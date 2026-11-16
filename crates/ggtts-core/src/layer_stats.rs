//! Bounded same-draw talker layer aggregation.
use crate::engine::Stop;
use crate::hooks::{LayerStep, Step, TensorHook, Wants};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerSummary {
    pub layer: usize,
    pub count: u64,
    pub mean: Vec<f64>,
    pub mean_l2: f64,
    pub eos: bool,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerRecord {
    pub chunk: u32,
    pub draw: u32,
    pub seed: u64,
    pub stop: &'static str,
    pub layers: Vec<LayerSummary>,
}
struct Acc {
    count: u64,
    sum: Vec<f64>,
    norm_sum: f64,
    eos: bool,
}
struct Active {
    key: (u32, u32, u64),
    layers: BTreeMap<usize, Acc>,
    steps: u64,
}
struct State {
    selected: Vec<usize>,
    active: Option<Active>,
    records: Vec<LayerRecord>,
    error: Option<String>,
}
/// Cloneable hook and pipeline handle. Only one draw's activation sums are retained.
#[derive(Clone)]
pub struct LayerStats(Rc<RefCell<State>>);
impl LayerStats {
    pub fn new(mut layers: Vec<usize>) -> Result<Self, String> {
        layers.sort_unstable();
        layers.dedup();
        if layers.is_empty() {
            return Err("select at least one layer".into());
        }
        Ok(Self(Rc::new(RefCell::new(State {
            selected: layers,
            active: None,
            records: Vec::new(),
            error: None,
        }))))
    }
    /// Retrieve records only after a successful render and after checking hook failures.
    pub fn records(&self) -> Result<Vec<LayerRecord>, String> {
        let s = self.0.borrow();
        if let Some(e) = &s.error {
            return Err(e.clone());
        }
        if s.active.is_some() {
            return Err("incomplete layer capture".into());
        }
        Ok(s.records.clone())
    }
    pub(crate) fn end_draw(&self, chunk: u32, draw: u32, seed: u64, stop: Stop, keep: bool) -> Result<(), String> {
        let mut s = self.0.borrow_mut();
        if let Some(e) = &s.error {
            return Err(e.clone());
        }
        if s.active.is_none() {
            s.error = Some("missing layer capture".into());
            return Err("missing layer capture".into());
        }
        let active = s.active.as_ref().unwrap();
        if active.key != (chunk, draw, seed) {
            return Err("layer capture draw identity mismatch".into());
        }
        if active.layers.len() != s.selected.len() {
            return Err("missing requested layer".into());
        }
        let mut count = None;
        let mut layers = Vec::new();
        for &layer in &s.selected {
            let a = active.layers.get(&layer).ok_or("missing requested layer")?;
            if a.count == 0 || a.count != active.steps || count.is_some_and(|n| n != a.count) {
                return Err("inconsistent layer counts".into());
            }
            if a.eos != (stop == Stop::Eos) {
                return Err("layer EOS does not match stop".into());
            }
            count = Some(a.count);
            layers.push(LayerSummary {
                layer,
                count: a.count,
                mean: a.sum.iter().map(|v| v / a.count as f64).collect(),
                mean_l2: a.norm_sum / a.count as f64,
                eos: a.eos,
            });
        }
        s.active = None;
        if keep {
            s.records.push(LayerRecord {
                chunk,
                draw,
                seed,
                stop: if stop == Stop::Eos { "eos" } else { "cap" },
                layers,
            });
        }
        Ok(())
    }
}
impl TensorHook for LayerStats {
    fn name(&self) -> &str {
        "layer-stats"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: true,
            logits: false,
        }
    }
    fn layers(&self) -> Vec<usize> {
        self.0.borrow().selected.clone()
    }
    fn step(&mut self, step: &Step<'_>) {
        let mut s = self.0.borrow_mut();
        if s.error.is_some() {
            return;
        }
        let key = (step.chunk, step.draw, step.seed);
        if s.active.is_none() {
            s.active = Some(Active {
                key,
                layers: BTreeMap::new(),
                steps: 0,
            });
        }
        let a = s.active.as_mut().unwrap();
        if a.key != key || step.frame as u64 != a.steps {
            s.error = Some("invalid step identity or position".into());
            return;
        }
        a.steps += 1;
    }
    fn layer(&mut self, step: &LayerStep<'_>) {
        let mut s = self.0.borrow_mut();
        if s.error.is_some() {
            return;
        }
        let key = (step.chunk, step.draw, step.seed);
        if s.active.is_none() {
            s.active = Some(Active {
                key,
                layers: BTreeMap::new(),
                steps: 0,
            });
        }
        let a = s.active.as_mut().unwrap();
        if a.key != key {
            s.error = Some("overlapping layer draws".into());
            return;
        }
        let entry = a.layers.entry(step.layer).or_insert_with(|| Acc {
            count: 0,
            sum: vec![0.; step.values.len()],
            norm_sum: 0.,
            eos: false,
        });
        if step.values.is_empty()
            || entry.sum.len() != step.values.len()
            || step.frame as u64 != entry.count
            || entry.eos
            || step.values.iter().any(|x| !x.is_finite())
        {
            s.error = Some("invalid layer position, dimension or value".into());
            return;
        }
        let mut sq = 0.;
        for (sum, value) in entry.sum.iter_mut().zip(step.values) {
            let v = f64::from(*value);
            *sum += v;
            sq += v * v;
        }
        entry.norm_sum += sq.sqrt();
        entry.count += 1;
        entry.eos = step.is_eos;
    }
    fn finish(&mut self) -> Result<(), String> {
        let s = self.0.borrow();
        if let Some(e) = &s.error {
            return Err(e.clone());
        }
        if s.active.is_some() {
            return Err("incomplete layer capture".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn step(stats: &mut LayerStats, draw: u32, frame: u32, layer: usize, values: &[f32]) {
        stats.layer(&LayerStep {
            chunk: 0,
            draw,
            seed: 10 + draw as u64,
            frame,
            token: 1,
            is_eos: false,
            layer,
            values,
        });
    }
    fn tick(stats: &mut LayerStats, draw: u32, frame: u32) {
        stats.step(&Step {
            chunk: 0,
            draw,
            seed: 10 + draw as u64,
            frame,
            token: 1,
            p_stop: Some(0.),
            p_stop_semantics: None,
            logits: None,
        });
    }
    #[test]
    fn aggregates_only_the_kept_draw() {
        let mut x = LayerStats::new(vec![1, 0]).unwrap();
        tick(&mut x, 0, 0);
        step(&mut x, 0, 0, 0, &[99., 0.]);
        step(&mut x, 0, 0, 1, &[99., 0.]);
        x.end_draw(0, 0, 10, Stop::Cap, false).unwrap();
        for (frame, v) in [[3., 4.], [0., 0.]].iter().enumerate() {
            tick(&mut x, 1, frame as u32);
            step(&mut x, 1, frame as u32, 0, v);
            step(&mut x, 1, frame as u32, 1, v);
        }
        x.end_draw(0, 1, 11, Stop::Cap, true).unwrap();
        let records = x.records().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].draw, 1);
        assert_eq!(records[0].layers[0].mean, vec![1.5, 2.]);
        assert_eq!(records[0].layers[0].mean_l2, 2.5);
    }
    #[test]
    fn missing_layer_and_missing_step_remain_incomplete() {
        let mut x = LayerStats::new(vec![0, 1]).unwrap();
        tick(&mut x, 0, 0);
        step(&mut x, 0, 0, 0, &[1.]);
        assert!(x.end_draw(0, 0, 10, Stop::Cap, true).is_err());
        assert!(x.records().is_err());
        let mut y = LayerStats::new(vec![0]).unwrap();
        tick(&mut y, 0, 0);
        step(&mut y, 0, 0, 0, &[1.]);
        tick(&mut y, 0, 1);
        assert!(y.end_draw(0, 0, 10, Stop::Cap, true).is_err());
        assert!(y.records().is_err());
    }
}
