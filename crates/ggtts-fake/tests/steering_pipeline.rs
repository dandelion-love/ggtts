//! End-to-end steering control through the real chunk pipeline and a deterministic test engine.
//! This engine consumes scaled vectors; it does not represent the numerical behavior of Qwen.

use ggtts_core::capabilities::Capabilities;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm, Stop};
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{exact, render_plan, DurationJudge, Observers, Output, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::steer::{write_safetensors, SteerFile, Steering};
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;

/// A steering-aware double: each layer and element affects a distinct output token.
/// The ordinary FakeEngine cannot be used directly: it correctly refuses steering.
type LayerVectors = Vec<(u32, Vec<f32>)>;

struct SteeringEngine {
    inner: FakeEngine,
    seen: Vec<LayerVectors>,
}

impl Engine for SteeringEngine {
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::of("qwen3-tts").unwrap()
    }
    fn generate(&mut self, request: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        let scaled = request
            .steer
            .ok_or_else(|| EngineError("steering missing".into()))?
            .scaled();
        let mut frames = Vec::new();
        for (layer, vector) in &scaled {
            for (index, value) in vector.iter().enumerate() {
                // Quantization tolerates f32 roundoff but distinguishes the 1% plants.
                frames
                    .push((10_000.0 + (*layer as f32) * 100.0 + index as f32 * 10.0 + value * 1_000.0).round() as u16);
            }
        }
        self.seen.push(scaled);
        Ok(Generated {
            codes: Codes { codebooks: 1, frames },
            stop: Stop::Eos,
        })
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        Ok(Pcm {
            samples: codes
                .frames
                .iter()
                .flat_map(|x| std::iter::repeat_n(f32::from(*x) / 65_535.0, 256))
                .collect(),
            sample_rate: 24_000,
        })
    }
}

fn file(tensors: &[(&str, &[f32])]) -> SteerFile {
    SteerFile::parse(&write_safetensors(tensors)).unwrap()
}

fn render(steer: Steering) -> (Vec<f32>, Vec<LayerVectors>) {
    let mut engine = SteeringEngine {
        inner: FakeEngine::default(),
        seen: Vec::new(),
    };
    let mut options = RenderOptions::for_profile(engine.profile());
    exact::options(&mut options);
    options.steer = Some(steer);
    let chunks = exact::chunks("Steering control.").unwrap();
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let read = render_plan(
        &mut engine,
        &chunks,
        &exact::seeds(42),
        &mut DurationJudge,
        &options,
        &mut Observers::new(&mut events, &mut hooks),
        Output::Whole,
    )
    .unwrap();
    (read.samples, engine.seen)
}

#[test]
fn unit_vectors_match_raw_vectors_through_pipeline_and_each_one_percent_plant_is_detected() {
    // The two vectors have distinct norms 5 and 8. Supply layers out of order to test pairing.
    let raw = file(&[("talker.layer.3", &[3.0, 4.0]), ("talker.layer.9", &[0.0, 8.0])]);
    let unit = file(&[("talker.layer.3", &[0.6, 0.8]), ("talker.layer.9", &[0.0, 1.0])]);
    let (baseline, baseline_seen) = render(raw.select(Some(&[9, 3]), 0.5).unwrap());
    let (equivalent, equivalent_seen) = render(unit.select_scaled(&[9, 3], &[4.0, 2.5]).unwrap());
    assert!(!baseline.is_empty());
    assert_eq!(
        baseline, equivalent,
        "equivalent pushes must produce identical pipeline output"
    );
    assert_eq!(baseline_seen.len(), 1, "the engine must receive the steering once");
    assert_eq!(baseline_seen[0].iter().map(|(l, _)| *l).collect::<Vec<_>>(), vec![3, 9]);
    for ((_, expected), (_, actual)) in baseline_seen[0].iter().zip(&equivalent_seen[0]) {
        for (a, b) in expected.iter().zip(actual) {
            assert!((a - b).abs() < 1e-6, "scaled vector differs: {a} vs {b}");
        }
    }
    for plant in [[4.04, 2.5], [4.0, 2.525]] {
        let (bad, _) = render(unit.select_scaled(&[9, 3], &plant).unwrap());
        assert_ne!(baseline, bad, "the 1% plant on each layer must change pipeline output");
    }
}
