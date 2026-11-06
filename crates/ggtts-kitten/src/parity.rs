//! The numbers the parity check reports, and the bars stated in DESIGN.md before any measurement.

/// Relative L2 error of `a` against the reference `b`.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (f64::from(*x) - f64::from(*y)).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

pub fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    (s / a.len().max(1) as f64).sqrt()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

pub fn argmax(a: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if v > a[best] {
            best = i;
        }
    }
    best
}

/// The single-token path against the batched path on the same model and device (DESIGN.md, level 2): the logits'
/// cosine at every compared position, at least. Every argmax must also be equal.
pub const STEP_MIN_COSINE: f64 = 0.99999;

/// Fixed pseudo-random token ids in `0..n_vocab` (a 64-bit LCG), for a check that needs inputs and has no recording.
pub fn pseudo_ids(n: usize, n_vocab: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % n_vocab as u64) as u32
        })
        .collect()
}

/// A bar, as DESIGN.md states it for one arm.
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    /// Hidden state after every layer, relative L2, at most. `None`: not barred.
    pub hidden_rel_l2: Option<f64>,
    pub logits_rmse: f64,
    pub min_cosine: f64,
    /// Positions whose argmax must equal the reference's, out of the positions checked.
    pub min_argmax_equal: usize,
}

/// The f32 arm (weights exact, graph isolated).
pub const BAR_F32: Bar = Bar {
    hidden_rel_l2: Some(1e-3),
    logits_rmse: 0.01,
    min_cosine: 0.0,
    min_argmax_equal: 32,
};
/// The ternary runtime arm (padded TQ2_0, Q8_K activations, bf16 head).
pub const BAR_TERNARY: Bar = Bar {
    hidden_rel_l2: None,
    logits_rmse: 0.06,
    min_cosine: 0.9999,
    min_argmax_equal: 30,
};

/// The logits half of a verdict.
#[derive(Debug, Clone)]
pub struct LogitsReport {
    pub rmse: f64,
    pub min_cosine: f64,
    pub argmax_equal: usize,
    pub positions: usize,
}

pub fn logits_report(ours: &[f32], reference: &[f32], n_vocab: usize) -> LogitsReport {
    let positions = reference.len() / n_vocab;
    let mut min_cosine = f64::INFINITY;
    let mut argmax_equal = 0;
    for p in 0..positions {
        let a = &ours[p * n_vocab..(p + 1) * n_vocab];
        let b = &reference[p * n_vocab..(p + 1) * n_vocab];
        min_cosine = min_cosine.min(cosine(a, b));
        argmax_equal += usize::from(argmax(a) == argmax(b));
    }
    LogitsReport {
        rmse: rmse(ours, reference),
        min_cosine,
        argmax_equal,
        positions,
    }
}

impl Bar {
    pub fn passes(&self, worst_hidden: f64, l: &LogitsReport) -> bool {
        self.hidden_rel_l2.is_none_or(|b| worst_hidden <= b)
            && l.rmse <= self.logits_rmse
            && l.min_cosine >= self.min_cosine
            && l.argmax_equal >= self.min_argmax_equal.min(l.positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_no_op_reads_as_no_change_and_a_shift_does_not() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        assert_eq!(rel_l2(&a, &a), 0.0);
        assert_eq!(rmse(&a, &a), 0.0);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-12);
        let b: Vec<f32> = a.iter().map(|x| x + 0.1).collect();
        assert!(rmse(&b, &a) > 0.09);
        let r = logits_report(&a, &a, 50);
        assert!(BAR_F32.passes(0.0, &r) && r.argmax_equal == 2);
        let r = logits_report(&b, &a, 50);
        assert!(
            !BAR_TERNARY.passes(0.0, &r),
            "an offset of 0.1 must fail the ternary bar"
        );
    }
}
