//! Kitten-TTS 2's sampler, in the PyTorch path's order (`kittenml/logits.py:96-117`): the token-run penalty, the
//! windowed repetition penalty (exempting `speech_end` and `stop`, and deliberately not the silence token), then
//! temperature, top-k, top-p and min-p, then a draw from the softmax.
//!
//! The draw uses this crate's own generator, so a seed means the same thing on every backend; it is not PyTorch's
//! stream, and sampled tokens are not expected to match PyTorch's (DESIGN.md, parity level 4).

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// The last N generated positions the repetition penalty scans; 0 scans everything.
    pub repetition_window: usize,
    pub run_penalty: f32,
    pub run_grace: usize,
}

impl Default for Settings {
    /// The `stable` preset and `kittenml/model.py`'s `ADVANCED_OPTIONS`, with the model's window of 50.
    fn default() -> Self {
        Settings {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.8,
            min_p: 0.0,
            repetition_penalty: 1.1,
            repetition_window: 50,
            run_penalty: 1.3,
            run_grace: 10,
        }
    }
}

/// The ids the penalties need.
#[derive(Debug, Clone, Copy)]
pub struct Ids {
    pub audio_lo: u32,
    pub audio_hi: u32,
    pub speech_end: u32,
    pub stop: u32,
}

/// SplitMix64: small, fast, and the same everywhere.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Apply the processors to `scores` in place. `history` is the prompt and everything generated so far.
pub fn process(scores: &mut [f32], history: &[u32], prompt_len: usize, ids: Ids, s: &Settings) {
    // TokenRunPenalty.
    if let Some(&last) = history.last() {
        if s.run_penalty != 0.0 {
            let run = history.iter().rev().take_while(|&&t| t == last).count();
            if (ids.audio_lo..ids.audio_hi).contains(&last) && run >= s.run_grace {
                scores[last as usize] -= (run - s.run_grace + 1) as f32 * s.run_penalty;
            }
        }
    }
    // RepetitionPenaltyExempt: each distinct id in the window once.
    if s.repetition_penalty != 1.0 {
        let start = if s.repetition_window > 0 {
            prompt_len.max(history.len().saturating_sub(s.repetition_window))
        } else {
            0
        };
        let mut seen: Vec<u32> = history[start.min(history.len())..].to_vec();
        seen.sort_unstable();
        seen.dedup();
        for id in seen {
            if id == ids.speech_end || id == ids.stop {
                continue;
            }
            let v = &mut scores[id as usize];
            *v = if *v < 0.0 {
                *v * s.repetition_penalty
            } else {
                *v / s.repetition_penalty
            };
        }
    }
    if s.temperature != 1.0 {
        for v in scores.iter_mut() {
            *v /= s.temperature;
        }
    }
    // TopKLogitsWarper: remove scores below the k-th largest.
    if s.top_k > 0 && s.top_k < scores.len() {
        let mut tmp: Vec<f32> = scores.to_vec();
        let k = s.top_k;
        tmp.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        let cutoff = tmp[k - 1];
        for v in scores.iter_mut() {
            if *v < cutoff {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    // TopPLogitsWarper: ascending sort, drop the low tail whose cumulative probability is at most 1 - top_p,
    // always keeping the most likely token.
    if s.top_p < 1.0 {
        let mut order: Vec<usize> = (0..scores.len()).filter(|&i| scores[i].is_finite()).collect();
        order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
        let peak = scores[*order.last().unwrap()];
        let sum: f64 = order.iter().map(|&i| f64::from(scores[i] - peak).exp()).sum();
        let mut acc = 0f64;
        for &i in &order[..order.len() - 1] {
            acc += f64::from(scores[i] - peak).exp() / sum;
            if acc <= 1.0 - f64::from(s.top_p) {
                scores[i] = f32::NEG_INFINITY;
            }
        }
    }
    if s.min_p > 0.0 {
        let peak = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let cutoff = peak + s.min_p.ln();
        for v in scores.iter_mut() {
            if *v < cutoff {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

/// Draw from the softmax of the finite scores, by inverse CDF in id order.
pub fn sample(scores: &[f32], rng: &mut Rng) -> u32 {
    let peak = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, f64::from(v - peak).exp()))
        .collect();
    let total: f64 = w.iter().map(|x| x.1).sum();
    let mut u = rng.uniform() * total;
    for &(i, p) in &w {
        if u < p {
            return i as u32;
        }
        u -= p;
    }
    w.last().map_or(0, |x| x.0 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: Ids = Ids {
        audio_lo: 10,
        audio_hi: 20,
        speech_end: 2,
        stop: 3,
    };

    #[test]
    fn same_seed_same_draws() {
        let scores: Vec<f32> = (0..30).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let da: Vec<u32> = (0..50).map(|_| sample(&scores, &mut a)).collect();
        let db: Vec<u32> = (0..50).map(|_| sample(&scores, &mut b)).collect();
        assert_eq!(da, db);
        assert!(da.iter().collect::<std::collections::BTreeSet<_>>().len() > 3);
    }

    #[test]
    fn top_k_one_is_greedy_and_end_tokens_are_exempt() {
        let mut scores = vec![0f32; 30];
        scores[2] = 5.0;
        scores[12] = 4.9;
        let s = Settings {
            top_k: 1,
            ..Settings::default()
        };
        process(&mut scores, &[1, 2, 12], 1, IDS, &s);
        assert_eq!(sample(&scores, &mut Rng::new(1)), 2);
    }

    #[test]
    fn the_run_penalty_bites_after_grace() {
        let mut a = vec![1f32; 30];
        let hist = vec![15u32; 12];
        process(
            &mut a,
            &hist,
            0,
            IDS,
            &Settings {
                repetition_penalty: 1.0,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                ..Settings::default()
            },
        );
        assert!((a[15] - (1.0 - 3.0 * 1.3)).abs() < 1e-6);
    }
}
