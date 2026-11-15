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

/// The largest top-k at which computing only the head rows below was measured exact (the default is 50).
pub const HEAD_ROWS_MAX_TOP_K: usize = 50;

/// `Ok` if `rows` is a non-empty range of the output head's `n_vocab` rows. [`head_rows`] always gives one for a token
/// map that passed [`crate::prompt::TokenMap::checked`]; anything else is a caller's error, refused rather than
/// silently computed as the whole head.
pub fn check_head_rows(rows: &std::ops::Range<usize>, n_vocab: usize) -> Result<(), String> {
    if rows.start < rows.end && rows.end <= n_vocab {
        Ok(())
    } else {
        Err(format!(
            "head_rows {}..{} is not a non-empty range of the output head's {n_vocab} rows",
            rows.start, rows.end
        ))
    }
}

/// The output-head rows a generation needs, or `None` for the whole head.
///
/// The rows run from the first end id to the end of the vocabulary: the end ids, the audio ids, and the special ids
/// after them, among them the reference's own end marker (`reference_speech_end_id`), which reaches the top-k set at
/// the end of a line. Computing only these rows is a measured shortcut, not a rule of the model: the sampler draws
/// over the whole vocabulary, as both references do, so it is exact only while no other id reaches the top-k set.
/// That was measured at the default settings (`examples/head_rows_check.rs`), so the subset is used only inside that
/// envelope: top-k on and at most [`HEAD_ROWS_MAX_TOP_K`], and top-p below 1. `full_head` is the value of
/// `GGTTS_KITTEN_FULL_HEAD`: exactly `"1"` asks for the whole head; any other value, or none, does not.
pub fn head_rows(
    s: &Settings,
    tm: &crate::prompt::TokenMap,
    n_vocab: usize,
    full_head: Option<&str>,
) -> Option<std::ops::Range<usize>> {
    if full_head == Some("1") {
        return None;
    }
    if !(s.top_k > 0 && s.top_k <= HEAD_ROWS_MAX_TOP_K && s.top_p < 1.0) {
        return None;
    }
    let lo = tm.speech_end.min(tm.stop).min(tm.audio_base) as usize;
    Some(lo..n_vocab)
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
///
/// Refuses logits that are not all finite: a NaN or an infinity means the forward pass failed, and drawing from
/// what is left would hide it (and leave nothing finite to draw from when every logit is NaN).
pub fn process(scores: &mut [f32], history: &[u32], prompt_len: usize, ids: Ids, s: &Settings) -> Result<(), String> {
    if let Some(i) = scores.iter().position(|v| !v.is_finite()) {
        let bad = scores.iter().filter(|v| !v.is_finite()).count();
        return Err(format!(
            "non-finite logits: {bad} of {} (the first, id {i}, is {}); the sampler refuses them",
            scores.len(),
            scores[i]
        ));
    }
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
        let Some(&last) = order.last() else {
            return Err("no finite score is left for top-p".into());
        };
        let peak = scores[last];
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
    Ok(())
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

    /// The token map of the released Kitten-TTS 2 LMs (`kitten.token_map`).
    fn kitten_map() -> crate::prompt::TokenMap {
        crate::prompt::TokenMap {
            audio_base: 151_675,
            audio_count: 6561,
            speech_start: 151_669,
            speech_end: 151_670,
            text_start: 151_671,
            start: 151_673,
            stop: 151_674,
            final_seg: 158_236,
            ref_text_start: 158_239,
            ref_text_end: 158_240,
            ref_speech_start: 158_237,
            ref_speech_end: 158_238,
        }
    }

    #[test]
    fn head_rows_outside_the_head_are_refused_not_widened() {
        assert!(check_head_rows(&(151_670..158_241), 158_241).is_ok());
        for bad in [
            151_670..158_242,
            158_241..158_241,
            std::ops::Range { start: 9, end: 3 },
            0..0,
        ] {
            let e = check_head_rows(&bad, 158_241).unwrap_err();
            assert!(
                e.contains(&format!("head_rows {}..{}", bad.start, bad.end)) && e.contains("158241 rows"),
                "{e}"
            );
        }
        // Every range head_rows gives for a checked map passes.
        let tm = kitten_map().checked(158_241).expect("the released map fits");
        let rows = head_rows(&Settings::default(), &tm, 158_241, None).unwrap();
        assert!(check_head_rows(&rows, 158_241).is_ok());
    }

    #[test]
    fn the_head_rows_cover_every_id_a_step_can_keep_and_the_reference_end_marker() {
        let tm = kitten_map();
        let rows = head_rows(&Settings::default(), &tm, 158_241, None).expect("the default settings use the subset");
        assert_eq!(rows, 151_670..158_241);
        for id in [
            tm.speech_end,
            tm.stop,
            tm.audio_base,
            tm.audio_base + tm.audio_count - 1,
        ] {
            assert!(rows.contains(&(id as usize)), "id {id}");
        }
        // Measured: this id reaches the top-k set at the end of a line, so the rows must cover it.
        assert!(rows.contains(&(tm.ref_speech_end as usize)), "reference_speech_end_id");
        for id in [tm.final_seg, tm.ref_speech_start, tm.ref_text_start, tm.ref_text_end] {
            assert!(rows.contains(&(id as usize)), "special id {id} after the audio ids");
        }
    }

    #[test]
    fn the_head_rows_are_used_only_inside_the_measured_sampler_envelope() {
        let tm = kitten_map();
        let with = |top_k: usize, top_p: f32| {
            let s = Settings {
                top_k,
                top_p,
                ..Settings::default()
            };
            head_rows(&s, &tm, 158_241, None).is_some()
        };
        assert!(with(50, 0.8), "the defaults");
        assert!(with(1, 0.8));
        assert!(!with(0, 0.8), "top-k off: every id is a candidate");
        assert!(!with(51, 0.8), "above the measured top-k");
        assert!(!with(64, 0.8));
        assert!(!with(50, 1.0), "top-p off");
    }

    #[test]
    fn only_exactly_one_asks_for_the_whole_head() {
        let tm = kitten_map();
        let s = Settings::default();
        assert_eq!(head_rows(&s, &tm, 158_241, Some("1")), None);
        for v in ["0", "", "true", "yes", " 1"] {
            assert!(
                head_rows(&s, &tm, 158_241, Some(v)).is_some(),
                "GGTTS_KITTEN_FULL_HEAD={v:?}"
            );
        }
    }

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
        process(&mut scores, &[1, 2, 12], 1, IDS, &s).unwrap();
        assert_eq!(sample(&scores, &mut Rng::new(1)), 2);
    }

    #[test]
    fn non_finite_logits_are_refused_not_sampled() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut scores = vec![0.5f32; 30];
            scores[7] = bad;
            let e = process(&mut scores, &[1], 1, IDS, &Settings::default()).unwrap_err();
            assert!(e.contains("non-finite logits: 1 of 30") && e.contains("id 7"), "{e}");
        }
        // Every logit NaN: the case that panicked in top-p before this check.
        let mut scores = vec![f32::NAN; 30];
        let e = process(&mut scores, &[1], 1, IDS, &Settings::default()).unwrap_err();
        assert!(e.contains("30 of 30"), "{e}");
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
        )
        .unwrap();
        assert!((a[15] - (1.0 - 3.0 * 1.3)).abs() < 1e-6);
    }
}
