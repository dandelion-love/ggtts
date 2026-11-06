//! Edge trimming and the judge (render spec sections 5 and 6).

use crate::numeric::{log10, mean_f64};
use crate::text::cjk_len;

/// A frame is loud when its RMS is more than this many dB below the loudest frame's, strictly.
pub const TOP_DB: f64 = 40.0;
/// Added to every frame's RMS before the ratio, so a silent frame has a finite level.
const RMS_EPS: f64 = 1e-12;

/// Two channels to one, by mean, in float32 (render spec section 6, step 1).
pub fn downmix_stereo(frames: &[[f32; 2]]) -> Vec<f32> {
    frames.iter().map(|[l, r]| ((l + r) as f64 / 2.0) as f32).collect()
}

/// What [`trim_edges`] found and kept. Seconds are unrounded; the sample bounds are exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimInfo {
    pub raw_s: f64,
    pub trimmed_s: f64,
    pub lead_cut_s: f64,
    pub tail_cut_s: f64,
    pub lead_silence_s: f64,
    pub tail_silence_s: f64,
    /// The kept audio is `y[start_sample..end_sample]`.
    pub start_sample: usize,
    pub end_sample: usize,
    pub raw_samples: usize,
}

/// Cap the silence at each end to `max_edge_s` seconds.
///
/// Frames are 10 ms (`sr / 100` samples, the last one may be short). A frame is loud when
/// `20 log10(rms / loudest) > -40`, strictly, with `1e-12` added to every RMS. Silence beyond the cap is cut, not
/// removed outright: the kept audio starts `round(max_edge_s * sr)` samples before the first loud frame and ends
/// as many after the last. Input is mono; see [`downmix_stereo`].
pub fn trim_edges(y: &[f64], sr: u32, max_edge_s: f64) -> TrimInfo {
    let n = y.len();
    let srf = f64::from(sr);
    let raw_s = n as f64 / srf;
    if n == 0 {
        return TrimInfo {
            raw_s: 0.0,
            trimmed_s: 0.0,
            lead_cut_s: 0.0,
            tail_cut_s: 0.0,
            lead_silence_s: 0.0,
            tail_silence_s: 0.0,
            start_sample: 0,
            end_sample: 0,
            raw_samples: 0,
        };
    }
    let hop = std::cmp::max(1, (sr / 100) as usize);
    let mut sq = Vec::with_capacity(hop);
    let rms: Vec<f64> = y
        .chunks(hop)
        .map(|f| {
            sq.clear();
            sq.extend(f.iter().map(|x| x * x));
            mean_f64(&sq).sqrt() + RMS_EPS
        })
        .collect();
    let loudest = rms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let loud: Vec<bool> = rms.iter().map(|&r| 20.0 * log10(r / loudest) > -TOP_DB).collect();
    let (Some(first), Some(last)) = (loud.iter().position(|&b| b), loud.iter().rposition(|&b| b)) else {
        return TrimInfo {
            raw_s,
            trimmed_s: raw_s,
            lead_cut_s: 0.0,
            tail_cut_s: 0.0,
            lead_silence_s: raw_s,
            tail_silence_s: 0.0,
            start_sample: 0,
            end_sample: n,
            raw_samples: n,
        };
    };
    let keep = (max_edge_s * srf).round_ties_even() as usize;
    let start = (first * hop).saturating_sub(keep);
    let end = std::cmp::min(n, (last + 1) * hop + keep);
    TrimInfo {
        raw_s,
        trimmed_s: (end - start) as f64 / srf,
        lead_cut_s: start as f64 / srf,
        tail_cut_s: (n - end) as f64 / srf,
        lead_silence_s: (first * hop) as f64 / srf,
        tail_silence_s: ((loud.len() - 1 - last) * hop) as f64 / srf,
        start_sample: start,
        end_sample: end,
        raw_samples: n,
    }
}

/// [`trim_edges`] on float32 samples, which widen to float64 exactly.
pub fn trim_edges_f32(y: &[f32], sr: u32, max_edge_s: f64) -> TrimInfo {
    let wide: Vec<f64> = y.iter().map(|&x| f64::from(x)).collect();
    trim_edges(&wide, sr, max_edge_s)
}

/// A functional failure the judge can name without a listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Speech over [`RUNAWAY_SPC`] seconds per character.
    Runaway,
    /// Speech under [`SILENT_SPC`] seconds per character.
    Silent,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Runaway => "runaway",
            Verdict::Silent => "silent",
        }
    }
}

/// Above this many seconds of speech per character, a draw is a runaway (strict).
pub const RUNAWAY_SPC: f64 = 0.60;
/// Below this many seconds of speech per character, a draw is near-silent (strict).
pub const SILENT_SPC: f64 = 0.07;

/// The judge's character count: CJK characters plus a third of the ASCII letters and digits, at least one.
pub fn judge_chars(text: &str) -> usize {
    let ascii_alnum = text.chars().filter(|c| c.is_ascii_alphanumeric()).count();
    std::cmp::max(1, cjk_len(text) + ascii_alnum / 3)
}

/// The judge's reading of one draw: its verdict and its unrounded seconds of speech per character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Judgement {
    pub verdict: Option<Verdict>,
    pub s_per_char: f64,
}

/// Judge one draw by its speech span: the loud frames' extent ([`trim_edges`] with no edge kept) over
/// [`judge_chars`].
pub fn judge(y: &[f64], sr: u32, text: &str) -> Judgement {
    let edge = trim_edges(y, sr, 0.0);
    let speech = (edge.end_sample - edge.start_sample) as f64 / f64::from(sr);
    let spc = speech / judge_chars(text) as f64;
    let verdict = if spc > RUNAWAY_SPC {
        Some(Verdict::Runaway)
    } else if spc < SILENT_SPC {
        Some(Verdict::Silent)
    } else {
        None
    };
    Judgement {
        verdict,
        s_per_char: spc,
    }
}

/// [`judge`] on float32 samples.
pub fn judge_f32(y: &[f32], sr: u32, text: &str) -> Judgement {
    let wide: Vec<f64> = y.iter().map(|&x| f64::from(x)).collect();
    judge(&wide, sr, text)
}
