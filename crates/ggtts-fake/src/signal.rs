//! Synthetic signals the parity cases are built from. Nothing here is recorded audio.

/// Silence, a 220 Hz sine of amplitude 0.3 for `speech_s` seconds, silence; computed in float64, returned as
/// float32. Each length is `round(seconds * sample_rate)` samples, rounded half to even.
pub fn tone(speech_s: f64, lead_s: f64, tail_s: f64, sample_rate: u32) -> Vec<f32> {
    let sr = f64::from(sample_rate);
    let n = (speech_s * sr).round_ties_even() as usize;
    let lead = (lead_s * sr).round_ties_even() as usize;
    let tail = (tail_s * sr).round_ties_even() as usize;
    let mut y = vec![0.0f32; lead];
    y.extend(sine(n, sample_rate));
    y.resize(y.len() + tail, 0.0);
    y
}

/// `0.3 * sin(2π · 220 · i / sample_rate)` for `n` samples, as float32.
pub fn sine(n: usize, sample_rate: u32) -> Vec<f32> {
    sine_f64(n, sample_rate).into_iter().map(|x| x as f32).collect()
}

/// [`sine`] before the float32 rounding.
pub fn sine_f64(n: usize, sample_rate: u32) -> Vec<f64> {
    let sr = f64::from(sample_rate);
    let w = 2.0 * std::f64::consts::PI * 220.0;
    (0..n).map(|i| 0.3 * libm::sin(w * (i as f64 / sr))).collect()
}

/// Two channels, 3.6 s at 24 kHz, that mix to mono by mean as something neither channel is.
///
/// Left: 0.3 s of silence, then the tone to 1.9 s. Right: silence to 1.3 s, the negated left tone to 1.9 s,
/// silence to 2.4 s, the tone to 3.4 s, silence. The mean is loud from 0.3 to 1.3 s and from 2.4 to 3.4 s, with
/// a 1.1 s gap where the channels cancel.
pub fn stereo_cancel() -> Vec<[f32; 2]> {
    let sr = 24_000u32;
    let at = |s: f64| (s * f64::from(sr)).round_ties_even() as usize;
    let n = at(3.6);
    let (a, b, c, d, e) = (at(0.3), at(1.9), at(1.3), at(2.4), at(3.4));
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];
    left[a..b].copy_from_slice(&sine_f64(b - a, sr));
    for (r, l) in right[c..b].iter_mut().zip(&left[c..b]) {
        *r = -l;
    }
    right[d..e].copy_from_slice(&sine_f64(e - d, sr));
    left.iter()
        .zip(&right)
        .map(|(&l, &r)| [l as f32, r as f32])
        .collect()
}

/// `edge_n` samples of DC `edge`, `middle_n` of DC `middle`, `edge_n` of DC `edge`, in float64.
pub fn dc(edge: f64, edge_n: usize, middle: f64, middle_n: usize) -> Vec<f64> {
    let mut y = vec![edge; edge_n];
    y.resize(edge_n + middle_n, middle);
    y.resize(2 * edge_n + middle_n, edge);
    y
}
