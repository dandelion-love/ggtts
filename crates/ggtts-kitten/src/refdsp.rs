//! Host-side signal processing for reference-clip preparation: reading a WAV file, torchaudio's sinc resampler,
//! librosa's Slaney mel filterbank, S3Gen's 24 kHz log-mel (`kittenml/s3gen.py` `mel_spectrogram`), the S3
//! tokenizer's Whisper-style log-mel (`S3Tokenizer.log_mel_spectrogram`) and Kaldi's fbank
//! (`torchaudio.compliance.kaldi.fbank` with its defaults, as CAMPPlus's `extract_feature` calls it).
//!
//! Everything is f64 on the host: these run once per reference clip, not per render.

use std::f64::consts::PI;
use std::path::Path;

/// Mono samples and their rate from a RIFF WAV file: PCM 16, 24 or 32 bit, or IEEE float 32. Channels are averaged.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF WAVE file", path.display()));
    }
    let (mut fmt, mut channels, mut rate, mut bits, mut data) = (0u16, 0u16, 0u32, 0u16, None);
    let mut p = 12;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let n = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = &b[p + 8..(p + 8 + n).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes([body[14], body[15]]);
            if fmt == 0xfffe && body.len() >= 26 {
                fmt = u16::from_le_bytes([body[24], body[25]]); // WAVE_FORMAT_EXTENSIBLE: the sub-format
            }
        } else if id == b"data" {
            data = Some(body);
        }
        p += 8 + n + (n & 1);
    }
    let data = data.ok_or("no data chunk")?;
    let ch = usize::from(channels.max(1));
    let samples: Vec<f64> = match (fmt, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|c| f64::from(i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) / 8388608.0)
            .collect(),
        (1, 32) => data
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])) / 2147483648.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        _ => return Err(format!("unsupported WAV format {fmt} with {bits} bits")),
    };
    let mono = samples
        .chunks_exact(ch)
        .map(|f| (f.iter().sum::<f64>() / ch as f64) as f32)
        .collect();
    Ok((mono, rate))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `torchaudio.functional.resample` with its defaults: windowed sinc, Hann window, `lowpass_filter_width` 6,
/// `rolloff` 0.99.
pub fn resample(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new {
        return x.to_vec();
    }
    let g = gcd(orig, new);
    let (o, n) = ((orig / g) as usize, (new / g) as usize);
    let lpw = 6.0f64;
    let base = f64::from(o.min(n) as u32) * 0.99;
    let width = (lpw * o as f64 / base).ceil() as usize;
    let klen = 2 * width + o;
    let scale = base / o as f64;
    let mut kernels = vec![0f64; n * klen];
    for ph in 0..n {
        for (j, k) in kernels[ph * klen..(ph + 1) * klen].iter_mut().enumerate() {
            let idx = (j as f64 - width as f64) / o as f64;
            let mut t = (-(ph as f64) / n as f64 + idx) * base;
            t = t.clamp(-lpw, lpw);
            let w = (t * PI / lpw / 2.0).cos().powi(2);
            let t = t * PI;
            *k = if t == 0.0 { 1.0 } else { t.sin() / t } * w * scale;
        }
    }
    // Pad (width, width + orig), then strided correlation, phases interleaved.
    let len = x.len();
    let at = |i: isize| -> f64 {
        if i < 0 || i as usize >= len {
            0.0
        } else {
            f64::from(x[i as usize])
        }
    };
    let frames = (len + 2 * width + o - klen) / o + 1;
    let target = (n * len).div_ceil(o);
    let mut out = Vec::with_capacity(frames * n);
    for f in 0..frames {
        for ph in 0..n {
            let k = &kernels[ph * klen..(ph + 1) * klen];
            let start = (f * o) as isize - width as isize;
            let v: f64 = k.iter().enumerate().map(|(j, w)| w * at(start + j as isize)).sum();
            out.push(v as f32);
        }
    }
    out.truncate(target);
    out
}

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let (min_log_hz, logstep) = (1000.0, (6.4f64).ln() / 27.0);
    if f >= min_log_hz {
        min_log_hz / f_sp + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let (min_log_hz, logstep) = (1000.0, (6.4f64).ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` with its defaults (Slaney scale and area normalisation),
/// row-major `[n_mels, n_fft / 2 + 1]`.
pub fn mel_filters(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let nb = n_fft / 2 + 1;
    let fft: Vec<f64> = (0..nb).map(|i| sr / 2.0 * i as f64 / (nb - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut w = vec![0f64; n_mels * nb];
    for m in 0..n_mels {
        let (d0, d1) = (mel_f[m + 1] - mel_f[m], mel_f[m + 2] - mel_f[m + 1]);
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (j, f) in fft.iter().enumerate() {
            let lower = (f - mel_f[m]) / d0;
            let upper = (mel_f[m + 2] - f) / d1;
            w[m * nb + j] = lower.min(upper).max(0.0) * enorm;
        }
    }
    w
}

/// The power or magnitude spectrum of one real frame by direct DFT with a precomputed table.
struct Dft {
    n: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Dft {
    fn new(n: usize) -> Self {
        let nb = n / 2 + 1;
        let mut cos = vec![0f64; nb * n];
        let mut sin = vec![0f64; nb * n];
        for k in 0..nb {
            for m in 0..n {
                let a = 2.0 * PI * ((k * m) % n) as f64 / n as f64;
                cos[k * n + m] = a.cos();
                sin[k * n + m] = a.sin();
            }
        }
        Dft { n, cos, sin }
    }
    /// `|X_k|^2` for k in 0..=n/2.
    fn power(&self, frame: &[f64], out: &mut [f64]) {
        for (k, o) in out.iter_mut().enumerate() {
            let (c, s) = (
                &self.cos[k * self.n..(k + 1) * self.n],
                &self.sin[k * self.n..(k + 1) * self.n],
            );
            let (mut re, mut im) = (0f64, 0f64);
            for m in 0..self.n {
                re += frame[m] * c[m];
                im -= frame[m] * s[m];
            }
            *o = re * re + im * im;
        }
    }
}

fn reflect(x: &[f32], i: isize) -> f64 {
    let n = x.len() as isize;
    let j = if i < 0 {
        -i
    } else if i >= n {
        2 * (n - 1) - i
    } else {
        i
    };
    f64::from(x[j as usize])
}

/// Periodic Hann, `torch.hann_window(n)`.
fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect()
}

/// Apply `f` to every frame on up to four threads, collecting rows in order.
fn par_frames<F: Fn(usize) -> Vec<f32> + Sync>(frames: usize, f: F) -> Vec<f32> {
    let threads = 4.min(frames.max(1));
    let per = frames.div_ceil(threads);
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let f = &f;
                s.spawn(move || {
                    ((t * per)..((t + 1) * per).min(frames))
                        .flat_map(f)
                        .collect::<Vec<f32>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.concat()
}

/// S3Gen's prompt mel at 24 kHz: reflect padding by 720 each side, 1920-point frames every 480 with a periodic
/// Hann window, `sqrt(|X|^2 + 1e-9)`, the Slaney mel filterbank (80 mels, 0-8000 Hz), `log(max(x, 1e-5))`.
/// Returns `[frames, 80]` frame-major, as `prompt_feat[0]`.
pub fn mel24(y: &[f32]) -> Vec<f32> {
    let (n_fft, hop, pad) = (1920usize, 480usize, 720isize);
    let fb = mel_filters(24000.0, n_fft, 80, 0.0, 8000.0);
    let win = hann_periodic(n_fft);
    let dft = Dft::new(n_fft);
    let padded = y.len() + 2 * pad as usize;
    let frames = (padded - n_fft) / hop + 1;
    let nb = n_fft / 2 + 1;
    par_frames(frames, |f| {
        let frame: Vec<f64> = (0..n_fft)
            .map(|m| reflect(y, (f * hop + m) as isize - pad) * win[m])
            .collect();
        let mut p = vec![0f64; nb];
        dft.power(&frame, &mut p);
        let mag: Vec<f64> = p.iter().map(|v| (v + 1e-9).sqrt()).collect();
        (0..80)
            .map(|m| {
                let e: f64 = fb[m * nb..(m + 1) * nb].iter().zip(&mag).map(|(a, b)| a * b).sum();
                e.max(1e-5).ln() as f32
            })
            .collect()
    })
}

/// The S3 tokenizer's log-mel: 400-point frames every 160 (centred, reflect), periodic Hann, `|X|^2` without the
/// last frame, the given `[128, 201]` filters, `log10(max(x, 1e-10))`, floored at the maximum minus 8, then
/// `(x + 4) / 4`. Returns `[128, frames]` channel-major.
pub fn whisper_mel(y: &[f32], filters: &[f32]) -> Vec<f32> {
    let (n_fft, hop) = (400usize, 160usize);
    let win = hann_periodic(n_fft);
    let dft = Dft::new(n_fft);
    let nb = n_fft / 2 + 1;
    let frames = y.len() / hop; // centre framing gives len/hop + 1; the last is dropped
    let rows = par_frames(frames, |f| {
        let frame: Vec<f64> = (0..n_fft)
            .map(|m| reflect(y, (f * hop + m) as isize - (n_fft / 2) as isize) * win[m])
            .collect();
        let mut p = vec![0f64; nb];
        dft.power(&frame, &mut p);
        (0..128)
            .map(|m| {
                let e: f64 = filters[m * nb..(m + 1) * nb]
                    .iter()
                    .zip(&p)
                    .map(|(a, b)| f64::from(*a) * b)
                    .sum();
                e.max(1e-10).log10() as f32
            })
            .collect()
    });
    let mx = rows.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0f32; 128 * frames];
    for f in 0..frames {
        for m in 0..128 {
            out[m * frames + f] = (rows[f * 128 + m].max(mx - 8.0) + 4.0) / 4.0;
        }
    }
    out
}

fn kaldi_mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// `torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)` at 16 kHz with every other argument at its default (25 ms
/// frames every 10 ms, edges snipped, no dither, DC removed, pre-emphasis 0.97, the Povey window, 512-point power
/// spectrum, Kaldi mel banks 20-8000 Hz, natural log floored at f32 epsilon), then CAMPPlus's per-bin mean
/// subtraction. `[frames, 80]` frame-major.
pub fn kaldi_fbank(x: &[f32], plant_no_preemph: bool) -> Vec<f32> {
    let (len, shift, nfft, bins) = (400usize, 160usize, 512usize, 80usize);
    if x.len() < len {
        return Vec::new();
    }
    let frames = 1 + (x.len() - len) / shift;
    let win: Vec<f64> = (0..len)
        .map(|n| (0.5 - 0.5 * (2.0 * PI * n as f64 / (len - 1) as f64).cos()).powf(0.85))
        .collect();
    let (lo, hi) = (kaldi_mel(20.0), kaldi_mel(8000.0));
    let delta = (hi - lo) / (bins + 1) as f64;
    let width = 16000.0 / nfft as f64;
    let mut banks = vec![0f64; bins * (nfft / 2)];
    for b in 0..bins {
        let (l, c, r) = (
            lo + b as f64 * delta,
            lo + (b + 1) as f64 * delta,
            lo + (b + 2) as f64 * delta,
        );
        for j in 0..nfft / 2 {
            let m = kaldi_mel(width * j as f64);
            let up = (m - l) / (c - l);
            let down = (r - m) / (r - c);
            banks[b * (nfft / 2) + j] = up.min(down).max(0.0);
        }
    }
    let dft = Dft::new(nfft);
    let eps = f64::from(f32::EPSILON);
    let rows = par_frames(frames, |f| {
        let s: Vec<f64> = x[f * shift..f * shift + len].iter().map(|&v| f64::from(v)).collect();
        let mean = s.iter().sum::<f64>() / len as f64;
        let s: Vec<f64> = s.iter().map(|v| v - mean).collect();
        let mut frame = vec![0f64; nfft];
        for n in 0..len {
            let prev = if n == 0 { s[0] } else { s[n - 1] };
            let v = if plant_no_preemph { s[n] } else { s[n] - 0.97 * prev };
            frame[n] = v * win[n];
        }
        let mut p = vec![0f64; nfft / 2 + 1];
        dft.power(&frame, &mut p);
        (0..bins)
            .map(|b| {
                let e: f64 = banks[b * (nfft / 2)..(b + 1) * (nfft / 2)]
                    .iter()
                    .zip(&p)
                    .map(|(a, q)| a * q)
                    .sum();
                e.max(eps).ln() as f32
            })
            .collect()
    });
    let mut out = rows;
    for b in 0..bins {
        let m = (0..frames).map(|f| f64::from(out[f * bins + b])).sum::<f64>() / frames as f64;
        for f in 0..frames {
            out[f * bins + b] = (f64::from(out[f * bins + b]) - m) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_a_low_tone_and_the_length() {
        let x: Vec<f32> = (0..2400)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 24000.0).sin() as f32)
            .collect();
        let y = resample(&x, 24000, 16000);
        assert_eq!(y.len(), 1600);
        let want: Vec<f32> = (0..1600)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 16000.0).sin() as f32)
            .collect();
        let err = y[100..1500]
            .iter()
            .zip(&want[100..1500])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-2, "{err}");
    }

    #[test]
    fn slaney_filters_are_triangles_with_unit_area_scale() {
        let w = mel_filters(16000.0, 400, 128, 0.0, 8000.0);
        assert_eq!(w.len(), 128 * 201);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!((0..128).all(|m| w[m * 201..(m + 1) * 201].iter().any(|&v| v > 0.0)));
    }
}
