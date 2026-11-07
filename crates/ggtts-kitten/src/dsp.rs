//! The HiFT vocoder's host-side signal processing (`kittenml/s3gen.py` `SineGen`, `SourceModuleHnNSF`,
//! `HiFTGenerator._stft` and `._istft`): the harmonic source from f0, and the 16-point STFT and its inverse. These
//! are small and sequential (a cumulative phase, an overlap-add), so they run in f64 on the host rather than in the
//! graph.

use std::f64::consts::PI;

pub const N_FFT: usize = 16;
pub const HOP: usize = 4;
pub const N_BINS: usize = N_FFT / 2 + 1;

/// `scipy.signal.get_window("hann", 16, fftbins=True)`: the periodic Hann window.
pub fn hann() -> [f64; N_FFT] {
    let mut w = [0.0; N_FFT];
    for (n, x) in w.iter_mut().enumerate() {
        *x = 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos();
    }
    w
}

/// `torch.stft(x, 16, 4, 16, hann, center=True, pad_mode="reflect", return_complex=True)`, as `[real; imag]` rows:
/// 18 channels of `frames` values, channel-major (the layout `HiFTGenerator.decode` concatenates).
pub fn stft(x: &[f32]) -> (Vec<f32>, usize) {
    let pad = N_FFT / 2;
    let n = x.len();
    let at = |i: isize| -> f64 {
        // Reflect padding without repeating the edge.
        let j = if i < 0 {
            -i
        } else if i >= n as isize {
            2 * (n as isize - 1) - i
        } else {
            i
        };
        f64::from(x[j as usize])
    };
    let frames = n / HOP + 1;
    let w = hann();
    let mut out = vec![0f32; 2 * N_BINS * frames];
    for f in 0..frames {
        let start = (f * HOP) as isize - pad as isize;
        for k in 0..N_BINS {
            let (mut re, mut im) = (0f64, 0f64);
            for (m, wm) in w.iter().enumerate() {
                let v = at(start + m as isize) * wm;
                let ang = -2.0 * PI * (k * m) as f64 / N_FFT as f64;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            out[k * frames + f] = re as f32;
            out[(N_BINS + k) * frames + f] = im as f32;
        }
    }
    (out, frames)
}

/// `HiFTGenerator._istft`: magnitude clipped at 100, the inverse 16-point STFT with the Hann window, overlap-add
/// normalised by the window's squared envelope, `center=True` trimming; `4 * (frames - 1)` samples. `mag` and
/// `phase` are `N_BINS` channels of `frames`, channel-major.
pub fn istft(mag: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    let w = hann();
    let total = N_FFT + HOP * (frames - 1);
    let mut y = vec![0f64; total];
    let mut env = vec![0f64; total];
    for f in 0..frames {
        let mut re = [0f64; N_BINS];
        let mut im = [0f64; N_BINS];
        for k in 0..N_BINS {
            let m = f64::from(mag[k * frames + f]).min(1e2);
            let p = f64::from(phase[k * frames + f]);
            re[k] = m * p.cos();
            im[k] = m * p.sin();
        }
        for n in 0..N_FFT {
            // irfft: the imaginary parts of DC and Nyquist are ignored.
            let mut v = re[0] + re[N_BINS - 1] * if n % 2 == 0 { 1.0 } else { -1.0 };
            for k in 1..N_BINS - 1 {
                let ang = 2.0 * PI * (k * n) as f64 / N_FFT as f64;
                v += 2.0 * (re[k] * ang.cos() - im[k] * ang.sin());
            }
            v /= N_FFT as f64;
            y[f * HOP + n] += v * w[n];
            env[f * HOP + n] += w[n] * w[n];
        }
    }
    let len = HOP * (frames - 1);
    (0..len)
        .map(|i| {
            let j = i + N_FFT / 2;
            (if env[j] > 1e-11 { y[j] / env[j] } else { y[j] }) as f32
        })
        .collect()
}

/// A Gaussian and a uniform stream from SplitMix64.
pub struct Noise(crate::sampler::Rng);

impl Noise {
    pub fn new(seed: u64) -> Self {
        Noise(crate::sampler::Rng::new(seed))
    }
    pub fn uniform(&mut self) -> f64 {
        self.0.uniform()
    }
    /// Box-Muller.
    pub fn gauss(&mut self) -> f64 {
        let u1 = self.0.uniform().max(1e-300);
        let u2 = self.0.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// `SourceModuleHnNSF` on frame-rate f0: upsampled by `up` (nearest), eight harmonics with a random phase each (the
/// fundamental's is zero), unvoiced below 10 Hz, Gaussian noise, then `tanh(linear(9 -> 1))`.
pub fn source(f0: &[f32], up: usize, sr: f64, lin_w: &[f32], lin_b: f32, noise: &mut Noise) -> Vec<f32> {
    const H: usize = 9;
    let (amp, noise_std, thresh) = (0.1, 0.003, 10.0);
    let phases: Vec<f64> = (0..H)
        .map(|i| if i == 0 { 0.0 } else { noise.uniform() * 2.0 * PI - PI })
        .collect();
    let mut acc = [0f64; H];
    let mut out = Vec::with_capacity(f0.len() * up);
    for &f in f0 {
        let f = f64::from(f);
        let uv = if f > thresh { 1.0 } else { 0.0 };
        let namp = uv * noise_std + (1.0 - uv) * amp / 3.0;
        for _ in 0..up {
            let mut s = f64::from(lin_b);
            for (i, a) in acc.iter_mut().enumerate() {
                *a = (*a + f * (i + 1) as f64 / sr).fract();
                let sine = amp * (2.0 * PI * *a + phases[i]).sin();
                let v = sine * uv + namp * noise.gauss();
                s += f64::from(lin_w[i]) * v;
            }
            out.push(s.tanh() as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_then_istft_is_the_identity() {
        let x: Vec<f32> = (0..400)
            .map(|i| ((i as f32) * 0.13).sin() * 0.5 + ((i * 7 % 13) as f32) * 0.01)
            .collect();
        let (spec, frames) = stft(&x);
        let mut mag = vec![0f32; N_BINS * frames];
        let mut phase = vec![0f32; N_BINS * frames];
        for i in 0..N_BINS * frames {
            let (re, im) = (spec[i], spec[N_BINS * frames + i]);
            mag[i] = (re * re + im * im).sqrt();
            phase[i] = im.atan2(re);
        }
        let y = istft(&mag, &phase, frames);
        assert_eq!(y.len(), x.len());
        let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-5, "round trip error {err}");
    }
}
