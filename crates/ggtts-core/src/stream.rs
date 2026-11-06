//! Streaming output: a read's pieces written as they arrive, in a container a player can read before it ends.
//!
//! Two containers, both 16-bit mono PCM:
//!
//! * [`Container::Wav`]: a RIFF/WAVE header whose RIFF and `data` sizes are `0xFFFFFFFF`, the convention for a
//!   stream of unknown length, then the samples. Players and decoders read it as it grows.
//! * [`Container::RawS16`]: one header line, `ggtts-pcm s16le rate=<Hz> channels=1\n`, then little-endian samples.
//!
//! Samples are converted with [`to_i16`], the same conversion the FLAC writer uses, so a stream of the levelled
//! read holds the FLAC's samples exactly. The header is written with the first piece, because the rate is the
//! first chunk's. Each piece is flushed as it is written.
//!
//! The pieces of a story read come from [`render_story_streamed`](crate::pipeline::render_story_streamed), before
//! the level step. [`first_sample_diff`] is the parity check between a stream and its batch read.
//!
//! **Two streams.** A listener gets the pieces levelled by [`FirstChunkLevel`]: the level step's own rule (gain to the
//! profile's target over the loud samples, then the clip), with the gain taken from the FIRST piece and held for the
//! read. The batch read's gain is known only after the last chunk; the first piece's is known the moment that piece
//! leaves, so this adds no latency. A parity check asks for the RAW stream instead, the pieces before the level step,
//! which is bit-exact to the batch read before its level step.

use crate::pipeline::{apply_gain, level_gain};
use crate::profile::Profile;

use std::io::{self, Write};

use crate::flac::to_i16;

/// The stream's container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Wav,
    RawS16,
}

impl Container {
    /// `wav` or `s16le`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "wav" => Some(Container::Wav),
            "s16le" | "raw" => Some(Container::RawS16),
            _ => None,
        }
    }
}

/// The size field of a WAV stream of unknown length.
pub const UNKNOWN_LENGTH: u32 = 0xFFFF_FFFF;

/// The raw container's header line for `rate`.
pub fn raw_header(rate: u32) -> String {
    format!("ggtts-pcm s16le rate={rate} channels=1\n")
}

/// The 44-byte header of a mono 16-bit WAV stream of unknown length at `rate`.
pub fn wav_header(rate: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&UNKNOWN_LENGTH.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    h[24..28].copy_from_slice(&rate.to_le_bytes());
    h[28..32].copy_from_slice(&(rate * 2).to_le_bytes()); // bytes per second
    h[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&UNKNOWN_LENGTH.to_le_bytes());
    h
}

/// Writes a stream's pieces to `w` as they arrive.
#[derive(Debug)]
pub struct StreamWriter<W: Write> {
    w: W,
    container: Container,
    rate: Option<u32>,
    samples: usize,
}

impl<W: Write> StreamWriter<W> {
    pub fn new(w: W, container: Container) -> Self {
        Self {
            w,
            container,
            rate: None,
            samples: 0,
        }
    }

    /// Write one piece at `rate`, the header first if this is the first piece, and flush. A later piece at another
    /// rate is refused: the container has one rate.
    pub fn write(&mut self, samples: &[f32], rate: u32) -> io::Result<()> {
        match self.rate {
            None => {
                match self.container {
                    Container::Wav => self.w.write_all(&wav_header(rate))?,
                    Container::RawS16 => self.w.write_all(raw_header(rate).as_bytes())?,
                }
                self.rate = Some(rate);
            }
            Some(r) if r != rate => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("a {rate} Hz piece in a {r} Hz stream"),
                ))
            }
            Some(_) => {}
        }
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &x in samples {
            bytes.extend_from_slice(&to_i16(x).to_le_bytes());
        }
        self.w.write_all(&bytes)?;
        self.w.flush()?;
        self.samples += samples.len();
        Ok(())
    }

    /// Samples written so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn into_inner(self) -> W {
        self.w
    }
}

/// Levels a stream's pieces for a listener, with one gain fixed from the first piece.
///
/// Why the first piece and not a per-profile constant: the level step's gain is a property of the voice and the
/// text, not of the engine (on the committed stories it is +3.96 dB for one Qwen3-TTS voice and -3.51 dB for one
/// Kitten-TTS 2 voice), so a constant is right for one voice only. The first piece must be complete before anything
/// is emitted anyway, so its gain costs no latency. If it has no sample above the floor, the gain is 1 (the samples
/// pass unchanged, as [`level`](crate::pipeline::level) leaves such a read alone) until a piece that has one.
#[derive(Debug, Clone)]
pub struct FirstChunkLevel {
    profile: Profile,
    gain: Option<f64>,
}

impl FirstChunkLevel {
    pub fn new(profile: &Profile) -> Self {
        Self {
            profile: *profile,
            gain: None,
        }
    }

    /// The gain held for the read, once a piece with a loud sample has set it.
    pub fn gain(&self) -> Option<f64> {
        self.gain
    }

    /// `samples` levelled with the read's gain, setting it from this piece if none is set yet.
    pub fn apply(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.gain.is_none() {
            self.gain = level_gain(&self.profile, samples);
        }
        let mut out = samples.to_vec();
        if let Some(g) = self.gain {
            apply_gain(&self.profile, &mut out, g);
        }
        out
    }
}

/// Levels a stream's pieces with one gain taken from at least the first `hold_s` seconds of audio.
///
/// Pieces are held back, whole and in order, until the held audio reaches `hold_s` seconds (always at least the
/// first piece). The gain is then computed over everything held, by the level step's own rule, the held pieces leave
/// levelled, and every later piece uses the same gain. A read shorter than `hold_s` is levelled at
/// [`finish`](Self::finish), over all of it. `hold_s = 0` is [`FirstChunkLevel`]: the first piece sets the gain.
///
/// Why hold: a first chunk unlike the rest sets the wrong gain for the whole read (#44). A loud first chunk makes
/// the stream too quiet; a quiet one makes it clamp every loud chunk after it. Holding more audio makes the gain
/// closer to the read's, at the cost of first-audio latency, which is the listener's trade to make.
#[derive(Debug, Clone)]
pub struct StreamLevel {
    profile: Profile,
    hold_s: f64,
    held: Vec<Vec<f32>>,
    held_samples: usize,
    rate: u32,
    gain: Option<f64>,
    decided: bool,
}

impl StreamLevel {
    pub fn new(profile: &Profile, hold_s: f64) -> Self {
        Self {
            profile: *profile,
            hold_s: hold_s.max(0.0),
            held: Vec::new(),
            held_samples: 0,
            rate: 0,
            gain: None,
            decided: false,
        }
    }

    /// The gain held for the read, once it is decided and the audio had a sample above the floor.
    pub fn gain(&self) -> Option<f64> {
        self.gain
    }

    /// Seconds of audio held when the gain was decided (or held now, before that).
    pub fn held_seconds(&self) -> f64 {
        if self.rate == 0 {
            0.0
        } else {
            self.held_samples as f64 / f64::from(self.rate)
        }
    }

    /// Take one piece; return the pieces ready to leave, levelled, in order. Empty while holding.
    pub fn push(&mut self, samples: &[f32], rate: u32) -> Vec<Vec<f32>> {
        if self.decided {
            return vec![self.level(samples)];
        }
        if self.rate == 0 {
            self.rate = rate;
        }
        self.held.push(samples.to_vec());
        self.held_samples += samples.len();
        if self.held_seconds() >= self.hold_s {
            self.decide()
        } else {
            Vec::new()
        }
    }

    /// The end of the read: decide over whatever is held (a read shorter than `hold_s`) and release it.
    pub fn finish(&mut self) -> Vec<Vec<f32>> {
        if self.decided || self.held.is_empty() {
            return Vec::new();
        }
        self.decide()
    }

    fn decide(&mut self) -> Vec<Vec<f32>> {
        let all: Vec<f32> = self.held.iter().flatten().copied().collect();
        self.gain = level_gain(&self.profile, &all);
        if self.gain.is_none() {
            // Nothing loud yet: release as is and decide on the next audio, as FirstChunkLevel does.
            self.held_samples = 0;
            return std::mem::take(&mut self.held);
        }
        self.decided = true;
        let held = std::mem::take(&mut self.held);
        held.iter().map(|p| self.level(p)).collect()
    }

    fn level(&self, samples: &[f32]) -> Vec<f32> {
        let mut out = samples.to_vec();
        if let Some(g) = self.gain {
            apply_gain(&self.profile, &mut out, g);
        }
        out
    }
}

/// The first index where `a` and `b` differ, bit for bit, or where one ends; `None` when they are identical.
pub fn first_sample_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i).map(|x| x.to_bits()) != b.get(i).map(|x| x.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wav_header_is_a_mono_16_bit_stream_of_unknown_length() {
        let h = wav_header(24_000);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), UNKNOWN_LENGTH);
        assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(h[28..32].try_into().unwrap()), 48_000);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes(h[40..44].try_into().unwrap()), UNKNOWN_LENGTH);
    }

    #[test]
    fn a_second_rate_is_refused() {
        let mut w = StreamWriter::new(Vec::new(), Container::RawS16);
        w.write(&[0.0; 4], 24_000).unwrap();
        assert!(w.write(&[0.0; 4], 22_050).is_err());
        assert_eq!(w.samples(), 4);
    }

    #[test]
    fn first_sample_diff_sees_a_sign_and_a_length() {
        assert_eq!(first_sample_diff(&[0.0, 1.0], &[0.0, 1.0]), None);
        assert_eq!(first_sample_diff(&[0.0, 0.0], &[0.0, -0.0]), Some(1));
        assert_eq!(first_sample_diff(&[0.0], &[0.0, 0.0]), Some(1));
    }
}
