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
