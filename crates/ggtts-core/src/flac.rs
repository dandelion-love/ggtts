//! FLAC output (render spec section 8): mono, 16-bit PCM, whatever suffix the caller asked for.
//!
//! The encoder writes VERBATIM subframes: valid FLAC that any decoder reads, lossless, but not compressed. A
//! predictive encoder is a later change; the samples, and so the parity cases, do not depend on it.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Samples per FLAC block.
const BLOCK: usize = 4096;

/// One float sample to 16-bit PCM: scaled by 32767, rounded half to even, saturated.
pub fn to_i16(x: f32) -> i16 {
    (f64::from(x) * 32767.0)
        .round_ties_even()
        .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

/// Write `samples` as FLAC, PCM_16, mono, at `sample_rate`. The suffix is forced to `.flac`; the path written is
/// returned.
pub fn write_flac(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<PathBuf> {
    let out = path.with_extension("flac");
    let pcm: Vec<i16> = samples.iter().map(|&x| to_i16(x)).collect();
    let bytes = encode_flac(&pcm, sample_rate)?;
    let mut f = std::fs::File::create(&out)?;
    f.write_all(&bytes)?;
    f.flush()?;
    Ok(out)
}

struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }
    fn put(&mut self, value: u64, bits: u32) {
        for i in (0..bits).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                self.out.push(self.acc as u8);
                self.acc = 0;
                self.n = 0;
            }
        }
    }
    fn bytes(self) -> Vec<u8> {
        assert_eq!(self.n, 0, "FLAC structures end on a byte boundary");
        self.out
    }
}

fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 };
        }
    }
    crc
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// The frame number in FLAC's UTF-8-like coding.
fn utf8_number(v: u64) -> Vec<u8> {
    if v < 0x80 {
        return vec![v as u8];
    }
    let mut n = 2;
    while n < 7 && v >= 1u64 << (5 * n + 1) {
        n += 1;
    }
    let mut out = vec![0u8; n];
    let mut rest = v;
    for i in (1..n).rev() {
        out[i] = 0x80 | (rest & 0x3f) as u8;
        rest >>= 6;
    }
    out[0] = ((0xff00u16 >> n) as u8) | rest as u8;
    out
}

/// Encode mono 16-bit PCM as a FLAC stream.
pub fn encode_flac(pcm: &[i16], sample_rate: u32) -> io::Result<Vec<u8>> {
    if sample_rate == 0 || sample_rate >= 1 << 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sample rate out of FLAC's range",
        ));
    }
    let mut out = b"fLaC".to_vec();
    // STREAMINFO, the last metadata block.
    let mut si = Bits::new();
    si.put(1, 1);
    si.put(0, 7);
    si.put(34, 24);
    si.put(BLOCK as u64, 16);
    si.put(BLOCK as u64, 16);
    si.put(0, 24);
    si.put(0, 24);
    si.put(u64::from(sample_rate), 20);
    si.put(0, 3); // channels - 1
    si.put(15, 5); // bits per sample - 1
    si.put(pcm.len() as u64, 36);
    si.put(0, 64); // MD5 not computed: all zero means unknown
    si.put(0, 64);
    out.extend(si.bytes());

    for (index, block) in pcm.chunks(BLOCK).enumerate() {
        let mut h = Bits::new();
        h.put(0x3ffe, 14);
        h.put(0, 1);
        h.put(0, 1); // fixed block size
        let size_code = if block.len() == BLOCK { 0b1100 } else { 0b0111 };
        h.put(size_code, 4);
        h.put(0, 4); // sample rate from STREAMINFO
        h.put(0, 4); // mono
        h.put(0b100, 3); // 16 bits per sample
        h.put(0, 1);
        let mut header = h.bytes();
        header.extend(utf8_number(index as u64));
        if size_code == 0b0111 {
            header.extend(((block.len() - 1) as u16).to_be_bytes());
        }
        header.push(crc8(&header));

        let mut frame = header;
        frame.push(0b0000_0010); // pad bit, VERBATIM subframe (000001), no wasted bits
        for &s in block {
            frame.extend(s.to_be_bytes());
        }
        let crc = crc16(&frame);
        frame.extend(crc.to_be_bytes());
        out.extend(frame);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_numbers() {
        assert_eq!(utf8_number(0), vec![0]);
        assert_eq!(utf8_number(0x7f), vec![0x7f]);
        assert_eq!(utf8_number(0x80), vec![0xc2, 0x80]);
        assert_eq!(utf8_number(0x7ff), vec![0xdf, 0xbf]);
        assert_eq!(utf8_number(0x800), vec![0xe0, 0xa0, 0x80]);
    }

    #[test]
    fn crcs_match_flac_reference_values() {
        // CRC-8 (poly 0x07) and CRC-16 (poly 0x8005), both with zero init, of "123456789".
        assert_eq!(crc8(b"123456789"), 0xf4);
        assert_eq!(crc16(b"123456789"), 0xfee8);
    }
}
