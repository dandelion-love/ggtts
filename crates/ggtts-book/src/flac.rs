//! FLAC for books: tags on a chapter file, and a writer that streams a whole book into one file.
//!
//! ggtts writes FLAC with VERBATIM subframes in fixed blocks of 4096 samples, mono, 16-bit (`ggtts_core::flac`).
//! [`with_tags`] adds a VORBIS_COMMENT block to such a stream. [`StreamWriter`] writes the same format a block at a
//! time, so an export of many hours of audio never holds more than one block in memory. The frame layout and the
//! CRCs follow `ggtts_core::flac` and the FLAC format specification; the tests decode the result with `claxon`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const BLOCK: usize = 4096;
const VORBIS_COMMENT: u8 = 4;

/// The vendor string written in every VORBIS_COMMENT block.
pub fn vendor() -> String {
    format!("ggtts-book {}", ggtts_core::VERSION)
}

fn vorbis_comment(tags: &[(String, String)]) -> Vec<u8> {
    let vendor = vendor();
    let mut body = Vec::new();
    body.extend((vendor.len() as u32).to_le_bytes());
    body.extend(vendor.as_bytes());
    body.extend((tags.len() as u32).to_le_bytes());
    for (key, value) in tags {
        let field = format!("{key}={value}");
        body.extend((field.len() as u32).to_le_bytes());
        body.extend(field.as_bytes());
    }
    body
}

fn block_header(last: bool, kind: u8, len: usize) -> io::Result<[u8; 4]> {
    if len >= 1 << 24 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a FLAC metadata block is limited to 16 MiB",
        ));
    }
    let n = len as u32;
    Ok([(u8::from(last) << 7) | kind, (n >> 16) as u8, (n >> 8) as u8, n as u8])
}

/// `flac` with a VORBIS_COMMENT block holding `tags` added as its last metadata block. Any VORBIS_COMMENT already
/// there is replaced.
pub fn with_tags(flac: &[u8], tags: &[(String, String)]) -> io::Result<Vec<u8>> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    if flac.get(..4) != Some(b"fLaC") {
        return Err(bad("not a FLAC stream"));
    }
    let mut out = b"fLaC".to_vec();
    let mut at = 4;
    loop {
        let header = flac
            .get(at..at + 4)
            .ok_or_else(|| bad("a metadata block header is cut short"))?;
        let last = header[0] & 0x80 != 0;
        let kind = header[0] & 0x7f;
        let len = (usize::from(header[1]) << 16) | (usize::from(header[2]) << 8) | usize::from(header[3]);
        let body = flac
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| bad("a metadata block is cut short"))?;
        if kind != VORBIS_COMMENT {
            out.extend(block_header(false, kind, len)?);
            out.extend(body);
        }
        at += 4 + len;
        if last {
            break;
        }
    }
    let comment = vorbis_comment(tags);
    out.extend(block_header(true, VORBIS_COMMENT, comment.len())?);
    out.extend(comment);
    out.extend(&flac[at..]);
    Ok(out)
}

/// A mono 16-bit FLAC written a block at a time. The total length is declared up front (STREAMINFO carries it) and
/// checked at [`finish`](Self::finish).
pub struct StreamWriter {
    out: BufWriter<File>,
    block: Vec<i16>,
    frame: u64,
    written: u64,
    total: u64,
}

impl StreamWriter {
    /// Create `path` for `total` samples at `sample_rate`, with `tags` in a VORBIS_COMMENT block.
    pub fn create(path: &Path, sample_rate: u32, total: u64, tags: &[(String, String)]) -> io::Result<Self> {
        if sample_rate == 0 || sample_rate >= 1 << 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sample rate out of FLAC's range",
            ));
        }
        if total >= 1 << 36 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many samples for one FLAC stream",
            ));
        }
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(b"fLaC")?;
        let mut si = Bits::default();
        si.put(BLOCK as u64, 16);
        si.put(BLOCK as u64, 16);
        si.put(0, 24);
        si.put(0, 24);
        si.put(u64::from(sample_rate), 20);
        si.put(0, 3); // channels - 1
        si.put(15, 5); // bits per sample - 1
        si.put(total, 36);
        si.put(0, 64); // MD5 not computed: all zero means unknown
        si.put(0, 64);
        let streaminfo = si.bytes();
        out.write_all(&block_header(false, 0, streaminfo.len())?)?;
        out.write_all(&streaminfo)?;
        let comment = vorbis_comment(tags);
        out.write_all(&block_header(true, VORBIS_COMMENT, comment.len())?)?;
        out.write_all(&comment)?;
        Ok(StreamWriter {
            out,
            block: Vec::with_capacity(BLOCK),
            frame: 0,
            written: 0,
            total,
        })
    }

    /// Append samples.
    pub fn push(&mut self, samples: &[i16]) -> io::Result<()> {
        for &s in samples {
            self.block.push(s);
            if self.block.len() == BLOCK {
                self.flush_block()?;
            }
        }
        Ok(())
    }

    /// Append `n` samples of silence.
    pub fn push_silence(&mut self, n: usize) -> io::Result<()> {
        for _ in 0..n {
            self.push(&[0])?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.block.is_empty() {
            return Ok(());
        }
        let mut h = Bits::default();
        h.put(0x3ffe, 14);
        h.put(0, 1);
        h.put(0, 1); // fixed block size
        let size_code = if self.block.len() == BLOCK { 0b1100 } else { 0b0111 };
        h.put(size_code, 4);
        h.put(0, 4); // sample rate from STREAMINFO
        h.put(0, 4); // mono
        h.put(0b100, 3); // 16 bits per sample
        h.put(0, 1);
        let mut frame = h.bytes();
        frame.extend(utf8_number(self.frame));
        if size_code == 0b0111 {
            frame.extend(((self.block.len() - 1) as u16).to_be_bytes());
        }
        frame.push(crc8(&frame));
        frame.push(0b0000_0010); // pad bit, VERBATIM subframe, no wasted bits
        for &s in &self.block {
            frame.extend(s.to_be_bytes());
        }
        let crc = crc16(&frame);
        frame.extend(crc.to_be_bytes());
        self.out.write_all(&frame)?;
        self.written += self.block.len() as u64;
        self.frame += 1;
        self.block.clear();
        Ok(())
    }

    /// Write the last block and flush. Refuses a stream whose length is not the one declared.
    pub fn finish(mut self) -> io::Result<()> {
        self.flush_block()?;
        if self.written != self.total {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("wrote {} samples, declared {}", self.written, self.total),
            ));
        }
        self.out.flush()
    }
}

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn put(&mut self, value: u64, bits: u32) {
        for i in (0..bits).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                self.bytes.push(self.acc as u8);
                self.acc = 0;
                self.n = 0;
            }
        }
    }

    fn bytes(self) -> Vec<u8> {
        debug_assert_eq!(self.n, 0, "a FLAC header is a whole number of bytes");
        self.bytes
    }
}

fn utf8_number(v: u64) -> Vec<u8> {
    if v < 0x80 {
        return vec![v as u8];
    }
    let mut n = 2;
    while v >= 1u64 << (5 * n + 1) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_the_core_writer() {
        assert_eq!(utf8_number(0x7ff), vec![0xdf, 0xbf]);
        assert_eq!(utf8_number(0x800), vec![0xe0, 0xa0, 0x80]);
        assert_eq!(crc8(b"123456789"), 0xf4);
        assert_eq!(crc16(b"123456789"), 0xfee8);
    }

    #[test]
    fn a_stream_matches_the_core_encoder_frame_for_frame() {
        // The same samples through both writers give the same audio frames (only the metadata differs).
        let pcm: Vec<i16> = (0..10_000).map(|i| ((i * 37) % 2000 - 1000) as i16).collect();
        let core = ggtts_core::flac::encode_flac(&pcm, 24_000).unwrap();
        let dir = std::env::temp_dir().join(format!("ggtts-book-flac-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.flac");
        let mut w = StreamWriter::create(&path, 24_000, pcm.len() as u64, &[]).unwrap();
        w.push(&pcm[..5000]).unwrap();
        w.push(&pcm[5000..]).unwrap();
        w.finish().unwrap();
        let ours = std::fs::read(&path).unwrap();
        let frames = |f: &[u8]| -> Vec<u8> {
            let start = f.windows(2).position(|w| w == [0xff, 0xf8]).unwrap();
            f[start..].to_vec()
        };
        assert_eq!(frames(&ours), frames(&core));
        std::fs::remove_dir_all(&dir).ok();
    }
}
