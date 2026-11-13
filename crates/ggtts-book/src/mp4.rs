//! An `.m4b` audiobook: lossless ALAC audio, the book's metadata, and chapters in the two forms players read.
//!
//! * **Audio:** ALAC frames of 4096 samples, mono, 16-bit, each written in ALAC's uncompressed ("escape") form, so
//!   the samples are the chapter files' own and no compressor is needed.
//! * **Chapters:** a QuickTime chapter text track (referenced from the audio track by `tref/chap`, the form Apple's
//!   players read) and a Nero `chpl` list in `moov/udta` (the form FFmpeg-based players read). `chpl` holds at most
//!   255 chapters; a longer book keeps the text track and leaves `chpl` out.
//! * **Metadata:** `©nam` (the book's title), `©alb`, one `©ART` per author, in `moov/udta/meta/ilst`.
//!
//! The file is `ftyp`, then `mdat` (the audio frames, then the chapter titles), then `moov`, so it is written in one
//! pass a frame at a time: every size is known before the audio is written.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Samples per ALAC frame.
pub const FRAME: u64 = 4096;

/// One chapter: its title and its first sample.
#[derive(Debug, Clone)]
pub struct Mark {
    pub title: String,
    pub start: u64,
}

/// The bytes of an ALAC frame of `n` samples in the escape form.
fn frame_bytes(n: u64) -> u64 {
    // tag 3 + instance 4 + unused 12 + partial 1 + shift 2 + escape 1, [32 for a partial frame's sample count],
    // 16 bits per sample, end tag 3, then padding to a byte.
    let bits = 23 + if n == FRAME { 0 } else { 32 } + 16 * n + 3;
    bits.div_ceil(8)
}

fn alac_frame(samples: &[i16]) -> Vec<u8> {
    let mut b = Bits::default();
    b.put(0, 3); // ID_SCE: a single channel element
    b.put(0, 4); // element instance tag
    b.put(0, 12); // unused
    let partial = samples.len() as u64 != FRAME;
    b.put(u64::from(partial), 1);
    b.put(0, 2); // bytes shifted
    b.put(1, 1); // escape: the samples follow uncompressed
    if partial {
        b.put(samples.len() as u64, 32);
    }
    for &s in samples {
        b.put(u64::from(s as u16), 16);
    }
    b.put(7, 3); // ID_END
    b.finish()
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

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.bytes.push((self.acc << (8 - self.n)) as u8);
        }
        self.bytes
    }
}

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend(((body.len() + 8) as u32).to_be_bytes());
    out.extend(kind);
    out.extend(body);
    out
}

fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version];
    b.extend(&flags.to_be_bytes()[1..]);
    b.extend(body);
    boxed(kind, &b)
}

fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn matrix() -> Vec<u8> {
    MATRIX.iter().flat_map(|v| v.to_be_bytes()).collect()
}

fn tkhd(id: u32, duration_ms: u64, audio: bool, enabled: bool) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u64.to_be_bytes()); // creation, modification (v1: 64-bit)
    b.extend(0u64.to_be_bytes());
    b.extend(id.to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(duration_ms.to_be_bytes());
    b.extend([0u8; 8]);
    b.extend(0u16.to_be_bytes()); // layer
    b.extend(0u16.to_be_bytes()); // alternate group
    b.extend(if audio { 0x0100u16 } else { 0 }.to_be_bytes()); // volume
    b.extend(0u16.to_be_bytes());
    b.extend(matrix());
    b.extend(0u32.to_be_bytes()); // width, height
    b.extend(0u32.to_be_bytes());
    full(b"tkhd", 1, if enabled { 3 } else { 2 }, &b)
}

fn mdhd(timescale: u32, duration: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0u64.to_be_bytes());
    b.extend(0u64.to_be_bytes());
    b.extend(timescale.to_be_bytes());
    b.extend(duration.to_be_bytes());
    b.extend(0x55c4u16.to_be_bytes()); // language: "und"
    b.extend(0u16.to_be_bytes());
    full(b"mdhd", 1, 0, &b)
}

fn hdlr(kind: &[u8; 4], name: &str) -> Vec<u8> {
    let mut b = vec![0u8; 4];
    b.extend(kind);
    b.extend([0u8; 12]);
    b.extend(name.as_bytes());
    b.push(0);
    full(b"hdlr", 0, 0, &b)
}

fn dinf() -> Vec<u8> {
    let url = full(b"url ", 0, 1, &[]);
    let mut dref = 1u32.to_be_bytes().to_vec();
    dref.extend(url);
    boxed(b"dinf", &full(b"dref", 0, 0, &dref))
}

/// `stts`, `stsc` (all samples in one chunk), `stsz` and `co64` for samples of the given durations and sizes.
fn sample_tables(durations: &[u32], sizes: &[u32], chunk_offset: u64) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        match runs.last_mut() {
            Some((count, delta)) if *delta == d => *count += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut stts = (runs.len() as u32).to_be_bytes().to_vec();
    for (count, delta) in &runs {
        stts.extend(count.to_be_bytes());
        stts.extend(delta.to_be_bytes());
    }
    let mut stsc = 1u32.to_be_bytes().to_vec();
    stsc.extend(1u32.to_be_bytes());
    stsc.extend((sizes.len() as u32).to_be_bytes());
    stsc.extend(1u32.to_be_bytes());
    let mut stsz = 0u32.to_be_bytes().to_vec();
    stsz.extend((sizes.len() as u32).to_be_bytes());
    for s in sizes {
        stsz.extend(s.to_be_bytes());
    }
    let mut co64 = 1u32.to_be_bytes().to_vec();
    co64.extend(chunk_offset.to_be_bytes());
    cat(&[
        full(b"stts", 0, 0, &stts),
        full(b"stsc", 0, 0, &stsc),
        full(b"stsz", 0, 0, &stsz),
        full(b"co64", 0, 0, &co64),
    ])
}

fn alac_entry(rate: u32) -> Vec<u8> {
    // The ALAC magic cookie (ALACSpecificConfig), in an `alac` full box inside the sample entry.
    let mut cookie = Vec::new();
    cookie.extend((FRAME as u32).to_be_bytes()); // frameLength
    cookie.push(0); // compatibleVersion
    cookie.push(16); // bitDepth
    cookie.extend([40, 10, 14]); // pb, mb, kb (the reference encoder's defaults)
    cookie.push(1); // numChannels
    cookie.extend(255u16.to_be_bytes()); // maxRun
    cookie.extend((frame_bytes(FRAME) as u32).to_be_bytes()); // maxFrameBytes
    cookie.extend((16 * rate).to_be_bytes()); // avgBitRate
    cookie.extend(rate.to_be_bytes());
    let mut e = vec![0u8; 6];
    e.extend(1u16.to_be_bytes()); // data reference index
    e.extend([0u8; 8]);
    e.extend(1u16.to_be_bytes()); // channels
    e.extend(16u16.to_be_bytes()); // sample size
    e.extend([0u8; 4]);
    e.extend(((rate.min(0xffff)) << 16).to_be_bytes()); // sample rate, 16.16
    e.extend(full(b"alac", 0, 0, &cookie));
    let mut stsd = 1u32.to_be_bytes().to_vec();
    stsd.extend(boxed(b"alac", &e));
    full(b"stsd", 0, 0, &stsd)
}

fn text_entry() -> Vec<u8> {
    // A QuickTime text sample entry, as Apple's chapter tracks carry.
    let mut e = vec![0u8; 6];
    e.extend(1u16.to_be_bytes());
    e.extend(1u32.to_be_bytes()); // display flags
    e.extend(1u32.to_be_bytes()); // text justification
    e.extend([0u8; 6]); // background colour
    e.extend([0u8; 8]); // default text box
    e.extend([0u8; 8]); // reserved
    e.extend(0u16.to_be_bytes()); // font number
    e.extend(0u16.to_be_bytes()); // font face
    e.push(0); // reserved
    e.extend(0u16.to_be_bytes());
    e.extend([0u8; 6]); // foreground colour
    e.push(0); // empty font name (Pascal string)
    let mut stsd = 1u32.to_be_bytes().to_vec();
    stsd.extend(boxed(b"text", &e));
    full(b"stsd", 0, 0, &stsd)
}

fn chapter_title(title: &str) -> Vec<u8> {
    let bytes = title.as_bytes();
    let mut s = (bytes.len() as u16).to_be_bytes().to_vec();
    s.extend(bytes);
    s
}

fn chpl(marks: &[Mark], rate: u32) -> Option<Vec<u8>> {
    if marks.len() > 255 {
        return None;
    }
    let mut b = vec![0u8; 4]; // the reserved word FFmpeg reads after version 1
    b.push(marks.len() as u8);
    for m in marks {
        // 100-nanosecond units.
        let t = (u128::from(m.start) * 10_000_000 / u128::from(rate)) as u64;
        b.extend(t.to_be_bytes());
        let mut end = m.title.len().min(255);
        while !m.title.is_char_boundary(end) {
            end -= 1;
        }
        b.push(end as u8);
        b.extend(&m.title.as_bytes()[..end]);
    }
    Some(full(b"chpl", 1, 0, &b))
}

fn ilst_item(kind: &[u8; 4], value: &str) -> Vec<u8> {
    let mut data = 1u32.to_be_bytes().to_vec(); // type: UTF-8
    data.extend(0u32.to_be_bytes()); // locale
    data.extend(value.as_bytes());
    boxed(kind, &boxed(b"data", &data))
}

/// The `.m4b` metadata.
pub struct Book<'a> {
    pub title: &'a str,
    pub authors: &'a [String],
}

/// Write an `.m4b` of `total` samples at `rate`, pulling audio from `fill` (called with a buffer to fill completely,
/// once per frame) and marking `marks`.
pub fn write_m4b(
    path: &Path,
    rate: u32,
    total: u64,
    marks: &[Mark],
    book: &Book<'_>,
    mut fill: impl FnMut(&mut [i16]) -> io::Result<()>,
) -> io::Result<()> {
    if total == 0 || marks.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "nothing to write"));
    }
    let frames = total.div_ceil(FRAME);
    let last = total - (frames - 1) * FRAME;
    let audio_bytes: u64 = (frames - 1) * frame_bytes(FRAME) + frame_bytes(last);
    let titles: Vec<Vec<u8>> = marks.iter().map(|m| chapter_title(&m.title)).collect();
    let text_bytes: u64 = titles.iter().map(|t| t.len() as u64).sum();

    let ftyp = {
        let mut b = b"M4B ".to_vec();
        b.extend(0u32.to_be_bytes());
        for brand in [b"M4B ", b"M4A ", b"mp42", b"isom"] {
            b.extend(brand);
        }
        boxed(b"ftyp", &b)
    };
    // mdat with a 64-bit size, so a long book is not limited to 4 GiB.
    let mdat_header = 16u64;
    let audio_offset = ftyp.len() as u64 + mdat_header;
    let text_offset = audio_offset + audio_bytes;

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&ftyp)?;
    out.write_all(&1u32.to_be_bytes())?;
    out.write_all(b"mdat")?;
    out.write_all(&(mdat_header + audio_bytes + text_bytes).to_be_bytes())?;
    let mut buf = vec![0i16; FRAME as usize];
    for f in 0..frames {
        let n = if f + 1 == frames { last } else { FRAME } as usize;
        fill(&mut buf[..n])?;
        out.write_all(&alac_frame(&buf[..n]))?;
    }
    for t in &titles {
        out.write_all(t)?;
    }

    // moov
    let duration_ms = total * 1000 / u64::from(rate);
    let mut sizes = vec![frame_bytes(FRAME) as u32; (frames - 1) as usize];
    sizes.push(frame_bytes(last) as u32);
    let mut durations = vec![FRAME as u32; (frames - 1) as usize];
    durations.push(last as u32);
    let audio_stbl = boxed(
        b"stbl",
        &cat(&[alac_entry(rate), sample_tables(&durations, &sizes, audio_offset)]),
    );
    let audio_trak = boxed(
        b"trak",
        &cat(&[
            tkhd(1, duration_ms, true, true),
            boxed(b"tref", &boxed(b"chap", &2u32.to_be_bytes())),
            boxed(
                b"mdia",
                &cat(&[
                    mdhd(rate, total),
                    hdlr(b"soun", "SoundHandler"),
                    boxed(b"minf", &cat(&[full(b"smhd", 0, 0, &[0; 4]), dinf(), audio_stbl])),
                ]),
            ),
        ]),
    );
    let text_durations: Vec<u32> = marks
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let end = marks.get(i + 1).map_or(total, |n| n.start);
            (end - m.start) as u32
        })
        .collect();
    let text_sizes: Vec<u32> = titles.iter().map(|t| t.len() as u32).collect();
    let text_stbl = boxed(
        b"stbl",
        &cat(&[text_entry(), sample_tables(&text_durations, &text_sizes, text_offset)]),
    );
    let gmhd = boxed(
        b"gmhd",
        &full(
            b"gmin",
            0,
            0,
            &[0, 0x40, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0, 0, 0, 0],
        ),
    );
    let text_trak = boxed(
        b"trak",
        &cat(&[
            tkhd(2, duration_ms, false, false),
            boxed(
                b"mdia",
                &cat(&[
                    mdhd(rate, total),
                    hdlr(b"text", "ChapterHandler"),
                    boxed(b"minf", &cat(&[gmhd, dinf(), text_stbl])),
                ]),
            ),
        ]),
    );
    let mvhd = {
        let mut b = Vec::new();
        b.extend(0u64.to_be_bytes());
        b.extend(0u64.to_be_bytes());
        b.extend(1000u32.to_be_bytes());
        b.extend(duration_ms.to_be_bytes());
        b.extend(0x0001_0000u32.to_be_bytes()); // rate 1.0
        b.extend(0x0100u16.to_be_bytes()); // volume 1.0
        b.extend([0u8; 10]);
        b.extend(matrix());
        b.extend([0u8; 24]);
        b.extend(3u32.to_be_bytes()); // next track id
        full(b"mvhd", 1, 0, &b)
    };
    let mut items = vec![ilst_item(b"\xa9nam", book.title), ilst_item(b"\xa9alb", book.title)];
    items.extend(book.authors.iter().map(|a| ilst_item(b"\xa9ART", a)));
    let mut meta_body = vec![0u8; 4];
    meta_body.extend(hdlr(b"mdir", ""));
    meta_body.extend(boxed(b"ilst", &items.concat()));
    let mut udta = boxed(b"meta", &meta_body);
    if let Some(c) = chpl(marks, rate) {
        udta.extend(c);
    }
    let moov = boxed(b"moov", &cat(&[mvhd, audio_trak, text_trak, boxed(b"udta", &udta)]));
    out.write_all(&moov)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sizes_match_the_frames() {
        let full: Vec<i16> = (0..FRAME as i16).collect();
        assert_eq!(alac_frame(&full).len() as u64, frame_bytes(FRAME));
        for n in [1u64, 7, 4095] {
            assert_eq!(alac_frame(&full[..n as usize]).len() as u64, frame_bytes(n), "{n}");
        }
    }

    #[test]
    fn chpl_truncates_long_titles_at_a_character() {
        let long = "霧".repeat(100); // 300 bytes
        let marks = [Mark { title: long, start: 0 }];
        let b = chpl(&marks, 24_000).unwrap();
        let len = b[8 + 4 + 4 + 1 + 8] as usize;
        assert_eq!(len, 255);
        let title = &b[8 + 4 + 4 + 1 + 8 + 1..];
        assert!(std::str::from_utf8(title).is_ok());
        assert!(chpl(
            &vec![
                Mark {
                    title: "x".into(),
                    start: 0
                };
                256
            ],
            24_000
        )
        .is_none());
    }
}
