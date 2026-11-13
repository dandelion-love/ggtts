//! Issue 226: the `.m4b` export. These tests read the file back without any outside tool: they walk the MP4 boxes,
//! decode the ALAC frames (written in ALAC's uncompressed "escape" form), and read the chapter lists. The opt-in test
//! at the end checks the same file with FFmpeg's `ffprobe` and `ffmpeg` when they are installed
//! (`cargo test -p ggtts-book --test m4b -- --ignored`).

mod common;

use std::path::{Path, PathBuf};

use common::*;
use ggtts_book::export::export;
use ggtts_book::render::{render_command, LoadedEngine};
use ggtts_fake::FakeEngine;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-m4b-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rendered(d: &Path) -> PathBuf {
    let book = nested().write_temp(&format!("m4b-{}", d.file_name().unwrap().to_string_lossy()));
    let out = d.join("out");
    render_command(&book, &out, "root", || {
        Ok(LoadedEngine {
            engine: Box::new(FakeEngine::default()),
            files: vec![],
            speaker: None,
        })
    })
    .unwrap();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
    out
}

/// The boxes directly inside `data`: (type, body).
fn boxes(data: &[u8]) -> Vec<(String, &[u8])> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 8 <= data.len() {
        let size32 = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        // Box types are four bytes read as Latin-1: `©nam` begins with 0xA9.
        let kind: String = data[at + 4..at + 8].iter().map(|&b| char::from(b)).collect();
        let (header, size) = if size32 == 1 {
            (
                16,
                u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize,
            )
        } else {
            (8, size32)
        };
        out.push((kind, &data[at + header..at + size]));
        at += size;
    }
    out
}

fn child<'a>(data: &'a [u8], kind: &str) -> &'a [u8] {
    boxes(data)
        .into_iter()
        .find(|(k, _)| k == kind)
        .unwrap_or_else(|| panic!("no {kind} box"))
        .1
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

/// Decode one escape-form ALAC frame (mono, 16-bit) into samples.
fn alac_frame(frame: &[u8]) -> Vec<i16> {
    let mut bits = frame.iter().flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1));
    let mut read = |n: u32| (0..n).fold(0u64, |acc, _| (acc << 1) | u64::from(bits.next().unwrap()));
    assert_eq!(read(3), 0, "a single channel element");
    read(4);
    assert_eq!(read(12), 0, "unused header bits");
    let partial = read(1) == 1;
    assert_eq!(read(2), 0, "no bytes shifted");
    assert_eq!(read(1), 1, "the escape (uncompressed) form");
    let n = if partial { read(32) as usize } else { 4096 };
    let samples = (0..n).map(|_| read(16) as u16 as i16).collect();
    assert_eq!(read(3), 7, "the end element");
    samples
}

struct M4b {
    samples: Vec<i16>,
    rate: u32,
    text_titles: Vec<String>,
    text_durations: Vec<u32>,
    chpl: Vec<(u64, String)>,
    tags: Vec<(String, String)>,
    brand: String,
}

/// Read the stbl tables of a track: (stts runs, sizes, the single chunk offset).
fn tables(stbl: &[u8]) -> (Vec<(u32, u32)>, Vec<u32>, u64) {
    let stts = child(stbl, "stts");
    let runs = (0..u32_at(stts, 4) as usize)
        .map(|i| (u32_at(stts, 8 + 8 * i), u32_at(stts, 12 + 8 * i)))
        .collect();
    let stsz = child(stbl, "stsz");
    let sizes = (0..u32_at(stsz, 8) as usize)
        .map(|i| u32_at(stsz, 12 + 4 * i))
        .collect();
    let co64 = child(stbl, "co64");
    assert_eq!(u32_at(co64, 4), 1, "one chunk");
    (runs, sizes, u64::from_be_bytes(co64[8..16].try_into().unwrap()))
}

fn read_m4b(path: &Path) -> M4b {
    let file = std::fs::read(path).unwrap();
    let top = boxes(&file);
    let kinds: Vec<&str> = top.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(kinds, ["ftyp", "mdat", "moov"]);
    let brand = String::from_utf8_lossy(&top[0].1[..4]).into_owned();
    let moov = top[2].1;
    let traks: Vec<&[u8]> = boxes(moov)
        .into_iter()
        .filter(|(k, _)| k == "trak")
        .map(|(_, b)| b)
        .collect();
    assert_eq!(traks.len(), 2, "an audio track and a chapter track");
    // The audio track points at the chapter track.
    assert_eq!(u32_at(child(child(traks[0], "tref"), "chap"), 0), 2);

    let mdia = child(traks[0], "mdia");
    let mdhd = child(mdia, "mdhd");
    let rate = u32_at(mdhd, 20);
    let stbl = child(child(mdia, "minf"), "stbl");
    let stsd = child(stbl, "stsd");
    let entry = boxes(&stsd[8..]);
    assert_eq!(entry[0].0, "alac");
    let (runs, sizes, offset) = tables(stbl);
    let mut samples = Vec::new();
    let mut at = offset as usize;
    for size in &sizes {
        samples.extend(alac_frame(&file[at..at + *size as usize]));
        at += *size as usize;
    }
    let declared: u64 = runs.iter().map(|(c, d)| u64::from(*c) * u64::from(*d)).sum();
    assert_eq!(declared, samples.len() as u64, "stts covers every sample");

    let tstbl = child(child(child(traks[1], "mdia"), "minf"), "stbl");
    let (truns, tsizes, toffset) = tables(tstbl);
    let mut text_titles = Vec::new();
    let mut at = toffset as usize;
    for size in &tsizes {
        let n = u16::from_be_bytes([file[at], file[at + 1]]) as usize;
        text_titles.push(String::from_utf8(file[at + 2..at + 2 + n].to_vec()).unwrap());
        at += *size as usize;
    }
    let text_durations = truns
        .iter()
        .flat_map(|(c, d)| std::iter::repeat_n(*d, *c as usize))
        .collect();

    let udta = child(moov, "udta");
    let chpl_box = child(udta, "chpl");
    let count = chpl_box[8] as usize;
    let mut chpl = Vec::new();
    let mut at = 9;
    for _ in 0..count {
        let t = u64::from_be_bytes(chpl_box[at..at + 8].try_into().unwrap());
        let n = chpl_box[at + 8] as usize;
        chpl.push((t, String::from_utf8(chpl_box[at + 9..at + 9 + n].to_vec()).unwrap()));
        at += 9 + n;
    }
    let meta = child(udta, "meta");
    let ilst = child(&meta[4..], "ilst");
    let tags = boxes(ilst)
        .into_iter()
        .map(|(k, b)| {
            let data = child(b, "data");
            (k, String::from_utf8(data[8..].to_vec()).unwrap())
        })
        .collect();
    M4b {
        samples,
        rate,
        text_titles,
        text_durations,
        chpl,
        tags,
        brand,
    }
}

fn flac_samples(path: &Path) -> Vec<i16> {
    let mut r = claxon::FlacReader::open(path).unwrap();
    r.samples().map(|s| s.unwrap() as i16).collect()
}

/// The `.m4b` holds exactly the FLAC export's samples, its chapters in both forms at the FLAC export's marks, and the
/// book's tags.
#[test]
fn m4b_matches_the_flac_export() {
    let d = dir("match");
    let out = rendered(&d);
    let flac = export(&out, &d.join("book.flac")).unwrap();
    let report = export(&out, &d.join("book.m4b")).unwrap();
    assert_eq!(report["container"], "m4b");
    let m = read_m4b(&d.join("book.m4b"));
    assert_eq!(m.brand, "M4B ");
    assert_eq!(
        m.samples,
        flac_samples(&d.join("book.flac")),
        "the audio is not the FLAC export's"
    );
    assert_eq!(m.rate, flac["sample_rate"].as_u64().unwrap() as u32);

    let marks = flac["chapters"].as_array().unwrap();
    let titles: Vec<String> = marks.iter().map(|c| c["title"].as_str().unwrap().to_string()).collect();
    let starts: Vec<u64> = marks.iter().map(|c| c["start_sample"].as_u64().unwrap()).collect();
    assert_eq!(m.text_titles, titles);
    let mut text_starts = vec![0u64];
    for d in &m.text_durations[..m.text_durations.len() - 1] {
        text_starts.push(text_starts.last().unwrap() + u64::from(*d));
    }
    assert_eq!(
        text_starts, starts,
        "the chapter track's samples start where the chapters do"
    );
    assert_eq!(
        m.text_durations.iter().map(|d| u64::from(*d)).sum::<u64>(),
        m.samples.len() as u64
    );
    let chpl_titles: Vec<String> = m.chpl.iter().map(|(_, t)| t.clone()).collect();
    assert_eq!(chpl_titles, titles);
    for ((t, _), start) in m.chpl.iter().zip(&starts) {
        assert_eq!(*t, start * 10_000_000 / u64::from(m.rate), "chpl is in 100 ns units");
    }
    assert!(m
        .tags
        .contains(&("\u{a9}nam".to_string(), "The Lighthouse Keeper's Almanac".to_string())));
    let artists: Vec<&str> = m
        .tags
        .iter()
        .filter(|(k, _)| k == "\u{a9}ART")
        .map(|(_, v)| v.as_str())
        .collect();
    assert_eq!(artists, ["Ada Test-Writer", "Bo Fixture"]);
    std::fs::remove_dir_all(&d).ok();
}

/// The same render exports to the same `.m4b` bytes.
#[test]
fn m4b_is_byte_identical() {
    let d = dir("same");
    let out = rendered(&d);
    export(&out, &d.join("a.m4b")).unwrap();
    export(&out, &d.join("b.m4b")).unwrap();
    assert!(std::fs::read(d.join("a.m4b")).unwrap() == std::fs::read(d.join("b.m4b")).unwrap());
    std::fs::remove_dir_all(&d).ok();
}

/// AC3: an unknown extension is refused, naming the two supported, before anything is read or written; a refusal
/// that holds for `.flac` holds for `.m4b`.
#[test]
fn unknown_extensions_and_unfinished_renders_are_refused() {
    let d = dir("refuse");
    let out = rendered(&d);
    let err = export(&out, &d.join("book.mp3")).unwrap_err();
    assert!(err.contains(".flac or .m4b"), "{err}");
    assert!(!d.join("book.mp3").exists());
    let err = export(&d.join("nowhere"), &d.join("book.m4b")).unwrap_err();
    assert!(err.contains("book.json"), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// The same file, read by FFmpeg: the chapters, the codec, and decoded samples equal to the FLAC export's.
/// Opt-in: needs `ffprobe` and `ffmpeg` on PATH.
#[test]
#[ignore = "needs ffprobe and ffmpeg; run with --ignored"]
fn m4b_reads_back_with_ffmpeg() {
    use std::process::Command;
    let d = dir("ffmpeg");
    let out = rendered(&d);
    export(&out, &d.join("book.flac")).unwrap();
    export(&out, &d.join("book.m4b")).unwrap();
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_chapters", "-show_streams", "-of", "json"])
        .arg(d.join("book.m4b"))
        .output()
        .expect("ffprobe on PATH");
    assert!(probe.status.success(), "{}", String::from_utf8_lossy(&probe.stderr));
    let json: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    assert_eq!(json["streams"][0]["codec_name"], "alac");
    assert_eq!(json["chapters"].as_array().unwrap().len(), 6);
    assert_eq!(json["chapters"][1]["tags"]["title"], "1. Low Tide");
    let pcm = |p: PathBuf| {
        let o = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(p)
            .args(["-f", "s16le", "-"])
            .output()
            .unwrap();
        assert!(o.status.success());
        o.stdout
    };
    assert!(
        pcm(d.join("book.m4b")) == pcm(d.join("book.flac")),
        "FFmpeg decodes different samples"
    );
    std::fs::remove_dir_all(&d).ok();
}
