//! One chaptered audiobook file from a rendered book's directory.
//!
//! `export` reads the chapter FLACs `render` wrote, in order, and streams them into one file with the book's tags and a
//! chapter mark per chapter. The container follows the output's extension: `.flac` carries the marks as Vorbis
//! comments `CHAPTER001=00:00:00.000` and `CHAPTER001NAME=<title>` (read as chapters by ffmpeg-based players such as
//! mpv and VLC); `.m4b` is ALAC audio with a QuickTime chapter track and a Nero `chpl` list, for audiobook apps
//! ([`crate::mp4`]). Chapters are separated by [`CHAPTER_GAP_S`] of silence. The audio
//! stays lossless: the samples are the chapter files' own.
//!
//! Nothing is exported from a directory that is not a finished render of one book: a chapter whose manifest has a
//! chunk not done, a chapter directory with no chapter file, chapters rendered with different seed roots or engines,
//! or a directory with no `book.json`, is refused, naming the chapter.

use std::io;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::flac::StreamWriter;
use crate::manifest::Manifest;
use crate::render::BOOK_MARKER;

/// Silence between two chapters in the export, in seconds.
pub const CHAPTER_GAP_S: f64 = 2.0;

struct ChapterFile {
    stem: String,
    file: PathBuf,
    title: String,
    samples: u64,
    rate: u32,
}

/// `HH:MM:SS.mmm` for a sample offset.
pub fn timestamp(sample: u64, rate: u32) -> String {
    let ms = sample * 1000 / u64::from(rate);
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

fn tag(reader: &claxon::FlacReader<std::fs::File>, name: &str) -> Vec<String> {
    reader.get_tag(name).map(String::from).collect()
}

/// The container an export is written in, chosen by the output's extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `.flac`: one FLAC with Vorbis-comment chapter marks.
    Flac,
    /// `.m4b`: ALAC audio with a chapter track and a `chpl` list ([`crate::mp4`]).
    M4b,
}

impl Format {
    /// The format for `out`'s extension, or a refusal naming the supported ones.
    pub fn for_path(out: &Path) -> Result<Format, String> {
        match out
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("flac") => Ok(Format::Flac),
            Some("m4b") => Ok(Format::M4b),
            _ => Err(format!(
                "{}: export writes .flac or .m4b, chosen by the extension",
                out.display()
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Flac => "flac",
            Format::M4b => "m4b",
        }
    }
}

/// The book's samples in order: each chapter file's own, with `gap` samples of silence between chapters, read a
/// FLAC block at a time.
struct Source<'a> {
    chapters: &'a [ChapterFile],
    gap: u64,
    next: usize,
    reader: Option<claxon::FlacReader<std::fs::File>>,
    queue: std::collections::VecDeque<i16>,
    silence: u64,
}

impl<'a> Source<'a> {
    fn new(chapters: &'a [ChapterFile], gap: u64) -> Self {
        Source {
            chapters,
            gap,
            next: 0,
            reader: None,
            queue: Default::default(),
            silence: 0,
        }
    }

    /// Fill `buf` completely; an error if the chapters run out first.
    fn fill(&mut self, buf: &mut [i16]) -> io::Result<()> {
        for slot in buf.iter_mut() {
            *slot = self
                .sample()?
                .ok_or_else(|| io::Error::other("the chapters ended before the declared length"))?;
        }
        Ok(())
    }

    fn sample(&mut self) -> io::Result<Option<i16>> {
        loop {
            if let Some(s) = self.queue.pop_front() {
                return Ok(Some(s));
            }
            if self.silence > 0 {
                self.silence -= 1;
                return Ok(Some(0));
            }
            if let Some(reader) = &mut self.reader {
                let mut blocks = reader.blocks();
                match blocks.read_next_or_eof(Vec::new()).map_err(io::Error::other)? {
                    Some(block) => self.queue.extend(block.channel(0).iter().map(|&s| s as i16)),
                    None => self.reader = None,
                }
                continue;
            }
            let Some(c) = self.chapters.get(self.next) else {
                return Ok(None);
            };
            if self.next > 0 {
                self.silence = self.gap;
            }
            self.next += 1;
            self.reader = Some(claxon::FlacReader::open(&c.file).map_err(io::Error::other)?);
        }
    }

    /// Refuse a source with samples left over: the declared length was too short.
    fn finish(mut self) -> io::Result<()> {
        match self.sample()? {
            None => Ok(()),
            Some(_) => Err(io::Error::other("the chapters hold more samples than declared")),
        }
    }
}

/// Export the rendered book in `dir` to `out` (`.flac` or `.m4b`). Returns a report of the chapters and their marks.
pub fn export(dir: &Path, out: &Path) -> Result<Value, String> {
    let format = Format::for_path(out)?;
    let marker = std::fs::read_to_string(dir.join(BOOK_MARKER)).map_err(|_| {
        format!(
            "{} has no {BOOK_MARKER}: not a directory ggtts-book render wrote",
            dir.display()
        )
    })?;
    let book_id = serde_json::from_str::<Value>(&marker)
        .ok()
        .and_then(|v| v.get("book_id").and_then(Value::as_str).map(String::from))
        .ok_or_else(|| {
            format!(
                "{} is not a book marker this tool wrote",
                dir.join(BOOK_MARKER).display()
            )
        })?;

    let mut stems: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.len() > 4 && name.as_bytes()[..3].iter().all(u8::is_ascii_digit) && &name[3..4] == "-")
        .collect();
    stems.sort();
    if stems.is_empty() {
        return Err(format!("{} holds no rendered chapters", dir.display()));
    }

    let mut chapters = Vec::new();
    let mut seed_root: Option<String> = None;
    let mut engine: Option<Value> = None;
    let mut album = String::new();
    let mut artists: Vec<String> = Vec::new();
    for stem in &stems {
        let body = Manifest::read_body(&dir.join(stem)).map_err(|e| format!("chapter {stem}: {e}"))?;
        if body["book_id"].as_str() != Some(book_id.as_str()) {
            return Err(format!("chapter {stem} belongs to another book"));
        }
        let chunks = body["chunks"].as_array().cloned().unwrap_or_default();
        if let Some(c) = chunks.iter().find(|c| c["state"] != "done") {
            return Err(format!(
                "chapter {stem} is not fully rendered: chunk {} is not done; run render again to finish it",
                c["index"]
            ));
        }
        let root = body["seed_root"].as_str().unwrap_or_default().to_string();
        if seed_root.get_or_insert_with(|| root.clone()) != &root {
            return Err(format!(
                "chapter {stem} was rendered with seed root {root:?}, the chapters before it with {:?}",
                seed_root.unwrap_or_default()
            ));
        }
        let this_engine = body["engine"].clone();
        if engine.get_or_insert_with(|| this_engine.clone()) != &this_engine {
            return Err(format!(
                "chapter {stem} was rendered by another engine than the chapters before it"
            ));
        }
        let file = dir.join(format!("{stem}.flac"));
        let reader = claxon::FlacReader::open(&file)
            .map_err(|e| format!("chapter {stem} has no chapter file {}: {e}", file.display()))?;
        let info = reader.streaminfo();
        if info.channels != 1 || info.bits_per_sample != 16 {
            return Err(format!("{}: not mono 16-bit", file.display()));
        }
        let samples = info
            .samples
            .ok_or_else(|| format!("{}: its length is not recorded", file.display()))?;
        if album.is_empty() {
            album = tag(&reader, "ALBUM").into_iter().next().unwrap_or_default();
            artists = tag(&reader, "ARTIST");
        }
        let title = tag(&reader, "TITLE").into_iter().next().unwrap_or_else(|| stem.clone());
        chapters.push(ChapterFile {
            stem: stem.clone(),
            file,
            title,
            samples,
            rate: info.sample_rate,
        });
    }
    let rate = chapters[0].rate;
    if let Some(c) = chapters.iter().find(|c| c.rate != rate) {
        return Err(format!(
            "chapter {} is at {} Hz, the first chapter at {rate} Hz",
            c.stem, c.rate
        ));
    }
    let gap = (CHAPTER_GAP_S * f64::from(rate)) as u64;
    let total: u64 = chapters.iter().map(|c| c.samples).sum::<u64>() + gap * (chapters.len() as u64 - 1);

    let mut tags = vec![
        ("TITLE".to_string(), album.clone()),
        ("ALBUM".to_string(), album.clone()),
    ];
    tags.extend(artists.iter().map(|a| ("ARTIST".to_string(), a.clone())));
    tags.push(("GGTTS_BOOK_ID".to_string(), book_id.clone()));
    tags.push(("GGTTS_SEED_ROOT".to_string(), seed_root.clone().unwrap_or_default()));
    let mut marks = Vec::new();
    let mut mark_starts = Vec::new();
    let mut at = 0u64;
    for (i, c) in chapters.iter().enumerate() {
        let n = i + 1;
        tags.push((format!("CHAPTER{n:03}"), timestamp(at, rate)));
        tags.push((format!("CHAPTER{n:03}NAME"), c.title.clone()));
        marks.push(json!({"chapter": c.stem, "title": c.title, "start_sample": at, "start": timestamp(at, rate)}));
        mark_starts.push(at);
        at += c.samples + gap;
    }

    let partial = out.with_file_name(format!(
        ".{}.partial",
        out.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    let mut source = Source::new(&chapters, gap);
    let failed = |e: io::Error| format!("{}: {e}", out.display());
    match format {
        Format::Flac => {
            let mut w = StreamWriter::create(&partial, rate, total, &tags).map_err(failed)?;
            let mut block = vec![0i16; 4096];
            let mut left = total;
            while left > 0 {
                let n = left.min(4096) as usize;
                source.fill(&mut block[..n]).map_err(failed)?;
                w.push(&block[..n]).map_err(failed)?;
                left -= n as u64;
            }
            w.finish().map_err(failed)?;
        }
        Format::M4b => {
            let mp4_marks: Vec<crate::mp4::Mark> = mark_starts
                .iter()
                .zip(&chapters)
                .map(|(&start, c)| crate::mp4::Mark {
                    title: c.title.clone(),
                    start,
                })
                .collect();
            let book = crate::mp4::Book {
                title: &album,
                authors: &artists,
            };
            crate::mp4::write_m4b(&partial, rate, total, &mp4_marks, &book, |buf| source.fill(buf)).map_err(failed)?;
        }
    }
    source.finish().map_err(failed)?;
    std::fs::rename(&partial, out).map_err(failed)?;
    Ok(json!({
        "format": "ggtts-book-export/1",
        "container": format.name(),
        "book_id": book_id,
        "file": out.to_string_lossy(),
        "sample_rate": rate,
        "samples": total,
        "chapter_gap_s": CHAPTER_GAP_S,
        "chapters": marks,
    }))
}

#[cfg(test)]
mod tests {
    use super::timestamp;

    #[test]
    fn timestamps() {
        assert_eq!(timestamp(0, 24_000), "00:00:00.000");
        assert_eq!(timestamp(24_000 * 3_725 + 12_000, 24_000), "01:02:05.500");
    }
}
