//! Rendering a book through one engine, chapter by chapter, resumable.
//!
//! The engine is passed in already loaded and is used for every chunk of every chapter, so a book loads its model
//! once. Each chunk is generated at its planned seed ([`ChapterPlan::seed_at`], draw 0), decoded, trimmed at its
//! edges by the profile's rule, and stored as FLAC through [`run_chapter`], so a killed render resumes where it
//! stopped. A finished chapter is then joined from the stored chunks with the profile's pauses, levelled by the
//! story path's level step, and written as one FLAC.
//!
//! Chunks are joined from the stored files, never from memory, so a chapter rendered in one run and a chapter
//! resumed after a kill are built from the same bytes and come out identical.

use std::path::{Path, PathBuf};

use ggtts_core::audio::trim_edges_f32;
use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::flac::{encode_flac, write_flac};
use ggtts_core::pipeline::{gap_samples, level};
use ggtts_core::profile::Profile;

use crate::book::{Book, Chapter};
use crate::manifest::{chunk_file, run_chapter, Resume, RunError};
use crate::plan::ChapterPlan;

/// What happened to one chapter.
#[derive(Debug, Clone, PartialEq)]
pub enum ChapterOutcome {
    /// Rendered (or resumed) and written.
    Written {
        /// The chapter's FLAC.
        file: PathBuf,
        /// Chunks rendered in this run.
        rendered: Vec<usize>,
        /// Chunks taken from an earlier run.
        resumed: usize,
        /// Why an earlier manifest was not used, if one was refused.
        refused: Option<String>,
    },
    /// Front or back matter, left out.
    Skipped { reason: String },
}

/// A chapter's outcome, by its id.
#[derive(Debug, Clone, PartialEq)]
pub struct ChapterReport {
    pub id: String,
    pub outcome: ChapterOutcome,
}

/// Generate, decode and edge-trim one planned chunk at its first draw's seed; the FLAC bytes to store.
pub fn render_chunk(engine: &mut dyn Engine, plan: &ChapterPlan, index: usize) -> Result<Vec<u8>, String> {
    let chunk = &plan.chunks[index];
    let profile = *engine.profile();
    let req = ChunkRequest {
        text: &chunk.text,
        seed: u64::from(plan.seed_at(index, 0)),
        chunk: index as u32,
        draw: 0,
        instruction: None,
    };
    let generated = engine.generate(&req).map_err(|e| e.to_string())?;
    let pcm = engine.decode(&generated.codes).map_err(|e| e.to_string())?;
    let t = trim_edges_f32(&profile, &pcm.samples, pcm.sample_rate, profile.edge_s);
    let kept: Vec<i16> = pcm.samples[t.start_sample..t.end_sample]
        .iter()
        .map(|&x| to_i16(x))
        .collect();
    encode_flac(&kept, pcm.sample_rate).map_err(|e| e.to_string())
}

fn to_i16(x: f32) -> i16 {
    (f64::from(x).clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// The directory a chapter's chunks and manifest live in, and the chapter's FLAC beside it: `NNN-<id>`.
pub fn chapter_stem(index: usize, chapter: &Chapter) -> String {
    let id: String = chapter
        .id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{index:03}-{id}")
}

/// Join a finished chapter's stored chunks with the profile's pauses, level it, and write `out` (a `.flac` path).
pub fn join_chapter(profile: &Profile, dir: &Path, plan: &ChapterPlan, out: &Path) -> Result<PathBuf, String> {
    let mut read: Vec<f32> = Vec::new();
    let mut rate = None;
    for (i, chunk) in plan.chunks.iter().enumerate() {
        let path = dir.join(chunk_file(i));
        let mut reader = claxon::FlacReader::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sr = reader.streaminfo().sample_rate;
        if *rate.get_or_insert(sr) != sr {
            return Err(format!(
                "{}: sample rate {sr}, not {}",
                path.display(),
                rate.unwrap_or(sr)
            ));
        }
        for s in reader.samples() {
            let s = s.map_err(|e| format!("{}: {e}", path.display()))?;
            read.push(s as f32 / 32768.0);
        }
        if i + 1 < plan.chunks.len() {
            read.extend(std::iter::repeat_n(
                0.0,
                gap_samples(profile, &chunk.punct, chunk.last_in_para, sr),
            ));
        }
    }
    let rate = rate.ok_or_else(|| format!("chapter {} has no chunks", plan.chapter_id))?;
    level(profile, &mut read);
    write_flac(out, &read, rate).map_err(|e| format!("{}: {e}", out.display()))
}

/// Render every chapter of `book` into `out`, with `engine` loaded once for all of them.
///
/// Skippable chapters (front and back matter) are left out. A chapter whose chunk fails stops the render with that
/// chunk named; what was finished is kept, and the next run resumes from it.
pub fn render_book(
    engine: &mut dyn Engine,
    book: &Book,
    plans: &[ChapterPlan],
    out: &Path,
) -> Result<Vec<ChapterReport>, String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut reports = Vec::new();
    for (i, (chapter, plan)) in book.chapters.iter().zip(plans).enumerate() {
        if let Some(reason) = chapter.skip_reason.clone().filter(|_| chapter.skippable) {
            reports.push(ChapterReport {
                id: chapter.id.clone(),
                outcome: ChapterOutcome::Skipped { reason },
            });
            continue;
        }
        if plan.chunks.is_empty() {
            let reason = "nothing to speak".to_string();
            reports.push(ChapterReport {
                id: chapter.id.clone(),
                outcome: ChapterOutcome::Skipped { reason },
            });
            continue;
        }
        let stem = chapter_stem(i, chapter);
        let dir = out.join(&stem);
        let mut rendered = Vec::new();
        let start: Resume = run_chapter(&dir, plan, |p, index| {
            rendered.push(index);
            render_chunk(engine, p, index)
        })
        .map_err(|e| match e {
            RunError::Render { index, reason } => format!("chapter {}, chunk {index}: {reason}", chapter.id),
            RunError::Io(e) => format!("chapter {}: {e}", chapter.id),
        })?;
        let profile = *engine.profile();
        let file = join_chapter(&profile, &dir, plan, &out.join(format!("{stem}.flac")))?;
        reports.push(ChapterReport {
            id: chapter.id.clone(),
            outcome: ChapterOutcome::Written {
                file,
                resumed: plan.chunks.len() - start.todo.len(),
                rendered,
                refused: start.refused.map(|e| e.to_string()),
            },
        });
    }
    Ok(reports)
}
