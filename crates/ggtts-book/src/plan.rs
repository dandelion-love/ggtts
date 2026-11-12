//! The chunk plan and the seed per chunk.
//!
//! A chapter's blocks are chunked by the engine's own chunker, [`ggtts_core::text::plan`], at the engine profile's
//! [`story_ceiling`](Profile::story_ceiling): there is no second chunker here and no ceiling of its own.
//!
//! Each chunk's seed is fixed by where it sits and what it says, so any one chunk can be rendered alone and gives the
//! same seed on every platform. See [`seed_key`] for the key.

use ggtts_core::profile::Profile;
use ggtts_core::seed::{hex, sha256};
use ggtts_core::text::{plan, validate_speech};
use serde_json::Value;

use crate::book::{Book, Chapter};

/// The seed root a plan uses when none is given: ggtts-render's default seed.
pub const DEFAULT_SEED_ROOT: &str = "20260905";

/// The version of the seed key. Changing the key's fields or their encoding is a new version, with new expected
/// seeds committed.
pub const SEED_KEY_VERSION: &str = "ggtts-book-seed/1";

/// The fields a chunk's seed is derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedKey<'a> {
    pub seed_root: &'a str,
    pub book_id: &'a str,
    pub chapter_id: &'a str,
    /// The block's index in the chapter.
    pub paragraph: usize,
    /// The chunk's index within its paragraph.
    pub chunk: usize,
    /// 0 for a first draw; a redraw of a chunk is draw 1, 2, ...
    pub draw: u32,
    /// The SHA-256 of the chunk's text, lower-case hex.
    pub text_sha256: &'a str,
}

/// The string a seed is hashed from: the version, then each field as `{byte length}:{value}`, joined by `|`.
///
/// The length prefix makes the encoding unambiguous: no two different keys give the same string, whatever their
/// fields contain (a `|` in a chapter id included).
pub fn seed_key(k: &SeedKey<'_>) -> String {
    let paragraph = k.paragraph.to_string();
    let chunk = k.chunk.to_string();
    let draw = k.draw.to_string();
    let fields = [
        k.seed_root,
        k.book_id,
        k.chapter_id,
        &paragraph,
        &chunk,
        &draw,
        k.text_sha256,
    ];
    let mut out = String::from(SEED_KEY_VERSION);
    for f in fields {
        out.push('|');
        out.push_str(&f.len().to_string());
        out.push(':');
        out.push_str(f);
    }
    out
}

/// The seed: the first four bytes of the SHA-256 of [`seed_key`], big-endian, as ggtts's own story seeds are read.
pub fn seed(k: &SeedKey<'_>) -> u32 {
    let d = sha256(seed_key(k).as_bytes());
    u32::from_be_bytes([d[0], d[1], d[2], d[3]])
}

/// The SHA-256 of a chunk's text, lower-case hex.
pub fn text_sha256(text: &str) -> String {
    hex(&sha256(text.as_bytes()))
}

/// One chunk to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedChunk {
    /// Its position in the chapter, counted from 0.
    pub index: usize,
    /// The block it comes from, as an index into the chapter's blocks.
    pub paragraph: usize,
    /// Its index within that block.
    pub chunk: usize,
    /// The text the engine reads: the chunk and its mark.
    pub text: String,
    /// The mark that ended it, or empty.
    pub punct: String,
    /// Whether it is its block's last chunk.
    pub last_in_para: bool,
    pub text_sha256: String,
    /// The first draw's seed.
    pub seed: u32,
}

/// Everything about one chapter's render that must match for a manifest to be trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterPlan {
    pub book_id: String,
    pub chapter_id: String,
    pub seed_root: String,
    /// The engine profile's name, for example `story-v1/qwen3-tts`.
    pub profile: String,
    /// The chunk ceiling the plan was made at: the profile's `story_ceiling`.
    pub ceiling: usize,
    pub chunks: Vec<PlannedChunk>,
    /// What renders the chunks: engine, backend, model files' SHA-256, speaker and ggtts build, as the renderer
    /// records them ([`render::provenance`](crate::render::provenance)). `Null` for a plan not yet tied to an engine.
    /// A manifest written by another engine is not resumed.
    pub engine: Value,
}

impl ChapterPlan {
    /// The seed of chunk `index` at draw `draw` (0 is the first draw, whose seed is [`PlannedChunk::seed`]).
    pub fn seed_at(&self, index: usize, draw: u32) -> u32 {
        let c = &self.chunks[index];
        seed(&SeedKey {
            seed_root: &self.seed_root,
            book_id: &self.book_id,
            chapter_id: &self.chapter_id,
            paragraph: c.paragraph,
            chunk: c.chunk,
            draw,
            text_sha256: &c.text_sha256,
        })
    }
}

/// Plan one chapter: chunk its blocks at the profile's ceiling and seed every chunk.
///
/// A block with nothing to speak (a scene break such as `* * *`) has no chunks; the block indices of the others
/// do not change.
pub fn plan_chapter(
    profile: &Profile,
    book_id: &str,
    chapter: &Chapter,
    seed_root: &str,
) -> Result<ChapterPlan, String> {
    let spoken: Vec<usize> = (0..chapter.blocks.len())
        .filter(|&i| validate_speech(chapter.blocks[i].text()).is_ok())
        .collect();
    let paragraphs: Vec<String> = spoken.iter().map(|&i| chapter.blocks[i].text().to_string()).collect();
    let planned =
        plan(profile, &paragraphs, profile.story_ceiling).map_err(|e| format!("chapter {}: {e}", chapter.id))?;
    let mut chunks = Vec::with_capacity(planned.len());
    let mut within = 0;
    let mut previous_para = usize::MAX;
    for (index, p) in planned.into_iter().enumerate() {
        if p.para != previous_para {
            within = 0;
            previous_para = p.para;
        }
        let paragraph = spoken[p.para];
        let sha = text_sha256(&p.text);
        let seed = seed(&SeedKey {
            seed_root,
            book_id,
            chapter_id: &chapter.id,
            paragraph,
            chunk: within,
            draw: 0,
            text_sha256: &sha,
        });
        chunks.push(PlannedChunk {
            index,
            paragraph,
            chunk: within,
            text: p.text,
            punct: p.punct,
            last_in_para: p.last_in_para,
            text_sha256: sha,
            seed,
        });
        within += 1;
    }
    Ok(ChapterPlan {
        book_id: book_id.to_string(),
        chapter_id: chapter.id.clone(),
        seed_root: seed_root.to_string(),
        profile: profile.name.to_string(),
        ceiling: profile.story_ceiling,
        chunks,
        engine: Value::Null,
    })
}

/// Plan every chapter of a book, skippable ones included (a renderer decides whether to leave them out).
pub fn plan_book(profile: &Profile, book: &Book, seed_root: &str) -> Result<Vec<ChapterPlan>, String> {
    let id = book.book_id();
    book.chapters
        .iter()
        .map(|c| plan_chapter(profile, &id, c, seed_root))
        .collect()
}

/// The profile an engine renders with, by the engine's name, as `ggtts-render --engine` names it.
///
/// Only profiles this binary is built with can plan: `qwen3-tts` and `fake` (the fake engine renders on the
/// Qwen3-TTS profile). Kitten-TTS 2's profile is compiled only into ggml builds of its engine, so it is refused here
/// rather than guessed.
pub fn engine_profile(engine: &str) -> Result<Profile, String> {
    match engine {
        "qwen3-tts" | "fake" => Ok(ggtts_qwen3_tts::PROFILE),
        "kitten-tts-2" => Err(
            "engine kitten-tts-2: its profile is compiled only with the ggml feature, which ggtts-book does not build \
             with yet"
                .to_string(),
        ),
        other => Err(format!("unknown engine {other:?}: expected qwen3-tts or fake")),
    }
}
