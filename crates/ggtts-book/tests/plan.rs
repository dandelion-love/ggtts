//! Story 2: the chunk plan and the seed per chunk.

mod common;

use common::*;
use ggtts_book::plan::{
    engine_profile, plan_book, plan_chapter, seed, seed_key, text_sha256, SeedKey, DEFAULT_SEED_ROOT,
};
use ggtts_book::{Block, Book, Chapter};
use ggtts_core::profile::Profile;
use ggtts_qwen3_tts::PROFILE;

fn book() -> Book {
    Book::read(nested().bytes()).unwrap()
}

fn chapter<'a>(book: &'a Book, id: &str) -> &'a Chapter {
    book.chapters.iter().find(|c| c.id == id).unwrap()
}

const BASE: SeedKey<'static> = SeedKey {
    seed_root: "20260905",
    book_id: "urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d",
    chapter_id: "ch1",
    paragraph: 1,
    chunk: 0,
    draw: 0,
    text_sha256: "6a6e0c7b1f0e5bd1f7f3a3b4f53d7c2b9c0c5a3e7d1e9d2a4f8b6c0e1d2f3a4b",
};

/// AC2: the key's exact encoding, with its version and each field length-prefixed.
#[test]
fn the_seed_key_is_versioned_and_length_prefixed() {
    assert_eq!(
        seed_key(&BASE),
        "ggtts-book-seed/1|8:20260905|45:urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d|3:ch1|1:1|1:0|1:0|\
         64:6a6e0c7b1f0e5bd1f7f3a3b4f53d7c2b9c0c5a3e7d1e9d2a4f8b6c0e1d2f3a4b"
    );
    // A boundary moved between two fields is a different key.
    let a = SeedKey {
        seed_root: "ab",
        book_id: "c",
        ..BASE
    };
    let b = SeedKey {
        seed_root: "a",
        book_id: "bc",
        ..BASE
    };
    assert_ne!(seed_key(&a), seed_key(&b));
    assert_ne!(seed(&a), seed(&b));
    // A separator inside a field cannot forge another field.
    let c = SeedKey {
        chapter_id: "ch1|1:1",
        paragraph: 1,
        ..BASE
    };
    let d = SeedKey {
        chapter_id: "ch1",
        paragraph: 1,
        ..BASE
    };
    assert_ne!(seed_key(&c), seed_key(&d));
}

/// AC3: the same inputs give these seeds on every platform. CI runs this on Linux and Windows.
#[test]
fn committed_seeds() {
    assert_eq!(seed(&BASE), EXPECTED_BASE_SEED);
    let plans = plan_book(&PROFILE, &book(), DEFAULT_SEED_ROOT).unwrap();
    let firsts: Vec<(String, u32)> = plans
        .iter()
        .map(|p| (p.chapter_id.clone(), p.chunks.first().map(|c| c.seed).unwrap_or(0)))
        .collect();
    let expected: Vec<(String, u32)> = EXPECTED_FIRST_SEEDS
        .iter()
        .map(|(id, s)| (id.to_string(), *s))
        .collect();
    assert_eq!(firsts, expected);
}

/// Checked against an independent implementation when committed: Python's `hashlib.sha256` of the key string above,
/// first four bytes big-endian.
const EXPECTED_BASE_SEED: u32 = 893500943;
/// The first chunk's seed in each chapter of the nested fixture at seed root `20260905` on the Qwen3-TTS profile.
const EXPECTED_FIRST_SEEDS: &[(&str, u32)] = &[
    ("titlepage", 2081165468),
    ("copyright", 2130566388),
    ("nav", 1563784238),
    ("part1", 3735897234),
    ("ch1", 3621667395),
    ("ch2a", 3242085839),
    ("part2", 1421272188),
    ("part2#c3", 3257862672),
    ("part2#c4", 3636425970),
    ("colophon", 1748389444),
];

/// AC4: changing any one field of the key changes the seed.
#[test]
fn every_field_changes_the_seed() {
    let variants = [
        (
            "seed root",
            SeedKey {
                seed_root: "20260906",
                ..BASE
            },
        ),
        (
            "book id",
            SeedKey {
                book_id: "urn:uuid:another",
                ..BASE
            },
        ),
        (
            "chapter",
            SeedKey {
                chapter_id: "ch2a",
                ..BASE
            },
        ),
        ("paragraph", SeedKey { paragraph: 2, ..BASE }),
        ("chunk", SeedKey { chunk: 1, ..BASE }),
        ("draw", SeedKey { draw: 1, ..BASE }),
        (
            "text",
            SeedKey {
                text_sha256: &"0".repeat(64),
                ..BASE
            },
        ),
    ];
    let base = seed(&BASE);
    let mut seen = vec![base];
    for (field, key) in &variants {
        let s = seed(key);
        assert_ne!(s, base, "changing the {field} left the seed at {base}");
        assert!(!seen.contains(&s), "the {field} change collides with another field's");
        seen.push(s);
    }
}

/// AC4 through the plan: a chunk's seed follows its text and its place, and a redraw has its own seed.
#[test]
fn plan_seeds_follow_text_and_place() {
    let book = book();
    let ch1 = chapter(&book, "ch1");
    let plan = plan_chapter(&PROFILE, "id", ch1, "root").unwrap();
    let c = &plan.chunks[1];
    assert_eq!(c.text_sha256, text_sha256(&c.text));
    assert_eq!(
        c.seed,
        seed(&SeedKey {
            seed_root: "root",
            book_id: "id",
            chapter_id: "ch1",
            paragraph: c.paragraph,
            chunk: c.chunk,
            draw: 0,
            text_sha256: &c.text_sha256,
        })
    );
    assert_eq!(plan.seed_at(1, 0), c.seed);
    assert_ne!(plan.seed_at(1, 1), c.seed);
    assert_ne!(
        plan_chapter(&PROFILE, "id", ch1, "root2").unwrap().chunks[1].seed,
        c.seed
    );
    assert_ne!(
        plan_chapter(&PROFILE, "id2", ch1, "root").unwrap().chunks[1].seed,
        c.seed
    );
    let mut edited = ch1.clone();
    edited.blocks[1] = Block::Paragraph {
        text: "The keeper counted the terns before breakfast.".into(),
    };
    assert_ne!(
        plan_chapter(&PROFILE, "id", &edited, "root").unwrap().chunks[1].seed,
        c.seed
    );
    // One chapter's plan does not depend on another's: a chunk can be planned (and rendered) alone.
    assert_eq!(
        plan_book(&PROFILE, &book, "root").unwrap()[4].chunks,
        plan_chapter(&PROFILE, &book.book_id(), ch1, "root").unwrap().chunks
    );
}

/// AC1: chunks come from ggtts-core's chunker at the profile's ceiling, never a ceiling of the book's own.
#[test]
fn the_plan_is_the_engine_chunker_at_the_profile_ceiling() {
    let book = book();
    for profile in [
        PROFILE,
        Profile {
            story_ceiling: 12,
            ..PROFILE
        },
    ] {
        for ch in &book.chapters {
            let plan = plan_chapter(&profile, "id", ch, "root").unwrap();
            let paragraphs: Vec<String> = ch.blocks.iter().map(|b| b.text().to_string()).collect();
            let core = ggtts_core::text::plan(&profile, &paragraphs, profile.story_ceiling).unwrap();
            let ours: Vec<(usize, &str)> = plan.chunks.iter().map(|c| (c.paragraph, c.text.as_str())).collect();
            let theirs: Vec<(usize, &str)> = core.iter().map(|c| (c.para, c.text.as_str())).collect();
            assert_eq!(ours, theirs, "chapter {}", ch.id);
            assert_eq!(plan.ceiling, profile.story_ceiling);
        }
    }
    // The planted ceiling changes the plan: the profile, not a default, sets it.
    let ch1 = chapter(&book, "ch1");
    let at_45 = plan_chapter(&PROFILE, "id", ch1, "root").unwrap();
    let at_12 = plan_chapter(
        &Profile {
            story_ceiling: 12,
            ..PROFILE
        },
        "id",
        ch1,
        "root",
    )
    .unwrap();
    assert!(at_12.chunks.len() > at_45.chunks.len());
}

/// A block with nothing to speak (a scene break) has no chunks, and the other blocks keep their indices.
#[test]
fn a_scene_break_has_no_chunks_and_moves_no_index() {
    let ch = Chapter {
        id: "x".into(),
        title: String::new(),
        depth: 0,
        skippable: false,
        skip_reason: None,
        files: vec![],
        blocks: vec![
            Block::Paragraph {
                text: "The door opened.".into(),
            },
            Block::Paragraph { text: "* * *".into() },
            Block::Paragraph {
                text: "It closed.".into(),
            },
        ],
    };
    let plan = plan_chapter(&PROFILE, "id", &ch, "root").unwrap();
    let paragraphs: Vec<usize> = plan.chunks.iter().map(|c| c.paragraph).collect();
    assert_eq!(paragraphs, [0, 2]);
}

/// The book id is the package's unique identifier, else a hash of the package document.
#[test]
fn the_book_id() {
    assert_eq!(book().book_id(), "urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d");
    let spine = Book::read(spine_only().bytes()).unwrap();
    assert!(spine.identifier.is_none());
    let id = spine.book_id();
    assert!(id.starts_with("sha256:") && id.len() == 7 + 64, "{id}");
    assert_eq!(Book::read(spine_only().bytes()).unwrap().book_id(), id);
    let epub = spine_only();
    let opf = epub.text("OEBPS/content.opf").replace("Two Rooms", "Three Rooms");
    assert_ne!(
        Book::read(epub.with("OEBPS/content.opf", &opf).bytes())
            .unwrap()
            .book_id(),
        id
    );
}

/// The profile comes from the engine; an engine whose profile this build lacks is refused, not guessed.
#[test]
fn engine_profiles() {
    assert_eq!(engine_profile("qwen3-tts").unwrap(), PROFILE);
    assert_eq!(engine_profile("fake").unwrap(), PROFILE);
    assert!(engine_profile("kitten-tts-2").unwrap_err().contains("ggml"));
    assert!(engine_profile("nope").unwrap_err().contains("unknown engine"));
}
