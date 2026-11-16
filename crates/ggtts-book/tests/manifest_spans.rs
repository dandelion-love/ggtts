//! Book-manifest tests for final joined sample spans and redraw invalidation.
mod common;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use common::*;
use ggtts_book::manifest::{
    chunk_file, resume, run_chapter, ChunkOutput, Draws, Manifest, ManifestError, SampleSpan, MANIFEST_FILE,
};
use ggtts_book::plan::{plan_chapter, ChapterPlan};
use ggtts_book::Book;
use ggtts_core::profile::Profile;
use ggtts_qwen3_tts::PROFILE;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-spans-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn plan() -> ChapterPlan {
    let profile = Profile {
        story_ceiling: 12,
        ..PROFILE
    };
    let book = Book::read(nested().bytes()).unwrap();
    let chapter = book.chapters.iter().find(|c| c.id == "ch1").unwrap();
    plan_chapter(&profile, &book.book_id(), chapter, "span-test").unwrap()
}

fn chunk_audio(plan: &ChapterPlan, i: usize) -> Vec<u8> {
    format!("chunk {i} at seed {}", plan.chunks[i].seed).into_bytes()
}

fn complete_with_draws(dir: &Path, plan: &ChapterPlan) {
    run_chapter(dir, plan, |p, i| {
        Ok(ChunkOutput {
            bytes: chunk_audio(p, i),
            draws: Some(Draws {
                kept: 0,
                stop: Some("eos".into()),
                count: 1,
                verdict: None,
                reads: vec![],
            }),
        })
    })
    .unwrap();
}

fn spans(plan: &ChapterPlan, first_chunk_samples: u64) -> (u64, Vec<SampleSpan>) {
    let mut out = Vec::new();
    let mut start = 0;
    let mut sample_count = 0;
    for i in 0..plan.chunks.len() {
        let end = start + if i == 0 { first_chunk_samples } else { 100 };
        out.push(SampleSpan { start, end });
        sample_count = end;
        if i + 1 < plan.chunks.len() {
            start = end + 20; // Pause samples are outside chunk spans.
        }
    }
    (sample_count, out)
}

fn resign(outer: &mut serde_json::Value) {
    let body = serde_json::to_string(&outer["manifest"]).unwrap();
    outer["sha256"] = ggtts_core::seed::hex(&ggtts_core::seed::sha256(body.as_bytes())).into();
}

#[test]
fn chapter_spans_and_draw_identity_round_trip_even_without_redraws() {
    let d = dir("roundtrip");
    let mut p = plan();
    p.redraw = 0;
    complete_with_draws(&d, &p);
    let chapter_bytes = b"joined chapter bytes";
    let (sample_count, expected) = spans(&p, 100);
    Manifest::record_chapter_with_spans(
        &d,
        &p,
        "chapter.flac",
        chapter_bytes,
        24000,
        sample_count,
        expected.clone(),
    )
    .unwrap();

    let loaded = Manifest::load(&d, &p).unwrap();
    let actual = loaded.chapter_spans.unwrap();
    assert_eq!(actual.sample_rate, 24000);
    assert_eq!(actual.sample_count, sample_count);
    assert_eq!(actual.chunks, expected);
    assert_eq!(
        actual.chapter_sha256,
        ggtts_core::seed::hex(&ggtts_core::seed::sha256(chapter_bytes))
    );

    let outer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(outer["manifest"]["chunks"][0]["kept_stop"], "eos");
    assert_eq!(outer["manifest"]["chunks"][0]["kept_draw"], 0);
    assert_eq!(outer["manifest"]["chunks"][0]["draws"], 1);
}

#[test]
fn old_chapter_record_loads_without_inventing_spans() {
    let d = dir("legacy");
    let p = plan();
    run_chapter(&d, &p, |plan, i| Ok(chunk_audio(plan, i))).unwrap();
    Manifest::record_chapter(&d, &p, "chapter.flac", b"legacy chapter").unwrap();
    let loaded = Manifest::load(&d, &p).unwrap();
    assert!(loaded.chapter.is_some());
    assert!(loaded.chapter_spans.is_none());
    let outer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert!(outer["manifest"].get("chapter_spans").is_none());
}

#[test]
fn stale_hash_and_overlapping_spans_are_refused() {
    let d = dir("invalid");
    let p = plan();
    complete_with_draws(&d, &p);
    let (sample_count, chunk_spans) = spans(&p, 100);
    Manifest::record_chapter_with_spans(&d, &p, "chapter.flac", b"chapter", 24000, sample_count, chunk_spans).unwrap();
    let original = std::fs::read_to_string(d.join(MANIFEST_FILE)).unwrap();

    let mut stale: serde_json::Value = serde_json::from_str(&original).unwrap();
    stale["manifest"]["chapter_spans"]["chapter_sha256"] = "0".repeat(64).into();
    resign(&mut stale);
    assert!(matches!(
        Manifest::parse(&stale.to_string(), &p),
        Err(ManifestError::Format(message)) if message.contains("stale")
    ));

    let mut overlap: serde_json::Value = serde_json::from_str(&original).unwrap();
    overlap["manifest"]["chapter_spans"]["chunks"][1][0] = overlap["manifest"]["chapter_spans"]["chunks"][0][1].clone();
    overlap["manifest"]["chapter_spans"]["chunks"][1][1] = overlap["manifest"]["chapter_spans"]["chunks"][0][1].clone();
    resign(&mut overlap);
    assert!(matches!(
        Manifest::parse(&overlap.to_string(), &p),
        Err(ManifestError::Format(message)) if message.contains("span")
    ));
}

#[test]
fn rerender_clears_old_spans_and_rebuilds_downstream_offsets() {
    let d = dir("redraw");
    let mut p = plan();
    p.redraw = 1;
    complete_with_draws(&d, &p);
    let (old_count, old_spans) = spans(&p, 100);
    Manifest::record_chapter_with_spans(
        &d,
        &p,
        "chapter.flac",
        b"before redraw",
        24000,
        old_count,
        old_spans.clone(),
    )
    .unwrap();

    std::fs::write(d.join(chunk_file(0)), b"changed duration").unwrap();
    let resumed = resume(&d, &p);
    assert_eq!(resumed.todo, [0]);
    assert!(resumed.manifest.chapter.is_none());
    assert!(resumed.manifest.chapter_spans.is_none());
    assert!(resumed.manifest.kept_stop[0].is_none());
    assert!(resumed.manifest.kept_stop[1..].iter().all(Option::is_some));

    let calls = RefCell::new(Vec::new());
    run_chapter(&d, &p, |plan, i| {
        calls.borrow_mut().push(i);
        let bytes = if i == 0 {
            b"redrawn longer chunk".to_vec()
        } else {
            chunk_audio(plan, i)
        };
        Ok(ChunkOutput {
            bytes,
            draws: Some(Draws {
                kept: 0,
                stop: Some("eos".into()),
                count: 1,
                verdict: None,
                reads: vec![],
            }),
        })
    })
    .unwrap();
    assert_eq!(*calls.borrow(), [0]);
    assert!(Manifest::load(&d, &p).unwrap().chapter_spans.is_none());

    let (new_count, new_spans) = spans(&p, 140);
    Manifest::record_chapter_with_spans(
        &d,
        &p,
        "chapter.flac",
        b"after redraw",
        24000,
        new_count,
        new_spans.clone(),
    )
    .unwrap();
    let rebuilt = Manifest::load(&d, &p).unwrap().chapter_spans.unwrap();
    assert_eq!(rebuilt.chunks, new_spans);
    assert!(rebuilt.chunks[1].start > old_spans[1].start);
}
