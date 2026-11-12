//! Story 2: the chapter manifest, its refusals, and resume.

mod common;

use std::cell::RefCell;
use std::path::PathBuf;

use common::*;
use ggtts_book::manifest::{chunk_file, resume, run_chapter, Manifest, ManifestError, Output, RunError, MANIFEST_FILE};
use ggtts_book::plan::{plan_chapter, ChapterPlan};
use ggtts_book::{Block, Book};
use ggtts_core::profile::Profile;
use ggtts_qwen3_tts::PROFILE;

/// A fresh, empty directory for one test.
fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-manifest-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Chapter 1 of the nested fixture at a small ceiling, so it has many chunks.
fn plan_with(profile: &Profile, seed_root: &str) -> ChapterPlan {
    let book = Book::read(nested().bytes()).unwrap();
    let ch1 = book.chapters.iter().find(|c| c.id == "ch1").unwrap();
    plan_chapter(profile, &book.book_id(), ch1, seed_root).unwrap()
}

fn small() -> Profile {
    Profile {
        story_ceiling: 12,
        ..PROFILE
    }
}

fn plan() -> ChapterPlan {
    plan_with(&small(), "root")
}

fn audio(plan: &ChapterPlan, index: usize) -> Vec<u8> {
    format!("audio of chunk {index} at seed {}", plan.chunks[index].seed).into_bytes()
}

/// AC5: a manifest round-trips, records what the plan says, and leaves no temporary file behind.
#[test]
fn a_manifest_round_trips() {
    let d = dir("roundtrip");
    let plan = plan();
    assert!(
        plan.chunks.len() >= 5,
        "the fixture plan should have several chunks, has {}",
        plan.chunks.len()
    );
    let mut m = Manifest::new(plan.clone());
    m.done[1] = Some(Output {
        file: chunk_file(1),
        sha256: "ab".repeat(32),
    });
    m.save(&d).unwrap();
    assert_eq!(Manifest::load(&d, &plan).unwrap(), m);
    let names: Vec<String> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, [MANIFEST_FILE]);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join(MANIFEST_FILE)).unwrap()).unwrap();
    let body = &json["manifest"];
    assert_eq!(body["format"], "ggtts-book-manifest/1");
    assert_eq!(body["profile"], "story-v1/qwen3-tts");
    assert_eq!(body["ceiling"], 12);
    assert_eq!(body["chunks"][1]["state"], "done");
    assert_eq!(body["chunks"][0]["state"], "planned");
    assert_eq!(body["chunks"][0]["seed"], plan.chunks[0].seed);
}

/// AC6: a manifest cut short mid-write is not trusted, and the chapter starts again from its plan.
#[test]
fn a_truncated_manifest_is_refused() {
    let d = dir("truncated");
    let plan = plan();
    let mut m = Manifest::new(plan.clone());
    m.done[0] = Some(Output {
        file: chunk_file(0),
        sha256: "00".repeat(32),
    });
    m.save(&d).unwrap();
    let full = std::fs::read(d.join(MANIFEST_FILE)).unwrap();
    std::fs::write(d.join(MANIFEST_FILE), &full[..full.len() / 2]).unwrap();
    assert!(matches!(Manifest::load(&d, &plan), Err(ManifestError::NotJson(_))));
    let r = resume(&d, &plan);
    assert!(matches!(r.refused, Some(ManifestError::NotJson(_))));
    assert_eq!(r.todo, (0..plan.chunks.len()).collect::<Vec<_>>());
}

/// AC6: an edited manifest fails its checksum.
#[test]
fn an_edited_manifest_fails_its_checksum() {
    let d = dir("checksum");
    let plan = plan();
    Manifest::new(plan.clone()).save(&d).unwrap();
    let text = std::fs::read_to_string(d.join(MANIFEST_FILE)).unwrap();
    let forged = text.replacen("\"planned\"", "\"done\"", 1);
    assert_ne!(forged, text);
    assert_eq!(Manifest::parse(&forged, &plan), Err(ManifestError::Checksum));
    // Unedited, it is trusted.
    assert!(Manifest::parse(&text, &plan).is_ok());
}

/// AC6: a manifest for another book, chapter, seed root, profile or ceiling is refused, naming the field.
#[test]
fn a_manifest_for_another_plan_is_refused() {
    let plan = plan();
    let text = Manifest::new(plan.clone()).to_json_string();
    let field = |other: &ChapterPlan| match Manifest::parse(&text, other) {
        Err(ManifestError::Mismatch { field, .. }) => field,
        other => panic!("expected a mismatch, got {other:?}"),
    };
    assert_eq!(
        field(&ChapterPlan {
            book_id: "another".into(),
            ..plan.clone()
        }),
        "book"
    );
    assert_eq!(
        field(&ChapterPlan {
            chapter_id: "ch2a".into(),
            ..plan.clone()
        }),
        "chapter"
    );
    assert_eq!(field(&plan_with(&small(), "another root")), "seed root");
    assert_eq!(
        field(&plan_with(
            &Profile {
                name: "story-v2/qwen3-tts",
                ..small()
            },
            "root"
        )),
        "profile"
    );
    assert_eq!(
        field(&plan_with(
            &Profile {
                story_ceiling: 13,
                ..PROFILE
            },
            "root"
        )),
        "ceiling"
    );
}

/// AC6: a chapter whose text changed since the manifest was written is re-planned, not resumed.
#[test]
fn a_changed_chapter_is_not_resumed() {
    let plan = plan();
    let text = Manifest::new(plan.clone()).to_json_string();
    let book = Book::read(nested().bytes()).unwrap();
    let mut ch1 = book.chapters.iter().find(|c| c.id == "ch1").unwrap().clone();
    ch1.blocks[2] = Block::Paragraph {
        text: "There were eight of them.".into(),
    };
    let changed = plan_chapter(&small(), &book.book_id(), &ch1, "root").unwrap();
    assert!(matches!(
        Manifest::parse(&text, &changed),
        Err(ManifestError::PlanChanged { .. })
    ));
}

/// AC6: an unknown format is refused.
#[test]
fn an_unknown_format_is_refused() {
    let plan = plan();
    let body = serde_json::json!({"format": "ggtts-book-manifest/9"});
    let sha = ggtts_core::seed::hex(&ggtts_core::seed::sha256(
        serde_json::to_string(&body).unwrap().as_bytes(),
    ));
    let text = serde_json::json!({"manifest": body, "sha256": sha}).to_string();
    assert!(matches!(Manifest::parse(&text, &plan), Err(ManifestError::Format(_))));
}

/// AC7: a run killed partway resumes at the first chunk not done and never renders a finished chunk again.
#[test]
fn a_killed_run_resumes_and_skips_finished_chunks() {
    let d = dir("resume");
    let plan = plan();
    let n = plan.chunks.len();
    let calls = RefCell::new(Vec::new());

    // The first run dies at chunk 3, the way a killed process would: chunks 0 to 2 are saved, 3 is not.
    let first = run_chapter(&d, &plan, |p, i| {
        calls.borrow_mut().push(i);
        if i == 3 {
            Err("killed".into())
        } else {
            Ok(audio(p, i))
        }
    });
    assert!(matches!(first, Err(RunError::Render { index: 3, .. })));
    assert_eq!(*calls.borrow(), [0, 1, 2, 3]);

    // The second run renders 3 onwards, and nothing before it.
    calls.borrow_mut().clear();
    let second = run_chapter(&d, &plan, |p, i| {
        calls.borrow_mut().push(i);
        Ok(audio(p, i))
    })
    .unwrap();
    assert_eq!(second.refused, None);
    assert_eq!(
        *calls.borrow(),
        (3..n).collect::<Vec<_>>(),
        "a resume rendered a finished chunk again"
    );

    // A third run has nothing to do.
    calls.borrow_mut().clear();
    run_chapter(&d, &plan, |p, i| {
        calls.borrow_mut().push(i);
        Ok(audio(p, i))
    })
    .unwrap();
    assert!(
        calls.borrow().is_empty(),
        "a finished chapter rendered {:?} again",
        calls.borrow()
    );

    // Every output is the renderer's, at its own seed, and the manifest says so.
    let m = Manifest::load(&d, &plan).unwrap();
    for i in 0..n {
        assert_eq!(std::fs::read(d.join(chunk_file(i))).unwrap(), audio(&plan, i));
        assert!(m.done[i].is_some());
    }
}

/// AC7: a chunk marked done whose output is gone or changed is rendered again, and only that chunk.
#[test]
fn a_missing_or_changed_output_is_rendered_again() {
    let d = dir("damaged");
    let plan = plan();
    run_chapter(&d, &plan, |p, i| Ok(audio(p, i))).unwrap();
    std::fs::remove_file(d.join(chunk_file(1))).unwrap();
    std::fs::write(d.join(chunk_file(4)), b"not what was rendered").unwrap();
    let calls = RefCell::new(Vec::new());
    run_chapter(&d, &plan, |p, i| {
        calls.borrow_mut().push(i);
        Ok(audio(p, i))
    })
    .unwrap();
    assert_eq!(*calls.borrow(), [1, 4]);
}

/// AC6 and AC7 together: a manifest for another seed root in the directory is refused, and every chunk is rendered.
#[test]
fn a_refused_manifest_renders_everything() {
    let d = dir("other-root");
    let other = plan_with(&small(), "another root");
    run_chapter(&d, &other, |p, i| Ok(audio(p, i))).unwrap();
    let plan = plan();
    let calls = RefCell::new(Vec::new());
    let start = run_chapter(&d, &plan, |p, i| {
        calls.borrow_mut().push(i);
        Ok(audio(p, i))
    })
    .unwrap();
    assert!(matches!(
        start.refused,
        Some(ManifestError::Mismatch { field: "seed root", .. })
    ));
    assert_eq!(*calls.borrow(), (0..plan.chunks.len()).collect::<Vec<_>>());
}
