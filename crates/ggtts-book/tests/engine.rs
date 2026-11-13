//! Story 3, AC2, AC5 and AC6: one engine load per book, the engine's provenance in every manifest and checked on
//! resume, render times recorded, and an output directory that holds another book refused. Driven with the fake
//! engine, through the same `render_command` the CLI runs.

mod common;

use std::cell::Cell;
use std::path::PathBuf;

use common::*;
use ggtts_book::render::{render_command, LoadedEngine, BOOK_MARKER};
use ggtts_fake::FakeEngine;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-engine-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fake(speaker: &str) -> LoadedEngine {
    LoadedEngine {
        engine: Box::new(FakeEngine::default()),
        files: vec![("talker".into(), "ab".repeat(32)), ("codec".into(), "cd".repeat(32))],
        speaker: Some(speaker.into()),
    }
}

/// AC2: rendering a book of six spoken chapters loads the engine once. (The loader is `FnOnce`, so a second call
/// would not compile; the count says so at run time too.)
#[test]
fn a_book_loads_its_engine_once() {
    let d = dir("once");
    let book = nested().write_temp("engine-once");
    let loads = Cell::new(0);
    let report = render_command(&book, &d.join("out"), "root", || {
        loads.set(loads.get() + 1);
        Ok(fake("voice_a"))
    })
    .unwrap();
    assert_eq!(loads.get(), 1);
    let written = report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .count();
    assert_eq!(written, 6);
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC6: a book refused is refused before the engine is loaded, so a bad book costs no model load.
#[test]
fn a_refused_book_loads_no_engine() {
    let d = dir("refused");
    let book = nested()
        .with("META-INF/encryption.xml", &encryption_xml("OEBPS/Text/ch1.xhtml", AES))
        .write_temp("engine-drm");
    let err = render_command(&book, &d.join("out"), "root", || -> Result<LoadedEngine, String> {
        panic!("the engine was loaded for a refused book")
    })
    .unwrap_err();
    assert!(err.starts_with("drm-protected:"), "{err}");
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC5: every chapter manifest records the engine's provenance and each done chunk's render time.
#[test]
fn manifests_record_the_engine_and_render_times() {
    let d = dir("provenance");
    let book = nested().write_temp("engine-prov");
    let out = d.join("out");
    let report = render_command(&book, &out, "root", || Ok(fake("voice_a"))).unwrap();
    let engine = &report["engine"];
    assert_eq!(engine["family"], "fake");
    assert_eq!(engine["speaker"], "voice_a");
    assert_eq!(engine["files"][0]["sha256"], "ab".repeat(32));
    assert_eq!(engine["ggtts_commit"], ggtts_core::SOURCE_COMMIT);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("004-ch1").join("manifest.json")).unwrap()).unwrap();
    assert_eq!(&manifest["manifest"]["engine"], engine);
    for chunk in manifest["manifest"]["chunks"].as_array().unwrap() {
        assert_eq!(chunk["state"], "done");
        assert!(chunk["render_ms"].is_u64(), "{chunk}");
    }
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC5 and AC6: a resume with another engine (here another speaker; equally another model file, backend or ggtts
/// build) does not splice two renders together: the earlier manifests are refused, naming the engine, and every
/// chunk is rendered again.
#[test]
fn a_resume_with_another_engine_renders_everything_again() {
    let d = dir("other-engine");
    let book = nested().write_temp("engine-other");
    let out = d.join("out");
    render_command(&book, &out, "root", || Ok(fake("voice_a"))).unwrap();
    let same = render_command(&book, &out, "root", || Ok(fake("voice_a"))).unwrap();
    assert!(same["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["rendered"].as_u64().unwrap_or(0) == 0));
    let other = render_command(&book, &out, "root", || Ok(fake("voice_b"))).unwrap();
    for c in other["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
    {
        assert_eq!(c["resumed"], 0, "{c}");
        let refused = c["refused_manifest"].as_str().unwrap_or("");
        assert!(refused.contains("for engine"), "{c}");
    }
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// Review finding 211-A: a resume with another model file (same engine, same speaker, a different SHA-256) is not
/// resumed either; every chunk is rendered again.
#[test]
fn a_resume_with_another_model_file_renders_everything_again() {
    let d = dir("other-model");
    let book = nested().write_temp("engine-other-model");
    let out = d.join("out");
    render_command(&book, &out, "root", || Ok(fake("voice_a"))).unwrap();
    let mut other = fake("voice_a");
    other.files[0].1 = "ef".repeat(32);
    let report = render_command(&book, &out, "root", || Ok(other)).unwrap();
    for c in report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
    {
        assert_eq!(c["resumed"], 0, "{c}");
        let refused = c["refused_manifest"].as_str().unwrap_or("");
        assert!(
            refused.contains("for engine") && refused.contains(&"ab".repeat(32)),
            "{c}"
        );
    }
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC6: an output directory holding another book is refused before anything is rendered or overwritten, and so is a
/// non-empty directory this tool did not write.
#[test]
fn an_output_directory_holding_another_book_is_refused() {
    let d = dir("other-book");
    let first = nested().write_temp("engine-first");
    let second = ncx_only().write_temp("engine-second");
    let out = d.join("out");
    render_command(&first, &out, "root", || Ok(fake("voice_a"))).unwrap();
    let marker = std::fs::read_to_string(out.join(BOOK_MARKER)).unwrap();
    let err = render_command(&second, &out, "root", || Ok(fake("voice_a"))).unwrap_err();
    assert!(err.contains("holds another book"), "{err}");
    assert_eq!(
        std::fs::read_to_string(out.join(BOOK_MARKER)).unwrap(),
        marker,
        "the marker was changed"
    );

    let stranger = d.join("stranger");
    std::fs::create_dir_all(&stranger).unwrap();
    std::fs::write(stranger.join("notes.txt"), "not a render").unwrap();
    let err = render_command(&first, &stranger, "root", || Ok(fake("voice_a"))).unwrap_err();
    assert!(err.contains("is not empty"), "{err}");
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(first.parent().unwrap()).ok();
    std::fs::remove_dir_all(second.parent().unwrap()).ok();
}
