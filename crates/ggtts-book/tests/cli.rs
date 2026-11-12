//! Story 1, AC6: `ggtts-book inspect` prints the chapter model, and refuses with exit 2 and the named reason.

mod common;

use std::process::Command;

use common::*;

fn ggtts_book() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ggtts-book"))
}

/// Story 2, AC8: `inspect` lists each chapter's planned chunks and seeds, the same the library plans.
#[test]
fn inspect_prints_the_planned_chunks_and_seeds() {
    use ggtts_book::plan::plan_book;
    let path = nested().write_temp("inspect-plan");
    let out = ggtts_book()
        .arg("inspect")
        .arg(&path)
        .args(["--seed-root", "r1", "--engine", "fake"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["plan"]["seed_root"], "r1");
    assert_eq!(json["plan"]["profile"], "story-v1/qwen3-tts");
    assert_eq!(json["plan"]["ceiling"], ggtts_qwen3_tts::PROFILE.story_ceiling);
    assert_eq!(json["book_id"], "urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d");
    let book = ggtts_book::Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&ggtts_qwen3_tts::PROFILE, &book, "r1").unwrap();
    for (chapter, plan) in json["chapters"].as_array().unwrap().iter().zip(&plans) {
        let seeds: Vec<u64> = chapter["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["seed"].as_u64().unwrap())
            .collect();
        let expected: Vec<u64> = plan.chunks.iter().map(|c| u64::from(c.seed)).collect();
        assert_eq!(seeds, expected, "chapter {}", plan.chapter_id);
    }
    // Another seed root, other seeds.
    let other = ggtts_book()
        .arg("inspect")
        .arg(&path)
        .args(["--seed-root", "r2"])
        .output()
        .unwrap();
    let other: serde_json::Value = serde_json::from_slice(&other.stdout).unwrap();
    assert_ne!(
        other["chapters"][4]["chunks"][0]["seed"],
        json["chapters"][4]["chunks"][0]["seed"]
    );
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

/// Story 2, AC8: an engine whose profile this build lacks is refused before the book is read.
#[test]
fn inspect_refuses_an_engine_it_has_no_profile_for() {
    let path = nested().write_temp("inspect-engine");
    for (engine, reason) in [("kitten-tts-2", "ggml feature"), ("nope", "unknown engine")] {
        let out = ggtts_book()
            .arg("inspect")
            .arg(&path)
            .args(["--engine", engine])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{engine}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(reason), "{engine}");
    }
    let out = ggtts_book()
        .arg("inspect")
        .arg(&path)
        .arg("--seed-root")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn inspect_prints_the_chapter_model() {
    let path = nested().write_temp("inspect-ok");
    let out = ggtts_book().arg("inspect").arg(&path).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["format"], "ggtts-book-chapters/1");
    assert_eq!(json["title"], "The Lighthouse Keeper's Almanac");
    assert_eq!(json["chapters"].as_array().unwrap().len(), 10);
    assert_eq!(json["chapters"][5]["title"], "2. The Second Lamp");
    assert_eq!(json["chapters"][5]["skippable"], false);
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn inspect_refuses_a_drm_protected_book() {
    let epub = nested().with("META-INF/encryption.xml", &encryption_xml("OEBPS/Text/ch1.xhtml", AES));
    let path = epub.write_temp("inspect-drm");
    let out = ggtts_book().arg("inspect").arg(&path).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("drm-protected:"), "{stderr}");
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn inspect_refuses_a_missing_file_and_a_bad_command_line() {
    let out = ggtts_book().args(["inspect", "no-such-book.epub"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("io:"));

    let out = ggtts_book().arg("render").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage:"));
}

#[test]
fn version_names_the_binary() {
    let out = ggtts_book().arg("--version").output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("ggtts-book "));
}

/// Story 3, AC1: `ggtts-book render --engine fake` renders a book end to end and reports each chapter.
#[test]
fn render_through_the_fake_engine() {
    let path = nested().write_temp("render-fake");
    let out = path.parent().unwrap().join("out");
    let run = || {
        ggtts_book()
            .arg("render")
            .arg(&path)
            .arg("--out")
            .arg(&out)
            .args(["--engine", "fake"])
            .output()
            .unwrap()
    };
    let first = run();
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let report: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    let chapters = report["chapters"].as_array().unwrap();
    assert_eq!(chapters.len(), 10);
    assert_eq!(chapters.iter().filter(|c| c["state"] == "written").count(), 6);
    assert_eq!(report["engine"]["family"], "fake");
    assert_eq!(report["engine"]["profile"], "story-v1/qwen3-tts");
    assert_eq!(report["engine"]["ggtts_version"], ggtts_core::VERSION);
    let file = out.join(chapters[4]["file"].as_str().unwrap());
    assert!(file.exists(), "{}", file.display());
    // A second run has nothing left to render.
    let again: serde_json::Value = serde_json::from_slice(&run().stdout).unwrap();
    assert!(again["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["rendered"].as_u64().unwrap_or(0) == 0));
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

/// An engine that cannot load is refused with its reason, before anything is written: in a build without engines,
/// the default `qwen3-tts` names the missing feature; Kitten-TTS 2 has no default files and needs a model; an
/// unknown engine is named.
#[test]
fn render_refuses_an_engine_it_cannot_load() {
    let path = nested().write_temp("render-refuse");
    let out = path.parent().unwrap().join("out");
    // With engines built, an empty cache offline names the missing files; without, the missing build feature.
    let no_engines = if ggtts_load::engines_built() {
        "offline mode does not download"
    } else {
        "--features ggml"
    };
    let cases: [(Vec<&str>, &str); 4] = [
        (vec!["--offline"], no_engines),
        (vec!["--engine", "qwen3-tts", "--offline"], no_engines),
        (vec!["--engine", "kitten-tts-2"], "--model is required"),
        (vec!["--engine", "parrot"], "unknown engine \"parrot\""),
    ];
    for (args, reason) in cases {
        let o = ggtts_book()
            .arg("render")
            .arg(&path)
            .arg("--out")
            .arg(&out)
            .args(&args)
            .env("GGTTS_CACHE_DIR", path.parent().unwrap().join("empty-cache"))
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(stderr.contains(reason), "{args:?}: {stderr}");
        assert!(!out.exists(), "{args:?} wrote output before refusing");
    }
    // A model flag is for render only.
    let o = ggtts_book()
        .arg("inspect")
        .arg(&path)
        .args(["--model", "m.gguf"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let o = ggtts_book()
        .arg("render")
        .arg(&path)
        .args(["--engine", "fake"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2), "no --out");
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}
