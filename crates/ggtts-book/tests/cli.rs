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
