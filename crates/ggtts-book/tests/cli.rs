//! Story 1, AC6: `ggtts-book inspect` prints the chapter model, and refuses with exit 2 and the named reason.

mod common;

use std::process::Command;

use common::*;

fn ggtts_book() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ggtts-book"))
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
