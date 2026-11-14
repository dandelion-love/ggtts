//! `ggtts-book render` on the shared options table: the options a book does not take are refused by name, before
//! anything is written, and steering on an engine that cannot steer is refused with no chunk written.

mod common;

use std::path::Path;
use std::process::{Command, Output};

use common::nested;

fn ggtts_book() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ggtts-book"))
}

fn render(book: &Path, out: &Path, args: &[&str]) -> Output {
    ggtts_book()
        .arg("render")
        .arg(book)
        .arg("--out")
        .arg(out)
        .args(["--engine", "fake"])
        .args(args)
        .output()
        .unwrap()
}

/// Every chunk file under `dir`, at any depth.
fn chunk_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("chunk-")) {
                found.push(p);
            }
        }
    }
    found
}

/// AC3: an option of the shared table a book does not take is refused by its flag and schema name, with the reason,
/// and nothing is written.
fn refused_by_name(tag: &str, args: &[&str], key: &str, reason: &str) {
    let book = nested().write_temp(tag);
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, args);
    assert_eq!(o.status.code(), Some(2), "{args:?}");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains(&format!("{} ({key})", args[0])), "{args:?}: {stderr}");
    assert!(stderr.contains(reason), "{args:?}: {stderr}");
    assert!(!out.exists(), "{args:?} wrote output before refusing");
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

#[test]
fn text_is_refused_by_name() {
    refused_by_name(
        "refuse-text",
        &["--text", "hello"],
        "render.text",
        "not a text or a story file",
    );
}

#[test]
fn stream_is_refused_by_name() {
    refused_by_name("refuse-stream", &["--stream", "-"], "output.stream", "not streamed");
}

#[test]
fn seed_is_refused_by_name() {
    refused_by_name("refuse-seed", &["--seed", "7"], "render.seed", "--seed-root");
}

/// The other options of the table a book does not take are refused the same way: a report, an instrument, the log.
#[test]
fn reports_instruments_and_log_flags_are_refused_by_name() {
    refused_by_name(
        "refuse-timing",
        &["--timing-report", "t.json"],
        "observe.timing_report",
        "no timing or duration report",
    );
    refused_by_name(
        "refuse-instrument",
        &["--instrument", "loudness"],
        "observe.instrument",
        "no instruments",
    );
    refused_by_name("refuse-log", &["--log-level", "debug"], "observe.log_level", "no log");
}

/// A render option is refused by `inspect`, which plans and renders nothing.
#[test]
fn inspect_refuses_a_render_option_by_name() {
    let book = nested().write_temp("inspect-instruction");
    let o = ggtts_book()
        .arg("inspect")
        .arg(&book)
        .args(["--instruction", "calm"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("--instruction (render.instruction)") && stderr.contains("for render"),
        "{stderr}"
    );
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC5: `--steer` on the fake engine, which cannot steer, is refused by the engine's own per-request refusal, and no
/// chunk file is written.
#[test]
fn steering_an_engine_that_cannot_steer_writes_no_chunk() {
    let book = nested().write_temp("steer-fake");
    let d = book.parent().unwrap();
    let steer = d.join("s.safetensors");
    std::fs::write(
        &steer,
        ggtts_core::steer::write_safetensors(&[("talker.layer.0", &[0.5f32; 8])]),
    )
    .unwrap();
    let out = d.join("out");
    let o = render(&book, &out, &["--steer", steer.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("the fake engine cannot apply steering"), "{stderr}");
    assert_eq!(
        chunk_files(&out),
        Vec::<std::path::PathBuf>::new(),
        "a refused steer wrote a chunk"
    );
    // The same render unsteered writes chunks: the check above can see them.
    let o = render(&book, &d.join("plain"), &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!chunk_files(&d.join("plain")).is_empty());
    std::fs::remove_dir_all(d).ok();
}

/// A bad steering file is refused before the engine is loaded or anything is written.
#[test]
fn a_steering_file_that_does_not_read_is_refused_before_anything_is_written() {
    let book = nested().write_temp("steer-missing");
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, &["--steer", "no-such-file.safetensors"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("render.steer"));
    assert!(!out.exists());
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}
