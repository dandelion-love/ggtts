//! AC2 on a real engine: a book chapter rendered with `--steer` on the tiny real model differs from the same chapter
//! unsteered, and the steered render is refused as a resume of the unsteered one. The steering reaches the engine
//! through the book, not only the fake engine's request record.
//!
//! Needs the tiny fixture (`scripts/tiny_fixture.sh`): `GGTTS_TINY_TALKER` and `GGTTS_CODEC`.
//!
//! ```text
//! eval "$(scripts/tiny_fixture.sh)"
//! cargo test --release -p ggtts-book --features ggml --test steer_tiny -- --ignored --nocapture
//! ```
#![cfg(feature = "ggml")]

mod common;

use std::path::{Path, PathBuf};

use common::Epub;
use ggtts_book::render::{render_command_options, LoadedEngine};

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} (see scripts/tiny_fixture.sh)")))
}

/// A one-chapter book with one short paragraph: one chunk, so the tiny model (which never stops and runs every
/// request to its cap) renders it in seconds.
fn one_chunk_book() -> Epub {
    let opf = r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:7c0a51f2-3d2b-4f7e-8e1a-2b9d4c6e8f10</dc:identifier>
    <dc:title>Steer Test</dc:title>
    <dc:language>yue</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>"#;
    let nav = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>n</title></head>
<body><nav epub:type="toc"><ol><li><a href="ch1.xhtml">One</a></li></ol></nav></body></html>"#;
    let ch1 = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c</title></head><body><p>你好。</p></body></html>"#;
    Epub::new(&[
        ("META-INF/container.xml", common::CONTAINER),
        ("OEBPS/content.opf", opf),
        ("OEBPS/nav.xhtml", nav),
        ("OEBPS/ch1.xhtml", ch1),
    ])
}

fn render(book: &Path, out: &Path, extra: &[String]) -> Result<serde_json::Value, String> {
    let (talker, codec) = (env_path("GGTTS_TINY_TALKER"), env_path("GGTTS_CODEC"));
    let mut args = vec![
        book.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--engine".into(),
        "qwen3-tts".into(),
        "--model".into(),
        talker.to_string_lossy().into_owned(),
        "--codec".into(),
        codec.to_string_lossy().into_owned(),
        "--offline".into(),
    ];
    args.extend(extra.iter().cloned());
    let (path, options) = ggtts_book::options::parse("render", &args, &|_| None)?;
    let req = options.load.clone();
    render_command_options(&path, &options, || {
        let engine = ggtts_load::load(&req)?;
        Ok(LoadedEngine {
            engine,
            files: Vec::new(),
            speaker: req.speaker.clone(),
        })
    })
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn a_steered_chapter_differs_from_an_unsteered_one_on_the_tiny_model() {
    let book = one_chunk_book().write_temp("steer-tiny");
    let d = book.parent().unwrap().to_path_buf();
    let steer = d.join("steer.safetensors");
    let v = vec![4.0f32; 2048];
    std::fs::write(&steer, ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v)])).unwrap();
    let chunk = |out: &Path| {
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("book.json")).unwrap()).unwrap();
        let stem = report["chapters"][0].as_str().unwrap().to_string();
        std::fs::read(out.join(stem).join("chunk-00000.flac")).unwrap()
    };
    let plain = d.join("plain");
    render(&book, &plain, &[]).unwrap();
    let again = d.join("again");
    render(&book, &again, &[]).unwrap();
    assert_eq!(
        chunk(&plain),
        chunk(&again),
        "the unsteered chunk repeats at its seed: a difference is the steering"
    );
    let steered = d.join("steered");
    let report = render(
        &book,
        &steered,
        &["--steer".into(), steer.to_string_lossy().into_owned()],
    )
    .unwrap();
    assert!(report["render"]["steer"]["sha256"].is_string(), "{report}");
    assert_ne!(chunk(&plain), chunk(&steered), "the steered chunk is the unsteered one");
    // The steered render over the unsteered one is a different render: its manifest is refused, naming the steering.
    let resumed = render(&book, &plain, &["--steer".into(), steer.to_string_lossy().into_owned()]).unwrap();
    let refused = resumed["chapters"][0]["refused_manifest"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(refused.contains("steering"), "{resumed}");
    std::fs::remove_dir_all(&d).ok();
}
