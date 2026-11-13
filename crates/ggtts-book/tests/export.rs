//! Story 4: chapter FLACs carry the book's tags, and `export` joins a finished render into one FLAC with a chapter
//! mark per chapter. Rendered with the fake engine; the chapter FLACs and the export are read back with `claxon`.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use ggtts_book::export::{export, timestamp, CHAPTER_GAP_S};
use ggtts_book::render::{render_command, render_command_with, LoadedEngine};
use ggtts_core::seed::{hex, sha256};
use ggtts_fake::FakeEngine;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-export-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fake() -> Result<LoadedEngine, String> {
    Ok(LoadedEngine {
        engine: Box::new(FakeEngine::default()),
        files: vec![],
        speaker: None,
    })
}

/// Render the nested fixture into `d/out`; returns (out, the report).
fn rendered(d: &Path, include_skippable: bool) -> (PathBuf, serde_json::Value) {
    let book = nested().write_temp(&format!("export-{}", d.file_name().unwrap().to_string_lossy()));
    let out = d.join("out");
    let report = render_command_with(&book, &out, "root", include_skippable, fake).unwrap();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
    (out, report)
}

fn tags(path: &Path, name: &str) -> Vec<String> {
    let r = claxon::FlacReader::open(path).unwrap();
    r.get_tag(name).map(String::from).collect()
}

fn samples(path: &Path) -> Vec<i32> {
    let mut r = claxon::FlacReader::open(path).unwrap();
    r.samples().map(Result::unwrap).collect()
}

/// AC1: each chapter FLAC carries TITLE, ALBUM, one ARTIST per author, TRACKNUMBER, and the render's ids.
#[test]
fn chapter_files_carry_the_books_tags() {
    let d = dir("tags");
    let (out, report) = rendered(&d, false);
    let ch1 = out.join(report["chapters"][4]["file"].as_str().unwrap());
    assert_eq!(tags(&ch1, "TITLE"), ["1. Low Tide"]);
    assert_eq!(tags(&ch1, "ALBUM"), ["The Lighthouse Keeper's Almanac"]);
    assert_eq!(tags(&ch1, "ARTIST"), ["Ada Test-Writer", "Bo Fixture"]);
    assert_eq!(
        tags(&ch1, "TRACKNUMBER"),
        ["2"],
        "part1 is track 1 of the written chapters"
    );
    assert_eq!(tags(&ch1, "GGTTS_CHAPTER_ID"), ["ch1"]);
    assert_eq!(tags(&ch1, "GGTTS_SEED_ROOT"), ["root"]);
    std::fs::remove_dir_all(&d).ok();
}

/// AC2: one FLAC, the chapters' own samples in order with the gap between them, and a chapter mark at each start.
#[test]
fn export_joins_the_chapters_with_a_mark_each() {
    let d = dir("join");
    let (out, report) = rendered(&d, false);
    let book_flac = d.join("book.flac");
    let exported = export(&out, &book_flac).unwrap();

    let files: Vec<PathBuf> = report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .map(|c| out.join(c["file"].as_str().unwrap()))
        .collect();
    assert_eq!(files.len(), 6);
    let rate = claxon::FlacReader::open(&files[0]).unwrap().streaminfo().sample_rate;
    // The docs and the changelog promise 2 seconds between chapters.
    assert_eq!(CHAPTER_GAP_S, 2.0);
    let gap = (CHAPTER_GAP_S * f64::from(rate)) as usize;
    assert_eq!(gap, 2 * rate as usize);

    let mut expected = Vec::new();
    let mut starts = Vec::new();
    for (i, f) in files.iter().enumerate() {
        if i > 0 {
            expected.extend(std::iter::repeat_n(0, gap));
        }
        starts.push(expected.len() as u64);
        expected.extend(samples(f));
    }
    assert_eq!(
        samples(&book_flac),
        expected,
        "the export is not the chapters' samples with the gaps"
    );
    assert_eq!(
        claxon::FlacReader::open(&book_flac).unwrap().streaminfo().samples,
        Some(expected.len() as u64)
    );

    for (i, (f, start)) in files.iter().zip(&starts).enumerate() {
        let n = i + 1;
        assert_eq!(tags(&book_flac, &format!("CHAPTER{n:03}")), [timestamp(*start, rate)]);
        assert_eq!(tags(&book_flac, &format!("CHAPTER{n:03}NAME")), tags(f, "TITLE"));
    }
    assert!(tags(&book_flac, "CHAPTER007").is_empty());
    assert_eq!(tags(&book_flac, "ALBUM"), ["The Lighthouse Keeper's Almanac"]);
    assert_eq!(exported["chapters"].as_array().unwrap().len(), 6);
    std::fs::remove_dir_all(&d).ok();
}

/// The same render exports to the same bytes.
#[test]
fn export_is_byte_identical() {
    let d = dir("same");
    let (out, _) = rendered(&d, false);
    export(&out, &d.join("a.flac")).unwrap();
    export(&out, &d.join("b.flac")).unwrap();
    let h = |p: PathBuf| hex(&sha256(&std::fs::read(p).unwrap()));
    assert_eq!(h(d.join("a.flac")), h(d.join("b.flac")));
    std::fs::remove_dir_all(&d).ok();
}

/// AC4: front and back matter is left out unless the render includes it; then the export has it too.
#[test]
fn skippable_chapters_are_rendered_and_exported_only_when_asked() {
    let d = dir("skippable");
    let (out, report) = rendered(&d, true);
    let written = report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .count();
    assert_eq!(written, 10);
    let exported = export(&out, &d.join("book.flac")).unwrap();
    assert_eq!(exported["chapters"].as_array().unwrap().len(), 10);
    assert_eq!(exported["chapters"][0]["title"], "The Lighthouse Keeper's Almanac");
    std::fs::remove_dir_all(&d).ok();
}

/// AC3: a chapter not fully rendered is refused, naming it.
#[test]
fn an_unfinished_render_is_refused() {
    let d = dir("unfinished");
    let book = nested().write_temp("export-unfinished");
    let out = d.join("out");
    struct Dies(FakeEngine, usize);
    impl ggtts_core::engine::Engine for Dies {
        fn info(&self) -> &ggtts_core::engine::EngineInfo {
            self.0.info()
        }
        fn profile(&self) -> &ggtts_core::profile::Profile {
            self.0.profile()
        }
        fn generate(
            &mut self,
            req: &ggtts_core::engine::ChunkRequest<'_>,
        ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
            self.1 += 1;
            if self.1 == 4 {
                return Err(ggtts_core::engine::EngineError("killed".into()));
            }
            self.0.generate(req)
        }
        fn decode(
            &mut self,
            codes: &ggtts_core::engine::Codes,
        ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
            self.0.decode(codes)
        }
    }
    let err = render_command(&book, &out, "root", || {
        Ok(LoadedEngine {
            engine: Box::new(Dies(FakeEngine::default(), 0)),
            files: vec![],
            speaker: None,
        })
    })
    .unwrap_err();
    assert!(err.contains("killed"), "{err}");
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(
        err.contains("not fully rendered") || err.contains("has no chapter file"),
        "{err}"
    );
    assert!(!d.join("book.flac").exists());
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC3: a chapter whose file exists but whose manifest has a chunk not done (a later render of it was cut short) is
/// refused, naming the chapter.
#[test]
fn a_chapter_with_a_chunk_not_done_is_refused() {
    let d = dir("not-done");
    let (out, report) = rendered(&d, false);
    let stem = Path::new(report["chapters"][4]["file"].as_str().unwrap())
        .file_stem()
        .unwrap()
        .to_owned();
    let manifest = out.join(&stem).join("manifest.json");
    let mut outer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    outer["manifest"]["chunks"][1]["state"] = "planned".into();
    outer["manifest"]["chunks"][1]["output"] = serde_json::Value::Null;
    outer["sha256"] = hex(&sha256(serde_json::to_string(&outer["manifest"]).unwrap().as_bytes())).into();
    std::fs::write(&manifest, outer.to_string()).unwrap();
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(
        err.contains("not fully rendered") && err.contains(&*stem.to_string_lossy()),
        "{err}"
    );
    std::fs::remove_dir_all(&d).ok();
}

/// AC3: chapters rendered with two seed roots are refused, naming the chapter.
#[test]
fn chapters_from_two_seed_roots_are_refused() {
    let d = dir("roots");
    let (out, report) = rendered(&d, false);
    let stem = Path::new(report["chapters"][5]["file"].as_str().unwrap())
        .file_stem()
        .unwrap()
        .to_owned();
    let manifest = out.join(&stem).join("manifest.json");
    let mut outer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    outer["manifest"]["seed_root"] = "another root".into();
    outer["sha256"] = hex(&sha256(serde_json::to_string(&outer["manifest"]).unwrap().as_bytes())).into();
    std::fs::write(&manifest, outer.to_string()).unwrap();
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(
        err.contains("seed root") && err.contains(&*stem.to_string_lossy()),
        "{err}"
    );
    std::fs::remove_dir_all(&d).ok();
}

/// The directory name of the written chapter at `i` in a render report.
fn stem(report: &serde_json::Value, i: usize) -> String {
    Path::new(report["chapters"][i]["file"].as_str().unwrap())
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// Rewrite a chapter's manifest body with `f`, keeping its checksum valid.
fn edit_manifest(out: &Path, stem: &str, f: impl FnOnce(&mut serde_json::Value)) {
    let manifest = out.join(stem).join("manifest.json");
    let mut outer: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    f(&mut outer["manifest"]);
    outer["sha256"] = hex(&sha256(serde_json::to_string(&outer["manifest"]).unwrap().as_bytes())).into();
    std::fs::write(&manifest, outer.to_string()).unwrap();
}

/// A render with `--include-skippable` and then one without it leave the front matter's directories behind; the
/// export takes only what the latest render planned.
#[test]
fn a_stale_skippable_chapter_is_not_exported() {
    let d = dir("stale");
    let book = nested().write_temp("export-stale");
    let out = d.join("out");
    let first = render_command_with(&book, &out, "root", true, fake).unwrap();
    let second = render_command_with(&book, &out, "root", false, fake).unwrap();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
    assert_eq!(second["chapters"][0]["state"], "skipped");
    let skipped = stem(&first, 0);
    assert!(
        out.join(&skipped).is_dir() && out.join(format!("{skipped}.flac")).is_file(),
        "the first render's front matter is still on disk"
    );
    let exported = export(&out, &d.join("book.flac")).unwrap();
    let names: Vec<&str> = exported["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["chapter"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 6, "{names:?}");
    assert!(!names.contains(&skipped.as_str()), "{names:?}");
    std::fs::remove_dir_all(&d).ok();
}

/// A chapter the latest render planned whose directory is gone is refused, naming it.
#[test]
fn a_planned_chapter_that_is_missing_is_refused() {
    let d = dir("planned-missing");
    let (out, report) = rendered(&d, false);
    let s = stem(&report, 5);
    std::fs::remove_dir_all(out.join(&s)).unwrap();
    std::fs::remove_file(out.join(format!("{s}.flac"))).unwrap();
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("planned but is not rendered") && err.contains(&s), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// A chapter file replaced after the render (here, by another chapter's file) is refused: its SHA-256 is not the one
/// the manifest recorded.
#[test]
fn a_chapter_file_changed_after_the_render_is_refused() {
    let d = dir("sha");
    let (out, report) = rendered(&d, false);
    let (a, b) = (stem(&report, 4), stem(&report, 5));
    std::fs::copy(out.join(format!("{a}.flac")), out.join(format!("{b}.flac"))).unwrap();
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(
        err.contains("not the one its manifest records") && err.contains(&b),
        "{err}"
    );
    std::fs::remove_dir_all(&d).ok();
}

/// A manifest with no chapter file recorded (a render cut short after the join, or written before the record) is
/// refused rather than trusted.
#[test]
fn a_chapter_file_with_no_record_is_refused() {
    let d = dir("no-record");
    let (out, report) = rendered(&d, false);
    let s = stem(&report, 4);
    edit_manifest(&out, &s, |m| {
        m.as_object_mut().unwrap().remove("chapter_file");
    });
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("no chapter file recorded") && err.contains(&s), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// Chapters rendered by two engines are refused, naming the chapter.
#[test]
fn chapters_from_two_engines_are_refused() {
    let d = dir("engines");
    let (out, report) = rendered(&d, false);
    let s = stem(&report, 5);
    edit_manifest(&out, &s, |m| m["engine"]["family"] = "another-engine".into());
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("another engine") && err.contains(&s), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// A chapter whose manifest names another book is refused, naming the chapter.
#[test]
fn a_chapter_of_another_book_is_refused() {
    let d = dir("other-book");
    let (out, report) = rendered(&d, false);
    let s = stem(&report, 5);
    edit_manifest(&out, &s, |m| m["book_id"] = "urn:uuid:another-book".into());
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("another book") && err.contains(&s), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// A chapter at another sample rate is refused, naming the chapter, even when its file is the one recorded.
#[test]
fn a_chapter_at_another_sample_rate_is_refused() {
    let d = dir("rate");
    let (out, report) = rendered(&d, false);
    let s = stem(&report, 5);
    let file = out.join(format!("{s}.flac"));
    let rate = claxon::FlacReader::open(&file).unwrap().streaminfo().sample_rate;
    let pcm: Vec<i16> = samples(&file).into_iter().map(|x| x as i16).collect();
    let bytes = ggtts_core::flac::encode_flac(&pcm, rate / 2).unwrap();
    std::fs::write(&file, &bytes).unwrap();
    let sha = hex(&sha256(&bytes));
    edit_manifest(&out, &s, |m| m["chapter_file"]["sha256"] = sha.into());
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("Hz") && err.contains(&s), "{err}");
    std::fs::remove_dir_all(&d).ok();
}

/// A stray directory with a non-ASCII name beside the render (here `009é`, where a byte slice at 3..4 would split
/// a character) neither panics the export nor is exported.
#[test]
fn a_stray_non_ascii_directory_is_ignored() {
    let d = dir("non-ascii");
    let (out, _) = rendered(&d, false);
    std::fs::create_dir_all(out.join("009é")).unwrap();
    std::fs::write(out.join("009é.flac"), b"not a flac").unwrap();
    let exported = export(&out, &d.join("book.flac")).unwrap();
    assert_eq!(exported["chapters"].as_array().unwrap().len(), 6);
    std::fs::remove_dir_all(&d).ok();
}

/// AC3: a chapter whose file is gone, and a directory that is not a render, are refused.
#[test]
fn a_missing_chapter_file_or_a_stranger_directory_is_refused() {
    let d = dir("missing");
    let (out, report) = rendered(&d, false);
    std::fs::remove_file(out.join(report["chapters"][4]["file"].as_str().unwrap())).unwrap();
    let err = export(&out, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("has no chapter file"), "{err}");
    let stranger = d.join("stranger");
    std::fs::create_dir_all(&stranger).unwrap();
    let err = export(&stranger, &d.join("book.flac")).unwrap_err();
    assert!(err.contains("book.json"), "{err}");
    std::fs::remove_dir_all(&d).ok();
}
