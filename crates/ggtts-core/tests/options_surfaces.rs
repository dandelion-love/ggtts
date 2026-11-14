//! Every option a front end takes today is a row of the options table (`ggtts_core::options::TABLE`), with its phase.
//!
//! The four surfaces are read from their source: `ggtts-render`'s flags (its usage line and the instrument flags it
//! hands on; since #289 it has no parser of its own, only the table's), the Python binding's `ggtts.load` and render methods, the C ABI's `ggtts_load_with`,
//! `ggtts_settings` (version 3), `ggtts_render` and the options-document functions, and Kotlin's `Ggtts.load` and render methods. A spelling there that
//! is not in the table fails, unless it is listed as not an option (`surfaces::NOT_OPTIONS`), with why. The readers
//! are in `surfaces/mod.rs`, shared with `ggtts-cli`'s conformance test.

mod surfaces;

use ggtts_core::options::{Descriptor, TABLE};
use surfaces::*;

/// The reader finds what it should: a reader that found nothing would pass the check below vacuously.
#[test]
fn each_surface_is_read_from_its_source() {
    let s = surfaces();
    let has = |surface: Surface, name: &str| s.iter().any(|(x, l)| *x == surface && l.iter().any(|n| n == name));
    for f in [
        "--model",
        "--stream-raw",
        "--instrument-timeout",
        "--log-format",
        "--seed",
        "--version",
    ] {
        assert!(has(Surface::Flag, f), "{f} not read from ggtts-render: {:?}", s[0].1);
    }
    for n in ["engine", "offline", "instruction", "text", "seed", "progress"] {
        assert!(
            has(Surface::Python, n),
            "{n} not read from the Python binding: {:?}",
            s[1].1
        );
    }
    for n in [
        "size",
        "speaker",
        "device",
        "cache_dir",
        "allow_download",
        "threads",
        "seed",
    ] {
        assert!(has(Surface::C, n), "{n} not read from ggtts.h: {:?}", s[2].1);
    }
    for n in ["engine", "cacheDir", "allowDownload", "instruction", "seed"] {
        assert!(has(Surface::Kotlin, n), "{n} not read from Ggtts.kt: {:?}", s[3].1);
    }
}

#[test]
fn every_front_end_option_is_a_row_of_the_table() {
    let mut gaps = Vec::new();
    for (surface, found) in surfaces() {
        for name in missing(TABLE, surface, &found) {
            gaps.push(format!("{surface:?} {name}"));
        }
    }
    assert!(gaps.is_empty(), "options no row of the table spells: {gaps:?}");
}

/// The check fails when a row is taken out: the table without its `offline` row misses the flag and each binding's
/// spelling of it.
#[test]
fn a_table_without_a_row_fails_the_check() {
    let planted: Vec<Descriptor> = TABLE.iter().filter(|d| d.name != "offline").copied().collect();
    let s = surfaces();
    assert_eq!(missing(&planted, Surface::Flag, &s[0].1), vec!["--offline"]);
    assert_eq!(missing(&planted, Surface::Python, &s[1].1), vec!["offline"]);
    assert_eq!(missing(&planted, Surface::C, &s[2].1), vec!["allow_download"]);
    assert_eq!(missing(&planted, Surface::Kotlin, &s[3].1), vec!["allowDownload"]);
}

/// Every not-an-option entry is still on its surface, so the list cannot hide a spelling that has gone.
#[test]
fn every_exemption_is_still_on_its_surface() {
    let s = surfaces();
    for (surface, name, why) in NOT_OPTIONS {
        assert!(!why.is_empty());
        let found = &s.iter().find(|(x, _)| x == surface).unwrap().1;
        assert!(
            found.iter().any(|n| n == name),
            "{surface:?} {name} is exempt and no longer there"
        );
    }
}

/// The phase each binding gives a value at today, beside the table's: the instruction is a render option that the
/// bindings fix at load, as every render's default. The rest of a binding's load arguments are load options.
#[test]
fn binding_load_arguments_are_load_options_except_the_instruction() {
    let py = read("crates/ggtts-py/python/ggtts/__init__.py");
    let load = names(between(&py, "\ndef load(", ") -> TTS:"));
    for name in load.iter().filter(|n| *n != "progress") {
        let d = TABLE.iter().find(|d| d.python.contains(&name.as_str())).unwrap();
        let expect = if name == "instruction" { "render" } else { "load" };
        assert_eq!(d.phase.name(), expect, "python {name}");
    }
}
