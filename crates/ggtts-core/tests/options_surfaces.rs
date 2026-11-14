//! Every option a front end takes today is a row of the options table (`ggtts_core::options::TABLE`), with its phase.
//!
//! The four surfaces are read from their source: `ggtts-render`'s flags (its usage line, its parser and the instrument
//! flags it hands on), the Python binding's `ggtts.load` and render methods, the C ABI's `ggtts_load_with`,
//! `ggtts_settings` (version 3) and `ggtts_render`, and Kotlin's `Ggtts.load` and render methods. A spelling there that
//! is not in the table fails, unless it is listed below as not an option, with why.

use std::path::Path;

use ggtts_core::options::{Descriptor, TABLE};

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// A source file with LF line ends: a Windows checkout may hold CRLF (core.autocrlf), and the readers below split on
/// `\n`.
fn read(path: &str) -> String {
    std::fs::read_to_string(root().join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .replace("\r\n", "\n")
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Surface {
    Flag,
    Python,
    C,
    Kotlin,
}

/// Spellings that are not options: commands, callbacks, out-parameters and handles.
const NOT_OPTIONS: &[(Surface, &str, &str)] = &[
    (Surface::Flag, "--version", "a command: prints the version"),
    (Surface::Flag, "--help", "a command: prints the usage"),
    (Surface::Flag, "-h", "a command: prints the usage"),
    (
        Surface::Python,
        "progress",
        "a callback for download messages, not a value",
    ),
    (Surface::Python, "on_chunk", "the stream's callback, not a value"),
    (Surface::C, "size", "the struct's own size, its version"),
    (Surface::C, "settings", "the struct the settings are in"),
    (Surface::C, "handle", "the engine handle"),
    (Surface::C, "pcm", "an out-parameter"),
    (Surface::C, "len", "an out-parameter"),
    (Surface::C, "rate", "an out-parameter"),
];

fn spellings(d: &Descriptor, s: Surface) -> &'static [&'static str] {
    match s {
        Surface::Flag => d.flags,
        Surface::Python => d.python,
        Surface::C => d.c,
        Surface::Kotlin => d.kotlin,
    }
}

/// The spellings in `found` that no row of `table` has and that are not listed as not options.
fn missing(table: &[Descriptor], surface: Surface, found: &[String]) -> Vec<String> {
    found
        .iter()
        .filter(|name| !NOT_OPTIONS.iter().any(|(s, n, _)| *s == surface && n == name))
        .filter(|name| !table.iter().any(|d| spellings(d, surface).contains(&name.as_str())))
        .cloned()
        .collect()
}

/// The text between `start` and the first `end` after it.
fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    let i = text.find(start).unwrap_or_else(|| panic!("{start:?} not found")) + start.len();
    let j = text[i..]
        .find(end)
        .unwrap_or_else(|| panic!("{end:?} not found after {start:?}"));
    &text[i..i + j]
}

/// Every occurrence of the text between `start` and `end`.
fn all_between<'a>(text: &'a str, start: &str, end: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(start) {
        let after = &rest[i + start.len()..];
        let j = after
            .find(end)
            .unwrap_or_else(|| panic!("{end:?} not found after {start:?}"));
        out.push(&after[..j]);
        rest = &after[j..];
    }
    out
}

/// Splits a parameter list at the commas outside brackets.
fn split_params(list: &str) -> Vec<&str> {
    let (mut depth, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, c) in list.char_indices() {
        match c {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&list[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&list[start..]);
    out.into_iter().map(str::trim).filter(|p| !p.is_empty()).collect()
}

/// A parameter's name: Python and Kotlin `name: Type = default`, pyo3 `name=default`, C `type *name`.
fn param_name(p: &str) -> String {
    let p = p.split(['=', ':']).next().unwrap().trim();
    let last = p
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|s| !s.is_empty())
        .unwrap();
    last.to_string()
}

fn names(list: &str) -> Vec<String> {
    split_params(list)
        .into_iter()
        .map(param_name)
        .filter(|n| n != "self")
        .collect()
}

fn flags(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    for (i, _) in text.match_indices('-') {
        if i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'-') {
            continue;
        }
        let end = text[i..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .map_or(text.len(), |j| i + j);
        let token = &text[i..end];
        let long = token.starts_with("--") && token.len() > 2 && token.as_bytes()[2].is_ascii_lowercase();
        let short = token.len() == 2 && token.as_bytes()[1].is_ascii_lowercase();
        if (long || short) && !out.iter().any(|f| f == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn render_flags() -> Vec<String> {
    let src = read("crates/ggtts-cli/src/bin/ggtts-render.rs");
    let usage = between(&src, "const USAGE: &str = \"", "\";");
    let parse = between(&src, "fn parse(args: &[String]) -> Result<Args, String> {", "\n}\n");
    let instruments = read("crates/ggtts-instruments/src/config.rs");
    let names = between(&instruments, "pub const NAMES", "];");
    let mut out = flags(usage);
    // Only the parser's quoted flags: its error messages name flags too, and those are in the list already.
    for quoted in all_between(parse, "\"-", "\"") {
        for f in flags(&format!("-{quoted}")) {
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }
    for f in flags(names) {
        if !out.contains(&f) {
            out.push(f);
        }
    }
    let main = between(&src, "fn main() -> ExitCode {", "\n}\n");
    for f in flags(main)
        .into_iter()
        .filter(|f| ["--version", "--help", "-h"].contains(&f.as_str()))
    {
        if !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

fn python_names() -> Vec<String> {
    let py = read("crates/ggtts-py/python/ggtts/__init__.py");
    let mut out = names(between(&py, "\ndef load(", ") -> TTS:"));
    for method in ["def render(", "def render_stream(", "def stream("] {
        out.extend(names(between(&py, method, ") ->")));
    }
    let native = read("crates/ggtts-py/src/python.rs");
    out.extend(names(between(&native, "#[pyo3(signature = (", "))]")));
    out.sort();
    out.dedup();
    out
}

fn c_names() -> Vec<String> {
    let h = read("crates/ggtts-ffi/include/ggtts.h");
    let settings = between(&h, "typedef struct ggtts_settings {", "} ggtts_settings;");
    let mut out: Vec<String> = settings
        .split(';')
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(param_name)
        .collect();
    out.extend(names(between(&h, "int32_t ggtts_load_with(", ");")));
    out.extend(names(between(&h, "int32_t ggtts_render(", ");")));
    out.sort();
    out.dedup();
    out
}

fn kotlin_names() -> Vec<String> {
    let kt = read("android/ggtts/src/main/kotlin/ggtts/Ggtts.kt");
    let class = between(&kt, "class Ggtts private constructor", "\n}\n");
    let mut out = Vec::new();
    for list in all_between(class, "fun load(", "): Ggtts") {
        out.extend(names(list));
    }
    out.extend(names(between(class, "fun render(", "): Pcm")));
    out.sort();
    out.dedup();
    out
}

fn surfaces() -> Vec<(Surface, Vec<String>)> {
    vec![
        (Surface::Flag, render_flags()),
        (Surface::Python, python_names()),
        (Surface::C, c_names()),
        (Surface::Kotlin, kotlin_names()),
    ]
}

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
        "-j",
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
