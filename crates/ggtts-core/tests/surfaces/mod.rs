//! The readers of each front end's option spellings, from its source: `ggtts-render`'s flags, the Python binding's
//! `ggtts.load` and render methods, the C ABI's `ggtts.h`, and Kotlin's `Ggtts.kt`. Shared by `options_surfaces.rs`
//! (every spelling is a row of the options table) and `ggtts-cli`'s conformance test (the binding surface snapshots),
//! which includes this file by path.

#![allow(dead_code)]

use std::path::Path;

use ggtts_core::options::Descriptor;

pub fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// A source file with LF line ends: a Windows checkout may hold CRLF (core.autocrlf), and the readers below split on
/// `\n`.
pub fn read(path: &str) -> String {
    std::fs::read_to_string(root().join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .replace("\r\n", "\n")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surface {
    Flag,
    Python,
    C,
    Kotlin,
}

/// Spellings that are not options: commands, callbacks, out-parameters and handles.
pub const NOT_OPTIONS: &[(Surface, &str, &str)] = &[
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

pub fn spellings(d: &Descriptor, s: Surface) -> &'static [&'static str] {
    match s {
        Surface::Flag => d.flags,
        Surface::Python => d.python,
        Surface::C => d.c,
        Surface::Kotlin => d.kotlin,
    }
}

/// Whether `name` on `surface` is spelled by a row of `table` or listed as not an option.
pub fn known(table: &[Descriptor], surface: Surface, name: &str) -> bool {
    NOT_OPTIONS.iter().any(|(s, n, _)| *s == surface && *n == name)
        || table.iter().any(|d| spellings(d, surface).contains(&name))
}

/// The spellings in `found` that no row of `table` has and that are not listed as not options.
pub fn missing(table: &[Descriptor], surface: Surface, found: &[String]) -> Vec<String> {
    found
        .iter()
        .filter(|name| !known(table, surface, name))
        .cloned()
        .collect()
}

/// The text between `start` and the first `end` after it.
pub fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    let i = text.find(start).unwrap_or_else(|| panic!("{start:?} not found")) + start.len();
    let j = text[i..]
        .find(end)
        .unwrap_or_else(|| panic!("{end:?} not found after {start:?}"));
    &text[i..i + j]
}

/// Every occurrence of the text between `start` and `end`.
pub fn all_between<'a>(text: &'a str, start: &str, end: &str) -> Vec<&'a str> {
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
pub fn split_params(list: &str) -> Vec<&str> {
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
pub fn param_name(p: &str) -> String {
    let p = p.split(['=', ':']).next().unwrap().trim();
    let last = p
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|s| !s.is_empty())
        .unwrap();
    last.to_string()
}

pub fn names(list: &str) -> Vec<String> {
    split_params(list)
        .into_iter()
        .map(param_name)
        .filter(|n| n != "self")
        .collect()
}

pub fn flags(text: &str) -> Vec<String> {
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

pub fn render_flags() -> Vec<String> {
    let src = read("crates/ggtts-cli/src/bin/ggtts-render.rs");
    let usage = between(&src, "const USAGE: &str = \"", "\";");
    let instruments = read("crates/ggtts-instruments/src/config.rs");
    let names = between(&instruments, "pub const NAMES", "];");
    let mut out = flags(usage);
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

pub fn python_names() -> Vec<String> {
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

pub fn c_names() -> Vec<String> {
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

pub fn kotlin_names() -> Vec<String> {
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

pub fn surfaces() -> Vec<(Surface, Vec<String>)> {
    vec![
        (Surface::Flag, render_flags()),
        (Surface::Python, python_names()),
        (Surface::C, c_names()),
        (Surface::Kotlin, kotlin_names()),
    ]
}

/// One line of a binding surface snapshot: the surface, the function or struct, and its parameter or field names.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub surface: Surface,
    pub item: String,
    pub names: Vec<String>,
}

impl Entry {
    pub fn line(&self) -> String {
        let s = match self.surface {
            Surface::Flag => "flag",
            Surface::Python => "python",
            Surface::C => "c",
            Surface::Kotlin => "kotlin",
        };
        if self.names.is_empty() {
            format!("{s} {}:", self.item)
        } else {
            format!("{s} {}: {}", self.item, self.names.join(", "))
        }
    }
}

/// The pyo3 functions of `python.rs` (its source text): `load`'s `#[pyo3(signature = ...)]`, and the parameters of the
/// engine's `render` and `render_stream` methods (less `self` and the interpreter token `py`).
pub fn pyo3_entries(src: &str) -> Vec<Entry> {
    let mut out = vec![Entry {
        surface: Surface::Python,
        item: "load".into(),
        names: names(between(src, "#[pyo3(signature = (", "))]")),
    }];
    for f in ["render", "render_stream"] {
        let params = between(src, &format!("    fn {f}"), ") ->");
        let params = &params[params.find('(').expect("a parameter list") + 1..];
        out.push(Entry {
            surface: Surface::Python,
            item: f.into(),
            names: names(params).into_iter().filter(|n| n != "py").collect(),
        });
    }
    out
}

/// `ggtts.h` (its text): the `ggtts_settings` struct's fields, then every exported function in header order.
pub fn header_entries(h: &str) -> Vec<Entry> {
    let mut text = String::new();
    let mut rest = h;
    while let Some(i) = rest.find("/*") {
        text.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("a closed comment");
        rest = &rest[i + j + 2..];
    }
    text.push_str(rest);
    let settings = between(&text, "typedef struct ggtts_settings {", "} ggtts_settings;");
    let mut out = vec![Entry {
        surface: Surface::C,
        item: "ggtts_settings".into(),
        names: settings
            .split(';')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(param_name)
            .collect(),
    }];
    let body = text.replace(settings, "");
    for statement in body.split(';').map(str::trim) {
        let statement = statement
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join(" ");
        let Some(open) = statement.find('(') else {
            continue;
        };
        if statement.trim_start().starts_with("typedef") || statement.contains('}') {
            continue;
        }
        let name = param_name(&statement[..open]);
        if !name.starts_with("ggtts_") {
            continue;
        }
        let close = statement.rfind(')').expect("a closed parameter list");
        let params: Vec<String> = names(&statement[open + 1..close])
            .into_iter()
            .filter(|n| n != "void")
            .collect();
        out.push(Entry {
            surface: Surface::C,
            item: name,
            names: params,
        });
    }
    out
}

/// `Ggtts.kt` (its text): every `external fun`, with its parameters.
pub fn kotlin_entries(kt: &str) -> Vec<Entry> {
    all_between(kt, "external fun ", ")")
        .into_iter()
        .map(|decl| {
            let open = decl.find('(').expect("a parameter list");
            Entry {
                surface: Surface::Kotlin,
                item: decl[..open].trim().to_string(),
                names: names(&decl[open + 1..]),
            }
        })
        .collect()
}

/// The three binding surfaces as they are in the tree.
pub fn binding_entries() -> Vec<Entry> {
    let mut out = pyo3_entries(&read("crates/ggtts-py/src/python.rs"));
    out.extend(header_entries(&read("crates/ggtts-ffi/include/ggtts.h")));
    out.extend(kotlin_entries(&read("android/ggtts/src/main/kotlin/ggtts/Ggtts.kt")));
    out
}
