//! The book's command line, read by the shared options parser ([`ggtts_core::options`]).
//!
//! `ggtts-book inspect` and `render` take their flags from the one options table, so a flag means in a book what it
//! means in `ggtts-render`. The book takes the load section (the engine and its files), the style instruction, the
//! steering, the trace, the output directory (`--out`) and the book section (`--seed-root`, `--include-skippable`).
//! Every other option in the table is refused by name, with the reason a book does not take it ([`refusal`]). The
//! positional argument is the book.
//!
//! An option a book does not take that comes only from the environment (`GGTTS_LOG_LEVEL`, for example) is ambient,
//! as the table's environment rule says: it is dropped, never refused.

use std::path::PathBuf;

use ggtts_core::options::{descriptor, Kind, Options, Source, TABLE};

/// What `ggtts-book render` takes, by schema name.
pub const RENDER_TAKES: &[&str] = &[
    "load.engine",
    "load.model",
    "load.codec",
    "load.voice",
    "load.speaker",
    "load.device",
    "load.threads",
    "load.offline",
    "load.cache_dir",
    "render.instruction",
    "render.steer",
    "render.steer_scale",
    "render.steer_layers",
    "observe.trace",
    "observe.trace_level",
    "output.out",
    "book.seed_root",
    "book.include_skippable",
];

/// What `ggtts-book inspect` takes, by schema name: the engine whose profile plans the chunks, and the seed root.
pub const INSPECT_TAKES: &[&str] = &["load.engine", "book.seed_root"];

/// The engine a book renders with when none is named: `qwen3-tts`, as `ggtts-render` with no model.
pub const DEFAULT_ENGINE: &str = "qwen3-tts";

/// Why a book does not take the option `key` (a schema name), for any option outside [`RENDER_TAKES`].
pub fn refusal(key: &str) -> &'static str {
    match key {
        "render.text" | "render.story" => "a book renders its chapters, not a text or a story file",
        "render.seed" | "render.seed_salt" => "a book's seeds come from --seed-root and each chunk's place in the book",
        "output.stream" | "output.stream_format" | "output.stream_raw" | "output.stream_level_seconds" => {
            "a book is written as one FLAC per chapter, not streamed"
        }
        "observe.timing_report" | "observe.duration_report" => "a book writes no timing or duration report",
        "observe.log_level" | "observe.log_format" => "a book writes no log",
        "observe.instruments"
        | "observe.instrument"
        | "observe.require"
        | "observe.on_instrument_error"
        | "observe.instrument_timeout"
        | "observe.instrument_device"
        | "observe.report"
        | "observe.run" => "a book runs no instruments",
        _ => "a book does not take it",
    }
}

/// Split `args` (after the command) into the book and the flags, by the table's spellings: a flag that takes a value
/// takes the argument after it.
fn split(args: &[String]) -> Result<(String, Vec<String>), String> {
    let mut book = None;
    let mut flags = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(d) = TABLE.iter().find(|d| d.flags.contains(&arg.as_str())) {
            flags.push(arg.clone());
            if d.kind != Kind::Bool {
                if let Some(value) = it.next() {
                    flags.push(value.clone());
                }
            }
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg}"));
        } else if book.is_none() {
            book = Some(arg.clone());
        } else {
            return Err("one book at a time".into());
        }
    }
    Ok((book.ok_or("no book named")?, flags))
}

/// Read `command`'s arguments (`inspect` or `render`, the arguments after it) with the environment `var` beneath the
/// flags: the book's path and the options value. A flag the table does not have, a value the table refuses, and an
/// option the command does not take are refused by name.
pub fn parse(
    command: &str,
    args: &[String],
    var: &dyn Fn(&str) -> Option<String>,
) -> Result<(PathBuf, Options), String> {
    let (book, flags) = split(args)?;
    let given = Options::from_args(&flags).map_err(|e| e.to_string())?;
    let o = Options::from_env(var)
        .map_err(|e| e.to_string())?
        .layer(given)
        .map_err(|e| e.to_string())?;
    let takes = if command == "render" {
        RENDER_TAKES
    } else {
        INSPECT_TAKES
    };
    let mut dropped = Vec::new();
    for (key, source) in &o.marks.sources {
        if takes.contains(&key.as_str()) {
            continue;
        }
        if *source == Source::Environment {
            dropped.push(key.clone());
            continue;
        }
        let flag = descriptor(key).and_then(|d| d.flags.first()).copied().unwrap_or("");
        let reason = if command != "render" && RENDER_TAKES.contains(&key.as_str()) {
            "it is for render"
        } else {
            refusal(key)
        };
        return Err(format!("{flag} ({key}) is refused by ggtts-book {command}: {reason}"));
    }
    let mut o = without(o, &dropped)?;
    if !o.marks.sources.contains_key("load.engine") {
        o.load.engine = DEFAULT_ENGINE.into();
    }
    Ok((PathBuf::from(book), o))
}

/// `o` with the options `keys` reset to their defaults and their marks removed.
fn without(o: Options, keys: &[String]) -> Result<Options, String> {
    if keys.is_empty() {
        return Ok(o);
    }
    let mut doc = o.to_json();
    for key in keys {
        let (section, name) = key.split_once('.').expect("a mark is section.name");
        if let Some(entries) = doc[section].as_object_mut() {
            entries.remove(name);
        }
    }
    let mut n: Options = serde_json::from_value(doc).map_err(|e| e.to_string())?;
    n.marks = o.marks;
    for key in keys {
        n.marks.sources.remove(key);
    }
    n.render.steering = o.render.steering;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn every_option_render_takes_is_a_row_of_the_table() {
        for key in RENDER_TAKES.iter().chain(INSPECT_TAKES) {
            assert!(descriptor(key).is_some(), "{key}");
        }
    }

    #[test]
    fn an_ambient_option_a_book_does_not_take_is_dropped_not_refused() {
        let env = |name: &str| (name == "GGTTS_LOG_LEVEL").then(|| "debug".to_string());
        let (_, o) = parse("render", &args(&["b.epub", "--out", "d"]), &env).unwrap();
        assert_eq!(o.observe.log_level, None);
        assert!(!o.marks.sources.contains_key("observe.log_level"));
        let err = parse("render", &args(&["b.epub", "--log-level", "debug"]), &|_| None).unwrap_err();
        assert!(err.contains("observe.log_level"), "{err}");
    }

    #[test]
    fn the_book_may_come_anywhere_among_the_flags() {
        let none = |_: &str| None;
        let (book, o) = parse("render", &args(&["--out", "d", "b.epub", "--offline"]), &none).unwrap();
        assert_eq!(book, PathBuf::from("b.epub"));
        assert!(o.load.offline);
        assert_eq!(o.load.engine, DEFAULT_ENGINE);
    }
}
