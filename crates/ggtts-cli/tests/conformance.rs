//! The capability conformance test: every option of the options table (`ggtts_core::options::TABLE`), for every front
//! end, is carried, refused with a reason, or a known gap from a baseline that can only shrink.
//!
//! * **The table** ([`FRONT_ENDS`]) gives each front end one cell per option: [`Cell::Carried`],
//!   [`Cell::Refused`] with the reason, or [`Cell::Gap`] with the issue that closes it. An option with no cell on a
//!   front end fails, naming the option, and so does a cell for an option the table does not have.
//! * **Declared is not exercised.** A cell is exercised when a probe ([`probes`]) drives the front end with the option
//!   set and sees it where it is read (an output file, a trace line, the seed a trace records) or sees the refusal. A
//!   probe that sees `Carried` closes a gap: the test then fails until the gap's line leaves the baseline. Every other
//!   cell is declared, and the test prints how many.
//! * **The gap baseline**, `conformance/gaps.txt`: exactly the table's gaps, one line each. A gap missing from it
//!   fails (so a new option cannot enter as a gap), and so does a line whose cell is no longer a gap.
//!   `scripts/check_conformance_base.py --base REF` refuses a line that is not in the file at `REF`, so the file
//!   only shrinks.
//! * **Binding surface snapshots**, `conformance/surfaces.txt`: the pyo3 `load`, `render` and `render_stream`
//!   parameters, `ggtts.h`'s `ggtts_settings` fields and exported functions, and `Ggtts.kt`'s `external fun`s. Any
//!   change fails, naming each added name and whether a row of the options table spells it.
//! * **The render path**: nothing outside the shared pipeline calls an engine's `generate`, `generate_traced`,
//!   `generate_hooked` or `decode` to render, except the files in `conformance/render-path-exceptions.txt`, which the
//!   same `--base` check keeps shrinking. See [`engine_calls`] for what the scan counts and leaves out.
//!
//! The counts are written to stderr directly (libtest captures only the print macros), so they reach the CI log.

#[path = "../../ggtts-core/tests/surfaces/mod.rs"]
mod surfaces;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use ggtts_core::options::{Descriptor, Kind, TABLE};
use surfaces::{known, read, root, Entry, Surface};

// ── the table ──────────────────────────────────────────────────────────────────────────────────────────────────────

/// What a front end does with an option.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cell {
    /// Passed through to where it is read.
    Carried,
    /// Refused, for this reason.
    Refused(&'static str),
    /// Not carried yet: the issue that closes it (a story of the one-render-surface epic, or the bug that tracks it).
    Gap(u32),
}

use Cell::{Carried, Gap, Refused};

/// One front end and its cells: groups of option keys (`section.name`), each group sharing a cell.
struct FrontEnd {
    id: &'static str,
    cells: &'static [(&'static [&'static str], Cell)],
}

const LOAD: &[&str] = &[
    "load.engine",
    "load.model",
    "load.codec",
    "load.voice",
    "load.speaker",
    "load.device",
    "load.threads",
    "load.offline",
    "load.cache_dir",
];
const STEER: &[&str] = &["render.steer", "render.steer_scale", "render.steer_layers"];
const TRACE: &[&str] = &["observe.trace", "observe.trace_level"];
const REPORTS: &[&str] = &["observe.timing_report", "observe.duration_report"];
const LOG: &[&str] = &["observe.log_level", "observe.log_format"];
const INSTRUMENTS: &[&str] = &[
    "observe.instruments",
    "observe.instrument",
    "observe.require",
    "observe.on_instrument_error",
    "observe.instrument_timeout",
    "observe.instrument_device",
    "observe.report",
    "observe.run",
];
const FILES_OUT: &[&str] = &["output.out", "output.stream", "output.stream_format"];
const STREAM_LEVEL: &[&str] = &["output.stream_raw", "output.stream_level_seconds"];
const STREAM: &[&str] = &[
    "output.stream",
    "output.stream_format",
    "output.stream_raw",
    "output.stream_level_seconds",
];
const BOOK: &[&str] = &["book.seed_root", "book.include_skippable"];

const NOT_A_BOOK: &str = "ggtts-book's own options: ggtts-render renders a text or a story, not a book";
const ONE_OF: &str = "--text and --story are one or the other";
const TEXT_SALT: &str = "a one-chunk plan renders at --seed: there is no story seed to salt";
const NO_INSTRUMENTS: &str =
    "instruments are programs run on a finished file; run them with `ggtts-render` or on the file";
const NO_FILE: &str = "a binding hands the PCM to its caller and writes no file";
const BINDING_SEED: &str = "a binding's seed is the story seed's salt (render.seed_salt); render.seed is the sampler seed of ggtts-render --text";
const NO_STORY_FILE: &str = "a binding renders the text it is given; a story file is ggtts-render --story's";
const NOT_BOOK_FRONT: &str = "ggtts-book's own options";
const ANDROID_TRACE: &str = "the trace is a desktop observer: refused on Android";
const CPU_ONLY: &str = "the Android build is CPU only";
const NEVER_DOWNLOADS: &str = "the app never downloads: its models are installed with it";
const PARITY_TOOL: &str = "an engine parity tool; render with ggtts-render --engine kitten-tts-2";
// `ggtts_book::options::refusal`'s reasons, word for word (checked by `book_refusals_are_the_books_own`).
const BOOK_TEXT: &str = "a book renders its chapters, not a text or a story file";
const BOOK_SEED: &str = "a book's seeds come from --seed-root and each chunk's place in the book";
const BOOK_STREAM: &str = "a book is written as one FLAC per chapter, not streamed";
const BOOK_REPORTS: &str = "a book writes no timing or duration report";
const BOOK_LOG: &str = "a book writes no log";
const BOOK_TIMINGS: &str = "a book does not take it";

/// Every front end of the direction document (`docs/architecture-direction.md`), with `ggtts-render`'s two modes
/// apart because they carry different options.
const FRONT_ENDS: &[FrontEnd] = &[
    FrontEnd {
        id: "render-story",
        cells: &[
            (LOAD, Carried),
            (
                &[
                    "render.story",
                    "render.instruction",
                    "render.redraw",
                    "render.measure_timings",
                ],
                Carried,
            ),
            (STEER, Carried),
            (&["render.text"], Refused(ONE_OF)),
            (&["render.seed"], Gap(215)),
            (&["render.seed_salt", "render.edge_s"], Gap(283)),
            (TRACE, Carried),
            (REPORTS, Carried),
            (LOG, Carried),
            (INSTRUMENTS, Carried),
            (&["output.out"], Carried),
            (STREAM, Carried),
            (BOOK, Refused(NOT_A_BOOK)),
        ],
    },
    FrontEnd {
        id: "render-text",
        cells: &[
            (LOAD, Carried),
            (
                &[
                    "render.text",
                    "render.seed",
                    "render.instruction",
                    "render.measure_timings",
                ],
                Carried,
            ),
            (STEER, Carried),
            (&["render.story"], Refused(ONE_OF)),
            (&["render.seed_salt"], Refused(TEXT_SALT)),
            (&["render.redraw"], Gap(294)),
            (&["render.edge_s"], Gap(180)),
            (TRACE, Gap(294)),
            (REPORTS, Carried),
            (LOG, Carried),
            (INSTRUMENTS, Carried),
            (&["output.out", "output.stream", "output.stream_format"], Carried),
            (STREAM_LEVEL, Gap(294)),
            (BOOK, Refused(NOT_A_BOOK)),
        ],
    },
    FrontEnd {
        id: "book",
        cells: &[
            (LOAD, Carried),
            (&["render.instruction", "render.redraw"], Carried),
            (STEER, Carried),
            (&["render.text", "render.story"], Refused(BOOK_TEXT)),
            (&["render.seed", "render.seed_salt"], Refused(BOOK_SEED)),
            (&["render.edge_s"], Gap(283)),
            (&["render.measure_timings"], Refused(BOOK_TIMINGS)),
            (TRACE, Carried),
            (REPORTS, Refused(BOOK_REPORTS)),
            (LOG, Refused(BOOK_LOG)),
            (INSTRUMENTS, Carried),
            (&["output.out"], Carried),
            (STREAM, Refused(BOOK_STREAM)),
            (BOOK, Carried),
        ],
    },
    FrontEnd {
        id: "c-abi",
        cells: &[
            (LOAD, Carried),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (STEER, Gap(280)),
            (&["render.redraw", "render.edge_s", "render.measure_timings"], Gap(280)),
            (TRACE, Gap(280)),
            (REPORTS, Gap(280)),
            (LOG, Gap(280)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (FILES_OUT, Refused(NO_FILE)),
            (STREAM_LEVEL, Gap(284)),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "python",
        cells: &[
            (
                &[
                    "load.engine",
                    "load.model",
                    "load.codec",
                    "load.voice",
                    "load.speaker",
                    "load.device",
                    "load.threads",
                    "load.offline",
                ],
                Carried,
            ),
            (&["load.cache_dir"], Gap(291)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (STEER, Gap(291)),
            (&["render.redraw", "render.edge_s", "render.measure_timings"], Gap(291)),
            (TRACE, Gap(291)),
            (REPORTS, Gap(291)),
            (LOG, Gap(291)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (FILES_OUT, Refused(NO_FILE)),
            (STREAM_LEVEL, Gap(284)),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "kotlin",
        cells: &[
            (
                &[
                    "load.engine",
                    "load.model",
                    "load.codec",
                    "load.voice",
                    "load.speaker",
                    "load.threads",
                    "load.offline",
                    "load.cache_dir",
                ],
                Carried,
            ),
            (&["load.device"], Refused(CPU_ONLY)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (STEER, Gap(292)),
            (&["render.redraw", "render.edge_s", "render.measure_timings"], Gap(292)),
            (TRACE, Refused(ANDROID_TRACE)),
            (REPORTS, Gap(292)),
            (LOG, Gap(292)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (FILES_OUT, Refused(NO_FILE)),
            (STREAM_LEVEL, Gap(284)),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "android",
        cells: &[
            (
                &[
                    "load.engine",
                    "load.model",
                    "load.codec",
                    "load.voice",
                    "load.speaker",
                    "load.threads",
                ],
                Carried,
            ),
            (&["load.device"], Refused(CPU_ONLY)),
            (&["load.offline"], Refused(NEVER_DOWNLOADS)),
            (&["load.cache_dir"], Gap(282)),
            (&["render.text", "render.seed_salt"], Carried),
            (&["render.instruction"], Gap(282)),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (STEER, Gap(282)),
            (&["render.redraw", "render.edge_s", "render.measure_timings"], Gap(282)),
            (TRACE, Refused(ANDROID_TRACE)),
            (REPORTS, Gap(282)),
            (LOG, Gap(282)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (FILES_OUT, Refused(NO_FILE)),
            (STREAM_LEVEL, Gap(284)),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "kitten",
        cells: &[
            (LOAD, Refused(PARITY_TOOL)),
            (
                &[
                    "render.text",
                    "render.story",
                    "render.seed",
                    "render.seed_salt",
                    "render.instruction",
                    "render.redraw",
                    "render.edge_s",
                    "render.measure_timings",
                ],
                Refused(PARITY_TOOL),
            ),
            (STEER, Refused(PARITY_TOOL)),
            (TRACE, Refused(PARITY_TOOL)),
            (REPORTS, Refused(PARITY_TOOL)),
            (LOG, Refused(PARITY_TOOL)),
            (INSTRUMENTS, Refused(PARITY_TOOL)),
            (&["output.out"], Refused(PARITY_TOOL)),
            (STREAM, Refused(PARITY_TOOL)),
            (BOOK, Refused(PARITY_TOOL)),
        ],
    },
];

/// Each front end's cell per option key, or every defect of the table against `table`: an option with no cell, a key
/// given twice, a key the options table does not have, a refusal with no reason.
fn cells(table: &[Descriptor], front_ends: &[FrontEnd]) -> Result<BTreeMap<(String, String), Cell>, Vec<String>> {
    let keys: BTreeSet<String> = table.iter().map(Descriptor::key).collect();
    let mut out = BTreeMap::new();
    let mut errors = Vec::new();
    for fe in front_ends {
        for (group, cell) in fe.cells {
            for key in *group {
                if !keys.contains(*key) {
                    errors.push(format!("{}: {key} is not an option of the table", fe.id));
                }
                if matches!(cell, Refused(r) if r.trim().is_empty()) {
                    errors.push(format!("{}: {key} is refused with no reason", fe.id));
                }
                if out.insert((fe.id.to_string(), key.to_string()), *cell).is_some() {
                    errors.push(format!("{}: {key} has two cells", fe.id));
                }
            }
        }
        for key in &keys {
            if !out.contains_key(&(fe.id.to_string(), key.clone())) {
                errors.push(format!(
                    "{}: {key} has no cell: say Carried or Refused(\"why\") (a new option never enters as a gap)",
                    fe.id
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

// ── the gap baseline ───────────────────────────────────────────────────────────────────────────────────────────────

const GAPS: &str = "conformance/gaps.txt";
const EXCEPTIONS: &str = "conformance/render-path-exceptions.txt";
const SNAPSHOT: &str = "conformance/surfaces.txt";

/// A list file's entries: its lines less blank lines and `# ` comments.
fn entries(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn gap_line(front_end: &str, key: &str, issue: u32) -> String {
    format!("{front_end}\t{key}\t#{issue}")
}

/// The baseline's defects against the table's cells: a gap not in it, and a line whose cell is no longer that gap.
fn check_gaps(cells: &BTreeMap<(String, String), Cell>, baseline: &str) -> Vec<String> {
    let listed: BTreeSet<&str> = entries(baseline).into_iter().collect();
    let mut want = BTreeSet::new();
    let mut errors = Vec::new();
    for ((fe, key), cell) in cells {
        if let Gap(issue) = cell {
            let line = gap_line(fe, key, *issue);
            if !listed.contains(line.as_str()) {
                errors.push(format!(
                    "{fe}: {key} is a gap not in {GAPS} (a new option enters only as Carried or Refused): {line:?}"
                ));
            }
            want.insert(line);
        }
    }
    for line in listed {
        if !want.contains(line) {
            errors.push(format!(
                "{GAPS} lists {line:?}, which is no longer a gap of the table: remove the line"
            ));
        }
    }
    errors
}

// ── probes: a front end driven with the option set ─────────────────────────────────────────────────────────────────

/// What a probe saw.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Seen {
    /// The option reached where it is read, or the engine's refusal of it.
    Carried,
    /// The front end refused it, naming it.
    Refused,
    /// Accepted with no effect.
    Dropped,
}

/// Whether what a probe saw is what the cell says: `Carried` must be seen carried, `Refused` refused, and a gap must
/// not be carried (a carried gap is closed, and its baseline line must go).
fn verdict(front_end: &str, key: &str, cell: Cell, seen: Seen) -> Result<(), String> {
    match (cell, seen) {
        (Carried, Seen::Carried) | (Refused(_), Seen::Refused) => Ok(()),
        (Gap(_), Seen::Refused | Seen::Dropped) => Ok(()),
        (Gap(i), Seen::Carried) => Err(format!(
            "{front_end}: {key} is carried now, so gap #{i} is closed: mark it Carried and remove its line from {GAPS}"
        )),
        (cell, seen) => Err(format!(
            "{front_end}: {key} is {cell:?} in the table, and the probe saw it {seen:?}"
        )),
    }
}

struct Run {
    ok: bool,
    stderr: String,
    dir: PathBuf,
}

/// A fresh directory with a one-paragraph story in it.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-conformance-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("s.txt"), "Title\n你好。\n").unwrap();
    dir
}

/// `ggtts-render` on the fake engine, in `mode` (`story` or `text`), with `extra`; `--out o.flac` unless `extra`
/// names its own output.
fn render(name: &str, mode: &str, extra: &[&str]) -> Run {
    let dir = scratch(name);
    let mut args: Vec<&str> = vec!["--engine", "fake", "--log-level", "error"];
    match mode {
        "story" => args.extend(["--story", "s.txt"]),
        _ => args.extend(["--text", "你好。"]),
    }
    if !extra.iter().any(|a| *a == "--stream" || *a == "--out") {
        args.extend(["--out", "o.flac"]);
    }
    args.extend(extra);
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(&args)
        .current_dir(&dir)
        .env_remove("GGTTS_LOG_LEVEL")
        .env_remove("GGTTS_LOG_FORMAT")
        .output()
        .unwrap();
    Run {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        dir,
    }
}

fn nonempty(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() > 0)
}

/// A run whose success shows the option by a non-empty file.
fn by_file(run: &Run, file: &str, name: &str) -> Seen {
    if !run.ok {
        return if run.stderr.contains(name) {
            Seen::Refused
        } else {
            panic!("the probe's render failed: {}", run.stderr)
        };
    }
    if nonempty(&run.dir.join(file)) {
        Seen::Carried
    } else {
        Seen::Dropped
    }
}

/// A run on an engine without the option: the engine's refusal naming it shows it was carried there.
fn by_engine_refusal(run: &Run, name: &str) -> Seen {
    match (run.ok, run.stderr.contains(name)) {
        (false, true) => Seen::Carried,
        (false, false) => panic!("the probe's render failed: {}", run.stderr),
        (true, _) => Seen::Dropped,
    }
}

/// A run that sets a flag the front end does not take: refused, naming the flag.
fn by_front_end_refusal(run: &Run, flag: &str) -> Seen {
    match (run.ok, run.stderr.contains(flag)) {
        (false, true) => Seen::Refused,
        (false, false) => panic!("refused, but not naming {flag}: {}", run.stderr),
        (true, _) => Seen::Dropped,
    }
}

/// The `seed` of every `chunk_begin` in a trace.
fn trace_seeds(path: &Path) -> Vec<u64> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["ev"] == "chunk_begin")
        .filter_map(|v| v["seed"].as_u64())
        .collect()
}

/// A value to give a flag of `d` in a probe.
fn sample_value(d: &Descriptor) -> Option<&'static str> {
    match d.kind {
        Kind::Bool => None,
        Kind::Int { .. } | Kind::Number { .. } | Kind::Seed => Some("1"),
        Kind::Choice(choices) => choices.first().copied(),
        Kind::Layers => Some("0"),
        _ => Some("x"),
    }
}

/// A probe: the front end, the option key, and what it saw.
type Probe = (&'static str, String, Seen);

fn render_probes() -> Vec<Probe> {
    let mut out = Vec::new();
    for mode in ["story", "text"] {
        let fe: &'static str = if mode == "story" { "render-story" } else { "render-text" };
        let p = |name: &str| format!("{mode}-{name}");
        let mut push = |key: &str, seen: Seen| out.push((fe, key.to_string(), seen));

        let r = render(&p("instruction"), mode, &["--instruction", "slowly"]);
        push("render.instruction", by_engine_refusal(&r, "instruction"));
        let r = render(&p("steer"), mode, &["--steer", "none.safetensors"]);
        push("render.steer", by_engine_refusal(&r, "steer"));
        let r = render(&p("out"), mode, &[]);
        push("output.out", by_file(&r, "o.flac", "--out"));
        push(
            if mode == "story" { "render.story" } else { "render.text" },
            by_file(&r, "o.flac", "--out"),
        );
        let r = render(&p("trace"), mode, &["--trace", "t.jsonl"]);
        push("observe.trace", by_file(&r, "t.jsonl", "--trace"));
        let r = render(&p("level"), mode, &["--trace", "t.jsonl", "--trace-level", "2"]);
        let header = std::fs::read_to_string(r.dir.join("t.jsonl")).unwrap_or_default();
        push(
            "observe.trace_level",
            if header.lines().next().is_some_and(|l| l.contains("\"trace_level\":2")) {
                Seen::Carried
            } else {
                Seen::Dropped
            },
        );
        let r = render(&p("timing"), mode, &["--timing-report", "tr.json"]);
        push("observe.timing_report", by_file(&r, "tr.json", "--timing-report"));
        let r = render(&p("duration"), mode, &["--duration-report", "dr.json"]);
        push("observe.duration_report", by_file(&r, "dr.json", "--duration-report"));
        let r = render(&p("stream"), mode, &["--stream", "st.wav"]);
        push("output.stream", by_file(&r, "st.wav", "--stream"));

        // The seed: two renders with the trace, at two seeds. Carried when the chunks' recorded seeds differ.
        let seeds = |n: &str| {
            let r = render(&p(&format!("seed{n}")), mode, &["--seed", n, "--trace", "t.jsonl"]);
            assert!(r.ok, "{}", r.stderr);
            trace_seeds(&r.dir.join("t.jsonl"))
        };
        if mode == "story" {
            let (a, b) = (seeds("1"), seeds("2"));
            assert!(!a.is_empty(), "the story trace records chunk seeds");
            push("render.seed", if a != b { Seen::Carried } else { Seen::Dropped });
        }

        // The options ggtts-render does not take: refused by name.
        for d in TABLE.iter().filter(|d| BOOK.contains(&d.key().as_str())) {
            let flag = d.flags[0];
            let mut extra = vec![flag];
            extra.extend(sample_value(d));
            let r = render(&p(d.name), mode, &extra);
            push(&d.key(), by_front_end_refusal(&r, flag));
        }
        let other = if mode == "story" {
            ["--text", "你好。"]
        } else {
            ["--story", "s.txt"]
        };
        let r = render(&p("other"), mode, &other);
        push(
            if mode == "story" { "render.text" } else { "render.story" },
            by_front_end_refusal(&r, "--text"),
        );
    }
    out
}

/// `ggtts-book render`'s options parser: every option with a flag that the book does not take is refused by name.
fn book_probes() -> Vec<Probe> {
    let mut out = Vec::new();
    for d in TABLE.iter().filter(|d| !d.flags.is_empty()) {
        let key = d.key();
        if ggtts_book::options::RENDER_TAKES.contains(&key.as_str()) {
            continue;
        }
        let mut args = vec!["b.epub".to_string()];
        // An option that needs another (`--stream-format` needs `--stream`) is given with it; the book may then
        // refuse the one it needs first, which refuses both.
        let parent = d
            .requires
            .and_then(|name| TABLE.iter().find(|p| p.name == name && p.section == d.section));
        for o in parent.into_iter().chain([d]) {
            args.push(o.flags[0].to_string());
            args.extend(sample_value(o).map(str::to_string));
        }
        let named = |e: &str| e.contains(d.flags[0]) || parent.is_some_and(|p| e.contains(p.flags[0]));
        let seen = match ggtts_book::options::parse("render", &args, &|_| None) {
            Err(e) if named(&e) => Seen::Refused,
            Err(e) => panic!("book: {key} refused without naming {}: {e}", d.flags[0]),
            Ok(_) => Seen::Dropped,
        };
        out.push(("book", key, seen));
    }
    out
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(ggtts_ffi::ggtts_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The C ABI on the fake engine: a style instruction reaches the engine's refusal; a text comes back as samples.
fn c_abi_probes() -> Vec<Probe> {
    use ggtts_ffi::{ggtts_free, ggtts_load_with, ggtts_pcm_free, ggtts_render, GgttsSettings, GGTTS_OK};
    let engine = CString::new("fake").unwrap();
    let load = |instruction: Option<&CString>| {
        let settings = GgttsSettings {
            size: std::mem::size_of::<GgttsSettings>(),
            speaker: std::ptr::null(),
            instruction: instruction.map_or(std::ptr::null(), |s| s.as_ptr()),
            device: std::ptr::null(),
            cache_dir: std::ptr::null(),
            allow_download: 0,
        };
        let mut h = 0u64;
        let null = std::ptr::null();
        let code = unsafe { ggtts_load_with(engine.as_ptr(), null, null, null, 0, &settings, &mut h) };
        (code, h)
    };
    let mut out = Vec::new();
    let slowly = CString::new("slowly").unwrap();
    let (code, h) = load(Some(&slowly));
    out.push((
        "c-abi",
        "render.instruction".to_string(),
        if code == GGTTS_OK {
            ggtts_free(h);
            Seen::Dropped
        } else if last_error().contains("instruction") {
            Seen::Carried
        } else {
            panic!("c-abi: the load failed: {}", last_error())
        },
    ));
    let (code, h) = load(None);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let text = CString::new("你好。").unwrap();
    let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
    let code = unsafe { ggtts_render(h, text.as_ptr(), 7, &mut p, &mut n, &mut r) };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    if !p.is_null() {
        unsafe { ggtts_pcm_free(p, n) };
    }
    ggtts_free(h);
    out.push((
        "c-abi",
        "render.text".to_string(),
        if n > 0 { Seen::Carried } else { Seen::Dropped },
    ));
    out
}

fn probes() -> Vec<Probe> {
    let mut out = render_probes();
    out.extend(book_probes());
    out.extend(c_abi_probes());
    out
}

fn report(line: &str) {
    let _ = writeln!(std::io::stderr(), "conformance: {line}");
}

/// AC1: every option has a cell on every front end, the baseline is exactly the table's gaps, and every probe sees
/// what its cell says. Prints the gap count and the declared-not-exercised count.
#[test]
fn every_front_end_carries_refuses_or_lists_every_option() {
    let cells = cells(TABLE, FRONT_ENDS).unwrap_or_else(|e| panic!("the conformance table:\n{}", e.join("\n")));
    let baseline = read(GAPS);
    let gap_errors = check_gaps(&cells, &baseline);
    let mut errors = gap_errors.clone();
    let mut exercised = BTreeSet::new();
    for (fe, key, seen) in probes() {
        let cell = cells[&(fe.to_string(), key.clone())];
        if let Err(e) = verdict(fe, &key, cell, seen) {
            errors.push(e);
        }
        exercised.insert((fe.to_string(), key));
    }
    let gaps = cells.values().filter(|c| matches!(c, Gap(_))).count();
    report(&format!(
        "{} front ends x {} options = {} cells: {} carried, {} refused, {} gaps",
        FRONT_ENDS.len(),
        TABLE.len(),
        cells.len(),
        cells.values().filter(|c| **c == Carried).count(),
        cells.values().filter(|c| matches!(c, Refused(_))).count(),
        gaps
    ));
    report(&format!(
        "exercised: {}; declared, not exercised: {}",
        exercised.len(),
        cells.len() - exercised.len()
    ));
    for fe in FRONT_ENDS {
        let of = |f: &dyn Fn(&Cell) -> bool| cells.iter().filter(|((x, _), c)| x == fe.id && f(c)).count();
        let ex = exercised.iter().filter(|(x, _)| x == fe.id).count();
        report(&format!(
            "  {:<13} gaps {:>2}  exercised {:>2}  declared {:>2}",
            fe.id,
            of(&|c| matches!(c, Gap(_))),
            ex,
            TABLE.len() - ex
        ));
    }
    if !gap_errors.is_empty() {
        let mut want: Vec<String> = cells
            .iter()
            .filter_map(|((fe, key), c)| match c {
                Gap(i) => Some(gap_line(fe, key, *i)),
                _ => None,
            })
            .collect();
        want.sort();
        errors.push(format!("the table's gaps, one per line:\n{}", want.join("\n")));
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// The book's refusal reasons in the table are the ones `ggtts-book` gives.
#[test]
fn book_refusals_are_the_books_own() {
    let cells = cells(TABLE, FRONT_ENDS).unwrap();
    for ((fe, key), cell) in &cells {
        if let (true, Refused(why)) = (fe == "book", cell) {
            assert_eq!(*why, ggtts_book::options::refusal(key), "book {key}");
        }
    }
}

// ── plants: each check shown red ───────────────────────────────────────────────────────────────────────────────────

/// AC2: an option the front ends do not list fails, naming it.
#[test]
fn a_new_option_with_no_cell_fails_naming_it() {
    let mut planted: Vec<Descriptor> = TABLE.to_vec();
    planted.push(Descriptor {
        name: "planted_option",
        ..TABLE[0]
    });
    let errors = cells(&planted, FRONT_ENDS).unwrap_err();
    assert_eq!(errors.len(), FRONT_ENDS.len(), "{errors:?}");
    assert!(
        errors.iter().all(|e| e.contains("load.planted_option has no cell")),
        "{errors:?}"
    );
}

/// AC2: an option entered as a gap fails: its line is not in the baseline (and `--base` refuses adding it there).
#[test]
fn a_new_option_entered_as_a_gap_fails_naming_it() {
    let mut cells = cells(TABLE, FRONT_ENDS).unwrap();
    cells.insert(("render-story".into(), "load.planted_option".into()), Gap(283));
    let errors = check_gaps(&cells, &read(GAPS));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("load.planted_option is a gap not in"), "{errors:?}");
}

/// AC4: a gap closed in the table while its baseline line stays fails, naming the line.
#[test]
fn a_closed_gap_still_listed_fails() {
    let mut cells = cells(TABLE, FRONT_ENDS).unwrap();
    let key = ("c-abi".to_string(), "observe.trace".to_string());
    assert_eq!(cells[&key], Gap(280));
    cells.insert(key, Carried);
    let errors = check_gaps(&cells, &read(GAPS));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("c-abi\\tobserve.trace\\t#280"), "{errors:?}");
}

/// AC3: a front end that drops an option its cell says it carries fails, and a gap a probe sees carried is closed.
#[test]
fn a_dropped_carried_option_fails() {
    let e = verdict("render-story", "render.instruction", Carried, Seen::Dropped).unwrap_err();
    assert!(
        e.contains("render.instruction is Carried") && e.contains("Dropped"),
        "{e}"
    );
    let e = verdict("render-text", "observe.trace", Gap(294), Seen::Carried).unwrap_err();
    assert!(e.contains("gap #294 is closed"), "{e}");
    assert!(verdict("book", "observe.report", Gap(295), Seen::Refused).is_ok());
}

// ── binding surface snapshots ──────────────────────────────────────────────────────────────────────────────────────

/// The difference between a snapshot and the surfaces, or nothing: every line added or removed, and each name a line
/// adds that no row of the options table spells.
fn snapshot_diff(snapshot: &str, now: &[Entry]) -> Vec<String> {
    let old: Vec<&str> = entries(snapshot);
    let new: Vec<String> = now.iter().map(Entry::line).collect();
    let mut out = Vec::new();
    for e in now {
        let line = e.line();
        if old.contains(&line.as_str()) {
            continue;
        }
        out.push(format!("+ {line}"));
        let before = old
            .iter()
            .find(|l| l.split(':').next() == line.split(':').next())
            .map(|l| l.split(": ").nth(1).unwrap_or("").split(", ").collect::<Vec<_>>())
            .unwrap_or_default();
        for name in e.names.iter().filter(|n| !before.contains(&n.as_str())) {
            if !known(TABLE, e.surface, name) {
                out.push(format!(
                    "  {name}: no row of the options table spells it, so no front end's cell covers it"
                ));
            }
        }
    }
    for line in &old {
        if !new.iter().any(|n| n == line) {
            out.push(format!("- {line}"));
        }
    }
    out
}

#[test]
fn the_binding_surfaces_match_their_snapshot() {
    let now = surfaces::binding_entries();
    assert!(now.len() >= 15, "the readers found the surfaces: {now:?}");
    let diff = snapshot_diff(&read(SNAPSHOT), &now);
    let lines: Vec<String> = now.iter().map(Entry::line).collect();
    assert!(
        diff.is_empty(),
        "a binding surface changed: give each new name a row in the options table and a cell per front end, then \
         update {SNAPSHOT}\n{}\nthe surfaces now:\n{}",
        diff.join("\n"),
        lines.join("\n")
    );
}

/// AC5: a Python keyword `foo=None` added to `load` fails the snapshot, naming it as no table row's.
#[test]
fn a_new_python_keyword_fails_the_snapshot() {
    let src = read("crates/ggtts-py/src/python.rs");
    let planted = src.replacen(
        "#[pyo3(signature = (engine,",
        "#[pyo3(signature = (engine, foo=None,",
        1,
    );
    assert_ne!(planted, src, "the plant applies");
    let mut now = surfaces::pyo3_entries(&planted);
    now.extend(surfaces::header_entries(&read("crates/ggtts-ffi/include/ggtts.h")));
    now.extend(surfaces::kotlin_entries(&read(
        "android/ggtts/src/main/kotlin/ggtts/Ggtts.kt",
    )));
    let diff = snapshot_diff(&read(SNAPSHOT), &now);
    assert!(
        diff.iter().any(|l| l.starts_with("+ python load: engine, foo,")),
        "{diff:?}"
    );
    assert!(
        diff.iter().any(|l| l.contains("foo: no row of the options table")),
        "{diff:?}"
    );
    assert!(known(TABLE, Surface::Python, "model"));
}

// ── the render path ────────────────────────────────────────────────────────────────────────────────────────────────

/// The `Engine` trait's render methods and how many arguments each takes.
const ENGINE_METHODS: [(&str, usize); 4] = [
    ("generate", 1),
    ("generate_traced", 2),
    ("generate_hooked", 3),
    ("decode", 1),
];

/// `src` with comments, string literals and character literals blanked (newlines kept), so a call in a comment or a
/// string is not a call.
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let blank = |out: &mut Vec<u8>, c: u8| out.push(if c == b'\n' { b'\n' } else { b' ' });
    let mut i = 0;
    while i < b.len() {
        let rest = &b[i..];
        if rest.starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                blank(&mut out, b[i]);
                i += 1;
            }
        } else if rest.starts_with(b"/*") {
            let mut depth = 0;
            while i < b.len() {
                if b[i..].starts_with(b"/*") {
                    depth += 1;
                    out.extend(b"  ");
                    i += 2;
                } else if b[i..].starts_with(b"*/") {
                    depth -= 1;
                    out.extend(b"  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    blank(&mut out, b[i]);
                    i += 1;
                }
            }
        } else if (rest.starts_with(b"r\"")
            || rest.starts_with(b"r#")
            || rest.starts_with(b"br\"")
            || rest.starts_with(b"br#"))
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'))
        {
            let start = i + if b[i] == b'b' { 2 } else { 1 };
            let hashes = b[start..].iter().take_while(|&&c| c == b'#').count();
            if b.get(start + hashes) != Some(&b'"') {
                out.push(b[i]);
                i += 1;
                continue;
            }
            let mut close = vec![b'"'];
            close.extend(std::iter::repeat_n(b'#', hashes));
            let body = start + hashes + 1;
            let end = b[body..]
                .windows(close.len())
                .position(|w| w == close.as_slice())
                .map_or(b.len(), |p| body + p + close.len());
            for &c in &b[i..end] {
                blank(&mut out, c);
            }
            i = end;
        } else if b[i] == b'"' {
            blank(&mut out, b[i]);
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    blank(&mut out, b[i]);
                    i += 1;
                }
                blank(&mut out, b[i]);
                i += 1;
            }
            if i < b.len() {
                blank(&mut out, b[i]);
                i += 1;
            }
        } else if b[i] == b'\'' {
            // A character literal ('x', '\n', '\u{..}') is blanked; a lifetime ('a) is kept.
            let len = if b.get(i + 1) == Some(&b'\\') {
                b[i + 3..].iter().position(|&c| c == b'\'').map(|p| p + 4)
            } else {
                let ch = std::str::from_utf8(&b[i + 1..]).ok().and_then(|s| s.chars().next());
                ch.and_then(|c| (b.get(i + 1 + c.len_utf8()) == Some(&b'\'')).then_some(c.len_utf8() + 2))
            };
            let n = len.unwrap_or(1);
            for &c in &b[i..i + n] {
                blank(&mut out, c);
            }
            i += n;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("blanking keeps UTF-8")
}

/// The index just past the brace block that opens at or after `from` (the item's body), or past the `;` that ends it
/// first (an item with no body).
fn item_end(code: &str, from: usize) -> usize {
    let b = code.as_bytes();
    let mut i = from;
    while i < b.len() && b[i] != b'{' && b[i] != b';' {
        i += 1;
    }
    if i >= b.len() || b[i] == b';' {
        return (i + 1).min(b.len());
    }
    let mut depth = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// The byte ranges of `code` that are not a render by rule: `#[cfg(test)]` items, `impl ... Engine for ...` blocks (an
/// engine delegating to the one it wraps) and the `Engine` trait's own definition (its default methods).
fn excluded(code: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, _) in code.match_indices("#[cfg(test)]") {
        out.push((i, item_end(code, i + "#[cfg(test)]".len())));
    }
    for (keyword, is_engine) in [
        (
            "impl",
            (|w: &[&str]| {
                w.windows(2)
                    .any(|p| (p[0] == "Engine" || p[0].ends_with("::Engine")) && p[1] == "for")
            }) as fn(&[&str]) -> bool,
        ),
        ("trait", |w: &[&str]| {
            w.get(1)
                .is_some_and(|n| *n == "Engine" || n.starts_with("Engine<") || n.starts_with("Engine:"))
        }),
    ] {
        for (i, _) in code.match_indices(keyword) {
            let bounded = (i == 0 || !code.as_bytes()[i - 1].is_ascii_alphanumeric() && code.as_bytes()[i - 1] != b'_')
                && code[i + keyword.len()..].starts_with(|c: char| c.is_whitespace() || c == '<');
            if !bounded {
                continue;
            }
            let head_end = code[i..].find(['{', ';']).map_or(code.len(), |j| i + j);
            let words: Vec<&str> = code[i..head_end].split_whitespace().collect();
            if is_engine(&words) {
                out.push((i, item_end(code, i)));
            }
        }
    }
    out
}

/// The calls of an `Engine` render method in `src`: `.generate(` with one argument, `.generate_traced(` with two,
/// `.generate_hooked(` with three, `.decode(` with one, outside comments, strings, `#[cfg(test)]` items, delegating
/// `impl Engine for` blocks and the trait's definition. The receiver's type is not read: the argument count tells an
/// engine's `decode(&codes)` from a vocoder's three-argument `decode`, and bindgen's `.generate()` has none. Each:
/// (line number, the line).
fn engine_calls(src: &str) -> Vec<(usize, String)> {
    let code = code_only(src);
    let skip = excluded(&code);
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (name, arity) in ENGINE_METHODS {
        let pattern = format!(".{name}(");
        for (i, _) in code.match_indices(&pattern) {
            if skip.iter().any(|(a, b)| (*a..*b).contains(&i)) {
                continue;
            }
            let open = i + pattern.len();
            let mut depth = 1;
            let mut j = open;
            let b = code.as_bytes();
            while j < b.len() && depth > 0 {
                match b[j] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let args = surfaces::split_params(&code[open..j.saturating_sub(1)]).len();
            if args == arity {
                let line = code[..i].matches('\n').count();
                out.push((line + 1, lines.get(line).unwrap_or(&"").trim().to_string()));
            }
        }
    }
    out.sort();
    out
}

/// Whether `path` (relative to the root, `/`-separated) is out of the scan by rule: build scripts, examples, tests,
/// the shared pipeline itself, and the engine crates' library code (their binaries are in).
fn out_of_scope(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    path.ends_with("/build.rs")
        || parts
            .iter()
            .any(|p| ["examples", "tests", "target", "spike"].contains(p))
        || path == "crates/ggtts-core/src/pipeline.rs"
        || [
            "crates/ggtts-qwen3-tts/src/",
            "crates/ggtts-kitten/src/",
            "crates/ggtts-fake/src/",
        ]
        .iter()
        .any(|c| path.starts_with(c) && !path[c.len()..].starts_with("bin/"))
}

/// The crate directory a source file belongs to (the nearest with a `Cargo.toml`), and whether that crate uses the
/// engine interface (it is `ggtts-core`, or depends on it).
fn uses_engine(path: &Path) -> bool {
    let mut dir = path.parent();
    while let Some(d) = dir {
        if let Ok(manifest) = std::fs::read_to_string(d.join("Cargo.toml")) {
            return manifest.contains("name = \"ggtts-core\"") || manifest.contains("ggtts-core");
        }
        dir = d.parent();
    }
    false
}

fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        let name = e.file_name();
        if p.is_dir() {
            if name != "target" && name != ".git" && name != "ggml" && name != "node_modules" {
                rust_files(&p, found);
            }
        } else if p.extension().is_some_and(|x| x == "rs") {
            found.push(p);
        }
    }
}

/// Every source file in the scan, relative to the root, with its engine calls.
fn scan() -> (usize, BTreeMap<String, Vec<(usize, String)>>) {
    let root = root().canonicalize().unwrap();
    let mut files = Vec::new();
    for top in ["crates", "instruments"] {
        rust_files(&root.join(top), &mut files);
    }
    let mut scanned = 0;
    let mut calls = BTreeMap::new();
    for f in files {
        let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        if out_of_scope(&rel) || !uses_engine(&f) {
            continue;
        }
        scanned += 1;
        let found = engine_calls(&std::fs::read_to_string(&f).unwrap().replace("\r\n", "\n"));
        if !found.is_empty() {
            calls.insert(rel, found);
        }
    }
    (scanned, calls)
}

/// The render-path check: a file with engine calls must be listed; a listed file must still have one.
fn check_render_path(calls: &BTreeMap<String, Vec<(usize, String)>>, exceptions: &str) -> Vec<String> {
    let listed: BTreeMap<&str, &str> = entries(exceptions)
        .into_iter()
        .map(|l| l.split_once('\t').unwrap_or((l, "")))
        .collect();
    let mut errors = Vec::new();
    for (file, found) in calls {
        if !listed.contains_key(file.as_str()) {
            let lines: Vec<String> = found.iter().map(|(n, l)| format!("  {file}:{n}: {l}")).collect();
            errors.push(format!(
                "{file} calls an engine directly: render through ggtts_core::pipeline::render_plan or render_chunk\n{}",
                lines.join("\n")
            ));
        }
    }
    for (file, why) in &listed {
        if why.trim().is_empty() {
            errors.push(format!("{EXCEPTIONS}: {file} names no issue or reason"));
        }
        if !calls.contains_key(*file) {
            errors.push(format!(
                "{EXCEPTIONS} lists {file}, which no longer calls an engine directly: remove the line"
            ));
        }
    }
    errors
}

/// AC8: the render path on this tree, with exactly the listed exceptions; prints what it scanned and matched.
#[test]
fn only_the_listed_files_call_an_engine_directly() {
    let (scanned, calls) = scan();
    assert!(scanned > 20, "the scan found the sources: {scanned}");
    report(&format!(
        "render path: scanned {scanned} source files; engine calls in {}:",
        calls.len()
    ));
    for (file, found) in &calls {
        for (n, l) in found {
            report(&format!("  {file}:{n}: {l}"));
        }
    }
    let errors = check_render_path(&calls, &read(EXCEPTIONS));
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// AC6: a direct engine call in a new front-end file fails; the delegating and out-of-scope shapes do not count.
#[test]
fn a_planted_engine_call_fails_the_render_path() {
    let planted =
        "fn speak(engine: &mut dyn Engine, req: &ChunkRequest) {\n    let g = engine.generate(&req).unwrap();\n    \
                   let pcm = engine\n        .decode(&g.codes)\n        .unwrap();\n}\n";
    assert_eq!(
        engine_calls(planted).iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        vec![2, 4]
    );
    let (_, mut calls) = scan();
    calls.insert("crates/ggtts-cli/src/bin/planted.rs".into(), engine_calls(planted));
    let errors = check_render_path(&calls, &read(EXCEPTIONS));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].starts_with("crates/ggtts-cli/src/bin/planted.rs calls an engine directly"),
        "{errors:?}"
    );

    let not_renders = "// engine.generate(&req) in a comment\nconst S: &str = \"engine.generate(&req)\";\n\
                       impl ggtts_core::engine::Engine for Wrap {\n    fn generate(&mut self, r: &ChunkRequest<'_>) -> R {\n        \
                       self.inner.generate(r)\n    }\n}\n#[cfg(test)]\nmod tests {\n    fn t() { e.generate(&r); }\n}\n\
                       fn vocode() { hift.decode(&mel, frames, &spec); bindings.generate(); }\n\
                       pub trait Engine {\n    fn generate_traced(&mut self, r: &R, e: &mut E) -> G {\n        self.generate(r)\n    }\n}\n";
    assert_eq!(engine_calls(not_renders), vec![]);
    assert!(out_of_scope("crates/ggtts-ggml-sys/build.rs"));
    assert!(out_of_scope("crates/ggtts-qwen3-tts/examples/generate.rs"));
    assert!(out_of_scope("crates/ggtts-qwen3-tts/src/talker.rs"));
    assert!(!out_of_scope("crates/ggtts-kitten/src/bin/ggtts-kitten.rs"));
}

/// An exception whose file no longer calls an engine fails: the list shrinks with the code.
#[test]
fn a_stale_exception_fails() {
    let (_, calls) = scan();
    let planted = format!("{}crates/ggtts-cli/src/bin/ggtts-render.rs\t#278\n", read(EXCEPTIONS));
    let errors = check_render_path(&calls, &planted);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("lists crates/ggtts-cli/src/bin/ggtts-render.rs, which no longer"),
        "{errors:?}"
    );
}
