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
use std::sync::atomic::{AtomicU64, Ordering};

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
const KITTEN_HEAD: &[&str] = &["load.kitten_full_head"];
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
const NO_TIMINGS: &str = "a binding returns the samples, not timings or reports";
const NO_LOG: &str = "a binding writes no log: its errors are its codes and messages";
const ANDROID_TRACE: &str = "the trace is a desktop observer: refused on Android";
const CPU_ONLY: &str = "the Android build is CPU only";
const ENGINE_READS_ENV: &str =
    "the kitten-tts-2 engine reads GGTTS_KITTEN_FULL_HEAD=1 from the environment itself: a binding takes no option for it";
const NEVER_DOWNLOADS: &str = "the app never downloads: its models are installed with it";
const PARITY_TOOL: &str = "an engine parity tool; render with ggtts-render --engine kitten-tts-2";
const TEXT_HOLD: &str =
    "--text renders one chunk, and the stream levels it by its own gain whatever the hold; give it with --story";
const TEXT_JUDGE: &str = "--text renders one draw and keeps it; the judge redraws a story's chunks (--story)";
const TEXT_REDRAW: &str = "--text renders one draw and keeps it; redraws are a story's (--story)";
const NO_JUDGE: &str =
    "the ASR judge starts a Python process beside the render: judge with ggtts-render --story or ggtts-book";
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
            (KITTEN_HEAD, Carried),
            (
                &[
                    "render.story",
                    "render.instruction",
                    "render.redraw",
                    "render.judge",
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
            (KITTEN_HEAD, Carried),
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
            (&["render.judge"], Refused(TEXT_JUDGE)),
            (&["render.redraw"], Refused(TEXT_REDRAW)),
            (&["render.edge_s"], Gap(180)),
            (TRACE, Carried),
            (REPORTS, Carried),
            (LOG, Carried),
            (INSTRUMENTS, Carried),
            (&["output.out", "output.stream", "output.stream_format"], Carried),
            (&["output.stream_raw"], Carried),
            (&["output.stream_level_seconds"], Refused(TEXT_HOLD)),
            (BOOK, Refused(NOT_A_BOOK)),
        ],
    },
    FrontEnd {
        id: "book",
        cells: &[
            (LOAD, Carried),
            (KITTEN_HEAD, Carried),
            (&["render.instruction", "render.redraw", "render.judge"], Carried),
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
            (KITTEN_HEAD, Refused(ENGINE_READS_ENV)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (&["render.judge"], Refused(NO_JUDGE)),
            (STEER, Carried),
            (&["render.redraw", "render.edge_s"], Carried),
            (&["render.measure_timings"], Refused(NO_TIMINGS)),
            (TRACE, Carried),
            (REPORTS, Refused(NO_TIMINGS)),
            (LOG, Refused(NO_LOG)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (&["output.out", "output.stream_format"], Refused(NO_FILE)),
            (&["output.stream"], Carried),
            (STREAM_LEVEL, Carried),
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
            (&["load.cache_dir"], Carried),
            (KITTEN_HEAD, Refused(ENGINE_READS_ENV)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (&["render.judge"], Refused(NO_JUDGE)),
            (STEER, Carried),
            (&["render.edge_s", "render.redraw"], Carried),
            (&["render.measure_timings"], Refused(NO_TIMINGS)),
            (TRACE, Carried),
            (REPORTS, Refused(NO_TIMINGS)),
            (LOG, Refused(NO_LOG)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (&["output.out", "output.stream_format"], Refused(NO_FILE)),
            (&["output.stream"], Carried),
            (STREAM_LEVEL, Carried),
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
            (KITTEN_HEAD, Refused(ENGINE_READS_ENV)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (&["render.judge"], Refused(NO_JUDGE)),
            (&["render.steer"], Carried),
            // Probed on `Recorder`, an engine that steers and redraws, loaded through `kotlin::load_over`.
            (&["render.steer_scale", "render.steer_layers"], Carried),
            (&["render.edge_s"], Carried),
            (&["render.redraw"], Carried),
            (&["render.measure_timings"], Refused(NO_TIMINGS)),
            (TRACE, Refused(ANDROID_TRACE)),
            (REPORTS, Refused(NO_TIMINGS)),
            (LOG, Refused(NO_LOG)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (&["output.out", "output.stream_format"], Refused(NO_FILE)),
            (&["output.stream"], Carried),
            (STREAM_LEVEL, Carried),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "android",
        // The app's request, its saved voices and its render calls hold the options document and hand it to the
        // Kotlin binding unread (android/demo/.../TtsBinding.kt, GgttsBinding), so it carries and refuses what the
        // binding does, by the binding's own refusal; `android_probes` checks the hand-over and takes the Kotlin
        // probes' verdicts. The app's own choices: it never downloads, and it streams raw unless the document says.
        cells: &[
            (
                &[
                    "load.engine",
                    "load.model",
                    "load.codec",
                    "load.voice",
                    "load.speaker",
                    "load.threads",
                    "load.cache_dir",
                ],
                Carried,
            ),
            (&["load.device"], Refused(CPU_ONLY)),
            (&["load.offline"], Refused(NEVER_DOWNLOADS)),
            (KITTEN_HEAD, Refused(ENGINE_READS_ENV)),
            (&["render.text", "render.seed_salt", "render.instruction"], Carried),
            (&["render.story"], Refused(NO_STORY_FILE)),
            (&["render.seed"], Refused(BINDING_SEED)),
            (&["render.judge"], Refused(NO_JUDGE)),
            (&["render.steer", "render.edge_s"], Carried),
            // Handed to the binding in the document like the rest, so the app carries them exactly when the Kotlin
            // binding does (its probes on `Recorder`). The app cannot carry more than the binding:
            // `the_app_carries_no_more_than_its_binding` holds the two rows together.
            (&["render.steer_scale", "render.steer_layers", "render.redraw"], Carried),
            (&["render.measure_timings"], Refused(NO_TIMINGS)),
            (TRACE, Refused(ANDROID_TRACE)),
            (REPORTS, Refused(NO_TIMINGS)),
            (LOG, Refused(NO_LOG)),
            (INSTRUMENTS, Refused(NO_INSTRUMENTS)),
            (&["output.out", "output.stream_format"], Refused(NO_FILE)),
            (&["output.stream"], Carried),
            (STREAM_LEVEL, Carried),
            (BOOK, Refused(NOT_BOOK_FRONT)),
        ],
    },
    FrontEnd {
        id: "kitten",
        cells: &[
            (LOAD, Refused(PARITY_TOOL)),
            (KITTEN_HEAD, Refused(PARITY_TOOL)),
            (
                &[
                    "render.text",
                    "render.story",
                    "render.seed",
                    "render.seed_salt",
                    "render.instruction",
                    "render.redraw",
                    "render.judge",
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

// ── introspection ──────────────────────────────────────────────────────────────────────────────────────────────

/// What the C ABI states about itself since version 5: the options schema (`ggtts_options_schema`), a loaded engine's
/// capability statement (`ggtts_capabilities`) and the key of a refused option (`ggtts_last_error_key`).
const INTROSPECTION: [&str; 3] = ["options_schema", "capabilities", "refused_key"];

const CLI_SCHEMA: &str = "a command-line tool takes flags, which its --help lists; the schema to read is the library's";
const CLI_CAPABILITIES: &str =
    "a command-line tool refuses an option its engine lacks before rendering; the statement to read is the library's";
const CLI_KEY: &str = "a command-line tool prints the refusal and exits; there is no error value to carry a key";
const APP: &str = "the app is not a programming interface: it sets its options on its own screens";

/// Each front end's cell for the three, in [`INTROSPECTION`]'s order. Carried or refused, never a gap.
const INTROSPECTION_CELLS: &[(&str, [Cell; 3])] = &[
    (
        "render-story",
        [Refused(CLI_SCHEMA), Refused(CLI_CAPABILITIES), Refused(CLI_KEY)],
    ),
    (
        "render-text",
        [Refused(CLI_SCHEMA), Refused(CLI_CAPABILITIES), Refused(CLI_KEY)],
    ),
    (
        "book",
        [Refused(CLI_SCHEMA), Refused(CLI_CAPABILITIES), Refused(CLI_KEY)],
    ),
    ("c-abi", [Carried, Carried, Carried]),
    ("python", [Carried, Carried, Carried]),
    ("kotlin", [Carried, Carried, Carried]),
    ("android", [Refused(APP), Refused(APP), Refused(APP)]),
    (
        "kitten",
        [Refused(PARITY_TOOL), Refused(PARITY_TOOL), Refused(PARITY_TOOL)],
    ),
];

/// What each binding states, beside the C ABI's own: the schema, the fake engine's statement and the key of a
/// refused option, through the binding's own core. A binding carries one only when it equals the C ABI's.
fn introspection_probes() -> Vec<(&'static str, &'static str, bool)> {
    use ggtts_ffi::{ggtts_capabilities, ggtts_load, ggtts_options_schema, ggtts_string_free};
    let value = |t: &str| -> serde_json::Value { serde_json::from_str(t).unwrap() };
    let c_schema = value(unsafe { CStr::from_ptr(ggtts_options_schema()) }.to_str().unwrap());
    let c_statement = {
        let e = CString::new("fake").unwrap();
        let (mut h, null) = (0u64, std::ptr::null());
        assert_eq!(unsafe { ggtts_load(e.as_ptr(), null, null, null, 0, &mut h) }, 0);
        let _freed = Freed(h);
        let mut json = std::ptr::null_mut();
        assert_eq!(unsafe { ggtts_capabilities(h, &mut json) }, 0);
        let text = unsafe { CStr::from_ptr(json) }.to_str().unwrap().to_string();
        unsafe { ggtts_string_free(json) };
        value(&text)
    };
    let steer = r#"{"render": {"steer": "s.safetensors"}}"#;
    let mut out = Vec::new();
    // The C ABI: its schema is the table's, its statement the shared one's, and a refusal leaves its key.
    out.push(("c-abi", "options_schema", c_schema == ggtts_core::options::schema()));
    out.push((
        "c-abi",
        "capabilities",
        c_statement["format"] == ggtts_core::capabilities::STATEMENT_FORMAT,
    ));
    let doc = CString::new(r#"{"load": {"engine": "fake"}}"#).unwrap();
    let mut h = 0u64;
    assert_eq!(unsafe { ggtts_ffi::ggtts_load_opts(doc.as_ptr(), &mut h) }, 0);
    let freed = Freed(h);
    let (text, steer_c) = (CString::new("你好。").unwrap(), CString::new(steer).unwrap());
    let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
    let code = unsafe {
        ggtts_ffi::ggtts_render_opts(
            h,
            text.as_ptr(),
            steer_c.as_ptr(),
            None,
            std::ptr::null_mut(),
            &mut p,
            &mut n,
            &mut r,
        )
    };
    let key = ggtts_ffi::ggtts_last_error_key();
    let c_key = (!key.is_null()).then(|| unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned());
    out.push((
        "c-abi",
        "refused_key",
        code != 0 && c_key.as_deref() == Some("render.steer"),
    ));
    drop(freed);
    // Python, through its core.
    {
        use ggtts_py::core::{guard_keyed, load, LoadRequest};
        out.push((
            "python",
            "options_schema",
            value(&ggtts_py::core::options_schema()) == c_schema,
        ));
        let req = LoadRequest {
            engine: Some("fake".into()),
            ..Default::default()
        };
        let (handle, _) = load(&req, &mut |_| {}).unwrap();
        out.push((
            "python",
            "capabilities",
            value(&handle.capabilities().unwrap()) == c_statement,
        ));
        let refused = guard_keyed(|| handle.render_with("你好。", None, Some(steer)));
        out.push((
            "python",
            "refused_key",
            matches!(refused, Err((_, Some(k))) if k == "render.steer"),
        ));
    }
    // Kotlin, through the core under its JNI layer.
    {
        use ggtts_ffi::kotlin;
        out.push(("kotlin", "options_schema", value(&kotlin::options_schema()) == c_schema));
        let h = kotlin::load(r#"{"load": {"engine": "fake"}}"#, &kotlin::LoadArgs::default()).unwrap();
        let _freed = Freed(h);
        out.push((
            "kotlin",
            "capabilities",
            value(&kotlin::capabilities(h).unwrap()) == c_statement,
        ));
        let refused = ggtts_py::core::guard_keyed(|| kotlin::render_opts(h, "你好。", Some(steer), None));
        out.push((
            "kotlin",
            "refused_key",
            matches!(refused, Err((_, Some(k))) if k == "render.steer"),
        ));
    }
    out
}

/// #369 AC4: every front end has a cell for each of the three, carried or refused with a reason, and every carried
/// cell is probed and seen: the binding states what the C ABI states.
#[test]
fn every_front_end_carries_or_refuses_the_introspection() {
    let ids: BTreeSet<&str> = FRONT_ENDS.iter().map(|f| f.id).collect();
    let rows: BTreeSet<&str> = INTROSPECTION_CELLS.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, rows, "every front end has one introspection row, and only they do");
    let probes = introspection_probes();
    for (id, cells) in INTROSPECTION_CELLS {
        for (item, cell) in INTROSPECTION.iter().zip(cells) {
            match cell {
                Carried => {
                    let seen = probes.iter().find(|(f, i, _)| f == id && i == item);
                    assert!(
                        matches!(seen, Some((_, _, true))),
                        "{id}: {item} is Carried and the probe saw {seen:?}"
                    );
                }
                Refused(why) => assert!(!why.is_empty(), "{id}: {item} is refused with no reason"),
                Gap(i) => panic!("{id}: {item} is a gap (#{i}): the introspection is carried or refused"),
            }
        }
    }
    report(&format!(
        "introspection: {} front ends x {} = {} cells, {} probed",
        INTROSPECTION_CELLS.len(),
        INTROSPECTION.len(),
        INTROSPECTION_CELLS.len() * INTROSPECTION.len(),
        probes.len()
    ));
}

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

// Tests in this file run at the same time and call the same probes. What they share, and why it is safe:
// - scratch folders: one per call (`unique_dir`), removed only by the call that made it;
// - C-ABI and Kotlin handles: one per probe, held by `Freed`, so a panic leaves none in the registry. The
//   registry and its per-handle defaults and load cache are keyed by handle id, so separate handles never meet;
// - the last error and the refused key: thread-local;
// - the process environment: never written here. A probe that needs an environment value must pass it to the
//   child process (`Command::env`), never `std::env::set_var`.

/// A scratch folder path unique to this call: the process id, a counter and `name`. Tests run at the same time
/// (several of them call the same probes) so a folder per process would be shared, and one probe's clean-up could
/// remove another's files.
fn unique_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ggtts-conformance-{}-{n}-{name}", std::process::id()))
}

/// A C-ABI or Kotlin handle, freed when this goes out of scope, so a probe that panics while holding one leaves
/// nothing in the handle registry for the tests still running.
struct Freed(u64);

impl Drop for Freed {
    fn drop(&mut self) {
        ggtts_ffi::ggtts_free(self.0);
    }
}

#[test]
fn each_call_gets_its_own_scratch_folder() {
    let (a, b) = (unique_dir("x"), unique_dir("x"));
    assert_ne!(a, b);
    assert!(a.to_string_lossy().contains(&std::process::id().to_string()));
}

/// A fresh directory with a one-paragraph story in it.
fn scratch(name: &str) -> PathBuf {
    let dir = unique_dir(name);
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
        let r = render(&p("kitten-head"), mode, &["--kitten-full-head"]);
        push("load.kitten_full_head", by_engine_refusal(&r, "load.kitten_full_head"));
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
        let levelled = std::fs::read(r.dir.join("st.wav")).unwrap_or_default();
        let r = render(&p("stream-raw"), mode, &["--stream", "st.wav", "--stream-raw"]);
        let before_level = std::fs::read(r.dir.join("st.wav")).unwrap_or_default();
        push(
            "output.stream_raw",
            if !r.ok {
                by_front_end_refusal(&r, "--stream-raw")
            } else if !before_level.is_empty() && before_level != levelled {
                Seen::Carried
            } else {
                Seen::Dropped
            },
        );
        if mode == "text" {
            let r = render(&p("hold"), mode, &["--stream", "st.wav", "--stream-level-seconds", "0"]);
            push(
                "output.stream_level_seconds",
                by_front_end_refusal(&r, "--stream-level-seconds"),
            );
        }

        // The judge, through the stand-in ASR worker: carried when the trace records what it read back.
        let model = scratch(&p("judge-model"));
        let spec = format!(
            "asr-qwen,model={},command={},interpreter=python",
            model.display(),
            root().join("fixtures/judge/fake_judge_worker.py").display()
        );
        let r = render(
            &p("judge"),
            mode,
            &["--judge", &spec, "--redraw", "1", "--trace", "t.jsonl"],
        );
        let trace = std::fs::read_to_string(r.dir.join("t.jsonl")).unwrap_or_default();
        push(
            "render.judge",
            if !r.ok {
                by_front_end_refusal(&r, "--judge")
            } else if trace.contains("\"ev\":\"read\"") {
                Seen::Carried
            } else {
                Seen::Dropped
            },
        );

        // Redraws on --text: refused by name (a story's redraws are probed by the judge above).
        if mode == "text" {
            let r = render(&p("redraw"), mode, &["--redraw", "1"]);
            push("render.redraw", by_front_end_refusal(&r, "--redraw"));
        }

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
    let freed = Freed(h);
    let text = CString::new("你好。").unwrap();
    let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
    let code = unsafe { ggtts_render(h, text.as_ptr(), 7, &mut p, &mut n, &mut r) };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    if !p.is_null() {
        unsafe { ggtts_pcm_free(p, n) };
    }
    drop(freed);
    out.push((
        "c-abi",
        "render.text".to_string(),
        if n > 0 { Seen::Carried } else { Seen::Dropped },
    ));
    out.extend(c_abi_document_probes(n));
    out
}

/// The C ABI's options documents (`ggtts_load_opts`, `ggtts_render_opts`) on the fake engine: each option where it
/// is read, or its refusal by name. `plain` is the sample count of the same text rendered with no document.
fn c_abi_document_probes(plain: usize) -> Vec<Probe> {
    use ggtts_ffi::{ggtts_load_opts, ggtts_pcm_free, ggtts_render_opts, GGTTS_ERR_ARG, GGTTS_OK};
    let dir = unique_dir("c-abi");
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = 0u64;
    let load = CString::new(r#"{"load": {"engine": "fake"}}"#).unwrap();
    assert_eq!(
        unsafe { ggtts_load_opts(load.as_ptr(), &mut h) },
        GGTTS_OK,
        "{}",
        last_error()
    );
    let _freed = Freed(h);
    let text = CString::new("你好。").unwrap();
    // A whole read with `doc`: its code and sample count.
    let render = |doc: &str| {
        let doc = CString::new(doc).unwrap();
        let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
        let code = unsafe {
            ggtts_render_opts(
                h,
                text.as_ptr(),
                doc.as_ptr(),
                None,
                std::ptr::null_mut(),
                &mut p,
                &mut n,
                &mut r,
            )
        };
        if !p.is_null() {
            unsafe { ggtts_pcm_free(p, n) };
        }
        (code, n)
    };
    // Refused by name, or carried to where the engine refuses it by name.
    let named = |doc: &str, key: &str, carried: bool| match render(doc) {
        (GGTTS_ERR_ARG, _) if last_error().starts_with(&format!("{key}: ")) => {
            if carried {
                Seen::Carried
            } else {
                Seen::Refused
            }
        }
        (code, _) => panic!("c-abi: {doc} gave {code}: {}", last_error()),
    };
    let mut out = Vec::new();
    let mut push = |key: &str, seen| out.push(("c-abi", key.to_string(), seen));
    let head = CString::new(r#"{"load": {"engine": "fake", "kitten_full_head": true}}"#).unwrap();
    let mut refused_handle = 0u64;
    push(
        "load.kitten_full_head",
        match unsafe { ggtts_load_opts(head.as_ptr(), &mut refused_handle) } {
            GGTTS_ERR_ARG if last_error().starts_with("load.kitten_full_head: ") => Seen::Refused,
            code => panic!(
                "c-abi: a load document with load.kitten_full_head gave {code}: {}",
                last_error()
            ),
        },
    );
    // Steering reaches the engine's capabilities, which refuse it on the fake engine: carried only when the refusal
    // is exactly that one, not a binding's own refusal of the key.
    let capability = ggtts_core::capabilities::Capabilities::of("fake")
        .unwrap()
        .refusal("render.steer", None)
        .to_string();
    push(
        "render.steer",
        match render(r#"{"render": {"steer": "s.safetensors"}}"#) {
            (GGTTS_ERR_ARG, _) if last_error() == capability => Seen::Carried,
            (GGTTS_ERR_ARG, _) if last_error().starts_with("render.steer: ") => Seen::Refused,
            (code, _) => panic!("c-abi: steering gave {code}: {}", last_error()),
        },
    );
    // The edge kept at each chunk changes the read's length.
    let (code, n) = render(r#"{"render": {"edge_s": 1.5}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    push("render.edge_s", if n != plain { Seen::Carried } else { Seen::Dropped });
    // The trace file, and its level in the header.
    let trace = dir.join("t.jsonl");
    let doc = format!(
        r#"{{"observe": {{"trace": {}, "trace_level": 2}}}}"#,
        serde_json::to_string(&trace.to_string_lossy()).unwrap()
    );
    assert_eq!(render(&doc).0, GGTTS_OK, "{}", last_error());
    let header = std::fs::read_to_string(&trace).unwrap_or_default();
    push(
        "observe.trace",
        if header.lines().count() > 1 {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    push(
        "observe.trace_level",
        if header.lines().next().is_some_and(|l| l.contains("\"trace_level\":2")) {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The stream: pieces reach the callback.
    extern "C" fn count(user: *mut std::ffi::c_void, _: *const f32, _: usize, _: u32) -> i32 {
        unsafe { *(user as *mut usize) += 1 };
        0
    }
    let mut pieces = 0usize;
    let doc = CString::new(r#"{"output": {"stream": "raw"}}"#).unwrap();
    let null = std::ptr::null_mut();
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            doc.as_ptr(),
            Some(count),
            &mut pieces as *mut usize as *mut std::ffi::c_void,
            null,
            null as *mut usize,
            null as *mut u32,
        )
    };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    push("output.stream", if pieces > 0 { Seen::Carried } else { Seen::Dropped });
    // The options a binding does not take, refused by name.
    for (key, doc) in [
        ("render.measure_timings", r#"{"render": {"measure_timings": true}}"#),
        ("observe.timing_report", r#"{"observe": {"timing_report": "t.json"}}"#),
        (
            "observe.duration_report",
            r#"{"observe": {"duration_report": "d.json"}}"#,
        ),
        ("observe.log_level", r#"{"observe": {"log_level": "info"}}"#),
        ("observe.log_format", r#"{"observe": {"log_format": "json"}}"#),
        ("output.out", r#"{"output": {"out": "o.flac"}}"#),
        ("render.judge", r#"{"render": {"judge": "asr-qwen,model=m"}}"#),
    ] {
        push(key, named(doc, key, false));
    }
    let _ = std::fs::remove_dir_all(dir);
    out
}

/// A binding call's result: the value, or the failure (code and message) and the refused key.
type Keyed<T> = Result<T, ((i32, String), Option<String>)>;

/// Python's `options=` on the fake engine, through the binding's own core (`ggtts_py::core`, the plain Rust under
/// the pyo3 module): each option where it is read, or its refusal by its key. Each render is on a fresh engine, since
/// the fake engine's output depends on what it rendered before.
fn python_probes() -> Vec<Probe> {
    use ggtts_py::core::{guard_keyed, load, LoadRequest, ERR_ARG};
    let dir = unique_dir("python");
    std::fs::create_dir_all(&dir).unwrap();
    let fresh = |options: Option<&str>| {
        let req = LoadRequest {
            options: Some(options.unwrap_or(r#"{"load": {"engine": "fake"}}"#).to_string()),
            ..Default::default()
        };
        guard_keyed(|| load(&req, &mut |_| {})).map(|(h, _)| h)
    };
    let text = "你好。";
    // A whole read with the render document `doc`: its samples, or the failure and the refused key.
    let render = |doc: &str| guard_keyed(|| fresh(None).map_err(|(e, _)| e)?.render_with(text, None, Some(doc)));
    let by_key = |r: Keyed<(Vec<f32>, u32)>, key: &str| match r {
        Err(((ERR_ARG, msg), Some(k))) if k == key && msg.starts_with(&format!("{key}: ")) => Seen::Refused,
        other => panic!("python: {key} gave {:?}", other.map(|(s, _)| s.len())),
    };
    let mut out = Vec::new();
    let mut push = |key: &str, seen| out.push(("python", key.to_string(), seen));
    let plain = render("{}").unwrap().0;
    // Steering and a style instruction reach the engine's capabilities, which refuse them on the fake engine: carried
    // only when the refusal is exactly that one, not a binding's own refusal of the key.
    let caps = ggtts_core::capabilities::Capabilities::of("fake").unwrap();
    for (key, doc) in [
        ("render.steer", r#"{"render": {"steer": "s.safetensors"}}"#),
        ("render.instruction", r#"{"render": {"instruction": "Speak slowly."}}"#),
    ] {
        let capability = caps.refusal(key, None).to_string();
        push(
            key,
            match render(doc) {
                Err(((ERR_ARG, msg), Some(k))) if k == key && msg == capability => Seen::Carried,
                Err(((ERR_ARG, _), Some(k))) if k == key => Seen::Refused,
                other => panic!("python: {key} gave {:?}", other.map(|(s, _)| s.len())),
            },
        );
    }
    // The edge kept at each chunk, given in a render document, changes the read's length.
    let edged = render(r#"{"render": {"edge_s": 1.5}}"#).unwrap().0;
    push(
        "render.edge_s",
        if edged.len() != plain.len() {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The salt moves every chunk's sampler seed, as the trace records it.
    let seeds = |salt: &str| {
        let trace = dir.join(format!("salt-{salt}.jsonl"));
        let doc = format!(
            r#"{{"render": {{"seed_salt": "{salt}"}}, "observe": {{"trace": {}}}}}"#,
            serde_json::to_string(&trace.to_string_lossy()).unwrap()
        );
        render(&doc).unwrap();
        trace_seeds(&trace)
    };
    let (seven, eight) = (seeds("7"), seeds("8"));
    push(
        "render.seed_salt",
        if !seven.is_empty() && seven != eight {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The load document's engine is the engine loaded: "fake", where the document's default ("auto", with no model)
    // would be qwen3-tts.
    let named = load(
        &LoadRequest {
            options: Some(r#"{"load": {"engine": "fake"}}"#.into()),
            ..Default::default()
        },
        &mut |_| {},
    );
    push(
        "load.engine",
        match named {
            Ok((_, loaded)) if loaded.engine == "fake" && loaded.family == "fake" => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    // The trace file, and its level in the header.
    let trace = dir.join("t.jsonl");
    let doc = format!(
        r#"{{"observe": {{"trace": {}, "trace_level": 2}}}}"#,
        serde_json::to_string(&trace.to_string_lossy()).unwrap()
    );
    render(&doc).unwrap();
    let header = std::fs::read_to_string(&trace).unwrap_or_default();
    push(
        "observe.trace",
        if header.lines().count() > 1 {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    push(
        "observe.trace_level",
        if header.lines().next().is_some_and(|l| l.contains("\"trace_level\":2")) {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The stream: pieces reach the callback.
    let mut pieces = 0usize;
    let streamed = guard_keyed(|| {
        fresh(None).map_err(|(e, _)| e)?.render_stream_with(
            text,
            None,
            Some(r#"{"output": {"stream": "raw"}}"#),
            &mut |_, _| {
                pieces += 1;
                true
            },
        )
    });
    assert!(streamed.is_ok(), "python: the stream failed: {streamed:?}");
    push("output.stream", if pieces > 0 { Seen::Carried } else { Seen::Dropped });
    // The options a binding does not take, refused by their keys.
    for (key, doc) in [
        ("render.measure_timings", r#"{"render": {"measure_timings": true}}"#),
        ("render.seed", r#"{"render": {"seed": 3}}"#),
        ("observe.timing_report", r#"{"observe": {"timing_report": "t.json"}}"#),
        (
            "observe.duration_report",
            r#"{"observe": {"duration_report": "d.json"}}"#,
        ),
        ("observe.log_level", r#"{"observe": {"log_level": "info"}}"#),
        ("observe.log_format", r#"{"observe": {"log_format": "json"}}"#),
        ("output.out", r#"{"output": {"out": "o.flac"}}"#),
        ("render.judge", r#"{"render": {"judge": "asr-qwen,model=m"}}"#),
    ] {
        push(key, by_key(render(doc), key));
    }
    out.extend(python_engine_probes(&dir));
    let _ = std::fs::remove_dir_all(dir);
    out
}

/// What [`Recorder`] saw: each request's draw, and the steering it carried (its scale and layers).
#[derive(Default)]
struct Recorded {
    draws: Vec<u32>,
    steer: Vec<(f64, Vec<u32>)>,
}

/// An engine that steers, for the probes the fake engine cannot answer: the fake engine stating Qwen3-TTS's
/// capabilities, recording each request's draw and steering, and handing the fake engine the request without the
/// steering (which it would refuse). Its first draw of a chunk is the fake engine's speech for a text twenty times
/// as long, which the duration judge rejects as a runaway, so a redraw is asked for when one is allowed.
struct Recorder {
    fake: ggtts_fake::FakeEngine,
    seen: std::sync::Arc<std::sync::Mutex<Recorded>>,
}

impl ggtts_core::engine::Engine for Recorder {
    fn capabilities(&self) -> ggtts_core::capabilities::Capabilities {
        ggtts_core::capabilities::Capabilities::of("qwen3-tts").unwrap()
    }
    fn info(&self) -> &ggtts_core::engine::EngineInfo {
        self.fake.info()
    }
    fn profile(&self) -> &ggtts_core::profile::Profile {
        self.fake.profile()
    }
    fn generate(
        &mut self,
        req: &ggtts_core::engine::ChunkRequest<'_>,
    ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
        let mut seen = self.seen.lock().unwrap();
        seen.draws.push(req.draw);
        if let Some(s) = req.steer {
            seen.steer.push((s.scale, s.layer_indices()));
        }
        drop(seen);
        let long = req.text.repeat(20);
        let text = if req.draw == 0 { long.as_str() } else { req.text };
        self.fake.generate(&ggtts_core::engine::ChunkRequest {
            text,
            steer: None,
            ..*req
        })
    }
    fn decode(
        &mut self,
        codes: &ggtts_core::engine::Codes,
    ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
        self.fake.decode(codes)
    }
}

/// Python's `options=` through its load path (`ggtts_py::core::load_over`) over [`Recorder`], loaded as qwen3-tts
/// so the document is checked against an engine that steers: `render.steer_scale` and `render.steer_layers` are
/// seen in the steering the engine is handed, `render.redraw` in the draws it is asked for, and `load.cache_dir` in
/// the offline refusal of an uncached steering URL, which names the cache the load gave.
fn python_engine_probes(dir: &Path) -> Vec<Probe> {
    use ggtts_py::core::{guard_keyed, load_over, LoadRequest};
    let text = "你好。";
    let steer = dir.join("steer.safetensors");
    let v = [0.5f32; 8];
    std::fs::write(
        &steer,
        ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v), ("talker.layer.1", &v)]),
    )
    .unwrap();
    let steer = serde_json::to_string(&steer.to_string_lossy()).unwrap();
    // Render `text` with the render document `doc` on a fresh recorder loaded with `load_doc`: what it saw, and the
    // render's result.
    let run = |load_doc: &str, doc: &str| {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Recorded::default()));
        let engine = Recorder {
            fake: ggtts_fake::FakeEngine::default(),
            seen: seen.clone(),
        };
        let req = LoadRequest {
            options: Some(load_doc.to_string()),
            ..Default::default()
        };
        let result = guard_keyed(|| {
            load_over(&req, Box::new(engine))?
                .0
                .render_with(text, Some(7), Some(doc))
        });
        let seen = std::sync::Arc::try_unwrap(seen).ok().unwrap().into_inner().unwrap();
        (seen, result)
    };
    let qwen = r#"{"load": {"engine": "qwen3-tts"}}"#;
    let mut out = Vec::new();
    let mut push = |key: &str, seen| out.push(("python", key.to_string(), seen));
    // Steering: the scale and the layers the engine is handed are the document's (the file has layers 0 and 1).
    let (plain, r) = run(qwen, &format!(r#"{{"render": {{"steer": {steer}, "redraw": 0}}}}"#));
    assert!(r.is_ok(), "python: a steered render failed: {r:?}");
    let (scaled, _) = run(
        qwen,
        &format!(r#"{{"render": {{"steer": {steer}, "steer_scale": 0.25, "redraw": 0}}}}"#),
    );
    let (layered, _) = run(
        qwen,
        &format!(r#"{{"render": {{"steer": {steer}, "steer_layers": [1], "redraw": 0}}}}"#),
    );
    let first = |s: &Recorded| s.steer.first().cloned();
    push(
        "render.steer_scale",
        match (first(&plain), first(&scaled)) {
            (Some((1.0, _)), Some((0.25, _))) => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    push(
        "render.steer_layers",
        match (first(&plain), first(&layered)) {
            (Some((_, all)), Some((_, one))) if all == vec![0, 1] && one == vec![1] => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    // Redraws: with none allowed the runaway first draw is the only one; with one allowed, a second draw is asked for.
    let (none, _) = run(qwen, r#"{"render": {"redraw": 0}}"#);
    let (one, _) = run(qwen, r#"{"render": {"redraw": 1}}"#);
    push(
        "render.redraw",
        if none.draws == vec![0] && one.draws == vec![0, 1] {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The load's cache directory: an offline engine refuses an uncached steering URL, naming that cache. The host is
    // `.invalid`, so a resolution that ignored the load would fail on the network instead.
    let cache = dir.join("python-cache");
    let load_doc =
        serde_json::json!({"load": {"engine": "qwen3-tts", "offline": true, "cache_dir": cache}}).to_string();
    let (_, r) = run(
        &load_doc,
        r#"{"render": {"steer": "https://example.invalid/presets/x.safetensors"}}"#,
    );
    push(
        "load.cache_dir",
        match r {
            Err(((_, msg), _))
                if msg.contains("offline mode does not download it") && msg.contains(&cache.display().to_string()) =>
            {
                Seen::Carried
            }
            _ => Seen::Dropped,
        },
    );
    out
}

/// Kotlin's `options` on the fake engine, through the binding's own core under its JNI layer (`ggtts_ffi::kotlin`,
/// what `Native.loadOpts`, `renderOpts` and `renderStreamOpts` call): each option where it is read, or its refusal by
/// its key.
fn kotlin_probes() -> Vec<Probe> {
    use ggtts_ffi::kotlin::{load, render_opts, render_stream_opts, LoadArgs};
    use ggtts_py::core::{guard_keyed, ERR_ARG};
    let text = "你好。";
    let fresh = |doc: &str| load(doc, &LoadArgs::default()).unwrap();
    let fake = r#"{"load": {"engine": "fake"}}"#;
    // A whole read with the render document `doc` on a fresh engine.
    let render = |doc: &str| {
        let h = Freed(fresh(fake));
        guard_keyed(|| render_opts(h.0, text, Some(doc), None))
    };
    let mut out = Vec::new();
    let mut push = |key: &str, seen| out.push(("kotlin", key.to_string(), seen));
    let plain = render("{}").unwrap().0;
    let caps = ggtts_core::capabilities::Capabilities::of("fake").unwrap();
    for (key, doc) in [
        ("render.steer", r#"{"render": {"steer": "s.safetensors"}}"#),
        ("render.instruction", r#"{"render": {"instruction": "Speak slowly."}}"#),
    ] {
        let capability = caps.refusal(key, None).to_string();
        push(
            key,
            match render(doc) {
                Err(((ERR_ARG, msg), Some(k))) if k == key && msg == capability => Seen::Carried,
                Err(((ERR_ARG, _), Some(k))) if k == key => Seen::Refused,
                other => panic!("kotlin: {key} gave {:?}", other.map(|(s, _)| s.len())),
            },
        );
    }
    let edged = render(r#"{"render": {"edge_s": 1.5}}"#).unwrap().0;
    push(
        "render.edge_s",
        if edged.len() != plain.len() {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The load document's engine is the engine loaded: "fake" loads, where the document's default ("auto", with no
    // model) would be qwen3-tts, which this build cannot load from no files.
    push(
        "load.engine",
        match guard_keyed(|| load(fake, &LoadArgs::default())) {
            Ok(h) => {
                ggtts_ffi::ggtts_free(h);
                Seen::Carried
            }
            Err(_) => Seen::Dropped,
        },
    );
    // The model path reaches the engine's file check, which names it.
    let dir = unique_dir("kotlin");
    std::fs::create_dir_all(&dir).unwrap();
    let not_gguf = dir.join("not-a-model.gguf");
    std::fs::write(&not_gguf, b"not a model").unwrap();
    let doc = format!(
        r#"{{"load": {{"engine": "qwen3-tts", "model": {0}, "codec": {0}, "offline": true}}}}"#,
        serde_json::to_string(&not_gguf.to_string_lossy()).unwrap()
    );
    push(
        "load.model",
        match guard_keyed(|| load(&doc, &LoadArgs::default())) {
            Err(((_, msg), _)) if msg.contains("not-a-model.gguf") => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    let _ = std::fs::remove_dir_all(&dir);
    // A speaker reaches the engine's capabilities, which refuse it on the fake engine in their own words.
    let speaker = caps.refusal("load.speaker", None).to_string();
    push(
        "load.speaker",
        match guard_keyed(|| load(r#"{"load": {"engine": "fake", "speaker": "a"}}"#, &LoadArgs::default())) {
            Err(((ERR_ARG, msg), Some(k))) if k == "load.speaker" && msg == speaker => Seen::Carried,
            Err(((ERR_ARG, _), Some(k))) if k == "load.speaker" => Seen::Refused,
            other => panic!("kotlin: load.speaker gave {:?}", other.is_ok()),
        },
    );
    // The stream: pieces reach the callback.
    let h = Freed(fresh(fake));
    let mut pieces = 0usize;
    let streamed = guard_keyed(|| {
        render_stream_opts(
            h.0,
            text,
            Some(r#"{"output": {"stream": "raw"}}"#),
            None,
            &mut |_, _| {
                pieces += 1;
                true
            },
        )
    });
    drop(h);
    assert!(streamed.is_ok(), "kotlin: the stream failed: {streamed:?}");
    push("output.stream", if pieces > 0 { Seen::Carried } else { Seen::Dropped });
    // What the binding refuses, by its key: in a render document, and a device in a load document.
    let by_key = |r: Keyed<(Vec<f32>, u32)>, key: &str| match r {
        Err(((ERR_ARG, msg), Some(k))) if k == key && msg.starts_with(&format!("{key}: ")) => Seen::Refused,
        other => panic!("kotlin: {key} gave {:?}", other.map(|(s, _)| s.len())),
    };
    for (key, doc) in [
        ("observe.trace", r#"{"observe": {"trace": "t.jsonl"}}"#),
        ("observe.trace_level", r#"{"observe": {"trace_level": 2}}"#),
        ("observe.instruments", r#"{"observe": {"instruments": "i"}}"#),
        ("render.measure_timings", r#"{"render": {"measure_timings": true}}"#),
        ("render.seed", r#"{"render": {"seed": 3}}"#),
        ("observe.timing_report", r#"{"observe": {"timing_report": "t.json"}}"#),
        ("observe.log_level", r#"{"observe": {"log_level": "info"}}"#),
        ("output.out", r#"{"output": {"out": "o.flac"}}"#),
    ] {
        push(key, by_key(render(doc), key));
    }
    push(
        "load.device",
        match guard_keyed(|| load(r#"{"load": {"engine": "fake", "device": "cpu"}}"#, &LoadArgs::default())) {
            Err(((ERR_ARG, msg), Some(k))) if k == "load.device" && msg.starts_with("load.device: ") => Seen::Refused,
            other => panic!("kotlin: load.device gave {:?}", other.is_ok()),
        },
    );
    out.extend(kotlin_engine_probes());
    out
}

/// Kotlin's `load(options=…)` through its native load path (`ggtts_ffi::kotlin::load_over`) over [`Recorder`], loaded
/// as qwen3-tts so the document is checked against an engine that steers, and rendered through `render_opts` as the
/// binding renders: `render.steer_scale` and `render.steer_layers` are seen in the steering the engine is handed,
/// `render.redraw` in the draws it is asked for, and `load.cache_dir` in the offline refusal of an uncached steering
/// URL, which names the cache the load gave. As Python's ([`python_engine_probes`]).
fn kotlin_engine_probes() -> Vec<Probe> {
    use ggtts_ffi::kotlin::{load_over, render_opts, LoadArgs};
    use ggtts_py::core::guard_keyed;
    let dir = unique_dir("kotlin-engine");
    std::fs::create_dir_all(&dir).unwrap();
    let text = "你好。";
    let steer = dir.join("steer.safetensors");
    let v = [0.5f32; 8];
    std::fs::write(
        &steer,
        ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v), ("talker.layer.1", &v)]),
    )
    .unwrap();
    let steer = serde_json::to_string(&steer.to_string_lossy()).unwrap();
    // Render `text` with the render document `doc` on a fresh recorder loaded with `load_doc`: what it saw, and the
    // render's result.
    let run = |load_doc: &str, doc: &str| {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Recorded::default()));
        let engine = Recorder {
            fake: ggtts_fake::FakeEngine::default(),
            seen: seen.clone(),
        };
        let result = guard_keyed(|| {
            let h = Freed(load_over(load_doc, &LoadArgs::default(), Box::new(engine))?);
            render_opts(h.0, text, Some(doc), Some(7))
        });
        let seen = std::sync::Arc::try_unwrap(seen).ok().unwrap().into_inner().unwrap();
        (seen, result)
    };
    let qwen = r#"{"load": {"engine": "qwen3-tts"}}"#;
    let mut out = Vec::new();
    let mut push = |key: &str, seen| out.push(("kotlin", key.to_string(), seen));
    // Steering: the scale and the layers the engine is handed are the document's (the file has layers 0 and 1).
    let (plain, r) = run(qwen, &format!(r#"{{"render": {{"steer": {steer}, "redraw": 0}}}}"#));
    assert!(r.is_ok(), "kotlin: a steered render failed: {r:?}");
    let (scaled, _) = run(
        qwen,
        &format!(r#"{{"render": {{"steer": {steer}, "steer_scale": 0.25, "redraw": 0}}}}"#),
    );
    let (layered, _) = run(
        qwen,
        &format!(r#"{{"render": {{"steer": {steer}, "steer_layers": [1], "redraw": 0}}}}"#),
    );
    let first = |s: &Recorded| s.steer.first().cloned();
    push(
        "render.steer_scale",
        match (first(&plain), first(&scaled)) {
            (Some((1.0, _)), Some((0.25, _))) => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    push(
        "render.steer_layers",
        match (first(&plain), first(&layered)) {
            (Some((_, all)), Some((_, one))) if all == vec![0, 1] && one == vec![1] => Seen::Carried,
            _ => Seen::Dropped,
        },
    );
    // Redraws: with none allowed the runaway first draw is the only one; with one allowed, a second draw is asked for.
    let (none, _) = run(qwen, r#"{"render": {"redraw": 0}}"#);
    let (one, _) = run(qwen, r#"{"render": {"redraw": 1}}"#);
    push(
        "render.redraw",
        if none.draws == vec![0] && one.draws == vec![0, 1] {
            Seen::Carried
        } else {
            Seen::Dropped
        },
    );
    // The load's cache directory: an offline engine refuses an uncached steering URL, naming that cache. The host is
    // `.invalid`, so a resolution that ignored the load would fail on the network instead.
    let cache = dir.join("kotlin-cache");
    let load_doc =
        serde_json::json!({"load": {"engine": "qwen3-tts", "offline": true, "cache_dir": cache}}).to_string();
    let (_, r) = run(
        &load_doc,
        r#"{"render": {"steer": "https://example.invalid/presets/x.safetensors"}}"#,
    );
    push(
        "load.cache_dir",
        match r {
            Err(((_, msg), _))
                if msg.contains("offline mode does not download it") && msg.contains(&cache.display().to_string()) =>
            {
                Seen::Carried
            }
            _ => Seen::Dropped,
        },
    );
    let _ = std::fs::remove_dir_all(dir);
    out
}

// ── the stream mode through each binding's document ──

/// Two paragraphs, so the fake engine's two chunks come back 6 dB apart and a level step has something to do.
const STREAM_TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

/// A binding's stream on a fresh fake engine, from a document (`None`: the call without one), the pieces joined.
type StreamFn = fn(Option<&str>) -> Vec<f32>;

/// The C ABI's stream on a fresh fake engine: `ggtts_render_opts` with `doc` and a callback, or, with `doc` `None`,
/// the old `ggtts_render_stream` at salt 7. The pieces, joined.
fn c_abi_stream(doc: Option<&str>) -> Vec<f32> {
    use ggtts_ffi::{ggtts_load_opts, ggtts_render_opts, ggtts_render_stream, GGTTS_OK};
    extern "C" fn keep(user: *mut std::ffi::c_void, samples: *const f32, len: usize, _: u32) -> i32 {
        let out = unsafe { &mut *(user as *mut Vec<f32>) };
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(samples, len) });
        0
    }
    let load = CString::new(r#"{"load": {"engine": "fake"}}"#).unwrap();
    let mut h = 0u64;
    assert_eq!(
        unsafe { ggtts_load_opts(load.as_ptr(), &mut h) },
        GGTTS_OK,
        "{}",
        last_error()
    );
    let _freed = Freed(h);
    let text = CString::new(STREAM_TEXT).unwrap();
    let mut out: Vec<f32> = Vec::new();
    let user = &mut out as *mut Vec<f32> as *mut std::ffi::c_void;
    let code = match doc {
        None => unsafe { ggtts_render_stream(h, text.as_ptr(), 7, Some(keep), user) },
        Some(doc) => {
            let doc = CString::new(doc).unwrap();
            let null = std::ptr::null_mut();
            unsafe {
                ggtts_render_opts(
                    h,
                    text.as_ptr(),
                    doc.as_ptr(),
                    Some(keep),
                    user,
                    null,
                    null as *mut usize,
                    null as *mut u32,
                )
            }
        }
    };
    assert_eq!(code, GGTTS_OK, "c-abi stream {doc:?}: {}", last_error());
    out
}

/// Python's stream on a fresh fake engine, through its core: `render_stream_with` with `doc`, or the old
/// `render_stream` at salt 7 with `doc` `None`.
fn python_stream(doc: Option<&str>) -> Vec<f32> {
    use ggtts_py::core::{load, LoadRequest};
    let req = LoadRequest {
        engine: Some("fake".into()),
        ..Default::default()
    };
    let (h, _) = load(&req, &mut |_| {}).unwrap();
    let mut out: Vec<f32> = Vec::new();
    let mut keep = |s: &[f32], _: u32| {
        out.extend_from_slice(s);
        true
    };
    match doc {
        None => h.render_stream(STREAM_TEXT, 7, &mut keep),
        Some(doc) => h.render_stream_with(STREAM_TEXT, None, Some(doc), &mut keep),
    }
    .unwrap_or_else(|e| panic!("python stream {doc:?}: {e:?}"));
    out
}

/// Kotlin's stream on a fresh fake engine, through the core under its JNI layer: `render_stream_opts` with `doc`, or,
/// with `doc` `None`, the call `renderStream` makes without options (the C ABI's `ggtts_render_stream`).
fn kotlin_stream(doc: Option<&str>) -> Vec<f32> {
    use ggtts_ffi::kotlin;
    let Some(doc) = doc else {
        return c_abi_stream(None);
    };
    let h = Freed(kotlin::load(r#"{"load": {"engine": "fake"}}"#, &kotlin::LoadArgs::default()).unwrap());
    let mut out: Vec<f32> = Vec::new();
    kotlin::render_stream_opts(h.0, STREAM_TEXT, Some(doc), None, &mut |s, _| {
        out.extend_from_slice(s);
        true
    })
    .unwrap_or_else(|e| panic!("kotlin stream {doc}: {e:?}"));
    out
}

/// What a probe of a binding's stream saw: `old` is the binding's call without a document (the pieces before the
/// level step), `default` its document stream with no mode given, `raw` with `"stream_raw": true`, and `held` with
/// `"stream_level_seconds": 0`. `output.stream_raw` is carried when it gives the old pieces and the default does not
/// (the default is levelled); `output.stream_level_seconds` when the hold changes the levelled stream.
fn stream_seen(old: &[f32], default: &[f32], raw: &[f32], held: &[f32]) -> [(&'static str, Seen); 2] {
    let raw_seen = if raw == old && default != old && default.len() == old.len() {
        Seen::Carried
    } else {
        Seen::Dropped
    };
    let hold_seen = if held != default && held.len() == default.len() {
        Seen::Carried
    } else {
        Seen::Dropped
    };
    [
        ("output.stream_raw", raw_seen),
        ("output.stream_level_seconds", hold_seen),
    ]
}

/// The documents a stream probe renders: no mode (levelled), `stream_raw`, and a hold of 0 s (the first chunk alone).
const STREAM_DOCS: [&str; 3] = [
    r#"{"render": {"seed_salt": "7"}}"#,
    r#"{"render": {"seed_salt": "7"}, "output": {"stream": "levelled", "stream_raw": true}}"#,
    r#"{"render": {"seed_salt": "7"}, "output": {"stream": "levelled", "stream_level_seconds": 0}}"#,
];

fn stream_probes() -> Vec<Probe> {
    let bindings: [(&'static str, StreamFn); 3] = [
        ("c-abi", c_abi_stream),
        ("python", python_stream),
        ("kotlin", kotlin_stream),
    ];
    let mut out = Vec::new();
    for (id, stream) in bindings {
        // The C ABI's document default needs a callback and no output.stream; the stream calls of Python and Kotlin
        // stream whatever document they are given.
        let [default, raw, held] = STREAM_DOCS.map(|d| stream(Some(d)));
        for (key, seen) in stream_seen(&stream(None), &default, &raw, &held) {
            out.push((id, key.to_string(), seen));
        }
    }
    out
}

fn probes() -> Vec<Probe> {
    let mut out = render_probes();
    out.extend(book_probes());
    out.extend(c_abi_probes());
    out.extend(python_probes());
    out.extend(kotlin_probes());
    out.extend(stream_probes());
    let kotlin: Vec<Probe> = out.iter().filter(|(fe, _, _)| *fe == "kotlin").cloned().collect();
    out.extend(android_probes(&read(ANDROID_SEAM), &kotlin));
    out
}

// ── the Android app: its document handed to the Kotlin binding ──

/// The demo app's binding seam, where its request becomes the binding's call.
const ANDROID_SEAM: &str = "android/demo/src/main/kotlin/ggtts/demo/TtsBinding.kt";

/// What `GgttsBinding` must hand the Kotlin binding for the app to carry its document: the load document at load, and
/// the request's render and stream documents on every render.
const ANDROID_HANDOVER: [&str; 3] = [
    "Ggtts.load(allowDownload = false, options = resolved.loadDocument(cacheDir))",
    "tts.render(text, seed, options = whole)",
    "tts.renderStream(text, seed, options = stream)",
];

/// The Android app's probes. The app runs on a phone, not in this test, so it is probed at its seam: when `seam` (the
/// source of `GgttsBinding`) hands every call its document ([`ANDROID_HANDOVER`]), the app sees each option as the
/// Kotlin binding does (`kotlin`, that binding's probes); when any call is made without it, every option is dropped.
/// The app's own JVM tests and `scripts/kotlin_binding_test.sh` (android/demo/src/hostTest) run the seam itself.
fn android_probes(seam: &str, kotlin: &[Probe]) -> Vec<Probe> {
    let code = code_only(seam);
    let handed = ANDROID_HANDOVER.iter().all(|h| code.contains(h));
    kotlin
        .iter()
        .map(|(_, key, seen)| ("android", key.clone(), if handed { *seen } else { Seen::Dropped }))
        .collect()
}

/// The options the app claims to carry that its binding does not: the app hands its document to the Kotlin binding,
/// so where the binding's cell is a gap or a refusal, the app's cannot be Carried.
fn wider_than_its_binding(cells: &BTreeMap<(String, String), Cell>) -> Vec<String> {
    let mut wrong = Vec::new();
    for d in TABLE {
        let key = d.key();
        let kotlin = cells[&("kotlin".to_string(), key.clone())];
        let android = cells[&("android".to_string(), key.clone())];
        if android == Carried && kotlin != Carried {
            wrong.push(format!("{key}: android is Carried, kotlin is {kotlin:?}"));
        }
    }
    wrong
}

/// The app carries no more than its binding, and a planted android cell wider than Kotlin's fails, naming it.
#[test]
fn the_app_carries_no_more_than_its_binding() {
    let mut cells = cells(TABLE, FRONT_ENDS).unwrap();
    let wrong = wider_than_its_binding(&cells);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    // The plant: the app claims a timing measurement, which the binding refuses (it returns samples, not timings).
    cells.insert(("android".to_string(), "render.measure_timings".to_string()), Carried);
    let planted = wider_than_its_binding(&cells);
    assert_eq!(planted.len(), 1, "{planted:?}");
    assert!(
        planted[0].starts_with("render.measure_timings: android is Carried, kotlin is Refused("),
        "{planted:?}"
    );
}

/// #282 AC6: the app's carried options are exercised through its seam, and an app whose binding call drops the
/// document fails: a planted seam that renders without it sees the instruction dropped.
#[test]
fn an_app_that_renders_without_its_document_fails() {
    let seam = read(ANDROID_SEAM);
    let kotlin = kotlin_probes();
    let seen = android_probes(&seam, &kotlin);
    assert!(!seen.is_empty());
    let cells = cells(TABLE, FRONT_ENDS).unwrap();
    for (fe, key, s) in &seen {
        verdict(fe, key, cells[&(fe.to_string(), key.clone())], *s).unwrap();
    }
    let planted = seam.replace("tts.render(text, seed, options = whole)", "tts.render(text, seed)");
    assert_ne!(
        planted, seam,
        "the seam no longer has the render hand-over this test plants out"
    );
    let dropped = android_probes(&planted, &kotlin);
    let instruction = dropped.iter().find(|(_, k, _)| k == "render.instruction").unwrap();
    let e = verdict("android", "render.instruction", Carried, instruction.2).unwrap_err();
    assert!(
        e.contains("render.instruction is Carried in the table, and the probe saw it Dropped"),
        "{e}"
    );
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
    let key = ("book".to_string(), "render.edge_s".to_string());
    assert_eq!(cells[&key], Gap(283));
    cells.insert(key, Carried);
    let errors = check_gaps(&cells, &read(GAPS));
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("book\\trender.edge_s\\t#283"), "{errors:?}");
}

/// #284 AC3: a document path that ignores `raw` (it levels anyway) fails: the probe sees `output.stream_raw` dropped,
/// and the carried cell turns red. So does one that ignores the hold.
#[test]
fn a_document_stream_that_ignores_raw_fails() {
    let old = c_abi_stream(None);
    let [default, _, held] = STREAM_DOCS.map(|d| c_abi_stream(Some(d)));
    // Planted: the raw document levelled like the default.
    let [(key, seen), _] = stream_seen(&old, &default, &default, &held);
    let e = verdict("c-abi", key, Carried, seen).unwrap_err();
    assert!(
        e.contains("output.stream_raw is Carried in the table, and the probe saw it Dropped"),
        "{e}"
    );
    // Planted: the hold ignored, so the held stream is the default one.
    let [_, (key, seen)] = stream_seen(&old, &default, &old, &default);
    assert!(verdict("c-abi", key, Carried, seen).is_err());
    // Unplanted, both are carried.
    let [raw, hold] = stream_seen(&old, &default, &c_abi_stream(Some(STREAM_DOCS[1])), &held);
    assert_eq!((raw.1, hold.1), (Seen::Carried, Seen::Carried));
}

/// #284 AC2: a binding's document stream with no mode given is `ggtts-render --stream`'s levelled stream for the same
/// story and seed, sample for sample at the stream's 16 bits; `"raw"` is the old call, and so is a stream call with no
/// document.
#[test]
fn a_document_stream_is_ggtts_renders_levelled_stream() {
    let dir = scratch("stream-levelled");
    // The story ggtts-render renders is the binding's: its title is the binding's story id, and the salt is the
    // binding's seed 7 against ggtts-render's (unset) story salt, so both are given as the empty salt.
    std::fs::write(
        dir.join("b.txt"),
        format!("{}\n{STREAM_TEXT}\n", ggtts_load::binding::STORY),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--log-level", "error", "--story", "b.txt"])
        .args(["--stream", "s.pcm", "--stream-format", "s16le"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(dir.join("s.pcm")).unwrap();
    let body = &bytes[bytes.iter().position(|&b| b == b'\n').unwrap() + 1..];
    let cli: Vec<i16> = body.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    let doc = r#"{"render": {"seed_salt": ""}}"#;
    let raw = r#"{"render": {"seed_salt": ""}, "output": {"stream": "raw"}}"#;
    let bindings: [(&str, StreamFn); 3] = [
        ("c-abi", c_abi_stream),
        ("python", python_stream),
        ("kotlin", kotlin_stream),
    ];
    for (id, stream) in bindings {
        let levelled: Vec<i16> = stream(Some(doc)).iter().map(|&x| ggtts_core::flac::to_i16(x)).collect();
        assert!(!cli.is_empty(), "ggtts-render streamed nothing");
        assert_eq!(
            levelled, cli,
            "{id}: the document's default stream is not ggtts-render's levelled stream"
        );
        assert_eq!(
            stream(Some(raw)).len(),
            cli.len(),
            "{id}: the raw stream has another length"
        );
        assert_ne!(stream(Some(raw)), stream(Some(doc)), "{id}: the raw stream is levelled");
    }
    // Without a document the stream calls are the old ones: the pieces before the level step.
    assert_eq!(python_stream(None), c_abi_stream(None));
    let _ = std::fs::remove_dir_all(dir);
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
        "#[pyo3(signature = (engine=None,",
        "#[pyo3(signature = (engine=None, foo=None,",
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
