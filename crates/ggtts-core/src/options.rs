//! The options value: every option a front end takes, defined once, as data.
//!
//! [`Options`] holds them in sections ([`Load`], [`Render`], [`Observe`], [`Output`], [`Book`]) behind a
//! [`version`](Options::version). [`TABLE`] has one [`Descriptor`] per option: its name, section, phase, type, default,
//! every spelling (flag, environment variable, and each binding's parameter) and one line of help. The schema
//! (`schemas/ggtts-options-1.schema.json`) is generated from the table ([`schema`]) and checked against the committed
//! file, and the parsers ([`Options::from_args`], [`Options::from_json`], [`Options::from_env`]) read the table, so a
//! spelling the table does not have is refused rather than guessed.
//!
//! Each value the parsers set is marked with where it came from ([`Source`]). A value from a flag or a document is a
//! request for this render; a value that comes only from the environment is ambient, and an engine without that
//! option ignores it with one warning and lists it in the record ([`Options::apply_engine_rule`]). A flag or document
//! value always wins over the environment ([`Options::layer`]).
//!
//! [`Options::render_options`] turns the render section into the pipeline's [`RenderOptions`], and [`record`] writes
//! the provenance record; [`reported`] is the part of it every report carries. `ggtts-render` reads its command line
//! and environment through this module, and `ggtts-book` its command line. `docs/options.md` describes the sections,
//! the phases, the seed keys and the environment rule.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::capabilities::Capabilities;
use crate::engine::EngineInfo;
use crate::pipeline::RenderOptions;
use crate::profile::Profile;
use crate::steer::{SteerFile, Steering};

/// The options value's format version, written as `version` in every document.
pub const VERSION: u32 = 1;

/// The schema's `$id`.
pub const SCHEMA_ID: &str = "https://github.com/dandelion-love/ggtts/schemas/ggtts-options-1.schema.json";

/// The seed `--text` renders at when no `seed` is given.
pub const DEFAULT_SEED: u64 = 20260905;

/// When an option is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Fixed when the engine is opened.
    Load,
    /// Chosen per render.
    Render,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Load => "load",
            Phase::Render => "render",
        }
    }
}

/// A section of the options value, as its document key names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Section {
    Load,
    Render,
    Observe,
    Output,
    Book,
}

impl Section {
    pub const ALL: [Section; 5] = [
        Section::Load,
        Section::Render,
        Section::Observe,
        Section::Output,
        Section::Book,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Section::Load => "load",
            Section::Render => "render",
            Section::Observe => "observe",
            Section::Output => "output",
            Section::Book => "book",
        }
    }

    fn by_name(name: &str) -> Option<Section> {
        Section::ALL.into_iter().find(|s| s.name() == name)
    }
}

/// An option's type, with the values it allows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// `true` or `false`; a flag spelling takes no value and sets `true`.
    Bool,
    /// A whole number in `[min, max]`.
    Int { min: i64, max: i64 },
    /// A whole number from 0 to `u64::MAX`.
    Seed,
    /// A finite number, at least `min` when given.
    Number { min: Option<f64> },
    /// A string; `nonempty` refuses one that is empty or only spaces.
    Text { nonempty: bool },
    /// A file or directory path: a non-empty string.
    Path,
    /// One of the listed strings.
    Choice(&'static [&'static str]),
    /// A list of strings; a flag spelling may be repeated, each adding one.
    Texts,
    /// A list of layer indices, none twice and at least one; a flag spelling takes `23,24,25`.
    Layers,
}

/// One option: one row of the table.
#[derive(Debug, Clone, Copy)]
pub struct Descriptor {
    /// The document key within its section. Unique across the table.
    pub name: &'static str,
    pub section: Section,
    pub phase: Phase,
    pub kind: Kind,
    /// The value in `Options::default()`, as JSON text. `null` is unset: the help says what unset means.
    pub default: &'static str,
    /// Command-line spellings.
    pub flags: &'static [&'static str],
    /// The environment variable, when there is one. Environment values are ambient ([`Source::Environment`]).
    pub env: Option<&'static str>,
    /// The Python binding's argument (`ggtts.load` or a render method).
    pub python: &'static [&'static str],
    /// The C ABI's argument or `ggtts_settings` field.
    pub c: &'static [&'static str],
    /// The Kotlin binding's parameter (`Ggtts.load` or a render method).
    pub kotlin: &'static [&'static str],
    /// Another option of the same section that must be given with this one.
    pub requires: Option<&'static str>,
    pub help: &'static str,
}

impl Descriptor {
    /// The schema name: `section.name`, as a [`Refusal`] and the record name it.
    pub fn key(&self) -> String {
        format!("{}.{}", self.section.name(), self.name)
    }

    /// The default as a JSON value.
    pub fn default_value(&self) -> Value {
        serde_json::from_str(self.default).expect("a table default is JSON")
    }
}

const NONE: &[&str] = &[];

const fn row(name: &'static str, section: Section, phase: Phase, kind: Kind, default: &'static str) -> Descriptor {
    Descriptor {
        name,
        section,
        phase,
        kind,
        default,
        flags: NONE,
        env: None,
        python: NONE,
        c: NONE,
        kotlin: NONE,
        requires: None,
        help: "",
    }
}

const fn path() -> Kind {
    Kind::Path
}

const fn text() -> Kind {
    Kind::Text { nonempty: false }
}

const fn named() -> Kind {
    Kind::Text { nonempty: true }
}

const fn seconds() -> Kind {
    Kind::Number { min: Some(0.0) }
}

use Phase::{Load as L, Render as R};
use Section::{Book as SB, Load as SL, Observe as SO, Output as SU, Render as SR};

/// Every option, one row each. The schema, the parsers and the record read this table.
pub const TABLE: &[Descriptor] = &[
    // ── load ──
    Descriptor {
        flags: &["--engine"],
        python: &["engine"],
        c: &["engine"],
        kotlin: &["engine"],
        help: "The engine: auto picks it from the model's architecture, and is qwen3-tts with no model.",
        ..row(
            "engine",
            SL,
            L,
            Kind::Choice(&["auto", "qwen3-tts", "kitten-tts-2", "fake"]),
            "\"auto\"",
        )
    },
    Descriptor {
        flags: &["--model"],
        python: &["model"],
        c: &["model"],
        kotlin: &["model"],
        help: "The model GGUF (Qwen3-TTS: the talker; Kitten-TTS 2: the LM); unset is the engine's pinned default.",
        ..row("model", SL, L, path(), "null")
    },
    Descriptor {
        flags: &["--codec"],
        python: &["codec"],
        c: &["codec"],
        kotlin: &["codec"],
        help: "The codec GGUF (Kitten-TTS 2: the S3Gen GGUF); unset is the engine's pinned default.",
        ..row("codec", SL, L, path(), "null")
    },
    Descriptor {
        flags: &["--voice"],
        python: &["voice"],
        c: &["voice"],
        kotlin: &["voice"],
        help: "A reference voice, for an engine that clones one (Kitten-TTS 2).",
        ..row("voice", SL, L, path(), "null")
    },
    Descriptor {
        flags: &["--speaker"],
        python: &["speaker"],
        c: &["speaker"],
        kotlin: &["speaker"],
        help: "A named Qwen3-TTS speaker; unset is the model's default speaker.",
        ..row("speaker", SL, L, named(), "null")
    },
    Descriptor {
        flags: &["--device"],
        python: &["device"],
        c: &["device"],
        help: "cpu or cuda; unset is cpu.",
        ..row("device", SL, L, Kind::Choice(&["cpu", "cuda"]), "null")
    },
    Descriptor {
        flags: &["--threads", "-j"],
        python: &["threads"],
        c: &["threads"],
        kotlin: &["threads"],
        help: "CPU threads asked for, capped at the CPUs this process may use; unset is 4.",
        ..row(
            "threads",
            SL,
            L,
            Kind::Int {
                min: 1,
                max: i32::MAX as i64,
            },
            "null",
        )
    },
    Descriptor {
        flags: &["--offline"],
        python: &["offline"],
        c: &["allow_download"],
        kotlin: &["allowDownload"],
        help: "Never download a default model file. The C ABI and Kotlin spell it negated, as allow_download.",
        ..row("offline", SL, L, Kind::Bool, "false")
    },
    Descriptor {
        env: Some("GGTTS_CACHE_DIR"),
        c: &["cache_dir"],
        kotlin: &["cacheDir"],
        help: "The root of the default model files' cache; unset is the platform's per-user cache.",
        ..row("cache_dir", SL, L, path(), "null")
    },
    // ── render ──
    Descriptor {
        flags: &["--text", "-t"],
        python: &["text"],
        c: &["text"],
        kotlin: &["text"],
        help: "The text to render, one paragraph per line.",
        ..row("text", SR, R, named(), "null")
    },
    Descriptor {
        flags: &["--story"],
        help: "A story file to render: a title line, then one paragraph per line.",
        ..row("story", SR, R, path(), "null")
    },
    Descriptor {
        flags: &["--seed"],
        help: "The sampler seed of a one-chunk plan only (ggtts-render --text); unset is 20260905.",
        ..row("seed", SR, R, Kind::Seed, "null")
    },
    Descriptor {
        python: &["seed"],
        c: &["seed"],
        kotlin: &["seed"],
        help: "The story seed derivation's salt, appended to every chunk's seed key; what each binding's seed is.",
        ..row("seed_salt", SR, R, text(), "null")
    },
    Descriptor {
        flags: &["--instruction"],
        python: &["instruction"],
        c: &["instruction"],
        kotlin: &["instruction"],
        help: "A style instruction for every chunk; empty is none. The bindings take it at load, as every render's.",
        ..row("instruction", SR, R, text(), "null")
    },
    Descriptor {
        flags: &["--steer"],
        help: "A steering file (safetensors, one F32 vector per talker layer).",
        ..row("steer", SR, R, path(), "null")
    },
    Descriptor {
        flags: &["--steer-scale"],
        requires: Some("steer"),
        help: "Multiplies every steering vector; unset is 1.0.",
        ..row("steer_scale", SR, R, Kind::Number { min: None }, "null")
    },
    Descriptor {
        flags: &["--steer-layers"],
        requires: Some("steer"),
        help: "The layers of the steering file to apply; unset is every layer in it.",
        ..row("steer_layers", SR, R, Kind::Layers, "null")
    },
    Descriptor {
        flags: &["--redraw"],
        help: "Redraws allowed for a chunk the judge rejects; unset is the engine profile's.",
        ..row(
            "redraw",
            SR,
            R,
            Kind::Int {
                min: 0,
                max: u32::MAX as i64,
            },
            "null",
        )
    },
    Descriptor {
        flags: &["--judge"],
        help: "An ASR that reads each draw back; a draw that lost or repeated a run of its text is redrawn: \
               asr-qwen,model=DIR[,device=cpu|cuda][,min_run=N]. Needs --redraw. Unset: the duration judge alone.",
        ..row("judge", SR, R, named(), "null")
    },
    Descriptor {
        help: "Seconds of silence kept at each edge of a chunk; unset is the engine profile's.",
        ..row("edge_s", SR, R, seconds(), "null")
    },
    Descriptor {
        help: "Collect wall-clock stage durations; also on whenever observe.timing_report is given.",
        ..row("measure_timings", SR, R, Kind::Bool, "false")
    },
    // ── observe ──
    Descriptor {
        flags: &["--trace"],
        help: "Where the trace (ggtts-trace/2 JSON lines) goes.",
        ..row("trace", SO, R, path(), "null")
    },
    Descriptor {
        flags: &["--trace-level"],
        requires: Some("trace"),
        help: "1 or 2 (2 adds every codebook); unset is 1.",
        ..row("trace_level", SO, R, Kind::Int { min: 1, max: 2 }, "null")
    },
    Descriptor {
        flags: &["--timing-report"],
        help: "A new file for the stage timing report.",
        ..row("timing_report", SO, R, path(), "null")
    },
    Descriptor {
        flags: &["--duration-report"],
        help: "A new file for the per-chunk duration report.",
        ..row("duration_report", SO, R, path(), "null")
    },
    Descriptor {
        flags: &["--log-level"],
        env: Some("GGTTS_LOG_LEVEL"),
        help: "error, warn, info, debug or trace; unset is info.",
        ..row(
            "log_level",
            SO,
            R,
            Kind::Choice(&["error", "warn", "info", "debug", "trace"]),
            "null",
        )
    },
    Descriptor {
        flags: &["--log-format"],
        env: Some("GGTTS_LOG_FORMAT"),
        help: "text or json; unset is text.",
        ..row("log_format", SO, R, Kind::Choice(&["text", "json"]), "null")
    },
    Descriptor {
        flags: &["--instruments"],
        help: "An instruments file (YAML); a flag always overrides it.",
        ..row("instruments", SO, R, path(), "null")
    },
    Descriptor {
        flags: &["--instrument"],
        help: "An instrument to run on the finished render, NAME[,KEY=VALUE]...; repeatable.",
        ..row("instrument", SO, R, Kind::Texts, "[]")
    },
    Descriptor {
        flags: &["--require"],
        help: "An instrument that must be ok for exit 0; repeatable.",
        ..row("require", SO, R, Kind::Texts, "[]")
    },
    Descriptor {
        flags: &["--on-instrument-error"],
        help: "continue or abort (skip the rest); unset is continue.",
        ..row(
            "on_instrument_error",
            SO,
            R,
            Kind::Choice(&["continue", "abort"]),
            "null",
        )
    },
    Descriptor {
        flags: &["--instrument-timeout"],
        help: "Seconds each instrument may run; unset is the instrument's own.",
        ..row("instrument_timeout", SO, R, seconds(), "null")
    },
    Descriptor {
        flags: &["--instrument-device"],
        help: "cpu, cuda or auto, for an instrument that runs a model; unset is auto.",
        ..row(
            "instrument_device",
            SO,
            R,
            Kind::Choice(&["cpu", "cuda", "auto"]),
            "null",
        )
    },
    Descriptor {
        flags: &["--report"],
        help: "The instruments' report directory; unset is the output's name with .report.",
        ..row("report", SO, R, path(), "null")
    },
    Descriptor {
        flags: &["--run"],
        help: "The run's identity in every envelope; unset is the output's name and the time.",
        ..row("run", SO, R, named(), "null")
    },
    // ── output ──
    Descriptor {
        flags: &["--out", "-o"],
        help: "The FLAC to write.",
        ..row("out", SU, R, path(), "null")
    },
    Descriptor {
        flags: &["--stream"],
        help: "Stream the audio as it is produced, to a file or - for stdout.",
        ..row("stream", SU, R, named(), "null")
    },
    Descriptor {
        flags: &["--stream-format"],
        requires: Some("stream"),
        help: "wav (plays as it arrives) or s16le after a one-line header (raw is s16le too); unset is wav.",
        ..row("stream_format", SU, R, Kind::Choice(&["wav", "s16le", "raw"]), "null")
    },
    Descriptor {
        flags: &["--stream-raw"],
        requires: Some("stream"),
        help: "Stream the pieces before the level step, bit-exact to the batch read.",
        ..row("stream_raw", SU, R, Kind::Bool, "false")
    },
    Descriptor {
        flags: &["--stream-level-seconds"],
        requires: Some("stream"),
        help: "Seconds of audio that set the levelled stream's gain; 0 is the first chunk alone; unset is 5.",
        ..row("stream_level_seconds", SU, R, seconds(), "null")
    },
    // ── book ──
    Descriptor {
        flags: &["--seed-root"],
        help: "The book's seed root: every chunk's seed is derived from it and the chunk's place in the book.",
        ..row("seed_root", SB, R, text(), "null")
    },
    Descriptor {
        flags: &["--include-skippable"],
        help: "Render the parts a reader may skip (front and back matter) as well.",
        ..row("include_skippable", SB, R, Kind::Bool, "false")
    },
];

/// The row named `key` (`section.name`).
pub fn descriptor(key: &str) -> Option<&'static Descriptor> {
    let (section, name) = key.split_once('.')?;
    TABLE.iter().find(|d| d.section.name() == section && d.name == name)
}

fn by_flag(flag: &str) -> Option<&'static Descriptor> {
    TABLE.iter().find(|d| d.flags.contains(&flag))
}

/// One command-line argument as the flag parser reads it: the flag as given, its row (`None` for a spelling the table
/// does not have), and its value as given (`None` for a boolean, for a flag the table does not have, and for a value
/// missing at the end).
#[derive(Debug, Clone, Copy)]
pub struct Arg<'a> {
    pub flag: &'a str,
    pub row: Option<&'static Descriptor>,
    pub value: Option<&'a str>,
}

/// The arguments as [`Options::from_args`] walks them, checking nothing: a row's flag takes the next argument as its
/// value unless the row is a boolean, and a flag the table does not have takes none. A front end reads it to find what
/// a refused command line asked for (the log format to report the refusal in, the value it refused), never to accept
/// a flag.
pub fn walk_args(args: &[String]) -> Vec<Arg<'_>> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let row = by_flag(flag);
        let value = match row {
            Some(d) if d.kind != Kind::Bool => it.next().map(String::as_str),
            _ => None,
        };
        out.push(Arg { flag, row, value });
    }
    out
}

fn in_section(section: Section, name: &str) -> Option<&'static Descriptor> {
    TABLE.iter().find(|d| d.section == section && d.name == name)
}

// ── the value ──

fn is_false(b: &bool) -> bool {
    !*b
}

/// The load section: what opens an engine. `ggtts_load::Request` is this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Load {
    pub engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec: Option<PathBuf>,
    /// A reference voice, for an engine that clones one (kitten-tts-2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<PathBuf>,
    /// A named speaker in a Qwen3-TTS talker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// `cpu` (the default) or `cuda`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// CPU threads; 4 when not given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<i32>,
    /// Never download a default file.
    #[serde(skip_serializing_if = "is_false")]
    pub offline: bool,
    /// The root of the default model files' cache; the per-user cache when not given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            engine: "auto".into(),
            model: None,
            codec: None,
            voice: None,
            speaker: None,
            device: None,
            threads: None,
            offline: false,
            cache_dir: None,
        }
    }
}

/// The render section: what one render is asked to do.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Render {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<PathBuf>,
    /// The sampler seed of a one-chunk plan only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The story seed derivation's salt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_salt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_layers: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redraw: Option<u32>,
    /// The ASR judge's spec (`ggtts_instruments::judge`); unset is the duration judge alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_s: Option<f64>,
    #[serde(skip_serializing_if = "is_false")]
    pub measure_timings: bool,
    /// The steering read from [`steer`](Self::steer) by [`Options::read_steering`]; never in a document.
    #[serde(skip)]
    pub steering: Option<Steering>,
}

/// The observe section: what watches a render (trace, reports, logging, instruments).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Observe {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_level: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_report: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_report: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruments: Option<PathBuf>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub instrument: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub require: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_instrument_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instrument_timeout: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instrument_device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
}

/// The output section: where the audio goes, and in what form.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Output {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_format: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    pub stream_raw: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_level_seconds: Option<f64>,
}

/// The book section: what only a book render takes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Book {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_root: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    pub include_skippable: bool,
}

/// Where a value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Flag,
    Document,
    Environment,
}

/// An environment value an engine without the option ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ignored {
    /// The option's schema name.
    pub key: String,
    /// The environment variable it came from.
    pub variable: String,
}

/// Where each value came from, and what was ignored. Provenance, not value: two options values with the same values
/// are equal wherever their values came from.
#[derive(Debug, Clone, Default)]
pub struct Marks {
    /// By schema name. A value with no mark is the default.
    pub sources: BTreeMap<String, Source>,
    pub ignored: Vec<Ignored>,
}

impl PartialEq for Marks {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Every option a front end takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub version: u32,
    pub load: Load,
    pub render: Render,
    pub observe: Observe,
    pub output: Output,
    pub book: Book,
    #[serde(skip)]
    pub marks: Marks,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            version: VERSION,
            load: Load::default(),
            render: Render::default(),
            observe: Observe::default(),
            output: Output::default(),
            book: Book::default(),
            marks: Marks::default(),
        }
    }
}

/// Why options were refused: the option by its schema name (`section.name`; a flag as given for an unknown flag;
/// `$` for a document that is not a JSON object), and what is wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub key: String,
    pub message: String,
}

impl Refusal {
    fn new(key: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.key, self.message)
    }
}

impl std::error::Error for Refusal {}

// ── checking values ──

fn describe(kind: Kind) -> String {
    match kind {
        Kind::Bool => "true or false".into(),
        Kind::Int { min, max } => format!("a whole number from {min} to {max}"),
        Kind::Seed => "a whole number from 0 to 18446744073709551615".into(),
        Kind::Number { min: Some(m) } => format!("a finite number, at least {m}"),
        Kind::Number { min: None } => "a finite number".into(),
        Kind::Text { nonempty: true } => "a string that is not empty".into(),
        Kind::Text { nonempty: false } => "a string".into(),
        Kind::Path => "a path: a string that is not empty".into(),
        Kind::Choice(list) => format!("one of {}", list.join(", ")),
        Kind::Texts => "a list of strings".into(),
        Kind::Layers => "a list of layer indices, at least one and none twice".into(),
    }
}

fn check_value(d: &Descriptor, v: &Value) -> Result<(), Refusal> {
    let ok = match d.kind {
        Kind::Bool => v.is_boolean(),
        Kind::Int { min, max } => v.as_i64().is_some_and(|n| (min..=max).contains(&n)),
        Kind::Seed => v.as_u64().is_some(),
        Kind::Number { min } => v.as_f64().is_some_and(|x| x.is_finite() && min.is_none_or(|m| x >= m)),
        Kind::Text { nonempty } => v.as_str().is_some_and(|s| !nonempty || !s.trim().is_empty()),
        Kind::Path => v.as_str().is_some_and(|s| !s.is_empty()),
        Kind::Choice(list) => v.as_str().is_some_and(|s| list.contains(&s)),
        Kind::Texts => v.as_array().is_some_and(|a| a.iter().all(Value::is_string)),
        Kind::Layers => v.as_array().is_some_and(|a| {
            let layers: Vec<u64> = a.iter().filter_map(Value::as_u64).collect();
            !a.is_empty()
                && layers.len() == a.len()
                && layers.iter().all(|&l| l <= u64::from(u32::MAX))
                && layers.iter().enumerate().all(|(i, l)| !layers[..i].contains(l))
        }),
    };
    if ok {
        Ok(())
    } else {
        Err(Refusal::new(d.key(), format!("must be {}, not {v}", describe(d.kind))))
    }
}

/// A flag or environment string as the option's JSON value.
fn from_text(d: &Descriptor, s: &str) -> Result<Value, Refusal> {
    let bad = || Refusal::new(d.key(), format!("must be {}, not {s:?}", describe(d.kind)));
    let v = match d.kind {
        Kind::Bool => match s {
            "true" | "1" => Value::Bool(true),
            "false" | "0" => Value::Bool(false),
            _ => return Err(bad()),
        },
        Kind::Int { .. } => s.parse::<i64>().map_err(|_| bad())?.into(),
        Kind::Seed => s.parse::<u64>().map_err(|_| bad())?.into(),
        // A NaN or an infinity has no JSON form and becomes null, which `check_value` below refuses by name.
        Kind::Number { .. } => s.parse::<f64>().map_err(|_| bad())?.into(),
        Kind::Text { .. } | Kind::Path | Kind::Choice(_) | Kind::Texts => Value::String(s.into()),
        Kind::Layers => crate::steer::parse_layers(s)
            .map_err(|e| Refusal::new(d.key(), e.0))?
            .into(),
    };
    if d.kind == Kind::Texts {
        check_value(d, &json!([v]))?;
    } else {
        check_value(d, &v)?;
    }
    Ok(v)
}

/// Checks a whole document and returns the keys it gives. `render_only` refuses a load-phase key.
fn check_document(doc: &Value, render_only: bool) -> Result<Vec<String>, Refusal> {
    let top = doc
        .as_object()
        .ok_or_else(|| Refusal::new("$", "an options document is a JSON object"))?;
    let mut given = Vec::new();
    for (k, v) in top {
        if k == "version" {
            if v.as_u64() != Some(u64::from(VERSION)) {
                return Err(Refusal::new("version", format!("must be {VERSION}, not {v}")));
            }
            continue;
        }
        let section = Section::by_name(k).ok_or_else(|| Refusal::new(k.clone(), "is not an options section"))?;
        let entries = v
            .as_object()
            .ok_or_else(|| Refusal::new(k.clone(), format!("must be an object, not {v}")))?;
        for (name, value) in entries {
            let key = format!("{k}.{name}");
            let d = in_section(section, name).ok_or_else(|| Refusal::new(key.clone(), "is not an option"))?;
            if render_only && d.phase == Phase::Load {
                return Err(Refusal::new(
                    key,
                    "is a load option, fixed when the engine is opened: give it when loading, not per render",
                ));
            }
            check_value(d, value)?;
            given.push(key);
        }
    }
    check_requires(top)?;
    Ok(given)
}

fn check_requires(top: &Map<String, Value>) -> Result<(), Refusal> {
    for d in TABLE {
        let Some(needed) = d.requires else { continue };
        let section = top.get(d.section.name()).and_then(Value::as_object);
        if section.is_some_and(|s| s.contains_key(d.name) && !s.contains_key(needed)) {
            return Err(Refusal::new(d.key(), format!("needs {}.{needed}", d.section.name())));
        }
    }
    Ok(())
}

impl Options {
    /// The value as a document. Unset options are absent: `false` booleans, empty lists and `None`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("the options value serializes")
    }

    fn from_checked(doc: Value, keys: Vec<String>, source: Source) -> Result<Self, Refusal> {
        let mut o: Options = serde_json::from_value(doc).map_err(|e| Refusal::new("$", e.to_string()))?;
        o.marks.sources = keys.into_iter().map(|k| (k, source)).collect();
        Ok(o)
    }

    /// Reads a whole options document (every section). Refuses JSON that is not an object, an unknown section or key,
    /// a value of the wrong type or out of range, a wrong `version`, and an option given without the one it requires.
    pub fn from_json(text: &str) -> Result<Self, Refusal> {
        Self::from_value(serde_json::from_str(text).map_err(|e| Refusal::new("$", format!("not JSON: {e}")))?)
    }

    /// [`from_json`](Self::from_json) for a parsed document.
    pub fn from_value(doc: Value) -> Result<Self, Refusal> {
        let keys = check_document(&doc, false)?;
        Self::from_checked(doc, keys, Source::Document)
    }

    /// Reads a render document: the options of one render on an engine already loaded. As
    /// [`from_json`](Self::from_json), and a load-phase option (`load.device`) is refused too.
    pub fn from_render_json(text: &str) -> Result<Self, Refusal> {
        let doc: Value = serde_json::from_str(text).map_err(|e| Refusal::new("$", format!("not JSON: {e}")))?;
        let keys = check_document(&doc, true)?;
        Self::from_checked(doc, keys, Source::Document)
    }

    /// Reads command-line arguments (without the program name) by the table's flag spellings. A flag the table does not
    /// have is refused under its own spelling. A list flag may be repeated; any other given twice keeps the last.
    pub fn from_args(args: &[String]) -> Result<Self, Refusal> {
        let mut doc = Map::new();
        for Arg { flag, row, value } in walk_args(args) {
            let d = row.ok_or_else(|| Refusal::new(flag, "is not an option"))?;
            let value = if d.kind == Kind::Bool {
                Value::Bool(true)
            } else {
                let text = value.ok_or_else(|| Refusal::new(d.key(), format!("{flag} needs a value")))?;
                from_text(d, text)?
            };
            let section = doc
                .entry(d.section.name())
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .expect("a section is an object");
            if d.kind == Kind::Texts {
                let list = section.entry(d.name).or_insert_with(|| json!([]));
                list.as_array_mut().expect("a list").push(value);
            } else {
                section.insert(d.name.into(), value);
            }
        }
        let doc = Value::Object(doc);
        let keys = check_document(&doc, false)?;
        Self::from_checked(doc, keys, Source::Flag)
    }

    /// Reads the options the table gives an environment spelling, through `var` (an empty value is unset).
    pub fn from_env(var: &dyn Fn(&str) -> Option<String>) -> Result<Self, Refusal> {
        let mut doc = Map::new();
        for d in TABLE {
            let Some(name) = d.env else { continue };
            let Some(text) = var(name).filter(|v| !v.is_empty()) else {
                continue;
            };
            let value = from_text(d, &text).map_err(|r| Refusal::new(r.key, format!("{name}: {}", r.message)))?;
            let value = if d.kind == Kind::Texts { json!([value]) } else { value };
            doc.entry(d.section.name())
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .expect("a section is an object")
                .insert(d.name.into(), value);
        }
        let doc = Value::Object(doc);
        let keys = check_document(&doc, false)?;
        Self::from_checked(doc, keys, Source::Environment)
    }

    /// [`from_env`](Self::from_env) on this process's environment.
    pub fn from_process_env() -> Result<Self, Refusal> {
        Self::from_env(&|name| std::env::var(name).ok())
    }

    /// `over` laid on this value: every value `over` marks replaces this one's, with its mark. So
    /// `Options::from_env(..)?.layer(Options::from_args(..)?)` is the environment with every flag winning over it.
    pub fn layer(self, over: Options) -> Result<Self, Refusal> {
        let mut base = self.to_json();
        let top = over.to_json();
        let mut marks = self.marks;
        for (key, source) in &over.marks.sources {
            let (section, name) = key.split_once('.').expect("a mark is section.name");
            let value = top[section].get(name).cloned();
            let entries = base[section].as_object_mut().expect("a section is an object");
            match value {
                Some(v) => entries.insert(name.into(), v),
                None => entries.remove(name),
            };
            marks.sources.insert(key.clone(), *source);
        }
        marks.ignored.extend(over.marks.ignored);
        check_requires(base.as_object().expect("an object"))?;
        let mut o: Options = serde_json::from_value(base).map_err(|e| Refusal::new("$", e.to_string()))?;
        o.marks = marks;
        o.render.steering = over.render.steering.or(self.render.steering);
        Ok(o)
    }

    /// Applies the rule for an option the engine lacks, to every value given: `has(key)` says whether the engine named
    /// `engine` has the option. A value from a flag or a document is a request for this render, and is refused. A
    /// value only from the environment is ambient: it is reset to the default, `warn` gets one line naming the
    /// variable, and the record lists it under `ignored`.
    pub fn apply_engine_rule(
        &mut self,
        engine: &str,
        has: &dyn Fn(&str) -> bool,
        warn: &mut dyn FnMut(&str),
    ) -> Result<(), Refusal> {
        let lacking: Vec<(String, Source)> = self
            .marks
            .sources
            .iter()
            .filter(|(k, _)| !has(k))
            .map(|(k, s)| (k.clone(), *s))
            .collect();
        if let Some((key, _)) = lacking.iter().find(|(_, s)| *s != Source::Environment) {
            return Err(Refusal::new(
                key.clone(),
                format!("the {engine} engine has no such option"),
            ));
        }
        if lacking.is_empty() {
            return Ok(());
        }
        let mut doc = self.to_json();
        for (key, _) in &lacking {
            let d = descriptor(key).expect("a mark names a row");
            let variable = d.env.expect("an environment value has a variable");
            doc[d.section.name()]
                .as_object_mut()
                .expect("a section is an object")
                .remove(d.name);
            warn(&format!(
                "ggtts: {variable} is set, and the {engine} engine has no {key}: ignored"
            ));
            self.marks.sources.remove(key);
            self.marks.ignored.push(Ignored {
                key: key.clone(),
                variable: variable.into(),
            });
        }
        let mut o: Options = serde_json::from_value(doc).map_err(|e| Refusal::new("$", e.to_string()))?;
        o.marks = std::mem::take(&mut self.marks);
        o.render.steering = self.render.steering.take();
        *self = o;
        Ok(())
    }

    /// Reads the steering file the render section names, with its layers and scale, into
    /// [`Render::steering`]. Nothing to do without one.
    pub fn read_steering(&mut self) -> Result<(), Refusal> {
        let Some(path) = &self.render.steer else {
            self.render.steering = None;
            return Ok(());
        };
        let file = SteerFile::read(path).map_err(|e| Refusal::new("render.steer", e.0))?;
        let steering = file
            .select(
                self.render.steer_layers.as_deref(),
                self.render.steer_scale.unwrap_or(1.0),
            )
            .map_err(|e| Refusal::new("render.steer", e.0))?;
        self.render.steering = Some(steering);
        Ok(())
    }

    /// The pipeline's options from the render section, with `profile`'s defaults where the section leaves them unset
    /// (as [`RenderOptions::for_profile`]). Timings are measured when the section asks or a timing report is wanted.
    /// Refuses a steering file not yet read ([`read_steering`](Self::read_steering)).
    pub fn render_options(&self, profile: &Profile) -> Result<RenderOptions, Refusal> {
        let r = &self.render;
        if r.steer.is_some() && r.steering.is_none() {
            return Err(Refusal::new(
                "render.steer",
                "the steering file has not been read: call Options::read_steering first",
            ));
        }
        let mut o = RenderOptions::for_profile(profile);
        if let Some(n) = r.redraw {
            o.redraw = n;
        }
        if let Some(salt) = &r.seed_salt {
            o.seed_salt.clone_from(salt);
        }
        if let Some(edge) = r.edge_s {
            o.edge_s = edge;
        }
        o.measure_timings = r.measure_timings || self.observe.timing_report.is_some();
        o.instruction.clone_from(&r.instruction);
        o.steer.clone_from(&r.steering);
        Ok(o)
    }
}

/// An options value an engine has been checked against ([`check`]): the value to render with, and the warning line
/// for each environment value the engine has no use for, which the value lists in its record as ignored.
#[derive(Debug, Clone)]
pub struct Checked {
    pub options: Options,
    pub warnings: Vec<String>,
}

/// The options of `o` the engine `caps` describes does not support, by schema key, in table order: a non-empty
/// instruction, steering, a speaker, and a device it does not run on.
fn lacking(o: &Options, caps: &Capabilities) -> Vec<&'static str> {
    let mut out = Vec::new();
    if o.load.speaker.is_some() && !caps.speakers {
        out.push("load.speaker");
    }
    if o.load.device.as_deref().is_some_and(|d| !caps.devices.contains(&d)) {
        out.push("load.device");
    }
    if o.render.instruction.as_deref().is_some_and(|i| !i.is_empty()) && !caps.instruction {
        out.push("render.instruction");
    }
    if o.render.steer.is_some() && !caps.steering {
        out.push("render.steer");
    }
    out
}

/// Checks `options` against what an engine supports (`caps`), before anything is loaded or rendered. An option the
/// engine does not support is refused with [`Capabilities::refusal`], under its schema key, when it was asked for: by
/// a flag, in a document, or set by the caller. One that came only from the environment is ambient: it is reset, one
/// warning line names its variable, and the record lists it as ignored.
pub fn check(options: &Options, caps: &Capabilities) -> Result<Checked, Refusal> {
    let mut o = options.clone();
    let mut warnings = Vec::new();
    for key in lacking(options, caps) {
        let d = descriptor(key).expect("a lacking key is a row");
        if o.marks.sources.get(key) != Some(&Source::Environment) {
            let value = match key {
                "load.device" => options.load.device.as_deref(),
                _ => None,
            };
            return Err(caps.refusal(key, value));
        }
        // Its variable; a row with no environment spelling cannot be marked so by the parsers, only by a caller.
        let variable = d.env.unwrap_or("the environment");
        let mut doc = o.to_json();
        doc[d.section.name()]
            .as_object_mut()
            .expect("a section is an object")
            .remove(d.name);
        let mut next: Options = serde_json::from_value(doc).map_err(|e| Refusal::new("$", e.to_string()))?;
        next.marks = std::mem::take(&mut o.marks);
        next.render.steering = o.render.steering.take();
        next.marks.sources.remove(key);
        next.marks.ignored.push(Ignored {
            key: key.to_string(),
            variable: variable.to_string(),
        });
        warnings.push(format!(
            "ggtts: {variable} is set, and the {} engine has no {key}: ignored",
            caps.family
        ));
        o = next;
    }
    Ok(Checked { options: o, warnings })
}

/// The provenance record of a render with `options` on the engine `info` describes: the engine, backend and parity;
/// the instruction (when not empty) and the steering record, as the trace header and every report write them; the
/// seed keys given; the speaker given; the device; the trace level (0 untraced); and the environment values the
/// engine ignored. A steering file not yet read is recorded by its path.
pub fn record(options: &Options, info: &EngineInfo) -> Value {
    let mut m = Map::new();
    m.insert("engine".into(), info.family.clone().into());
    m.insert("backend".into(), info.backend.clone().into());
    m.insert("parity".into(), info.parity.clone().into());
    let r = &options.render;
    if let Some(i) = r.instruction.as_ref().filter(|i| !i.is_empty()) {
        m.insert("instruction".into(), i.clone().into());
    }
    match (&r.steering, &r.steer) {
        (Some(s), _) => {
            m.insert("steer".into(), s.record());
        }
        (None, Some(path)) => {
            m.insert("steer".into(), json!({"file": path}));
        }
        (None, None) => {}
    }
    if let Some(seed) = r.seed {
        m.insert("seed".into(), seed.into());
    }
    if let Some(salt) = &r.seed_salt {
        m.insert("seed_salt".into(), salt.clone().into());
    }
    if let Some(root) = &options.book.seed_root {
        m.insert("seed_root".into(), root.clone().into());
    }
    if let Some(speaker) = &options.load.speaker {
        m.insert("speaker".into(), speaker.clone().into());
    }
    m.insert(
        "device".into(),
        options.load.device.clone().unwrap_or_else(|| "cpu".into()).into(),
    );
    let level = if options.observe.trace.is_some() {
        options.observe.trace_level.unwrap_or(1)
    } else {
        0
    };
    m.insert("trace_level".into(), level.into());
    if !options.marks.ignored.is_empty() {
        let ignored: Vec<Value> = options
            .marks
            .ignored
            .iter()
            .map(|i| json!({"key": i.key, "variable": i.variable}))
            .collect();
        m.insert("ignored".into(), ignored.into());
    }
    Value::Object(m)
}

/// The fields of a [`record`] that every `ggtts-render` report (timing, duration, each instrument's inputs) and its
/// trace header carry beside the model files: what the render was asked to do differently.
pub const REPORTED: [&str; 2] = ["instruction", "steer"];

/// The [`REPORTED`] fields of `record`, those it has. A report writes these, so it cannot disagree with the record.
pub fn reported(record: &Value) -> Map<String, Value> {
    REPORTED
        .iter()
        .filter_map(|k| record.get(*k).map(|v| ((*k).to_string(), v.clone())))
        .collect()
}

// ── the schema ──

fn kind_schema(kind: Kind) -> Map<String, Value> {
    let v = match kind {
        Kind::Bool => json!({"type": "boolean"}),
        Kind::Int { min, max } => json!({"type": "integer", "minimum": min, "maximum": max}),
        Kind::Seed => json!({"type": "integer", "minimum": 0, "maximum": u64::MAX}),
        Kind::Number { min: Some(m) } => json!({"type": "number", "minimum": m}),
        Kind::Number { min: None } => json!({"type": "number"}),
        Kind::Text { nonempty: true } => json!({"type": "string", "pattern": "\\S"}),
        Kind::Text { nonempty: false } => json!({"type": "string"}),
        Kind::Path => json!({"type": "string", "minLength": 1}),
        Kind::Choice(list) => json!({"enum": list}),
        Kind::Texts => json!({"type": "array", "items": {"type": "string"}}),
        Kind::Layers => json!({
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": u32::MAX},
            "minItems": 1,
            "uniqueItems": true
        }),
    };
    match v {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

/// The JSON Schema of an options document, generated from [`TABLE`]. Each property carries its phase and spellings as
/// `x-ggtts-*` annotations. `schemas/ggtts-options-1.schema.json` is this, pretty-printed.
pub fn schema() -> Value {
    let mut sections = Map::new();
    for section in Section::ALL {
        let mut properties = Map::new();
        let mut dependent = Map::new();
        for d in TABLE.iter().filter(|d| d.section == section) {
            let mut p = kind_schema(d.kind);
            p.insert("description".into(), d.help.into());
            let default = d.default_value();
            if !default.is_null() {
                p.insert("default".into(), default);
            }
            p.insert("x-ggtts-phase".into(), d.phase.name().into());
            let spellings: Vec<(&str, &[&str])> = vec![
                ("x-ggtts-flags", d.flags),
                ("x-ggtts-python", d.python),
                ("x-ggtts-c", d.c),
                ("x-ggtts-kotlin", d.kotlin),
            ];
            for (k, list) in spellings {
                if !list.is_empty() {
                    p.insert(k.into(), json!(list));
                }
            }
            if let Some(env) = d.env {
                p.insert("x-ggtts-env".into(), env.into());
            }
            if let Some(needed) = d.requires {
                dependent.insert(d.name.into(), json!([needed]));
            }
            properties.insert(d.name.into(), Value::Object(p));
        }
        let mut s = Map::new();
        s.insert("type".into(), "object".into());
        s.insert("additionalProperties".into(), false.into());
        s.insert("properties".into(), Value::Object(properties));
        if !dependent.is_empty() {
            s.insert("dependentRequired".into(), Value::Object(dependent));
        }
        sections.insert(section.name().into(), Value::Object(s));
    }
    let mut properties = Map::new();
    properties.insert("version".into(), json!({"const": VERSION}));
    properties.extend(sections);
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "ggtts options, version 1",
        "description": "Every option a ggtts front end takes, in sections. Generated from ggtts_core::options::TABLE; docs/options.md describes it.",
        "type": "object",
        "additionalProperties": false,
        "properties": properties
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_requires_name_a_row_of_the_same_section() {
        for (i, d) in TABLE.iter().enumerate() {
            assert!(TABLE[..i].iter().all(|e| e.name != d.name), "{} is named twice", d.name);
            assert!(!d.help.is_empty(), "{} has no help", d.key());
            if let Some(r) = d.requires {
                assert!(in_section(d.section, r).is_some(), "{} requires {r}", d.key());
            }
            for f in d.flags {
                assert_eq!(
                    TABLE.iter().filter(|e| e.flags.contains(f)).count(),
                    1,
                    "{f} is spelled twice"
                );
            }
            // Every default is a value the option allows, or unset.
            let v = d.default_value();
            let unset = v.is_null() || v == json!(false) || v == json!([]);
            assert!(unset || check_value(d, &v).is_ok(), "{}'s default {v}", d.key());
        }
    }

    #[test]
    fn every_load_section_row_is_load_phase_and_no_other_is() {
        for d in TABLE {
            assert_eq!(d.phase == Phase::Load, d.section == Section::Load, "{}", d.key());
        }
    }
}
