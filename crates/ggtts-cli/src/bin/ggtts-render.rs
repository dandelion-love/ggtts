//! ggtts-render: text in, FLAC out.
//!
//! ```text
//! ggtts-render [--model M.gguf] [--codec C.gguf] [--offline] --out x.flac [--engine auto|qwen3-tts|kitten-tts-2|fake]
//!              [--threads N] [--device cpu|cuda] [--trace trace.jsonl [--trace-level 1|2]] [--timing-report FILE] [--voice REF.json] [--speaker NAME]
//!              [--stream FILE|- [--stream-raw | --stream-level-seconds S] [--stream-format wav|s16le]] (--text "..." [--seed S] | --story story.txt)
//!              [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE.yaml] [--require NAME]...
//!              [--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort] [--instrument-timeout S] [--report DIR] [--run ID]
//! ```
//!
//! **Model files.** A path given with `--model` or `--codec` is used as it is. A file not given is the engine's
//! default (`ggtts_models`): Qwen3-TTS's talker and codec, pinned by sha256, downloaded into the per-user cache
//! (`GGTTS_CACHE_DIR` overrides where) the first time and verified before use. With no `--model` the engine is
//! qwen3-tts. `--offline` never downloads: a default file not already in the cache is an error naming its URL.
//!
//! `--voice` is a reference voice for an engine that clones one; an engine that does not ignores it.
//! `--speaker` selects a named Qwen3-TTS speaker (for example `hk_voice_343`); it is refused for other engines.
//! Without it the default talker uses its default speaker (`hk_voice_343`), and a given talker retains the legacy
//! `hk_cantonese_speaker` default; an absent or unknown speaker is an error.
//! `--engine auto` (the default) picks the engine from the model's `general.architecture`. `--text` renders one
//! request at one seed (default 20260905). `--story` reads a story file (title line, then one paragraph per line)
//! and renders it on the core's story path. For stories `--trace` writes a complete ggtts-trace/2 event stream;
//! tracing is read-only, so the FLAC is the same with or without it. Engines need the `ggml` feature (and
//! `--device cuda` the `cuda` feature); without it only `--version` and the fake engine work.
//!
//! `--stream` writes the audio as it is produced, to a file or `-` for stdout, as a WAV of unknown length (the
//! default) or raw s16le after a one-line header. A story streams chunk by chunk, each piece as soon as its chunk
//! is kept. Which stream:
//!
//! * **To listen: `--stream`.** Levelled with the level step's own rule: one gain, set from at least the first
//!   `--stream-level-seconds` of audio (default `stream::DEFAULT_HOLD_S`, 5 s; 0 takes the first chunk alone), and
//!   kept for the rest of the read (`stream::StreamLevel`). Audio is held back until that much is in hand, so a hold
//!   longer than the first chunk delays first audio; one no longer than the first chunk costs nothing. Its loudness
//!   is close to the FLAC's but not equal: the FLAC's gain is computed over the whole read, known only after the last
//!   chunk.
//! * **To check parity: `--stream --stream-raw`.** The pieces before the level step, bit-exact to the batch read
//!   before its level step; applying the FLAC's gain (printed on stderr at the end) gives the FLAC's samples.
//!
//! `--out` is optional with `--stream` and still writes the levelled FLAC. A `--text` render has no level step on
//! either output, so its stream holds the FLAC's samples, raw or not.
//!
//! `--engine fake` renders with the fake engine (a tone per chunk, no model), so the pipeline and the stream can be
//! run end to end without one. `--model` is not needed for it.
//!
//! The instrument flags (see `ggtts_instruments::config`) run instruments on the finished render, which needs
//! `--out`. Each runs in its own process, with its own log, a timeout and a crash boundary, and is passed the run's
//! identity (`--run`, default the output's name and the time) and the render's (the output's name). The report goes
//! to `--report` (default: the output's name with `.report`): one envelope per instrument, merged into `report.json`,
//! `.md` and `.html`, plus `run.log`. An unknown instrument is refused before rendering. The exit is 0 when the FLAC
//! is written and every `--require`d instrument is ok, 3 when the FLAC is written but the measurement is INCOMPLETE,
//! and 2 when the render failed. `--on-instrument-error abort` skips the instruments after a failure; it never
//! changes the FLAC or the exit code.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::flac::write_flac;
use ggtts_core::logging::{Format as LogFormat, Level as LogLevel, Settings as LogSettings};
use ggtts_core::pipeline::{plan_story, render_story_streamed, RenderOptions};
use ggtts_core::stream::{Container, StreamLevel, StreamWriter};
use ggtts_core::text::read_text;
use ggtts_core::trace::{emit_header, Event, Events, Header, JsonlEvents, NullEvents};

// The pipeline ends before a held stream is flushed. Delay the trace end marker until the CLI has recorded
// the actual first write; readers stop at that marker.
struct DeferredEnd<'a> {
    inner: &'a mut dyn Events,
    end: Option<(u64, f64)>,
}

impl Events for DeferredEnd<'_> {
    fn level(&self) -> u8 {
        self.inner.level()
    }
    fn emit(&mut self, event: Event<'_>) {
        if let Event::End { frames, seconds } = event {
            self.end = Some((frames, seconds));
        } else {
            self.inner.emit(event);
        }
    }
}

#[path = "../gguf_arch.rs"]
mod gguf_arch;

const BINARY: &str = "ggtts-render";

#[derive(Default)]
struct StreamTiming {
    first_audio_ms: Option<u128>,
    writes: u64,
    write_ns: u128,
}

impl StreamTiming {
    /// An empty PCM piece is not audio, even if a container could write a header for it.
    fn write<W: std::io::Write>(
        &mut self,
        writer: &mut StreamWriter<W>,
        samples: &[f32],
        rate: u32,
        render_start: Instant,
    ) -> std::io::Result<()> {
        let first = (!samples.is_empty() && self.first_audio_ms.is_none()).then(|| render_start.elapsed().as_millis());
        let start = Instant::now();
        writer.write(samples, rate)?;
        self.write_ns += start.elapsed().as_nanos();
        self.writes += 1;
        if self.first_audio_ms.is_none() && first.is_some() {
            self.first_audio_ms = first;
        }
        Ok(())
    }
}
const USAGE: &str = "usage: ggtts-render [--model M.gguf] [--codec C.gguf] [--offline] --out x.flac \
[--engine auto|qwen3-tts|kitten-tts-2|fake] [--threads N] [--device cpu|cuda] [--trace FILE [--trace-level 1|2]] [--voice REF] [--speaker NAME] \
[--stream FILE|- [--stream-raw | --stream-level-seconds S] [--stream-format wav|s16le]] (--text TEXT [--seed S] | --story FILE) \
[--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]... \
[--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort] [--instrument-timeout S] [--report DIR] [--run ID] \
[--log-level error|warn|info|debug|trace] [--log-format text|json] [--timing-report FILE] [--duration-report FILE]\n       ggtts-render --version";

/// Which stream to ask for, printed by `--help`.
const STREAMS: &str = concat!(
    "streams:\n",
    "  --stream FILE|-             to listen: levelled as it is produced, with one gain set from the first\n",
    "                              5 s of audio (held back until then)\n",
    "  --stream-level-seconds S    how much audio sets that gain: more is closer to the FLAC's loudness,\n",
    "                              less is earlier first audio; 0 = the first chunk alone (default 5)\n",
    "  --stream FILE --stream-raw  for parity checks: before the level step, bit-exact to the batch read;\n",
    "                              the FLAC is it times the gain printed at the end\n",
    "  --stream-format wav|s16le   wav (default: plays as it arrives) or raw s16le after a one-line header",
);

/// What `--help` says about the default model files.
const MODELS_HELP: &str =
    "\nmodel files: a path given with --model or --codec is used as it is. A file not given is the engine's \
default, pinned by sha256 and downloaded once into the per-user cache (GGTTS_CACHE_DIR overrides where). With no \
--model the engine is qwen3-tts and the speaker hk_voice_343. --offline never downloads.";

/// The engine a render with no `--model` uses: the one whose default files `ggtts_models` pins.
const DEFAULT_ENGINE: &str = "qwen3-tts";

/// Where `--help` says the script instruments are found.
const INSTRUMENTS_HELP: &str = "\ninstruments: --instrument NAME runs a shipped instrument on the finished render \
(instruments/instruments.yaml lists them).\nScript (Python) instruments are found in the instruments/ directory next to \
ggtts-render (a release archive), or the nearest one above it (a source checkout); GGTTS_INSTRUMENTS_DIR overrides \
that.\nThey need Python 3.8 or later: GGTTS_PYTHON, else python3 (python on Windows) on PATH.";

#[derive(Default)]
struct Args {
    model: Option<PathBuf>,
    codec: Option<PathBuf>,
    out: Option<PathBuf>,
    engine: Option<String>,
    text: Option<String>,
    story: Option<PathBuf>,
    seed: Option<u64>,
    threads: Option<i32>,
    device: Option<String>,
    trace: Option<PathBuf>,
    trace_level: Option<u8>,
    timing_report: Option<PathBuf>,
    duration_report: Option<PathBuf>,
    /// A reference voice, for engines that clone one (kitten-tts-2: a `ggtts-kitten prepare` JSON). Others ignore it.
    voice: Option<PathBuf>,
    /// Named speaker in a Qwen talker; distinct from a cloned reference voice.
    speaker: Option<String>,
    stream: Option<String>,
    stream_format: Option<String>,
    stream_raw: bool,
    stream_level_seconds: Option<f64>,
    instruments: ggtts_instruments::Flags,
    /// Where the instruments' report goes.
    report: Option<PathBuf>,
    /// The run's identity, for every envelope.
    run: Option<String>,
    /// Never download a default model file.
    offline: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        if ggtts_instruments::Flags::NAMES.contains(&flag.as_str()) {
            a.instruments.take(flag, value()?);
            continue;
        }
        match flag.as_str() {
            "--report" => a.report = Some(value()?.into()),
            "--run" => a.run = Some(value()?),
            "--model" => a.model = Some(value()?.into()),
            "--codec" => a.codec = Some(value()?.into()),
            "--offline" => a.offline = true,
            "--out" | "-o" => a.out = Some(value()?.into()),
            "--engine" => a.engine = Some(value()?),
            "--text" | "-t" => a.text = Some(value()?),
            "--story" => a.story = Some(value()?.into()),
            "--seed" => a.seed = Some(value()?.parse().map_err(|_| "--seed needs an integer")?),
            "--device" => a.device = Some(value()?),
            "--trace" => a.trace = Some(value()?.into()),
            "--timing-report" => a.timing_report = Some(value()?.into()),
            "--duration-report" => a.duration_report = Some(value()?.into()),
            "--trace-level" => {
                let level = value()?.parse::<u8>().map_err(|_| "--trace-level must be 1 or 2")?;
                if !(1..=2).contains(&level) {
                    return Err("--trace-level must be 1 or 2".into());
                }
                a.trace_level = Some(level);
            }
            "--voice" => a.voice = Some(value()?.into()),
            "--speaker" => {
                let speaker = value()?;
                if speaker.trim().is_empty() {
                    return Err("--speaker must not be empty".into());
                }
                a.speaker = Some(speaker);
            }
            "--stream" => a.stream = Some(value()?),
            "--stream-format" => a.stream_format = Some(value()?),
            "--stream-raw" => a.stream_raw = true,
            "--stream-level-seconds" => {
                let v: f64 = value()?
                    .parse()
                    .map_err(|_| "--stream-level-seconds needs a number of seconds")?;
                if !(v.is_finite() && v >= 0.0) {
                    return Err("--stream-level-seconds must be 0 or more".into());
                }
                a.stream_level_seconds = Some(v);
            }
            "--threads" | "-j" => a.threads = Some(value()?.parse().map_err(|_| "--threads needs an integer")?),
            other => return Err(format!("invalid argument: {other}")),
        }
    }
    Ok(a)
}

/// The model engines this binary can carry, by architecture name: `load_engine` has an arm for each, and
/// `engine_lines` lists them when the build has them. `every_engine_has_a_load_arm` holds the two together.
const ENGINES: &[&str] = &["qwen3-tts", "kitten-tts-2"];

#[cfg(feature = "ggml")]
fn qwen_voice(a: &Args) -> Result<ggtts_qwen3_tts::engine::Voice, String> {
    let device = a.device.as_deref().unwrap_or("cpu");
    let mut voice = ggtts_qwen3_tts::engine::Voice {
        device: ggtts_qwen3_tts::cpu::Device::parse(device)
            .ok_or(format!("--device must be cpu or cuda, not {device}"))?,
        // A story's --trace is the pipeline event stream, not the engine's legacy step-only file.
        trace: if a.story.is_some() { None } else { a.trace.clone() },
        ..Default::default()
    };
    if let Some(speaker) = &a.speaker {
        voice.speaker.clone_from(speaker);
    }
    Ok(voice)
}

/// The engine for a model, by architecture name. One arm per engine; an engine left out of the build is reported,
/// not hidden.
fn load_engine(name: &str, a: &Args, model: &Path) -> Result<Box<dyn Engine>, String> {
    let threads = a.threads.unwrap_or(4);
    match name {
        #[cfg(feature = "ggml")]
        "qwen3-tts" => {
            let codec = a.codec.as_ref().ok_or("the qwen3-tts engine needs --codec")?;
            let voice = qwen_voice(a)?;
            ggtts_qwen3_tts::engine::Qwen3Tts::load(model, codec, threads, voice)
                .map(|e| Box::new(e) as Box<dyn Engine>)
                .map_err(|e| e.to_string())
        }
        // Kitten-TTS 2: --model is the LM GGUF, --codec the S3Gen GGUF, --voice the reference (`ggtts-kitten prepare`).
        #[cfg(feature = "ggml")]
        "kitten-tts-2" => {
            let codec = a
                .codec
                .as_ref()
                .ok_or("the kitten-tts-2 engine needs --codec (the S3Gen GGUF)")?;
            let voice = a
                .voice
                .as_ref()
                .ok_or("the kitten-tts-2 engine needs --voice, a ggtts-kitten prepare JSON")?;
            let device = match a.device.as_deref().unwrap_or("cpu") {
                "cuda" => ggtts_kitten::backend::Device::Cuda,
                _ => ggtts_kitten::backend::Device::Cpu,
            };
            ggtts_kitten::engine::KittenEngine::open(model, codec, voice, threads as usize, device)
                .map(|e| Box::new(e) as Box<dyn Engine>)
        }
        #[cfg(not(feature = "ggml"))]
        name if ENGINES.contains(&name) => {
            let _ = (threads, model, &a.voice);
            Err("this build has no engines: rebuild with --features ggml".into())
        }
        "fake" => Ok(Box::new(ggtts_fake::FakeEngine::default())),
        other => Err(format!("no engine for architecture {other:?}")),
    }
}

/// The model files a render of engine `name` uses, in the order of its command-line paths (`--model`, `--codec`,
/// `--voice`). A given path is used as it is. A path not given is the engine's pinned default, from the cache,
/// downloaded first when missing (never with `--offline`). Fills the paths in `a`, and the default speaker when the
/// talker is the default one and no `--speaker` was given.
fn resolve_models(a: &mut Args, name: &str, log: LogSettings) -> Result<Vec<ggtts_models::ModelFile>, String> {
    let given = [a.model.clone(), a.codec.clone(), a.voice.clone()];
    let Some(defaults) = ggtts_models::defaults_for(name) else {
        if a.model.is_none() && name != "fake" {
            return Err(format!(
                "--model is required: the {name} engine has no default model files"
            ));
        }
        let roles = ["model", "codec", "voice"];
        return Ok(roles
            .iter()
            .zip(given)
            .filter_map(|(role, path)| path.map(|p| ggtts_models::ModelFile::given(role, p)))
            .collect());
    };
    let n = defaults.files.len();
    if given[..n].iter().all(Option::is_some) {
        return Ok(defaults
            .files
            .iter()
            .zip(given)
            .filter_map(|(p, path)| path.map(|path| ggtts_models::ModelFile::given(p.role, path)))
            .collect());
    }
    // Nothing is fetched for a build that cannot render it.
    if !cfg!(feature = "ggml") {
        return Err("this build has no engines: rebuild with --features ggml".into());
    }
    let cache = ggtts_models::Cache::new(ggtts_models::Cache::default_root()?);
    let files = ggtts_models::resolve(
        defaults,
        &given[..n],
        &cache,
        a.offline,
        &ggtts_models::Http::default(),
        &mut |m| log.stderr(LogLevel::Info, BINARY, "models", format_args!("{m}")),
    )?;
    if a.model.is_none() && a.speaker.is_none() {
        a.speaker = defaults.default_speaker.map(str::to_string);
    }
    let mut paths = files.iter().map(|f| Some(f.path.clone()));
    a.model = paths.next().flatten();
    a.codec = paths.next().flatten().or(a.codec.take());
    a.voice = paths.next().flatten().or(a.voice.take());
    Ok(files)
}

/// Where `--stream` writes: a file, or stdout for `-`.
fn open_stream(a: &Args) -> Result<Option<StreamWriter<Box<dyn std::io::Write>>>, String> {
    let Some(dest) = a.stream.as_deref() else {
        if a.stream_format.is_some() {
            return Err("--stream-format needs --stream".into());
        }
        if a.stream_raw {
            return Err("--stream-raw needs --stream".into());
        }
        if a.stream_level_seconds.is_some() {
            return Err("--stream-level-seconds needs --stream".into());
        }
        return Ok(None);
    };
    if a.stream_raw && a.stream_level_seconds.is_some() {
        return Err("--stream-level-seconds levels the stream, and --stream-raw does not: give one".into());
    }
    let fmt = a.stream_format.as_deref().unwrap_or("wav");
    let container = Container::parse(fmt).ok_or(format!("--stream-format must be wav or s16le, not {fmt}"))?;
    let w: Box<dyn std::io::Write> = if dest == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(std::fs::File::create(dest).map_err(|e| format!("{dest}: {e}"))?)
    };
    Ok(Some(StreamWriter::new(w, container)))
}

/// Render, then run the instruments. The exit code: 0, or 3 when a required instrument is not ok (the measurement is
/// INCOMPLETE; the FLAC is written).
fn render(a: Args, log: LogSettings) -> Result<u8, String> {
    #[cfg(feature = "ggml")]
    ggtts_qwen3_tts::native_logging::install(log).map_err(str::to_string)?;
    render_with_loader(a, log, load_engine)
}

fn render_with_loader(
    a: Args,
    log: LogSettings,
    loader: impl FnOnce(&str, &Args, &Path) -> Result<Box<dyn Engine>, String>,
) -> Result<u8, String> {
    render_with_story_reader(a, log, loader, |path| {
        std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
    })
}

/// `path` made absolute against the current directory with `.` and `..` resolved by name only, so two spellings of
/// a directory that does not exist yet compare equal. Symlinks are not followed.
fn lexical_absolute(path: &Path) -> PathBuf {
    let mut out = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_default()
    };
    for part in path.components() {
        match part {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

fn render_with_story_reader(
    a: Args,
    log: LogSettings,
    loader: impl FnOnce(&str, &Args, &Path) -> Result<Box<dyn Engine>, String>,
    read_story: impl FnOnce(&Path) -> Result<String, String>,
) -> Result<u8, String> {
    render_with_stream_opener(a, log, loader, read_story, open_stream)
}

fn render_with_stream_opener(
    mut a: Args,
    log: LogSettings,
    loader: impl FnOnce(&str, &Args, &Path) -> Result<Box<dyn Engine>, String>,
    read_story: impl FnOnce(&Path) -> Result<String, String>,
    stream_opener: impl FnOnce(&Args) -> Result<Option<StreamWriter<Box<dyn std::io::Write>>>, String>,
) -> Result<u8, String> {
    if a.trace_level.is_some() && (a.trace.is_none() || a.story.is_none()) {
        return Err("--trace-level requires a story and --trace".into());
    }
    if a.timing_report.as_ref().is_some_and(|p| p.exists()) {
        return Err("--timing-report must be a new file".into());
    }
    if a.duration_report
        .as_ref()
        .is_some_and(|p| std::fs::symlink_metadata(p).is_ok())
    {
        return Err("--duration-report must be a new file".into());
    }
    if a.out.is_none() && a.stream.is_none() {
        return Err("--out is required (or --stream)".into());
    }
    // Reject universally invalid input before probing the model's architecture.
    // Ceiling-dependent validation follows once the engine profile is known.
    let story_input = match (&a.text, &a.story) {
        (Some(text), None) => {
            if text.chars().all(ggtts_core::text::is_strip_space) {
                return Err("text is empty".into());
            }
            ggtts_core::text::validate_speech(text).map_err(|e| e.to_string())?;
            None
        }
        (None, Some(story)) => {
            let contents = read_story(story)?;
            let (title, paragraphs) = read_text(&contents);
            if paragraphs.is_empty() {
                return Err("the story has no chunk to render".into());
            }
            for paragraph in &paragraphs {
                ggtts_core::text::validate_speech(paragraph).map_err(|e| e.to_string())?;
            }
            Some((title, paragraphs))
        }
        _ => return Err("give exactly one of --text or --story".into()),
    };
    // Refuse a bad instrument config before spending the render on it.
    let instruments = if a.instruments.any() {
        if a.out.is_none() {
            return Err("instruments measure the --out FLAC: give --out".into());
        }
        let c = resolve_instrument_config(&a)?;
        ggtts_instruments::runner::check(&c)?;
        Some(c)
    } else {
        None
    };
    // Refuse a duration report that cannot be written before spending the render on it. The instrument report
    // directory is the one parent that may not exist yet: the runner creates it.
    if let Some(parent) = a
        .duration_report
        .as_ref()
        .and_then(|p| p.parent())
        .filter(|d| !d.as_os_str().is_empty() && !d.is_dir())
    {
        let report_dir = instruments.as_ref().and(a.out.as_ref()).map(|out| {
            a.report
                .clone()
                .unwrap_or_else(|| out.with_extension("flac").with_extension("report"))
        });
        if report_dir.as_deref().map(lexical_absolute) != Some(lexical_absolute(parent)) {
            return Err(format!("--duration-report: {} is not a directory", parent.display()));
        }
    }
    let name = match (a.engine.as_deref(), &a.model) {
        (None | Some("auto"), Some(model)) => {
            gguf_arch::architecture(model).map_err(|e| format!("{}: {e}", model.display()))?
        }
        // No model: the default engine, on its default files.
        (None | Some("auto"), None) => DEFAULT_ENGINE.to_string(),
        (Some(e), _) => e.to_string(),
    };
    if a.speaker.is_some() && name != "qwen3-tts" {
        return Err("--speaker is only supported by the qwen3-tts engine".into());
    }
    // Validate and plan before opening output files or loading model weights.
    let preplanned = if let Some((title, paragraphs)) = story_input {
        let profile = match name.as_str() {
            "fake" => *ggtts_fake::FakeEngine::default().profile(),
            "qwen3-tts" => ggtts_qwen3_tts::PROFILE,
            #[cfg(feature = "ggml")]
            "kitten-tts-2" => ggtts_kitten::engine::PROFILE,
            _ => return Err(format!("no available profile for architecture {name:?}")),
        };
        let chunks = plan_story(&profile, &paragraphs).map_err(|e| e.to_string())?;
        if chunks.is_empty() {
            return Err("the story has no chunk to render".into());
        }
        Some((title, chunks))
    } else {
        None
    };
    // Every check that needs no model has passed: only now is anything downloaded.
    let models = resolve_models(&mut a, &name, log)?;
    for m in &models {
        log.stderr(
            LogLevel::Info,
            BINARY,
            "model_file",
            format_args!("{} {} ({})", m.role, m.path.display(), m.source.as_str()),
        );
    }
    let model = a.model.clone().unwrap_or_default();
    let mut stream = stream_opener(&a)?;
    let t0 = Instant::now();
    let mut engine = loader(&name, &a, &model)?;
    let load_ms = t0.elapsed().as_millis();
    let t1 = Instant::now();
    let measured_render_ms;
    let frames_generated;
    let mut stage_times = None;
    let mut trace_stats = None;
    let mut stream_stats = None;
    let mut duration_chunks = None;
    let (samples, rate) = match (&a.text, &a.story) {
        (Some(text), None) => {
            let req = ChunkRequest {
                text,
                seed: a.seed.unwrap_or(20260905),
                chunk: 0,
                draw: 0,
            };
            let g = engine.generate(&req).map_err(|e| e.to_string())?;
            frames_generated = g.codes.n_frames() as u64;
            let gen_ns = t1.elapsed().as_nanos();
            let gen_ms = gen_ns / 1_000_000;
            let t2 = Instant::now();
            let pcm = engine.decode(&g.codes).map_err(|e| e.to_string())?;
            let decode_ns = t2.elapsed().as_nanos();
            if a.timing_report.is_some() {
                stage_times = Some(ggtts_core::pipeline::StageTimings {
                    generate_ns: gen_ns,
                    decode_ns,
                    judge_ns: 0,
                });
            }
            if let Some(w) = stream.as_mut() {
                let mut timing = StreamTiming::default();
                timing
                    .write(w, &pcm.samples, pcm.sample_rate, t1)
                    .map_err(|e| format!("--stream: {e}"))?;
                stream_stats = Some((timing.first_audio_ms, timing.writes, None, timing.write_ns));
            }
            measured_render_ms = t1.elapsed().as_millis();
            log.stderr(
                LogLevel::Info,
                BINARY,
                "draw",
                format_args!(
                    "{} frames, stop {:?}, generate {gen_ms} ms, decode {} ms",
                    g.codes.n_frames(),
                    g.stop,
                    decode_ns / 1_000_000
                ),
            );
            (pcm.samples, pcm.sample_rate)
        }
        (None, Some(_story)) => {
            let (title, chunks) = preplanned.expect("story was planned before loading");
            let profile = *engine.profile();
            let mut timing = StreamTiming::default();
            let mut level_ns = 0u128;
            let hold_s = a.stream_level_seconds.unwrap_or(ggtts_core::stream::DEFAULT_HOLD_S);
            let mut leveller = StreamLevel::new(&profile, hold_s);
            let raw = a.stream_raw;
            let trace_level = a.trace_level.unwrap_or(1);
            let mut trace = a
                .trace
                .as_ref()
                .map(|path| {
                    std::fs::File::create(path)
                        .map(|file| JsonlEvents::new(file, trace_level))
                        .map_err(|e| format!("--trace {}: {e}", path.display()))
                })
                .transpose()?;
            let mut null = NullEvents;
            let events: &mut dyn Events = match trace.as_mut() {
                Some(writer) => writer,
                None => &mut null,
            };
            let mut events = DeferredEnd {
                inner: events,
                end: None,
            };
            emit_header(&mut events, &Header::for_engine(engine.as_ref(), trace_level));
            let mut options = RenderOptions::for_profile(&profile);
            options.measure_timings = a.timing_report.is_some();
            let read = render_story_streamed(engine.as_mut(), &title, &chunks, &options, &mut events, &mut |p| {
                if let Some(w) = stream.as_mut() {
                    let started = Instant::now();
                    let ready = if raw {
                        vec![p.samples.to_vec()]
                    } else {
                        leveller.push(p.samples, p.sample_rate)
                    };
                    level_ns += started.elapsed().as_nanos();
                    for piece in ready {
                        timing.write(w, &piece, p.sample_rate, t1).map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            })
            .map_err(|e| e.to_string())?;
            if let Some(w) = stream.as_mut() {
                // A read shorter than the hold leaves at the end, levelled over all of it.
                let started = Instant::now();
                let ready = leveller.finish();
                level_ns += started.elapsed().as_nanos();
                for piece in ready {
                    timing
                        .write(w, &piece, read.sample_rate, t1)
                        .map_err(|e| format!("--stream: {e}"))?;
                }
            }
            let render_ms = t1.elapsed().as_millis();
            measured_render_ms = render_ms;
            frames_generated = read.frames_generated;
            stage_times = read.timings;
            let first_audio_ms = timing.first_audio_ms;
            let stream_writes = timing.writes;
            let stream_write_ns = timing.write_ns;
            if stream.is_some() {
                stream_stats = Some((first_audio_ms, stream_writes, Some(level_ns), stream_write_ns));
            }
            if events.level() > 0 {
                if let Some(ms) = first_audio_ms {
                    events.emit(Event::Timing {
                        stage: "first_audio",
                        chunk: None,
                        draw: None,
                        ms: ms as u64,
                    });
                }
                events.emit(Event::Timing {
                    stage: "render",
                    chunk: None,
                    draw: None,
                    ms: render_ms as u64,
                });
                if let Some((frames, seconds)) = events.end.take() {
                    events.inner.emit(Event::End { frames, seconds });
                }
            }
            if let Some(writer) = trace {
                let (_, stats) = writer.finish_with_stats().map_err(|e| format!("--trace: {e}"))?;
                trace_stats = Some((trace_level, stats));
                if log.enabled(LogLevel::Info) {
                    log.stderr_fields(LogLevel::Info, BINARY, "trace_sink", format_args!("trace sink {} events, {} ns serialization/write, {} ns flush; generation includes trace emission",
                        stats.events_written, stats.sink_ns, stats.flush_ns), serde_json::json!({"events":stats.events_written,
                        "sink_ns":stats.sink_ns as u64, "flush_ns":stats.flush_ns as u64, "trace_level":trace_level}));
                }
            }
            if let Some(w) = &stream {
                let db = |g: Option<f64>| g.map_or("none".to_string(), |g| format!("{:+.4} dB", 20.0 * g.log10()));
                let what = if raw {
                    "before the level step".to_string()
                } else {
                    format!(
                        "levelled by the gain of its first {:.1} s (--stream-level-seconds {hold_s}) {}",
                        leveller.held_seconds(),
                        db(leveller.gain())
                    )
                };
                log.stderr(
                    LogLevel::Info,
                    BINARY,
                    "stream",
                    format_args!(
                        "streamed {} samples {what}, first audio {}; the read's level gain {} \
                     (the --out FLAC applies it)",
                        w.samples(),
                        first_audio_ms.map_or_else(|| "not written".to_string(), |ms| format!("at {ms} ms")),
                        db(read.gain),
                    ),
                );
                if log.enabled(LogLevel::Info) {
                    log.stderr_fields(LogLevel::Info, BINARY, "stream_handoff", format_args!("stream handoffs {stream_writes} writes, {level_ns} ns preparing/levelling, {stream_write_ns} ns writing"),
                        serde_json::json!({"writes":stream_writes, "preparation_ns":level_ns as u64,
                                           "write_ns":stream_write_ns as u64, "first_audio_ms":first_audio_ms}));
                }
            }
            log.stderr(
                LogLevel::Info,
                BINARY,
                "story",
                format_args!(
                    "{} chunks, {} first-draw defects, {:.2} s of audio",
                    read.chunks.len(),
                    read.first_draw_defects(),
                    read.seconds()
                ),
            );
            if a.duration_report.is_some() || instruments.is_some() {
                duration_chunks = Some(
                    read.chunks
                        .iter()
                        .map(|c| {
                            let trimmed = c.span.1 - c.span.0;
                            let sr = c.decoded_sample_rate;
                            serde_json::json!({"chunk": c.i, "kept_draw": c.redraws,
                        "sample_rate": (sr > 0).then_some(sr),
                        "decoded_samples": c.decoded_samples,
                        "decoded_seconds": (sr > 0).then(|| c.decoded_samples as f64 / f64::from(sr)),
                        "trimmed_samples": trimmed,
                        "trimmed_seconds": (sr > 0).then(|| trimmed as f64 / f64::from(sr))})
                        })
                        .collect::<Vec<_>>(),
                );
            }
            (read.samples, read.sample_rate)
        }
        _ => return Err("give exactly one of --text and --story".into()),
    };
    let render_ms = measured_render_ms;
    let duration_data = if a.duration_report.is_some() || instruments.is_some() {
        let story = a.story.is_some();
        let chunk_samples = duration_chunks
            .as_ref()
            .map(|chunks| chunks.iter().filter_map(|c| c["trimmed_samples"].as_u64()).sum::<u64>());
        let joins = chunk_samples.and_then(|n| (samples.len() as u64).checked_sub(n));
        Some(serde_json::json!({
            "format": "ggtts-duration/1", "path": if story { "story" } else { "text" },
            "chunks": duration_chunks,
            "discarded_draws_included": false,
            "output_sample_rate": (rate > 0).then_some(rate), "output_samples": samples.len(),
            "output_seconds": (rate > 0).then(|| samples.len() as f64 / f64::from(rate)),
            "join_samples": joins,
            "join_seconds": joins.and_then(|n| (rate > 0).then(|| n as f64 / f64::from(rate))),
            "text_decoded_samples": (!story).then_some(samples.len()),
            "text_decoded_seconds": (!story && rate > 0).then(|| samples.len() as f64 / f64::from(rate)),
            "rate_note": "Chunk seconds use each decoder rate. The pipeline concatenates without resampling and outputs at the first chunk rate; differing rates change playback duration. Output and join seconds use the output rate."
        }))
    } else {
        None
    };
    // Instruments run after rendering. Release model-owned resources before loading
    // any GPU-backed instrument, while retaining provenance for logs and timing.
    let engine_info = engine.info().clone();
    drop(engine);
    let mut written = None;
    let mut artifact_write_ns = None;
    if let Some(out) = &a.out {
        let started = Instant::now();
        let path = write_flac(out, &samples, rate).map_err(|e| format!("{}: {e}", out.display()))?;
        artifact_write_ns = Some(started.elapsed().as_nanos());
        if log.enabled(LogLevel::Info) {
            log.stderr_fields(
                LogLevel::Info,
                BINARY,
                "render_complete",
                format_args!(
            "wrote {} ({} samples at {rate} Hz) with {} on {}, parity {}; load {load_ms} ms, render {render_ms} ms",
            path.display(),
            samples.len(),
            engine_info.family,
            engine_info.backend,
            engine_info.parity
        ),
                serde_json::json!({"samples":samples.len(), "sample_rate":rate, "load_ms":load_ms as u64,
                            "render_ms":render_ms as u64, "engine":engine_info.family,
                            "backend":engine_info.backend, "parity":engine_info.parity}),
            );
        }
        written = Some(path);
    }
    let write_duration = |path: &Path, data: &serde_json::Value| -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("duration report {}: {e}", path.display()))?;
        writeln!(file, "{data}").map_err(|e| format!("duration report {}: {e}", path.display()))
    };
    let write_timing = |instrument_ms: Option<u128>| -> Result<(), String> {
        let Some(path) = &a.timing_report else {
            return Ok(());
        };
        use std::io::Write;
        let (log_records, log_sink_ns, log_lock_wait_ns) = ggtts_core::logging::stats();
        let stage = stage_times.map(|t| serde_json::json!({
            "generate_ns": t.generate_ns as u64, "decode_ns": t.decode_ns as u64,
            "judge_ns": a.story.as_ref().map(|_| t.judge_ns as u64),
            "generated_frames_per_second": (t.generate_ns > 0).then(|| frames_generated as f64 * 1e9 / t.generate_ns as f64)
        }));
        let trace = trace_stats.map(|(level, s): (u8, ggtts_core::trace::TraceSinkStats)| {
            serde_json::json!({
                "level": level, "events": s.events_written, "sink_ns": s.sink_ns as u64, "flush_ns": s.flush_ns as u64
            })
        });
        let stream = stream_stats.map(
            |(first, writes, level_ns, write_ns): (Option<u128>, u64, Option<u128>, u128)| {
                serde_json::json!({
                    "first_audio_ms": first.map(|v| v as u64), "writes": writes,
                    "preparation_ns": level_ns.map(|v| v as u64), "write_ns": write_ns as u64
                })
            },
        );
        let audio_seconds = (rate > 0).then(|| samples.len() as f64 / f64::from(rate));
        // Every model file the render used: a cached file by its verified pin, a given one hashed now.
        let model_files = models
            .iter()
            .map(ggtts_models::ModelFile::report)
            .collect::<Result<Vec<_>, _>>()?;
        let report = serde_json::json!({
            "models": model_files,
            "format": "ggtts-timing/1", "engine": engine_info.family, "backend": engine_info.backend,
            "parity": engine_info.parity, "load_ms": load_ms as u64, "render_ms": render_ms as u64,
            "audio_seconds": audio_seconds, "frames_generated": frames_generated, "stages": stage,
            "render_rtf": audio_seconds.filter(|s| *s > 0.0).map(|s| render_ms as f64 / 1000.0 / s),
            "stream": stream, "trace_sink": trace, "artifact_write_ns": artifact_write_ns.map(|v| v as u64),
            "instrument_phase_ms": instrument_ms.map(|v| v as u64),
            "logging": {"level": log.level.name(), "format": log.format.name(), "records": log_records,
                        "sink_ns": log_sink_ns, "lock_wait_ns": log_lock_wait_ns},
            "limits": "generation is host wall time including observer callbacks; no exclusive GPU or asynchronous synchronization claim; timings overlap; logger counters omit caller preparation and external child logs"
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("--timing-report {}: {e}", path.display()))?;
        writeln!(file, "{report}").map_err(|e| format!("--timing-report {}: {e}", path.display()))
    };
    let (Some(config), Some(path)) = (instruments, written) else {
        if let (Some(path), Some(data)) = (&a.duration_report, &duration_data) {
            write_duration(path, data)?;
        }
        write_timing(None)?;
        return Ok(ggtts_instruments::runner::EXIT_OK);
    };
    let render_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("render")
        .to_string();
    let run_id = a.run.clone().unwrap_or_else(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("{render_id}-{secs}")
    });
    let inputs = ggtts_instruments::Inputs {
        run: run_id,
        render: render_id,
        report_dir: a.report.clone().unwrap_or_else(|| path.with_extension("report")),
        audio: path.clone(),
        audio_state: if a.story.is_some() {
            ggtts_core::report::AudioState::Trimmed
        } else {
            ggtts_core::report::AudioState::Raw
        },
        audio_seconds: (rate > 0).then(|| samples.len() as f64 / f64::from(rate)),
        events: a.trace.clone(),
        bin_dir: None,
        instruments_dir: None,
    };
    let summary = ggtts_instruments::runner::run_with_logging(&config, &inputs, log)?;
    if let Some(data) = &duration_data {
        // The report directory's own files are replaced on a rerun, as the runner replaces its run log and
        // instrument reports there. Only an explicit --duration-report must be a new file.
        let file = inputs.report_dir.join("duration.json");
        std::fs::write(&file, format!("{data}\n")).map_err(|e| format!("duration report {}: {e}", file.display()))?;
        // Resolve after the runner creates its directory, including path aliases.
        if let Some(explicit) = &a.duration_report {
            let same_file = std::fs::canonicalize(explicit)
                .ok()
                .zip(std::fs::canonicalize(&file).ok())
                .is_some_and(|(a, b)| a == b);
            if !same_file {
                write_duration(explicit, data)?;
            }
        }
        let md = inputs.report_dir.join("duration.md");
        std::fs::write(&md, format!("# Audio durations\n\nKept decoded and trimmed chunk durations, join contribution, and emitted duration are recorded below and in [duration.json](duration.json). Rejected redraws are excluded.\n\n```json\n{data}\n```\n"))
            .map_err(|e| format!("{}: {e}", md.display()))?;
        log.stderr(
            LogLevel::Info,
            BINARY,
            "duration_report",
            format_args!(
                "kept decoded, trimmed, join and emitted durations: {} and {}",
                file.display(),
                md.display()
            ),
        );
    }
    for e in &summary.envelopes {
        let why = e.error.as_ref().map_or(String::new(), |w| format!(": {w}"));
        let level = match e.status {
            ggtts_core::report::Status::Ok => LogLevel::Info,
            ggtts_core::report::Status::Failed
            | ggtts_core::report::Status::Timeout
            | ggtts_core::report::Status::Invalid => LogLevel::Error,
            ggtts_core::report::Status::Skipped | ggtts_core::report::Status::Truncated => LogLevel::Warn,
        };
        log.stderr(
            level,
            BINARY,
            "instrument",
            format_args!("instrument {} {}: {}{why}", e.name, e.version, e.status.as_str()),
        );
    }
    log.stderr(
        LogLevel::Info,
        BINARY,
        "instrument_report",
        format_args!(
            "instrument report in {}; runner {} ms (budget {} ms)",
            inputs.report_dir.display(),
            summary.overhead.own_ms,
            summary.overhead.budget_ms
        ),
    );
    if !summary.complete() {
        log.stderr(
            LogLevel::Error,
            BINARY,
            "measurement_incomplete",
            format_args!("measurement INCOMPLETE: {}", summary.incomplete.join("; ")),
        );
    }
    write_timing(Some(summary.overhead.total_ms))?;
    Ok(summary.exit_code())
}

fn resolve_instrument_config(a: &Args) -> Result<ggtts_instruments::Config, String> {
    let mut config = a.instruments.resolve()?;
    if config.device.is_none() {
        config.set_device(a.device.as_deref().unwrap_or("cpu"))?;
    }
    Ok(config)
}

/// The model engines compiled into this binary, one line each, with the backends each can run on. Read from the
/// build's features, not written by hand, so a build that dropped an engine says so (`scripts/buildinfo_engines.sh`
/// turns these lines into the release's BUILDINFO list and refuses an empty or short one). The fake engine is a test
/// instrument, always present, and not listed.
fn engine_lines() -> Vec<String> {
    let mut backends = Vec::new();
    if cfg!(feature = "ggml") {
        backends.push("cpu");
    }
    if cfg!(feature = "cuda") {
        backends.push("cuda");
    }
    let engines: &[&str] = if cfg!(feature = "ggml") { ENGINES } else { &[] };
    if engines.is_empty() {
        return vec!["engines: none (built without the ggml feature)".into()];
    }
    engines
        .iter()
        .map(|e| format!("engine {e} backends {}", backends.join(",")))
        .collect()
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (log, args) = match LogSettings::from_args(&raw) {
        Ok(parsed) => parsed,
        Err(e) => {
            e.settings.stderr(
                LogLevel::Error,
                "error",
                "logging_config",
                format_args!("{}", e.message),
            );
            return ExitCode::from(ggtts_instruments::runner::EXIT_RENDER_FAILED);
        }
    };
    match args.first().map(String::as_str) {
        None | Some("--version") | Some("-V") => {
            for line in ggtts_core::version_lines(BINARY) {
                println!("{line}");
            }
            for line in engine_lines() {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Some("-h") | Some("--help") => {
            println!("{USAGE}\n\n{STREAMS}");
            println!("{MODELS_HELP}");
            println!("{INSTRUMENTS_HELP}");
            ExitCode::SUCCESS
        }
        Some(_) => match parse(&args).and_then(|a| render(a, log)) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                log.stderr(LogLevel::Error, "error", "command_failed", format_args!("{e}"));
                if log.format == LogFormat::Text {
                    eprintln!("{USAGE}");
                }
                ExitCode::from(ggtts_instruments::runner::EXIT_RENDER_FAILED)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn instrument_device_inherits_render_and_respects_override() {
        let args = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let parsed = super::parse(&args(&["--device", "cuda"])).unwrap();
        assert_eq!(
            super::resolve_instrument_config(&parsed).unwrap().device.as_deref(),
            Some("cuda")
        );
        let parsed = super::parse(&args(&["--device", "cuda", "--instrument-device", "cpu"])).unwrap();
        assert_eq!(
            super::resolve_instrument_config(&parsed).unwrap().device.as_deref(),
            Some("cpu")
        );
        let parsed = super::parse(&args(&[])).unwrap();
        assert_eq!(
            super::resolve_instrument_config(&parsed).unwrap().device.as_deref(),
            Some("cpu")
        );
    }
    use super::*;
    use ggtts_core::engine::{Codes, EngineError, EngineInfo, Generated, Pcm};
    use ggtts_core::profile::Profile;

    #[test]
    fn engine_is_dropped_before_instrument_child_and_timing_keeps_metadata() {
        struct MarkedEngine {
            inner: ggtts_fake::FakeEngine,
            marker: PathBuf,
        }
        impl Drop for MarkedEngine {
            fn drop(&mut self) {
                std::fs::write(&self.marker, b"dropped").unwrap();
            }
        }
        impl Engine for MarkedEngine {
            fn info(&self) -> &EngineInfo {
                self.inner.info()
            }
            fn profile(&self) -> &Profile {
                self.inner.profile()
            }
            fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
                self.inner.generate(req)
            }
            fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
                self.inner.decode(codes)
            }
        }
        let root = std::env::temp_dir().join(format!("ggtts-engine-drop-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("instrument.py");
        let marker = root.join("dropped");
        let sdk = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments/python");
        std::fs::write(
            &script,
            format!(
                "import sys,pathlib\nsys.path.insert(0,{:?})\nfrom ggtts_instrument import run\ndef measure(c,e):\n assert pathlib.Path(c.config['marker']).read_text()=='dropped'\n e.metric('checked',1,'1','neutral',render=c.render)\nsys.exit(run(measure,'lifecycle','1',hooks=[]))\n",
                sdk.to_string_lossy()
            ),
        )
        .unwrap();
        let python = std::env::var("GGTTS_PYTHON").unwrap_or_else(|_| {
            if cfg!(windows) {
                "python".into()
            } else {
                "python3".into()
            }
        });
        let timing = root.join("timing.json");
        let mut args = Args {
            engine: Some("fake".into()),
            text: Some("hello".into()),
            out: Some(root.join("out.flac")),
            report: Some(root.join("report")),
            timing_report: Some(timing.clone()),
            ..Args::default()
        };
        args.instruments.take(
            "--instrument",
            format!(
                "lifecycle,command={},interpreter={python},version=1,arg.marker={}",
                script.display(),
                marker.display()
            ),
        );
        args.instruments.take("--require", "lifecycle".into());
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        let code = render_with_loader(args, log, move |_, _, _| {
            Ok(Box::new(MarkedEngine {
                inner: ggtts_fake::FakeEngine::default(),
                marker,
            }))
        })
        .unwrap();
        assert_eq!(code, 0);
        let report: serde_json::Value = serde_json::from_slice(&std::fs::read(timing).unwrap()).unwrap();
        assert_eq!(report["engine"], "fake");
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(feature = "ggml")]
    #[test]
    fn qwen_voice_uses_explicit_speaker_or_legacy_default() {
        assert_eq!(qwen_voice(&Args::default()).unwrap().speaker, "hk_cantonese_speaker");
        let args = Args {
            speaker: Some("hk_voice_343".into()),
            ..Args::default()
        };
        assert_eq!(qwen_voice(&args).unwrap().speaker, "hk_voice_343");
        let args = Args {
            speaker: Some("hk_voice_777".into()),
            story: Some("story".into()),
            ..Args::default()
        };
        assert_eq!(qwen_voice(&args).unwrap().speaker, "hk_voice_777");
    }

    #[test]
    fn speaker_selection_is_qwen_only_and_reaches_the_loader() {
        let a = parse(&["--speaker".into(), "hk_voice_343".into()]).unwrap();
        assert_eq!(a.speaker.as_deref(), Some("hk_voice_343"));
        assert!(parse(&["--speaker".into(), "  ".into()]).is_err());
        assert!(parse(&["--speaker".into()]).is_err());
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        let args = Args {
            engine: Some("qwen3-tts".into()),
            model: Some("unused".into()),
            // Both paths given, so nothing is resolved from the cache.
            codec: Some("unused-codec".into()),
            text: Some("hello".into()),
            out: Some("unused.flac".into()),
            speaker: a.speaker,
            ..Args::default()
        };
        let err = render_with_loader(args, log, |name, a, _| {
            assert_eq!(name, "qwen3-tts");
            assert_eq!(a.speaker.as_deref(), Some("hk_voice_343"));
            Err("loader reached".into())
        })
        .unwrap_err();
        assert_eq!(err, "loader reached");
        let args = Args {
            engine: Some("fake".into()),
            text: Some("hello".into()),
            out: Some("unused.flac".into()),
            speaker: Some("hk_voice_343".into()),
            ..Args::default()
        };
        let err = render_with_loader(args, log, |_, _, _| panic!("must reject before loading")).unwrap_err();
        assert!(err.contains("only supported by the qwen3-tts engine"));
    }

    struct EmptyThenAudio {
        inner: ggtts_fake::FakeEngine,
        profile: Profile,
        calls: usize,
        all_empty: bool,
    }

    impl EmptyThenAudio {
        fn new(all_empty: bool) -> Self {
            let inner = ggtts_fake::FakeEngine::default();
            let profile = Profile {
                min_chunk_chars: 0,
                redraw: 0,
                pauses: &[],
                default_pause_s: 0.0,
                para_extra_s: 0.0,
                ..*inner.profile()
            };
            Self {
                inner,
                profile,
                calls: 0,
                all_empty,
            }
        }
    }

    impl Engine for EmptyThenAudio {
        fn info(&self) -> &EngineInfo {
            self.inner.info()
        }
        fn profile(&self) -> &Profile {
            &self.profile
        }
        fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
            self.inner.generate(req)
        }
        fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
            self.calls += 1;
            if self.all_empty || self.calls == 1 {
                Ok(Pcm {
                    samples: Vec::new(),
                    sample_rate: 24_000,
                })
            } else {
                self.inner.decode(codes)
            }
        }
    }

    #[test]
    fn story_read_delay_is_outside_render_timing() {
        let root = std::env::temp_dir().join(format!("ggtts-story-clock-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let story = root.join("story.txt");
        std::fs::write(&story, "Title\nA short sentence.\n").unwrap();
        let report = root.join("timing.json");
        let args = Args {
            engine: Some("fake".into()),
            story: Some(story),
            stream: Some(root.join("stream.wav").to_string_lossy().into_owned()),
            stream_raw: true,
            timing_report: Some(report.clone()),
            ..Args::default()
        };
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        let wall = Instant::now();
        let code = render_with_story_reader(
            args,
            log,
            |_, _, _| Ok(Box::new(ggtts_fake::FakeEngine::default())),
            |path| {
                std::thread::sleep(std::time::Duration::from_millis(1200));
                std::fs::read_to_string(path).map_err(|e| e.to_string())
            },
        )
        .unwrap();
        let elapsed = wall.elapsed().as_millis();
        assert_eq!(code, 0);
        let timing: serde_json::Value = serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
        let render_ms = u128::from(timing["render_ms"].as_u64().unwrap());
        assert!(timing["stream"]["first_audio_ms"].is_number());
        assert!(
            elapsed.saturating_sub(render_ms) >= 900,
            "story read delay was included in render timing"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn slow_stream_handoff_is_in_the_cli_timing_report_without_changing_output() {
        use std::io::Write;
        use std::sync::{Arc, Mutex};
        use std::time::Duration;

        struct Capture {
            bytes: Arc<Mutex<Vec<u8>>>,
            delay: Option<Duration>,
        }
        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if let Some(delay) = self.delay.take() {
                    std::thread::sleep(delay);
                }
                self.bytes.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let root = std::env::temp_dir().join(format!("ggtts-handoff-control-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let run = |name: &str, delay: Option<Duration>| {
            let timing_path = root.join(format!("{name}-timing.json"));
            let trace_path = root.join(format!("{name}-trace.jsonl"));
            let args = Args {
                engine: Some("fake".into()),
                story: Some(root.join("story.txt")),
                stream: Some("in-memory".into()),
                stream_raw: true,
                timing_report: Some(timing_path.clone()),
                trace: Some(trace_path.clone()),
                trace_level: Some(2),
                ..Args::default()
            };
            let log = LogSettings {
                level: LogLevel::Error,
                format: LogFormat::Text,
            };
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let capture = bytes.clone();
            let code = render_with_stream_opener(
                args,
                log,
                |_, _, _| Ok(Box::new(ggtts_fake::FakeEngine::default())),
                |_| Ok("Title\nA short sentence.\nAnother sentence.\n".into()),
                move |_| {
                    Ok(Some(StreamWriter::new(
                        Box::new(Capture { bytes: capture, delay }),
                        Container::RawS16,
                    )))
                },
            )
            .unwrap();
            assert_eq!(code, 0);
            let timing: serde_json::Value = serde_json::from_slice(&std::fs::read(timing_path).unwrap()).unwrap();
            let tokens: Vec<_> = std::fs::read_to_string(trace_path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .filter(|event| event["ev"] == "step")
                .map(|event| (event["token"].clone(), event["cb"].clone()))
                .collect();
            let audio = bytes.lock().unwrap().clone();
            (timing, tokens, audio)
        };
        let (baseline, baseline_tokens, baseline_audio) = run("baseline", None);
        let (slow, slow_tokens, slow_audio) = run("slow", Some(Duration::from_millis(150)));
        assert!(!baseline_audio.is_empty());
        assert!(!baseline_tokens.is_empty());
        assert_eq!(baseline_audio, slow_audio);
        assert_eq!(baseline_tokens, slow_tokens);
        assert!(baseline["stream"]["write_ns"].is_number());
        assert!(slow["stream"]["write_ns"].as_u64().unwrap() >= 150_000_000);
        assert!(slow["render_ms"].as_u64().unwrap() >= 150);
        assert!(slow["stream"]["writes"].as_u64().unwrap() > 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_audio_requires_a_successful_nonempty_pcm_write() {
        for container in [Container::Wav, Container::RawS16] {
            let mut writer = StreamWriter::new(Vec::new(), container);
            let mut timing = StreamTiming::default();
            let start = Instant::now();
            timing.write(&mut writer, &[], 24_000, start).unwrap();
            assert_eq!(timing.first_audio_ms, None);
            assert_eq!(timing.writes, 1);
            assert_eq!(writer.samples(), 0);
            timing.write(&mut writer, &[0.25], 24_000, start).unwrap();
            let first = timing.first_audio_ms;
            assert!(first.is_some());
            assert_eq!(timing.writes, 2);
            timing.write(&mut writer, &[], 24_000, start).unwrap();
            assert_eq!(timing.first_audio_ms, first);
            assert_eq!(timing.writes, 3);
            assert_eq!(writer.samples(), 1);
            assert!(!writer.into_inner().is_empty());
        }
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("broken"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = StreamWriter::new(Broken, Container::Wav);
        let mut timing = StreamTiming::default();
        assert!(timing.write(&mut writer, &[0.25], 24_000, Instant::now()).is_err());
        assert_eq!(timing.first_audio_ms, None);
        assert_eq!(timing.writes, 0);
    }

    #[test]
    fn empty_decode_has_no_first_audio_in_text_or_story_reports() {
        let root = std::env::temp_dir().join(format!("ggtts-empty-first-audio-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        for mode in ["text", "raw-story", "held-story", "mixed-story"] {
            let stream = root.join(format!("{mode}.wav"));
            let report = root.join(format!("{mode}.json"));
            let trace = root.join(format!("{mode}.jsonl"));
            let story = root.join(format!("{mode}.txt"));
            let mut args = Args {
                engine: Some("fake".into()),
                stream: Some(stream.to_string_lossy().into_owned()),
                timing_report: Some(report.clone()),
                ..Args::default()
            };
            if mode == "text" {
                args.text = Some("A test.".into());
            } else {
                std::fs::write(&story, "Title\nFirst sentence.\nSecond sentence.\n").unwrap();
                args.story = Some(story);
                args.trace = Some(trace.clone());
                args.stream_raw = mode == "raw-story" || mode == "mixed-story";
                if !args.stream_raw {
                    args.stream_level_seconds = Some(60.0);
                }
            }
            let all_empty = mode != "mixed-story";
            let code =
                render_with_loader(args, log, move |_, _, _| Ok(Box::new(EmptyThenAudio::new(all_empty)))).unwrap();
            assert_eq!(code, 0);
            let timing: serde_json::Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
            if all_empty {
                assert!(timing["stream"]["first_audio_ms"].is_null(), "{mode}");
                assert!(timing["stream"]["writes"].as_u64().unwrap() >= 1, "{mode}");
                assert_eq!(std::fs::read(&stream).unwrap().len(), 44, "{mode}");
            } else {
                assert!(timing["stream"]["first_audio_ms"].is_number(), "{mode}");
                assert!(timing["stream"]["writes"].as_u64().unwrap() >= 2, "{mode}");
            }
            if mode != "text" {
                let events: Vec<serde_json::Value> = std::fs::read_to_string(&trace)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                let firsts = events
                    .iter()
                    .filter(|v| v["ev"] == "timing" && v["stage"] == "first_audio")
                    .count();
                assert_eq!(firsts, usize::from(!all_empty), "{mode}");
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_stream_makes_required_timing_incomplete() {
        let root = std::env::temp_dir().join(format!("ggtts-empty-required-timing-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let story = root.join("story.txt");
        std::fs::write(&story, "Title\nA short sentence.\n").unwrap();
        let trace = root.join("trace.jsonl");
        let report = root.join("report");
        let mut args = Args {
            engine: Some("fake".into()),
            story: Some(story),
            out: Some(root.join("out.flac")),
            stream: Some(root.join("stream.wav").to_string_lossy().into_owned()),
            trace: Some(trace.clone()),
            report: Some(report.clone()),
            ..Args::default()
        };
        args.instruments.take("--instrument", "render-timing".into());
        args.instruments.take("--require", "render-timing".into());
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        let code = render_with_loader(args, log, |_, _, _| Ok(Box::new(EmptyThenAudio::new(true)))).unwrap();
        assert_eq!(
            code, 3,
            "missing first audio must make the required instrument incomplete"
        );
        let events: Vec<serde_json::Value> = std::fs::read_to_string(trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(events.iter().any(|v| v["ev"] == "end"));
        assert!(!events
            .iter()
            .any(|v| v["ev"] == "timing" && v["stage"] == "first_audio"));
        let envelope: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report.join("instruments/render-timing/envelope.json")).unwrap())
                .unwrap();
        assert_eq!(envelope["status"], "skipped");
        assert_eq!(
            envelope["error"],
            "missing or invalid first_audio timing; a streaming v2 trace is required"
        );

        // The same instrument configuration must complete when a normal decoder supplies audio.
        let mut control = Args {
            engine: Some("fake".into()),
            story: Some(root.join("story.txt")),
            out: Some(root.join("control.flac")),
            stream: Some(root.join("control.wav").to_string_lossy().into_owned()),
            trace: Some(root.join("control.jsonl")),
            report: Some(root.join("control-report")),
            ..Args::default()
        };
        control.instruments.take("--instrument", "render-timing".into());
        control.instruments.take("--require", "render-timing".into());
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        let code = render_with_loader(control, log, |_, _, _| Ok(Box::new(ggtts_fake::FakeEngine::default()))).unwrap();
        assert_eq!(code, 0, "a normal stream must complete the required measurement");
        let envelope: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("control-report/instruments/render-timing/envelope.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(envelope["status"], "ok");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_resolution_uses_given_paths_and_never_fetches_for_them() {
        let log = LogSettings {
            level: LogLevel::Error,
            format: LogFormat::Text,
        };
        // The fake engine needs no model; a given one is still recorded.
        let mut a = Args::default();
        assert!(resolve_models(&mut a, "fake", log).unwrap().is_empty());
        let mut a = Args {
            model: Some("m.bin".into()),
            ..Args::default()
        };
        let files = resolve_models(&mut a, "fake", log).unwrap();
        assert_eq!(
            (files[0].role.as_str(), files[0].source),
            ("model", ggtts_models::Source::Path)
        );
        // An engine with no default files needs its paths.
        let mut a = Args::default();
        let err = resolve_models(&mut a, "kitten-tts-2", log).unwrap_err();
        assert!(err.contains("--model is required"), "{err}");
        // Both Qwen paths given: used as they are, in the table's roles, with nothing downloaded or hashed.
        let mut a = Args {
            model: Some("talker.gguf".into()),
            codec: Some("codec.gguf".into()),
            offline: true,
            ..Args::default()
        };
        let files = resolve_models(&mut a, "qwen3-tts", log).unwrap();
        let got: Vec<_> = files
            .iter()
            .map(|f| (f.role.as_str(), f.path.clone(), f.sha256.clone()))
            .collect();
        assert_eq!(
            got,
            [
                ("talker", PathBuf::from("talker.gguf"), None),
                ("codec", PathBuf::from("codec.gguf"), None)
            ]
        );
        assert_eq!(a.speaker, None, "a given talker keeps its own default speaker");
        // A build without engines downloads nothing for a render it cannot run.
        if !cfg!(feature = "ggml") {
            let mut a = Args::default();
            let err = resolve_models(&mut a, "qwen3-tts", log).unwrap_err();
            assert!(err.contains("no engines"), "{err}");
        }
    }

    /// Every engine `--version` can list reaches its own arm in `load_engine`, never the unknown-architecture refusal.
    #[test]
    fn every_engine_has_a_load_arm() {
        let a = Args::default();
        let refused = |name: &str| {
            load_engine(name, &a, Path::new("no-such-model.gguf"))
                .err()
                .unwrap_or_default()
        };
        for name in ENGINES {
            let err = refused(name);
            assert!(!err.is_empty(), "{name}: loaded with no model");
            assert!(!err.starts_with("no engine for architecture"), "{name}: {err}");
        }
        assert!(refused("not-an-engine").starts_with("no engine for architecture"));
    }
}
