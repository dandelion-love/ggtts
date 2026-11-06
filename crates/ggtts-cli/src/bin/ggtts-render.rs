//! ggtts-render: text in, FLAC out.
//!
//! ```text
//! ggtts-render --model M.gguf [--codec C.gguf] --out x.flac [--engine auto|qwen3-tts|kitten-tts-2|fake]
//!              [--threads N] [--device cpu|cuda] [--trace trace.jsonl] [--voice REF.json]
//!              [--stream FILE|- [--stream-format wav|s16le]] (--text "..." [--seed S] | --story story.txt)
//!              [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE.yaml] [--require NAME]...
//!              [--on-instrument-error continue|abort] [--instrument-timeout S] [--report DIR] [--run ID]
//! ```
//!
//! `--voice` is a reference voice for an engine that clones one; an engine that does not ignores it.
//! `--engine auto` (the default) picks the engine from the model's `general.architecture`. `--text` renders one
//! request at one seed (default 20260905). `--story` reads a story file (title line, then one paragraph per line)
//! and renders it on the core's story path. `--trace` writes the engine's per-step stop probabilities as JSON
//! lines; it is read-only, so the FLAC is the same with or without it. Engines need the `ggml` feature (and
//! `--device cuda` the `cuda` feature); without it only `--version` and the fake engine work.
//!
//! `--stream` writes the audio as it is produced, to a file or `-` for stdout, as a WAV of unknown length (the
//! default) or raw s16le after a one-line header. A story streams chunk by chunk, each piece as soon as its chunk
//! is kept, **before the level step**: the read's level gain is known only once every chunk is in, so it is
//! reported on stderr at the end, and `--out` (optional with `--stream`) still writes the levelled FLAC. A `--text`
//! render has no level step, so its stream holds the FLAC's samples.
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
use ggtts_core::pipeline::{plan_story, render_story_streamed, RenderOptions};
use ggtts_core::stream::{Container, StreamWriter};
use ggtts_core::text::read_text;
use ggtts_core::trace::NullEvents;

#[path = "../gguf_arch.rs"]
mod gguf_arch;

const BINARY: &str = "ggtts-render";
const USAGE: &str = "usage: ggtts-render --model M.gguf [--codec C.gguf] --out x.flac \
[--engine auto|qwen3-tts|kitten-tts-2|fake] [--threads N] [--device cpu|cuda] [--trace FILE] [--voice REF] \
[--stream FILE|- [--stream-format wav|s16le]] (--text TEXT [--seed S] | --story FILE) \
[--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]... \
[--on-instrument-error continue|abort] [--instrument-timeout S] [--report DIR] [--run ID]\n       ggtts-render --version";

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
    /// A reference voice, for engines that clone one (kitten-tts-2: a `ggtts-kitten prepare` JSON). Others ignore it.
    voice: Option<PathBuf>,
    stream: Option<String>,
    stream_format: Option<String>,
    instruments: ggtts_instruments::Flags,
    /// Where the instruments' report goes.
    report: Option<PathBuf>,
    /// The run's identity, for every envelope.
    run: Option<String>,
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
            "--out" | "-o" => a.out = Some(value()?.into()),
            "--engine" => a.engine = Some(value()?),
            "--text" | "-t" => a.text = Some(value()?),
            "--story" => a.story = Some(value()?.into()),
            "--seed" => a.seed = Some(value()?.parse().map_err(|_| "--seed needs an integer")?),
            "--device" => a.device = Some(value()?),
            "--trace" => a.trace = Some(value()?.into()),
            "--voice" => a.voice = Some(value()?.into()),
            "--stream" => a.stream = Some(value()?),
            "--stream-format" => a.stream_format = Some(value()?),
            "--threads" | "-j" => a.threads = Some(value()?.parse().map_err(|_| "--threads needs an integer")?),
            other => return Err(format!("invalid argument: {other}")),
        }
    }
    Ok(a)
}

/// The model engines this binary can carry, by architecture name: `load_engine` has an arm for each, and
/// `engine_lines` lists them when the build has them. `every_engine_has_a_load_arm` holds the two together.
const ENGINES: &[&str] = &["qwen3-tts", "kitten-tts-2"];

/// The engine for a model, by architecture name. One arm per engine; an engine left out of the build is reported,
/// not hidden.
fn load_engine(name: &str, a: &Args, model: &Path) -> Result<Box<dyn Engine>, String> {
    let threads = a.threads.unwrap_or(4);
    match name {
        #[cfg(feature = "ggml")]
        "qwen3-tts" => {
            let codec = a.codec.as_ref().ok_or("the qwen3-tts engine needs --codec")?;
            let device = a.device.as_deref().unwrap_or("cpu");
            let voice = ggtts_qwen3_tts::engine::Voice {
                device: ggtts_qwen3_tts::cpu::Device::parse(device)
                    .ok_or(format!("--device must be cpu or cuda, not {device}"))?,
                trace: a.trace.clone(),
                ..Default::default()
            };
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

/// Where `--stream` writes: a file, or stdout for `-`.
fn open_stream(a: &Args) -> Result<Option<StreamWriter<Box<dyn std::io::Write>>>, String> {
    let Some(dest) = a.stream.as_deref() else {
        if a.stream_format.is_some() {
            return Err("--stream-format needs --stream".into());
        }
        return Ok(None);
    };
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
fn render(a: Args) -> Result<u8, String> {
    let mut stream = open_stream(&a)?;
    if a.out.is_none() && stream.is_none() {
        return Err("--out is required (or --stream)".into());
    }
    // Refuse a bad instrument config before spending the render on it.
    let instruments = if a.instruments.any() {
        if a.out.is_none() {
            return Err("instruments measure the --out FLAC: give --out".into());
        }
        let c = a.instruments.resolve()?;
        ggtts_instruments::runner::check(&c)?;
        Some(c)
    } else {
        None
    };
    let fake = a.engine.as_deref() == Some("fake");
    let model = match (&a.model, fake) {
        (Some(m), _) => m.clone(),
        (None, true) => PathBuf::new(),
        (None, false) => return Err("--model is required".into()),
    };
    let name = match a.engine.as_deref() {
        None | Some("auto") => gguf_arch::architecture(&model).map_err(|e| format!("{}: {e}", model.display()))?,
        Some(e) => e.to_string(),
    };
    let t0 = Instant::now();
    let mut engine = load_engine(&name, &a, &model)?;
    let load_ms = t0.elapsed().as_millis();
    let t1 = Instant::now();
    let (samples, rate) = match (&a.text, &a.story) {
        (Some(text), None) => {
            let req = ChunkRequest {
                text,
                seed: a.seed.unwrap_or(20260905),
                chunk: 0,
                draw: 0,
            };
            let g = engine.generate(&req).map_err(|e| e.to_string())?;
            let gen_ms = t1.elapsed().as_millis();
            let t2 = Instant::now();
            let pcm = engine.decode(&g.codes).map_err(|e| e.to_string())?;
            if let Some(w) = stream.as_mut() {
                w.write(&pcm.samples, pcm.sample_rate)
                    .map_err(|e| format!("--stream: {e}"))?;
            }
            eprintln!(
                "{BINARY}: {} frames, stop {:?}, generate {gen_ms} ms, decode {} ms",
                g.codes.n_frames(),
                g.stop,
                t2.elapsed().as_millis()
            );
            (pcm.samples, pcm.sample_rate)
        }
        (None, Some(story)) => {
            let contents = std::fs::read_to_string(story).map_err(|e| format!("{}: {e}", story.display()))?;
            let (title, paragraphs) = read_text(&contents);
            let profile = *engine.profile();
            let chunks = plan_story(&profile, &paragraphs);
            let mut first_audio_ms = None;
            let read = render_story_streamed(
                engine.as_mut(),
                &title,
                &chunks,
                &RenderOptions::for_profile(&profile),
                &mut NullEvents,
                &mut |p| {
                    if let Some(w) = stream.as_mut() {
                        first_audio_ms.get_or_insert(t1.elapsed().as_millis());
                        // A failed write stops the render: nothing can receive the rest.
                        w.write(p.samples, p.sample_rate).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                },
            )
            .map_err(|e| e.to_string())?;
            if let Some(w) = &stream {
                let gain = read.gain.map_or("none".to_string(), |g| {
                    format!("{:+.4} dB (the --out FLAC applies it)", 20.0 * g.log10())
                });
                eprintln!(
                    "{BINARY}: streamed {} samples before the level step, first audio at {} ms; level gain {gain}",
                    w.samples(),
                    first_audio_ms.unwrap_or(0),
                );
            }
            eprintln!(
                "{BINARY}: {} chunks, {} first-draw defects, {:.2} s of audio",
                read.chunks.len(),
                read.first_draw_defects(),
                read.seconds()
            );
            (read.samples, read.sample_rate)
        }
        _ => return Err("give exactly one of --text and --story".into()),
    };
    let render_ms = t1.elapsed().as_millis();
    let mut written = None;
    if let Some(out) = &a.out {
        let path = write_flac(out, &samples, rate).map_err(|e| format!("{}: {e}", out.display()))?;
        eprintln!(
            "{BINARY}: wrote {} ({} samples at {rate} Hz) with {} on {}, parity {}; load {load_ms} ms, render {render_ms} ms",
            path.display(),
            samples.len(),
            engine.info().family,
            engine.info().backend,
            engine.info().parity
        );
        written = Some(path);
    }
    let (Some(config), Some(path)) = (instruments, written) else {
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
        audio_seconds: (rate > 0).then(|| samples.len() as f64 / f64::from(rate)),
        events: a.trace.clone(),
        bin_dir: None,
    };
    let summary = ggtts_instruments::run(&config, &inputs)?;
    for e in &summary.envelopes {
        let why = e.error.as_ref().map_or(String::new(), |w| format!(": {w}"));
        eprintln!(
            "{BINARY}: instrument {} {}: {}{why}",
            e.name,
            e.version,
            e.status.as_str()
        );
    }
    eprintln!(
        "{BINARY}: instrument report in {}; runner {} ms (budget {} ms)",
        inputs.report_dir.display(),
        summary.overhead.own_ms,
        summary.overhead.budget_ms
    );
    if !summary.complete() {
        eprintln!("{BINARY}: measurement INCOMPLETE: {}", summary.incomplete.join("; "));
    }
    Ok(summary.exit_code())
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
    let args: Vec<String> = std::env::args().skip(1).collect();
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
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(_) => match parse(&args).and_then(render) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("{USAGE}");
                ExitCode::from(ggtts_instruments::runner::EXIT_RENDER_FAILED)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
