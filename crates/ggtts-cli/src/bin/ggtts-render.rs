//! ggtts-render: text in, FLAC out.
//!
//! ```text
//! ggtts-render --model M.gguf [--codec C.gguf] --out x.flac [--engine auto|qwen3-tts] [--threads N]
//!              [--device cpu|cuda] [--trace trace.jsonl] [--voice REF.json] (--text "..." [--seed S] | --story story.txt)
//! ```
//!
//! `--voice` is a reference voice for an engine that clones one; an engine that does not ignores it.
//! `--engine auto` (the default) picks the engine from the model's `general.architecture`. `--text` renders one
//! request at one seed (default 20260905). `--story` reads a story file (title line, then one paragraph per line)
//! and renders it on the core's story path. `--trace` writes the engine's per-step stop probabilities as JSON
//! lines; it is read-only, so the FLAC is the same with or without it. Engines need the `ggml` feature (and
//! `--device cuda` the `cuda` feature); without it only `--version` works.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::flac::write_flac;
use ggtts_core::pipeline::{plan_story, render_story, RenderOptions};
use ggtts_core::text::read_text;

#[path = "../gguf_arch.rs"]
mod gguf_arch;

const BINARY: &str = "ggtts-render";
const USAGE: &str = "usage: ggtts-render --model M.gguf [--codec C.gguf] --out x.flac [--engine auto|qwen3-tts] \
[--threads N] [--device cpu|cuda] [--trace FILE] [--voice REF] (--text TEXT [--seed S] | --story FILE)\n       ggtts-render --version";

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
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
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
            "--threads" | "-j" => a.threads = Some(value()?.parse().map_err(|_| "--threads needs an integer")?),
            other => return Err(format!("invalid argument: {other}")),
        }
    }
    Ok(a)
}

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
        "qwen3-tts" | "kitten-tts-2" => {
            let _ = (threads, model, &a.voice);
            Err("this build has no engines: rebuild with --features ggml".into())
        }
        other => Err(format!("no engine for architecture {other:?}")),
    }
}

fn render(a: Args) -> Result<(), String> {
    let model = a.model.clone().ok_or("--model is required")?;
    let out = a.out.clone().ok_or("--out is required")?;
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
            let read = render_story(engine.as_mut(), &title, &chunks, &RenderOptions::for_profile(&profile))
                .map_err(|e| e.to_string())?;
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
    let path = write_flac(&out, &samples, rate).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "{BINARY}: wrote {} ({} samples at {rate} Hz) with {}; load {load_ms} ms, render {render_ms} ms",
        path.display(),
        samples.len(),
        engine.info().family
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("--version") | Some("-V") => {
            for line in ggtts_core::version_lines(BINARY) {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Some("-h") | Some("--help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(_) => match parse(&args).and_then(render) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
    }
}
