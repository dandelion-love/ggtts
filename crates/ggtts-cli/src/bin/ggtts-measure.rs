//! ggtts-measure: run instruments on audio that already exists, without rendering.
//!
//! ```text
//! ggtts-measure --audio IN.flac [--events E.jsonl] [--render ID] [--run ID] --report DIR
//!               [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]...
//!               [--on-instrument-error continue|abort] [--instrument-timeout S]
//! ```
//!
//! The instrument phase of `ggtts-render`, on a file you already have: a render kept from an earlier run, a release
//! sample, a recording. The same config (`ggtts-instruments/1`), the same runner, the same report layout and the same
//! exit codes: 0 when the measurement is complete, 3 when a required instrument is not ok (INCOMPLETE), 2 when the
//! command is refused and no instrument runs.
//!
//! * **It loads no engine.** Nothing here constructs one, and no model file is read.
//! * **It never writes to the given files.** The instruments get copies under `<report>/input/`. The runner's own
//!   check catches an instrument that writes to the copy (`failed`, the copy restored), and the originals' hashes are
//!   compared before and after the run.
//! * **The ids default as ggtts-render's do:** the render is the audio's file stem, and the run is `<render>-<secs>`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_instruments::runner::{input_file, EXIT_RENDER_FAILED};

const BINARY: &str = "ggtts-measure";
const USAGE: &str = "usage: ggtts-measure --audio IN.flac [--events E.jsonl] [--render ID] [--run ID] --report DIR \
[--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]... \
[--on-instrument-error continue|abort] [--instrument-timeout S]";

#[derive(Default)]
struct Args {
    audio: Option<PathBuf>,
    events: Option<PathBuf>,
    render: Option<String>,
    run: Option<String>,
    report: Option<PathBuf>,
    instruments: ggtts_instruments::Flags,
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
            "--audio" => a.audio = Some(value()?.into()),
            "--events" => a.events = Some(value()?.into()),
            "--render" => a.render = Some(value()?),
            "--run" => a.run = Some(value()?),
            "--report" => a.report = Some(value()?.into()),
            other => return Err(format!("invalid argument: {other}")),
        }
    }
    Ok(a)
}

/// The audio's length in seconds from a FLAC's STREAMINFO, for the runner's overhead figure. None for anything else.
fn flac_seconds(bytes: &[u8]) -> Option<f64> {
    let si = bytes.get(8..8 + 34).filter(|_| bytes.starts_with(b"fLaC"))?;
    let rate = (u32::from(si[10]) << 12) | (u32::from(si[11]) << 4) | (u32::from(si[12]) >> 4);
    let total = (u64::from(si[13] & 0x0f) << 32) | u64::from(u32::from_be_bytes([si[14], si[15], si[16], si[17]]));
    (rate > 0 && total > 0).then(|| total as f64 / f64::from(rate))
}

/// Refuse what cannot be measured before any instrument runs: a missing or empty file, or one that is neither FLAC
/// nor WAV.
fn readable_audio(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|e| format!("--audio {}: {e}", path.display()))?;
    let wav = bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE";
    if !(bytes.starts_with(b"fLaC") || wav) {
        return Err(format!("--audio {}: not a FLAC or WAV file", path.display()));
    }
    Ok(bytes)
}

/// A copy of `src` under `dir`, keeping its file name, for the instruments to read.
fn copy_into(src: &Path, dir: &Path) -> Result<PathBuf, String> {
    let name = src.file_name().ok_or(format!("{}: no file name", src.display()))?;
    let dst = dir.join(name);
    fs::copy(src, &dst).map_err(|e| format!("copying {} to {}: {e}", src.display(), dst.display()))?;
    Ok(dst)
}

fn measure(a: Args) -> Result<u8, String> {
    let audio = a.audio.clone().ok_or("--audio is required")?;
    let report = a.report.clone().ok_or("--report is required")?;
    if !a.instruments.any() {
        return Err("no instrument given: use --instrument NAME or --instruments FILE".into());
    }
    let config = a.instruments.resolve()?;
    ggtts_instruments::runner::check(&config)?;
    let bytes = readable_audio(&audio)?;
    if let Some(e) = &a.events {
        if !e.is_file() {
            return Err(format!("--events {}: no such file", e.display()));
        }
    }
    let before: Vec<_> = std::iter::once(&audio)
        .chain(a.events.as_ref())
        .map(|p| input_file(p))
        .collect();

    let input_dir = report.join("input");
    fs::create_dir_all(&input_dir).map_err(|e| format!("{}: {e}", input_dir.display()))?;
    let audio_copy = copy_into(&audio, &input_dir)?;
    let events_copy = a.events.as_deref().map(|e| copy_into(e, &input_dir)).transpose()?;
    let render = a.render.clone().unwrap_or_else(|| {
        audio
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("render")
            .to_string()
    });
    let run = a.run.clone().unwrap_or_else(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("{render}-{secs}")
    });
    let inputs = ggtts_instruments::Inputs {
        run,
        render,
        report_dir: report.clone(),
        audio: audio_copy,
        audio_seconds: flac_seconds(&bytes),
        events: events_copy,
        bin_dir: None,
        instruments_dir: None,
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
    let after: Vec<_> = std::iter::once(&audio)
        .chain(a.events.as_ref())
        .map(|p| input_file(p))
        .collect();
    if before != after {
        return Err("a given file changed during the run, though no instrument was handed it".into());
    }
    eprintln!(
        "{BINARY}: instrument report in {}; runner {} ms (budget {} ms)",
        report.display(),
        summary.overhead.own_ms,
        summary.overhead.budget_ms
    );
    if !summary.complete() {
        eprintln!("{BINARY}: measurement INCOMPLETE: {}", summary.incomplete.join("; "));
    }
    Ok(summary.exit_code())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("-h") | Some("--help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("--version") | Some("-V") => {
            for line in ggtts_core::version_lines(BINARY) {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Some(_) => match parse(&args).and_then(measure) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("{USAGE}");
                ExitCode::from(EXIT_RENDER_FAILED)
            }
        },
    }
}
