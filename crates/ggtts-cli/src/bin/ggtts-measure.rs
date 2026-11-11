//! ggtts-measure: run instruments on audio that already exists, without rendering.
//!
//! ```text
//! ggtts-measure --audio IN.flac [--events E.jsonl] [--render ID] [--run ID] --report DIR
//!               [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]...
//!               [--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort] [--instrument-timeout S]
//! ```
//!
//! The instrument phase of `ggtts-render`, on a file you already have: a render kept from an earlier run, a release
//! sample, a recording. The same config (`ggtts-instruments/1`), the same runner, the same report layout and the same
//! exit codes: 0 when the measurement is complete, 3 when a required instrument is not ok (INCOMPLETE), 2 when the
//! command is refused and no instrument runs.
//!
//! * **It loads no engine.** Nothing here constructs one, and no model file is read.
//! * **It never writes to the given files.** The report directory must not already exist. The instruments get copies
//!   under separate `<report>/input/audio/` and `<report>/input/events/` directories. The runner's own
//!   check catches an instrument that writes to the copy (`failed`, the copy restored), and the originals' hashes are
//!   compared before and after the run.
//! * **The ids default as ggtts-render's do:** the render is the audio's file stem, and the run is `<render>-<secs>`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_core::logging::{Format as LogFormat, Level as LogLevel, Settings as LogSettings};
use ggtts_instruments::runner::{input_file, EXIT_RENDER_FAILED};

const BINARY: &str = "ggtts-measure";
const USAGE: &str = "usage: ggtts-measure --audio IN.flac [--events E.jsonl] [--render ID] [--run ID] --report DIR \
[--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]... \
[--audio-state raw|trimmed|unknown] \
[--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort] [--instrument-timeout S] \
[--log-level error|warn|info|debug|trace] [--log-format text|json]\n\
  --report DIR must not already exist";

#[derive(Default)]
struct Args {
    audio: Option<PathBuf>,
    audio_state: Option<ggtts_core::report::AudioState>,
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
            "--audio-state" => {
                a.audio_state = Some(match value()?.as_str() {
                    "raw" => ggtts_core::report::AudioState::Raw,
                    "trimmed" => ggtts_core::report::AudioState::Trimmed,
                    "unknown" => ggtts_core::report::AudioState::Unknown,
                    _ => return Err("--audio-state must be raw, trimmed or unknown".into()),
                });
            }
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

fn measure(a: Args, log: LogSettings) -> Result<u8, String> {
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

    // A reused report can contain source files, symlinks or hard links at any output path.
    // Reserve a fresh directory atomically before writing anything into it.
    if let Some(parent) = report.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::create_dir(&report).map_err(|e| format!("--report {} must be a new directory: {e}", report.display()))?;
    let input_dir = report.join("input");
    let audio_dir = input_dir.join("audio");
    fs::create_dir_all(&audio_dir).map_err(|e| format!("{}: {e}", audio_dir.display()))?;
    let audio_copy = copy_into(&audio, &audio_dir)?;
    let events_copy = if let Some(events) = a.events.as_deref() {
        let events_dir = input_dir.join("events");
        fs::create_dir_all(&events_dir).map_err(|e| format!("{}: {e}", events_dir.display()))?;
        Some(copy_into(events, &events_dir)?)
    } else {
        None
    };
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
        audio_state: a.audio_state.unwrap_or(ggtts_core::report::AudioState::Unknown),
        audio_seconds: flac_seconds(&bytes),
        events: events_copy,
        bin_dir: None,
        instruments_dir: None,
        models: None,
        instruction: None,
    };
    let summary = ggtts_instruments::runner::run_with_logging(&config, &inputs, log)?;
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
    let after: Vec<_> = std::iter::once(&audio)
        .chain(a.events.as_ref())
        .map(|p| input_file(p))
        .collect();
    if before != after {
        return Err("a given file changed during the run, though no instrument was handed it".into());
    }
    log.stderr(
        LogLevel::Info,
        BINARY,
        "instrument_report",
        format_args!(
            "instrument report in {}; runner {} ms (budget {} ms)",
            report.display(),
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
    Ok(summary.exit_code())
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
            return ExitCode::from(EXIT_RENDER_FAILED);
        }
    };
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
        Some(_) => match parse(&args).and_then(|a| measure(a, log)) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                log.stderr(LogLevel::Error, "error", "command_failed", format_args!("{e}"));
                if log.format == LogFormat::Text {
                    eprintln!("{USAGE}");
                }
                ExitCode::from(EXIT_RENDER_FAILED)
            }
        },
    }
}
