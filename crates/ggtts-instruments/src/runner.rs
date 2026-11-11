//! Run the configured instruments on a finished render, each behind its own boundary, and merge their envelopes into
//! the run's report.
//!
//! An instrument is a child process, started once with:
//!
//! ```text
//! PROGRAM --run RUN --render RENDER [--events EVENTS.jsonl] --audio OUT.flac --envelope ENVELOPE.json --name NAME
//!         [--KEY VALUE]...
//! ```
//!
//! It writes one `ggtts-report/1` envelope ([`ggtts_core::report`]) whose identity is `{run: RUN, instrument: NAME}`.
//! Its stdout and stderr go to its own log, `<report>/instruments/<name>/<name>.log`. It has a timeout, after which
//! it is killed. It may not change the run's audio: the runner compares the file's bytes before and after, and puts
//! them back if they changed. Whatever happens, the runner keeps one envelope per instrument at
//! `<report>/instruments/<name>/envelope.json`, merges them into `report.json`, `report.md` and `report.html`
//! ([`ggtts_core::report::merge`]), and writes a line per instrument to `<report>/run.log`: name, version, status,
//! exit and time.
//!
//! The runner owns `status`, `log` and `error`. A crash, a non-zero exit or a write to the audio is `failed`; the
//! timeout is `timeout`; an envelope that does not validate, or names another run or instrument, is `invalid`; an
//! instrument that needs the event stream is `skipped` without one and `truncated` with one that has no `end` event.
//!
//! The render is finished before any instrument starts, so an instrument can make the *report* fail but can never
//! change the render, and the time to first audio is unchanged. Under [`OnError::Abort`] the instruments after a
//! failure are `skipped`.
//!
//! **Render completion and measurement completeness are separate.** The render is complete when its audio is
//! written. The measurement is complete only when every instrument in `require` is present, valid and `ok` in the
//! merged report; otherwise it is INCOMPLETE, with each reason kept in [`RunSummary::incomplete`] and `run.log`,
//! never a pass and never a zero. The exit codes keep the three outcomes apart: [`EXIT_OK`], [`EXIT_RENDER_FAILED`],
//! [`EXIT_INCOMPLETE`].
//!
//! **This boundary is for child processes only.** A tensor hook runs inside the engine, so it is trusted and
//! feature-gated, and gets no timeout and no crash boundary. [`in_process`] catches its panic and reports it as
//! `failed`. A hook that hangs hangs the run, and one that aborts the process (`std::process::abort`, a stack
//! overflow, a panic under `panic = "abort"`) ends the run. Nothing here can contain either, so nothing here claims
//! to.

use std::fs::{self, File};
use std::panic::{catch_unwind, UnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use ggtts_core::logging::{Format as LogFormat, Level as LogLevel, Settings as LogSettings};
use ggtts_core::report::{
    self, AudioState, Envelope, Identity, Input, InputFile, Metric, Provenance, Report, Status, FORMAT,
};
use serde_json::{json, Map, Value};

use crate::catalogue;
use crate::config::{Config, Entry, OnError};

/// The render was written, and the measurement is complete.
pub const EXIT_OK: u8 = 0;
/// The render failed, or the command line was wrong: there is no audio to measure.
pub const EXIT_RENDER_FAILED: u8 = 2;
/// The render was written, and a required instrument or input is missing: the measurement is INCOMPLETE.
pub const EXIT_INCOMPLETE: u8 = 3;

/// The runner's own overhead budget (stated before measuring): this much per instrument, plus [`BUDGET_RUN_MS`].
pub const BUDGET_PER_INSTRUMENT_MS: u128 = 50;
pub const BUDGET_RUN_MS: u128 = 100;
/// The runner's own share of the real-time factor, on a read of 30 s or longer.
pub const BUDGET_RTF: f64 = 0.01;

/// What the instruments read, who they report for, and where the report goes.
#[derive(Debug, Clone)]
pub struct Inputs {
    /// The run's id: every envelope of the run carries it.
    pub run: String,
    /// The render's id, for metrics scoped to it.
    pub render: String,
    /// The report directory; created if absent.
    pub report_dir: PathBuf,
    /// The render's audio.
    pub audio: PathBuf,
    /// The known trim state of the audio, or Unknown when the caller cannot establish it.
    pub audio_state: AudioState,
    /// Its length in seconds, for the overhead's real-time factor.
    pub audio_seconds: Option<f64>,
    /// The run's event stream, when it has one.
    pub events: Option<PathBuf>,
    /// Where shipped instruments' programs are. Default: beside the running binary.
    pub bin_dir: Option<PathBuf>,
    /// The shipped `instruments/` directory, which holds the script instruments. Default: `GGTTS_INSTRUMENTS_DIR`,
    /// else the nearest `instruments/` above the running binary (a source checkout's `target/` sits beside it).
    pub instruments_dir: Option<PathBuf>,
    /// Every model file the render used, for `report.json` and `run.log`; `None` when the audio was not rendered by
    /// this run (a measurement of existing audio).
    pub models: Option<Vec<serde_json::Value>>,
}

/// What the instrument phase cost, after the render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Overhead {
    /// The whole phase: every instrument and the runner around them.
    pub total_ms: u128,
    /// The instruments' own time, spawn to exit.
    pub instruments_ms: u128,
    /// `total_ms - instruments_ms`: spawning, supervising, hashing, envelopes and the merge.
    pub own_ms: u128,
    /// The budget for `own_ms`.
    pub budget_ms: u128,
}

/// What a run of the instruments found.
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub envelopes: Vec<Envelope>,
    /// The merged report, as written to `report.json`, `.md` and `.html`.
    pub report: Report,
    /// An instrument failed under [`OnError::Abort`].
    pub aborted: bool,
    /// Why the measurement is INCOMPLETE, one reason per unmet requirement. Empty: complete.
    pub incomplete: Vec<String>,
    pub overhead: Overhead,
}

impl RunSummary {
    /// Every required instrument is present, valid and `ok`.
    pub fn complete(&self) -> bool {
        self.incomplete.is_empty()
    }

    /// [`EXIT_OK`] or [`EXIT_INCOMPLETE`]: the render was written either way.
    pub fn exit_code(&self) -> u8 {
        if self.complete() {
            EXIT_OK
        } else {
            EXIT_INCOMPLETE
        }
    }

    /// Instruments that failed, timed out or wrote an invalid envelope.
    pub fn failures(&self) -> usize {
        self.envelopes
            .iter()
            .filter(|e| matches!(e.status, Status::Failed | Status::Timeout | Status::Invalid))
            .count()
    }
}

/// Refuse a config whose instruments cannot be found, before anything is rendered.
pub fn check(config: &Config) -> Result<(), String> {
    for e in &config.instruments {
        if e.command.is_none() && catalogue::find(&e.name).is_none() {
            let names: Vec<&str> = catalogue::CATALOGUE.iter().map(|k| k.name).collect();
            return Err(format!(
                "instrument {:?} is not shipped (shipped: {}); give it a command to run your own",
                e.name,
                names.join(", ")
            ));
        }
    }
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A file's path and content hash, for an envelope's provenance.
pub fn input_file(path: &Path) -> Option<InputFile> {
    let bytes = fs::read(path).ok()?;
    Some(InputFile {
        path: path.display().to_string(),
        sha256: hex(&ggtts_core::seed::sha256(&bytes)),
    })
}

/// The entry's effective configuration, defaults included, for the envelope's provenance.
fn entry_config(config: &Config, entry: &Entry, needs_events: bool) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(
        "requested_device".into(),
        json!(config.device.as_deref().unwrap_or("cpu")),
    );
    m.insert("timeout_s".into(), json!(config.timeout_for(entry)));
    m.insert("needs_events".into(), json!(needs_events));
    if let Some(c) = &entry.command {
        m.insert("command".into(), json!(c));
    }
    if let Some(i) = &entry.interpreter {
        m.insert("interpreter".into(), json!(i));
    }
    let args: Map<String, Value> = entry.args.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    m.insert("args".into(), Value::Object(args));
    m
}

/// Run every configured instrument in order, then merge. `Err` only when the report itself cannot be written.
pub fn run(config: &Config, inputs: &Inputs) -> Result<RunSummary, String> {
    run_with_logging(config, inputs, LogSettings::default())
}

/// Run with explicit log settings; child-process logs remain separate, unmodified files.
pub fn run_with_logging(config: &Config, inputs: &Inputs, log: LogSettings) -> Result<RunSummary, String> {
    let t_run = Instant::now();
    let report_dir = &inputs.report_dir;
    fs::create_dir_all(report_dir).map_err(|e| format!("{}: {e}", report_dir.display()))?;
    let run_log_path = report_dir.join("run.log");
    let run_log_file = File::create(&run_log_path).map_err(|e| format!("{}: {e}", run_log_path.display()))?;
    let run_log = ggtts_core::logging::LogSink::new(log, run_log_file);
    macro_rules! record {
        ($level:expr, $event:expr, $($args:tt)*) => {{
            if log.enabled($level) {
                let component = if log.format == LogFormat::Text { "" } else { "instrument-runner" };
                run_log.emit($level, component, $event, format_args!($($args)*), serde_json::Value::Null);
                if run_log.failures() != 0 {
                    return Err(format!("{}: unable to write instrument run log", run_log_path.display()));
                }
            }
        }};
    }
    let bin_dir = match &inputs.bin_dir {
        Some(d) => d.clone(),
        None => std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_default(),
    };
    let on_error = match config.on_error {
        OnError::Continue => "continue",
        OnError::Abort => "abort",
    };
    record!(
        LogLevel::Info,
        "start",
        "ggtts instruments {}: run {}, render {}, {} instrument(s), on_error {on_error}, require [{}], audio {}, \
         events {}",
        env!("CARGO_PKG_VERSION"),
        inputs.run,
        inputs.render,
        config.instruments.len(),
        config.require.join(", "),
        inputs.audio.display(),
        inputs
            .events
            .as_ref()
            .map_or("none".into(), |p| p.display().to_string()),
    );
    let mut provided = Vec::new();
    provided.extend(input_file(&inputs.audio));
    // An event stream that does not end with its `end` event was cut short: an instrument that needs it is
    // `truncated` rather than handed half a run.
    let events_problem = match &inputs.events {
        None => Some((Status::Skipped, "this run has no event stream".to_string())),
        Some(p) => match fs::read_to_string(p) {
            Err(e) => Some((
                Status::Skipped,
                format!("the event stream {} cannot be read: {e}", p.display()),
            )),
            Ok(t) => {
                provided.extend(input_file(p));
                let last = t.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or_default();
                let ended = serde_json::from_str::<Value>(last)
                    .map(|v| v["ev"] == "end")
                    .unwrap_or(false);
                (!ended).then(|| {
                    (
                        Status::Truncated,
                        format!("the event stream {} is truncated: no end event", p.display()),
                    )
                })
            }
        },
    };

    let mut envelopes = Vec::with_capacity(config.instruments.len());
    let mut merge_inputs = Vec::with_capacity(config.instruments.len());
    let mut instruments_ms = 0u128;
    let mut aborted = false;
    for entry in &config.instruments {
        let version = entry
            .version
            .clone()
            .or_else(|| catalogue::find(&entry.name).map(|k| k.version.to_string()))
            .unwrap_or_else(|| "unknown".into());
        let needs_events = entry
            .needs_events
            .or_else(|| catalogue::find(&entry.name).map(|k| k.needs_events))
            .unwrap_or(false);
        let dir = report_dir.join("instruments").join(&entry.name);
        let log_rel = format!("instruments/{0}/{0}.log", entry.name);
        let made = |status: Status, why: String| Envelope {
            name: entry.name.clone(),
            version: version.clone(),
            format: FORMAT.into(),
            identity: Identity {
                run: inputs.run.clone(),
                instrument: entry.name.clone(),
            },
            provenance: Provenance {
                config: entry_config(config, entry, needs_events),
                ggtts: Some(env!("CARGO_PKG_VERSION").into()),
                engine: None,
                backend: None,
                model_sha256: None,
                seeds: Vec::new(),
                inputs: provided.clone(),
                audio: AudioState::NotApplicable,
            },
            hooks: Vec::new(),
            status,
            metrics: Vec::new(),
            artifacts: Vec::new(),
            log: Some(log_rel.clone()),
            error: Some(why),
        };
        let (mut env, exit, ms) = if aborted {
            let why = "skipped: an earlier instrument failed and on_error is abort".to_string();
            (made(Status::Skipped, why), "not started".to_string(), 0)
        } else if let (true, Some((status, problem))) = (needs_events, &events_problem) {
            let why = format!("it needs the event stream, and {problem}");
            (made(*status, why), "not started".to_string(), 0)
        } else {
            one(config, entry, inputs, &bin_dir, &dir, &log_rel, &made)
        };
        // The caller owns the input and its provenance; do not let an instrument invent its trim state.
        // Event-only instruments retain n/a. The merged report contains v2 envelopes, including accepted v1 inputs.
        env.format = FORMAT.into();
        if env.provenance.audio != AudioState::NotApplicable || env.hooks.iter().any(|h| h == "artifact.audio") {
            env.provenance.audio = inputs.audio_state;
        }
        instruments_ms += ms;
        let _ = fs::create_dir_all(&dir);
        let env_path = dir.join("envelope.json");
        let text = serde_json::to_string_pretty(&env.to_json()).unwrap_or_default() + "\n";
        fs::write(&env_path, &text).map_err(|e| format!("{}: {e}", env_path.display()))?;
        merge_inputs.push(Input {
            source: format!("instruments/{}/envelope.json", entry.name),
            text,
        });
        let level = match env.status {
            Status::Ok => LogLevel::Info,
            Status::Failed | Status::Timeout | Status::Invalid => LogLevel::Error,
            Status::Skipped | Status::Truncated => LogLevel::Warn,
        };
        record!(
            level,
            "instrument",
            "instrument {} {}: {} ({exit}, {ms} ms){}",
            env.name,
            env.version,
            env.status.as_str(),
            env.error.as_ref().map_or(String::new(), |e| format!(": {e}"))
        );
        if matches!(env.status, Status::Failed | Status::Timeout | Status::Invalid) && config.on_error == OnError::Abort
        {
            aborted = true;
        }
        envelopes.push(env);
    }

    let required: Vec<&str> = config.require.iter().map(String::as_str).collect();
    let mut merged = report::merge(&inputs.run, &merge_inputs, &required);
    merged.models = inputs.models.clone();
    for m in inputs.models.iter().flatten() {
        let text = |k: &str| {
            m.get(k)
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string()
        };
        record!(
            LogLevel::Info,
            "model_file",
            "model {} {} sha256 {} ({})",
            text("role"),
            text("path"),
            text("sha256"),
            text("source")
        );
    }
    merged
        .write(report_dir)
        .map_err(|e| format!("{}: {e}", report_dir.display()))?;
    let incomplete: Vec<String> = merged
        .required
        .iter()
        .filter(|r| !r.met())
        .map(|r| match envelopes.iter().find(|e| e.name == r.name) {
            None => format!("required instrument {} was not configured to run", r.name),
            Some(e) => format!(
                "required instrument {} is {}{}",
                r.name,
                e.status.as_str(),
                e.error.as_ref().map_or(String::new(), |w| format!(": {w}"))
            ),
        })
        .collect();

    let total_ms = t_run.elapsed().as_millis();
    let own_ms = total_ms.saturating_sub(instruments_ms);
    let overhead = Overhead {
        total_ms,
        instruments_ms,
        own_ms,
        budget_ms: BUDGET_PER_INSTRUMENT_MS * config.instruments.len() as u128 + BUDGET_RUN_MS,
    };
    let summary = RunSummary {
        envelopes,
        report: merged,
        aborted,
        incomplete,
        overhead,
    };
    let count = |s: Status| summary.envelopes.iter().filter(|e| e.status == s).count();
    record!(
        LogLevel::Info,
        "done",
        "done: {} ok, {} failed, timed out or invalid, {} skipped, {} truncated{}",
        count(Status::Ok),
        summary.failures(),
        count(Status::Skipped),
        count(Status::Truncated),
        if summary.aborted { "; aborted" } else { "" }
    );
    let within = |ok: bool| if ok { "within" } else { "OVER" };
    record!(
        LogLevel::Info,
        "overhead",
        "overhead: time to first audio +0 ms (instruments start after the audio is written); instrument phase {} ms \
         = instruments {} ms + runner {} ms (budget {} ms: {})",
        overhead.total_ms,
        overhead.instruments_ms,
        overhead.own_ms,
        overhead.budget_ms,
        within(overhead.own_ms <= overhead.budget_ms)
    );
    if let Some(s) = inputs.audio_seconds.filter(|s| *s > 0.0) {
        let rtf = overhead.total_ms as f64 / 1000.0 / s;
        let own = overhead.own_ms as f64 / 1000.0 / s;
        record!(
            LogLevel::Info,
            "overhead_rtf",
            "overhead: RTF added {rtf:.4} over {s:.2} s of audio, runner's own share {own:.4} (budget {BUDGET_RTF} on \
             30 s or more: {})",
            if s >= 30.0 {
                within(own <= BUDGET_RTF)
            } else {
                "not judged, under 30 s"
            }
        );
    }
    if summary.complete() {
        record!(LogLevel::Info, "measurement_complete", "measurement: complete");
    } else {
        record!(
            LogLevel::Error,
            "measurement_incomplete",
            "measurement: INCOMPLETE: {}",
            summary.incomplete.join("; ")
        );
    }
    Ok(summary)
}

/// Run one instrument. The envelope to keep, how its process ended, and its time from spawn to exit.
fn one(
    config: &Config,
    entry: &Entry,
    inputs: &Inputs,
    bin_dir: &Path,
    dir: &Path,
    log_rel: &str,
    made: &dyn Fn(Status, String) -> Envelope,
) -> (Envelope, String, u128) {
    let fail = |status: Status, why: String, exit: &str, ms: u128| (made(status, why), exit.to_string(), ms);
    let (program, interpreter) = match (&entry.command, catalogue::find(&entry.name)) {
        (Some(c), _) => (PathBuf::from(c), entry.interpreter.clone()),
        (None, Some(k)) if k.interpreter.is_some() => match instruments_dir(inputs) {
            Some(d) => (
                d.join(k.program),
                Some(entry.interpreter.clone().unwrap_or_else(default_python)),
            ),
            None => {
                let why = "the shipped instruments/ directory was not found; set GGTTS_INSTRUMENTS_DIR".to_string();
                return fail(Status::Failed, why, "not started", 0);
            }
        },
        (None, Some(k)) => (
            bin_dir.join(format!("{}{}", k.program, std::env::consts::EXE_SUFFIX)),
            entry.interpreter.clone(),
        ),
        (None, None) => {
            return fail(
                Status::Failed,
                "not shipped and no command given".into(),
                "not started",
                0,
            )
        }
    };
    if let Err(e) = fs::create_dir_all(dir) {
        return fail(Status::Failed, format!("{}: {e}", dir.display()), "not started", 0);
    }
    let env_path = dir.join("envelope.json");
    let _ = fs::remove_file(&env_path);
    let log_path = dir.join(format!("{}.log", entry.name));
    let (out, err) = match File::create(&log_path).and_then(|f| f.try_clone().map(|g| (f, g))) {
        Ok(p) => p,
        Err(e) => return fail(Status::Failed, format!("{}: {e}", log_path.display()), "not started", 0),
    };
    let audio_before = fs::read(&inputs.audio).ok();

    // A script runs under its interpreter: Windows cannot start a `.py` file, and on every system the interpreter is
    // then the one asked for, not whatever the file's first line names.
    let mut cmd = match &interpreter {
        Some(i) => {
            let mut c = Command::new(i);
            c.arg(&program);
            c
        }
        None => Command::new(&program),
    };
    // A generic hint for every instrument; it is not a claim that an instrument used a GPU.
    // Device-aware instruments report their actual device and may support an explicit override.
    cmd.env("GGTTS_INSTRUMENT_DEVICE", config.device.as_deref().unwrap_or("cpu"));
    cmd.arg("--run").arg(&inputs.run).arg("--render").arg(&inputs.render);
    if let Some(ev) = &inputs.events {
        cmd.arg("--events").arg(ev);
    }
    cmd.arg("--audio")
        .arg(&inputs.audio)
        .arg("--envelope")
        .arg(&env_path)
        .arg("--name")
        .arg(&entry.name);
    for (k, v) in &entry.args {
        cmd.arg(format!("--{k}")).arg(v);
    }
    cmd.stdin(Stdio::null()).stdout(out).stderr(err);
    let t0 = Instant::now();
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let why = match &interpreter {
                Some(i) => format!("could not start {i} {}: {e}", program.display()),
                None => format!("could not start {}: {e}", program.display()),
            };
            return fail(Status::Failed, why, "not started", 0);
        }
    };
    let timeout = config.timeout_for(entry);
    let waited = wait(child, Duration::from_secs_f64(timeout));
    let ms = t0.elapsed().as_millis();

    // The crash boundary around the run's audio: whatever the instrument did, the render's bytes come back.
    let tampered = match (&audio_before, fs::read(&inputs.audio).ok()) {
        (Some(before), after) if after.as_deref() != Some(before.as_slice()) => {
            let _ = fs::write(&inputs.audio, before);
            true
        }
        _ => false,
    };

    let status = match waited {
        Waited::TimedOut => {
            let why = format!("no exit within {timeout} s; killed");
            return fail(Status::Timeout, why, "killed at the timeout", ms);
        }
        Waited::Error(e) => return fail(Status::Failed, format!("waiting for it: {e}"), "unknown", ms),
        Waited::Exited(s) => s,
    };
    let exit = describe(status);
    if tampered {
        let why = "it changed the run's audio; the render's bytes were put back".to_string();
        return fail(Status::Failed, why, &exit, ms);
    }
    if !status.success() {
        return fail(Status::Failed, format!("it {exit}"), &exit, ms);
    }
    let text = match fs::read_to_string(&env_path) {
        Ok(t) => t,
        Err(e) => {
            let why = format!("no envelope at {}: {e}", env_path.display());
            return fail(Status::Invalid, why, &exit, ms);
        }
    };
    let checked = serde_json::from_str::<Value>(&text)
        .map_err(|e| vec![format!("not JSON: {e}")])
        .and_then(|v| report::validate(&v));
    match checked {
        Err(why) => fail(
            Status::Invalid,
            format!("invalid envelope: {}", why.join("; ")),
            &exit,
            ms,
        ),
        Ok(env) if env.name != entry.name || env.identity.instrument != entry.name => {
            let why = format!(
                "invalid envelope: it names instrument {:?} ({:?}), not {:?}",
                env.name, env.identity.instrument, entry.name
            );
            fail(Status::Invalid, why, &exit, ms)
        }
        Ok(env) if env.identity.run != inputs.run => {
            let why = format!(
                "invalid envelope: it belongs to run {:?}, not {:?}",
                env.identity.run, inputs.run
            );
            fail(Status::Invalid, why, &exit, ms)
        }
        Ok(env) if env.status == Status::Timeout => {
            let why = "invalid envelope: an instrument may not report timeout; only the runner sets it".to_string();
            fail(Status::Invalid, why, &exit, ms)
        }
        Ok(mut env) => {
            env.log = Some(log_rel.into());
            (env, exit, ms)
        }
    }
}

/// `GGTTS_PYTHON`, else `python3`, or `python` on Windows (where the installer names it so).
pub fn default_python() -> String {
    std::env::var("GGTTS_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.to_string())
}

/// The shipped `instruments/` directory: [`Inputs::instruments_dir`], `GGTTS_INSTRUMENTS_DIR`, or the nearest
/// `instruments/` holding the Python SDK above the running binary.
fn instruments_dir(inputs: &Inputs) -> Option<PathBuf> {
    if let Some(d) = &inputs.instruments_dir {
        return Some(d.clone());
    }
    if let Some(d) = std::env::var_os("GGTTS_INSTRUMENTS_DIR") {
        return Some(PathBuf::from(d));
    }
    let exe = std::env::current_exe().ok()?;
    exe.ancestors()
        .skip(1)
        .map(|a| a.join("instruments"))
        .find(|d| d.join("python/ggtts_instrument").is_dir())
}

enum Waited {
    Exited(ExitStatus),
    TimedOut,
    Error(std::io::Error),
}

fn wait(mut child: Child, timeout: Duration) -> Waited {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(s)) => return Waited::Exited(s),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Waited::TimedOut;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => return Waited::Error(e),
        }
    }
}

fn describe(s: ExitStatus) -> String {
    match s.code() {
        Some(c) => format!("exited with code {c}"),
        None => "was terminated by a signal".into(),
    }
}

/// Run a trusted in-process hook (a tensor hook) and turn its outcome into an envelope for `run`.
///
/// A panic is caught and reported as `failed` with its message, and the run goes on. An `Err` is `failed` too.
/// This is **not** the child-process boundary: there is no timeout, so a hook that hangs hangs the run, and a hook
/// that aborts the process ends it. Only code behind a feature, reviewed as part of the engine, belongs here.
pub fn in_process<F>(run: &str, name: &str, version: &str, hooks: &[&str], f: F) -> Envelope
where
    F: FnOnce() -> Result<Vec<Metric>, String> + UnwindSafe,
{
    let (status, metrics, error) = match catch_unwind(f) {
        Ok(Ok(m)) => (Status::Ok, m, None),
        Ok(Err(e)) => (Status::Failed, Vec::new(), Some(e)),
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".into());
            (Status::Failed, Vec::new(), Some(format!("panicked: {msg}")))
        }
    };
    Envelope {
        name: name.into(),
        version: version.into(),
        format: FORMAT.into(),
        identity: Identity {
            run: run.into(),
            instrument: name.into(),
        },
        provenance: Provenance {
            config: Map::new(),
            ggtts: Some(env!("CARGO_PKG_VERSION").into()),
            engine: None,
            backend: None,
            model_sha256: None,
            seeds: Vec::new(),
            inputs: Vec::new(),
            audio: AudioState::NotApplicable,
        },
        hooks: hooks.iter().map(|h| h.to_string()).collect(),
        status,
        metrics,
        artifacts: Vec::new(),
        log: None,
        error,
    }
}
