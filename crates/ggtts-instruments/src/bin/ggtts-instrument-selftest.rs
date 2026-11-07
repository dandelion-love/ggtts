//! ggtts-instrument-selftest: the runner's own instrument, and its planted failures.
//!
//! ```text
//! ggtts-instrument-selftest --run RUN --render RENDER [--events E.jsonl] --audio A.flac --envelope OUT.json
//!                           --name NAME [--mode MODE]
//! ```
//!
//! `--mode ok` (the default) counts the event lines and the audio's bytes and writes a valid `ggtts-report/1`
//! envelope. The other modes each plant one failure the runner must contain: `crash` panics, `hang` never exits,
//! `bad-envelope` writes an envelope of the wrong schema, and `tamper` appends to the run's audio.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_core::report::{AudioState, Direction, Envelope, Identity, Metric, Provenance, Status, FORMAT};
use ggtts_instruments::runner::input_file;
use serde_json::{json, Map};

fn main() -> ExitCode {
    let mut run = None;
    let mut render = None;
    let mut events: Option<PathBuf> = None;
    let mut audio: Option<PathBuf> = None;
    let mut envelope: Option<PathBuf> = None;
    let mut name = "selftest".to_string();
    let mut mode = "ok".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            eprintln!("{flag} needs a value");
            return ExitCode::from(2);
        };
        match flag.as_str() {
            "--run" => run = Some(value),
            "--render" => render = Some(value),
            "--events" => events = Some(value.into()),
            "--audio" => audio = Some(value.into()),
            "--envelope" => envelope = Some(value.into()),
            "--name" => name = value,
            "--mode" => mode = value,
            other => {
                eprintln!("invalid argument: {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(run), Some(render), Some(audio), Some(envelope)) = (run, render, audio, envelope) else {
        eprintln!("--run, --render, --audio and --envelope are required");
        return ExitCode::from(2);
    };
    eprintln!(
        "selftest: mode {mode}, run {run}, render {render}, audio {}",
        audio.display()
    );
    match mode.as_str() {
        "crash" => panic!("selftest: planted crash"),
        "hang" => loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        },
        "bad-envelope" => {
            let bad = json!({"format": "ggtts-report/0", "name": name, "status": "ok",
                "metrics": [{"name": "bytes", "value": 1}]});
            return write(&envelope, &bad.to_string());
        }
        "tamper" => {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&audio) {
                let _ = f.write_all(b"tampered");
            }
        }
        "ok" => {}
        other => {
            eprintln!("unknown --mode {other}");
            return ExitCode::from(2);
        }
    }
    let neutral = |metric: &str, value: f64, unit: &str| Metric {
        name: metric.into(),
        value: Some(value),
        unit: unit.into(),
        direction: Direction::Neutral,
        interval: None,
        render: Some(render.clone()),
        chunk: None,
    };
    let bytes = std::fs::metadata(&audio).map(|m| m.len()).unwrap_or(0);
    let mut metrics = vec![neutral("audio_bytes", bytes as f64, "bytes")];
    let mut inputs: Vec<_> = input_file(&audio).into_iter().collect();
    if let Some(ev) = &events {
        let n = std::fs::read_to_string(ev)
            .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);
        metrics.push(neutral("events", n as f64, "lines"));
        inputs.extend(input_file(ev));
    }
    let mut config = Map::new();
    config.insert("mode".into(), json!(mode));
    let env = Envelope {
        name: name.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        format: FORMAT.into(),
        identity: Identity { run, instrument: name },
        provenance: Provenance {
            config,
            ggtts: Some(env!("CARGO_PKG_VERSION").into()),
            engine: None,
            backend: None,
            model_sha256: None,
            seeds: Vec::new(),
            inputs,
            audio: AudioState::Raw,
        },
        hooks: vec!["per-run".into()],
        status: Status::Ok,
        metrics,
        artifacts: Vec::new(),
        log: None,
        error: None,
    };
    write(&envelope, &env.to_json().to_string())
}

fn write(path: &Path, text: &str) -> ExitCode {
    match std::fs::write(path, text) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            ExitCode::from(1)
        }
    }
}
