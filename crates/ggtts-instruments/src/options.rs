//! The instrument run, from the shared options value ([`ggtts_core::options`]).
//!
//! The instrument options (`--instrument`, `--instruments`, `--require`, `--instrument-device`,
//! `--on-instrument-error`, `--instrument-timeout`, `--report`, `--run`) are observer entries of the options value.
//! Every front end that runs instruments makes the same two calls: [`config`] before rendering, so a bad instrument
//! config is refused before the render is spent on it, and [`run`] on each finished audio file. `ggtts-render` runs
//! once per render; `ggtts-book` once per chapter it wrote, with the chapter's FLAC and identity.

use std::path::{Path, PathBuf};

use ggtts_core::logging::Settings as LogSettings;
use ggtts_core::options::Options;
use ggtts_core::report::AudioState;
use serde_json::{Map, Value};

use crate::config::{Config, Flags};
use crate::runner::{self, Inputs, RunSummary};

/// The instrument options of `o` as [`Flags`]: the file, then each option's values, every list in its order.
/// [`Flags::resolve`] checks them and applies them over the file.
pub fn flags(o: &Options) -> Flags {
    let o = &o.observe;
    let mut f = Flags::default();
    if let Some(p) = &o.instruments {
        f.take("--instruments", p.to_string_lossy().into_owned());
    }
    for v in &o.instrument {
        f.take("--instrument", v.clone());
    }
    for v in &o.require {
        f.take("--require", v.clone());
    }
    if let Some(v) = &o.on_instrument_error {
        f.take("--on-instrument-error", v.clone());
    }
    if let Some(v) = o.instrument_timeout {
        f.take("--instrument-timeout", v.to_string());
    }
    if let Some(v) = &o.instrument_device {
        f.take("--instrument-device", v.clone());
    }
    f
}

/// The config `o` asks for, its device defaulting to the render's (`load.device`, else `cpu`). Not checked: see
/// [`config`].
pub fn resolve(o: &Options) -> Result<Config, String> {
    let mut config = flags(o).resolve()?;
    if config.device.is_none() {
        config.set_device(o.load.device.as_deref().unwrap_or("cpu"))?;
    }
    Ok(config)
}

/// The instrument config `o` asks for, refused if an instrument cannot be found ([`runner::check`]); `None` when no
/// instrument is asked for or required. Call it before rendering.
pub fn config(o: &Options) -> Result<Option<Config>, String> {
    if !flags(o).any() {
        return Ok(None);
    }
    let c = resolve(o)?;
    runner::check(&c)?;
    Ok(Some(c))
}

/// The run's id: `observe.run`, else `<render>-<unix seconds>`. A front end that runs several files in one run (a
/// book's chapters) takes it once and passes it to every [`run`].
pub fn run_id(o: &Options, render: &str) -> String {
    o.observe.run.clone().unwrap_or_else(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("{render}-{secs}")
    })
}

/// A finished audio file to measure, and what the instruments are told about it.
#[derive(Debug, Clone)]
pub struct Finished {
    /// The run's id ([`run_id`]).
    pub run: String,
    /// The render's id: the file's name for `ggtts-render`, the chapter's for a book.
    pub render: String,
    /// The audio file.
    pub audio: PathBuf,
    /// Whether its edges were trimmed.
    pub audio_state: AudioState,
    /// Its length, when known.
    pub audio_seconds: Option<f64>,
    /// The run's event stream (the trace), when it has one that covers exactly this file.
    pub events: Option<PathBuf>,
    /// The model files it was rendered with.
    pub models: Option<Vec<Value>>,
    /// The render's reported provenance ([`ggtts_core::options::reported`]): the instruction and steering.
    pub reported: Map<String, Value>,
    /// The report directory when `observe.report` is not set.
    pub default_report_dir: PathBuf,
}

impl Finished {
    /// The runner's inputs: the report directory is `report_dir` when given, else the file's own default.
    pub fn inputs(&self, report_dir: Option<&Path>) -> Inputs {
        Inputs {
            run: self.run.clone(),
            render: self.render.clone(),
            report_dir: report_dir.map_or_else(|| self.default_report_dir.clone(), Path::to_path_buf),
            audio: self.audio.clone(),
            audio_state: self.audio_state,
            audio_seconds: self.audio_seconds,
            events: self.events.clone(),
            bin_dir: None,
            instruments_dir: None,
            models: self.models.clone(),
            instruction: self
                .reported
                .get("instruction")
                .and_then(Value::as_str)
                .map(str::to_string),
            steer: self.reported.get("steer").cloned(),
        }
    }
}

/// Run `config`'s instruments on `finished`, writing the report into `report_dir` (else the file's default). The
/// summary decides the exit: [`RunSummary::exit_code`].
pub fn run(
    config: &Config,
    finished: &Finished,
    report_dir: Option<&Path>,
    log: LogSettings,
) -> Result<(Inputs, RunSummary), String> {
    let inputs = finished.inputs(report_dir);
    let summary = runner::run_with_logging(config, &inputs, log)?;
    Ok((inputs, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(args: &[&str]) -> Options {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        Options::from_args(&args).unwrap()
    }

    #[test]
    fn no_instrument_is_no_config() {
        assert!(config(&opts(&[])).unwrap().is_none());
        // A device or a timeout alone asks for nothing.
        assert!(config(&opts(&["--instrument-device", "cpu"])).unwrap().is_none());
    }

    #[test]
    fn an_unknown_instrument_is_refused() {
        let e = config(&opts(&["--instrument", "no-such-instrument"])).unwrap_err();
        assert!(e.contains("\"no-such-instrument\" is not shipped"), "{e}");
    }

    #[test]
    fn the_device_inherits_the_render_and_a_flag_overrides_it() {
        let device = |args: &[&str]| resolve(&opts(args)).unwrap().device;
        assert_eq!(device(&[]).as_deref(), Some("cpu"));
        assert_eq!(device(&["--device", "cuda"]).as_deref(), Some("cuda"));
        assert_eq!(
            device(&["--device", "cuda", "--instrument-device", "cpu"]).as_deref(),
            Some("cpu")
        );
    }

    #[test]
    fn the_run_id_is_the_option_else_the_render_and_a_time() {
        assert_eq!(run_id(&opts(&["--run", "r1"]), "x"), "r1");
        assert!(run_id(&opts(&[]), "x").starts_with("x-"));
    }
}
