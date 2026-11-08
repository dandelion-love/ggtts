//! Consistent CLI and runner log records. Rendered audio and data output never share this sink.
use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_EVENTS: AtomicU64 = AtomicU64::new(0);
static LOG_SINK_NS: AtomicU64 = AtomicU64::new(0);
static LOG_LOCK_WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// Process-local logger cost. Excludes suppressed calls, caller-side message preparation and native work outside callback.
pub fn stats() -> (u64, u64, u64) {
    (
        LOG_EVENTS.load(Ordering::Relaxed),
        LOG_SINK_NS.load(Ordering::Relaxed),
        LOG_LOCK_WAIT_NS.load(Ordering::Relaxed),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "error" => Some(Self::Error),
            "warn" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    pub level: Level,
    pub format: Format,
}

/// A configuration failure, with the usable output format selected so far.
#[derive(Debug)]
pub struct ConfigError {
    pub message: String,
    pub settings: Settings,
}

fn config_error(message: &str, format: &str) -> ConfigError {
    ConfigError {
        message: message.into(),
        settings: Settings {
            level: Level::Error,
            format: if format == "json" { Format::Json } else { Format::Text },
        },
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            level: Level::Info,
            format: Format::Text,
        }
    }
}

impl Settings {
    /// Consume shared flags; explicit flags override environment settings. Errors are never suppressible.
    pub fn from_args(args: &[String]) -> Result<(Self, Vec<String>), ConfigError> {
        let mut level = std::env::var("GGTTS_LOG_LEVEL").unwrap_or_else(|_| "info".into());
        let mut format = std::env::var("GGTTS_LOG_FORMAT").unwrap_or_else(|_| "text".into());
        let mut rest = Vec::new();
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--log-level" => {
                    level = iter
                        .next()
                        .ok_or_else(|| config_error("--log-level needs a value", &format))?
                        .clone()
                }
                "--log-format" => {
                    format = iter
                        .next()
                        .ok_or_else(|| config_error("--log-format needs a value", &format))?
                        .clone()
                }
                _ => {
                    rest.push(arg.clone());
                    // Values may themselves begin with `--`; never reinterpret text or paths as logging flags.
                    if matches!(
                        arg.as_str(),
                        "--model"
                            | "--codec"
                            | "--out"
                            | "-o"
                            | "--engine"
                            | "--text"
                            | "-t"
                            | "--story"
                            | "--seed"
                            | "--threads"
                            | "-j"
                            | "--device"
                            | "--trace"
                            | "--trace-level"
                            | "--timing-report"
                            | "--voice"
                            | "--stream"
                            | "--stream-format"
                            | "--stream-level-seconds"
                            | "--report"
                            | "--run"
                            | "--audio"
                            | "--audio-state"
                            | "--events"
                            | "--render"
                            | "--instrument"
                            | "--instruments"
                            | "--require"
                            | "--on-instrument-error"
                            | "--instrument-timeout"
                    ) {
                        if let Some(value) = iter.next() {
                            rest.push(value.clone());
                        }
                    }
                }
            }
        }
        let level = Level::parse(&level)
            .ok_or_else(|| config_error("log level must be error, warn, info, debug or trace", &format))?;
        let format = match format.as_str() {
            "text" => Format::Text,
            "json" => Format::Json,
            _ => return Err(config_error("log format must be text or json", &format)),
        };
        Ok((Self { level, format }, rest))
    }

    pub fn enabled(self, level: Level) -> bool {
        level <= self.level
    }

    /// Emit a single record. JSON is one object per line; messages should not include input text or secrets.
    pub fn write(
        self,
        out: &mut dyn Write,
        level: Level,
        component: &str,
        event: &str,
        message: fmt::Arguments<'_>,
    ) -> io::Result<()> {
        self.write_fields(out, level, component, event, message, serde_json::Value::Null)
    }

    pub fn write_fields(
        self,
        out: &mut dyn Write,
        level: Level,
        component: &str,
        event: &str,
        message: fmt::Arguments<'_>,
        fields: serde_json::Value,
    ) -> io::Result<()> {
        if !self.enabled(level) {
            return Ok(());
        }
        let start = Instant::now();
        let result = match self.format {
            Format::Text if component.is_empty() => writeln!(out, "{message}"),
            Format::Text => writeln!(out, "{component}: {message}"),
            Format::Json => {
                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .and_then(|d| u64::try_from(d.as_millis()).ok());
                let record = serde_json::json!({"format":"ggtts-log/1", "time_unix_ms":timestamp_ms,
                    "level":level.name(), "component":component, "event":event, "message":message.to_string(),
                    "fields":fields});
                writeln!(out, "{record}")
            }
        };
        LOG_SINK_NS.fetch_add(
            start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        if result.is_ok() {
            LOG_EVENTS.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    pub fn stderr(self, level: Level, component: &str, event: &str, message: fmt::Arguments<'_>) {
        if !self.enabled(level) {
            return;
        }
        let start = Instant::now();
        let stderr = io::stderr();
        let mut locked = stderr.lock();
        LOG_LOCK_WAIT_NS.fetch_add(
            start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        let _ = self.write(&mut locked, level, component, event, message);
    }

    pub fn stderr_fields(
        self,
        level: Level,
        component: &str,
        event: &str,
        message: fmt::Arguments<'_>,
        fields: serde_json::Value,
    ) {
        if !self.enabled(level) {
            return;
        }
        let start = Instant::now();
        let stderr = io::stderr();
        let mut locked = stderr.lock();
        LOG_LOCK_WAIT_NS.fetch_add(
            start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        let _ = self.write_fields(&mut locked, level, component, event, message, fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_filter_without_hiding_errors_and_json_is_one_record() {
        let settings = Settings {
            level: Level::Warn,
            format: Format::Json,
        };
        let mut out = Vec::new();
        settings
            .write(&mut out, Level::Info, "test", "progress", format_args!("quiet"))
            .unwrap();
        assert!(out.is_empty());
        settings
            .write(&mut out, Level::Error, "test", "failure", format_args!("line\none"))
            .unwrap();
        let text = std::str::from_utf8(&out).unwrap();
        assert_eq!(text.lines().count(), 1);
        let record: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(record["format"], "ggtts-log/1");
        assert_eq!(record["level"], "error");
        assert_eq!(record["message"], "line\none");
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../schemas/ggtts-log-1.schema.json")).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&record));
        let mut planted = record.clone();
        planted["level"] = "made-up".into();
        assert!(
            !validator.is_valid(&planted),
            "the published schema must reject invalid severity"
        );
    }

    #[test]
    fn flags_do_not_consume_a_text_value_that_looks_like_a_flag() {
        let args = ["--text", "--log-level", "--log-level", "debug", "--log-format", "json"].map(str::to_string);
        let (settings, rest) = Settings::from_args(&args).unwrap();
        assert_eq!(settings.level, Level::Debug);
        assert_eq!(settings.format, Format::Json);
        assert_eq!(rest, ["--text", "--log-level"]);
    }
}
