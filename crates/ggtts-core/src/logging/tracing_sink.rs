//! `tracing` dispatch and subscriber-backed output for the CLI and instrument runner.
use super::{Format, Level, Settings, LOG_EVENTS, LOG_LOCK_WAIT_NS, LOG_SINK_NS};
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::{FmtContext, MakeWriter};
use tracing_subscriber::registry::LookupSpan;

type Output = Arc<Mutex<Box<dyn Write + Send>>>;

#[derive(Clone)]
struct MeasuredWriter {
    output: Output,
    failures: Arc<AtomicU64>,
}

struct Guard<'a> {
    output: MutexGuard<'a, Box<dyn Write + Send>>,
    failures: &'a AtomicU64,
}

impl Write for Guard<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = self.output.write(bytes);
        if result.is_err() || matches!(result, Ok(0)) && !bytes.is_empty() {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        let result = self.output.flush();
        if result.is_err() {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

impl<'a> MakeWriter<'a> for MeasuredWriter {
    type Writer = Guard<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        let start = Instant::now();
        let guard = self.output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        LOG_LOCK_WAIT_NS.fetch_add(elapsed_ns(start), Ordering::Relaxed);
        Guard {
            output: guard,
            failures: &self.failures,
        }
    }
}

#[derive(Default)]
struct Fields {
    component: String,
    event: String,
    message: String,
    fields_json: String,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "component" => self.component = value.into(),
            "event" => self.event = value.into(),
            "message" => self.message = value.into(),
            "fields_json" => self.fields_json = value.into(),
            _ => {}
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

struct GgttsFormat(Format);

impl<S, N> FormatEvent<S, N> for GgttsFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, _ctx: &FmtContext<'_, S, N>, mut writer: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let mut fields = Fields::default();
        event.record(&mut fields);
        match self.0 {
            Format::Text if fields.component.is_empty() => writeln!(writer, "{}", fields.message),
            Format::Text => writeln!(writer, "{}: {}", fields.component, fields.message),
            Format::Json => {
                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .and_then(|d| u64::try_from(d.as_millis()).ok());
                let extra: serde_json::Value = serde_json::from_str(&fields.fields_json).map_err(|_| fmt::Error)?;
                let level = event.metadata().level().as_str().to_ascii_lowercase();
                let record = serde_json::json!({"format":"ggtts-log/1", "time_unix_ms":timestamp_ms,
                    "level":level, "component":fields.component, "event":fields.event,
                    "message":fields.message, "fields":extra});
                writeln!(writer, "{record}")
            }
        }
    }
}

/// A reusable tracing subscriber and its output destination.
pub struct LogSink {
    settings: Settings,
    dispatch: tracing::Dispatch,
    failures: Arc<AtomicU64>,
}

impl LogSink {
    pub fn new(settings: Settings, output: impl Write + Send + 'static) -> Self {
        Self::from_output(settings, Arc::new(Mutex::new(Box::new(output))))
    }

    fn from_output(settings: Settings, output: Output) -> Self {
        let failures = Arc::new(AtomicU64::new(0));
        let writer = MeasuredWriter {
            output,
            failures: failures.clone(),
        };
        let subscriber = tracing_subscriber::fmt()
            .log_internal_errors(false)
            .event_format(GgttsFormat(settings.format))
            .with_writer(writer)
            .with_max_level(tracing::Level::TRACE)
            .finish();
        Self {
            settings,
            dispatch: tracing::Dispatch::new(subscriber),
            failures,
        }
    }

    /// Number of failed destination writes or flushes observed by the subscriber.
    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    pub fn emit(
        &self,
        level: Level,
        component: &str,
        event: &str,
        message: fmt::Arguments<'_>,
        fields: serde_json::Value,
    ) {
        if !self.settings.enabled(level) {
            return;
        }
        let start = Instant::now();
        let fields_json = fields.to_string();
        tracing::dispatcher::with_default(&self.dispatch, || {
            macro_rules! emit {
                ($level:expr) => {
                    tracing::event!($level, component = component, event = event, fields_json = fields_json.as_str(), message = %message)
                };
            }
            match level {
                Level::Error => emit!(tracing::Level::ERROR),
                Level::Warn => emit!(tracing::Level::WARN),
                Level::Info => emit!(tracing::Level::INFO),
                Level::Debug => emit!(tracing::Level::DEBUG),
                Level::Trace => emit!(tracing::Level::TRACE),
            }
        });
        LOG_SINK_NS.fetch_add(elapsed_ns(start), Ordering::Relaxed);
        LOG_EVENTS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(super) fn stderr(settings: Settings) -> Arc<LogSink> {
    static SINKS: OnceLock<Mutex<HashMap<Settings, Arc<LogSink>>>> = OnceLock::new();
    static STDERR: OnceLock<Output> = OnceLock::new();
    let start = Instant::now();
    let mut sinks = SINKS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    LOG_LOCK_WAIT_NS.fetch_add(elapsed_ns(start), Ordering::Relaxed);
    let sink = sinks
        .entry(settings)
        .or_insert_with(|| {
            let output = STDERR.get_or_init(|| Arc::new(Mutex::new(Box::new(io::stderr()))));
            Arc::new(LogSink::from_output(settings, output.clone()))
        })
        .clone();
    LOG_SINK_NS.fetch_add(elapsed_ns(start), Ordering::Relaxed);
    sink
}

#[derive(Clone, Default)]
struct Buffer(Arc<Mutex<Vec<u8>>>);
impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn write_to(
    settings: Settings,
    out: &mut dyn Write,
    level: Level,
    component: &str,
    event: &str,
    message: fmt::Arguments<'_>,
    fields: serde_json::Value,
) -> io::Result<()> {
    if !settings.enabled(level) {
        return Ok(());
    }
    let buffer = Buffer::default();
    LogSink::new(settings, buffer.clone()).emit(level, component, event, message, fields);
    let bytes = buffer.0.lock().unwrap_or_else(|e| e.into_inner());
    out.write_all(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_subscribers_filter_and_keep_structured_fields() {
        let a = Buffer::default();
        let b = Buffer::default();
        let quiet = LogSink::new(
            Settings {
                level: Level::Error,
                format: Format::Json,
            },
            a.clone(),
        );
        let verbose = LogSink::new(
            Settings {
                level: Level::Trace,
                format: Format::Json,
            },
            b.clone(),
        );
        quiet.emit(
            Level::Info,
            "quiet",
            "ignored",
            format_args!("no"),
            serde_json::Value::Null,
        );
        verbose.emit(
            Level::Trace,
            "verbose",
            "detail",
            format_args!("value {}", 42),
            serde_json::json!({"n": 3}),
        );
        quiet.emit(
            Level::Error,
            "quiet",
            "failure",
            format_args!("bad"),
            serde_json::Value::Null,
        );
        let quiet_text = String::from_utf8(a.0.lock().unwrap().clone()).unwrap();
        let verbose_text = String::from_utf8(b.0.lock().unwrap().clone()).unwrap();
        let q: serde_json::Value = serde_json::from_str(&quiet_text).unwrap();
        let v: serde_json::Value = serde_json::from_str(&verbose_text).unwrap();
        assert_eq!(q["event"], "failure");
        assert_eq!(q["level"], "error");
        assert_eq!(v["level"], "trace");
        assert_eq!(v["message"], "value 42");
        assert_eq!(v["fields"]["n"], 3);
    }

    #[test]
    fn borrowed_writer_failure_is_returned() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("broken"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let result = write_to(
            Settings::default(),
            &mut Broken,
            Level::Error,
            "cli",
            "error",
            format_args!("failure"),
            serde_json::Value::Null,
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Other);
    }

    #[test]
    fn real_subscriber_destination_failure_is_observable() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("broken"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let sink = LogSink::new(Settings::default(), Broken);
        sink.emit(
            Level::Error,
            "test",
            "failure",
            format_args!("failure"),
            serde_json::Value::Null,
        );
        assert_eq!(sink.failures(), 1);
    }

    #[test]
    fn zero_length_destination_write_is_observable() {
        struct Zero;
        impl Write for Zero {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Ok(0)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let sink = LogSink::new(Settings::default(), Zero);
        sink.emit(
            Level::Error,
            "test",
            "failure",
            format_args!("failure"),
            serde_json::Value::Null,
        );
        assert_eq!(sink.failures(), 1);
    }

    #[test]
    fn slow_destination_is_reflected_in_sink_timing() {
        struct Slow(Buffer);
        impl Write for Slow {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                std::thread::sleep(std::time::Duration::from_millis(15));
                self.0.write(bytes)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let buffer = Buffer::default();
        let sink = LogSink::new(Settings::default(), Slow(buffer.clone()));
        let before = LOG_SINK_NS.load(Ordering::Relaxed);
        sink.emit(
            Level::Info,
            "test",
            "slow",
            format_args!("control"),
            serde_json::Value::Null,
        );
        let elapsed = LOG_SINK_NS.load(Ordering::Relaxed) - before;
        assert!(
            elapsed >= 10_000_000,
            "sink timing missed the planted delay: {elapsed} ns"
        );
        assert_eq!(&*buffer.0.lock().unwrap(), b"test: control\n");
    }
}
