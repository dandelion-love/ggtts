//! The render's judge, from the shared options value: the duration judge alone, or with an ASR that reads each draw
//! back ([`ggtts_core::textjudge`]).
//!
//! `render.judge` (`--judge`) names the ASR and its settings: `asr-qwen,model=DIR[,device=cpu|cuda][,min_run=N]`.
//! Unset, a render keeps the duration judge alone, as before. Every front end that redraws (`ggtts-render --story`,
//! `ggtts-book render`) builds its judge with [`judge`] before rendering, so an ASR that cannot start (no model, no
//! Python, no `qwen-asr`) refuses the render in the same words everywhere, before anything is drawn.
//!
//! The ASR runs as one child process for the whole render ([`AsrRecogniser`]): the shipped
//! `instruments/examples/python/asr-qwen/judge_worker.py`, which loads the model once and reads one draw per request.
//! Each draw is written to a FLAC file in a temporary directory, its path sent on the worker's stdin as a JSON line,
//! and the transcript read back from its stdout. `command=` and `interpreter=` run another program speaking the same
//! protocol.
//!
//! **A worker that dies or stops answering ends the render.** Each answer has a deadline: `timeout=S` per draw (120 s
//! unless set) and `start_timeout=S` for the model to load (600 s). A worker that exits, closes its output, writes
//! something that is not a JSON line, or misses a deadline (it is then killed) is a fault
//! ([`ggtts_core::textjudge::Recogniser::fault`]): the pipeline stops the render there, so no draw it could not judge
//! is kept as passed, and a book's resume renders that chunk again. A worker that answers `{"error": ...}` for one
//! draw is not a fault: that draw is kept, with the error in its reading.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use ggtts_core::options::Options;
use ggtts_core::pipeline::{DurationJudge, Judge};
use ggtts_core::textjudge::{Fold, Recogniser, TextJudge, DEFAULT_MIN_RUN};
use serde_json::Value;

/// The ASRs a judge can run, by name.
pub const JUDGES: &[&str] = &["asr-qwen"];

/// A parsed `render.judge`.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeSpec {
    /// The ASR: one of [`JUDGES`].
    pub name: String,
    /// The local model directory the ASR loads.
    pub model: PathBuf,
    /// Where the ASR runs: `cpu` or `cuda`.
    pub device: String,
    /// The shortest lost or repeated run that rejects a draw.
    pub min_run: usize,
    /// Another program to run in place of the shipped worker, speaking its protocol.
    pub command: Option<String>,
    /// What runs `command` (`python` means [`crate::runner::default_python`]).
    pub interpreter: Option<String>,
    /// Seconds the ASR may take to read one draw back.
    pub timeout_s: f64,
    /// Seconds the ASR may take to say it is ready (its model loaded).
    pub start_timeout_s: f64,
}

/// [`JudgeSpec::timeout_s`] unless set.
pub const DEFAULT_TIMEOUT_S: f64 = 120.0;
/// [`JudgeSpec::start_timeout_s`] unless set.
pub const DEFAULT_START_TIMEOUT_S: f64 = 600.0;

/// Read `render.judge`: `NAME,model=DIR[,device=cpu|cuda][,min_run=N][,timeout=S][,start_timeout=S]
/// [,command=PROGRAM][,interpreter=PROGRAM]`.
pub fn parse(spec: &str) -> Result<JudgeSpec, String> {
    let bad = |why: String| format!("--judge (render.judge): {why}");
    let mut parts = spec.split(',');
    let name = parts.next().unwrap_or_default().trim().to_string();
    if !JUDGES.contains(&name.as_str()) {
        return Err(bad(format!("{name:?} is not a judge (judges: {})", JUDGES.join(", "))));
    }
    let mut model = None;
    let mut device = "cpu".to_string();
    let mut min_run = DEFAULT_MIN_RUN;
    let (mut command, mut interpreter) = (None, None);
    let (mut timeout_s, mut start_timeout_s) = (DEFAULT_TIMEOUT_S, DEFAULT_START_TIMEOUT_S);
    let seconds = |k: &str, v: &str| {
        v.parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| bad(format!("{k} must be a number of seconds above 0, not {v:?}")))
    };
    for p in parts {
        let (k, v) = p
            .split_once('=')
            .ok_or_else(|| bad(format!("{p:?} is not KEY=VALUE")))?;
        match k {
            "model" if !v.is_empty() => model = Some(PathBuf::from(v)),
            "device" if v == "cpu" || v == "cuda" => device = v.to_string(),
            "device" => return Err(bad(format!("device must be cpu or cuda, not {v:?}"))),
            "min_run" => {
                min_run = v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| bad(format!("min_run must be a whole number above 0, not {v:?}")))?
            }
            "timeout" => timeout_s = seconds(k, v)?,
            "start_timeout" => start_timeout_s = seconds(k, v)?,
            "command" if !v.is_empty() => command = Some(v.to_string()),
            "interpreter" if !v.is_empty() => interpreter = Some(v.to_string()),
            _ => {
                return Err(bad(format!(
                    "unknown or empty key {k:?} (keys: model, device, min_run, timeout, start_timeout, command, interpreter)"
                )))
            }
        }
    }
    let model = model.ok_or_else(|| bad(format!("{name} needs model=DIR, a local model directory")))?;
    Ok(JudgeSpec {
        name,
        model,
        device,
        min_run,
        command,
        interpreter,
        timeout_s,
        start_timeout_s,
    })
}

/// The render's judge from `o`: the duration judge alone when `render.judge` is unset, else a text judge reading
/// every draw the duration judge keeps back with the named ASR, started now. Refused (with the reason, under
/// `render.judge`) when the spec does not read, the script fold does not read, or the ASR does not start.
pub fn judge(o: &Options) -> Result<Box<dyn Judge>, String> {
    let Some(spec) = &o.render.judge else {
        return Ok(Box::new(DurationJudge));
    };
    let spec = parse(spec)?;
    if o.render.redraw.unwrap_or(0) == 0 {
        return Err(
            "--judge (render.judge): the judge rejects a draw so that it is drawn again; give --redraw N \
                    (N above 0)"
                .into(),
        );
    }
    let fold = Fold::t2s().map_err(|e| format!("--judge (render.judge): {e}"))?;
    let asr =
        AsrRecogniser::start(&spec).map_err(|e| format!("--judge (render.judge): {} did not start: {e}", spec.name))?;
    let mut judge = TextJudge::new(asr, fold);
    judge.min_run = spec.min_run;
    Ok(Box::new(judge))
}

/// The shipped worker for an ASR, under the `instruments/` directory.
fn worker_script(name: &str) -> String {
    format!("examples/python/{name}/judge_worker.py")
}

/// An ASR running as a child process for the whole render, reading one draw per request.
pub struct AsrRecogniser {
    child: Child,
    stdin: ChildStdin,
    /// The worker's stdout, a line at a time, from a reader thread, so a read can have a deadline.
    lines: Receiver<std::io::Result<String>>,
    dir: PathBuf,
    n: u64,
    timeout: Duration,
    /// Why the worker cannot read any more draws, once it cannot.
    fault: Option<String>,
}

impl AsrRecogniser {
    /// Start the ASR `spec` names and wait, up to its `start_timeout`, for it to say it is ready (its model loaded).
    /// Its stderr is discarded: an ASR's own messages can contain the speech it heard.
    pub fn start(spec: &JudgeSpec) -> Result<Self, String> {
        let (program, mut args) = match (&spec.command, &spec.interpreter) {
            (Some(c), Some(i)) => (interpreter(i), vec![c.clone()]),
            (Some(c), None) => (c.clone(), Vec::new()),
            (None, _) => {
                let dir = crate::runner::shipped_dir()
                    .ok_or("the shipped instruments/ directory was not found; set GGTTS_INSTRUMENTS_DIR")?;
                let script = dir.join(worker_script(&spec.name));
                if !script.is_file() {
                    return Err(format!("{} is missing", script.display()));
                }
                (
                    crate::runner::default_python(),
                    vec![script.to_string_lossy().into_owned()],
                )
            }
        };
        args.extend([
            "--model".to_string(),
            spec.model.to_string_lossy().into_owned(),
            "--device".to_string(),
            spec.device.clone(),
        ]);
        let dir = std::env::temp_dir().join(format!("ggtts-judge-{}-{}", std::process::id(), unique()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            // The protocol is UTF-8 JSON lines; a Python on Windows otherwise reads and writes its console code page.
            .env("PYTHONIOENCODING", "utf-8")
            .spawn()
            .map_err(|e| format!("{program}: {e}"))?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = BufReader::new(child.stdout.take().expect("piped"));
        let (tx, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut stdout = stdout;
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        });
        let mut asr = Self {
            child,
            stdin,
            lines,
            dir,
            n: 0,
            timeout: Duration::from_secs_f64(spec.timeout_s),
            fault: None,
        };
        let first = asr.read_line(Duration::from_secs_f64(spec.start_timeout_s))?;
        match first.get("ready") {
            Some(Value::Bool(true)) => Ok(asr),
            _ => Err(first
                .get("error")
                .and_then(Value::as_str)
                .map_or_else(|| format!("it said {first}, not ready"), str::to_string)),
        }
    }

    /// The worker's next line, within `deadline`. A worker that misses it is killed.
    fn read_line(&mut self, deadline: Duration) -> Result<Value, String> {
        match self.lines.recv_timeout(deadline) {
            Ok(Ok(line)) => {
                serde_json::from_str(&line).map_err(|e| format!("the ASR wrote a line that is not JSON: {e}"))
            }
            Ok(Err(e)) => Err(format!("reading the ASR: {e}")),
            Err(RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                Err(format!(
                    "the ASR did not answer within {} s, and was stopped",
                    deadline.as_secs_f64()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                let status = self.child.wait().ok();
                Err(match status {
                    Some(s) => format!("the ASR exited ({s})"),
                    None => "the ASR closed its output".into(),
                })
            }
        }
    }
}

impl Recogniser for AsrRecogniser {
    fn recognise(&mut self, samples: &[f32], sample_rate: u32, text: &str) -> Result<String, String> {
        if let Some(f) = &self.fault {
            return Err(f.clone());
        }
        self.n += 1;
        let path = self.dir.join(format!("draw-{}.flac", self.n));
        let path = ggtts_core::flac::write_flac(&path, samples, sample_rate).map_err(|e| e.to_string())?;
        let request = serde_json::json!({"audio": path.to_string_lossy(), "text": text});
        let sent = writeln!(self.stdin, "{request}").and_then(|()| self.stdin.flush());
        let answer = sent
            .map_err(|e| format!("writing to the ASR: {e}"))
            .and_then(|()| self.read_line(self.timeout));
        let _ = std::fs::remove_file(&path);
        // Anything but an answer is a worker that cannot read another draw.
        let answer = answer.inspect_err(|e| self.fault = Some(e.clone()))?;
        match (answer.get("text").and_then(Value::as_str), answer.get("error")) {
            (Some(t), _) => Ok(t.to_string()),
            (None, Some(e)) => Err(e.as_str().map_or_else(|| e.to_string(), str::to_string)),
            (None, None) => Err(format!("the ASR answered {answer}")),
        }
    }

    fn fault(&self) -> Option<String> {
        self.fault.clone()
    }
}

impl Drop for AsrRecogniser {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn interpreter(name: &str) -> String {
    if name == "python" {
        crate::runner::default_python()
    } else {
        name.to_string()
    }
}

/// A number no other judge in this process has used, for its temporary directory.
fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn a_spec_reads_or_is_refused_naming_the_problem() {
        let s = parse("asr-qwen,model=/m,device=cuda,min_run=6").unwrap();
        assert_eq!(
            (s.model.as_path(), s.device.as_str(), s.min_run),
            (Path::new("/m"), "cuda", 6)
        );
        let s = parse("asr-qwen,model=/m,timeout=5,start_timeout=9").unwrap();
        assert_eq!((s.timeout_s, s.start_timeout_s), (5.0, 9.0));
        let s = parse("asr-qwen,model=/m").unwrap();
        assert_eq!(
            (s.timeout_s, s.start_timeout_s),
            (DEFAULT_TIMEOUT_S, DEFAULT_START_TIMEOUT_S)
        );
        assert_eq!((s.device.as_str(), s.min_run), ("cpu", DEFAULT_MIN_RUN));
        for (spec, says) in [
            ("whisper,model=/m", "is not a judge"),
            ("asr-qwen", "needs model=DIR"),
            ("asr-qwen,model=/m,device=tpu", "device must be cpu or cuda"),
            ("asr-qwen,model=/m,min_run=0", "min_run must be"),
            ("asr-qwen,model=/m,colour=red", "unknown or empty key"),
            ("asr-qwen,model", "is not KEY=VALUE"),
            ("asr-qwen,model=/m,timeout=0", "timeout must be a number of seconds"),
            (
                "asr-qwen,model=/m,start_timeout=x",
                "start_timeout must be a number of seconds",
            ),
        ] {
            let e = parse(spec).unwrap_err();
            assert!(
                e.starts_with("--judge (render.judge): ") && e.contains(says),
                "{spec}: {e}"
            );
        }
    }

    #[test]
    fn unset_is_the_duration_judge() {
        let mut j = judge(&Options::default()).unwrap();
        assert!(j.take_reading().is_none());
    }

    #[test]
    fn an_asr_that_does_not_start_is_refused_by_name() {
        let mut o = Options::default();
        o.render.judge = Some("asr-qwen,model=/m,command=/no/such/asr".into());
        o.render.redraw = Some(1);
        let e = judge(&o).err().unwrap();
        assert!(e.starts_with("--judge (render.judge): asr-qwen did not start: "), "{e}");
    }
}
