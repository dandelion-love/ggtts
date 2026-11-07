//! `ggtts-measure` runs the instrument runner on audio that already exists: a required instrument decides
//! completeness as in a render's run, the given file is never written, no engine is loaded, a bad command runs no
//! instrument, and the same instrument on the same FLAC reports the same from `ggtts-render` and `ggtts-measure`.
//!
//! The instruments are small Python programs written here on the ggtts SDK (`instruments/python`): `bytes` passes
//! and reports the audio's size, `broken` raises, and `tamper` appends to the audio it is given. They need Python
//! 3.8 or later as `python3` or `python`; without one each test fails, unless `GGTTS_SKIP_PYTHON=1` says to skip
//! on purpose, as in `ggtts-core`'s Python harness.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const RENDER: &str = env!("CARGO_BIN_EXE_ggtts-render");
const MEASURE: &str = env!("CARGO_BIN_EXE_ggtts-measure");

#[test]
fn report_input_alias_never_modifies_the_source() {
    let Some(py) = python() else { return };
    let d = scratch("input-alias");
    let ins = Instruments::new(&d, py);
    let source = flac(&d);
    let report = d.join("report");
    std::fs::create_dir_all(report.join("input")).unwrap();
    let input = report.join("input").join("kept.flac");
    std::fs::copy(&source, &input).unwrap();
    let before = std::fs::read(&input).unwrap();
    let output = measure(
        &["--audio", input.to_str().unwrap(), "--report", report.to_str().unwrap()],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert!(std::fs::read(&input).unwrap() == before, "input must not be modified");
    assert_eq!(output.status.code(), Some(2), "existing report must be refused");
}

#[test]
fn existing_report_with_hardlink_is_refused_without_modifying_input() {
    let Some(py) = python() else { return };
    let d = scratch("hardlink-alias");
    let ins = Instruments::new(&d, py);
    let input = flac(&d);
    let before = std::fs::read(&input).unwrap();
    let report = d.join("report");
    std::fs::create_dir_all(report.join("input")).unwrap();
    std::fs::hard_link(&input, report.join("input").join("kept.flac")).unwrap();
    let output = measure(
        &["--audio", input.to_str().unwrap(), "--report", report.to_str().unwrap()],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(std::fs::read(&input).unwrap() == before);
}

#[cfg(unix)]
#[test]
fn existing_report_with_symlink_is_refused_without_modifying_input() {
    let Some(py) = python() else { return };
    let d = scratch("symlink-alias");
    let ins = Instruments::new(&d, py);
    let input = flac(&d);
    let before = std::fs::read(&input).unwrap();
    let report = d.join("report");
    std::fs::create_dir_all(report.join("input")).unwrap();
    std::os::unix::fs::symlink(&input, report.join("input").join("kept.flac")).unwrap();
    let output = measure(
        &["--audio", input.to_str().unwrap(), "--report", report.to_str().unwrap()],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(std::fs::read(&input).unwrap() == before);
}

#[test]
fn audio_and_events_with_same_filename_get_distinct_copies() {
    let Some(py) = python() else { return };
    let d = scratch("same-name");
    let ins = Instruments::new(&d, py);
    let audio = flac(&d);
    let events_dir = d.join("events");
    std::fs::create_dir_all(&events_dir).unwrap();
    let events = events_dir.join("kept.flac");
    std::fs::write(&events, b"test events").unwrap();
    let report = d.join("report");
    let output = measure(
        &[
            "--audio",
            audio.to_str().unwrap(),
            "--events",
            events.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(std::fs::read(report.join("input/audio/kept.flac")).unwrap() == std::fs::read(&audio).unwrap());
    assert!(std::fs::read(report.join("input/events/kept.flac")).unwrap() == b"test events");
}

fn python() -> Option<String> {
    for exe in ["python3", "python"] {
        let ok = Command::new(exe)
            .args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"])
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return Some(exe.into());
        }
    }
    if std::env::var("GGTTS_SKIP_PYTHON").as_deref() == Ok("1") {
        eprintln!("GGTTS_SKIP_PYTHON=1: the ggtts-measure tests were not run");
        return None;
    }
    panic!("no Python 3.8+ found as python3 or python; install one, or set GGTTS_SKIP_PYTHON=1 to skip on purpose");
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-measure-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const PROGRAM: &str = r#"import os, sys
sys.path.insert(0, sys.argv.pop(1))
from ggtts_instrument import run

MODE = os.path.splitext(os.path.basename(__file__))[0]

def measure(ctx, env):
    env.input(ctx.audio)
    if MODE == "broken":
        raise RuntimeError("planted failure")
    if MODE == "tamper":
        with open(ctx.audio, "ab") as f:
            f.write(b"planted")
    env.metric("bytes", os.path.getsize(ctx.audio), "1", "n/a", render=ctx.render)

sys.exit(run(measure, MODE, "0.1.0", hooks=["artifact.audio"], audio_state="trimmed"))
"#;

/// The three instruments, each a `--instrument` value: its program, the interpreter, and the SDK's path as its
/// first argument (the program takes it off before the runner's flags are read).
struct Instruments {
    py: String,
    dir: PathBuf,
}

impl Instruments {
    fn new(dir: &Path, py: String) -> Self {
        for mode in ["bytes", "broken", "tamper"] {
            std::fs::write(dir.join(format!("{mode}.py")), PROGRAM).unwrap();
        }
        Instruments {
            py,
            dir: dir.to_path_buf(),
        }
    }

    fn flag(&self, mode: &str) -> String {
        let sdk = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments/python");
        // The interpreter runs `program sdk-path --runner-flags...`: the SDK path rides as the program's first
        // argument, so nothing is installed and the environment is untouched.
        let wrapper = self.dir.join(format!("{mode}-run.py"));
        std::fs::write(
            &wrapper,
            format!(
                "import runpy, sys\nsys.argv = [{prog:?}, {sdk:?}] + sys.argv[1:]\nrunpy.run_path({prog:?}, run_name='__main__')\n",
                prog = self.dir.join(format!("{mode}.py")).display().to_string(),
                sdk = sdk.display().to_string()
            ),
        )
        .unwrap();
        format!(
            "{mode},command={},interpreter={},version=0.1.0",
            wrapper.display(),
            self.py
        )
    }
}

/// A fake render's FLAC, made without instruments.
fn flac(dir: &Path) -> PathBuf {
    let out = dir.join("kept.flac");
    let o = Command::new(RENDER)
        .args(["--engine", "fake", "--text", "今日天氣好好，我哋去海邊行下。", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    out
}

fn measure(args: &[&str], extra: &[String]) -> Output {
    Command::new(MEASURE).args(args).args(extra).output().unwrap()
}

fn run_log(report: &Path) -> String {
    std::fs::read_to_string(report.join("run.log")).unwrap_or_default()
}

fn envelope(report: &Path, name: &str) -> Value {
    let p = report.join("instruments").join(name).join("envelope.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn measurement_line(report: &Path) -> String {
    run_log(report)
        .lines()
        .find(|l| l.starts_with("measurement:"))
        .unwrap_or("")
        .to_string()
}

#[test]
fn a_required_instrument_decides_completeness_as_in_a_render() {
    let Some(py) = python() else { return };
    let d = scratch("ac1");
    let ins = Instruments::new(&d, py);
    let audio = flac(&d);
    let a = audio.to_str().unwrap();

    let ok = d.join("ok");
    let o = measure(
        &["--audio", a, "--report", ok.to_str().unwrap(), "--require", "bytes"],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(measurement_line(&ok), "measurement: complete");

    let bad = d.join("bad");
    let o = measure(
        &["--audio", a, "--report", bad.to_str().unwrap(), "--require", "broken"],
        &["--instrument".into(), ins.flag("broken")],
    );
    assert_eq!(o.status.code(), Some(3), "{}", String::from_utf8_lossy(&o.stderr));

    // The same failure in a render's run writes the same line.
    let rendered = d.join("rendered");
    let o = Command::new(RENDER)
        .args(["--engine", "fake", "--text", "今日天氣好好，我哋去海邊行下。", "--out"])
        .arg(d.join("r.flac"))
        .args([
            "--report",
            rendered.to_str().unwrap(),
            "--require",
            "broken",
            "--instrument",
            &ins.flag("broken"),
        ])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(3), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(measurement_line(&bad).starts_with("measurement: INCOMPLETE: required instrument broken is failed"));
    assert_eq!(measurement_line(&bad), measurement_line(&rendered));
}

#[test]
fn the_given_file_is_never_written() {
    let Some(py) = python() else { return };
    let d = scratch("ac2");
    let ins = Instruments::new(&d, py);
    let audio = flac(&d);
    let before = std::fs::read(&audio).unwrap();
    let rep = d.join("rep");
    let o = measure(
        &["--audio", audio.to_str().unwrap(), "--report", rep.to_str().unwrap()],
        &["--instrument".into(), ins.flag("tamper")],
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "not required, so the run completes: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(
        envelope(&rep, "tamper")["status"],
        "failed",
        "the runner must catch the write to its copy"
    );
    assert_eq!(
        std::fs::read(&audio).unwrap(),
        before,
        "the original must be byte-identical"
    );
    assert_eq!(
        std::fs::read(rep.join("input").join("audio").join("kept.flac")).unwrap(),
        before,
        "the copy is restored too"
    );
}

#[test]
fn no_engine_is_loaded() {
    let Some(py) = python() else { return };
    let d = scratch("ac3");
    let ins = Instruments::new(&d, py);
    let audio = flac(&d);
    // No --model, no --codec, no model file anywhere: the run still completes, and nothing names an engine.
    let rep = d.join("rep");
    let o = Command::new(MEASURE)
        .current_dir(&d)
        .args([
            "--audio",
            audio.to_str().unwrap(),
            "--report",
            rep.to_str().unwrap(),
            "--instrument",
            &ins.flag("bytes"),
        ])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.contains("load") && !err.contains("engine"), "{err}");
    assert!(std::fs::read_dir(&d)
        .unwrap()
        .all(|e| e.unwrap().path().extension() != Some(std::ffi::OsStr::new("gguf"))));
}

#[test]
fn the_same_instrument_reports_the_same_from_both_paths() {
    let Some(py) = python() else { return };
    let d = scratch("ac4");
    let ins = Instruments::new(&d, py);
    let out = d.join("seam.flac");
    let rendered = d.join("rendered");
    let o = Command::new(RENDER)
        .args(["--engine", "fake", "--text", "今日天氣好好，我哋去海邊行下。", "--out"])
        .arg(&out)
        .args([
            "--run",
            "seam",
            "--report",
            rendered.to_str().unwrap(),
            "--instrument",
            &ins.flag("bytes"),
        ])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let measured = d.join("measured");
    let o = measure(
        &[
            "--audio",
            out.to_str().unwrap(),
            "--run",
            "seam",
            "--report",
            measured.to_str().unwrap(),
        ],
        &["--instrument".into(), ins.flag("bytes")],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let (a, b) = (envelope(&rendered, "bytes"), envelope(&measured, "bytes"));
    // Equal field by field, apart from where the audio was read: ggtts-render hands over its own --out, and
    // ggtts-measure a copy under the report. The copy's hash must be the original's.
    for k in [
        "format",
        "name",
        "version",
        "identity",
        "hooks",
        "status",
        "metrics",
        "artifacts",
        "error",
    ] {
        assert_eq!(a[k], b[k], "{k}");
    }
    for k in ["config", "ggtts", "engine", "backend", "model_sha256", "seeds", "audio"] {
        assert_eq!(a["provenance"][k], b["provenance"][k], "provenance.{k}");
    }
    let sha = |e: &Value| {
        e["provenance"]["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["sha256"].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(sha(&a), sha(&b), "the inputs' hashes");
    assert!(!sha(&a).is_empty());
}

#[test]
fn a_bad_command_runs_no_instrument() {
    let Some(py) = python() else { return };
    let d = scratch("ac5");
    let ins = Instruments::new(&d, py);
    let audio = flac(&d);
    let junk = d.join("junk.flac");
    std::fs::write(&junk, b"not audio at all").unwrap();
    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "a missing --audio",
            vec!["--audio".into(), d.join("absent.flac").display().to_string()],
        ),
        (
            "an unreadable --audio",
            vec!["--audio".into(), junk.display().to_string()],
        ),
        (
            "a missing --events",
            vec![
                "--audio".into(),
                audio.display().to_string(),
                "--events".into(),
                d.join("absent.jsonl").display().to_string(),
            ],
        ),
        ("no instrument", vec!["--audio".into(), audio.display().to_string()]),
    ];
    for (i, (what, mut args)) in cases.into_iter().enumerate() {
        let rep = d.join(format!("rep{i}"));
        args.extend(["--report".into(), rep.display().to_string()]);
        if what != "no instrument" {
            args.extend(["--instrument".into(), ins.flag("bytes")]);
        }
        let o = Command::new(MEASURE).args(&args).output().unwrap();
        assert_eq!(
            o.status.code(),
            Some(2),
            "{what}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(String::from_utf8_lossy(&o.stderr).starts_with("error: "), "{what}");
        assert!(!rep.join("instruments").exists(), "{what}: an instrument ran");
    }
}
