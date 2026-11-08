use std::path::Path;
use std::process::Command;

#[test]
fn required_step_measurement_reports_missing_data_without_changing_audio() {
    let root = std::env::temp_dir().join(format!("ggtts-step-input-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let story = root.join("story.txt");
    std::fs::write(&story, "A short story\nThe quick fox crosses the field.").unwrap();
    let audio = root.join("audio.flac");
    let trace = root.join("events.jsonl");
    let instruments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    let render = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--story"])
        .arg(&story)
        .arg("--out")
        .arg(&audio)
        .arg("--trace")
        .arg(&trace)
        .output()
        .unwrap();
    assert!(render.status.success(), "{}", String::from_utf8_lossy(&render.stderr));
    let original = std::fs::read(&audio).unwrap();
    let instrumented_audio = root.join("instrumented.flac");
    let instrumented_trace = root.join("instrumented.jsonl");
    let instrumented_report = root.join("instrumented-report");
    let instrumented = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .env("GGTTS_INSTRUMENTS_DIR", &instruments)
        .args(["--engine", "fake", "--story"])
        .arg(&story)
        .arg("--out")
        .arg(&instrumented_audio)
        .arg("--trace")
        .arg(&instrumented_trace)
        .arg("--report")
        .arg(&instrumented_report)
        .args(["--instrument", "p-stop-early", "--require", "p-stop-early"])
        .output()
        .unwrap();
    assert_eq!(
        instrumented.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&instrumented.stderr)
    );
    assert_eq!(std::fs::read(&instrumented_audio).unwrap(), original);
    let run = |events: &Path, report: &Path| {
        Command::new(env!("CARGO_BIN_EXE_ggtts-measure"))
            .env("GGTTS_INSTRUMENTS_DIR", &instruments)
            .arg("--audio")
            .arg(&audio)
            .arg("--events")
            .arg(events)
            .arg("--report")
            .arg(report)
            .args(["--instrument", "p-stop-early", "--require", "p-stop-early"])
            .output()
            .unwrap()
    };
    let valid = root.join("valid-report");
    let ok = run(&trace, &valid);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let missing = root.join("missing.jsonl");
    let lines = std::fs::read_to_string(&trace).unwrap();
    let without_steps: String = lines
        .lines()
        .filter(|line| !line.contains("\"ev\":\"step\""))
        .map(|line| format!("{line}\n"))
        .collect();
    std::fs::write(&missing, without_steps).unwrap();
    let report = root.join("missing-report");
    let result = run(&missing, &report);
    assert_eq!(
        result.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let log = std::fs::read_to_string(report.join("run.log")).unwrap();
    assert!(log.contains("p_stop data is unavailable"), "{log}");
    assert_eq!(std::fs::read(&audio).unwrap(), original);
}

#[test]
fn story_trace_reaches_required_event_instrument_without_changing_audio() {
    let root = std::env::temp_dir().join(format!("ggtts-event-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let story = root.join("story.txt");
    std::fs::write(&story, "A short story\nThe quick fox crosses the field.").unwrap();
    let plain = root.join("plain.flac");
    let traced = root.join("traced.flac");
    let trace = root.join("events.jsonl");
    let report = root.join("report");
    let binary = env!("CARGO_BIN_EXE_ggtts-render");
    let instruments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    let run = |out: &Path, extra: &[&str]| {
        Command::new(binary)
            .env("GGTTS_INSTRUMENTS_DIR", &instruments)
            .args(["--engine", "fake", "--story"])
            .arg(&story)
            .arg("--out")
            .arg(out)
            .args(extra)
            .output()
            .unwrap()
    };
    let base = run(&plain, &[]);
    assert!(base.status.success(), "{}", String::from_utf8_lossy(&base.stderr));
    let traced_run = run(
        &traced,
        &[
            "--trace",
            trace.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
            "--instrument",
            "draws",
            "--require",
            "draws",
        ],
    );
    assert!(
        traced_run.status.success(),
        "{}",
        String::from_utf8_lossy(&traced_run.stderr)
    );
    assert!(std::fs::read(&plain).unwrap() == std::fs::read(&traced).unwrap());
    let events = std::fs::read_to_string(trace).unwrap();
    assert!(events.lines().any(|line| line.contains("\"ev\":\"trace\"")));
    assert!(events.lines().any(|line| line.contains("\"ev\":\"end\"")));
    let stream = root.join("stream.wav");
    let stream_out = root.join("stream.flac");
    let stream_trace = root.join("stream.jsonl");
    let stream_report = root.join("stream-report");
    let result = run(
        &stream_out,
        &[
            "--stream",
            stream.to_str().unwrap(),
            "--trace",
            stream_trace.to_str().unwrap(),
            "--report",
            stream_report.to_str().unwrap(),
            "--instrument",
            "draws",
            "--require",
            "draws",
        ],
    );
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(std::fs::read(&plain).unwrap() == std::fs::read(&stream_out).unwrap());
    let events = std::fs::read_to_string(stream_trace).unwrap();
    assert!(events.lines().any(|line| line.contains("\"ev\":\"emit\"")));
    assert!(events.lines().any(|line| line.contains("\"ev\":\"end\"")));
}
