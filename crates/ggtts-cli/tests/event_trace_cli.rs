use std::path::Path;
use std::process::Command;

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
