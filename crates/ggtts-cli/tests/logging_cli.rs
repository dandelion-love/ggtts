use std::path::Path;
use std::process::Command;

#[test]
fn invalid_logging_level_reports_json_for_both_commands() {
    for bin in [env!("CARGO_BIN_EXE_ggtts-render"), env!("CARGO_BIN_EXE_ggtts-measure")] {
        for args in [
            ["--log-format", "json", "--log-level", "invalid"],
            ["--log-level", "invalid", "--log-format", "json"],
        ] {
            let output = Command::new(bin).args(args).output().unwrap();
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
            assert_eq!(value["format"], "ggtts-log/1");
            assert_eq!(value["level"], "error");
            assert_eq!(value["event"], "logging_config");
        }
    }
}

#[test]
fn json_logs_are_consistent_and_do_not_contaminate_audio_stdout() {
    let root = std::env::temp_dir().join(format!("ggtts-json-logs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let story = root.join("story.txt");
    std::fs::write(&story, "Title\nA private test sentence.").unwrap();
    let audio = root.join("audio.flac");
    let report = root.join("report");
    let trace = root.join("trace.jsonl");
    let timing = root.join("timing.json");
    let instruments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    let result = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .env("GGTTS_INSTRUMENTS_DIR", &instruments)
        .args(["--engine", "fake", "--story"])
        .arg(&story)
        .arg("--out")
        .arg(&audio)
        .args(["--stream", "-"])
        .arg("--trace")
        .arg(&trace)
        .arg("--timing-report")
        .arg(&timing)
        .args(["--trace-level", "2"])
        .arg("--report")
        .arg(&report)
        .args(["--instrument", "audio-levels", "--require", "audio-levels"])
        .args(["--log-level", "info", "--log-format", "json"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert!(
        result.stdout.starts_with(b"RIFF"),
        "audio stdout must remain a WAV stream"
    );
    let stderr = std::str::from_utf8(&result.stderr).unwrap();
    assert!(!stderr.contains("private test sentence"));
    let lines: Vec<serde_json::Value> = stderr.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert!(lines.iter().all(|line| line["format"] == "ggtts-log/1"));
    assert!(lines
        .iter()
        .any(|line| line["event"] == "trace_sink" && line["fields"]["events"].is_number()));
    assert!(lines
        .iter()
        .any(|line| line["event"] == "stream_handoff" && line["fields"]["write_ns"].is_number()));
    let timings: serde_json::Value = serde_json::from_slice(&std::fs::read(&timing).unwrap()).unwrap();
    assert_eq!(timings["format"], "ggtts-timing/1");
    assert_eq!(timings["trace_sink"]["level"], 2);
    assert!(timings["stages"]["generated_frames_per_second"].as_f64().unwrap() > 0.0);
    assert!(timings["instrument_phase_ms"].is_number());
    let duration: serde_json::Value = serde_json::from_slice(&std::fs::read(report.join("duration.json")).unwrap()).unwrap();
    assert_eq!(duration["format"], "ggtts-duration/1");
    assert_eq!(duration["discarded_draws_included"], false);
    let chunk = &duration["chunks"][0];
    assert!(chunk["decoded_samples"].as_u64().unwrap() >= chunk["trimmed_samples"].as_u64().unwrap());
    assert!(std::fs::read_to_string(report.join("duration.md")).unwrap().contains("Audio durations"));
    assert!(timings["stream"]["write_ns"].is_number());
    let run_log = std::fs::read_to_string(report.join("run.log")).unwrap();
    assert!(run_log
        .lines()
        .all(|line| serde_json::from_str::<serde_json::Value>(line)
            .is_ok_and(|value| value["format"] == "ggtts-log/1" && value["component"] == "instrument-runner")));

    let missing = Command::new(env!("CARGO_BIN_EXE_ggtts-measure"))
        .env("GGTTS_INSTRUMENTS_DIR", &instruments)
        .arg("--audio")
        .arg(&audio)
        .arg("--report")
        .arg(root.join("missing"))
        .args([
            "--instrument",
            "p-stop-early",
            "--require",
            "p-stop-early",
            "--log-level",
            "error",
            "--log-format",
            "json",
        ])
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(3));
    let errors: Vec<serde_json::Value> = std::str::from_utf8(&missing.stderr)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!errors.is_empty() && errors.iter().all(|v| v["level"] == "error"));
    assert!(errors.iter().any(|v| v["event"] == "measurement_incomplete"));
    let run_log = std::fs::read_to_string(root.join("missing/run.log")).unwrap();
    assert!(run_log
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .any(|v| v["event"] == "measurement_incomplete" && v["level"] == "error"));
    let quiet_audio = root.join("quiet.flac");
    let quiet_timing = root.join("quiet.json");
    let quiet = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--story"])
        .arg(&story)
        .arg("--out")
        .arg(&quiet_audio)
        .arg("--timing-report")
        .arg(&quiet_timing)
        .args(["--log-level", "error", "--log-format", "json"])
        .output()
        .unwrap();
    assert!(quiet.status.success(), "{}", String::from_utf8_lossy(&quiet.stderr));
    assert!(quiet.stderr.is_empty());
    assert_eq!(std::fs::read(&quiet_audio).unwrap(), std::fs::read(&audio).unwrap());
    let timings: serde_json::Value = serde_json::from_slice(&std::fs::read(&quiet_timing).unwrap()).unwrap();
    assert!(timings["trace_sink"].is_null() && timings["stream"].is_null());
    assert!(timings["stages"]["generate_ns"].is_number());
    assert_eq!(timings["logging"]["records"], 0);
    let text_timing = root.join("text-timing.json");
    let text = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--text", "A short line."])
        .arg("--out")
        .arg(root.join("text.flac"))
        .arg("--stream")
        .arg(root.join("text.wav"))
        .arg("--timing-report")
        .arg(&text_timing)
        .args(["--log-level", "error"])
        .output()
        .unwrap();
    assert!(text.status.success(), "{}", String::from_utf8_lossy(&text.stderr));
    let text_report: serde_json::Value = serde_json::from_slice(&std::fs::read(text_timing).unwrap()).unwrap();
    assert!(text_report["stream"]["first_audio_ms"].is_number());
    assert!(text_report["stream"]["preparation_ns"].is_null());
    assert!(text_report["stages"]["judge_ns"].is_null());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn logging_probe_compares_modes_without_changing_fake_audio() {
    let root = std::env::temp_dir().join(format!("ggtts-log-probe-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let story = root.join("story.txt");
    std::fs::write(&story, "Fixture\nA short sentence.").unwrap();
    let work = root.join("work");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/logging_overhead_probe.py");
    let result = Command::new(ggtts_instruments::runner::default_python())
        .arg(script)
        .arg("--render")
        .arg(env!("CARGO_BIN_EXE_ggtts-render"))
        .arg("--story")
        .arg(&story)
        .arg("--work")
        .arg(&work)
        .args(["--engine", "fake", "--device", "cpu", "--repeats", "3"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(work.join("summary.json")).unwrap()).unwrap();
    assert_eq!(report["format"], "ggtts-logging-probe/1");
    assert_eq!(report["runs"].as_array().unwrap().len(), 15);
    assert!(report["summary"]["info_trace_2"]["frames_per_second"]["median"].is_number());
    let _ = std::fs::remove_dir_all(root);
}
