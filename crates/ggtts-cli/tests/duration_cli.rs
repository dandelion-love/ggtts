use std::process::Command;

#[test]
fn duration_report_is_trace_free_and_preserves_audio() {
    let dir = std::env::temp_dir().join(format!("ggtts-duration-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let story = dir.join("story.txt");
    std::fs::write(&story, "Title\nFirst sentence.\nSecond sentence.").unwrap();
    let render = |audio: &str, report: Option<&str>| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ggtts-render"));
        cmd.args(["--engine", "fake", "--story"]).arg(&story).arg("--out").arg(dir.join(audio));
        if let Some(report) = report { cmd.arg("--duration-report").arg(dir.join(report)); }
        let result = cmd.output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    };
    render("plain.flac", None);
    render("measured.flac", Some("duration.json"));
    assert_eq!(std::fs::read(dir.join("plain.flac")).unwrap(), std::fs::read(dir.join("measured.flac")).unwrap());
    let d: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("duration.json")).unwrap()).unwrap();
    let chunks = d["chunks"].as_array().unwrap();
    assert!(chunks.len() >= 2);
    let kept: u64 = chunks.iter().map(|c| c["trimmed_samples"].as_u64().unwrap()).sum();
    assert_eq!(kept + d["join_samples"].as_u64().unwrap(), d["output_samples"].as_u64().unwrap());
    assert!(d["join_samples"].as_u64().unwrap() > 0);
    for c in chunks {
        assert!(c["decoded_samples"].as_u64().unwrap() >= c["trimmed_samples"].as_u64().unwrap());
        assert!(c["sample_rate"].as_u64().unwrap() > 0);
    }
}

#[test]
fn text_reports_decode_without_story_trimming() {
    let dir = std::env::temp_dir().join(format!("ggtts-duration-text-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--text", "Hello.", "--out"]).arg(dir.join("out.flac"))
        .arg("--duration-report").arg(dir.join("duration.json")).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let d: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("duration.json")).unwrap()).unwrap();
    assert_eq!(d["path"], "text");
    assert_eq!(d["text_decoded_samples"], d["output_samples"]);
    assert!(d["chunks"].is_null() && d["join_samples"].is_null());
}

#[test]
fn instrumented_without_trace_writes_durations_and_refuses_existing_sidecar() {
    let dir = std::env::temp_dir().join(format!("ggtts-duration-auto-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let instruments = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    let run = |report: &str, out: &str| Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .env("GGTTS_INSTRUMENTS_DIR", &instruments)
        .args(["--engine", "fake", "--text", "Hello.", "--out"]).arg(dir.join(out))
        .args(["--instrument", "audio-levels", "--require", "audio-levels", "--report"])
        .arg(dir.join(report)).output().unwrap();
    let first = run("ok", "ok.flac");
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    assert!(dir.join("ok/duration.json").exists() && dir.join("ok/duration.md").exists());
    std::fs::create_dir(dir.join("collision")).unwrap();
    std::fs::write(dir.join("collision/duration.json"), b"keep").unwrap();
    assert!(!run("collision", "never.flac").status.success());
    assert_eq!(std::fs::read(dir.join("collision/duration.json")).unwrap(), b"keep");
    assert!(!dir.join("never.flac").exists());
}
