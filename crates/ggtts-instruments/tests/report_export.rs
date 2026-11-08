use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use ggtts_core::report::{merge, Input, FORMAT};
use serde_json::json;

#[test]
fn separate_export_preserves_original_and_refuses_bad_inputs() {
    let dir = std::env::temp_dir().join(format!(
        "ggtts-export-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let source = dir.join("private-path.json");
    let output = dir.join("share.json");
    let env = json!({"format":FORMAT,"name":"asr-private","version":"secret-version",
        "identity":{"run":"private-run","instrument":"private-id"},
        "provenance":{"config":{"text":"private-transcript"},"ggtts":null,"engine":null,"backend":null,
            "model_sha256":null,"seeds":[],"inputs":[],"audio":"trimmed"},
        "hooks":[],"status":"timeout","metrics":[],"artifacts":[],"log":"private-log","error":"private-error"});
    let report = merge(
        "private-run",
        &[Input {
            source: "private-source".into(),
            text: env.to_string(),
        }],
        &["asr-private"],
    );
    let original = report.to_json().to_string();
    std::fs::write(&source, &original).unwrap();
    let exe = env!("CARGO_BIN_EXE_ggtts-report-export");
    let result = Command::new(exe).arg(&source).arg(&output).output().unwrap();
    assert!(result.status.success());
    let shared = std::fs::read_to_string(&output).unwrap();
    assert!(!shared.contains("private"));
    assert!(!shared.contains("secret"));
    assert_eq!(std::fs::read_to_string(&source).unwrap(), original);
    let again = Command::new(exe).arg(&source).arg(&output).output().unwrap();
    assert!(!again.status.success());
    assert_eq!(std::fs::read_to_string(&output).unwrap(), shared);
    let bad = dir.join("bad.json");
    let missing = dir.join("must-not-exist.json");
    std::fs::write(&bad, "private-transcript malformed").unwrap();
    let failure = Command::new(exe).arg(&bad).arg(&missing).output().unwrap();
    assert!(!failure.status.success());
    assert!(!missing.exists());
    assert!(!String::from_utf8_lossy(&failure.stderr).contains("private"));
    std::fs::remove_dir_all(dir).unwrap();
}
