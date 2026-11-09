use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use ggtts_core::report::{merge, Input, FORMAT};
use serde_json::{json, Value};

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

#[test]
fn exports_the_timing_instruments_real_synthetic_envelope() {
    if std::env::var_os("GGTTS_SKIP_PYTHON").is_some() {
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "ggtts-export-timing-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let timing = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../instruments/templates/python/timing/test_instrument.py");
    // Run the actual timing producer with its own synthetic trace fixture.
    let script = "import importlib.util,json,sys\n\
        s=importlib.util.spec_from_file_location('timing_test',sys.argv[1])\n\
        m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n\
        trace,_=m.TimingTest().trace(sys.argv[2])\n\
        print(json.dumps(m.run_instrument(m.SCRIPT,sys.argv[2],events=trace)))\n";
    let produced = Command::new(ggtts_instruments::runner::default_python())
        .args(["-c", script])
        .arg(timing)
        .arg(&dir)
        .output()
        .unwrap();
    assert!(produced.status.success(), "synthetic timing producer failed");
    let envelope: Value = serde_json::from_slice(&produced.stdout).unwrap();
    assert_eq!(envelope["status"], "ok");
    let metrics = envelope["metrics"].as_array().unwrap();
    assert_eq!(metrics.len(), 6);
    let report = merge(
        "selftest",
        &[Input {
            source: "private-trace-path".into(),
            text: envelope.to_string(),
        }],
        &["render-timing"],
    );
    let source = dir.join("private-report.json");
    let destination = dir.join("share.json");
    std::fs::write(&source, report.to_json().to_string()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ggtts-report-export"))
        .arg(&source)
        .arg(&destination)
        .output()
        .unwrap();
    assert!(result.status.success(), "timing export failed");
    let shared: Value = serde_json::from_slice(&std::fs::read(destination).unwrap()).unwrap();
    let exported = shared["instruments"][0]["metrics"].as_array().unwrap();
    assert_eq!(exported.len(), metrics.len());
    for (before, after) in metrics.iter().zip(exported) {
        for key in ["name", "unit", "value", "direction"] {
            assert_eq!(before[key], after[key]);
        }
    }
    let schema_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/ggtts-shareable-report-1.schema.json");
    let schema: Value = serde_json::from_slice(&std::fs::read(schema_path).unwrap()).unwrap();
    assert!(jsonschema::validator_for(&schema).unwrap().is_valid(&shared));
    assert!(!shared.to_string().contains("private-trace-path"));
    std::fs::remove_dir_all(dir).unwrap();
}
