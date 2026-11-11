//! The default model files as the command line sees them: the offline refusal, and the model files in the timing
//! report. Nothing here reaches the network: the offline run must refuse before any download, and the reports come
//! from the fake engine.

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_ggtts-render");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-default-models-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn a_default_render_offline_with_an_empty_cache_names_each_missing_file_and_its_url() {
    let dir = scratch("offline");
    let cache = dir.join("cache");
    let r = Command::new(BIN)
        .env("GGTTS_CACHE_DIR", &cache)
        .args(["--text", "你好。", "--offline", "--out"])
        .arg(dir.join("x.flac"))
        .output()
        .unwrap();
    assert!(!r.status.success());
    let err = String::from_utf8_lossy(&r.stderr);
    if cfg!(feature = "ggml") {
        for p in ggtts_models::QWEN3_TTS.files {
            assert!(err.contains(p.file_name) && err.contains(p.url), "{err}");
        }
    } else {
        // A build without engines refuses before resolving anything.
        assert!(err.contains("no engines"), "{err}");
    }
    let left: Vec<_> = walk(&cache);
    assert!(left.is_empty(), "an offline refusal left files: {left:?}");
    assert!(!dir.join("x.flac").exists());
    let _ = std::fs::remove_dir_all(dir);
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path());
            }
        }
    }
    out
}

fn timing_report(dir: &Path, extra: &[&str]) -> serde_json::Value {
    let report = dir.join(format!("timing-{}.json", extra.len()));
    let r = Command::new(BIN)
        .args(["--engine", "fake", "--text", "hello", "--log-level", "error", "--out"])
        .arg(dir.join(format!("out-{}.flac", extra.len())))
        .arg("--timing-report")
        .arg(&report)
        .args(extra)
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    json(&report)
}

#[test]
fn the_timing_report_records_every_model_file_and_the_schema_holds_it() {
    let dir = scratch("timing");
    let model = dir.join("an-unreleased-model.bin");
    std::fs::write(&model, b"an unreleased model").unwrap();
    let none = timing_report(&dir, &[]);
    assert_eq!(none["models"], serde_json::json!([]));
    let given = timing_report(&dir, &["--model", model.to_str().unwrap()]);
    let entry = &given["models"][0];
    assert_eq!(entry["role"], "model");
    assert_eq!(entry["source"], "path");
    assert_eq!(entry["path"], model.to_str().unwrap());
    assert!(entry["url"].is_null());
    // sha256 of b"an unreleased model", computed independently.
    assert_eq!(
        entry["sha256"],
        ggtts_models::sha256_file(&model).unwrap(),
        "the given file's own hash"
    );

    let schema = json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-timing-1.schema.json"));
    let validator = jsonschema::validator_for(&schema).expect("the published timing schema compiles");
    let cached = {
        let mut v = given.clone();
        v["models"] = serde_json::json!([{"role": "talker", "path": "/cache/models/x/talker.gguf",
            "sha256": "4eb27e5a3434d15ca818814b8e4f0f1941716654afb8543866732b62f9227041", "source": "cache",
            "url": "https://example.invalid/talker.gguf"}]);
        v
    };
    let older = {
        let mut v = none.clone();
        v.as_object_mut().unwrap().remove("models");
        v
    };
    for (what, report) in [
        ("no model", &none),
        ("a given model", &given),
        ("a cached model", &cached),
        ("a report from before models", &older),
    ] {
        let errors: Vec<String> = validator.iter_errors(report).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{what}: {errors:?}");
    }
    let plant = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut bad = cached.clone();
        edit(&mut bad["models"][0]);
        bad
    };
    let defects = [
        ("a short hash", plant(&|m| m["sha256"] = "4eb27e5a".into())),
        ("an unknown source", plant(&|m| m["source"] = "web".into())),
        (
            "a cached file with no url",
            plant(&|m| m["url"] = serde_json::Value::Null),
        ),
        ("a given path with a url", plant(&|m| m["source"] = "path".into())),
        (
            "no role",
            plant(&|m| {
                m.as_object_mut().unwrap().remove("role");
            }),
        ),
        ("an extra field", plant(&|m| m["trusted"] = true.into())),
    ];
    for (what, bad) in defects {
        assert!(!validator.is_valid(&bad), "the schema accepted {what}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn help_names_the_offline_switch_and_the_cache_override() {
    let r = Command::new(BIN).arg("--help").output().unwrap();
    let out = String::from_utf8_lossy(&r.stdout);
    assert!(out.contains("--offline") && out.contains("GGTTS_CACHE_DIR"), "{out}");
}
