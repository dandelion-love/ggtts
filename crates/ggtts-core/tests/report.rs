//! The report envelope `ggtts-report/1` and the merger.
//!
//! - **Runtime check and schema agree.** `report::validate` and the published schema
//!   (`schemas/ggtts-report-1.schema.json`) agree on every valid envelope and every planted defect below. The one
//!   exception is the rule JSON Schema cannot state, an interval's low above its high, which is named.
//! - **Nothing is dropped.** An envelope that cannot be read, does not validate, names another run or repeats an
//!   instrument identity becomes an `invalid` entry with every reason.
//! - **Completeness.** A required instrument that is missing, invalid, or ended other than `ok` makes the measurement
//!   INCOMPLETE.
//! - **The merged report.** `report.json` validates against the schema's report definition, and the report does not
//!   depend on the order envelopes were found in.

use std::path::{Path, PathBuf};

use ggtts_core::report::{merge, validate, Input, Report, Status, FORMAT};
use serde_json::{json, Value};

const RUN: &str = "run-2026-11-06-a";

fn schema() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-report-2.schema.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// A validator for one of the schema's definitions: the root (the envelope) or `$defs/report`.
fn validator(def: Option<&str>) -> jsonschema::Validator {
    let mut s = schema();
    if let Some(d) = def {
        s["$ref"] = json!(format!("#/$defs/{d}"));
    }
    jsonschema::validator_for(&s).expect("the schema compiles")
}

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

/// A full envelope from `name`: per-chunk metrics scoped to a render, a descriptive and a diagnostic value, and
/// complete provenance.
fn env(name: &str) -> Value {
    json!({
        "format": FORMAT,
        "name": name,
        "version": "0.1.0",
        "identity": {"run": RUN, "instrument": format!("{name}#1")},
        "provenance": {
            "config": {"edge_s": 0.4, "threshold_db": -45},
            "ggtts": "0.0.1+3a9ee51",
            "engine": "qwen3-tts",
            "backend": "cpu",
            "model_sha256": sha('a'),
            "seeds": [{"render": "story-1", "chunk": 0, "seed": 20261106}, {"render": "story-1", "chunk": 1, "seed": 7}],
            "inputs": [{"path": "renders/story-1.flac", "sha256": sha('b')}],
            "audio": "trimmed"
        },
        "hooks": ["chunk_end", "end"],
        "status": "ok",
        "metrics": [
            {"name": "rtf", "value": 0.31, "unit": "1", "direction": "lower", "interval": [0.29, 0.33]},
            {"name": "spc", "value": 0.27, "unit": "s/char", "direction": "neutral", "render": "story-1", "chunk": 0},
            {"name": "spc", "value": null, "unit": "s/char", "direction": "neutral", "render": "story-1", "chunk": 1},
            {"name": "frames", "value": 412, "unit": "1", "direction": "n/a", "render": "story-1"}
        ],
        "artifacts": [{"path": "instruments/timings/steps.csv", "type": "text/csv", "sha256": sha('c')}],
        "log": "instruments/timings/log.txt"
    })
}

#[test]
fn v2_unknown_is_explicit_and_v1_remains_readable() {
    let mut old = env("timings");
    old["format"] = json!("ggtts-report/1");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-report-1.schema.json");
    let old_schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let old_validator = jsonschema::validator_for(&old_schema).unwrap();
    assert!(old_validator.is_valid(&old));
    assert!(validate(&old).is_ok());
    let report = merge(
        RUN,
        &[Input {
            source: "legacy.json".into(),
            text: old.to_string(),
        }],
        &[],
    );
    assert!(validator(Some("report")).is_valid(&report.to_json()));
    old["provenance"]["audio"] = json!("unknown");
    assert!(!old_validator.is_valid(&old));
    assert!(validate(&old).is_err());
    old["format"] = json!(FORMAT);
    assert!(validator(None).is_valid(&old));
    assert!(validate(&old).is_ok());
}

fn with(name: &str, f: impl Fn(&mut Value)) -> Value {
    let mut v = env(name);
    f(&mut v);
    v
}

/// Valid envelopes: one per status an instrument reports, the smallest one, and one carrying `$schema`.
fn valid() -> Vec<(&'static str, Value)> {
    vec![
        ("ok", env("timings")),
        (
            "failed",
            with("cer", |v| {
                v["status"] = json!("failed");
                v["error"] = json!("exit status 3");
                v["metrics"] = json!([]);
            }),
        ),
        (
            "timeout",
            with("pauses", |v| {
                v["status"] = json!("timeout");
                v["error"] = json!("no envelope after 30 s");
            }),
        ),
        (
            "skipped",
            with("loudness-ends", |v| {
                v["status"] = json!("skipped");
                v["log"] = json!(null);
            }),
        ),
        (
            "truncated",
            with("stop-probability", |v| {
                v["status"] = json!("truncated");
                v["error"] = json!("the stream has no end event");
            }),
        ),
        (
            "smallest",
            json!({"format": FORMAT, "name": "x", "version": "1", "identity": {"run": RUN, "instrument": "x"},
                   "provenance": {"config": {}, "ggtts": null, "engine": null, "backend": null, "model_sha256": null,
                                  "seeds": [], "inputs": [], "audio": "n/a"},
                   "hooks": [], "status": "ok", "metrics": [], "artifacts": [], "log": null}),
        ),
        (
            "with $schema",
            with("with-schema", |v| {
                v["$schema"] = json!("urn:ggtts:schema:ggtts-report:1")
            }),
        ),
    ]
}

fn rm(v: &mut Value, ptr: &str) {
    let (parent, key) = ptr.rsplit_once('/').unwrap();
    v.pointer_mut(parent).unwrap().as_object_mut().unwrap().remove(key);
}

/// Planted defects: each must be refused by both checks.
fn planted() -> Vec<(&'static str, Value)> {
    let e = |f: &dyn Fn(&mut Value)| {
        let mut v = env("timings");
        f(&mut v);
        v
    };
    vec![
        ("not an object", json!([1, 2])),
        ("no status", e(&|v| rm(v, "/status"))),
        ("a status not in the list", e(&|v| v["status"] = json!("done"))),
        ("the next format", e(&|v| v["format"] = json!("ggtts-report/3"))),
        ("an upper-case name", e(&|v| v["name"] = json!("Timings"))),
        ("an empty version", e(&|v| v["version"] = json!(""))),
        ("an empty hook", e(&|v| v["hooks"] = json!([""]))),
        ("a metric without a unit", e(&|v| rm(v, "/metrics/0/unit"))),
        ("a metric without a direction", e(&|v| rm(v, "/metrics/0/direction"))),
        (
            "a direction not in the list",
            e(&|v| v["metrics"][0]["direction"] = json!("up")),
        ),
        ("a value as a string", e(&|v| v["metrics"][0]["value"] = json!("0.31"))),
        (
            "a missing value left out instead of null",
            e(&|v| rm(v, "/metrics/0/value")),
        ),
        (
            "an interval of one number",
            e(&|v| v["metrics"][0]["interval"] = json!([0.3])),
        ),
        ("a chunk without its render", e(&|v| rm(v, "/metrics/1/render"))),
        (
            "an undeclared metric field",
            e(&|v| v["metrics"][0]["note"] = json!("x")),
        ),
        ("a metric outside the list", e(&|v| v["rtf"] = json!(0.31))),
        ("an artifact without a hash", e(&|v| rm(v, "/artifacts/0/sha256"))),
        (
            "an artifact with a short hash",
            e(&|v| v["artifacts"][0]["sha256"] = json!("abc")),
        ),
        ("no identity", e(&|v| rm(v, "/identity"))),
        ("no run in the identity", e(&|v| rm(v, "/identity/run"))),
        (
            "an empty instrument identity",
            e(&|v| v["identity"]["instrument"] = json!("")),
        ),
        ("no provenance", e(&|v| rm(v, "/provenance"))),
        ("no effective configuration", e(&|v| rm(v, "/provenance/config"))),
        (
            "a model hash left out instead of null",
            e(&|v| rm(v, "/provenance/model_sha256")),
        ),
        (
            "an upper-case model hash",
            e(&|v| v["provenance"]["model_sha256"] = json!(sha('A'))),
        ),
        (
            "a seed that is negative",
            e(&|v| v["provenance"]["seeds"][0]["seed"] = json!(-1)),
        ),
        ("an input without a hash", e(&|v| rm(v, "/provenance/inputs/0/sha256"))),
        ("no raw-or-trimmed state", e(&|v| rm(v, "/provenance/audio"))),
        (
            "an audio state not in the list",
            e(&|v| v["provenance"]["audio"] = json!("cut")),
        ),
        ("no log field", e(&|v| rm(v, "/log"))),
        ("an empty log path", e(&|v| v["log"] = json!(""))),
        ("an error that is a number", e(&|v| v["error"] = json!(3))),
    ]
}

#[test]
fn the_runtime_check_and_the_schema_agree() {
    let schema = validator(None);
    for (name, v) in valid() {
        assert!(
            validate(&v).is_ok(),
            "validate refused a valid envelope ({name}): {:?}",
            validate(&v)
        );
        let errs: Vec<String> = schema.iter_errors(&v).map(|e| e.to_string()).collect();
        assert!(
            errs.is_empty(),
            "the schema refused a valid envelope ({name}): {errs:?}"
        );
    }
    for (name, v) in planted() {
        assert!(validate(&v).is_err(), "validate accepted a planted defect: {name}");
        assert!(!schema.is_valid(&v), "the schema accepted a planted defect: {name}");
    }
    // The one rule JSON Schema cannot state: the runtime check is the stricter, and says so.
    let flipped = with("timings", |v| v["metrics"][0]["interval"] = json!([0.33, 0.29]));
    assert!(
        schema.is_valid(&flipped),
        "if the schema learns this rule, drop the exception here"
    );
    let why = validate(&flipped).unwrap_err();
    assert!(why.iter().any(|w| w.contains("low <= high")), "{why:?}");
}

#[test]
fn validate_names_every_reason_not_only_the_first() {
    let v = with("timings", |v| {
        v["status"] = json!("done");
        v["metrics"][0]["direction"] = json!("up");
        v["extra"] = json!(1);
        rm(v, "/provenance/audio");
    });
    let why = validate(&v).unwrap_err();
    assert_eq!(why.len(), 4, "{why:?}");
}

#[test]
fn a_missing_value_is_null_not_zero_and_neutral_directions_survive() {
    let e = validate(&env("timings")).unwrap();
    let spc: Vec<Option<f64>> = e.metrics.iter().filter(|m| m.name == "spc").map(|m| m.value).collect();
    assert_eq!(spc, vec![Some(0.27), None]);
    let r = merge(
        RUN,
        &[Input {
            source: "t.json".into(),
            text: env("timings").to_string(),
        }],
        &[],
    );
    let j = r.to_json();
    assert_eq!(
        j["instruments"][0]["metrics"][2]["value"],
        Value::Null,
        "null is written as null"
    );
    let md = r.to_markdown();
    assert!(
        md.contains("| spc | missing | s/char | neutral |  | story-1 chunk 1 |"),
        "{md}"
    );
    assert!(md.contains("| frames | 412 | 1 | n/a |  | story-1 |"));
    assert!(md.contains("| rtf | 0.31 | 1 | lower | [0.29, 0.33] | run |"));
}

#[test]
fn adding_a_metric_never_changes_the_format() {
    let more = with("timings", |v| {
        for i in 0..5 {
            v["metrics"]
                .as_array_mut()
                .unwrap()
                .push(json!({"name": format!("m{i}"), "value": i, "unit": "ms", "direction": "lower"}));
        }
    });
    let (a, b) = (validate(&env("timings")).unwrap(), validate(&more).unwrap());
    assert_eq!((a.format.as_str(), b.format.as_str()), (FORMAT, FORMAT));
    assert_eq!(b.metrics.len(), a.metrics.len() + 5);
    let r = merge(
        RUN,
        &[Input {
            source: "more.json".into(),
            text: more.to_string(),
        }],
        &[],
    );
    assert_eq!(r.to_json()["format"], FORMAT);
    assert!(validator(Some("report")).is_valid(&r.to_json()));
    assert!(r.to_markdown().contains("| m4 | 4 | ms | lower |"));
}

fn inputs() -> Vec<Input> {
    let mut v: Vec<Input> = valid()
        .into_iter()
        .filter(|(n, _)| *n != "with $schema")
        .map(|(n, e)| Input {
            source: format!("run/{}.json", n.replace(' ', "-")),
            text: e.to_string(),
        })
        .collect();
    v.push(Input {
        source: "run/<cer judge b>.json".into(),
        text: "{ not json".into(),
    });
    v.push(Input {
        source: "run/runaways.json".into(),
        text: with("runaways", |v| v["status"] = json!("done")).to_string(),
    });
    v.push(Input {
        source: "run/stale.json".into(),
        text: with("stale", |v| v["identity"]["run"] = json!("run-2026-11-05-z")).to_string(),
    });
    // Two envelopes claiming one instrument identity: neither can be trusted.
    for n in 1..=2 {
        v.push(Input {
            source: format!("run/dup-{n}.json"),
            text: with("dup", |v| v["identity"]["instrument"] = json!("dup#1")).to_string(),
        });
    }
    v
}

const REQUIRED: [&str; 4] = ["timings", "cer", "asr-judge", "runaways"];

#[test]
fn every_input_is_one_entry_and_an_invalid_one_is_visible_as_invalid() {
    let ins = inputs();
    let r = merge(RUN, &ins, &REQUIRED);
    assert_eq!(r.entries.len(), ins.len(), "an envelope was dropped");
    let counts: Vec<(Status, usize)> = r.counts().to_vec();
    // ok: timings + smallest. invalid: not JSON, bad status, another run, the two duplicates.
    assert_eq!(
        counts,
        vec![
            (Status::Ok, 2),
            (Status::Failed, 1),
            (Status::Timeout, 1),
            (Status::Skipped, 1),
            (Status::Truncated, 1),
            (Status::Invalid, 5)
        ]
    );
    let by_source = |s: &str| r.entries.iter().find(|e| e.source == s).unwrap();
    let not_json = by_source("run/<cer judge b>.json");
    assert_eq!(not_json.envelope.status, Status::Invalid);
    assert_eq!(
        not_json.envelope.name, "cer-judge-b-",
        "named from its source when it has no name of its own"
    );
    assert!(not_json.invalid[0].starts_with("not JSON"));
    let wrong = by_source("run/runaways.json");
    assert_eq!(
        (wrong.envelope.name.as_str(), wrong.envelope.version.as_str()),
        ("runaways", "0.1.0"),
        "keeps its own name and version when they are valid"
    );
    assert!(wrong.envelope.error.as_deref().unwrap().contains("status \"done\""));
    let stale = by_source("run/stale.json");
    assert!(
        stale.invalid[0].contains("belongs to run \"run-2026-11-05-z\""),
        "{:?}",
        stale.invalid
    );
    assert_eq!(
        stale.envelope.identity.run, RUN,
        "listed under this run, with the reason"
    );
    for d in ["run/dup-1.json", "run/dup-2.json"] {
        assert!(by_source(d).invalid[0].contains("appears more than once"), "{d}");
    }

    let j = r.to_json();
    let errs: Vec<String> = validator(Some("report"))
        .iter_errors(&j)
        .map(|e| e.to_string())
        .collect();
    assert!(errs.is_empty(), "{errs:?}\n{j:#}");
    assert_eq!(j["summary"]["invalid"], 5);
    assert_eq!(j["instruments"].as_array().unwrap().len(), ins.len());

    let (md, html) = (r.to_markdown(), r.to_html());
    for e in &r.entries {
        let st = e.envelope.status.as_str();
        assert!(
            md.contains(&format!("| {} |", e.envelope.name)),
            "{} missing from report.md",
            e.envelope.name
        );
        assert!(md.contains(&format!("**{st}**")));
        assert!(
            html.contains(&format!("<td>{}</td>", e.envelope.name)),
            "{} missing from report.html",
            e.envelope.name
        );
    }
    assert!(md.contains("invalid envelope: not JSON") && html.contains("invalid envelope: not JSON"));
    assert!(md.contains("no envelope after 30 s"), "a timeout's reason is shown");
    assert!(
        md.contains("| loudness-ends | 0.1.0 | **skipped** |"),
        "a skipped instrument is listed"
    );
}

#[test]
fn a_required_instrument_that_is_missing_failed_or_invalid_makes_the_measurement_incomplete() {
    let r = merge(RUN, &inputs(), &REQUIRED);
    let met: Vec<(&str, bool)> = r.required.iter().map(|q| (q.name.as_str(), q.met())).collect();
    assert_eq!(
        met,
        vec![
            ("timings", true),
            ("cer", false),
            ("asr-judge", false),
            ("runaways", false)
        ],
        "timings ok; cer failed; asr-judge absent; runaways invalid"
    );
    let q = |n: &str| r.required.iter().find(|q| q.name == n).unwrap();
    assert!(!q("asr-judge").present);
    assert!(q("runaways").present && !q("runaways").valid);
    assert!(!r.complete());
    let j = r.to_json();
    assert_eq!(j["measurement"], "incomplete");
    assert!(r.to_markdown().contains("## Measurement: **INCOMPLETE**"));
    assert!(r.to_markdown().contains("| asr-judge | **no** | yes | - |"));

    // Every status but ok leaves a required instrument unmet, and the no-op (all ok) is complete.
    for st in ["failed", "timeout", "skipped", "truncated", "invalid"] {
        let one = merge(
            RUN,
            &[Input {
                source: "a.json".into(),
                text: with("a", |v| v["status"] = json!(st)).to_string(),
            }],
            &["a"],
        );
        assert!(!one.complete(), "a required instrument with status {st} counted as met");
    }
    let fine = merge(
        RUN,
        &[Input {
            source: "a.json".into(),
            text: env("a").to_string(),
        }],
        &["a"],
    );
    assert!(fine.complete());
    assert_eq!(fine.to_json()["measurement"], "complete");
    assert!(merge(RUN, &[], &[]).complete(), "nothing required, nothing missing");
    assert!(
        !merge(RUN, &[], &["a"]).complete(),
        "an empty run with a requirement is incomplete"
    );
}

#[test]
fn the_report_does_not_depend_on_the_order_envelopes_were_found() {
    let ins = inputs();
    let mut rev = ins.clone();
    rev.reverse();
    let (a, b) = (merge(RUN, &ins, &REQUIRED), merge(RUN, &rev, &REQUIRED));
    assert_eq!(a.to_json(), b.to_json());
    assert_eq!(a.to_markdown(), b.to_markdown());
    assert_eq!(a.to_html(), b.to_html());
}

#[test]
fn nothing_an_envelope_says_can_break_the_html_or_the_table() {
    let e = with("timings", |v| {
        v["version"] = json!("1 | <script>alert(1)</script>");
        v["error"] = json!("a\nb | c");
        v["status"] = json!("failed");
    });
    let r = merge(
        RUN,
        &[Input {
            source: "x.json".into(),
            text: e.to_string(),
        }],
        &[],
    );
    let (md, html) = (r.to_markdown(), r.to_html());
    assert!(!html.contains("<script>") && html.contains("&lt;script&gt;"));
    let row = md.lines().find(|l| l.starts_with("| timings |")).unwrap();
    assert_eq!(
        row.matches(" | ").count(),
        4,
        "a pipe or newline in a value split the row: {row}"
    );
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-report-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn write_puts_the_three_files_side_by_side() {
    let d = tmp("write");
    let r: Report = merge(RUN, &inputs(), &REQUIRED);
    r.write(&d).unwrap();
    let j: Value = serde_json::from_str(&std::fs::read_to_string(d.join("report.json")).unwrap()).unwrap();
    assert_eq!(j, r.to_json());
    assert_eq!(std::fs::read_to_string(d.join("report.md")).unwrap(), r.to_markdown());
    assert_eq!(std::fs::read_to_string(d.join("report.html")).unwrap(), r.to_html());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_valid_envelope_round_trips() {
    for (name, v) in valid() {
        let e = validate(&v).unwrap();
        let back = validate(&e.to_json()).unwrap_or_else(|w| panic!("{name}: {w:?}"));
        assert_eq!(back, e, "{name}");
    }
}

#[test]
fn the_merged_report_carries_the_model_files_when_known() {
    let mut r = merge(RUN, &inputs(), &REQUIRED);
    assert!(
        r.to_json().get("models").is_none(),
        "unknown model files are left out, not invented"
    );
    r.models = Some(vec![
        json!({"role": "talker", "path": "/cache/models/t.gguf", "sha256": sha('a'),
        "source": "cache", "url": "https://example.invalid/t.gguf"}),
    ]);
    let v = r.to_json();
    assert_eq!(v["models"][0]["sha256"], sha('a'));
    assert!(validator(Some("report")).is_valid(&v));
    let mut bad = v.clone();
    bad["models"][0]["source"] = json!("web");
    assert!(!validator(Some("report")).is_valid(&bad));
}
