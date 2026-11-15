//! The options value (`ggtts_core::options`): its schema, its refusals, its provenance record, the conversion to
//! `RenderOptions`, and the environment rule.

use std::path::{Path, PathBuf};

use ggtts_core::engine::Engine;
use ggtts_core::options::{self, record, Options, Refusal, Source, TABLE};
use ggtts_core::pipeline::RenderOptions;
use ggtts_core::profile::Profile;
use ggtts_core::trace::{Event, Header};
use serde_json::{json, Value};

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

const SCHEMA: &str = "schemas/ggtts-options-1.schema.json";

fn fixtures() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = std::fs::read_dir(root().join("fixtures/options"))
        .expect("fixtures/options")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read_to_string(&p).unwrap(),
            )
        })
        .collect();
    out.sort();
    assert!(out.len() >= 5, "a fixture per section");
    out
}

fn validator() -> jsonschema::Validator {
    let text = std::fs::read_to_string(root().join(SCHEMA)).expect("the committed schema");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

// ── AC2: the schema ──

/// The committed schema is the one the table generates. `GGTTS_BLESS=1` rewrites it; the diff is then reviewed.
#[test]
fn the_committed_schema_is_generated_from_the_table() {
    let generated = serde_json::to_string_pretty(&options::schema()).unwrap() + "\n";
    let path = root().join(SCHEMA);
    if std::env::var_os("GGTTS_BLESS").is_some() {
        std::fs::write(&path, &generated).unwrap();
    }
    // A Windows checkout may hold the file with CRLF line ends (core.autocrlf); the content is what is compared.
    let committed = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
    assert!(
        committed == generated,
        "{SCHEMA} is not what options::schema() generates: run the test with GGTTS_BLESS=1 and review the diff"
    );
}

#[test]
fn the_schema_validates_the_default_and_every_fixture() {
    let v = validator();
    let default = Options::default().to_json();
    assert!(v.is_valid(&default), "{default}");
    for (name, text) in fixtures() {
        let doc: Value = serde_json::from_str(&text).unwrap();
        let errors: Vec<String> = v.iter_errors(&doc).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
        // What the parser writes back is valid too.
        let parsed = Options::from_json(&text).unwrap_or_else(|r| panic!("{name}: {r}"));
        assert!(v.is_valid(&parsed.to_json()), "{name} written back");
    }
}

#[test]
fn a_document_survives_a_write_and_a_read() {
    let x = Options::default();
    assert_eq!(Options::from_json(&x.to_json().to_string()).unwrap(), x);
    for (name, text) in fixtures() {
        let x = Options::from_json(&text).unwrap();
        let back = Options::from_json(&x.to_json().to_string()).unwrap();
        assert_eq!(back, x, "{name}");
        // Every value the document gave is written back as it was given.
        let doc: Value = serde_json::from_str(&text).unwrap();
        let written = x.to_json();
        for (section, entries) in doc.as_object().unwrap() {
            let Some(entries) = entries.as_object() else { continue };
            for (k, v) in entries {
                assert_eq!(written[section].get(k), Some(v), "{name}: {section}.{k}");
            }
        }
    }
}

/// The fixtures between them give every option in the table, so a row the value lacks (or a field the table lacks)
/// fails the read above rather than passing unseen.
#[test]
fn the_fixtures_give_every_option() {
    let mut given = std::collections::BTreeSet::new();
    for (_, text) in fixtures() {
        let doc: Value = serde_json::from_str(&text).unwrap();
        for (section, entries) in doc.as_object().unwrap() {
            if let Some(entries) = entries.as_object() {
                for k in entries.keys() {
                    given.insert(format!("{section}.{k}"));
                }
            }
        }
    }
    let missing: Vec<String> = TABLE.iter().map(|d| d.key()).filter(|k| !given.contains(k)).collect();
    assert!(missing.is_empty(), "no fixture gives {missing:?}");
}

/// Every table default is what `Options::default()` holds.
#[test]
fn the_table_defaults_are_the_value_defaults() {
    let doc = Options::default().to_json();
    for d in TABLE {
        let held = doc[d.section.name()].get(d.name).cloned();
        let unset = match d.default_value() {
            Value::Null => None,
            Value::Bool(false) => None,
            Value::Array(a) if a.is_empty() => None,
            v => Some(v),
        };
        assert_eq!(held, unset, "{}", d.key());
    }
}

// ── AC4: refusals ──

fn refused_key(r: Result<Options, Refusal>) -> String {
    match r {
        Ok(o) => panic!("accepted: {}", o.to_json()),
        Err(r) => {
            assert!(!r.message.is_empty());
            r.key
        }
    }
}

#[test]
fn an_unknown_key_is_refused_by_name() {
    assert_eq!(
        refused_key(Options::from_json(r#"{"render": {"speed": 2}}"#)),
        "render.speed"
    );
    assert_eq!(refused_key(Options::from_json(r#"{"voices": {}}"#)), "voices");
    assert_eq!(refused_key(Options::from_args(&args(&["--speed", "2"]))), "--speed");
}

#[test]
fn a_wrong_type_is_refused_by_name() {
    assert_eq!(
        refused_key(Options::from_json(r#"{"load": {"threads": "four"}}"#)),
        "load.threads"
    );
    assert_eq!(
        refused_key(Options::from_args(&args(&["--threads", "four"]))),
        "load.threads"
    );
}

#[test]
fn a_steer_scale_that_is_not_a_number_is_refused_by_name() {
    // JSON has no NaN: the flag can spell one, and a document can only spell a string.
    assert_eq!(
        refused_key(Options::from_args(&args(&[
            "--steer",
            "s.safetensors",
            "--steer-scale",
            "NaN"
        ]))),
        "render.steer_scale"
    );
    assert_eq!(
        refused_key(Options::from_args(&args(&[
            "--steer",
            "s.safetensors",
            "--steer-scale",
            "inf"
        ]))),
        "render.steer_scale"
    );
    assert_eq!(
        refused_key(Options::from_json(
            r#"{"render": {"steer": "s.safetensors", "steer_scale": "NaN"}}"#
        )),
        "render.steer_scale"
    );
}

#[test]
fn trace_level_3_is_refused_by_name() {
    assert_eq!(
        refused_key(Options::from_json(
            r#"{"observe": {"trace": "t.jsonl", "trace_level": 3}}"#
        )),
        "observe.trace_level"
    );
    assert_eq!(
        refused_key(Options::from_args(&args(&["--trace", "t.jsonl", "--trace-level", "3"]))),
        "observe.trace_level"
    );
}

#[test]
fn stream_raw_without_stream_is_refused_by_name() {
    assert_eq!(
        refused_key(Options::from_json(r#"{"output": {"stream_raw": true}}"#)),
        "output.stream_raw"
    );
    assert_eq!(
        refused_key(Options::from_args(&args(&["--out", "x.flac", "--stream-raw"]))),
        "output.stream_raw"
    );
}

#[test]
fn a_load_key_in_a_render_document_is_refused_by_name() {
    let doc = r#"{"render": {"seed_salt": "x"}, "load": {"device": "cuda"}}"#;
    // The same key is fine in a whole document, where the engine is opened with it.
    assert_eq!(Options::from_json(doc).unwrap().load.device.as_deref(), Some("cuda"));
    assert_eq!(refused_key(Options::from_render_json(doc)), "load.device");
    // A render document without one is read.
    let r = Options::from_render_json(r#"{"render": {"seed_salt": "x"}}"#).unwrap();
    assert_eq!(r.render.seed_salt.as_deref(), Some("x"));
}

/// Each refused document is refused by the schema too, except the ones only the parser can see (a load key in a
/// render document is a valid whole document).
#[test]
fn the_schema_refuses_what_the_parser_refuses() {
    let v = validator();
    for doc in [
        json!({"render": {"speed": 2}}),
        json!({"load": {"threads": "four"}}),
        json!({"render": {"steer": "s", "steer_scale": "NaN"}}),
        json!({"observe": {"trace": "t", "trace_level": 3}}),
        json!({"output": {"stream_raw": true}}),
        json!({"version": 2}),
    ] {
        assert!(!v.is_valid(&doc), "the schema accepts {doc}");
        assert!(Options::from_value(doc.clone()).is_err(), "the parser accepts {doc}");
    }
}

// ── AC6: RenderOptions from the render section ──

fn same(a: &RenderOptions, b: &RenderOptions) -> bool {
    a.redraw == b.redraw
        && a.seed_salt == b.seed_salt
        && a.edge_s.to_bits() == b.edge_s.to_bits()
        && a.measure_timings == b.measure_timings
        && a.instruction == b.instruction
        && a.steer == b.steer
}

fn profiles() -> Vec<Profile> {
    vec![*ggtts_fake::FakeEngine::default().profile(), ggtts_qwen3_tts::PROFILE]
}

#[test]
fn the_default_render_options_are_the_profiles() {
    for p in profiles() {
        let got = Options::default().render_options(&p).unwrap();
        assert!(same(&got, &RenderOptions::for_profile(&p)), "{}: {got:?}", p.name);
    }
}

fn steer_file(dir: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-options-{dir}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("steer.safetensors");
    let a = [0.5f32, -1.0, 2.0];
    let b = [1.0f32, 1.0, 1.0];
    std::fs::write(
        &path,
        ggtts_core::steer::write_safetensors(&[("talker.layer.3", &a), ("talker.layer.5", &b)]),
    )
    .unwrap();
    path
}

#[test]
fn every_render_field_lands_in_its_render_options_field() {
    let steer = steer_file("fields");
    for p in profiles() {
        let doc = json!({"render": {
            "redraw": p.redraw + 2,
            "seed_salt": "take-2",
            "edge_s": p.edge_s + 0.5,
            "measure_timings": true,
            "instruction": "Speak slowly and softly.",
            "steer": steer,
            "steer_scale": 0.125,
            "steer_layers": [5]
        }});
        let mut o = Options::from_value(doc).unwrap();
        // A steering file is read before it is rendered with, and not before.
        assert_eq!(o.render_options(&p).unwrap_err().key, "render.steer");
        o.read_steering().unwrap();
        let got = o.render_options(&p).unwrap();
        assert_eq!(got.redraw, p.redraw + 2);
        assert_eq!(got.seed_salt, "take-2");
        assert_eq!(got.edge_s, p.edge_s + 0.5);
        assert!(got.measure_timings);
        assert_eq!(got.instruction.as_deref(), Some("Speak slowly and softly."));
        let s = got.steer.expect("steering");
        assert_eq!(s.layer_indices(), vec![5]);
        assert_eq!(s.scale, 0.125);
    }
    // A timing report asks for timings too, as ggtts-render does today.
    let o = Options::from_json(r#"{"observe": {"timing_report": "t.json"}}"#).unwrap();
    assert!(o.render_options(&profiles()[0]).unwrap().measure_timings);
}

// ── AC3: the record is what ggtts-render writes ──

/// ggtts-render's header and report code, as it is on `rc` (`render_with_models`): the instruction when not empty,
/// the steering's record, the engine's info and the trace level.
fn render_header(
    engine: &dyn Engine,
    instruction: Option<&str>,
    steering: Option<&ggtts_core::steer::Steering>,
    level: u8,
) -> Value {
    let instruction = instruction.map(str::to_string).filter(|i| !i.is_empty());
    let steer_record = steering.map(ggtts_core::steer::Steering::record);
    let mut header = Header::for_engine(engine, level);
    header.instruction.clone_from(&instruction);
    header.steer.clone_from(&steer_record);
    Value::Object(Event::Trace(&header).to_json())
}

fn render_report(instruction: Option<&str>, steering: Option<&ggtts_core::steer::Steering>) -> Value {
    let instruction = instruction.map(str::to_string).filter(|i| !i.is_empty());
    let steer_record = steering.map(ggtts_core::steer::Steering::record);
    let mut doc = json!({});
    if let Some(i) = &instruction {
        doc["instruction"] = i.as_str().into();
    }
    if let Some(s) = &steer_record {
        doc["steer"] = s.clone();
    }
    doc
}

#[test]
fn the_record_has_the_fields_ggtts_render_writes() {
    let engine = ggtts_fake::FakeEngine::default();
    let steer = steer_file("record");
    let cases: Vec<(Option<&str>, bool, Option<u8>)> = vec![
        (None, false, None),
        (Some(""), false, Some(2)),
        (Some("Speak slowly and softly."), true, None),
        (Some("Speak quickly."), true, Some(2)),
    ];
    for (instruction, steered, level) in cases {
        let mut flags = vec!["--trace".to_string(), "t.jsonl".to_string()];
        if let Some(i) = instruction {
            flags.extend(["--instruction".into(), i.into()]);
        }
        if steered {
            flags.extend([
                "--steer".into(),
                steer.display().to_string(),
                "--steer-scale".into(),
                "0.5".into(),
            ]);
        }
        if let Some(l) = level {
            flags.extend(["--trace-level".into(), l.to_string()]);
        }
        let mut o = Options::from_args(&flags).unwrap();
        o.read_steering().unwrap();
        let rec = record(&o, engine.info());
        let header = render_header(&engine, instruction, o.render.steering.as_ref(), level.unwrap_or(1));
        for k in ["engine", "backend", "parity", "trace_level", "instruction", "steer"] {
            assert_eq!(rec.get(k), header.get(k), "{k} for {flags:?}");
        }
        let report = render_report(instruction, o.render.steering.as_ref());
        for k in ["instruction", "steer"] {
            assert_eq!(rec.get(k), report.get(k), "report {k} for {flags:?}");
        }
    }
}

#[test]
fn the_record_names_each_seed_key_apart() {
    let engine = ggtts_fake::FakeEngine::default();
    let o = Options::from_json(
        r#"{"load": {"speaker": "hk_voice_343"}, "render": {"seed": 7, "seed_salt": "s"}, "book": {"seed_root": "r"}}"#,
    )
    .unwrap();
    let rec = record(&o, engine.info());
    assert_eq!(rec["seed"], 7);
    assert_eq!(rec["seed_salt"], "s");
    assert_eq!(rec["seed_root"], "r");
    assert_eq!(rec["speaker"], "hk_voice_343");
    assert_eq!(rec["device"], "cpu");
    assert_eq!(rec["trace_level"], 0);
    assert!(rec.get("ignored").is_none());
    let bare = record(&Options::default(), engine.info());
    for k in ["seed", "seed_salt", "seed_root", "speaker", "instruction", "steer"] {
        assert!(bare.get(k).is_none(), "{k} in a record with none: {bare}");
    }
}

// ── AC7: the environment rule ──

fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
}

#[test]
fn each_value_is_marked_with_its_source() {
    let e = Options::from_env(&env(&[("GGTTS_LOG_LEVEL", "warn"), ("GGTTS_CACHE_DIR", "/c")])).unwrap();
    assert_eq!(e.marks.sources.get("observe.log_level"), Some(&Source::Environment));
    assert_eq!(e.marks.sources.get("load.cache_dir"), Some(&Source::Environment));
    let f = Options::from_args(&args(&["--log-level", "debug", "--threads", "2"])).unwrap();
    assert_eq!(f.marks.sources.get("observe.log_level"), Some(&Source::Flag));
    let d = Options::from_json(r#"{"load": {"threads": 2}}"#).unwrap();
    assert_eq!(d.marks.sources.get("load.threads"), Some(&Source::Document));
    assert_eq!(d.marks.sources.len(), 1, "only what was given is marked");

    // A flag wins over the environment.
    let both = e.layer(f).unwrap();
    assert_eq!(both.observe.log_level.as_deref(), Some("debug"));
    assert_eq!(both.marks.sources.get("observe.log_level"), Some(&Source::Flag));
    assert_eq!(both.load.cache_dir.as_deref(), Some(Path::new("/c")));
    assert_eq!(both.load.threads, Some(2));
    // An empty variable is unset; a bad one is refused by name.
    assert!(Options::from_env(&env(&[("GGTTS_LOG_LEVEL", "")]))
        .unwrap()
        .observe
        .log_level
        .is_none());
    assert_eq!(
        refused_key(Options::from_env(&env(&[("GGTTS_LOG_FORMAT", "yaml")]))),
        "observe.log_format"
    );
}

/// A fake engine that lacks `load.cache_dir` (it needs no files) and `observe.log_level`.
fn lacks(key: &str) -> bool {
    !matches!(key, "load.cache_dir" | "observe.log_level")
}

#[test]
fn an_environment_value_the_engine_lacks_is_ignored_with_one_warning_and_recorded() {
    let engine = ggtts_fake::FakeEngine::default();
    let mut o = Options::from_env(&env(&[("GGTTS_CACHE_DIR", "/c"), ("GGTTS_LOG_FORMAT", "json")]))
        .unwrap()
        .layer(Options::from_args(&args(&["--threads", "2"])).unwrap())
        .unwrap();
    let mut warnings = Vec::new();
    o.apply_engine_rule("fake", &lacks, &mut |w| warnings.push(w.to_string()))
        .unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("GGTTS_CACHE_DIR"), "{warnings:?}");
    assert!(o.load.cache_dir.is_none(), "an ignored value is not used");
    assert_eq!(
        o.observe.log_format.as_deref(),
        Some("json"),
        "a value the engine has stays"
    );
    assert_eq!(o.load.threads, Some(2));
    let rec = record(&o, engine.info());
    assert_eq!(
        rec["ignored"],
        json!([{"key": "load.cache_dir", "variable": "GGTTS_CACHE_DIR"}])
    );
}

#[test]
fn a_flag_or_document_value_the_engine_lacks_is_refused() {
    let mut warnings = 0;
    let mut f = Options::from_env(&env(&[("GGTTS_LOG_LEVEL", "warn")]))
        .unwrap()
        .layer(Options::from_args(&args(&["--log-level", "debug"])).unwrap())
        .unwrap();
    let r = f.apply_engine_rule("fake", &lacks, &mut |_| warnings += 1).unwrap_err();
    assert_eq!(r.key, "observe.log_level");
    let mut d = Options::from_json(r#"{"load": {"cache_dir": "/c"}}"#).unwrap();
    let r = d.apply_engine_rule("fake", &lacks, &mut |_| warnings += 1).unwrap_err();
    assert_eq!(r.key, "load.cache_dir");
    assert_eq!(warnings, 0);
}
