//! Explicit, allowlisted export of a private merged instrument report.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use ggtts_core::report::{validate, Status, FORMAT};
use serde_json::{json, Value};

// Public constants from the shipped instrument implementations. Arbitrary
// third-party names and units are not safe to copy into a shareable report.
const METRICS: &[(&str, &str)] = &[
    ("audio_bytes", "bytes"),
    ("events", "lines"),
    ("p_stop_early_max", "1"),
    ("early_stop_frames", "1"),
    ("draws", "1"),
    ("redraws", "1"),
    ("cap_stops", "1"),
    ("flagged_draws", "1"),
    ("chunk_draws", "1"),
    ("lead_s_max", "s"),
    ("tail_s_max", "s"),
    ("trim_s_total", "s"),
    ("edges_over_limit", "1"),
    ("seconds", "s"),
    ("frames", "1"),
    ("chunks", "1"),
    ("gain_db", "dB"),
    ("peak_dbfs", "dBFS"),
    ("rms_dbfs", "dBFS"),
    ("clipped_samples", "1"),
    ("internal_pauses", "1"),
    ("longest_pause_s", "s"),
    ("pause_s_total", "s"),
    ("pitch_voiced_frames", "1"),
    ("pitch_median_hz", "Hz"),
    ("pitch_range_st", "semitones"),
    ("voiced_frames", "1"),
    ("loud_p90_db", "dB"),
    ("soft_p10_db", "dB"),
    ("character_error_rate", "1"),
    ("ttfa_ms", "ms"),
    ("render_ms", "ms"),
    ("render_rtf", "1"),
    ("generate_ms_total", "ms"),
    ("decode_ms_total", "ms"),
    ("judge_ms_total", "ms"),
];

fn fields(v: &Value, required: &[&str], optional: &[&str]) -> Result<(), ()> {
    let m = v.as_object().ok_or(())?;
    if required.iter().any(|k| !m.contains_key(*k))
        || m.keys()
            .any(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(());
    }
    Ok(())
}

fn export(v: &Value) -> Result<Value, ()> {
    fields(
        v,
        &["format", "run", "measurement", "summary", "completeness", "instruments"],
        &[],
    )?;
    if v["format"] != FORMAT || v["run"].as_str().filter(|s| !s.is_empty()).is_none() {
        return Err(());
    }
    let entries = v["instruments"].as_array().ok_or(())?;
    let reqs = v["completeness"].as_array().ok_or(())?;
    let mut counts: BTreeMap<&str, u64> = Status::ALL.into_iter().map(|s| (s.as_str(), 0)).collect();
    let mut names = BTreeMap::<String, usize>::new();
    let mut records = Vec::new();
    let mut identities: Vec<(String, Status)> = Vec::new();
    for entry in entries {
        let mut raw = entry.clone();
        let object = raw.as_object_mut().ok_or(())?;
        if object
            .remove("source")
            .and_then(|v| v.as_str().map(str::to_owned))
            .is_none()
        {
            return Err(());
        }
        let invalid = object.remove("invalid");
        let env = validate(&raw).map_err(|_| ())?;
        if env.identity.run != v["run"]
            || identities.iter().any(|(id, status)| {
                id == &env.identity.instrument && (*status != Status::Invalid || env.status != Status::Invalid)
            })
        {
            return Err(());
        }
        identities.push((env.identity.instrument.clone(), env.status));
        if invalid.is_some() && env.status != Status::Invalid {
            return Err(());
        }
        if let Some(x) = invalid {
            if x.as_array()
                .filter(|a| !a.is_empty() && a.iter().all(Value::is_string))
                .is_none()
            {
                return Err(());
            }
        }
        *counts.get_mut(env.status.as_str()).ok_or(())? += 1;
        let next = names.len() + 1;
        let id = *names.entry(env.name.clone()).or_insert(next);
        let mut metrics = Vec::new();
        for m in &env.metrics {
            if !METRICS.contains(&(m.name.as_str(), m.unit.as_str())) {
                return Err(());
            }
            metrics.push(json!({"name":m.name,"unit":m.unit,"value":m.value,
                "direction":m.direction.as_str(),"interval":m.interval,"chunk":m.chunk,
                "render_scoped":m.render.is_some()}));
        }
        records.push(
            json!({"instrument":id,"status":env.status.as_str(),"audio":env.provenance.audio.as_str(),
            "model_sha256":env.provenance.model_sha256,
            "input_sha256":env.provenance.inputs.iter().map(|i| &i.sha256).collect::<Vec<_>>(),
            "artifact_sha256":env.artifacts.iter().map(|i| &i.sha256).collect::<Vec<_>>(),"metrics":metrics}),
        );
    }
    let summary = v["summary"].as_object().ok_or(())?;
    if summary.len() != counts.len()
        || counts
            .iter()
            .any(|(k, n)| summary.get(*k).and_then(Value::as_u64) != Some(*n))
    {
        return Err(());
    }
    let mut required = Vec::new();
    let mut all_met = true;
    for req in reqs {
        fields(req, &["name", "present", "valid", "statuses", "met"], &[])?;
        let name = req["name"].as_str().filter(|s| !s.is_empty()).ok_or(())?;
        let found: Vec<_> = entries.iter().filter(|e| e["name"] == name).collect();
        let statuses: Vec<_> = found.iter().map(|e| e["status"].as_str().unwrap_or("")).collect();
        let valid = found.iter().all(|e| e.get("invalid").is_none());
        let present = !found.is_empty();
        let met = present && valid && statuses.iter().all(|s| *s == "ok");
        if req["present"] != present || req["valid"] != valid || req["met"] != met || req["statuses"] != json!(statuses)
        {
            return Err(());
        }
        let next = names.len() + 1;
        let id = *names.entry(name.to_owned()).or_insert(next);
        required.push(json!({"instrument":id,"present":present,"valid":valid,"statuses":statuses,"met":met}));
        all_met &= met;
    }
    if v["measurement"] != if all_met { "complete" } else { "incomplete" } {
        return Err(());
    }
    Ok(
        json!({"format":"ggtts-shareable-report/1","measurement":v["measurement"],
        "summary":counts,"required":required,"instruments":records,
        "omitted":["identifiers","paths","configuration","logs","error details","unreviewed provenance"]}),
    )
}

fn run() -> Result<(), ()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(());
    }
    let input = std::fs::read(Path::new(&args[0])).map_err(|_| ())?;
    let value: Value = serde_json::from_slice(&input).map_err(|_| ())?;
    let out = export(&value)?;
    let encoded = serde_json::to_vec_pretty(&out).map_err(|_| ())?;
    publish(Path::new(&args[1]), &encoded)
}

// Publish only after the full report is written. The hard link is an atomic,
// no-clobber operation in the destination directory; unsupported filesystems
// refuse rather than falling back to a potentially partial destination.
fn publish(destination: &Path, encoded: &[u8]) -> Result<(), ()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| ())?.as_nanos();
    let temporary = parent.join(format!(".ggtts-export-{}-{stamp}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| ())?;
    let result = file.write_all(encoded).and_then(|_| file.sync_all());
    drop(file);
    let result = result.and_then(|_| std::fs::hard_link(&temporary, destination));
    let _ = std::fs::remove_file(&temporary);
    result.map_err(|_| ())
}

fn main() {
    if run().is_err() {
        eprintln!("Export refused: invalid, unsupported, or inaccessible report or destination.");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggtts_core::report::{merge, Input};

    fn report(status: &str, required: bool) -> Value {
        let e = json!({"format":FORMAT,"name":"sensitive-name","version":"secret-version",
            "identity":{"run":"private-run","instrument":"private-id"},
            "provenance":{"config":{"secret":"private-transcript"},"ggtts":null,"engine":null,
                "backend":null,"model_sha256":"a".repeat(64),"seeds":[],
                "inputs":[{"path":"private-path","sha256":"b".repeat(64)}],"audio":"trimmed"},
            "hooks":["private-hook"],"status":status,
            "metrics":[{"name":"character_error_rate","unit":"1","value":0.25,"direction":"neutral","render":"private-render"}],
            "artifacts":[{"path":"private-artifact","type":"private-type","sha256":"c".repeat(64)}],
            "log":"private-log","error":"private-error"});
        let i = Input {
            source: "private-source".into(),
            text: e.to_string(),
        };
        merge("private-run", &[i], if required { &["sensitive-name"] } else { &[] }).to_json()
    }

    #[test]
    fn decoder_provenance_does_not_export_configuration_or_paths() {
        let mut input = report("ok", true);
        let provenance = &mut input["instruments"][0]["provenance"];
        provenance["config"]["ggtts_audio_decoders"] =
            json!([{"kind":"libsndfile","sha256":"d".repeat(64),"private":"private-decoder-path"}]);
        provenance["inputs"].as_array_mut().unwrap().push(json!({
            "path":"decoder:libsndfile", "sha256":"d".repeat(64)
        }));
        let output = export(&input).unwrap();
        let serialized = output.to_string();
        assert!(!serialized.contains("private-decoder-path"));
        assert!(!serialized.contains("decoder:libsndfile"));
        assert!(serialized.contains(&"d".repeat(64)));
    }

    #[test]
    fn removes_planted_strings_for_success_and_failure() {
        for status in ["ok", "failed", "timeout", "skipped", "truncated"] {
            let input = report(status, true);
            let before = input.to_string();
            let out = export(&input).unwrap();
            let text = out.to_string();
            for secret in ["private", "sensitive-name", "secret-version"] {
                assert!(!text.contains(secret));
            }
            assert_eq!(
                out["measurement"],
                if status == "ok" { "complete" } else { "incomplete" }
            );
            assert_eq!(out["instruments"][0]["metrics"][0]["value"], 0.25);
            assert_eq!(before, input.to_string());
        }
    }

    #[test]
    fn rejects_unknown_fields_metrics_and_false_completeness() {
        let input = report("failed", true);
        let mut x = input.clone();
        x["instruments"][0]["private-extra"] = json!("secret");
        assert!(export(&x).is_err());
        let mut x = input.clone();
        x["instruments"][0]["metrics"][0]["unit"] = json!("private-unit");
        assert!(export(&x).is_err());
        let mut x = input.clone();
        x["measurement"] = json!("complete");
        assert!(export(&x).is_err());
        let mut x = input.clone();
        x["summary"]["failed"] = json!(0);
        assert!(export(&x).is_err());
        let mut x = input;
        x["instruments"][0]["status"] = json!("mystery");
        assert!(export(&x).is_err());
    }

    #[test]
    fn refuses_missing_unknown_and_inconsistent_fields() {
        let input = report("ok", true);
        for key in ["run", "instruments", "summary", "completeness", "measurement"] {
            let mut x = input.clone();
            x.as_object_mut().unwrap().remove(key);
            assert!(export(&x).is_err());
        }
        let mut x = input.clone();
        x["unexpected"] = json!("private");
        assert!(export(&x).is_err());
        let mut x = input.clone();
        x["instruments"][0]["provenance"]["extra"] = json!("private");
        assert!(export(&x).is_err());
        let mut x = input.clone();
        x["completeness"][0]["met"] = json!(false);
        assert!(export(&x).is_err());
        let mut x = input;
        x["instruments"][0]["identity"]["run"] = json!("other");
        assert!(export(&x).is_err());
    }

    #[test]
    fn publication_never_clobbers_and_leaves_no_temporary_file_on_error() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let dir = std::env::temp_dir().join(format!(
            "ggtts-export-publish-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let destination = dir.join("report.json");
        std::fs::write(&destination, "original").unwrap();
        assert!(publish(&destination, b"replacement").is_err());
        assert_eq!(std::fs::read_to_string(&destination).unwrap(), "original");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        assert!(publish(&dir.join("missing").join("report.json"), b"data").is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn optional_failure_and_missing_required_remain_distinct() {
        let optional = export(&report("failed", false)).unwrap();
        assert_eq!(optional["measurement"], "complete");
        let input = merge("private-run", &[], &["missing-private-name"]).to_json();
        let missing = export(&input).unwrap();
        assert_eq!(missing["measurement"], "incomplete");
        assert_eq!(missing["required"][0]["present"], false);
        assert!(!missing.to_string().contains("missing-private-name"));
    }

    #[test]
    fn preserves_sibling_order() {
        let mut input = report("timeout", false);
        let mut sibling = input["instruments"][0].clone();
        sibling["name"] = json!("second-private");
        sibling["identity"]["instrument"] = json!("second-private-id");
        sibling["status"] = json!("failed");
        input["instruments"].as_array_mut().unwrap().push(sibling);
        input["summary"]["failed"] = json!(1);
        let out = export(&input).unwrap();
        assert_eq!(out["instruments"][0]["status"], "timeout");
        assert_eq!(out["instruments"][1]["status"], "failed");
        assert_eq!(out["instruments"][0]["instrument"], 1);
        assert_eq!(out["instruments"][1]["instrument"], 2);
        assert!(!out.to_string().contains("private"));
    }

    #[test]
    fn all_shipped_metric_pairs_export_and_match_schema() {
        let mut input = report("ok", false);
        let metrics: Vec<Value> = METRICS
            .iter()
            .map(|(name, unit)| json!({"name":name,"unit":unit,"value":1.0,"direction":"neutral"}))
            .collect();
        input["instruments"][0]["metrics"] = json!(metrics);
        let mut sibling = input["instruments"][0].clone();
        sibling["name"] = json!("another-instrument");
        sibling["identity"]["instrument"] = json!("another-id");
        sibling["metrics"] = json!([{"name":"character_error_rate","unit":"1","value":0.2,"direction":"neutral"}]);
        input["instruments"].as_array_mut().unwrap().push(sibling);
        input["summary"]["ok"] = json!(2);
        let out = export(&input).unwrap();
        assert_eq!(
            out["instruments"][0]["metrics"].as_array().unwrap().len(),
            METRICS.len()
        );
        assert_eq!(out["instruments"][1]["metrics"][0]["value"], 0.2);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-shareable-report-1.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&out));
        let mut unreviewed = out;
        unreviewed["instruments"][0]["metrics"][0]["name"] = json!("private-name");
        assert!(!validator.is_valid(&unreviewed));
    }

    #[test]
    fn merger_invalid_and_instrument_invalid_stay_incomplete() {
        let broken = Input {
            source: "private-path".into(),
            text: "not json private".into(),
        };
        let merged = merge("private-run", &[broken], &["private-path"]);
        let out = export(&merged.to_json()).unwrap();
        assert_eq!(out["instruments"][0]["status"], "invalid");
        assert_eq!(out["measurement"], "incomplete");
        assert!(!out.to_string().contains("private"));
        let mut own = report("failed", true);
        own["instruments"][0]["status"] = json!("invalid");
        own["summary"]["failed"] = json!(0);
        own["summary"]["invalid"] = json!(1);
        own["completeness"][0]["statuses"] = json!(["invalid"]);
        assert_eq!(export(&own).unwrap()["measurement"], "incomplete");
    }
}
