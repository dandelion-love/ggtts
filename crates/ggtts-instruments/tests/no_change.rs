//! The ordinary CI proof uses the fake engine only. It does not establish real-model or GPU identity.
//! Event instruments run after rendering; this checks the real runner, its full catalogue, and the artifacts it
//! handles. The planted sample mutation is a test double for a bad observer, not a real in-process tensor hook.

use std::collections::HashMap;
use std::path::Path;

use ggtts_core::engine::{ChunkRequest, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::flac::write_flac;
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{plan_story, render_story_traced, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::seed::sha256;
use ggtts_core::trace::{emit_header, Events, Header, NullEvents, Recorder};
use ggtts_fake::FakeEngine;
use ggtts_instruments::{catalogue::CATALOGUE, Config, Inputs};

struct Capture<E> {
    engine: E,
    tokens: Vec<u8>,
    plant_sample: bool,
}

impl<E: Engine> Engine for Capture<E> {
    fn p_stop_semantics(&self) -> Option<&'static str> {
        self.engine.p_stop_semantics()
    }
    fn info(&self) -> &EngineInfo {
        self.engine.info()
    }
    fn profile(&self) -> &Profile {
        self.engine.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.engine.generate(req)
    }
    fn generate_hooked(
        &mut self,
        req: &ChunkRequest<'_>,
        events: &mut dyn Events,
        hooks: &mut Hooks,
    ) -> Result<Generated, EngineError> {
        let result = self.engine.generate_hooked(req, events, hooks)?;
        self.tokens.extend_from_slice(&result.codes.codebooks.to_le_bytes());
        self.tokens
            .extend_from_slice(&(result.codes.frames.len() as u64).to_le_bytes());
        for token in &result.codes.frames {
            self.tokens.extend_from_slice(&token.to_le_bytes());
        }
        Ok(result)
    }
    fn decode(&mut self, codes: &ggtts_core::engine::Codes) -> Result<Pcm, EngineError> {
        let mut pcm = self.engine.decode(codes)?;
        if self.plant_sample && !pcm.samples.is_empty() {
            pcm.samples[0] += 0.25;
        }
        Ok(pcm)
    }
}

fn fake() -> FakeEngine {
    FakeEngine::new(HashMap::new(), HashMap::new())
}

fn render<E: Engine>(engine: E, events: &mut dyn Events, plant_sample: bool) -> (Vec<u8>, Vec<f32>, u32) {
    let mut capture = Capture {
        engine,
        tokens: Vec::new(),
        plant_sample,
    };
    let profile = *capture.profile();
    let paragraphs = vec!["你好，今日天氣好好。".to_string()];
    let chunks = plan_story(&profile, &paragraphs);
    let mut opts = RenderOptions::for_profile(&profile);
    opts.redraw = 0;
    let level = events.level();
    emit_header(events, &Header::for_engine(&capture, level));
    let read = render_story_traced(&mut capture, "no-change-fixture", &chunks, &opts, events).unwrap();
    (capture.tokens, read.samples, read.sample_rate)
}

fn sample_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

#[test]
fn fake_engine_catalogue_preserves_tokens_and_samples_and_sees_a_sample_plant() {
    let root = std::env::temp_dir().join(format!("ggtts-no-change-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let (base_tokens, base_samples, rate) = render(fake(), &mut NullEvents, false);
    assert!(!base_tokens.is_empty() && !base_samples.is_empty());
    let mut trace = Recorder::new(1);
    let (tokens, samples, traced_rate) = render(fake(), &mut trace, false);
    assert_eq!(rate, traced_rate);
    assert_eq!(
        sha256(&base_tokens),
        sha256(&tokens),
        "tracing changed full codec tokens"
    );
    assert_eq!(
        sha256(&sample_bytes(&base_samples)),
        sha256(&sample_bytes(&samples)),
        "tracing changed samples"
    );

    let trace_path = root.join("events.jsonl");
    let lines: Vec<String> = trace
        .lines
        .iter()
        .map(|line| serde_json::Value::Object(line.clone()).to_string())
        .collect();
    std::fs::write(&trace_path, lines.join("\n") + "\n").unwrap();
    let audio = write_flac(&root.join("audio"), &samples, rate).unwrap();
    let before = sha256(&std::fs::read(&audio).unwrap());
    let mut config = Config::default();
    for item in CATALOGUE {
        let flag = if item.name == "render-timing" {
            "render-timing,arg.stage_only=true"
        } else {
            item.name
        };
        config.apply_instrument_flag(flag).unwrap();
        config.require.push(item.name.to_string());
    }
    let inputs = Inputs {
        run: "no-change".into(),
        render: "fixture".into(),
        report_dir: root.join("report"),
        audio: audio.clone(),
        audio_state: ggtts_core::report::AudioState::Trimmed,
        audio_seconds: Some(samples.len() as f64 / f64::from(rate)),
        events: Some(trace_path),
        bin_dir: Path::new(env!("CARGO_BIN_EXE_ggtts-instrument-selftest"))
            .parent()
            .map(Path::to_path_buf),
        instruments_dir: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
    };
    let result = ggtts_instruments::run(&config, &inputs).unwrap();
    assert!(
        result.complete(),
        "instruments did not all report OK: {:?}",
        result.incomplete
    );
    assert_eq!(result.envelopes.len(), CATALOGUE.len());
    let schema_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-report-2.schema.json");
    let schema: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(schema_path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).expect("the published report schema compiles");
    for envelope in &result.envelopes {
        assert!(
            validator.is_valid(&envelope.to_json()),
            "{} produced an envelope outside the published schema",
            envelope.name
        );
    }
    let mut invalid = result.envelopes[0].to_json();
    invalid["status"] = serde_json::json!("planted-invalid");
    assert!(!validator.is_valid(&invalid), "schema must reject an invalid status");
    assert_eq!(
        before,
        sha256(&std::fs::read(audio).unwrap()),
        "an instrument changed the audio"
    );

    let (_, planted_samples, _) = render(fake(), &mut Recorder::new(1), true);
    assert_ne!(
        sha256(&sample_bytes(&base_samples)),
        sha256(&sample_bytes(&planted_samples)),
        "the sample-mutation plant must make the identity check red"
    );
    println!(
        "fake backend; seed derived from no-change-fixture; {} event instruments; sample plant detected; \
              does not establish real-model/GPU identity, timing, or external-artifact safety",
        CATALOGUE.len()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn promotion_gate_and_real_probe_controls_are_tested_without_a_model() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for pattern in [
        "test_qwen_event_instrument_probe.py",
        "test_instrument_promotion_gate.py",
        "test_rc_gate.py",
    ] {
        let result = std::process::Command::new(ggtts_instruments::runner::default_python())
            .current_dir(&root)
            .args(["-m", "unittest", "discover", "-s", "scripts", "-p", pattern])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{pattern}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
