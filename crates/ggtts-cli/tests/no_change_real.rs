//! Local real-Qwen no-change proof. Ignored on hosted CI: it needs the published GGUFs and optional CUDA.
//! Event instruments run after generation. This compares full generated token vectors and decoded samples with
//! tracing off/on, then runs each catalogue instrument separately against the traced render and checks its audio.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};
use std::process::Command;

use ggtts_core::engine::{ChunkRequest, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::flac::write_flac;
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{plan_story, render_story_traced, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::seed::sha256;
use ggtts_core::trace::{emit_header, Events, Header, NullEvents, Recorder};
use ggtts_instruments::{catalogue::CATALOGUE, Config, Inputs};
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};

const TALKER_SHA: &str = "33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67";
const CODEC_SHA: &str = "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824";

struct Capture<E> {
    engine: E,
    tokens: Vec<u8>,
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
        self.tokens.extend_from_slice(&req.seed.to_le_bytes());
        self.tokens.extend_from_slice(&result.codes.codebooks.to_le_bytes());
        self.tokens
            .extend_from_slice(&(result.codes.frames.len() as u64).to_le_bytes());
        for token in &result.codes.frames {
            self.tokens.extend_from_slice(&token.to_le_bytes());
        }
        Ok(result)
    }
    fn decode(&mut self, codes: &ggtts_core::engine::Codes) -> Result<Pcm, EngineError> {
        self.engine.decode(codes)
    }
}

fn pinned(var: &str, want: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("set {var}")));
    // Python streams the multi-GB model; using the core's in-memory hash here would duplicate it in RAM.
    let code = "import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],'rb');\nwhile True:\n b=f.read(1048576)\n if not b: break\n h.update(b)\nprint(h.hexdigest())";
    let out = Command::new(ggtts_instruments::runner::default_python())
        .args(["-c", code])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "hash {var}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        want,
        "{var} is not the published v0.0.1 file"
    );
    path
}

fn render(engine: Qwen3Tts, events: &mut dyn Events) -> (Vec<u8>, Vec<f32>, u32) {
    let mut capture = Capture {
        engine,
        tokens: Vec::new(),
    };
    let profile = *capture.profile();
    let paragraphs = vec!["小朋友喺海邊砌咗一座好大嘅沙堡。".to_string()];
    let chunks = plan_story(&profile, &paragraphs).unwrap();
    let mut opts = RenderOptions::for_profile(&profile);
    opts.redraw = 0;
    let level = events.level();
    emit_header(events, &Header::for_engine(&capture, level));
    let read = render_story_traced(&mut capture, "no-change-fixture", &chunks, &opts, events).unwrap();
    (capture.tokens, read.samples, read.sample_rate)
}

fn samples_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn hex(bytes: &[u8]) -> String {
    sha256(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "requires the pinned Qwen GGUFs and prebuilt selftest; GGTTS_DEVICE=cpu or cuda"]
fn each_event_instrument_preserves_real_qwen_tokens_and_samples() {
    let talker = pinned("GGTTS_TALKER", TALKER_SHA);
    let codec = pinned("GGTTS_CODEC", CODEC_SHA);
    let backend = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = Device::parse(&backend).expect("GGTTS_DEVICE must be cpu or cuda");
    let load = || {
        Qwen3Tts::load(
            &talker,
            &codec,
            4,
            Voice {
                device,
                ..Voice::default()
            },
        )
        .unwrap()
    };
    let (baseline_tokens, baseline_samples, rate) = render(load(), &mut NullEvents);
    let mut trace = Recorder::new(1);
    let (tokens, samples, traced_rate) = render(load(), &mut trace);
    assert!(
        !tokens.is_empty() && !samples.is_empty(),
        "empty output cannot prove identity"
    );
    assert_eq!(rate, traced_rate);
    assert_eq!(
        sha256(&baseline_tokens),
        sha256(&tokens),
        "tracing changed full codec tokens"
    );
    assert_eq!(
        sha256(&samples_bytes(&baseline_samples)),
        sha256(&samples_bytes(&samples)),
        "tracing changed samples"
    );
    let root = std::env::temp_dir().join(format!("ggtts-real-each-{}-{backend}", std::process::id()));
    std::fs::create_dir(&root).expect("fresh scratch directory");
    let events = root.join("events.jsonl");
    assert_eq!(trace.lines[0]["flags"]["p_stop_semantics"], "qwen-sampler-weights/1");
    let lines: Vec<String> = trace
        .lines
        .iter()
        .map(|line| serde_json::Value::Object(line.clone()).to_string())
        .collect();
    let seeds: Vec<u64> = trace
        .lines
        .iter()
        .filter(|line| line.get("ev").and_then(|v| v.as_str()) == Some("chunk_begin"))
        .filter_map(|line| line.get("seed").and_then(|v| v.as_u64()))
        .collect();
    assert!(!seeds.is_empty(), "a proof without recorded seeds is incomplete");
    std::fs::write(&events, lines.join("\n") + "\n").unwrap();
    let audio = write_flac(&root.join("audio"), &samples, rate).unwrap();
    let audio_sha = sha256(&std::fs::read(&audio).unwrap());
    for item in CATALOGUE {
        let mut config = Config::default();
        let flag = if item.name == "render-timing" {
            "render-timing,arg.stage_only=true"
        } else {
            item.name
        };
        config.apply_instrument_flag(flag).unwrap();
        config.require.push(item.name.to_string());
        let inputs = Inputs {
            run: "real-no-change".into(),
            render: "fixture".into(),
            report_dir: root.join(item.name),
            audio: audio.clone(),
            audio_state: ggtts_core::report::AudioState::Trimmed,
            audio_seconds: Some(samples.len() as f64 / f64::from(rate)),
            events: Some(events.clone()),
            bin_dir: Path::new(env!("CARGO_BIN_EXE_ggtts-render"))
                .parent()
                .map(Path::to_path_buf),
            instruments_dir: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
        };
        let summary = ggtts_instruments::run(&config, &inputs).unwrap();
        assert!(summary.complete(), "{}: {:?}", item.name, summary.incomplete);
        assert_eq!(
            sha256(&std::fs::read(&audio).unwrap()),
            audio_sha,
            "{} changed audio",
            item.name
        );
        println!("instrument={} status=ok audio_unchanged=true", item.name);
    }
    println!(
        "backend={backend}; build={}; story=no-change-fixture; seeds={seeds:?}; tokens_sha256={}; \
              samples_sha256={}; limits: event instruments run after generation; no timing or external-artifact \
              identity; no cross-backend parity",
        env!("CARGO_PKG_VERSION"),
        hex(&tokens),
        hex(&samples_bytes(&samples))
    );
    std::fs::remove_dir_all(root).unwrap();
}
