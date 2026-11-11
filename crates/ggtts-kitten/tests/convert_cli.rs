//! `ggtts-kitten-convert` end to end, as a user runs it from a release archive: a synthetic Kitten-TTS 2 download
//! (one layer, hidden 256, no real weights) on disk, converted by the built binary, read back, and its tokenizer
//! found embedded where `ggtts-render` looks for it. Also `--help` and `--version`, which the release smoke runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ggtts_kitten::gguf::{Gguf, Value};

const BIN: &str = env!("CARGO_BIN_EXE_ggtts-kitten-convert");
const TOKENIZER: &str = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#;

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-kitten-convert-cli-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("lm")).unwrap();
    d
}

/// A ternary row in Kitten's rule: groups of 128, each one of {-s, 0, +s}, with s the group's absmax.
fn row(n: usize, seed: &mut u64) -> Vec<f32> {
    let mut next = || {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n / 128 {
        let s = f32::from_bits((0x3c00 + next() % 0x300) << 16);
        for j in 0..128 {
            out.push(match (j, next() % 3) {
                (0, _) | (_, 0) => s,
                (_, 1) => -s,
                _ => 0.0,
            });
        }
    }
    out
}

/// A one-layer model in Kitten's key layout, written where `--repo` expects it.
fn write_repo(d: &Path, with_tokenizer: bool) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    let mut add = |name: &str, shape: &[usize], v: &[f32]| {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()));
        header.insert(
            name.into(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    };
    let mut seed = 7u64;
    for t in ["q", "k", "v", "o"] {
        let w: Vec<f32> = (0..256).flat_map(|_| row(256, &mut seed)).collect();
        add(&format!("model.layers.0.self_attn.{t}_proj.weight"), &[256, 256], &w);
    }
    add("model.layers.0.input_layernorm.weight", &[256], &[1.5; 256]);
    add("model.embed_tokens.weight", &[8, 256], &[0.25; 8 * 256]);
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut st = (h.len() as u64).to_le_bytes().to_vec();
    st.extend(h);
    st.extend(data);
    std::fs::write(d.join("lm/model.safetensors"), st).unwrap();
    let model = serde_json::json!({
        "speaker_embedding_dim": 512, "sample_rate": 24000, "default_preset": "stable",
        "decode_presets": {"stable": {"temperature": 0.8, "top_p": 0.8, "top_k": 50, "min_p": 0.0}},
        "generation": {"repetition_window": 50},
        "token_map": {"audio_id_base": 4, "num_audio_tokens": 2}
    });
    let lm = serde_json::json!({
        "num_hidden_layers": 1, "hidden_size": 256, "intermediate_size": 256, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 128, "vocab_size": 8, "rms_norm_eps": 1e-6,
        "rope_theta": 1e6, "tie_word_embeddings": true
    });
    std::fs::write(d.join("config.json"), model.to_string()).unwrap();
    std::fs::write(d.join("lm/config.json"), lm.to_string()).unwrap();
    if with_tokenizer {
        std::fs::write(d.join("lm/tokenizer.json"), TOKENIZER).unwrap();
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn converts_a_download_and_embeds_its_tokenizer() {
    let d = dir("ok");
    write_repo(&d, true);
    let out = d.join("kitten-lm.gguf");
    let o = run(&["--repo", d.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert!(o.status.success(), "{}{}", text(&o.stdout), text(&o.stderr));
    assert!(text(&o.stdout).contains("READBACK PASS"), "{}", text(&o.stdout));
    let g = Gguf::open(&out).unwrap();
    assert_eq!(g.str("general.architecture").unwrap(), "kitten-tts-2");
    match g.kv.get("kitten.tokenizer.json") {
        Some(Value::Str(s)) => assert_eq!(s, TOKENIZER),
        other => panic!("no embedded tokenizer: {other:?}"),
    }
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_download_without_its_tokenizer_is_refused_before_writing() {
    let d = dir("no-tokenizer");
    write_repo(&d, false);
    let out = d.join("kitten-lm.gguf");
    let o = run(&["--repo", d.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    let e = text(&o.stderr);
    assert!(
        e.contains("tokenizer.json") && e.contains("ggtts-render refuses"),
        "{e}"
    );
    assert!(!out.exists(), "a GGUF without its tokenizer was written");
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn help_succeeds_and_names_both_conversions() {
    for flag in ["--help", "-h"] {
        let o = run(&[flag]);
        assert!(o.status.success(), "{flag}: {}", text(&o.stderr));
        let s = text(&o.stdout);
        assert!(
            s.contains("--repo") && s.contains("--s3gen") && s.contains("KittenML/kitten-tts-2"),
            "{s}"
        );
    }
}

#[test]
fn version_names_the_tag_the_ggtts_commit_and_the_ggml_pin() {
    let o = run(&["--version"]);
    assert!(o.status.success());
    let lines: Vec<String> = text(&o.stdout).lines().map(str::to_string).collect();
    assert_eq!(lines, ggtts_core::version_lines("ggtts-kitten-convert").to_vec());
    assert!(lines[1].contains(ggtts_core::SOURCE_COMMIT), "{lines:?}");
}

/// The renderer's refusal of a GGUF converted before the tokenizer was embedded names the shipped converter and
/// the download it reads.
#[cfg(feature = "ggml")]
#[test]
fn a_gguf_without_a_tokenizer_is_refused_naming_the_shipped_converter() {
    let d = dir("old-gguf");
    write_repo(&d, false);
    let st = ggtts_kitten::safetensors::SafeTensors::open(&d.join("lm/model.safetensors")).unwrap();
    let read = |p: &str| -> serde_json::Value { serde_json::from_slice(&std::fs::read(d.join(p)).unwrap()).unwrap() };
    let lm = d.join("old.gguf");
    ggtts_kitten::convert::convert(
        &st,
        &read("config.json"),
        &read("lm/config.json"),
        "tiny",
        &lm,
        ggtts_kitten::ternary::Rule::KITTEN,
        true,
        ggtts_kitten::ternary::Layout::Q4,
        None,
    )
    .unwrap();
    let e = ggtts_kitten::engine::KittenEngine::open(
        &lm,
        &d.join("absent-s3gen.gguf"),
        &d.join("absent-voice.json"),
        1,
        ggtts_kitten::backend::Device::Cpu,
    )
    .err()
    .expect("a GGUF without a tokenizer opened");
    assert_eq!(e, ggtts_kitten::engine::NO_TOKENIZER);
    assert!(
        e.contains("ggtts-kitten-convert --repo") && e.contains("KittenML/kitten-tts-2"),
        "{e}"
    );
    std::fs::remove_dir_all(&d).unwrap();
}
