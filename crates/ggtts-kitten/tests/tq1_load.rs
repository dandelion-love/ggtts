//! A whole (tiny) Kitten-TTS 2 language model in each ternary layout, loaded by `Model::load` and run forward.
//!
//! The source is ternary under Kitten's rule with one scale shared by both 128-weight groups of every 256-weight
//! block, so that the older lossy `tq1_0-256-experimental` layout holds it exactly too and every layout carries the
//! same values. Bars, written before running:
//!
//! * every layout loads, including a `tq1_0-256-experimental` file written as the older tools wrote it;
//! * with the matrices dequantized ([`Weights::F32`]) every layout gives bit-identical logits: the readers decode the
//!   same values;
//! * through the packed kernels ([`Weights::Ternary`]), padded `TQ1_0` gives bit-identical logits to padded `TQ2_0`
//!   (same trits, same scales, same padded `Q8_K` activations), and every layout is finite and within rel L2 `5e-2`
//!   of the f32 arm.
//!
//! `GGTTS_DEVICE=cuda` (with the `cuda` feature) runs the same on CUDA, where the ternary matrices become exact f16.
#![cfg(feature = "ggml")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ggtts_kitten::backend::{Backend, Device};
use ggtts_kitten::convert;
use ggtts_kitten::gguf::{self, GgmlType, Gguf, TensorOut, Value};
use ggtts_kitten::model::{Model, Session, Weights};
use ggtts_kitten::safetensors::SafeTensors;
use ggtts_kitten::ternary::{tq1_decode, tq1_encode, Layout, Rule, TQ1_0_BLOCK_BYTES};

const H: usize = 256;
const FF: usize = 512;
const KV: usize = 128;
const VOCAB: usize = 8;
const SPK: usize = 64;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `rows` x `cols` ternary, one bf16- and fp16-exact scale per 256 weights, each 128-group's absmax equal to it.
fn ternary(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut w = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols / 256 {
        let s = (1 + (rng.next() * 15.0) as u32) as f32 / 256.0;
        for j in 0..256 {
            let u = rng.next();
            w.push(if j % 128 == 0 || u > 0.7 {
                s
            } else if u < 0.3 {
                -s
            } else {
                0.0
            });
        }
    }
    w
}

fn bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A complete one-layer model in Kitten's key layout, as safetensors bytes.
fn source() -> SafeTensors {
    let mut rng = Lcg(3);
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    let mut add = |name: &str, shape: &[usize], v: Vec<f32>| {
        let start = data.len();
        data.extend(bf16(&v));
        header.insert(
            name.into(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    };
    let p = |s: &str| format!("model.layers.0.{s}.weight");
    for (name, rows, cols) in [
        ("self_attn.q_proj", H, H),
        ("self_attn.k_proj", KV, H),
        ("self_attn.v_proj", KV, H),
        ("self_attn.o_proj", H, H),
        ("mlp.gate_proj", FF, H),
        ("mlp.up_proj", FF, H),
        ("mlp.down_proj", H, FF),
    ] {
        add(&p(name), &[rows, cols], ternary(rows, cols, &mut rng));
    }
    let around_one = |n: usize, rng: &mut Lcg| (0..n).map(|_| 0.75 + rng.next() as f32 * 0.5).collect::<Vec<_>>();
    add(&p("input_layernorm"), &[H], around_one(H, &mut rng));
    add(&p("post_attention_layernorm"), &[H], around_one(H, &mut rng));
    add(&p("self_attn.q_norm"), &[128], around_one(128, &mut rng));
    add(&p("self_attn.k_norm"), &[128], around_one(128, &mut rng));
    add("model.norm.weight", &[H], around_one(H, &mut rng));
    let embed: Vec<f32> = (0..VOCAB * H).map(|_| rng.next() as f32 - 0.5).collect();
    add(convert::EMBED, &[VOCAB, H], embed);
    let spk: Vec<f32> = (0..H * SPK).map(|_| (rng.next() as f32 - 0.5) * 0.1).collect();
    add("spk_proj.0.weight", &[H, SPK], spk);
    add("spk_proj.0.bias", &[H], vec![0.0; H]);
    add("spk_proj.1.weight", &[H], vec![1.0; H]);
    add("spk_proj.1.bias", &[H], vec![0.0; H]);
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    SafeTensors::from_bytes(out).unwrap()
}

fn configs() -> (serde_json::Value, serde_json::Value) {
    let model = serde_json::json!({
        "speaker_embedding_dim": SPK, "sample_rate": 24000, "default_preset": "stable",
        "decode_presets": {"stable": {"temperature": 0.8, "top_p": 0.8, "top_k": 50, "min_p": 0.0}},
        "generation": {"repetition_window": 50},
        "token_map": {"audio_id_base": 4, "num_audio_tokens": 2}
    });
    let lm = serde_json::json!({
        "num_hidden_layers": 1, "hidden_size": H, "intermediate_size": FF, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 128, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
        "rope_theta": 1e6, "tie_word_embeddings": true
    });
    (model, lm)
}

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-kitten-tq1-load-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Written by this converter, with the readback check passing.
fn converted(st: &SafeTensors, d: &Path, layout: Layout) -> PathBuf {
    let (model, lm) = configs();
    let out = d.join(format!("{}.gguf", layout.name()));
    convert::convert(st, &model, &lm, "tiny", &out, Rule::KITTEN, true, layout, None).unwrap();
    convert::verify(st, &Gguf::open(&out).unwrap()).unwrap();
    out
}

/// A `tq1_0-256-experimental` file as the older tools wrote it: each 256-weight block holds both 128-groups and
/// one scale. Built from the padded TQ1 file's blocks, so it carries the same trits and scales.
fn legacy(padded: &Path, d: &Path) -> PathBuf {
    let g = Gguf::open(padded).unwrap();
    let mut kv: BTreeMap<String, Value> = g.kv.clone();
    kv.insert(
        "kitten.ternary.layout".into(),
        Value::Str("tq1_0-256-experimental".into()),
    );
    let mut tensors = Vec::new();
    for name in &g.order {
        let (t, data) = g.tensor(name).unwrap();
        if t.ty != GgmlType::TQ1_0 {
            tensors.push(TensorOut {
                name: name.clone(),
                dims: t.dims.clone(),
                ty: t.ty,
                data: data.to_vec(),
            });
            continue;
        }
        let mut out = Vec::with_capacity(data.len() / 2);
        for pair in data.chunks_exact(2 * TQ1_0_BLOCK_BYTES) {
            let (a, b) = pair.split_at(TQ1_0_BLOCK_BYTES);
            assert_eq!(a[52..54], b[52..54], "the source shares a scale per 256 weights");
            let (ta, tb) = (tq1_decode(a).unwrap(), tq1_decode(b).unwrap());
            let mut t = [0i8; 256];
            t[..128].copy_from_slice(&ta[..128]);
            t[128..].copy_from_slice(&tb[..128]);
            out.extend_from_slice(&tq1_encode(&t));
            out.extend_from_slice(&a[52..54]);
        }
        tensors.push(TensorOut {
            name: name.clone(),
            dims: vec![t.dims[0] / 2, t.dims[1]],
            ty: GgmlType::TQ1_0,
            data: out,
        });
    }
    let out = d.join("legacy.gguf");
    gguf::write(&out, &kv, &tensors).unwrap();
    out
}

fn logits(path: &Path, weights: Weights, be: &std::rc::Rc<Backend>) -> Vec<f32> {
    let g = Gguf::open(path).unwrap();
    let m = Model::load(&g, weights, be.clone()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut rng = Lcg(99);
    let t = 6;
    let embd: Vec<f32> = (0..t * H).map(|_| (rng.next() as f32 - 0.5) * 2.0).collect();
    let mut s = Session::new(&m, 16).unwrap();
    let out = s.forward(&embd, t, false).unwrap().logits;
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{}: non-finite logits",
        path.display()
    );
    out
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        n += (f64::from(*x) - f64::from(*y)).powi(2);
        d += f64::from(*y).powi(2);
    }
    (n / d).sqrt()
}

#[test]
fn every_ternary_layout_loads_and_agrees() {
    let device = Device::parse(&std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into())).expect("GGTTS_DEVICE");
    let be = Backend::new(device, 2).expect("backend");
    let st = source();
    let d = dir();
    let q4 = converted(&st, &d, Layout::Q4);
    let tq2 = converted(&st, &d, Layout::Tq2Pad);
    let tq1 = converted(&st, &d, Layout::Tq1Pad);
    let old = legacy(&tq1, &d);
    assert_eq!(
        Gguf::open(&old).unwrap().str("kitten.ternary.layout").unwrap(),
        "tq1_0-256-experimental"
    );

    let f32_arm: Vec<Vec<f32>> = [&q4, &tq2, &tq1, &old]
        .iter()
        .map(|p| logits(p, Weights::F32, &be))
        .collect();
    for (i, y) in f32_arm.iter().enumerate().skip(1) {
        assert!(bits_equal(y, &f32_arm[0]), "f32 arm: layout {i} differs from Q4_0");
    }

    let tern: Vec<Vec<f32>> = [&q4, &tq2, &tq1, &old]
        .iter()
        .map(|p| logits(p, Weights::Ternary, &be))
        .collect();
    assert!(
        bits_equal(&tern[2], &tern[1]),
        "{device:?}: padded TQ1_0 and padded TQ2_0 differ, rel L2 {:e}",
        rel_l2(&tern[2], &tern[1])
    );
    for (i, y) in tern.iter().enumerate() {
        let e = rel_l2(y, &f32_arm[0]);
        eprintln!("{device:?} ternary arm, layout {i}: rel L2 {e:.3e} against the f32 arm");
        assert!(e < 5e-2, "layout {i}: rel L2 {e}");
    }
    std::fs::remove_dir_all(&d).unwrap();
}
