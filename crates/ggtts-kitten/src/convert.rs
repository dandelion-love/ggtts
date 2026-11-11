//! Kitten-TTS 2's language model (or a QAT export in the same key layout) to a ggtts GGUF, and the readback check.
//!
//! Tensor names are kept as Kitten writes them. The 196 ternary linears become padded `TQ2_0` (see
//! [`crate::ternary`]); the token embedding, which is also the tied output head, stays bf16 byte for byte; every
//! other tensor (the norms and `spk_proj`) is widened to f32, which is exact.

use std::collections::BTreeMap;
use std::path::Path;

use crate::gguf::{self, GgmlType, Gguf, TensorOut, Value};
use crate::safetensors::{Dtype, SafeTensors};
use crate::ternary::{Layout, Rule, GROUP};

pub const ARCH: &str = "kitten-tts-2";
pub const EMBED: &str = "model.embed_tokens.weight";

/// True for the 196 ternary-born linears: `q/k/v/o_proj` and `gate/up/down_proj` of every layer.
pub fn is_ternary(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((layer, tail)) = rest.split_once('.') else {
        return false;
    };
    layer.bytes().all(|b| b.is_ascii_digit())
        && matches!(
            tail,
            "self_attn.q_proj.weight"
                | "self_attn.k_proj.weight"
                | "self_attn.v_proj.weight"
                | "self_attn.o_proj.weight"
                | "mlp.gate_proj.weight"
                | "mlp.up_proj.weight"
                | "mlp.down_proj.weight"
        )
}

fn get_u32(v: &serde_json::Value, key: &str) -> Result<u32, String> {
    v[key]
        .as_u64()
        .map(|x| x as u32)
        .ok_or_else(|| format!("config: {key} missing"))
}

fn get_f32(v: &serde_json::Value, key: &str) -> Result<f32, String> {
    v[key]
        .as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| format!("config: {key} missing"))
}

/// The GGUF metadata, from the model's `config.json` and `lm/config.json`.
pub fn metadata(
    model: &serde_json::Value,
    lm: &serde_json::Value,
    source: &str,
    layout: Layout,
    tokenizer_json: Option<&str>,
) -> Result<BTreeMap<String, Value>, String> {
    let mut kv = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        kv.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str(ARCH.into()));
    put("general.name", Value::Str("Kitten-TTS 2 language model".into()));
    put(
        "general.license",
        Value::Str("Stellon Labs Community License (the weights)".into()),
    );
    put("kitten.source", Value::Str(source.into()));
    for (k, j) in [
        ("block_count", "num_hidden_layers"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
        ("attention.key_length", "head_dim"),
        ("vocab_size", "vocab_size"),
    ] {
        put(&format!("kitten.{k}"), Value::U32(get_u32(lm, j)?));
    }
    put(
        "kitten.attention.layer_norm_rms_epsilon",
        Value::F32(get_f32(lm, "rms_norm_eps")?),
    );
    put("kitten.rope.freq_base", Value::F32(get_f32(lm, "rope_theta")?));
    if lm["tie_word_embeddings"].as_bool() != Some(true) {
        return Err("expected tie_word_embeddings: true".into());
    }
    put(
        "kitten.speaker_dim",
        Value::U32(get_u32(model, "speaker_embedding_dim")?),
    );
    put("kitten.sample_rate", Value::U32(get_u32(model, "sample_rate")?));
    let tm = model["token_map"].as_object().ok_or("config: token_map missing")?;
    for (k, v) in tm {
        put(
            &format!("kitten.token_map.{k}"),
            Value::U32(v.as_u64().ok_or("token_map value")? as u32),
        );
    }
    let preset = &model["decode_presets"][model["default_preset"].as_str().unwrap_or("stable")];
    for k in ["temperature", "top_p", "min_p"] {
        put(&format!("kitten.sampling.{k}"), Value::F32(get_f32(preset, k)?));
    }
    put("kitten.sampling.top_k", Value::U32(get_u32(preset, "top_k")?));
    let gen = &model["generation"];
    put(
        "kitten.sampling.repetition_window",
        Value::U32(gen["repetition_window"].as_u64().unwrap_or(0) as u32),
    );
    if let Some(e) = gen["emotion_control"].as_str() {
        put("kitten.emotion_control", Value::Str(e.into()));
    }
    put("kitten.ternary.group_size", Value::U32(GROUP as u32));
    if let Some(t) = tokenizer_json {
        put("kitten.tokenizer.json", Value::Str(t.to_string()));
    }
    put("kitten.ternary.layout", Value::Str(layout.name().into()));
    Ok(kv)
}

/// What a conversion did.
#[derive(Debug, Default)]
pub struct Summary {
    pub ternary: usize,
    pub ternary_weights: usize,
    pub bf16: usize,
    pub f32: usize,
    pub bytes: usize,
}

/// Convert. `strict` refuses a source that `rule` does not reproduce exactly; a planted fault passes
/// `strict = false` so that the fault reaches the file and the readback check has to find it.
#[allow(clippy::too_many_arguments)] // the source, two configs, a name, a path, three choices of rule and layout, and the tokenizer
pub fn convert(
    st: &SafeTensors,
    model: &serde_json::Value,
    lm: &serde_json::Value,
    source: &str,
    out: &Path,
    rule: Rule,
    strict: bool,
    layout: Layout,
    tokenizer_json: Option<&str>,
) -> Result<Summary, String> {
    if layout == Layout::Tq1 {
        return Err("TQ1_0 export is not supported; this runtime only reads existing TQ1 files".into());
    }
    let kv = metadata(model, lm, source, layout, tokenizer_json)?;
    let mut tensors = Vec::new();
    let mut sum = Summary::default();
    for (name, info) in &st.tensors {
        if name == "lm_head.weight" {
            return Err("lm_head.weight present, but the head is tied to the embedding".into());
        }
        let dims: Vec<u64> = info.shape.iter().rev().map(|&d| d as u64).collect();
        if is_ternary(name) {
            let [rows, cols] = info.shape[..] else {
                return Err(format!("{name}: not a matrix"));
            };
            if cols % GROUP != 0 {
                return Err(format!("{name}: {cols} columns is not a multiple of {GROUP}"));
            }
            let w = st.f32(name)?;
            let packed = pack_matrix(name, &w, rows, cols, rule, strict, layout)?;
            sum.ternary += 1;
            sum.ternary_weights += rows * cols;
            sum.bytes += packed.len();
            tensors.push(TensorOut {
                name: name.clone(),
                dims: vec![layout.stored_cols(cols) as u64, rows as u64],
                ty: if layout == Layout::Q4 {
                    GgmlType::Q4_0
                } else {
                    GgmlType::TQ2_0
                },
                data: packed,
            });
        } else if name == EMBED {
            if info.dtype != Dtype::BF16 {
                return Err(format!("{name}: expected bf16, found {:?}", info.dtype));
            }
            let data = st.raw(name)?.to_vec();
            sum.bf16 += 1;
            sum.bytes += data.len();
            tensors.push(TensorOut {
                name: name.clone(),
                dims,
                ty: GgmlType::BF16,
                data,
            });
        } else {
            let data: Vec<u8> = st.f32(name)?.iter().flat_map(|x| x.to_le_bytes()).collect();
            sum.f32 += 1;
            sum.bytes += data.len();
            tensors.push(TensorOut {
                name: name.clone(),
                dims,
                ty: GgmlType::F32,
                data,
            });
        }
    }
    gguf::write(out, &kv, &tensors).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(sum)
}

/// Pack a ternary matrix's rows on up to four threads.
fn pack_matrix(
    name: &str,
    w: &[f32],
    rows: usize,
    cols: usize,
    rule: Rule,
    strict: bool,
    layout: Layout,
) -> Result<Vec<u8>, String> {
    let row_bytes = layout.row_bytes(cols);
    let threads = 4.min(rows.max(1));
    let per = rows.div_ceil(threads);
    let parts: Vec<Result<Vec<u8>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut out = Vec::with_capacity(per * row_bytes);
                    for r in (t * per)..((t + 1) * per).min(rows) {
                        layout
                            .pack_row(&w[r * cols..(r + 1) * cols], rule, strict, &mut out)
                            .map_err(|e| format!("{name} row {r}: {e:?}"))?;
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("pack thread panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(rows * row_bytes);
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// What the readback check found.
#[derive(Debug, Default)]
pub struct Verified {
    pub ternary: usize,
    pub groups: usize,
    pub others: usize,
}

/// The readback check: every tensor of the source, read back from the GGUF, equals the source. Ternary matrices:
/// trit for trit, scale bits for scale bits, value for value, padding zero ([`ternary::check_row`]). The bf16
/// embedding: byte for byte. Everything else: f32 bits equal to the widened source.
pub fn verify(st: &SafeTensors, g: &Gguf) -> Result<Verified, String> {
    if g.str("general.architecture")? != ARCH {
        return Err("not a Kitten-TTS 2 GGUF".into());
    }
    let layout = Layout::parse(g.str("kitten.ternary.layout")?).ok_or("unknown ternary layout")?;
    if g.u32("kitten.ternary.group_size")? as usize != GROUP {
        return Err("unknown ternary group size".into());
    }
    if g.tensors.len() != st.tensors.len() {
        return Err(format!(
            "{} tensors in the GGUF, {} in the source",
            g.tensors.len(),
            st.tensors.len()
        ));
    }
    let mut v = Verified::default();
    for (name, info) in &st.tensors {
        let (t, data) = g.tensor(name)?;
        if is_ternary(name) {
            let (rows, cols) = (info.shape[0], info.shape[1]);
            let want = if layout == Layout::Q4 {
                GgmlType::Q4_0
            } else {
                GgmlType::TQ2_0
            };
            if t.ty != want || t.dims != [layout.stored_cols(cols) as u64, rows as u64] {
                return Err(format!("{name}: stored as {:?} {:?}", t.ty, t.dims));
            }
            let w = st.f32(name)?;
            let row_bytes = layout.row_bytes(cols);
            for r in 0..rows {
                layout
                    .check_row(&w[r * cols..(r + 1) * cols], &data[r * row_bytes..(r + 1) * row_bytes])
                    .map_err(|e| format!("{name} row {r}: {e}"))?;
            }
            v.ternary += 1;
            v.groups += rows * cols / GROUP;
        } else if name == EMBED {
            if t.ty != GgmlType::BF16 || data != st.raw(name)? {
                return Err(format!("{name}: bf16 bytes differ"));
            }
            v.others += 1;
        } else {
            let src = st.f32(name)?;
            if t.ty != GgmlType::F32 || data.len() != src.len() * 4 {
                return Err(format!("{name}: stored as {:?}", t.ty));
            }
            for (i, (c, s)) in data.chunks_exact(4).zip(&src).enumerate() {
                if f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits() != s.to_bits() {
                    return Err(format!("{name}[{i}]: differs"));
                }
            }
            v.others += 1;
        }
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::ScaleRule;

    #[test]
    fn the_196_names() {
        let mut n = 0;
        for l in 0..28 {
            for t in ["q", "k", "v", "o"] {
                n += usize::from(is_ternary(&format!("model.layers.{l}.self_attn.{t}_proj.weight")));
            }
            for t in ["gate", "up", "down"] {
                n += usize::from(is_ternary(&format!("model.layers.{l}.mlp.{t}_proj.weight")));
            }
            assert!(!is_ternary(&format!("model.layers.{l}.input_layernorm.weight")));
            assert!(!is_ternary(&format!("model.layers.{l}.self_attn.q_norm.weight")));
        }
        assert_eq!(n, 196);
        assert!(!is_ternary(EMBED));
        assert!(!is_ternary("spk_proj.0.weight"));
    }

    /// A tiny model in Kitten's key layout: one layer, hidden 256, as safetensors bytes.
    fn tiny_safetensors() -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        let mut add = |name: &str, shape: &[usize], dtype: &str, bytes: Vec<u8>| {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                name.into(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        };
        let bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let mut seed = 1;
        for t in ["q", "k", "v", "o"] {
            let w: Vec<f32> = (0..256)
                .flat_map(|_| {
                    crate::ternary::tests::kitten_row(256, {
                        seed += 1;
                        seed
                    })
                })
                .collect();
            add(
                &format!("model.layers.0.self_attn.{t}_proj.weight"),
                &[256, 256],
                "BF16",
                bf16(&w),
            );
        }
        add(
            "model.layers.0.input_layernorm.weight",
            &[256],
            "BF16",
            bf16(&[1.5; 256]),
        );
        add(EMBED, &[8, 256], "BF16", bf16(&[0.25; 8 * 256]));
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        out
    }

    fn configs() -> (serde_json::Value, serde_json::Value) {
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
        (model, lm)
    }

    fn round_trip(rule: Rule, strict: bool) -> Result<Verified, String> {
        round_trip_in(rule, strict, Layout::Tq2Pad)
    }

    fn round_trip_in(rule: Rule, strict: bool, layout: Layout) -> Result<Verified, String> {
        let st = SafeTensors::from_bytes(tiny_safetensors()).unwrap();
        let (model, lm) = configs();
        let tag = format!("{}-{:?}-{}", rule.group, rule.scale, layout.name());
        let dir = std::env::temp_dir().join(format!("ggtts-kitten-conv-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("m.gguf");
        convert(&st, &model, &lm, "tiny", &out, rule, strict, layout, None)?;
        let g = Gguf::open(&out).unwrap();
        let r = verify(&st, &g);
        std::fs::remove_dir_all(&dir).unwrap();
        r
    }

    #[test]
    fn converts_and_reads_back_exactly() {
        let v = round_trip(Rule::KITTEN, true).unwrap();
        assert_eq!(v.ternary, 4);
        assert_eq!(v.groups, 4 * 256 * 2);
        assert_eq!(v.others, 2);
    }

    #[test]
    fn planted_group_256_turns_the_check_red() {
        let wrong = Rule {
            group: 256,
            scale: ScaleRule::AbsMax,
        };
        assert!(round_trip(wrong, true).is_err(), "strict conversion refuses it");
        assert!(
            round_trip(wrong, false).is_err(),
            "and the readback finds it when it is written anyway"
        );
    }

    #[test]
    fn q4_0_converts_exactly_and_a_planted_group_256_is_red() {
        let v = round_trip_in(Rule::KITTEN, true, Layout::Q4).unwrap();
        assert_eq!(v.ternary, 4);
        let wrong = Rule {
            group: 256,
            scale: ScaleRule::AbsMax,
        };
        assert!(round_trip_in(wrong, false, Layout::Q4).is_err());
    }

    #[test]
    fn planted_absmean_turns_the_check_red() {
        let wrong = Rule {
            group: GROUP,
            scale: ScaleRule::AbsMean,
        };
        assert!(round_trip(wrong, false).is_err());
    }
}
