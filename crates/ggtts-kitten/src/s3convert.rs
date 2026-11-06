//! Chatterbox-Turbo's S3Gen (`s3gen_meanflow.safetensors`) to a ggtts GGUF: the flow (`flow.*`) and the HiFT
//! vocoder (`mel2wav.*`), every tensor f32 as shipped, with PyTorch's weight-norm parametrizations folded.
//!
//! A weight-normed layer is stored as `original0` (g, one scalar per index of dimension 0) and `original1` (v), and
//! its weight is `g * v / ||v||`, the norm taken over every dimension but the first (`torch.nn.utils.weight_norm`
//! with `dim=0`; for a `ConvTranspose1d` dimension 0 is the input channel). The folded tensor is written as
//! `<layer>.weight`. The tokenizer and the speaker encoder are not converted: reference conditioning is still
//! prepared by the PyTorch tool.

use std::collections::BTreeMap;
use std::path::Path;

use crate::gguf::{self, GgmlType, TensorOut, Value};
use crate::safetensors::SafeTensors;

pub const ARCH: &str = "s3gen-meanflow";
const G: &str = ".parametrizations.weight.original0";
const V: &str = ".parametrizations.weight.original1";

/// `g * v / ||v||` per index of dimension 0.
pub fn fold_weight_norm(g: &[f32], v: &[f32]) -> Vec<f32> {
    let per = v.len() / g.len();
    let mut out = Vec::with_capacity(v.len());
    for (i, gi) in g.iter().enumerate() {
        let row = &v[i * per..(i + 1) * per];
        let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let s = f64::from(*gi) / norm;
        out.extend(row.iter().map(|&x| (f64::from(x) * s) as f32));
    }
    out
}

/// Tensors of one source under `prefix` (renamed with `rename`): weight norm folded, and every BatchNorm folded to
/// `<bn>.bn_scale` and `<bn>.bn_shift` (`scale = weight / sqrt(running_var + 1e-5)`, `shift = bias - mean * scale`;
/// weight 1 and bias 0 for a BatchNorm without affine parameters). Integer counters are skipped.
fn collect(
    st: &SafeTensors,
    keep: &dyn Fn(&str) -> bool,
    rename: &dyn Fn(&str) -> String,
    tensors: &mut Vec<TensorOut>,
) -> Result<(), String> {
    let mut push = |name: String, dims: Vec<u64>, data: Vec<f32>| {
        let dims = if dims.is_empty() { vec![1] } else { dims };
        tensors.push(TensorOut {
            name,
            dims,
            ty: GgmlType::F32,
            data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        });
    };
    for (name, info) in &st.tensors {
        if !keep(name) || name.ends_with(G) || info.dtype == crate::safetensors::Dtype::I64 {
            continue;
        }
        if let Some(bn) = name.strip_suffix(".running_mean") {
            let mean = st.f32(name)?;
            let var = st.f32(&format!("{bn}.running_var"))?;
            let w = st
                .f32(&format!("{bn}.weight"))
                .unwrap_or_else(|_| vec![1.0; mean.len()]);
            let b = st.f32(&format!("{bn}.bias")).unwrap_or_else(|_| vec![0.0; mean.len()]);
            let scale: Vec<f32> = (0..mean.len())
                .map(|i| (f64::from(w[i]) / (f64::from(var[i]) + 1e-5).sqrt()) as f32)
                .collect();
            let shift: Vec<f32> = (0..mean.len()).map(|i| b[i] - mean[i] * scale[i]).collect();
            push(rename(&format!("{bn}.bn_scale")), vec![mean.len() as u64], scale);
            push(rename(&format!("{bn}.bn_shift")), vec![mean.len() as u64], shift);
            continue;
        }
        if name.ends_with(".running_var") {
            continue;
        }
        let (out_name, data) = if let Some(layer) = name.strip_suffix(V) {
            let g = st.f32(&format!("{layer}{G}"))?;
            (format!("{layer}.weight"), fold_weight_norm(&g, &st.f32(name)?))
        } else {
            (name.clone(), st.f32(name)?)
        };
        push(
            rename(&out_name),
            info.shape.iter().rev().map(|&d| d as u64).collect(),
            data,
        );
    }
    Ok(())
}

/// S3Gen (flow, vocoder, S3 tokenizer, CAMPPlus) and, when given, Kitten's pyannote speaker model (as `pyannote.*`).
pub fn convert(st: &SafeTensors, pyannote: Option<&SafeTensors>, out: &Path) -> Result<usize, String> {
    let mut tensors = Vec::new();
    let keep = |n: &str| {
        ["flow.", "mel2wav.", "tokenizer.", "speaker_encoder."]
            .iter()
            .any(|p| n.starts_with(p))
    };
    collect(st, &keep, &|n| n.to_string(), &mut tensors)?;
    if let Some(p) = pyannote {
        collect(p, &|_| true, &|n| format!("pyannote.{n}"), &mut tensors)?;
    }
    let mut kv = BTreeMap::new();
    kv.insert("general.architecture".to_string(), Value::Str(ARCH.into()));
    kv.insert(
        "general.name".to_string(),
        Value::Str("S3Gen meanflow decoder (Chatterbox-Turbo)".into()),
    );
    kv.insert("general.license".to_string(), Value::Str("MIT".into()));
    let n = tensors.len();
    gguf::write(out, &kv, &tensors).map_err(|e| e.to_string())?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_matches_the_definition() {
        let g = [2.0f32, 0.5];
        let v = [3.0f32, 4.0, 0.0, 1.0, 0.0, 0.0];
        let w = fold_weight_norm(&g, &v);
        assert_eq!(w, vec![2.0 * 3.0 / 5.0, 2.0 * 4.0 / 5.0, 0.0, 0.5, 0.0, 0.0]);
    }
}
