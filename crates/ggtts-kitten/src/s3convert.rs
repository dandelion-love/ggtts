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

pub fn convert(st: &SafeTensors, out: &Path) -> Result<usize, String> {
    let mut tensors = Vec::new();
    for (name, info) in &st.tensors {
        if !(name.starts_with("flow.") || name.starts_with("mel2wav.")) || name.ends_with(G) {
            continue;
        }
        let (out_name, data) = if let Some(layer) = name.strip_suffix(V) {
            let g = st.f32(&format!("{layer}{G}"))?;
            (format!("{layer}.weight"), fold_weight_norm(&g, &st.f32(name)?))
        } else {
            (name.clone(), st.f32(name)?)
        };
        let dims: Vec<u64> = info.shape.iter().rev().map(|&d| d as u64).collect();
        let dims = if dims.is_empty() { vec![1] } else { dims };
        tensors.push(TensorOut {
            name: out_name,
            dims,
            ty: GgmlType::F32,
            data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        });
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
