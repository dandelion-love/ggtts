//! A minimal safetensors reader: the header, and each tensor's dtype, shape and bytes.
//!
//! The format is an 8-byte little-endian header length, a JSON header naming every tensor's `dtype`, `shape` and
//! `data_offsets` (relative to the end of the header), then the bytes. Only what the converter reads is supported.

use std::collections::BTreeMap;
use std::path::Path;

/// A tensor's element type, as safetensors names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
    /// Present in some files (buffers, counters); read as raw bytes only.
    I64,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "BF16" => Ok(Dtype::BF16),
            "F16" => Ok(Dtype::F16),
            "F32" => Ok(Dtype::F32),
            "I64" => Ok(Dtype::I64),
            other => Err(format!("unsupported safetensors dtype {other}")),
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::I64 => 8,
        }
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone)]
pub struct Info {
    pub dtype: Dtype,
    /// Outermost dimension first, as PyTorch writes it.
    pub shape: Vec<usize>,
    /// Byte range inside the data section.
    pub start: usize,
    pub end: usize,
}

impl Info {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A whole safetensors file in memory.
pub struct SafeTensors {
    bytes: Vec<u8>,
    data_start: usize,
    pub tensors: BTreeMap<String, Info>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("file too short for a safetensors header".into());
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(n)
            .filter(|&e| e <= bytes.len())
            .ok_or("header overruns the file")?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..data_start]).map_err(|e| format!("safetensors header: {e}"))?;
        let obj = header.as_object().ok_or("safetensors header is not an object")?;
        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = Dtype::parse(v["dtype"].as_str().ok_or("missing dtype")?)?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or("missing shape")?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad shape"))
                .collect::<Result<_, _>>()?;
            let offs = v["data_offsets"].as_array().ok_or("missing data_offsets")?;
            let start = offs.first().and_then(|x| x.as_u64()).ok_or("bad data_offsets")? as usize;
            let end = offs.get(1).and_then(|x| x.as_u64()).ok_or("bad data_offsets")? as usize;
            let info = Info {
                dtype,
                shape,
                start,
                end,
            };
            if end < start || data_start + end > bytes.len() || end - start != info.numel() * dtype.size() {
                return Err(format!("{name}: data_offsets disagree with dtype and shape"));
            }
            tensors.insert(name.clone(), info);
        }
        Ok(SafeTensors {
            bytes,
            data_start,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Result<&Info, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not in the file"))
    }

    pub fn raw(&self, name: &str) -> Result<&[u8], String> {
        let i = self.info(name)?;
        Ok(&self.bytes[self.data_start + i.start..self.data_start + i.end])
    }

    /// The tensor widened to f32. bf16 and f16 widen exactly.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let i = self.info(name)?;
        let raw = self.raw(name)?;
        Ok(match i.dtype {
            Dtype::BF16 => raw
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F16 => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::I64 => return Err(format!("{name}: an integer tensor is not read as f32")),
        })
    }
}

/// bf16 is the top half of an f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// IEEE half to f32, exactly (subnormals, infinities and NaN included).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: m * 2^-24.
            let v = m as f32 * f32::from_bits(0x3380_0000);
            return if sign != 0 { -v } else { v };
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// `x` as an IEEE half, only if the half holds it exactly; `None` otherwise (out of range, or bits that would round).
pub fn f32_to_f16_exact(x: f32) -> Option<u16> {
    if !x.is_finite() {
        return None;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return Some(sign);
    }
    let h = if a >= f32::from_bits(0x3880_0000) {
        // Normal half range starts at 2^-14.
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        if exp >= 0x1f || bits & 0x1fff != 0 {
            return None;
        }
        sign | ((exp as u16) << 10) | (((bits >> 13) & 0x3ff) as u16)
    } else {
        // Subnormal half: a multiple of 2^-24 below 2^-14.
        let m = a / f32::from_bits(0x3380_0000);
        if m.fract() != 0.0 || m >= 1024.0 {
            return None;
        }
        sign | m as u16
    };
    (f16_to_f32(h).to_bits() == bits).then_some(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trips_every_finite_value() {
        for h in 0u16..=0xffff {
            let x = f16_to_f32(h);
            if !x.is_finite() {
                continue;
            }
            let back = f32_to_f16_exact(x).expect("a half's own value is exact");
            assert_eq!(f16_to_f32(back).to_bits(), x.to_bits(), "{h:#06x}");
        }
    }

    #[test]
    fn inexact_values_are_refused() {
        assert_eq!(f32_to_f16_exact(1.0 + f32::EPSILON), None);
        assert_eq!(f32_to_f16_exact(70000.0), None);
        assert_eq!(f32_to_f16_exact(1e-9), None);
        assert!(f32_to_f16_exact(0.0123).is_none());
        assert!(f32_to_f16_exact(bf16_to_f32(0x3c49)).is_some()); // a bf16 near 0.0123 is exact
    }
}
