//! A GGUF v3 writer and reader, for the value and tensor types this engine uses.
//!
//! Layout: magic `GGUF`, version, tensor count, key-value count, the key-values, the tensor infos (name, dims with
//! the fastest-varying first, ggml type, offset into the data section), padding to the alignment, then the data.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;

pub const ALIGNMENT: usize = 32;

/// ggml tensor types used here, by their ggml enum value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    TQ1_0 = 34,
    TQ2_0 = 35,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GgmlType::F32),
            1 => Some(GgmlType::F16),
            2 => Some(GgmlType::Q4_0),
            30 => Some(GgmlType::BF16),
            34 => Some(GgmlType::TQ1_0),
            35 => Some(GgmlType::TQ2_0),
            _ => None,
        }
    }

    /// Bytes for `n` elements of one row.
    pub fn row_bytes(self, n: usize) -> Result<usize, String> {
        let (block, bytes) = match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::TQ1_0 => (256, crate::ternary::TQ1_0_BLOCK_BYTES),
            GgmlType::TQ2_0 => (256, crate::ternary::TQ2_0_BLOCK_BYTES),
            GgmlType::Q4_0 => (32, crate::ternary::Q4_0_BLOCK_BYTES),
        };
        if n == 0 || n % block != 0 {
            return Err(format!("invalid {self:?} row dimension {n}"));
        }
        (n / block).checked_mul(bytes).ok_or_else(|| "row size overflow".into())
    }
}

fn tensor_bytes(dims: &[u64], ty: GgmlType) -> Result<usize, String> {
    if dims.is_empty() || dims.len() > 4 || dims.iter().any(|&d| d == 0 || d > i64::MAX as u64) {
        return Err("invalid tensor dimensions".into());
    }
    let cols = usize::try_from(dims[0]).map_err(|_| "tensor dimension overflow")?;
    let mut size = ty.row_bytes(cols)?;
    for &d in &dims[1..] {
        let d = usize::try_from(d).map_err(|_| "tensor dimension overflow")?;
        size = size.checked_mul(d).ok_or("tensor size overflow")?;
    }
    Ok(size)
}

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrI32(Vec<i32>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::I32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// A tensor to write: its dims (fastest first), type and bytes.
pub struct TensorOut {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub data: Vec<u8>,
}

fn put_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn put_value(w: &mut impl Write, v: &Value) -> io::Result<()> {
    match v {
        Value::U32(x) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Value::I32(x) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Value::F32(x) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Value::Bool(x) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[u8::from(*x)])
        }
        Value::Str(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            put_str(w, s)
        }
        Value::ArrI32(a) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            for x in a {
                w.write_all(&x.to_le_bytes())?;
            }
            Ok(())
        }
    }
}

fn pad(n: usize) -> usize {
    n.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// Write a GGUF file. Tensors are written in the order given.
pub fn write(path: &Path, kv: &BTreeMap<String, Value>, tensors: &[TensorOut]) -> io::Result<()> {
    let tmp = path.with_extension("gguf.partial");
    let mut w = io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        put_str(&mut head, k)?;
        put_value(&mut head, v)?;
    }
    let mut offset = 0usize;
    for t in tensors {
        let expect = tensor_bytes(&t.dims, t.ty).map_err(io::Error::other)?;
        if expect != t.data.len() {
            return Err(io::Error::other(format!(
                "{}: {} bytes for dims {:?}",
                t.name,
                t.data.len(),
                t.dims
            )));
        }
        put_str(&mut head, &t.name)?;
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&(t.ty as u32).to_le_bytes());
        head.extend_from_slice(&(offset as u64).to_le_bytes());
        offset = pad(offset + t.data.len());
    }
    head.resize(pad(head.len()), 0);
    w.write_all(&head)?;
    for t in tensors {
        w.write_all(&t.data)?;
        w.write_all(&vec![0u8; pad(t.data.len()) - t.data.len()])?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)
}

/// A tensor read back.
#[derive(Debug, Clone)]
pub struct TensorIn {
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte range in the file.
    pub start: usize,
    pub len: usize,
}

/// A GGUF file in memory.
pub struct Gguf {
    pub bytes: Vec<u8>,
    pub kv: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorIn>,
    /// Tensor names in file order.
    pub order: Vec<String>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.p.checked_add(n).ok_or("GGUF offset overflow")?;
        let s = self.b.get(self.p..end).ok_or("truncated GGUF")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = usize::try_from(self.u64()?).map_err(|_| "GGUF string length overflow")?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
    }
    fn scalar_size(ty: u32) -> Result<usize, String> {
        Ok(match ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return Err(format!("GGUF value type {ty} has no fixed size")),
        })
    }
    fn value(&mut self, ty: u32) -> Result<Option<Value>, String> {
        Ok(match ty {
            4 => Some(Value::U32(self.u32()?)),
            5 => Some(Value::I32(self.u32()? as i32)),
            6 => Some(Value::F32(f32::from_bits(self.u32()?))),
            7 => Some(Value::Bool(self.take(1)?[0] != 0)),
            8 => Some(Value::Str(self.str()?)),
            9 => {
                let et = self.u32()?;
                let n = usize::try_from(self.u64()?).map_err(|_| "GGUF array length overflow")?;
                if et == 5 {
                    let len = n.checked_mul(4).ok_or("GGUF array size overflow")?;
                    if len > self.b.len().saturating_sub(self.p) {
                        return Err("truncated GGUF array".into());
                    }
                    let mut a = Vec::with_capacity(n);
                    for _ in 0..n {
                        a.push(self.u32()? as i32);
                    }
                    Some(Value::ArrI32(a))
                } else if et == 8 {
                    if n > self.b.len().saturating_sub(self.p) / 8 {
                        return Err("truncated GGUF string array".into());
                    }
                    for _ in 0..n {
                        self.str()?;
                    }
                    None
                } else {
                    self.take(
                        n.checked_mul(Self::scalar_size(et)?)
                            .ok_or("GGUF array size overflow")?,
                    )?;
                    None
                }
            }
            t => {
                self.take(Self::scalar_size(t)?)?;
                None
            }
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        let mut kv = BTreeMap::new();
        let mut infos = Vec::new();
        let data_start;
        {
            let mut r = Rd { b: &bytes, p: 0 };
            if r.take(4)? != b"GGUF" {
                return Err("not a GGUF file".into());
            }
            let version = r.u32()?;
            if version != 3 {
                return Err(format!("GGUF version {version}, expected 3"));
            }
            let nt = usize::try_from(r.u64()?).map_err(|_| "tensor count overflow")?;
            let nkv = usize::try_from(r.u64()?).map_err(|_| "metadata count overflow")?;
            if nt > bytes.len() / 24 || nkv > bytes.len() / 12 {
                return Err("GGUF counts exceed file size".into());
            }
            for _ in 0..nkv {
                let k = r.str()?;
                let ty = r.u32()?;
                if let Some(v) = r.value(ty)? {
                    kv.insert(k, v);
                }
            }
            for _ in 0..nt {
                let name = r.str()?;
                let nd = r.u32()? as usize;
                if !(1..=4).contains(&nd) {
                    return Err(format!("{name}: invalid dimension count {nd}"));
                }
                let mut dims = Vec::with_capacity(nd);
                for _ in 0..nd {
                    dims.push(r.u64()?);
                }
                let tyv = r.u32()?;
                let ty = GgmlType::from_u32(tyv).ok_or_else(|| format!("{name}: ggml type {tyv} not supported"))?;
                let off = usize::try_from(r.u64()?).map_err(|_| "tensor offset overflow")?;
                infos.push((name, dims, ty, off));
            }
            data_start = r.p.checked_add(ALIGNMENT - 1).ok_or("GGUF alignment overflow")? / ALIGNMENT * ALIGNMENT;
        }
        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        if let Some(alignment) = kv.get("general.alignment") {
            if alignment.as_u32() != Some(ALIGNMENT as u32) {
                return Err("unsupported GGUF alignment".into());
            }
        }
        for (name, dims, ty, off) in infos {
            if off % ALIGNMENT != 0 {
                return Err(format!("{name}: unaligned tensor offset"));
            }
            let len = tensor_bytes(&dims, ty).map_err(|e| format!("{name}: {e}"))?;
            let start = data_start.checked_add(off).ok_or("tensor offset overflow")?;
            let end = start.checked_add(len).ok_or("tensor length overflow")?;
            if end > bytes.len() {
                return Err(format!("{name}: data overruns the file"));
            }
            if ty == GgmlType::TQ1_0 {
                if !matches!(
                    kv.get("kitten.ternary.layout").and_then(Value::as_str),
                    Some("tq1_0-256-experimental" | "tq1_0-pad128")
                ) {
                    return Err(format!("{name}: unsupported TQ1 layout"));
                }
                for block in bytes[start..end].chunks_exact(crate::ternary::TQ1_0_BLOCK_BYTES) {
                    let scale = crate::safetensors::f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
                    if !scale.is_finite() || scale < 0.0 {
                        return Err(format!("{name}: invalid TQ1 scale"));
                    }
                }
            }
            if tensors.contains_key(&name) {
                return Err(format!("{name}: duplicate tensor"));
            }
            order.push(name.clone());
            tensors.insert(name, TensorIn { dims, ty, start, len });
        }
        Ok(Gguf {
            bytes,
            kv,
            tensors,
            order,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<(&TensorIn, &[u8]), String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not in the GGUF"))?;
        Ok((t, &self.bytes[t.start..t.start + t.len]))
    }

    pub fn u32(&self, key: &str) -> Result<u32, String> {
        self.kv
            .get(key)
            .and_then(Value::as_u32)
            .ok_or_else(|| format!("GGUF key {key} missing or not an integer"))
    }

    pub fn f32(&self, key: &str) -> Result<f32, String> {
        self.kv
            .get(key)
            .and_then(Value::as_f32)
            .ok_or_else(|| format!("GGUF key {key} missing or not a float"))
    }

    pub fn str(&self, key: &str) -> Result<&str, String> {
        self.kv
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("GGUF key {key} missing or not a string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tq1_file(dims: &[u64], ty: u32, layout: &str, data: &[u8], offset: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        put_str(&mut b, "kitten.ternary.layout").unwrap();
        put_value(&mut b, &Value::Str(layout.into())).unwrap();
        put_str(&mut b, "w").unwrap();
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b.resize(pad(b.len()), 0);
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn tq1_parser_rejects_invalid_layout_shapes_scales_and_truncation() {
        let layout = "tq1_0-256-experimental";
        let mut block = [0u8; crate::ternary::TQ1_0_BLOCK_BYTES];
        block[52..54].copy_from_slice(&0x3c00u16.to_le_bytes());
        assert!(Gguf::parse(tq1_file(&[256, 1], 34, layout, &block, 0)).is_ok());
        assert!(Gguf::parse(tq1_file(&[255, 1], 34, layout, &block, 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[], 34, layout, &block, 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, i64::MAX as u64], 34, layout, &block, 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, 1], 999, layout, &block, 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, 1], 34, "unknown", &block, 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, 1], 34, layout, &block[..53], 0)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, 1], 34, layout, &block, u64::MAX - 31)).is_err());
        assert!(Gguf::parse(tq1_file(&[256, 1], 34, layout, &block, 1)).is_err());
        for scale in [0x7c00u16, 0x7e00, 0xbc00] {
            block[52..54].copy_from_slice(&scale.to_le_bytes());
            assert!(Gguf::parse(tq1_file(&[256, 1], 34, layout, &block, 0)).is_err());
        }
    }

    #[test]
    fn write_then_read() {
        let dir = std::env::temp_dir().join(format!("ggtts-kitten-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        let mut kv = BTreeMap::new();
        kv.insert("a.u".into(), Value::U32(7));
        kv.insert("a.s".into(), Value::Str("x".into()));
        kv.insert("a.f".into(), Value::F32(0.5));
        kv.insert("a.arr".into(), Value::ArrI32(vec![1, -2]));
        let t = vec![
            TensorOut {
                name: "w".into(),
                dims: vec![3, 2],
                ty: GgmlType::F32,
                data: vec![1u8; 24],
            },
            TensorOut {
                name: "b".into(),
                dims: vec![5],
                ty: GgmlType::BF16,
                data: vec![2u8; 10],
            },
        ];
        write(&path, &kv, &t).unwrap();
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.kv, kv);
        assert_eq!(g.order, vec!["w", "b"]);
        let (ti, d) = g.tensor("b").unwrap();
        assert_eq!(ti.dims, vec![5]);
        assert_eq!(d, &[2u8; 10][..]);
        assert_eq!(ti.start % ALIGNMENT, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
