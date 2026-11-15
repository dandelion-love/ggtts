//! A GGUF file's key-value section, read and rewritten without ggml and without touching tensor data.
//!
//! [`header`] reads a file up to the end of its tensor directory: its keys (each value kept as its encoded bytes, so
//! it can be written back unchanged), its tensor infos as bytes, and where the data section starts. [`rewrite`]
//! writes a copy with some string keys set, every other key and every tensor byte as they were: tensor offsets are
//! relative to the data section, so only the header grows and the data moves as a whole, still aligned.
//!
//! A path of exactly `/proc/self/fd/<n>` is read through that descriptor ([`crate::fdpath`]).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// GGUF's value type of a string.
pub const TYPE_STRING: u32 = 8;
const TYPE_U32: u32 = 4;
const TYPE_ARRAY: u32 = 9;
/// The data section's alignment when the file does not set `general.alignment`.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// One key and its value, the value as its encoded bytes (after the type).
#[derive(Debug, Clone, PartialEq)]
pub struct Kv {
    pub key: String,
    pub ty: u32,
    pub raw: Vec<u8>,
}

impl Kv {
    /// A string key.
    pub fn string(key: &str, value: &str) -> Kv {
        let mut raw = (value.len() as u64).to_le_bytes().to_vec();
        raw.extend_from_slice(value.as_bytes());
        Kv { key: key.to_string(), ty: TYPE_STRING, raw }
    }

    /// The value, when it is a string.
    pub fn as_str(&self) -> Option<&str> {
        if self.ty != TYPE_STRING || self.raw.len() < 8 {
            return None;
        }
        std::str::from_utf8(&self.raw[8..]).ok()
    }

    fn as_u32(&self) -> Option<u32> {
        (self.ty == TYPE_U32 && self.raw.len() == 4).then(|| u32::from_le_bytes(self.raw[..4].try_into().unwrap()))
    }
}

/// A GGUF header: its version, keys in file order, tensor count and directory bytes, and where the data starts.
#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub kv: Vec<Kv>,
    pub n_tensors: u64,
    pub tensor_infos: Vec<u8>,
    /// The byte where the header (keys and tensor directory) ends.
    pub end: u64,
    pub alignment: u64,
}

impl Header {
    /// The byte where the data section starts: [`Header::end`] rounded up to the alignment.
    pub fn data_start(&self) -> u64 {
        self.end.div_ceil(self.alignment) * self.alignment
    }

    /// The string keys that start with `prefix`, by key.
    pub fn strings(&self, prefix: &str) -> BTreeMap<String, String> {
        self.kv
            .iter()
            .filter(|k| k.key.starts_with(prefix))
            .filter_map(|k| k.as_str().map(|v| (k.key.clone(), v.to_string())))
            .collect()
    }
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

struct Counting<R> {
    inner: R,
    n: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let k = self.inner.read(buf)?;
        self.n += k as u64;
        Ok(k)
    }
}

fn take(r: &mut impl Read, n: u64, out: &mut Vec<u8>) -> io::Result<()> {
    if n > 1 << 30 {
        return Err(bad("GGUF value too large"));
    }
    let at = out.len();
    out.resize(at + n as usize, 0);
    r.read_exact(&mut out[at..])
}

fn u32_of(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64_of(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn string_of(r: &mut impl Read) -> io::Result<String> {
    let n = u64_of(r)?;
    if n > 1 << 20 {
        return Err(bad("GGUF string too long"));
    }
    let mut b = Vec::new();
    take(r, n, &mut b)?;
    String::from_utf8(b).map_err(|_| bad("GGUF string is not UTF-8"))
}

fn fixed_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Read one value of type `t`, appending its encoded bytes to `out`.
fn value_raw(r: &mut impl Read, t: u32, out: &mut Vec<u8>) -> io::Result<()> {
    if let Some(n) = fixed_size(t) {
        return take(r, n, out);
    }
    match t {
        TYPE_STRING => {
            let n = u64_of(r)?;
            if n > 1 << 30 {
                return Err(bad("GGUF string too long"));
            }
            out.extend_from_slice(&n.to_le_bytes());
            take(r, n, out)
        }
        TYPE_ARRAY => {
            let et = u32_of(r)?;
            let n = u64_of(r)?;
            out.extend_from_slice(&et.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
            match fixed_size(et) {
                Some(s) => take(r, n.checked_mul(s).ok_or_else(|| bad("GGUF array too large"))?, out),
                None => {
                    for _ in 0..n {
                        value_raw(r, et, out)?;
                    }
                    Ok(())
                }
            }
        }
        _ => Err(bad(format!("unknown GGUF value type {t}"))),
    }
}

/// Read a GGUF header from `r`, up to the end of the tensor directory. Tensor data is not read.
pub fn read_header(r: impl Read) -> io::Result<Header> {
    let mut r = Counting { inner: r, n: 0 };
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = u32_of(&mut r)?;
    if version < 2 {
        return Err(bad(format!("GGUF version {version} is not supported (2 or later)")));
    }
    let n_tensors = u64_of(&mut r)?;
    let n_kv = u64_of(&mut r)?;
    if n_kv > 1 << 20 || n_tensors > 1 << 24 {
        return Err(bad("GGUF header counts are implausible"));
    }
    let mut kv = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = string_of(&mut r)?;
        let ty = u32_of(&mut r)?;
        let mut raw = Vec::new();
        value_raw(&mut r, ty, &mut raw)?;
        kv.push(Kv { key, ty, raw });
    }
    let mut tensor_infos = Vec::new();
    for _ in 0..n_tensors {
        let name_len = u64_of(&mut r)?;
        tensor_infos.extend_from_slice(&name_len.to_le_bytes());
        take(&mut r, name_len, &mut tensor_infos)?;
        let n_dims = u32_of(&mut r)?;
        if n_dims > 8 {
            return Err(bad("GGUF tensor has more than 8 dimensions"));
        }
        tensor_infos.extend_from_slice(&n_dims.to_le_bytes());
        take(&mut r, u64::from(n_dims) * 8 + 4 + 8, &mut tensor_infos)?; // dims, type, offset
    }
    let alignment = kv
        .iter()
        .find(|k| k.key == "general.alignment")
        .and_then(Kv::as_u32)
        .map(u64::from)
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(bad(format!("GGUF alignment {alignment} is not a power of two")));
    }
    Ok(Header { version, kv, n_tensors, tensor_infos, end: r.n, alignment })
}

/// [`read_header`] of the file at `path`.
pub fn header(path: &Path) -> io::Result<Header> {
    read_header(BufReader::new(crate::fdpath::open(path)?))
}

/// Write `input` to `output` with the string keys in `set` given those values (replaced where the key exists, added
/// at the end otherwise), and every other key, the tensor directory and every tensor byte unchanged.
pub fn rewrite(input: &Path, output: &Path, set: &BTreeMap<String, String>) -> io::Result<()> {
    if input == output {
        return Err(bad("rewrite to a new file, not over the input"));
    }
    let h = header(input)?;
    let mut kv = h.kv.clone();
    for (k, v) in set {
        match kv.iter_mut().find(|e| &e.key == k) {
            Some(e) => *e = Kv::string(k, v),
            None => kv.push(Kv::string(k, v)),
        }
    }
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&h.version.to_le_bytes());
    head.extend_from_slice(&h.n_tensors.to_le_bytes());
    head.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for e in &kv {
        head.extend_from_slice(&(e.key.len() as u64).to_le_bytes());
        head.extend_from_slice(e.key.as_bytes());
        head.extend_from_slice(&e.ty.to_le_bytes());
        head.extend_from_slice(&e.raw);
    }
    head.extend_from_slice(&h.tensor_infos);
    let padded = (head.len() as u64).div_ceil(h.alignment) * h.alignment;
    head.resize(padded as usize, 0);

    let mut src = crate::fdpath::open(input)?;
    src.seek(SeekFrom::Start(h.data_start()))?;
    let mut out = BufWriter::new(File::create(output)?);
    out.write_all(&head)?;
    io::copy(&mut src, &mut out)?;
    out.flush()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small GGUF: `kv` string keys, `general.alignment` 32, and two tensors whose data is `data`.
    pub(crate) fn tiny(kv: &[(&str, &str)], data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&((kv.len() + 1) as u64).to_le_bytes());
        let mut put_key = |b: &mut Vec<u8>, k: &str| {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
        };
        put_key(&mut b, "general.alignment");
        b.extend_from_slice(&TYPE_U32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());
        for (k, v) in kv {
            put_key(&mut b, k);
            b.extend_from_slice(&TYPE_STRING.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        }
        // Two f32 tensors of 2 elements, at offsets 0 and 32 of the data section.
        for (name, off) in [("a", 0u64), ("b", 32u64)] {
            put_key(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&2u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&off.to_le_bytes());
        }
        let padded = b.len().div_ceil(32) * 32;
        b.resize(padded, 0);
        b.extend_from_slice(data);
        b
    }

    fn temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ggtts-gguf-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn a_header_reads_its_string_keys_and_where_the_data_starts() {
        let data: Vec<u8> = (0..64u8).collect();
        let bytes = tiny(&[("general.architecture", "qwen3-tts"), ("x.y", "z")], &data);
        let h = read_header(&bytes[..]).unwrap();
        assert_eq!(h.n_tensors, 2);
        assert_eq!(h.alignment, 32);
        assert_eq!(h.strings("general.").get("general.architecture").map(String::as_str), Some("qwen3-tts"));
        assert_eq!(&bytes[h.data_start() as usize..], &data[..]);
    }

    #[test]
    fn a_rewrite_sets_keys_and_leaves_every_tensor_byte() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37)).collect();
        let input = temp("in.gguf", &tiny(&[("general.architecture", "qwen3-tts"), ("ggtts.voice.a.name", "old")], &data));
        let output = input.with_file_name("out.gguf");
        let set = BTreeMap::from([
            ("ggtts.voice.a.name".to_string(), "Voice A".to_string()),
            ("ggtts.voice.a.language".to_string(), "yue-HK".to_string()),
        ]);
        rewrite(&input, &output, &set).unwrap();
        let before = read_header(&std::fs::read(&input).unwrap()[..]).unwrap();
        let out_bytes = std::fs::read(&output).unwrap();
        let after = read_header(&out_bytes[..]).unwrap();
        assert_eq!(after.tensor_infos, before.tensor_infos, "the tensor directory is unchanged");
        assert_eq!(&out_bytes[after.data_start() as usize..], &data[..], "every tensor byte is unchanged");
        assert_eq!(after.data_start() % 32, 0);
        let s = after.strings("ggtts.voice.");
        assert_eq!(s.get("ggtts.voice.a.name").map(String::as_str), Some("Voice A"));
        assert_eq!(s.get("ggtts.voice.a.language").map(String::as_str), Some("yue-HK"));
        assert_eq!(after.strings("general.").get("general.architecture").map(String::as_str), Some("qwen3-tts"));
        assert!(rewrite(&input, &input, &set).is_err(), "never over the input");
        let _ = std::fs::remove_dir_all(input.parent().unwrap());
    }

    #[test]
    fn what_is_not_a_gguf_is_refused() {
        assert!(read_header(&b"JSON{}"[..]).is_err());
        assert!(read_header(&b"GGUF\x01\x00\x00\x00"[..]).is_err(), "version 1");
    }
}
