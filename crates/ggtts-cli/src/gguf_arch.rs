//! Reads `general.architecture` from a GGUF file's header, without ggml, so `ggtts-render` can pick the engine for
//! a model file before loading it. Only the key-value section is read; tensor data is never touched.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

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

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn string_of(r: &mut impl Read) -> io::Result<String> {
    let n = u64_of(r)?;
    if n > 1 << 20 {
        return Err(bad("GGUF string too long"));
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| bad("GGUF string is not UTF-8"))
}

/// Size in bytes of a fixed-size GGUF value type, or `None` for strings and arrays.
fn fixed_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,
    }
}

fn skip_value(r: &mut impl Read, t: u32) -> io::Result<()> {
    if let Some(n) = fixed_size(t) {
        io::copy(&mut r.take(n), &mut io::sink())?;
        return Ok(());
    }
    match t {
        8 => string_of(r).map(|_| ()),
        9 => {
            let et = u32_of(r)?;
            let n = u64_of(r)?;
            match fixed_size(et) {
                Some(s) => {
                    io::copy(&mut r.take(n * s), &mut io::sink())?;
                }
                None => {
                    for _ in 0..n {
                        skip_value(r, et)?;
                    }
                }
            }
            Ok(())
        }
        _ => Err(bad("unknown GGUF value type")),
    }
}

/// The file's `general.architecture`, for example `qwen3-tts`.
pub fn architecture(path: &Path) -> io::Result<String> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = u32_of(&mut r)?;
    if version < 2 {
        return Err(bad("GGUF version 1 is not supported"));
    }
    let _n_tensors = u64_of(&mut r)?;
    let n_kv = u64_of(&mut r)?;
    for _ in 0..n_kv {
        let key = string_of(&mut r)?;
        let t = u32_of(&mut r)?;
        if key == "general.architecture" && t == 8 {
            return string_of(&mut r);
        }
        skip_value(&mut r, t)?;
    }
    Err(bad("no general.architecture key"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
        out.extend((key.len() as u64).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }

    #[test]
    fn finds_the_architecture_after_other_keys() {
        let mut f = b"GGUF".to_vec();
        f.extend(3u32.to_le_bytes());
        f.extend(0u64.to_le_bytes());
        f.extend(3u64.to_le_bytes());
        // An array of strings, then a u32, then the architecture.
        f.extend(9u64.to_le_bytes());
        f.extend(b"names.arr");
        f.extend(9u32.to_le_bytes());
        f.extend(8u32.to_le_bytes());
        f.extend(2u64.to_le_bytes());
        for s in ["a", "bc"] {
            f.extend((s.len() as u64).to_le_bytes());
            f.extend(s.as_bytes());
        }
        f.extend(5u64.to_le_bytes());
        f.extend(b"count");
        f.extend(4u32.to_le_bytes());
        f.extend(7u32.to_le_bytes());
        kv_string(&mut f, "general.architecture", "qwen3-tts");
        let dir = std::env::temp_dir().join(format!("ggtts-arch-{}", std::process::id()));
        std::fs::write(&dir, &f).unwrap();
        let got = architecture(&dir);
        std::fs::remove_file(&dir).unwrap();
        assert_eq!(got.unwrap(), "qwen3-tts");
    }

    #[test]
    fn refuses_a_file_that_is_not_gguf() {
        let p = std::env::temp_dir().join(format!("ggtts-notgguf-{}", std::process::id()));
        std::fs::write(&p, b"RIFF....").unwrap();
        let got = architecture(&p);
        std::fs::remove_file(&p).unwrap();
        assert!(got.is_err());
    }
}
