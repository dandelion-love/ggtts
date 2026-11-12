//! Steering: a vector added to the talker's hidden state at the output of chosen layers.
//!
//! Steering is a request input, not an instrument. An instrument never changes the output; steering changes it by
//! design. So it never goes through [`hooks`](crate::hooks) or the trace, and it follows the shape of
//! [`ChunkRequest::instruction`](crate::engine::ChunkRequest::instruction):
//!
//! * it is a field of the request ([`ChunkRequest::steer`](crate::engine::ChunkRequest::steer)), and `None` is the
//!   render exactly as without the field;
//! * an engine that cannot apply it refuses a request that carries it ([`refuse_steer`]), and never ignores it;
//! * it is recorded ([`Steering::record`]), so a steered render can never be taken for a plain one.
//!
//! The file is safetensors: an 8-byte little-endian header length, a JSON header naming each tensor's dtype, shape
//! and byte range, then the raw bytes. Only 1-D `F32` tensors named `talker.layer.<i>` are accepted, one per layer.
//! PyTorch writes the format directly, so a reference and this engine read the same bytes.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::engine::{ChunkRequest, EngineError};
use crate::seed::{hex, sha256};

/// The tensor name prefix: `talker.layer.<i>`.
pub const PREFIX: &str = "talker.layer.";

/// Why a steering file or selection was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SteerError(pub String);

impl std::fmt::Display for SteerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SteerError {}

fn refuse(msg: impl Into<String>) -> SteerError {
    SteerError(msg.into())
}

/// A steering file as read: its sha256 and one vector per layer, by layer index.
#[derive(Debug, Clone, PartialEq)]
pub struct SteerFile {
    /// The sha256 of the whole file, lower-case hex.
    pub sha256: String,
    /// One vector per layer, in layer order.
    pub layers: BTreeMap<u32, Vec<f32>>,
}

/// The layer index in a tensor name, when the name is `talker.layer.<i>` with `<i>` in plain decimal.
fn layer_of(name: &str) -> Option<u32> {
    let digits = name.strip_prefix(PREFIX)?;
    let plain =
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && (digits == "0" || !digits.starts_with('0'));
    if plain {
        digits.parse().ok()
    } else {
        None
    }
}

impl SteerFile {
    /// Reads and checks a steering file. A file that cannot be read is refused with its path.
    pub fn read(path: &Path) -> Result<Self, SteerError> {
        let bytes = std::fs::read(path).map_err(|e| refuse(format!("steering file {}: {e}", path.display())))?;
        Self::parse(&bytes).map_err(|e| refuse(format!("steering file {}: {e}", path.display())))
    }

    /// Parses and checks the bytes of a steering file.
    pub fn parse(bytes: &[u8]) -> Result<Self, SteerError> {
        let mut layers = BTreeMap::new();
        for (name, (shape, v)) in read_f32_tensors(bytes, |name| {
            if layer_of(name).is_none() {
                return Err(refuse(format!(
                    "tensor {name:?} is not named talker.layer.<i>; only talker layers can be steered"
                )));
            }
            Ok(())
        })? {
            let layer = layer_of(&name).expect("checked by the filter");
            if shape.len() != 1 {
                return Err(refuse(format!(
                    "tensor {name} has shape {shape:?}; a steering vector is one-dimensional"
                )));
            }
            if let Some(i) = v.iter().position(|x| !x.is_finite()) {
                return Err(refuse(format!(
                    "tensor {name} has a value that is not finite ({}) at element {i}",
                    v[i]
                )));
            }
            // A layer has one name (no leading zeros), so it cannot appear twice.
            layers.insert(layer, v);
        }
        if layers.is_empty() {
            return Err(refuse("the file holds no talker.layer.<i> tensor"));
        }
        Ok(Self {
            sha256: hex(&sha256(bytes)),
            layers,
        })
    }

    /// The steering to apply: the layers named in `layers` (every layer in the file when `None`), at `scale`.
    /// Refuses a layer the file does not hold, a layer named twice, an empty list, a scale that is not finite in f32,
    /// and a scale that makes any scaled value overflow f32 ([`Steering::check_scaled`]).
    pub fn select(&self, layers: Option<&[u32]>, scale: f64) -> Result<Steering, SteerError> {
        if !scale.is_finite() {
            return Err(refuse(format!("the steering scale {scale} is not a finite number")));
        }
        let wanted: Vec<u32> = match layers {
            None => self.layers.keys().copied().collect(),
            Some([]) => return Err(refuse("the steering layer list is empty")),
            Some(list) => {
                let mut seen = Vec::with_capacity(list.len());
                for &l in list {
                    if seen.contains(&l) {
                        return Err(refuse(format!("layer {l} is named twice in the steering layer list")));
                    }
                    if !self.layers.contains_key(&l) {
                        let held: Vec<String> = self.layers.keys().map(u32::to_string).collect();
                        return Err(refuse(format!(
                            "layer {l} is not in the steering file, which holds layers {}",
                            held.join(",")
                        )));
                    }
                    seen.push(l);
                }
                seen.sort_unstable();
                seen
            }
        };
        let s = Steering {
            sha256: self.sha256.clone(),
            scale,
            layers: wanted.iter().map(|l| (*l, self.layers[l].clone())).collect(),
        };
        s.check_scaled()?;
        Ok(s)
    }
}

/// F32 tensors by name, each with its shape and its values, as [`read_f32_tensors`] returns them.
pub type F32Tensors = BTreeMap<String, (Vec<u64>, Vec<f32>)>;

/// The tensors of a safetensors file, by name: each one's shape and its values. Every tensor must be `F32` and lie
/// inside the file; `check` sees each name first and may refuse it. The `__metadata__` entry is skipped.
pub fn read_f32_tensors(
    bytes: &[u8],
    mut check: impl FnMut(&str) -> Result<(), SteerError>,
) -> Result<F32Tensors, SteerError> {
    if bytes.len() < 8 {
        return Err(refuse("not a safetensors file: shorter than its 8-byte header length"));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let body = &bytes[8..];
    if n > body.len() as u64 {
        return Err(refuse(format!(
            "not a safetensors file: its header length {n} is past the end of the file"
        )));
    }
    let (head, data) = body.split_at(n as usize);
    let header: Value = serde_json::from_slice(head)
        .map_err(|e| refuse(format!("not a safetensors file: the header is not JSON ({e})")))?;
    let entries = header
        .as_object()
        .ok_or_else(|| refuse("not a safetensors file: the header is not a JSON object"))?;
    // A JSON object keeps the last of two equal keys, so a tensor named twice would lose one of its two entries
    // without a word. Count the header's own keys instead.
    let keys = top_level_keys(head);
    if keys.len() != entries.len() {
        let mut seen = std::collections::BTreeSet::new();
        let twice = keys
            .iter()
            .find(|k| !seen.insert(k.as_str()))
            .cloned()
            .unwrap_or_default();
        return Err(refuse(format!("the header names tensor {twice:?} more than once")));
    }
    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        check(name)?;
        let dtype = entry.get("dtype").and_then(Value::as_str);
        if dtype != Some("F32") {
            return Err(refuse(format!(
                "tensor {name} is {}, not F32",
                dtype.unwrap_or("of no dtype")
            )));
        }
        let shape: Vec<u64> = entry
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|s| s.iter().map(Value::as_u64).collect())
            .ok_or_else(|| refuse(format!("tensor {name} has no shape")))?;
        let len = shape
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .ok_or_else(|| refuse(format!("tensor {name} has shape {shape:?}, too large")))?;
        let offsets: Vec<u64> = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .and_then(|s| s.iter().map(Value::as_u64).collect())
            .ok_or_else(|| refuse(format!("tensor {name} has no data_offsets")))?;
        let [begin, end] = offsets[..] else {
            return Err(refuse(format!("tensor {name} has data_offsets {offsets:?}")));
        };
        if begin > end || end > data.len() as u64 || Some(end - begin) != len.checked_mul(4) {
            return Err(refuse(format!(
                "tensor {name}: data_offsets [{begin}, {end}] do not hold {len} f32 values in the file's {} data \
                 bytes",
                data.len()
            )));
        }
        let v: Vec<f32> = data[begin as usize..end as usize]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        out.insert(name.clone(), (shape, v));
    }
    Ok(out)
}

/// The keys of a JSON object's top level, in order, repeats kept. `json` must already have parsed as an object.
fn top_level_keys(json: &[u8]) -> Vec<String> {
    let mut keys = Vec::new();
    let (mut depth, mut i) = (0usize, 0usize);
    // At depth 1, a string is a key when it follows `{` or `,`.
    let mut expect_key = false;
    while i < json.len() {
        match json[i] {
            b'"' => {
                let start = i;
                i += 1;
                while json[i] != b'"' {
                    i += if json[i] == b'\\' { 2 } else { 1 };
                }
                if depth == 1 && expect_key {
                    // Valid JSON, so the string decodes.
                    keys.push(serde_json::from_slice(&json[start..=i]).unwrap_or_default());
                    expect_key = false;
                }
            }
            b'{' | b'[' => {
                depth += 1;
                expect_key = depth == 1 && json[i] == b'{';
            }
            b'}' | b']' => depth -= 1,
            b',' if depth == 1 => expect_key = true,
            _ => {}
        }
        i += 1;
    }
    keys
}

/// Parses a layer list such as `23,24,25`. Refuses an empty list, an entry that is not a whole number, and a layer
/// named twice.
pub fn parse_layers(list: &str) -> Result<Vec<u32>, SteerError> {
    let mut out = Vec::new();
    for part in list.split(',') {
        let part = part.trim();
        let l: u32 = part
            .parse()
            .map_err(|_| refuse(format!("{part:?} in the steering layer list is not a layer index")))?;
        if out.contains(&l) {
            return Err(refuse(format!("layer {l} is named twice in the steering layer list")));
        }
        out.push(l);
    }
    Ok(out)
}

/// The steering a request carries: the file's sha256, the layers in use with their vectors, and one scale.
#[derive(Debug, Clone, PartialEq)]
pub struct Steering {
    /// The steering file's sha256.
    pub sha256: String,
    /// One scale for every layer. An engine adds `scale * vector`, computed in f32.
    pub scale: f64,
    /// The layers applied, ascending, each with its vector as the file holds it (unscaled).
    pub layers: Vec<(u32, Vec<f32>)>,
}

impl Steering {
    /// The layer indices applied, ascending.
    pub fn layer_indices(&self) -> Vec<u32> {
        self.layers.iter().map(|(l, _)| *l).collect()
    }

    /// Each layer's vector times the scale, in f32, as an engine adds it.
    pub fn scaled(&self) -> Vec<(u32, Vec<f32>)> {
        let s = self.scale as f32;
        self.layers
            .iter()
            .map(|(l, v)| (*l, v.iter().map(|x| s * x).collect()))
            .collect()
    }

    /// Refuses a scale that is not finite in f32 (`1e39` is a finite f64 and infinite in f32), and a scale that makes
    /// any `scale * x` overflow f32: the engine computes in f32, so either would put an infinity in the hidden state.
    pub fn check_scaled(&self) -> Result<(), SteerError> {
        let s = self.scale as f32;
        if !s.is_finite() {
            return Err(refuse(format!(
                "the steering scale {} is not a finite number in f32, the precision the vectors are added in",
                self.scale
            )));
        }
        for (l, v) in &self.layers {
            if let Some(i) = v.iter().position(|x| !(s * x).is_finite()) {
                return Err(refuse(format!(
                    "the steering scale {} times element {i} of layer {l} ({}) is not a finite number in f32",
                    self.scale, v[i]
                )));
            }
        }
        Ok(())
    }

    /// Refuses steering a model with `n_layers` layers of `hidden_size` cannot take: a layer out of range, a vector
    /// of the wrong length, or a scale that overflows f32 ([`check_scaled`](Self::check_scaled)).
    pub fn check(&self, n_layers: usize, hidden_size: usize) -> Result<(), SteerError> {
        self.check_scaled()?;
        for (l, v) in &self.layers {
            if *l as usize >= n_layers {
                return Err(refuse(format!(
                    "steering layer {l} is out of range: the talker has layers 0 to {}",
                    n_layers.saturating_sub(1)
                )));
            }
            if v.len() != hidden_size {
                return Err(refuse(format!(
                    "the steering vector for layer {l} has {} values; the talker's hidden size is {hidden_size}",
                    v.len()
                )));
            }
        }
        Ok(())
    }

    /// What every report and the trace header record: `{"sha256", "layers", "scale"}`.
    pub fn record(&self) -> Value {
        json!({"sha256": self.sha256, "layers": self.layer_indices(), "scale": self.scale})
    }
}

/// The error an engine that cannot steer returns for a request that carries steering.
pub fn refuse_steer(req: &ChunkRequest<'_>, family: &str) -> Result<(), EngineError> {
    match req.steer {
        None => Ok(()),
        Some(_) => Err(EngineError(format!(
            "the {family} engine cannot apply steering; only the qwen3-tts engine can"
        ))),
    }
}

/// Builds a safetensors file from `(name, values)` pairs, every tensor 1-D F32, in the order given. For tests and
/// tools; the reader takes files from any writer.
pub fn write_safetensors(tensors: &[(&str, &[f32])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, v) in tensors {
        let end = offset + 4 * v.len() as u64;
        header.insert(
            (*name).to_string(),
            json!({"dtype": "F32", "shape": [v.len()], "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut head = Value::Object(header).to_string().into_bytes();
    // The format pads the header with spaces to a multiple of 8.
    while head.len() % 8 != 0 {
        head.push(b' ');
    }
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    for (_, v) in tensors {
        for x in *v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(tensors: &[(&str, &[f32])]) -> Result<SteerFile, SteerError> {
        SteerFile::parse(&write_safetensors(tensors))
    }

    fn refused(r: Result<SteerFile, SteerError>, want: &str) {
        let e = r.expect_err("must be refused");
        assert!(e.0.contains(want), "{e}");
    }

    #[test]
    fn a_good_file_reads_with_its_sha256() {
        let bytes = write_safetensors(&[("talker.layer.3", &[1.0, 2.0]), ("talker.layer.0", &[0.5, -0.5])]);
        let f = SteerFile::parse(&bytes).unwrap();
        assert_eq!(f.sha256, hex(&sha256(&bytes)));
        assert_eq!(f.layers.keys().copied().collect::<Vec<_>>(), [0, 3]);
        assert_eq!(f.layers[&3], [1.0, 2.0]);
    }

    #[test]
    fn a_name_that_is_not_a_talker_layer_is_refused() {
        for name in [
            "talker.layer.x",
            "talker.layer.",
            "talker.layer.01",
            "talker.layer.-1",
            "code_pred.layer.0",
            "model.layers.3",
        ] {
            refused(file(&[(name, &[1.0])]), "is not named talker.layer.<i>");
        }
    }

    #[test]
    fn a_value_that_is_not_finite_is_refused() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            refused(file(&[("talker.layer.1", &[0.0, bad])]), "not finite");
        }
    }

    /// A one-tensor file whose JSON header is edited by `edit`.
    fn with_header(edit: impl Fn(&str) -> String) -> Vec<u8> {
        let bytes = write_safetensors(&[("talker.layer.1", &[1.0, 2.0])]);
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let head = edit(std::str::from_utf8(&bytes[8..8 + n]).unwrap());
        let mut out = (head.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(head.as_bytes());
        out.extend_from_slice(&bytes[8 + n..]);
        out
    }

    /// A header naming one tensor twice is refused, rather than one of the two vectors being dropped. A nested key
    /// that repeats a top-level name is not a repeat.
    #[test]
    fn a_tensor_named_twice_is_refused() {
        let twice = with_header(|h| {
            let entry = h.trim_end().strip_prefix('{').unwrap().strip_suffix('}').unwrap();
            format!("{{{entry},{}}}", entry.replace("[0,8]", "[0,4]").replace("[2]", "[1]"))
        });
        refused(
            SteerFile::parse(&twice),
            "names tensor \"talker.layer.1\" more than once",
        );
        let nested = with_header(|h| {
            let body = h.trim_end().strip_suffix('}').unwrap();
            format!("{body},\"__metadata__\":{{\"talker.layer.1\":\"a note\"}}}}")
        });
        assert!(SteerFile::parse(&nested).is_ok());
    }

    #[test]
    fn a_scale_that_overflows_f32_is_refused() {
        let f = file(&[("talker.layer.1", &[1.0, 1e10])]).unwrap();
        // 1e39 is a finite f64 and an infinite f32.
        let e = f.select(None, 1e39).unwrap_err();
        assert!(e.0.contains("not a finite number in f32"), "{e}");
        // A finite f32 scale whose product with an element overflows.
        let e = f.select(None, 1e30).unwrap_err();
        assert!(e.0.contains("times element 1 of layer 1"), "{e}");
        assert!(f.select(None, 1e20).is_ok());
        // The model check refuses it too, for steering not made by `select`.
        let mut s = f.select(None, 1.0).unwrap();
        s.scale = 1e39;
        assert!(s.check(4, 2).unwrap_err().0.contains("not a finite number in f32"));
    }

    #[test]
    fn a_tensor_that_is_not_one_dimensional_f32_is_refused() {
        refused(SteerFile::parse(&with_header(|h| h.replace("F32", "F16"))), "not F32");
        refused(
            SteerFile::parse(&with_header(|h| h.replace("[2]", "[1,2]"))),
            "one-dimensional",
        );
        // The unedited file reads, so each refusal above is the edit's.
        assert!(SteerFile::parse(&with_header(str::to_string)).is_ok());
    }

    #[test]
    fn a_file_that_is_not_safetensors_is_refused() {
        refused(SteerFile::parse(b"abc"), "not a safetensors file");
        refused(SteerFile::parse(&[255; 16]), "not a safetensors file");
        let mut b = 4u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"nope");
        refused(SteerFile::parse(&b), "not JSON");
        refused(file(&[]), "holds no talker.layer.<i> tensor");
        // A byte range past the data.
        let mut bytes = write_safetensors(&[("talker.layer.1", &[1.0, 2.0])]);
        bytes.truncate(bytes.len() - 4);
        refused(SteerFile::parse(&bytes), "data_offsets");
    }

    #[test]
    fn an_unreadable_file_is_refused_with_its_path() {
        let e = SteerFile::read(Path::new("no-such-steering-file.safetensors")).unwrap_err();
        assert!(e.0.contains("no-such-steering-file.safetensors"), "{e}");
    }

    #[test]
    fn selection_refuses_a_missing_or_repeated_layer_and_keeps_layer_order() {
        let f = file(&[("talker.layer.2", &[1.0]), ("talker.layer.5", &[2.0])]).unwrap();
        let all = f.select(None, 1.0).unwrap();
        assert_eq!(all.layer_indices(), [2, 5]);
        let one = f.select(Some(&[5, 2]), 0.5).unwrap();
        assert_eq!(one.layer_indices(), [2, 5]);
        assert_eq!(one.scaled(), [(2, vec![0.5]), (5, vec![1.0])]);
        let e = f.select(Some(&[3]), 1.0).unwrap_err();
        assert!(
            e.0.contains("layer 3 is not in the steering file") && e.0.contains("2,5"),
            "{e}"
        );
        let e = f.select(Some(&[2, 2]), 1.0).unwrap_err();
        assert!(e.0.contains("named twice"), "{e}");
        assert!(f.select(Some(&[]), 1.0).is_err());
        for bad in [f64::NAN, f64::INFINITY] {
            assert!(f.select(None, bad).unwrap_err().0.contains("not a finite number"));
        }
    }

    #[test]
    fn a_layer_list_parses_and_refuses_repeats_and_junk() {
        assert_eq!(parse_layers("23,24,25").unwrap(), [23, 24, 25]);
        assert_eq!(parse_layers(" 7 ").unwrap(), [7]);
        assert!(parse_layers("1,1").unwrap_err().0.contains("named twice"));
        for bad in ["", "1,", "a", "-1", "1.5"] {
            assert!(parse_layers(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_model_check_refuses_a_wrong_length_or_an_out_of_range_layer() {
        let f = file(&[("talker.layer.1", &[1.0, 2.0])]).unwrap();
        let s = f.select(None, 1.0).unwrap();
        assert!(s.check(2, 2).is_ok());
        let e = s.check(1, 2).unwrap_err();
        assert!(e.0.contains("out of range") && e.0.contains("0 to 0"), "{e}");
        let e = s.check(4, 3).unwrap_err();
        assert!(e.0.contains("has 2 values") && e.0.contains("hidden size is 3"), "{e}");
    }

    #[test]
    fn the_record_names_the_file_the_layers_and_the_scale() {
        let s = file(&[("talker.layer.4", &[1.0]), ("talker.layer.9", &[1.0])])
            .unwrap()
            .select(None, 0.25)
            .unwrap();
        let r = s.record();
        assert_eq!(r["sha256"], s.sha256.as_str());
        assert_eq!(r["layers"], json!([4, 9]));
        assert_eq!(r["scale"], 0.25);
    }

    #[test]
    fn an_engine_that_cannot_steer_refuses_a_steered_request() {
        let s = file(&[("talker.layer.0", &[1.0])]).unwrap().select(None, 1.0).unwrap();
        let mut req = ChunkRequest {
            text: "好。",
            seed: 1,
            chunk: 0,
            draw: 0,
            instruction: None,
            steer: None,
        };
        assert!(refuse_steer(&req, "kitten-tts-2").is_ok());
        req.steer = Some(&s);
        let e = refuse_steer(&req, "kitten-tts-2").unwrap_err();
        assert!(
            e.0.contains("kitten-tts-2") && e.0.contains("cannot apply steering"),
            "{e}"
        );
    }
}
