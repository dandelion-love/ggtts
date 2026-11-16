//! Map final rendered PCM and kept draws to the offline feedback page.
use crate::flac::{encode_flac, to_i16};
use crate::pipeline::Read;
use crate::seed::{hex, sha256};
use serde_json::{json, Value};

const TEMPLATE: &str = include_str!("../../../feedback/page.html");
const MARKER: &str = "<!-- GGTTS_INPUT -->";

pub struct FeedbackBundle {
    pub input: Value,
    pub provenance: Value,
}
fn valid_id(id: &str) -> bool {
    let mut chars = id.bytes();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id.len() <= 128
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}
fn b64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(A[((n >> 18) & 63) as usize] as char);
        out.push(A[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            A[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            A[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
/// Python's ensure_ascii JSON serialization, including UTF-16 surrogate pairs.
fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            c if c <= '\u{001f}' || !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
fn canonical(value: &Value, out: &mut String) -> Result<(), String> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => json_string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(v, out)?;
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort_unstable();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_string(k, out);
                out.push(':');
                canonical(&m[*k], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}
pub fn input_digest(input: &Value) -> Result<String, String> {
    let mut data = String::new();
    canonical(input, &mut data)?;
    Ok(hex(&sha256(data.as_bytes())))
}
fn valid_audio(uri: &str) -> bool {
    let Some(data) = uri.strip_prefix("data:audio/flac;base64,") else {
        return false;
    };
    if data.is_empty() || data.len() % 4 != 0 {
        return false;
    }
    let mut bytes = Vec::with_capacity(data.len() / 4 * 3);
    let chunks = data.as_bytes().chunks_exact(4);
    let n = chunks.len();
    for (i, group) in chunks.enumerate() {
        let mut values = [0u8; 4];
        let mut pad = 0;
        for (j, c) in group.iter().enumerate() {
            let v = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' if i + 1 == n && j >= 2 => {
                    pad += 1;
                    0
                }
                _ => return false,
            };
            if pad > 0 && *c != b'=' {
                return false;
            }
            values[j] = v;
        }
        if pad > 2 {
            return false;
        }
        let v = (u32::from(values[0]) << 18)
            | (u32::from(values[1]) << 12)
            | (u32::from(values[2]) << 6)
            | u32::from(values[3]);
        bytes.push((v >> 16) as u8);
        if pad < 2 {
            bytes.push((v >> 8) as u8);
        }
        if pad < 1 {
            bytes.push(v as u8);
        }
    }
    if bytes.len() < 42 || &bytes[..4] != b"fLaC" || bytes[4] & 0x7f != 0 || bytes[5..8] != [0, 0, 34] {
        return false;
    }
    let stream = u64::from_be_bytes(bytes[18..26].try_into().unwrap());
    stream >> 44 != 0 && stream & ((1u64 << 36) - 1) != 0
}
/// Build page HTML for a validated ggtts-feedback-input/1 document.
pub fn render_html(input: &Value) -> Result<String, String> {
    let map = input.as_object().ok_or("invalid feedback input")?;
    if map.len() != 3
        || input.get("format").and_then(Value::as_str) != Some("ggtts-feedback-input/1")
        || !input.get("id").and_then(Value::as_str).is_some_and(valid_id)
    {
        return Err("invalid feedback input".into());
    }
    let items = input
        .get("items")
        .and_then(Value::as_array)
        .filter(|x| !x.is_empty())
        .ok_or("missing feedback items")?;
    let mut ids = std::collections::HashSet::new();
    for item in items {
        let object = item.as_object().ok_or("invalid feedback item")?;
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| valid_id(id))
            .ok_or("invalid item id")?;
        if object.len() != 3
            || !ids.insert(id)
            || item.get("text").and_then(Value::as_str).is_none_or(str::is_empty)
            || !item.get("audio").and_then(Value::as_str).is_some_and(valid_audio)
        {
            return Err("invalid feedback item".into());
        }
    }
    if TEMPLATE.matches(MARKER).count() != 1 {
        return Err("invalid feedback template".into());
    }
    let mut data = String::new();
    canonical(input, &mut data)?;
    let digest = hex(&sha256(data.as_bytes()));
    let embedded=format!("<meta name=\"feedback-input-sha256\" content=\"{digest}\"><script id=\"feedback-input\" type=\"application/json\">{data}</script>");
    Ok(TEMPLATE.replacen(MARKER, &embedded, 1))
}
/// Map final, levelled Read.samples, never decoded/pre-trim draw samples.
pub fn from_read(read: &Read, page_id: &str, provenance: Value) -> Result<FeedbackBundle, String> {
    if !valid_id(page_id) {
        return Err("invalid page id".into());
    }
    if provenance.as_object().is_none_or(|m| m.is_empty()) {
        return Err("missing render provenance".into());
    }
    if read.sample_rate == 0 || read.chunks.is_empty() || read.samples.is_empty() {
        return Err("missing rendered audio or chunks".into());
    }
    if read.samples.iter().any(|v| !v.is_finite()) {
        return Err("non-finite final audio".into());
    }
    let full: Vec<i16> = read.samples.iter().map(|&x| to_i16(x)).collect();
    let full_flac = encode_flac(&full, read.sample_rate).map_err(|e| e.to_string())?;
    let mut items = Vec::new();
    let mut records = Vec::new();
    let mut end = 0;
    for (i, c) in read.chunks.iter().enumerate() {
        let (start, stop) = c.span;
        if c.i != i
            || c.text.is_empty()
            || start < end
            || (i == 0 && start != 0)
            || start > full.len()
            || read.samples[end..start].iter().any(|v| *v != 0.0)
            || start >= stop
            || stop > full.len()
            || c.stop == crate::engine::Stop::Abort
            || c.decoded_sample_rate != read.sample_rate
            || stop - start > c.decoded_samples
        {
            return Err(format!("inconsistent chunk {i}"));
        }
        let audio = encode_flac(&full[start..stop], read.sample_rate).map_err(|e| e.to_string())?;
        let id = format!("chunk-{i}");
        records.push(
            json!({"id":id,"chunk":i,"draw":c.redraws,"seed":c.seed,"start":start,"end":stop,
            "text_sha256":hex(&sha256(c.text.as_bytes())),"audio_sha256":hex(&sha256(&audio)),
            "stop":c.stop.as_str()}),
        );
        items.push(json!({"id":id,"text":c.text,"audio":format!("data:audio/flac;base64,{}",b64(&audio))}));
        end = stop;
    }
    if end != full.len() {
        return Err("unmapped trailing audio".into());
    }
    let input = json!({"format":"ggtts-feedback-input/1","id":page_id,"items":items});
    let digest = input_digest(&input)?;
    Ok(FeedbackBundle {
        input,
        provenance: json!({
            "format":"ggtts-feedback-provenance/1","id":page_id,"input_sha256":digest,
            "output_sha256":hex(&sha256(&full_flac)),"sample_rate":read.sample_rate,
            "output_samples":full.len(),"render":provenance,"chunks":records
        }),
    })
}
