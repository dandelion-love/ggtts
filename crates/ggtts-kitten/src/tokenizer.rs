//! The model's text tokenizer: Qwen2's byte-level BPE, read from the model's `lm/tokenizer.json`.
//!
//! The pipeline is the file's own: added tokens matched literally first; then the split regex
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
//! written out by hand below; then GPT-2's byte-to-unicode map; then BPE by merge rank. Two simplifications, both
//! no-ops on ordinary text and both checked against the Hugging Face tokenizer's ids: `\p{L}` and `\p{N}` are
//! Rust's `char::is_alphabetic` and `char::is_numeric`, and the NFC normalizer is not applied (give NFC text).

use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// Added tokens by first char, longest first.
    added: HashMap<char, Vec<(String, u32)>>,
    byte_char: [char; 256],
}

fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

fn is_l(c: char) -> bool {
    c.is_alphabetic()
}
fn is_n(c: char) -> bool {
    c.is_numeric()
}

/// The split regex, alternative by alternative, leftmost first.
pub fn pre_split(s: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = s.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { cs[i].0 } else { s.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let mut end = None;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' {
            let low: String = cs[i + 1..n.min(i + 3)]
                .iter()
                .map(|x| x.1.to_ascii_lowercase())
                .collect();
            for suf in ["re", "ve", "ll", "s", "t", "m", "d"] {
                if low.starts_with(suf) {
                    end = Some(i + 1 + suf.len());
                    break;
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if end.is_none() {
            let start = if is_l(c) {
                Some(i)
            } else if c != '\r' && c != '\n' && !is_n(c) && i + 1 < n && is_l(cs[i + 1].1) {
                Some(i + 1)
            } else {
                None
            };
            if let Some(mut j) = start {
                while j < n && is_l(cs[j].1) {
                    j += 1;
                }
                end = Some(j);
            }
        }
        // \p{N}
        if end.is_none() && is_n(c) {
            end = Some(i + 1);
        }
        // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        if end.is_none() {
            let other = |x: char| !x.is_whitespace() && !is_l(x) && !is_n(x);
            let j0 = if c == ' ' && i + 1 < n && other(cs[i + 1].1) {
                i + 1
            } else {
                i
            };
            if j0 < n && other(cs[j0].1) {
                let mut j = j0;
                while j < n && other(cs[j].1) {
                    j += 1;
                }
                while j < n && (cs[j].1 == '\r' || cs[j].1 == '\n') {
                    j += 1;
                }
                end = Some(j);
            }
        }
        if end.is_none() && c.is_whitespace() {
            let mut j = i;
            while j < n && cs[j].1.is_whitespace() {
                j += 1;
            }
            // \s*[\r\n]+ : up to the last newline in the run.
            if let Some(last_nl) = (i..j).rev().find(|&k| cs[k].1 == '\r' || cs[k].1 == '\n') {
                end = Some(last_nl + 1);
            } else if j == n || j - i == 1 {
                // \s+(?!\S) at the end of the text, or \s+ for a single space before a non-space.
                end = Some(j);
            } else {
                // \s+(?!\S): leave the last whitespace for the next token.
                end = Some(j - 1);
            }
        }
        let e = end.unwrap_or(i + 1);
        out.push(&s[at(i)..at(e)]);
        i = e;
    }
    out
}

impl Tokenizer {
    pub fn open(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j: serde_json::Value = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        let model = &j["model"];
        if model["type"] != "BPE" {
            return Err("tokenizer.json: not a BPE model".into());
        }
        let vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or("tokenizer.json: vocab")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
            .collect();
        let mut ranks = HashMap::new();
        for (r, m) in model["merges"]
            .as_array()
            .ok_or("tokenizer.json: merges")?
            .iter()
            .enumerate()
        {
            let (a, b) = match m {
                serde_json::Value::Array(p) => (
                    p[0].as_str().unwrap_or("").to_string(),
                    p[1].as_str().unwrap_or("").to_string(),
                ),
                serde_json::Value::String(p) => {
                    let (a, b) = p.split_once(' ').ok_or("tokenizer.json: merge")?;
                    (a.to_string(), b.to_string())
                }
                _ => return Err("tokenizer.json: merge".into()),
            };
            ranks.insert((a, b), r);
        }
        let mut added: HashMap<char, Vec<(String, u32)>> = HashMap::new();
        for t in j["added_tokens"].as_array().ok_or("tokenizer.json: added_tokens")? {
            let content = t["content"].as_str().unwrap_or("").to_string();
            if let Some(c) = content.chars().next() {
                added
                    .entry(c)
                    .or_default()
                    .push((content, t["id"].as_u64().unwrap_or(0) as u32));
            }
        }
        for v in added.values_mut() {
            v.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        }
        Ok(Tokenizer {
            vocab,
            ranks,
            added,
            byte_char: bytes_to_unicode(),
        })
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) -> Result<(), String> {
        let mut syms: Vec<String> = piece.bytes().map(|b| self.byte_char[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        for s in syms {
            out.push(*self.vocab.get(&s).ok_or_else(|| format!("no vocab entry for {s:?}"))?);
        }
        Ok(())
    }

    fn encode_plain(&self, s: &str, out: &mut Vec<u32>) -> Result<(), String> {
        for piece in pre_split(s) {
            self.bpe(piece, out)?;
        }
        Ok(())
    }

    /// Token ids for `text`, with no special tokens added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        let mut plain_start = 0;
        let mut i = 0;
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            let hit = self
                .added
                .get(&c)
                .and_then(|v| v.iter().find(|(t, _)| text[i..].starts_with(t.as_str())));
            if let Some((t, id)) = hit {
                self.encode_plain(&text[plain_start..i], &mut out)?;
                out.push(*id);
                i += t.len();
                plain_start = i;
            } else {
                i += c.len_utf8();
            }
        }
        self.encode_plain(&text[plain_start..], &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_matches_the_regex_on_common_shapes() {
        assert_eq!(pre_split("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pre_split("it's 42!"), vec!["it", "'s", " ", "4", "2", "!"]);
        assert_eq!(pre_split("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pre_split("end.\n\nNext"), vec!["end", ".\n\n", "Next"]);
        assert_eq!(pre_split("x   "), vec!["x", "   "]);
        assert_eq!(pre_split("(((never)))"), vec!["(((", "never", ")))"]);
    }

    #[test]
    fn byte_map_is_gpt2s() {
        let m = bytes_to_unicode();
        assert_eq!(m[b' ' as usize], 'Ġ');
        assert_eq!(m[b'\n' as usize], 'Ċ');
        assert_eq!(m[b'A' as usize], 'A');
    }
}
