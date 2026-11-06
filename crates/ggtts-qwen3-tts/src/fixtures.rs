//! Reading codec-token files: the reference CLI's `--dump-generated-codes` format,
//! `{"frames": F, "codebooks": C, "codes": [...]}`, frame-major, and the golden fixtures in `fixtures/cpu/`
//! (format `ggtts-qwen3-tts-golden/1`, written by `scripts/record_fixtures.sh`).

use std::path::Path;

use serde_json::Value;

/// One golden fixture: a request, the reference's codec tokens for it, and the reference decoder's samples as a
/// hash.
#[derive(Debug, Clone)]
pub struct Golden {
    pub name: String,
    pub text: String,
    pub speaker: String,
    pub language: String,
    pub seed: u64,
    pub max_tokens: u32,
    pub talker_sha256: String,
    pub codec_sha256: String,
    pub codebooks: usize,
    /// Frame-major: `codes[f * codebooks + c]`.
    pub codes: Vec<i32>,
    pub sample_count: usize,
    pub samples_sha256: String,
    /// The CPU thread count the reference ran with (`reference.threads`). The checks run with the same count, since
    /// a different count can change how ggml splits a reduction, and so the bits.
    pub threads: u32,
}

impl Golden {
    pub fn frames(&self) -> usize {
        self.codes.len() / self.codebooks
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if v["format"] != "ggtts-qwen3-tts-golden/1" {
            return Err(format!("unknown format {}", v["format"]));
        }
        let s = |p: &str| -> Result<String, String> {
            v.pointer(p)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or(format!("missing {p}"))
        };
        let n = |p: &str| -> Result<u64, String> { v.pointer(p).and_then(Value::as_u64).ok_or(format!("missing {p}")) };
        let codes = v["codes"]
            .as_array()
            .ok_or("missing codes")?
            .iter()
            .map(|c| c.as_i64().map(|c| c as i32).ok_or("bad code"))
            .collect::<Result<Vec<_>, _>>()?;
        let codebooks = n("/codebooks")? as usize;
        let frames = n("/frames")? as usize;
        if codebooks == 0 || codes.len() != frames * codebooks {
            return Err(format!("{} codes for {frames} frames of {codebooks}", codes.len()));
        }
        Ok(Self {
            name: s("/name")?,
            text: s("/text")?,
            speaker: s("/request/speaker")?,
            language: s("/request/language")?,
            seed: n("/request/seed")?,
            max_tokens: n("/request/max_tokens")? as u32,
            talker_sha256: s("/talker_sha256")?,
            codec_sha256: s("/codec_sha256")?,
            codebooks,
            codes,
            sample_count: n("/samples/count")? as usize,
            samples_sha256: s("/samples/sha256_f32le")?,
            threads: n("/reference/threads")? as u32,
        })
    }

    /// Every `*.json` in `dir`, sorted by file name.
    pub fn load_all(dir: &Path) -> Result<Vec<Self>, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|p| {
                let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Self::parse(&t).map_err(|e| format!("{}: {e}", p.display()))
            })
            .collect()
    }
}

/// The one thread count a set of fixtures was recorded with. Refuses an empty set, or one that mixes counts.
pub fn recorded_threads(goldens: &[Golden]) -> Result<i32, String> {
    let first = goldens.first().ok_or("no fixtures")?.threads;
    if let Some(g) = goldens.iter().find(|g| g.threads != first) {
        return Err(format!(
            "{} was recorded with {} threads and {} with {first}: re-record the set at one count",
            g.name, g.threads, goldens[0].name
        ));
    }
    i32::try_from(first).map_err(|_| format!("{first} threads"))
}

/// Returns `(codebooks, codes)`, checking that the count is `frames * codebooks`.
pub fn read_codes_json(text: &str) -> Result<(usize, Vec<i32>), String> {
    let field = |key: &str| -> Result<usize, String> {
        let k = text.find(&format!("\"{key}\"")).ok_or(format!("no {key}"))?;
        let rest = &text[k + key.len() + 2..];
        let rest = rest
            .trim_start()
            .strip_prefix(':')
            .ok_or(format!("bad {key}"))?
            .trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse().map_err(|_| format!("bad {key}"))
    };
    let frames = field("frames")?;
    let codebooks = field("codebooks")?;
    let k = text.find("\"codes\"").ok_or("no codes")?;
    let open = k + text[k..].find('[').ok_or("no codes array")?;
    let close = open + text[open..].find(']').ok_or("unterminated codes array")?;
    let codes = text[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>().map_err(|_| format!("bad code {s:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    if codebooks == 0 || codes.len() != frames * codebooks {
        return Err(format!("{} codes for {frames} frames of {codebooks}", codes.len()));
    }
    Ok((codebooks, codes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_dump_format() {
        let t = "{\n  \"frames\": 2,\n  \"codebooks\": 2,\n  \"codes\": [\n    1,2,\n    3,4\n  ]\n}\n";
        assert_eq!(read_codes_json(t).unwrap(), (2, vec![1, 2, 3, 4]));
    }

    #[test]
    fn refuses_a_short_array() {
        let t = "{\"frames\": 2, \"codebooks\": 2, \"codes\": [1,2,3]}";
        assert!(read_codes_json(t).is_err());
    }
}
