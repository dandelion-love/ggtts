//! ggtts-kitten-convert: a Kitten-TTS 2 language model to a ggtts GGUF, then the readback check.
//!
//! ```text
//! ggtts-kitten-convert --repo DIR [--weights FILE] --out FILE.gguf [--layout q4_0|tq2_0|tq1_0] [--plant group256|absmean]
//! ggtts-kitten-convert --repo DIR [--weights FILE] --verify FILE.gguf
//! ggtts-kitten-convert --s3gen s3gen_meanflow.safetensors [--pyannote speaker/model.safetensors] --out FILE.gguf
//! ```
//!
//! `--repo` is a Kitten-TTS 2 model directory (its `config.json` and `lm/config.json` are read). `--weights` is the
//! language model's safetensors in Kitten's key layout, by default `<repo>/lm/model.safetensors`; a QAT export's
//! `ternary_bf16.safetensors` has the same layout. The readback check runs after every conversion and the exit
//! status is its verdict. `--plant` writes a deliberately wrong rule without the converter's own refusal, so the
//! readback check can be seen to fail.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ggtts_kitten::convert;
use ggtts_kitten::gguf::Gguf;
use ggtts_kitten::safetensors::SafeTensors;
use ggtts_kitten::ternary::{Rule, ScaleRule};

const USAGE: &str = "usage: ggtts-kitten-convert --repo DIR [--weights FILE] (--out FILE.gguf [--layout q4_0|tq2_0|tq1_0] [--plant group256|absmean] | --verify FILE.gguf)
       ggtts-kitten-convert --s3gen s3gen_meanflow.safetensors [--pyannote speaker/model.safetensors] --out FILE.gguf
       ggtts-kitten-convert --version

--repo: a KittenML/kitten-tts-2 download (config.json, lm/config.json, lm/model.safetensors, lm/tokenizer.json).
  Writes the language model GGUF for `ggtts-render --model`, with the tokenizer embedded, then checks it by readback.
--layout: how the ternary matrices are stored, each exact: q4_0 (the default, 4.5 bits a weight), tq2_0 (padded,
  4.125) or tq1_0 (padded TQ1_0, `tq1_0-pad128`, 3.375). Each 128-weight group keeps its own scale; the older
  256-weight TQ1_0 layout gives two groups one scale, is lossy for this model, and is read but never written.
--s3gen: ResembleAI/chatterbox-turbo's s3gen_meanflow.safetensors, with --pyannote the Kitten download's
  speaker/model.safetensors. Writes the decoder GGUF for `ggtts-render --codec` and `ggtts-kitten prepare --s3gguf`.";

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn read_json(p: &std::path::Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let (mut repo, mut weights, mut out, mut verify_only, mut plant) = (None, None, None, None, None);
    let mut s3gen: Option<PathBuf> = None;
    let mut pyannote: Option<PathBuf> = None;
    // Q4_0 by default: the one layout that passes the ternary arm's parity bar (DESIGN.md, Measured).
    let mut layout = ggtts_kitten::ternary::Layout::Q4;
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--repo" => repo = Some(PathBuf::from(val()?)),
            "--weights" => weights = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--verify" => verify_only = Some(PathBuf::from(val()?)),
            "--plant" => plant = Some(val()?),
            "--s3gen" => s3gen = Some(PathBuf::from(val()?)),
            "--pyannote" => pyannote = Some(PathBuf::from(val()?)),
            "--layout" => {
                let v = val()?;
                layout = ggtts_kitten::ternary::Layout::parse(&v).ok_or(format!("unknown --layout {v}"))?;
            }
            "--version" | "-V" => {
                for line in ggtts_core::version_lines("ggtts-kitten-convert") {
                    println!("{line}");
                }
                return Ok(true);
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(true);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if let Some(src) = s3gen {
        let out = out.ok_or("--s3gen needs --out")?;
        let st = SafeTensors::open(&src)?;
        let py = pyannote.map(|p| SafeTensors::open(&p)).transpose()?;
        let n = ggtts_kitten::s3convert::convert(&st, py.as_ref(), &out)?;
        println!(
            "wrote {}: {n} tensors (S3Gen flow, vocoder, tokenizer and CAMPPlus; pyannote if given)",
            out.display()
        );
        return Ok(true);
    }
    let repo = repo.ok_or("--repo is required")?;
    let weights = weights.unwrap_or_else(|| repo.join("lm").join("model.safetensors"));
    let (rule, strict) = match plant.as_deref() {
        None => (Rule::KITTEN, true),
        Some("group256") => (
            Rule {
                group: 256,
                scale: ScaleRule::AbsMax,
            },
            false,
        ),
        Some("absmean") => (
            Rule {
                group: 128,
                scale: ScaleRule::AbsMean,
            },
            false,
        ),
        Some(p) => return Err(format!("unknown planted fault {p}")),
    };
    let t0 = Instant::now();
    let st = SafeTensors::open(&weights)?;
    eprintln!(
        "read {} tensors from {} in {:.1}s",
        st.tensors.len(),
        weights.display(),
        t0.elapsed().as_secs_f64()
    );
    let gguf_path = match (out, verify_only) {
        (Some(out), None) => {
            let model = read_json(&repo.join("config.json"))?;
            let lm = read_json(&repo.join("lm").join("config.json"))?;
            let source = weights
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let t1 = Instant::now();
            // Required: a GGUF without its tokenizer converts and verifies, and fails only later, in the renderer.
            let tok_path = repo.join("lm").join("tokenizer.json");
            let tok = std::fs::read_to_string(&tok_path).map_err(|e| {
                format!(
                    "{}: {e} (the tokenizer is embedded in the GGUF; ggtts-render refuses a GGUF without it)",
                    tok_path.display()
                )
            })?;
            let s = convert::convert(&st, &model, &lm, &source, &out, rule, strict, layout, Some(&tok))?;
            eprintln!(
                "wrote {}: {} ternary matrices ({} weights) as {}, {} bf16, {} f32, {} bytes of tensors, in {:.1}s",
                out.display(),
                s.ternary,
                s.ternary_weights,
                layout.name(),
                s.bf16,
                s.f32,
                s.bytes,
                t1.elapsed().as_secs_f64()
            );
            out
        }
        (None, Some(v)) => v,
        _ => return Err("give exactly one of --out and --verify".into()),
    };
    let t2 = Instant::now();
    let g = Gguf::open(&gguf_path)?;
    match convert::verify(&st, &g) {
        Ok(v) => {
            println!(
                "READBACK PASS: {} ternary matrices, {} groups of 128 equal to the source trit for trit and scale bit for bit; {} other tensors equal ({:.1}s)",
                v.ternary,
                v.groups,
                v.others,
                t2.elapsed().as_secs_f64()
            );
            Ok(true)
        }
        Err(e) => {
            println!("READBACK FAIL: {e}");
            Ok(false)
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            usage()
        }
    }
}
