//! A template event instrument: what it measures from a trace, and the envelope it writes.

use std::collections::BTreeMap;

use ggtts_core::trace::{parse, TraceError};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// The instrument's name and version, as its envelope reports them.
pub const NAME: &str = "draw-lengths";
pub const VERSION: &str = "0.1.0";

/// What one trace measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub draws: u64,
    pub longest_draw_frames: Option<u64>,
    pub frames_total: u64,
    /// The stream ended with `end`.
    pub complete: bool,
    /// From the header, for the envelope's provenance.
    pub header: Map<String, Value>,
    /// Every draw's `(chunk, seed)`, for the envelope's seeds.
    pub seeds: Vec<(u64, u64)>,
}

/// Measure a trace. Refuses a stream that is not `ggtts-trace/1`.
pub fn measure(trace: &str) -> Result<Measured, TraceError> {
    let lines = parse(trace)?;
    let mut longest: Option<u64> = None;
    let (mut draws, mut total) = (0u64, 0u64);
    let mut seeds = Vec::new();
    for m in &lines {
        match m["ev"].as_str() {
            Some("chunk_end") => {
                let f = m.get("frames").and_then(Value::as_u64).unwrap_or(0);
                draws += 1;
                total += f;
                longest = Some(longest.map_or(f, |l| l.max(f)));
            }
            Some("chunk_begin") => {
                if let (Some(c), Some(s)) = (m["chunk"].as_u64(), m["seed"].as_u64()) {
                    seeds.push((c, s));
                }
            }
            _ => {}
        }
    }
    Ok(Measured {
        draws,
        longest_draw_frames: longest,
        frames_total: total,
        complete: lines.last().is_some_and(|m| m["ev"] == "end"),
        header: lines[0].clone(),
        seeds,
    })
}

/// The sha256 of `bytes`, lower-case hex.
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Who the envelope belongs to and what it was given.
pub struct Context {
    pub name: String,
    pub run: String,
    pub render: String,
    pub config: BTreeMap<String, String>,
    /// `(file name, sha256)` of each input read.
    pub inputs: Vec<(String, String)>,
}

/// The `ggtts-report/1` envelope for `m`, or for a failure when `m` is an error.
pub fn envelope(ctx: &Context, m: Result<&Measured, String>) -> Value {
    let s = |v: Option<&Value>| v.and_then(Value::as_str).filter(|x| !x.is_empty()).map(String::from);
    let (header, status, error) = match &m {
        Ok(m) if m.complete => (Some(&m.header), "ok", None),
        Ok(m) => (
            Some(&m.header),
            "truncated",
            Some("the stream has no end event".to_string()),
        ),
        Err(e) => (None, "failed", Some(e.clone())),
    };
    let metric = |name: &str, value: Value, direction: &str| json!({"name": name, "value": value, "unit": "1", "direction": direction, "render": ctx.render});
    let metrics: Vec<Value> = match &m {
        Ok(m) => vec![
            metric("draws", json!(m.draws), "neutral"),
            metric("longest_draw_frames", json!(m.longest_draw_frames), "lower"),
            metric("frames_total", json!(m.frames_total), "neutral"),
        ],
        Err(_) => Vec::new(),
    };
    let seeds: Vec<Value> = match &m {
        Ok(m) => m
            .seeds
            .iter()
            .map(|(c, s)| json!({"render": ctx.render, "chunk": c, "seed": s}))
            .collect(),
        Err(_) => Vec::new(),
    };
    let inputs: Vec<Value> = ctx
        .inputs
        .iter()
        .map(|(p, h)| json!({"path": p, "sha256": h}))
        .collect();
    let model = header
        .and_then(|h| h.get("model_sha256"))
        .and_then(Value::as_str)
        .map(String::from);
    let mut e = json!({
        "format": "ggtts-report/1",
        "name": ctx.name,
        "version": VERSION,
        "identity": {"run": ctx.run, "instrument": ctx.name},
        "provenance": {
            "config": ctx.config,
            "ggtts": s(header.and_then(|h| h.get("ggtts"))),
            "engine": s(header.and_then(|h| h.get("engine"))),
            "backend": s(header.and_then(|h| h.get("backend"))),
            "model_sha256": model,
            "seeds": seeds,
            "inputs": inputs,
            "audio": "n/a",
        },
        "hooks": ["chunk_begin", "chunk_end", "end"],
        "status": status,
        "metrics": metrics,
        "artifacts": [],
        "log": null,
    });
    if let Some(err) = error {
        e["error"] = json!(err);
    }
    e
}
