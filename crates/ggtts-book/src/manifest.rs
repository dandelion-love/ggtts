//! The chapter manifest: what a chapter's render planned and what it has finished, written so a killed run resumes.
//!
//! One `manifest.json` per chapter directory. Its body records the plan (book, chapter, seed root, profile, ceiling,
//! and every chunk's text, hash and seed), the engine, the render options that change the audio (the instruction and
//! the steering, only when there are any) and each chunk's state; the file carries the SHA-256 of its body beside it.
//! A manifest is written to a temporary file, flushed to disk and renamed over the old one, so a reader sees either
//! the old manifest or the new one, never half of one.
//!
//! A manifest is trusted only when it parses, its checksum matches, its format is known, and its plan is the plan
//! about to run. Anything else is refused with a [`ManifestError`] and the chapter starts again from its plan: a
//! wrong resume would splice another book's or another seed's audio into this one.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ggtts_core::seed::{hex, sha256};
use serde_json::{json, Value};

use crate::plan::ChapterPlan;

/// The manifest's format. A reader refuses any other.
pub const MANIFEST_FORMAT: &str = "ggtts-book-manifest/1";
/// The manifest's file name in a chapter directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// A finished chunk's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    /// The file's name in the chapter directory.
    pub file: String,
    pub sha256: String,
}

/// The draws the core chunk step made for a done chunk: which one was kept, how many were made, and the judge's verdict
/// on the kept one (`None` when it passed).
#[derive(Debug, Clone, PartialEq)]
pub struct Draws {
    /// The kept draw: 0 for the first, `n` for the `n`th redraw.
    pub kept: u32,
    /// The kept draw's termination reason, when recorded.
    pub stop: Option<String>,
    /// Draws made, the kept one included.
    pub count: u32,
    /// The judge's verdict on the kept draw (`runaway`, `truncated`, ...), or `None` when it passed.
    pub verdict: Option<String>,
    /// What a judge that reads back (`--judge`) read in each draw it read, with the draw's number: the transcript, the
    /// counts and the verdict the reading gave. Empty for the duration judge, and then not written.
    pub reads: Vec<Value>,
}

impl Draws {
    /// Whether a fresh render with `redraw` redraws allowed ends at the same draw: it stops at the first draw the judge
    /// passes, or at draw `redraw` either way. A chunk with no record (a render at redraw 0, or one written before
    /// draws were recorded) was never judged, so it holds only at redraw 0.
    pub fn holds_at(draws: Option<&Draws>, redraw: u32) -> bool {
        match draws {
            None => redraw == 0,
            Some(d) => d.kept <= redraw && (d.verdict.is_none() || d.kept == redraw),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleSpan {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterSpans {
    pub sample_rate: u32,
    pub sample_count: u64,
    pub chapter_sha256: String,
    pub chunks: Vec<SampleSpan>,
}

/// What a chunk renderer returns to [`run_chapter`]: the file's bytes and, when it judges its draws, the draws it made.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkOutput {
    pub bytes: Vec<u8>,
    pub draws: Option<Draws>,
}

impl From<Vec<u8>> for ChunkOutput {
    fn from(bytes: Vec<u8>) -> Self {
        ChunkOutput { bytes, draws: None }
    }
}

/// A chapter's manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub plan: ChapterPlan,
    /// One entry per planned chunk: `Some` once it is done.
    pub done: Vec<Option<Output>>,
    /// Draw identity and verdict for each done chunk when the renderer records them. New renders record even draw 0;
    /// older manifests without these optional fields remain readable.
    pub draws: Vec<Option<Draws>>,
    /// How long each done chunk took to render (generate, decode, trim and encode), in milliseconds. Recorded for
    /// reading back; never checked.
    pub render_ms: Vec<Option<u64>>,
    /// Kept draw termination reason, including when redraw is zero.
    pub kept_stop: Vec<Option<String>>,
    /// The chapter FLAC joined from these chunks, once written: its file name in the output directory and its
    /// SHA-256. `export` checks the file against it. Cleared whenever a chunk is rendered again.
    pub chapter: Option<Output>,
    /// Exact final joined-audio spans, absent for legacy manifests.
    pub chapter_spans: Option<ChapterSpans>,
}

/// Why a manifest is not trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The file could not be read.
    Io(String),
    /// Not JSON: a write cut short, most often.
    NotJson(String),
    /// Its checksum does not match its body.
    Checksum,
    /// A format this reader does not know, or a body missing a field.
    Format(String),
    /// Made for another book, chapter, seed root, profile, ceiling, engine, instruction or steering: names the field.
    Mismatch {
        field: &'static str,
        manifest: String,
        plan: String,
    },
    /// The chapter's text or seeds changed since the manifest was written: names the first chunk that differs.
    PlanChanged { index: usize },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "cannot read the manifest: {e}"),
            ManifestError::NotJson(e) => write!(f, "the manifest is not valid JSON (a write cut short?): {e}"),
            ManifestError::Checksum => write!(f, "the manifest's checksum does not match its contents"),
            ManifestError::Format(e) => write!(f, "the manifest's format is not {MANIFEST_FORMAT}: {e}"),
            ManifestError::Mismatch { field, manifest, plan } => {
                write!(f, "the manifest is for {field} {manifest:?}, not {plan:?}")
            }
            ManifestError::PlanChanged { index } => {
                write!(
                    f,
                    "the chapter's plan changed at chunk {index} since the manifest was written"
                )
            }
        }
    }
}

impl std::error::Error for ManifestError {}

impl Manifest {
    /// A manifest with nothing done.
    pub fn new(plan: ChapterPlan) -> Self {
        let done = vec![None; plan.chunks.len()];
        let render_ms = vec![None; plan.chunks.len()];
        let draws = vec![None; plan.chunks.len()];
        let kept_stop = vec![None; plan.chunks.len()];
        Manifest {
            plan,
            done,
            draws,
            render_ms,
            kept_stop,
            chapter: None,
            chapter_spans: None,
        }
    }

    /// Record a chapter file for legacy callers that have no joined-sample map.
    pub fn record_chapter(dir: &Path, plan: &ChapterPlan, file: &str, bytes: &[u8]) -> Result<(), String> {
        Self::record_chapter_inner(dir, plan, file, bytes, None)
    }

    /// Record the chapter file and exact final-audio spans in one atomic manifest replacement.
    pub fn record_chapter_with_spans(
        dir: &Path,
        plan: &ChapterPlan,
        file: &str,
        bytes: &[u8],
        sample_rate: u32,
        sample_count: u64,
        chunks: Vec<SampleSpan>,
    ) -> Result<(), String> {
        let spans = ChapterSpans {
            sample_rate,
            sample_count,
            chapter_sha256: hex(&sha256(bytes)),
            chunks,
        };
        validate_chapter_spans(&spans, plan.chunks.len())?;
        Self::record_chapter_inner(dir, plan, file, bytes, Some(spans))
    }

    fn record_chapter_inner(
        dir: &Path,
        plan: &ChapterPlan,
        file: &str,
        bytes: &[u8],
        spans: Option<ChapterSpans>,
    ) -> Result<(), String> {
        let mut m = Manifest::load(dir, plan).map_err(|e| e.to_string())?;
        if m.done.iter().any(Option::is_none) {
            return Err("the chapter is not fully rendered".to_string());
        }
        m.chapter = Some(Output {
            file: file.to_string(),
            sha256: hex(&sha256(bytes)),
        });
        m.chapter_spans = spans;
        m.save(dir).map_err(|e| e.to_string())
    }

    fn body(&self) -> Value {
        let p = &self.plan;
        let mut body = json!({
            "format": MANIFEST_FORMAT,
            "book_id": p.book_id,
            "chapter_id": p.chapter_id,
            "seed_root": p.seed_root,
            "profile": p.profile,
            "ceiling": p.ceiling,
            "engine": p.engine,
            "chapter_file": self.chapter.as_ref().map(|o| json!({"file": o.file, "sha256": o.sha256})),
            "chunks": p.chunks.iter().enumerate().map(|(i, c)| {
                let done = &self.done[i];
                let mut chunk = json!({
                    "index": c.index,
                    "paragraph": c.paragraph,
                    "chunk": c.chunk,
                    "text": c.text,
                    "text_sha256": c.text_sha256,
                    "seed": c.seed,
                    "state": if done.is_some() { "done" } else { "planned" },
                    "output": done.as_ref().map(|o| json!({"file": o.file, "sha256": o.sha256})),
                    "render_ms": self.render_ms[i],
                });
                if let Some(stop) = &self.kept_stop[i] {
                    chunk["kept_stop"] = stop.clone().into();
                }
                if let (Some(_), Some(d)) = (done, &self.draws[i]) {
                    chunk["kept_draw"] = d.kept.into();
                    chunk["draws"] = d.count.into();
                    chunk["verdict"] = d.verdict.clone().into();
                    if !d.reads.is_empty() {
                        chunk["reads"] = d.reads.clone().into();
                    }
                }
                chunk
            }).collect::<Vec<_>>(),
        });
        // The redraw limit is omitted at zero; kept-draw identity is recorded independently when available.
        if p.redraw > 0 {
            body["redraw"] = p.redraw.into();
        }
        // Written only when the render has an instruction or steering, so a manifest of a render with neither is the
        // same bytes it was before the field existed.
        if !p.render.is_null() {
            body["render"] = p.render.clone();
        }
        if let (Some(chapter), Some(spans)) = (&self.chapter, &self.chapter_spans) {
            if spans.chapter_sha256 == chapter.sha256 {
                body["chapter_spans"] = json!({
                    "sample_rate": spans.sample_rate,
                    "sample_count": spans.sample_count,
                    "chapter_sha256": spans.chapter_sha256,
                    "chunks": spans.chunks.iter().map(|span| json!([span.start, span.end])).collect::<Vec<_>>(),
                });
            }
        }
        body
    }

    /// The manifest file's contents: `{"manifest": body, "sha256": hex of the body's compact JSON}`.
    pub fn to_json_string(&self) -> String {
        let body = self.body();
        let sha = hex(&sha256(
            serde_json::to_string(&body).expect("JSON values serialize").as_bytes(),
        ));
        serde_json::to_string_pretty(&json!({"manifest": body, "sha256": sha})).expect("JSON values serialize")
    }

    /// Write `manifest.json` into `dir`: a temporary file, flushed, then renamed over the old manifest.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        write_atomic(&dir.join(MANIFEST_FILE), self.to_json_string().as_bytes())
    }

    /// Read a manifest and check it against `plan`. Nothing in it is trusted unless every check passes.
    pub fn load(dir: &Path, plan: &ChapterPlan) -> Result<Manifest, ManifestError> {
        let text = fs::read_to_string(dir.join(MANIFEST_FILE)).map_err(|e| ManifestError::Io(e.to_string()))?;
        Manifest::parse(&text, plan)
    }

    /// A manifest's body, checked for its checksum and format only: for a reader that has no plan, such as an export.
    pub fn read_body(dir: &Path) -> Result<Value, ManifestError> {
        let text = fs::read_to_string(dir.join(MANIFEST_FILE)).map_err(|e| ManifestError::Io(e.to_string()))?;
        let outer: Value = serde_json::from_str(&text).map_err(|e| ManifestError::NotJson(e.to_string()))?;
        let body = outer
            .get("manifest")
            .cloned()
            .ok_or_else(|| ManifestError::Format("no \"manifest\"".into()))?;
        let recorded = outer
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or(ManifestError::Checksum)?;
        if recorded
            != hex(&sha256(
                serde_json::to_string(&body).expect("JSON values serialize").as_bytes(),
            ))
        {
            return Err(ManifestError::Checksum);
        }
        let format = body.get("format").and_then(Value::as_str).unwrap_or("");
        if format != MANIFEST_FORMAT {
            return Err(ManifestError::Format(format!("it is {format:?}")));
        }
        Ok(body)
    }

    /// [`load`](Self::load) from the file's contents.
    pub fn parse(text: &str, plan: &ChapterPlan) -> Result<Manifest, ManifestError> {
        let outer: Value = serde_json::from_str(text).map_err(|e| ManifestError::NotJson(e.to_string()))?;
        let body = outer
            .get("manifest")
            .ok_or_else(|| ManifestError::Format("no \"manifest\"".into()))?;
        let recorded = outer
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or(ManifestError::Checksum)?;
        let actual = hex(&sha256(
            serde_json::to_string(body).expect("JSON values serialize").as_bytes(),
        ));
        if recorded != actual {
            return Err(ManifestError::Checksum);
        }
        let format = body.get("format").and_then(Value::as_str).unwrap_or("");
        if format != MANIFEST_FORMAT {
            return Err(ManifestError::Format(format!("it is {format:?}")));
        }
        let str_field = |name: &'static str| -> Result<String, ManifestError> {
            body.get(name)
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| ManifestError::Format(format!("no {name}")))
        };
        let checks: [(&'static str, String, String); 5] = [
            ("book", str_field("book_id")?, plan.book_id.clone()),
            ("chapter", str_field("chapter_id")?, plan.chapter_id.clone()),
            ("seed root", str_field("seed_root")?, plan.seed_root.clone()),
            ("profile", str_field("profile")?, plan.profile.clone()),
            (
                "ceiling",
                body.get("ceiling")
                    .and_then(Value::as_u64)
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
                plan.ceiling.to_string(),
            ),
        ];
        for (field, manifest, planned) in checks {
            if manifest != planned {
                return Err(ManifestError::Mismatch {
                    field,
                    manifest,
                    plan: planned,
                });
            }
        }
        // The engine that rendered the done chunks: a resume with another engine, backend, model, speaker or ggtts
        // build would splice two renders into one chapter.
        let engine = body.get("engine").cloned().unwrap_or(Value::Null);
        if engine != plan.engine {
            return Err(ManifestError::Mismatch {
                field: "engine",
                manifest: engine.to_string(),
                plan: plan.engine.to_string(),
            });
        }
        check_render(body.get("render").unwrap_or(&Value::Null), &plan.render)?;
        let chunks = body
            .get("chunks")
            .and_then(Value::as_array)
            .ok_or_else(|| ManifestError::Format("no chunks".into()))?;
        let mut done = Vec::with_capacity(plan.chunks.len());
        let mut draws = Vec::with_capacity(plan.chunks.len());
        let mut render_ms = Vec::with_capacity(plan.chunks.len());
        let mut kept_stop = Vec::with_capacity(plan.chunks.len());
        for (i, planned) in plan.chunks.iter().enumerate() {
            let Some(c) = chunks.get(i) else {
                return Err(ManifestError::PlanChanged { index: i });
            };
            let same = c.get("index").and_then(Value::as_u64) == Some(planned.index as u64)
                && c.get("paragraph").and_then(Value::as_u64) == Some(planned.paragraph as u64)
                && c.get("chunk").and_then(Value::as_u64) == Some(planned.chunk as u64)
                && c.get("text").and_then(Value::as_str) == Some(planned.text.as_str())
                && c.get("text_sha256").and_then(Value::as_str) == Some(planned.text_sha256.as_str())
                && c.get("seed").and_then(Value::as_u64) == Some(u64::from(planned.seed));
            if !same {
                return Err(ManifestError::PlanChanged { index: i });
            }
            let output = match (c.get("state").and_then(Value::as_str), c.get("output")) {
                (Some("done"), Some(o)) => {
                    let file = o.get("file").and_then(Value::as_str);
                    let sha = o.get("sha256").and_then(Value::as_str);
                    match (file, sha) {
                        (Some(file), Some(sha)) => Some(Output {
                            file: file.to_string(),
                            sha256: sha.to_string(),
                        }),
                        _ => return Err(ManifestError::Format(format!("chunk {i} is done with no output"))),
                    }
                }
                (Some("planned"), _) => None,
                (state, _) => return Err(ManifestError::Format(format!("chunk {i} has state {state:?}"))),
            };
            render_ms.push(output.as_ref().and(c.get("render_ms").and_then(Value::as_u64)));
            let stop = c.get("kept_stop").and_then(Value::as_str).map(String::from);
            if stop.as_deref().is_some_and(|stop| !matches!(stop, "eos" | "cap")) {
                return Err(ManifestError::Format(format!("chunk {i} has invalid kept stop")));
            }
            kept_stop.push(stop.clone());
            let kept = c
                .get("kept_draw")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            let count = c
                .get("draws")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            draws.push(match (&output, kept, count) {
                (Some(_), Some(kept), Some(count)) if kept < count => Some(Draws {
                    kept,
                    stop,
                    count,
                    verdict: c.get("verdict").and_then(Value::as_str).map(String::from),
                    reads: c.get("reads").and_then(Value::as_array).cloned().unwrap_or_default(),
                }),
                _ => None,
            });
            done.push(output);
        }
        if chunks.len() != plan.chunks.len() {
            return Err(ManifestError::PlanChanged {
                index: plan.chunks.len().min(chunks.len()),
            });
        }
        let chapter = body.get("chapter_file").and_then(|o| {
            Some(Output {
                file: o.get("file")?.as_str()?.to_string(),
                sha256: o.get("sha256")?.as_str()?.to_string(),
            })
        });
        let chapter_spans = match body.get("chapter_spans") {
            None => None,
            Some(value) => {
                let Some(recorded_chapter) = chapter.as_ref() else {
                    return Err(ManifestError::Format("chapter spans without chapter file".into()));
                };
                let sample_rate = value
                    .get("sample_rate")
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| ManifestError::Format("invalid chapter span sample rate".into()))?;
                let sample_count = value
                    .get("sample_count")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| ManifestError::Format("invalid chapter span sample count".into()))?;
                let chapter_sha256 = value
                    .get("chapter_sha256")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ManifestError::Format("invalid chapter span hash".into()))?
                    .to_string();
                if chapter_sha256 != recorded_chapter.sha256 {
                    return Err(ManifestError::Format(
                        "chapter spans are stale for the chapter file".into(),
                    ));
                }
                let records = value
                    .get("chunks")
                    .and_then(Value::as_array)
                    .ok_or_else(|| ManifestError::Format("missing chapter chunk spans".into()))?;
                let mut chunks_spans = Vec::with_capacity(records.len());
                for record in records {
                    let pair = record
                        .as_array()
                        .filter(|pair| pair.len() == 2)
                        .ok_or_else(|| ManifestError::Format("invalid chapter chunk span".into()))?;
                    let start = pair[0]
                        .as_u64()
                        .ok_or_else(|| ManifestError::Format("invalid chapter chunk span start".into()))?;
                    let end = pair[1]
                        .as_u64()
                        .ok_or_else(|| ManifestError::Format("invalid chapter chunk span end".into()))?;
                    chunks_spans.push(SampleSpan { start, end });
                }
                let parsed = ChapterSpans {
                    sample_rate,
                    sample_count,
                    chapter_sha256,
                    chunks: chunks_spans,
                };
                validate_chapter_spans(&parsed, plan.chunks.len()).map_err(ManifestError::Format)?;
                Some(parsed)
            }
        };
        Ok(Manifest {
            plan: plan.clone(),
            done,
            draws,
            render_ms,
            kept_stop,
            chapter,
            chapter_spans,
        })
    }

    /// The chunks still to render: those not done, those whose recorded output is missing from `dir` or no longer has
    /// its recorded SHA-256, and those whose recorded draws a render at the plan's [`redraw`](ChapterPlan::redraw)
    /// would not end at ([`Draws::holds_at`]): a chunk the judge rejected at a lower redraw setting is judged again
    /// under the new one, never kept silently.
    pub fn todo(&self, dir: &Path) -> Vec<usize> {
        (0..self.plan.chunks.len())
            .filter(|&i| match &self.done[i] {
                None => true,
                Some(o) => !output_intact(dir, o) || !Draws::holds_at(self.draws[i].as_ref(), self.plan.redraw),
            })
            .collect()
    }
}

fn validate_chapter_spans(spans: &ChapterSpans, expected_chunks: usize) -> Result<(), String> {
    if spans.sample_rate == 0
        || spans.sample_count == 0
        || spans.chapter_sha256.len() != 64
        || !spans.chapter_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("invalid joined chapter facts".into());
    }
    if spans.chunks.len() != expected_chunks {
        return Err("joined chapter span count does not match plan".into());
    }
    let mut previous_end = 0;
    for (index, span) in spans.chunks.iter().enumerate() {
        if span.start >= span.end
            || span.end > spans.sample_count
            || span.start < previous_end
            || (index == 0 && span.start != 0)
        {
            return Err(format!("invalid joined span for chunk {index}"));
        }
        previous_end = span.end;
    }
    if previous_end != spans.sample_count {
        return Err("joined spans do not cover the end of chapter audio".into());
    }
    Ok(())
}

/// The render options a manifest was written under against the ones about to run: the instruction, the steering and
/// the edge kept at each chunk (`--edge-s`), then anything else, the first difference named. A manifest without the field (written before it existed, or by a
/// render with neither) is a render with no instruction and no steering.
fn check_render(recorded: &Value, planned: &Value) -> Result<(), ManifestError> {
    let shown = |v: Option<&Value>| match v {
        None | Some(Value::Null) => "none".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
    };
    for (key, field) in [
        ("instruction", "instruction"),
        ("steer", "steering"),
        ("edge_s", "edge"),
    ] {
        let (m, p) = (recorded.get(key), planned.get(key));
        if m != p {
            return Err(ManifestError::Mismatch {
                field,
                manifest: shown(m),
                plan: shown(p),
            });
        }
    }
    if recorded != planned {
        return Err(ManifestError::Mismatch {
            field: "render options",
            manifest: shown(Some(recorded)),
            plan: shown(Some(planned)),
        });
    }
    Ok(())
}

fn output_intact(dir: &Path, o: &Output) -> bool {
    // A recorded name that is a path, not a file name, is never trusted.
    if o.file.contains(['/', '\\']) || o.file == ".." || o.file == "." {
        return false;
    }
    fs::read(dir.join(&o.file)).is_ok_and(|bytes| hex(&sha256(&bytes)) == o.sha256)
}

/// Where a chapter's render stands before it starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Resume {
    pub manifest: Manifest,
    /// The chunks to render, in order.
    pub todo: Vec<usize>,
    /// Why an existing manifest was not used, when one was found and refused.
    pub refused: Option<ManifestError>,
    /// The done chunks rendered again because their recorded draws do not hold at the plan's redraw setting
    /// ([`Draws::holds_at`]), in order. Also in [`todo`](Self::todo).
    pub rejudged: Vec<usize>,
}

/// Decide how to start a chapter in `dir`: resume a trusted manifest, else start again from `plan`.
pub fn resume(dir: &Path, plan: &ChapterPlan) -> Resume {
    let path = dir.join(MANIFEST_FILE);
    if !path.exists() {
        let manifest = Manifest::new(plan.clone());
        let todo = (0..plan.chunks.len()).collect();
        return Resume {
            manifest,
            todo,
            refused: None,
            rejudged: Vec::new(),
        };
    }
    match Manifest::load(dir, plan) {
        Ok(manifest) => {
            let todo = manifest.todo(dir);
            let rejudged = todo
                .iter()
                .copied()
                .filter(|&i| {
                    manifest.done[i].as_ref().is_some_and(|o| output_intact(dir, o))
                        && !Draws::holds_at(manifest.draws[i].as_ref(), plan.redraw)
                })
                .collect();
            let mut manifest = manifest;
            for &i in &todo {
                manifest.done[i] = None;
                manifest.draws[i] = None;
                manifest.render_ms[i] = None;
                manifest.kept_stop[i] = None;
            }
            if !todo.is_empty() {
                // A chunk is rendered again, so the chapter file and its sample spans no longer stand.
                manifest.chapter = None;
                manifest.chapter_spans = None;
            }
            Resume {
                manifest,
                todo,
                refused: None,
                rejudged,
            }
        }
        Err(e) => Resume {
            manifest: Manifest::new(plan.clone()),
            todo: (0..plan.chunks.len()).collect(),
            refused: Some(e),
            rejudged: Vec::new(),
        },
    }
}

/// The file a chunk's output is written to: `chunk-00042.flac`.
pub fn chunk_file(index: usize) -> String {
    format!("chunk-{index:05}.flac")
}

/// Why [`run_chapter`] stopped.
#[derive(Debug)]
pub enum RunError {
    /// The renderer failed on a chunk; everything before it is saved.
    Render {
        index: usize,
        reason: String,
    },
    Io(std::io::Error),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Render { index, reason } => write!(f, "chunk {index} failed: {reason}"),
            RunError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RunError {}

/// Render a chapter's outstanding chunks into `dir`, resuming from its manifest.
///
/// `render` is called once per chunk still to do, in order, and returns that chunk's audio file (bytes, or a
/// [`ChunkOutput`] with the draws it made). Each output is written atomically and the manifest saved after it, so a
/// run killed at any point resumes at the first chunk not recorded as done. Returns the [`Resume`] it started from, so
/// a caller can report what was skipped and why.
pub fn run_chapter<R: Into<ChunkOutput>>(
    dir: &Path,
    plan: &ChapterPlan,
    mut render: impl FnMut(&ChapterPlan, usize) -> Result<R, String>,
) -> Result<Resume, RunError> {
    fs::create_dir_all(dir).map_err(RunError::Io)?;
    let start = resume(dir, plan);
    let mut manifest = start.manifest.clone();
    manifest.save(dir).map_err(RunError::Io)?;
    for &index in &start.todo {
        let started = std::time::Instant::now();
        let ChunkOutput { bytes, draws } = render(plan, index)
            .map_err(|reason| RunError::Render { index, reason })?
            .into();
        let ms = started.elapsed().as_millis() as u64;
        let file = chunk_file(index);
        write_atomic(&dir.join(&file), &bytes).map_err(RunError::Io)?;
        manifest.done[index] = Some(Output {
            file,
            sha256: hex(&sha256(&bytes)),
        });
        manifest.kept_stop[index] = draws.as_ref().and_then(|d| d.stop.clone());
        manifest.draws[index] = draws;
        manifest.render_ms[index] = Some(ms);
        manifest.save(dir).map_err(RunError::Io)?;
    }
    Ok(start)
}

/// Write `bytes` to `path` through a temporary file in the same directory, flushed before it is renamed into place.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp: PathBuf = path.to_path_buf();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    tmp.set_file_name(format!(".{name}.partial"));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}
