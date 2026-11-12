//! The chapter manifest: what a chapter's render planned and what it has finished, written so a killed run resumes.
//!
//! One `manifest.json` per chapter directory. Its body records the plan (book, chapter, seed root, profile, ceiling,
//! and every chunk's text, hash and seed) and each chunk's state; the file carries the SHA-256 of its body beside it.
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

/// A chapter's manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub plan: ChapterPlan,
    /// One entry per planned chunk: `Some` once it is done.
    pub done: Vec<Option<Output>>,
    /// How long each done chunk took to render (generate, decode, trim and encode), in milliseconds. Recorded for
    /// reading back; never checked.
    pub render_ms: Vec<Option<u64>>,
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
    /// Made for another book, chapter, seed root, profile or ceiling: names the field.
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
        Manifest { plan, done, render_ms }
    }

    fn body(&self) -> Value {
        let p = &self.plan;
        json!({
            "format": MANIFEST_FORMAT,
            "book_id": p.book_id,
            "chapter_id": p.chapter_id,
            "seed_root": p.seed_root,
            "profile": p.profile,
            "ceiling": p.ceiling,
            "engine": p.engine,
            "chunks": p.chunks.iter().enumerate().map(|(i, c)| {
                let done = &self.done[i];
                json!({
                    "index": c.index,
                    "paragraph": c.paragraph,
                    "chunk": c.chunk,
                    "text": c.text,
                    "text_sha256": c.text_sha256,
                    "seed": c.seed,
                    "state": if done.is_some() { "done" } else { "planned" },
                    "output": done.as_ref().map(|o| json!({"file": o.file, "sha256": o.sha256})),
                    "render_ms": self.render_ms[i],
                })
            }).collect::<Vec<_>>(),
        })
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
        let chunks = body
            .get("chunks")
            .and_then(Value::as_array)
            .ok_or_else(|| ManifestError::Format("no chunks".into()))?;
        let mut done = Vec::with_capacity(plan.chunks.len());
        let mut render_ms = Vec::with_capacity(plan.chunks.len());
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
            done.push(output);
        }
        if chunks.len() != plan.chunks.len() {
            return Err(ManifestError::PlanChanged {
                index: plan.chunks.len().min(chunks.len()),
            });
        }
        Ok(Manifest {
            plan: plan.clone(),
            done,
            render_ms,
        })
    }

    /// The chunks still to render: those not done, and those whose recorded output is missing from `dir` or no
    /// longer has its recorded SHA-256.
    pub fn todo(&self, dir: &Path) -> Vec<usize> {
        (0..self.plan.chunks.len())
            .filter(|&i| match &self.done[i] {
                None => true,
                Some(o) => !output_intact(dir, o),
            })
            .collect()
    }
}

fn output_intact(dir: &Path, o: &Output) -> bool {
    // A recorded name that is a path, not a file name, is never trusted.
    if o.file.contains(['/', '\\']) || o.file == ".." || o.file == "." {
        return false;
    }
    fs::read(dir.join(&o.file)).is_ok_and(|bytes| hex(&sha256(&bytes)) == o.sha256)
}

/// Where a chapter's render stands before it starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resume {
    pub manifest: Manifest,
    /// The chunks to render, in order.
    pub todo: Vec<usize>,
    /// Why an existing manifest was not used, when one was found and refused.
    pub refused: Option<ManifestError>,
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
        };
    }
    match Manifest::load(dir, plan) {
        Ok(manifest) => {
            let todo = manifest.todo(dir);
            let mut manifest = manifest;
            for &i in &todo {
                manifest.done[i] = None;
                manifest.render_ms[i] = None;
            }
            Resume {
                manifest,
                todo,
                refused: None,
            }
        }
        Err(e) => Resume {
            manifest: Manifest::new(plan.clone()),
            todo: (0..plan.chunks.len()).collect(),
            refused: Some(e),
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
/// `render` is called once per chunk still to do, in order, and returns that chunk's audio file. Each output is
/// written atomically and the manifest saved after it, so a run killed at any point resumes at the first chunk not
/// recorded as done. Returns the [`Resume`] it started from, so a caller can report what was skipped and why.
pub fn run_chapter(
    dir: &Path,
    plan: &ChapterPlan,
    mut render: impl FnMut(&ChapterPlan, usize) -> Result<Vec<u8>, String>,
) -> Result<Resume, RunError> {
    fs::create_dir_all(dir).map_err(RunError::Io)?;
    let start = resume(dir, plan);
    let mut manifest = start.manifest.clone();
    manifest.save(dir).map_err(RunError::Io)?;
    for &index in &start.todo {
        let started = std::time::Instant::now();
        let bytes = render(plan, index).map_err(|reason| RunError::Render { index, reason })?;
        let ms = started.elapsed().as_millis() as u64;
        let file = chunk_file(index);
        write_atomic(&dir.join(&file), &bytes).map_err(RunError::Io)?;
        manifest.done[index] = Some(Output {
            file,
            sha256: hex(&sha256(&bytes)),
        });
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
