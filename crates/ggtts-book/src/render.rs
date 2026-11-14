//! Rendering a book through one engine, chapter by chapter, resumable.
//!
//! The engine is passed in already loaded and is used for every chunk of every chapter, so a book loads its model
//! once. Each chunk is rendered by the core chunk step ([`ggtts_core::pipeline::render_chunk`], the step every
//! render is made of) at its planned seeds ([`ChapterPlan::seed_at`]) with the render options (instruction,
//! steering, edge trim) and the observers (the trace), and stored as FLAC through [`run_chapter`], so a killed render
//! resumes where it stopped. A finished chapter is then joined from the stored chunks with the profile's pauses,
//! levelled by the story path's level step, and written as one FLAC.
//!
//! Chunks are joined from the stored files, never from memory, so a chapter rendered in one run and a chapter
//! resumed after a kill are built from the same bytes and come out identical.
//!
//! Instruments run on each chapter FLAC this run wrote, through the same call as `ggtts-render`'s
//! ([`ggtts_instruments::options`]), after every chapter is rendered and the engine is released
//! ([`render_command_options`]).

use std::io::BufWriter;
use std::path::{Path, PathBuf};

use ggtts_core::engine::{Engine, EngineInfo};
use ggtts_core::flac::encode_flac;
use ggtts_core::hooks::Hooks;
use ggtts_core::options::Options;
use ggtts_core::pipeline::{gap_samples, level, DurationJudge, Observers, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::trace::{emit_header, Events, Header, JsonlEvents, NullEvents};

use crate::book::{Book, Chapter};
use crate::manifest::{chunk_file, run_chapter, ChunkOutput, Draws, Resume, RunError};
use crate::plan::{ChapterPlan, DEFAULT_SEED_ROOT};

/// What happened to one chapter.
#[derive(Debug, Clone, PartialEq)]
pub enum ChapterOutcome {
    /// Rendered (or resumed) and written.
    Written {
        /// The chapter's FLAC.
        file: PathBuf,
        /// Chunks rendered in this run.
        rendered: Vec<usize>,
        /// Chunks taken from an earlier run.
        resumed: usize,
        /// Why an earlier manifest was not used, if one was refused.
        refused: Option<String>,
        /// Done chunks rendered again because the redraw setting changed what a render keeps.
        rejudged: usize,
    },
    /// Front or back matter, left out.
    Skipped { reason: String },
}

/// A chapter's outcome, by its id.
#[derive(Debug, Clone, PartialEq)]
pub struct ChapterReport {
    pub id: String,
    pub outcome: ChapterOutcome,
}

/// How a book renders its chunks: the core step's options, and whether front and back matter are rendered.
#[derive(Debug, Clone)]
pub struct BookRender {
    /// Render front and back matter too.
    pub include_skippable: bool,
    /// The core step's options: instruction, steering, edge trim, and the redraws a chunk the judge rejects gets.
    pub opts: RenderOptions,
}

impl BookRender {
    /// The profile's defaults (its redraws and edge trim), with no instruction and no steering, leaving front and
    /// back matter out.
    pub fn for_profile(profile: &Profile) -> Self {
        Self {
            include_skippable: false,
            opts: RenderOptions::for_profile(profile),
        }
    }
}

/// Render one planned chunk through the core chunk step ([`ggtts_core::pipeline::render_chunk`]) at the plan's seeds
/// ([`ChapterPlan::seed_at`], draw 0, then 1, 2, ... for each redraw) with the render spec's judge, `opts` and the
/// observers `obs`: a draw the judge rejects is redrawn up to `opts.redraw` times, as in a story. The FLAC bytes to
/// store, and the draws made.
pub fn render_chunk(
    engine: &mut dyn Engine,
    plan: &ChapterPlan,
    index: usize,
    opts: &RenderOptions,
    obs: &mut Observers<'_>,
) -> Result<ChunkOutput, String> {
    let seeds = |i: usize, _: &str, draw: u32| plan.seed_at(i, draw);
    let kept = ggtts_core::pipeline::render_chunk(
        engine,
        &plan.chunks[index].text,
        index,
        &seeds,
        &mut DurationJudge,
        opts,
        obs,
    )
    .map_err(|e| e.to_string())?;
    let pcm: Vec<i16> = kept.samples.iter().map(|&x| to_i16(x)).collect();
    let bytes = encode_flac(&pcm, kept.sample_rate).map_err(|e| e.to_string())?;
    let last = kept.kept();
    Ok(ChunkOutput {
        bytes,
        draws: Some(Draws {
            kept: last.draw,
            count: kept.draws.len() as u32,
            verdict: last.judgement.verdict.map(|v| v.as_str().to_string()),
        }),
    })
}

fn to_i16(x: f32) -> i16 {
    (f64::from(x).clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// The directory a chapter's chunks and manifest live in, and the chapter's FLAC beside it: `NNN-<id>`.
pub fn chapter_stem(index: usize, chapter: &Chapter) -> String {
    let id: String = chapter
        .id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{index:03}-{id}")
}

/// Join a finished chapter's stored chunks with the profile's pauses, level it, and write `out` (a `.flac` path)
/// with `tags` in its VORBIS_COMMENT block.
pub fn join_chapter(
    profile: &Profile,
    dir: &Path,
    plan: &ChapterPlan,
    out: &Path,
    tags: &[(String, String)],
) -> Result<PathBuf, String> {
    let mut read: Vec<f32> = Vec::new();
    let mut rate = None;
    for (i, chunk) in plan.chunks.iter().enumerate() {
        let path = dir.join(chunk_file(i));
        let mut reader = claxon::FlacReader::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sr = reader.streaminfo().sample_rate;
        if *rate.get_or_insert(sr) != sr {
            return Err(format!(
                "{}: sample rate {sr}, not {}",
                path.display(),
                rate.unwrap_or(sr)
            ));
        }
        for s in reader.samples() {
            let s = s.map_err(|e| format!("{}: {e}", path.display()))?;
            read.push(s as f32 / 32768.0);
        }
        if i + 1 < plan.chunks.len() {
            read.extend(std::iter::repeat_n(
                0.0,
                gap_samples(profile, &chunk.punct, chunk.last_in_para, sr),
            ));
        }
    }
    let rate = rate.ok_or_else(|| format!("chapter {} has no chunks", plan.chapter_id))?;
    level(profile, &mut read);
    let pcm: Vec<i16> = read.iter().map(|&x| ggtts_core::flac::to_i16(x)).collect();
    let flac = encode_flac(&pcm, rate).map_err(|e| format!("{}: {e}", out.display()))?;
    let tagged = crate::flac::with_tags(&flac, tags).map_err(|e| format!("{}: {e}", out.display()))?;
    crate::manifest::write_atomic(out, &tagged).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out.to_path_buf())
}

/// The tags a chapter FLAC carries: TITLE (the chapter), ALBUM (the book), one ARTIST per author, TRACKNUMBER (its
/// place among the chapters written), and the ids that tie it to its render.
pub fn chapter_tags(book: &Book, chapter: &Chapter, track: usize, plan: &ChapterPlan) -> Vec<(String, String)> {
    let mut tags = vec![
        ("TITLE".to_string(), chapter.title.clone()),
        ("ALBUM".to_string(), book.title.clone()),
    ];
    tags.extend(book.authors.iter().map(|a| ("ARTIST".to_string(), a.clone())));
    tags.extend([
        ("TRACKNUMBER".to_string(), track.to_string()),
        ("GGTTS_BOOK_ID".to_string(), plan.book_id.clone()),
        ("GGTTS_CHAPTER_ID".to_string(), plan.chapter_id.clone()),
        ("GGTTS_SEED_ROOT".to_string(), plan.seed_root.clone()),
    ]);
    tags
}

/// The file in an output directory that names the book rendered into it.
pub const BOOK_MARKER: &str = "book.json";

/// Refuse an output directory that holds another book; mark a fresh one as this book's. Without this, a second book
/// rendered into the same directory would refuse the first book's manifests and overwrite its chunks.
fn claim_output(out: &Path, book_id: &str) -> Result<(), String> {
    let marker = out.join(BOOK_MARKER);
    match std::fs::read_to_string(&marker) {
        Ok(text) => {
            let held = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("book_id").and_then(|b| b.as_str()).map(String::from));
            match held {
                Some(id) if id == book_id => Ok(()),
                Some(id) => Err(format!(
                    "{} holds another book ({id}), not {book_id}: render into another --out",
                    out.display()
                )),
                None => Err(format!("{} is not a book marker this tool wrote", marker.display())),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let has_entries = std::fs::read_dir(out).map_err(|e| e.to_string())?.next().is_some();
            if has_entries {
                return Err(format!(
                    "{} is not empty and holds no {BOOK_MARKER}: render into an empty directory or one this tool wrote",
                    out.display()
                ));
            }
            write_marker(out, book_id, &[])
        }
        Err(e) => Err(format!("{}: {e}", marker.display())),
    }
}

/// The format of `book.json`.
pub const MARKER_FORMAT: &str = "ggtts-book-output/1";

/// Write `book.json`: the book rendered into `out`, and the chapter directories its latest render planned, in order.
fn write_marker(out: &Path, book_id: &str, chapters: &[String]) -> Result<(), String> {
    let marker = out.join(BOOK_MARKER);
    let text = serde_json::json!({"format": MARKER_FORMAT, "book_id": book_id, "chapters": chapters}).to_string();
    crate::manifest::write_atomic(&marker, text.as_bytes()).map_err(|e| format!("{}: {e}", marker.display()))
}

/// What a render's chunks come from, recorded in every chapter manifest and checked on resume: the engine family,
/// backend and parity, the profile, each model file's role and SHA-256, the speaker, and the ggtts build.
pub fn provenance(engine: &dyn Engine, files: &[(String, String)], speaker: Option<&str>) -> serde_json::Value {
    let info = engine.info();
    serde_json::json!({
        "family": info.family,
        "backend": info.backend,
        "parity": info.parity,
        "profile": engine.profile().name,
        "files": files.iter().map(|(role, sha)| serde_json::json!({"role": role, "sha256": sha})).collect::<Vec<_>>(),
        "speaker": speaker,
        "ggtts_version": ggtts_core::VERSION,
        "ggtts_commit": ggtts_core::SOURCE_COMMIT,
    })
}

/// The engine, the files it came from (role and SHA-256) and its speaker: what [`render_command`]'s loader returns.
pub struct LoadedEngine {
    pub engine: Box<dyn Engine>,
    pub files: Vec<(String, String)>,
    pub speaker: Option<String>,
}

/// The whole `ggtts-book render`: read the book, load the engine once through `load`, plan every chapter with the
/// engine's own profile, tie the plans to the engine's provenance, and render into `out`. The book is read before the
/// engine is loaded, so a refused book costs no model load.
pub fn render_command(
    book_path: &Path,
    out: &Path,
    seed_root: &str,
    load: impl FnOnce() -> Result<LoadedEngine, String>,
) -> Result<serde_json::Value, String> {
    render_command_with(book_path, out, seed_root, false, load)
}

/// [`render_command`], rendering front and back matter too when `include_skippable`.
pub fn render_command_with(
    book_path: &Path,
    out: &Path,
    seed_root: &str,
    include_skippable: bool,
    load: impl FnOnce() -> Result<LoadedEngine, String>,
) -> Result<serde_json::Value, String> {
    let mut options = Options::default();
    options.output.out = Some(out.to_path_buf());
    options.book.seed_root = Some(seed_root.to_string());
    options.book.include_skippable = include_skippable;
    render_command_options(book_path, &options, load)
}

/// The render options that change the audio, from the provenance record ([`ggtts_core::options::record`]): the
/// style instruction (when not empty) and the steering. `Null` when there are neither. Every chapter manifest records
/// it, and a resume under other render options is refused ([`crate::manifest::ManifestError::Mismatch`]).
pub fn render_identity(options: &Options, info: &EngineInfo) -> serde_json::Value {
    let record = ggtts_core::options::record(options, info);
    let kept: serde_json::Map<String, serde_json::Value> = ["instruction", "steer"]
        .into_iter()
        .filter_map(|k| record.get(k).map(|v| (k.to_string(), v.clone())))
        .collect();
    if kept.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::Object(kept)
    }
}

/// [`render_command`] with the shared options value ([`ggtts_core::options`]): the output directory
/// (`output.out`), the book section (`book.seed_root`, else [`DEFAULT_SEED_ROOT`]; `book.include_skippable`), the
/// render section's instruction and steering, and the trace (`observe.trace`, `observe.trace_level`). The load
/// section is the caller's: `load` opens the engine. The steering file is read before the engine is loaded, so a bad
/// one costs no model load. Options that do not apply to a book are refused by the command line before this
/// ([`crate::options`]).
pub fn render_command_options(
    book_path: &Path,
    options: &Options,
    load: impl FnOnce() -> Result<LoadedEngine, String>,
) -> Result<serde_json::Value, String> {
    let out = options.output.out.as_deref().ok_or("render needs --out DIR")?;
    let seed_root = options.book.seed_root.as_deref().unwrap_or(DEFAULT_SEED_ROOT);
    let book = Book::open(book_path).map_err(|e| e.to_string())?;
    let mut options = options.clone();
    options.read_steering().map_err(|e| e.to_string())?;
    // An instrument that cannot be found, or a reference a book supplies itself, is refused before the engine
    // loads or any chapter is rendered.
    let instruments = ggtts_instruments::options::config(&options)?;
    if let Some(c) = &instruments {
        refuse_given_reference(c)?;
    }
    let LoadedEngine {
        mut engine,
        files,
        speaker,
    } = load()?;
    let profile = *engine.profile();
    let settings = BookRender {
        include_skippable: options.book.include_skippable,
        opts: options.render_options(&profile).map_err(|e| e.to_string())?,
    };
    let mut plans = crate::plan::plan_book(&profile, &book, seed_root)?;
    let engine_record = provenance(engine.as_ref(), &files, speaker.as_deref());
    let identity = render_identity(&options, engine.info());
    let reported = ggtts_core::options::reported(&ggtts_core::options::record(&options, engine.info()));
    for plan in &mut plans {
        plan.engine = engine_record.clone();
        plan.render = identity.clone();
    }
    let trace_level = options.observe.trace_level.unwrap_or(1);
    let mut trace = match &options.observe.trace {
        Some(path) => Some(
            std::fs::File::create(path)
                .map(|f| JsonlEvents::new(BufWriter::new(f), trace_level))
                .map_err(|e| format!("--trace {}: {e}", path.display()))?,
        ),
        None => None,
    };
    let reports = {
        let events: &mut dyn Events = match trace.as_mut() {
            Some(t) => t,
            None => &mut NullEvents,
        };
        let mut header = Header::for_engine(engine.as_ref(), trace_level);
        header.instruction = settings.opts.instruction.clone().filter(|i| !i.is_empty());
        header.steer = settings.opts.steer.as_ref().map(|s| s.record());
        emit_header(events, &header);
        render_book_observed(engine.as_mut(), &book, &plans, out, &settings, events)?
    };
    if let Some(t) = trace {
        t.finish().map_err(|e| format!("--trace: {e}"))?;
    }
    // The engine is released before any instrument runs, as in ggtts-render: an instrument may load a model of its
    // own on the same device.
    drop(engine);
    let measured = match &instruments {
        Some(config) => {
            let models: Vec<serde_json::Value> = files
                .iter()
                .map(|(role, sha)| serde_json::json!({"role": role, "sha256": sha}))
                .collect();
            measure_chapters(config, &options, &reports, &plans, &models, &reported)?
        }
        None => Vec::new(),
    };
    let chapters: Vec<_> = reports
        .iter()
        .map(|r| match &r.outcome {
            ChapterOutcome::Written {
                file,
                rendered,
                resumed,
                refused,
                rejudged,
            } => {
                let mut c = serde_json::json!({
                    "id": r.id,
                    "state": "written",
                    "file": file.strip_prefix(out).unwrap_or(file).to_string_lossy(),
                    "rendered": rendered.len(),
                    "resumed": resumed,
                    "refused_manifest": refused,
                });
                if *rejudged > 0 {
                    c["rejudged"] = (*rejudged).into();
                }
                if let Some(m) = measured.iter().find(|m| m.id == r.id) {
                    c["instruments"] = m.to_json(out);
                }
                c
            }
            ChapterOutcome::Skipped { reason } => serde_json::json!({"id": r.id, "state": "skipped", "reason": reason}),
        })
        .collect();
    let mut report = serde_json::json!({
        "format": "ggtts-book-render/1",
        "book_id": book.book_id(),
        "engine": engine_record,
        "seed_root": seed_root,
        "chapters": chapters,
    });
    if !identity.is_null() {
        report["render"] = identity;
    }
    if let Some(config) = &instruments {
        report["measurement"] = if measurement_complete(config, &measured) {
            "complete"
        } else {
            "incomplete"
        }
        .into();
    }
    Ok(report)
}

/// Whether a chapter was measured, and when.
#[derive(Debug, Clone, PartialEq)]
pub enum Measured {
    /// Measured in this run.
    ThisRun,
    /// Measured by an earlier run with the same chapter FLAC and the same settings: its `measured.json`.
    EarlierRun,
    /// Not measured, for this reason; the next run measures it.
    No(String),
}

/// One written chapter's instrument run: its id, its report directory, when it was measured, and what came of it
/// (the instruments' statuses and why the measurement is incomplete; empty when it was not measured).
pub struct ChapterMeasurement {
    pub id: String,
    pub report_dir: PathBuf,
    pub measured: Measured,
    pub status: serde_json::Map<String, serde_json::Value>,
    pub incomplete: Vec<String>,
}

impl ChapterMeasurement {
    /// The chapter's `instruments` entry in the render report: the report directory (relative to `out` when inside
    /// it), when it was measured, each instrument's status, and why the measurement is incomplete, if it is.
    pub fn to_json(&self, out: &Path) -> serde_json::Value {
        let (measured, reason) = match &self.measured {
            Measured::ThisRun => ("this run", None),
            Measured::EarlierRun => ("earlier run", None),
            Measured::No(why) => ("no", Some(why.clone())),
        };
        let mut v = serde_json::json!({
            "report": self.report_dir.strip_prefix(out).unwrap_or(&self.report_dir).to_string_lossy(),
            "measured": measured,
            "status": self.status,
            "complete": matches!(self.measured, Measured::ThisRun | Measured::EarlierRun) && self.incomplete.is_empty(),
            "incomplete": self.incomplete,
        });
        if let Some(r) = reason {
            v["reason"] = r.into();
        }
        v
    }
}

/// Whether the book's measurement is complete: every measured chapter is, and, when an instrument is required, no
/// written chapter was left unmeasured (as `ggtts-render` counts a required instrument that never ran).
pub fn measurement_complete(config: &ggtts_instruments::Config, measured: &[ChapterMeasurement]) -> bool {
    measured.iter().all(|m| match m.measured {
        Measured::ThisRun | Measured::EarlierRun => m.incomplete.is_empty(),
        Measured::No(_) => config.require.is_empty(),
    })
}

/// Refuse a `reference_file` given for a book: the book gives each chapter its own text, and one file cannot be every
/// chapter's.
pub fn refuse_given_reference(config: &ggtts_instruments::Config) -> Result<(), String> {
    use ggtts_instruments::catalogue::REFERENCE_ARG;
    match config
        .instruments
        .iter()
        .find(|e| e.args.iter().any(|(k, _)| k == REFERENCE_ARG))
    {
        Some(e) => Err(format!(
            "--instrument {}: arg.{REFERENCE_ARG} is refused by ggtts-book render: a book gives each chapter its own \
             text as the reference, so leave it out",
            e.name
        )),
        None => Ok(()),
    }
}

/// A chapter's spoken text, as its reference: its chunks, a line per paragraph.
pub fn chapter_reference(plan: &ChapterPlan) -> String {
    let mut text = String::new();
    for c in &plan.chunks {
        text.push_str(&c.text);
        if c.last_in_para {
            text.push('\n');
        }
    }
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// The file in a chapter's report directory that records a finished measurement.
pub const MEASURED_STAMP: &str = "measured.json";
/// Its format.
pub const MEASURED_FORMAT: &str = "ggtts-book-measured/1";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Measure every written chapter not yet measured with the same chapter FLAC and the same settings. A chapter whose
/// report directory holds a matching `measured.json` is taken from it ([`Measured::EarlierRun`]); any other is
/// measured now, and stamped when no instrument failed. So a resumed book measures the chapters an interrupted run never measured, and a re-run
/// with other instruments measures every chapter again.
///
/// Each chapter is one render of the run (its id is the chapter's directory name), with its report beside its FLAC
/// (`NNN-<id>.report`), or under `observe.report` when given (`<report>/NNN-<id>`). The run is `observe.run`, else
/// `book-<unix seconds>`, one for the whole book. An instrument that takes a reference (an ASR) is given the chapter's
/// own text, written to `reference.txt` in its report directory. The book's trace covers every chapter, so no chapter
/// is given an event stream, and an instrument that needs one is `skipped` with that reason. Under
/// `on_instrument_error: abort`, a chapter where an instrument failed stops the chapters after it ([`Measured::No`]).
pub fn measure_chapters(
    config: &ggtts_instruments::Config,
    options: &Options,
    reports: &[ChapterReport],
    plans: &[ChapterPlan],
    models: &[serde_json::Value],
    reported: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<ChapterMeasurement>, String> {
    use ggtts_instruments::catalogue::{find, REFERENCE_ARG};
    let run = ggtts_instruments::options::run_id(options, "book");
    let mut measured = Vec::new();
    let mut aborted: Option<String> = None;
    for (r, plan) in reports.iter().zip(plans) {
        let ChapterOutcome::Written { file, .. } = &r.outcome else {
            continue;
        };
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or(&r.id).to_string();
        let report_dir = options
            .observe
            .report
            .as_ref()
            .map_or_else(|| file.with_extension("report"), |d| d.join(&stem));
        let entry = |measured, status, incomplete| ChapterMeasurement {
            id: r.id.clone(),
            report_dir: report_dir.clone(),
            measured,
            status,
            incomplete,
        };
        if let Some(why) = &aborted {
            measured.push(entry(Measured::No(why.clone()), serde_json::Map::new(), Vec::new()));
            continue;
        }
        // The chapter's own config: every reference-taking instrument reads the chapter's text.
        let reference = report_dir.join("reference.txt");
        let mut chapter_config = config.clone();
        let mut needs_reference = false;
        for e in &mut chapter_config.instruments {
            if find(&e.name).is_some_and(|k| k.takes_reference) {
                e.args
                    .push((REFERENCE_ARG.to_string(), reference.to_string_lossy().into_owned()));
                e.args.sort();
                needs_reference = true;
            }
        }
        let flac = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let signature = hex(&ggtts_core::seed::sha256(
            format!("{}\n{chapter_config:?}", hex(&ggtts_core::seed::sha256(&flac))).as_bytes(),
        ));
        let stamp_path = report_dir.join(MEASURED_STAMP);
        let stamp = std::fs::read(&stamp_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .filter(|v| v["format"] == MEASURED_FORMAT && v["signature"] == signature.as_str());
        if let Some(v) = stamp {
            let status = v["status"].as_object().cloned().unwrap_or_default();
            let incomplete = v["incomplete"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            measured.push(entry(Measured::EarlierRun, status, incomplete));
            continue;
        }
        // A stamp that no longer matches is removed before the run, so an interrupted run leaves none.
        let _ = std::fs::remove_file(&stamp_path);
        if needs_reference {
            std::fs::create_dir_all(&report_dir).map_err(|e| format!("{}: {e}", report_dir.display()))?;
            std::fs::write(&reference, chapter_reference(plan)).map_err(|e| format!("{}: {e}", reference.display()))?;
        }
        let finished = ggtts_instruments::options::Finished {
            run: run.clone(),
            render: stem,
            audio: file.clone(),
            audio_state: ggtts_core::report::AudioState::Trimmed,
            audio_seconds: flac_seconds(file),
            events: None,
            models: Some(models.to_vec()),
            reported: reported.clone(),
            default_report_dir: report_dir.clone(),
        };
        let (_, summary) = ggtts_instruments::options::run(
            &chapter_config,
            &finished,
            None,
            ggtts_core::logging::Settings::default(),
        )
        .map_err(|e| format!("chapter {}: instruments: {e}", r.id))?;
        let status: serde_json::Map<String, serde_json::Value> = summary
            .envelopes
            .iter()
            .map(|e| (e.name.clone(), e.status.as_str().into()))
            .collect();
        // Only a measurement in which no instrument failed is kept for later runs: a failure (a crash, a timeout, an
        // invalid envelope) or an abort is measured again next time.
        if summary.failures() == 0 && !summary.aborted {
            let stamp = serde_json::json!({
                "format": MEASURED_FORMAT,
                "signature": signature,
                "run": run,
                "status": status,
                "incomplete": summary.incomplete,
            });
            crate::manifest::write_atomic(&stamp_path, stamp.to_string().as_bytes())
                .map_err(|e| format!("{}: {e}", stamp_path.display()))?;
        }
        if summary.aborted {
            aborted = Some(format!(
                "an instrument failed on chapter {} and on_instrument_error is abort",
                r.id
            ));
        }
        measured.push(entry(Measured::ThisRun, status, summary.incomplete.clone()));
    }
    Ok(measured)
}

/// The chapter's length in seconds, from its FLAC's stream info.
fn flac_seconds(path: &Path) -> Option<f64> {
    let info = claxon::FlacReader::open(path).ok()?.streaminfo();
    let samples = info.samples?;
    (info.sample_rate > 0).then(|| samples as f64 / f64::from(info.sample_rate))
}

/// Render every chapter of `book` into `out`, with `engine` loaded once for all of them.
///
/// Skippable chapters (front and back matter) are left out. A chapter whose chunk fails stops the render with that
/// chunk named; what was finished is kept, and the next run resumes from it.
pub fn render_book(
    engine: &mut dyn Engine,
    book: &Book,
    plans: &[ChapterPlan],
    out: &Path,
) -> Result<Vec<ChapterReport>, String> {
    render_book_with(engine, book, plans, out, false)
}

/// [`render_book`], rendering front and back matter too when `include_skippable`.
pub fn render_book_with(
    engine: &mut dyn Engine,
    book: &Book,
    plans: &[ChapterPlan],
    out: &Path,
    include_skippable: bool,
) -> Result<Vec<ChapterReport>, String> {
    let settings = BookRender {
        include_skippable,
        ..BookRender::for_profile(engine.profile())
    };
    render_book_observed(engine, book, plans, out, &settings, &mut NullEvents)
}

/// [`render_book`] with `settings` (the core step's options and whether front and back matter are rendered), feeding
/// every rendered chunk's trace events into `events`. Chunks resumed from an earlier run are not rendered, so they are
/// not traced.
pub fn render_book_observed(
    engine: &mut dyn Engine,
    book: &Book,
    plans: &[ChapterPlan],
    out: &Path,
    settings: &BookRender,
    events: &mut dyn Events,
) -> Result<Vec<ChapterReport>, String> {
    let include_skippable = settings.include_skippable;
    let mut hooks = Hooks::new();
    // A chapter's manifest records the redraws its chunks were rendered with, whoever planned it.
    let plans: Vec<ChapterPlan> = plans
        .iter()
        .map(|p| ChapterPlan {
            redraw: settings.opts.redraw,
            ..p.clone()
        })
        .collect();
    let plans = &plans[..];
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    claim_output(out, &book.book_id())?;
    // Record which chapters this render plans, before rendering any: `export` takes exactly these, so a directory a
    // render with other options left behind is never exported, and a planned chapter that is missing is refused.
    let planned: Vec<String> = book
        .chapters
        .iter()
        .zip(plans)
        .enumerate()
        .filter(|(_, (c, p))| (include_skippable || !c.skippable) && !p.chunks.is_empty())
        .map(|(i, (c, _))| chapter_stem(i, c))
        .collect();
    write_marker(out, &book.book_id(), &planned)?;
    let mut reports = Vec::new();
    let mut track = 0;
    for (i, (chapter, plan)) in book.chapters.iter().zip(plans).enumerate() {
        if let Some(reason) = chapter
            .skip_reason
            .clone()
            .filter(|_| chapter.skippable && !include_skippable)
        {
            reports.push(ChapterReport {
                id: chapter.id.clone(),
                outcome: ChapterOutcome::Skipped { reason },
            });
            continue;
        }
        if plan.chunks.is_empty() {
            let reason = "nothing to speak".to_string();
            reports.push(ChapterReport {
                id: chapter.id.clone(),
                outcome: ChapterOutcome::Skipped { reason },
            });
            continue;
        }
        let stem = chapter_stem(i, chapter);
        let dir = out.join(&stem);
        let mut rendered = Vec::new();
        let start: Resume = run_chapter(&dir, plan, |p, index| {
            rendered.push(index);
            render_chunk(
                engine,
                p,
                index,
                &settings.opts,
                &mut Observers::new(&mut *events, &mut hooks),
            )
        })
        .map_err(|e| match e {
            RunError::Render { index, reason } => format!(
                "render stopped at chapter {}, chunk {index}: {reason}; what was finished is kept, run again with the \
                 same --out to resume",
                chapter.id
            ),
            RunError::Io(e) => format!("chapter {}: {e}", chapter.id),
        })?;
        let profile = *engine.profile();
        track += 1;
        let tags = chapter_tags(book, chapter, track, plan);
        let file = join_chapter(&profile, &dir, plan, &out.join(format!("{stem}.flac")), &tags)?;
        let bytes = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        crate::manifest::Manifest::record_chapter(&dir, plan, &format!("{stem}.flac"), &bytes)
            .map_err(|e| format!("chapter {}: {e}", chapter.id))?;
        reports.push(ChapterReport {
            id: chapter.id.clone(),
            outcome: ChapterOutcome::Written {
                file,
                resumed: plan.chunks.len() - start.todo.len(),
                rendered,
                refused: start.refused.map(|e| e.to_string()),
                rejudged: start.rejudged.len(),
            },
        });
    }
    Ok(reports)
}
