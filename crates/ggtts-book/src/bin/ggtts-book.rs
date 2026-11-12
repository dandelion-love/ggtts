//! ggtts-book: EPUB books for ggtts.
//!
//! ```text
//! ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
//!                                   print the chapter model, and each chapter's planned chunks and seeds, as JSON
//! ggtts-book render BOOK.epub --out DIR --engine fake [--seed-root R]
//!                                   render every chapter into DIR, resuming what an earlier run finished
//! ggtts-book --version
//! ```
//!
//! `--engine` names the engine: for `inspect`, the one whose profile the chunks are planned with (default
//! `qwen3-tts`, ggtts-render's default engine); for `render`, the one that renders, which must be named. Only `fake`
//! renders today: it needs no model, so the whole path (chapters, seeds, manifests, resume, one FLAC per chapter) runs
//! end to end. `--seed-root` is the root every chunk's seed is derived from; the default is ggtts-render's default
//! seed, `20260905`.
//!
//! `render` writes `DIR/NNN-<chapter id>/` (the chapter's chunks and manifest) and `DIR/NNN-<chapter id>.flac`, and
//! prints a JSON report. Front and back matter is left out. Run it again on the same `DIR` to resume.
//!
//! Exit 0 on success; 2 when the book is refused (the reason, with its code, on stderr), a render fails (what was
//! finished is kept), or the command line is wrong.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_book::plan::{engine_profile, plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_book, ChapterOutcome};
use ggtts_book::Book;
use serde_json::json;

const USAGE: &str = "usage: ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]\n       \
ggtts-book render BOOK.epub --out DIR --engine fake [--seed-root R]\n       ggtts-book --version";

fn fail(message: &str) -> ExitCode {
    eprintln!("ggtts-book: {message}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") if args.len() == 1 => {
            for line in ggtts_core::version_lines("ggtts-book") {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Some("--help" | "-h") if args.len() == 1 => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(command @ ("inspect" | "render")) => match Options::parse(command, &args[1..]) {
            Ok(o) if command == "inspect" => inspect(o),
            Ok(o) => render(o),
            Err(e) => fail(&format!("{e}\n{USAGE}")),
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

struct Options {
    book: String,
    engine: Option<String>,
    seed_root: String,
    out: Option<PathBuf>,
}

impl Options {
    fn parse(command: &str, args: &[String]) -> Result<Options, String> {
        let mut o = Options {
            book: String::new(),
            engine: None,
            seed_root: DEFAULT_SEED_ROOT.to_string(),
            out: None,
        };
        let mut book = None;
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                flag @ ("--engine" | "--seed-root" | "--out") => {
                    let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?.clone();
                    match flag {
                        "--engine" => o.engine = Some(value),
                        "--seed-root" => o.seed_root = value,
                        _ if command == "render" => o.out = Some(PathBuf::from(value)),
                        _ => return Err("--out is for render".to_string()),
                    }
                }
                a if a.starts_with("--") => return Err(format!("unknown option {a}")),
                _ if book.is_none() => book = Some(arg.clone()),
                _ => return Err("one book at a time".to_string()),
            }
        }
        o.book = book.ok_or("no book named")?;
        Ok(o)
    }
}

fn inspect(o: Options) -> ExitCode {
    let engine = o.engine.unwrap_or_else(|| "qwen3-tts".to_string());
    let profile = match engine_profile(&engine) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let book = match Book::open(Path::new(&o.book)) {
        Ok(book) => book,
        Err(e) => return fail(&e.to_string()),
    };
    let plans = match plan_book(&profile, &book, &o.seed_root) {
        Ok(plans) => plans,
        Err(e) => return fail(&e),
    };
    let mut out = book.to_json();
    out["plan"] = json!({
        "engine": engine,
        "profile": profile.name,
        "ceiling": profile.story_ceiling,
        "seed_root": o.seed_root,
        "seed_key": ggtts_book::plan::SEED_KEY_VERSION,
    });
    for (chapter, plan) in out["chapters"]
        .as_array_mut()
        .expect("chapters is an array")
        .iter_mut()
        .zip(&plans)
    {
        chapter["chunks"] = plan
            .chunks
            .iter()
            .map(|c| {
                json!({
                    "index": c.index,
                    "paragraph": c.paragraph,
                    "chunk": c.chunk,
                    "text": c.text,
                    "text_sha256": c.text_sha256,
                    "seed": c.seed,
                })
            })
            .collect();
    }
    println!("{}", serde_json::to_string_pretty(&out).expect("JSON values serialize"));
    ExitCode::SUCCESS
}

fn render(o: Options) -> ExitCode {
    let Some(out_dir) = o.out else {
        return fail("render needs --out DIR");
    };
    let mut engine = match o.engine.as_deref() {
        Some("fake") => ggtts_fake::FakeEngine::default(),
        Some(other) => {
            return fail(&format!(
                "render: engine {other:?} cannot render from ggtts-book yet; only --engine fake renders until a real \
                 engine's loader is wired in"
            ))
        }
        None => return fail("render needs --engine: only fake renders until a real engine's loader is wired in"),
    };
    let book = match Book::open(Path::new(&o.book)) {
        Ok(book) => book,
        Err(e) => return fail(&e.to_string()),
    };
    let profile = *ggtts_core::engine::Engine::profile(&engine);
    let plans = match plan_book(&profile, &book, &o.seed_root) {
        Ok(plans) => plans,
        Err(e) => return fail(&e),
    };
    let reports = match render_book(&mut engine, &book, &plans, &out_dir) {
        Ok(r) => r,
        Err(e) => return fail(&format!("render stopped: {e}; run again with the same --out to resume")),
    };
    let chapters: Vec<_> = reports
        .iter()
        .map(|r| match &r.outcome {
            ChapterOutcome::Written {
                file,
                rendered,
                resumed,
                refused,
            } => json!({
                "id": r.id,
                "state": "written",
                "file": file.strip_prefix(&out_dir).unwrap_or(file).to_string_lossy(),
                "rendered": rendered.len(),
                "resumed": resumed,
                "refused_manifest": refused,
            }),
            ChapterOutcome::Skipped { reason } => json!({"id": r.id, "state": "skipped", "reason": reason}),
        })
        .collect();
    let report = json!({
        "format": "ggtts-book-render/1",
        "book_id": book.book_id(),
        "engine": "fake",
        "profile": profile.name,
        "seed_root": o.seed_root,
        "chapters": chapters,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("JSON values serialize")
    );
    ExitCode::SUCCESS
}
