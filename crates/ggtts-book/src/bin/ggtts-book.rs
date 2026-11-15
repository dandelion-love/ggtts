//! ggtts-book: EPUB books for ggtts.
//!
//! ```text
//! ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
//!                                   print the chapter model, and each chapter's planned chunks and seeds, as JSON
//! ggtts-book render BOOK.epub --out DIR [--engine qwen3-tts|kitten-tts-2|fake] [--seed-root R]
//!                  [--model M.gguf] [--codec C.gguf] [--voice REF.json] [--speaker NAME]
//!                  [--device cpu|cuda] [--threads N] [--offline]
//!                  [--instruction TEXT] [--steer FILE [--steer-scale X] [--steer-layers L,L]]
//!                  [--trace FILE [--trace-level 1|2]] [--redraw N [--judge asr-qwen,model=DIR[,device=cpu|cuda][,min_run=N]]]
//!                  [--include-skippable]
//!                  [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE.yaml] [--require NAME]...
//!                  [--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort]
//!                  [--instrument-timeout S] [--report DIR] [--run ID]
//!                                   render every chapter into DIR, resuming what an earlier run finished
//! ggtts-book export DIR --out BOOK.flac|BOOK.m4b
//!                                   join a finished render into one FLAC with a chapter mark per chapter
//! ggtts-book --version
//! ```
//!
//! `--engine` is `qwen3-tts` by default, as for `ggtts-render`. For `inspect` it names the profile the chunks are
//! planned with; for `render`, the engine that renders, loaded once for the whole book. The model flags are
//! `ggtts-render`'s: a file not given is the engine's pinned default, from the per-user cache (`GGTTS_CACHE_DIR`),
//! downloaded the first time unless `--offline`. Model engines need a build with the `ggml` feature
//! (`cargo build -p ggtts-book --features ggml`); without it `render` refuses them, and `--engine fake` (a tone per
//! chunk, no model) runs the whole path. `--seed-root` is the root every chunk's seed is derived from; the default is
//! ggtts-render's default seed, `20260905`.
//!
//! `render` writes `DIR/book.json`, `DIR/NNN-<chapter id>/` (the chapter's chunks and manifest) and
//! `DIR/NNN-<chapter id>.flac`, and prints a JSON report. Front and back matter is left out. Run it again on the same
//! `DIR` to resume; a `DIR` holding another book is refused. `--include-skippable` renders the front and back
//! matter too. Each chapter FLAC is tagged: TITLE, ALBUM, ARTIST, TRACKNUMBER.
//!
//! The flags are the shared options table's (`ggtts_core::options`), read by its parser, so each means what it means
//! in `ggtts-render`: `--instruction`, `--steer` and its scale and layers reach every chunk's request, and `--trace`
//! writes one ggtts-trace/2 stream for the chunks this run renders. Each chunk is judged, and one the judge rejects is
//! redrawn up to `--redraw` times (unset: the engine profile's), as in a story; each chunk's manifest entry records
//! the kept draw and the number of draws. `--judge` adds an ASR that reads each draw back, so a chunk that lost or
//! repeated a run of its text is redrawn too, and records what it read per chunk (`docs/book.md`). The instrument flags are ggtts-render's: the instruments run on each
//! chapter FLAC this run wrote (not on a chapter an earlier run finished), with one report per chapter beside its
//! FLAC (`NNN-<chapter id>.report`, or `--report DIR/NNN-<chapter id>`). An unknown instrument is refused before
//! anything is rendered. A table option a book does not take (`--text`, `--story`, `--seed`, `--stream`, the
//! timing and duration reports, the log) is refused by name with the reason. A chapter's
//! manifest records the instruction and steering, and a resume under different ones starts the chapter again, naming
//! the difference.
//!
//! `export` refuses a render that is not finished, or that mixes seed roots or engines, naming the chapter.
//!
//! Exit 0 on success; 3 when every chapter is written but a `--require`d instrument is not ok on some chapter (the
//! measurement is INCOMPLETE, as for ggtts-render); 2 when the book or the engine is refused (the reason on stderr),
//! a render fails (what was finished is kept), or the command line is wrong.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_book::plan::{engine_profile, plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_command_options, LoadedEngine};
use ggtts_book::Book;
use ggtts_core::options::Options;
use serde_json::json;

const USAGE: &str = "usage: ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]\n       \
ggtts-book render BOOK.epub --out DIR [--engine qwen3-tts|kitten-tts-2|fake] [--seed-root R]\n                  \
[--model M.gguf] [--codec C.gguf] [--voice REF.json] [--speaker NAME] [--device cpu|cuda] [--threads N] [--offline]\n                  \
[--instruction TEXT] [--steer FILE [--steer-scale X] [--steer-layers L,L]] [--trace FILE [--trace-level 1|2]]\n                  \
[--redraw N [--judge asr-qwen,model=DIR[,device=cpu|cuda][,min_run=N]]] [--include-skippable]\n                  \
[--instrument NAME[,KEY=VALUE]...]... [--instruments FILE] [--require NAME]...\n                  \
[--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort] [--instrument-timeout S] \
[--report DIR] [--run ID]\n       \
ggtts-book export DIR --out BOOK.flac|BOOK.m4b\n       \
ggtts-book --version";

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
        Some("export") => export(&args[1..]),
        Some(command @ ("inspect" | "render")) => {
            match ggtts_book::options::parse(command, &args[1..], &|name| std::env::var(name).ok()) {
                Ok((book, o)) if command == "inspect" => inspect(&book, o),
                Ok((book, o)) => render(&book, o),
                Err(e) => fail(&format!("{e}\n{USAGE}")),
            }
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn inspect(book_path: &Path, o: Options) -> ExitCode {
    let engine = o.load.engine;
    let seed_root = o.book.seed_root.unwrap_or_else(|| DEFAULT_SEED_ROOT.to_string());
    let profile = match engine_profile(&engine) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let book = match Book::open(book_path) {
        Ok(book) => book,
        Err(e) => return fail(&e.to_string()),
    };
    let plans = match plan_book(&profile, &book, &seed_root) {
        Ok(plans) => plans,
        Err(e) => return fail(&e),
    };
    let mut out = book.to_json();
    out["plan"] = json!({
        "engine": engine,
        "profile": profile.name,
        "ceiling": profile.story_ceiling,
        "seed_root": seed_root,
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

/// Load the engine through `ggtts-load`, hashing every model file for the manifest's provenance.
fn load_engine(req: ggtts_load::Request) -> Result<LoadedEngine, String> {
    let ggtts_load::Loaded {
        engine,
        mut files,
        speaker,
    } = ggtts_load::load_default(req)?;
    let mut hashed = Vec::new();
    for f in &mut files {
        f.ensure_sha256()?;
        hashed.push((f.role.clone(), f.sha256.clone().unwrap_or_default()));
    }
    Ok(LoadedEngine {
        engine,
        files: hashed,
        speaker,
    })
}

fn render(book_path: &Path, o: Options) -> ExitCode {
    if o.output.out.is_none() {
        return fail("render needs --out DIR");
    }
    let req = o.load.clone();
    match render_command_options(book_path, &o, || load_engine(req)) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("JSON values serialize")
            );
            if report["measurement"] == "incomplete" {
                eprintln!("ggtts-book: measurement INCOMPLETE: a required instrument is not ok (see each chapter's instruments)");
                return ExitCode::from(ggtts_instruments::runner::EXIT_INCOMPLETE);
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

fn export(args: &[String]) -> ExitCode {
    let mut dir = None;
    let mut out = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => match it.next() {
                Some(v) => out = Some(PathBuf::from(v)),
                None => return fail(&format!("--out needs a value\n{USAGE}")),
            },
            a if a.starts_with("--") => return fail(&format!("unknown option {a}\n{USAGE}")),
            _ if dir.is_none() => dir = Some(PathBuf::from(arg)),
            _ => return fail(&format!("one directory at a time\n{USAGE}")),
        }
    }
    let (Some(dir), Some(out)) = (dir, out) else {
        return fail(&format!("export needs DIR and --out BOOK.flac or BOOK.m4b\n{USAGE}"));
    };
    match ggtts_book::export::export(&dir, &out) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("JSON values serialize")
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}
