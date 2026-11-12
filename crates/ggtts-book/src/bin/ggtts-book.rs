//! ggtts-book: EPUB books for ggtts.
//!
//! ```text
//! ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
//!                                   print the chapter model, and each chapter's planned chunks and seeds, as JSON
//! ggtts-book render BOOK.epub --out DIR [--engine qwen3-tts|kitten-tts-2|fake] [--seed-root R]
//!                  [--model M.gguf] [--codec C.gguf] [--voice REF.json] [--speaker NAME]
//!                  [--device cpu|cuda] [--threads N] [--offline]
//!                  [--include-skippable]
//!                                   render every chapter into DIR, resuming what an earlier run finished
//! ggtts-book export DIR --out BOOK.flac
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
//! `export` refuses a render that is not finished, or that mixes seed roots or engines, naming the chapter.
//!
//! Exit 0 on success; 2 when the book or the engine is refused (the reason on stderr), a render fails (what was
//! finished is kept), or the command line is wrong.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ggtts_book::plan::{engine_profile, plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_command_with, LoadedEngine};
use ggtts_book::Book;
use serde_json::json;

const USAGE: &str = "usage: ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]\n       \
ggtts-book render BOOK.epub --out DIR [--engine qwen3-tts|kitten-tts-2|fake] [--seed-root R]\n                  \
[--model M.gguf] [--codec C.gguf] [--voice REF.json] [--speaker NAME] [--device cpu|cuda] [--threads N] [--offline]\n                  \
[--include-skippable]\n       \
ggtts-book export DIR --out BOOK.flac\n       \
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
    seed_root: String,
    out: Option<PathBuf>,
    include_skippable: bool,
    load: ggtts_load::Request,
}

impl Options {
    fn parse(command: &str, args: &[String]) -> Result<Options, String> {
        let mut o = Options {
            book: String::new(),
            seed_root: DEFAULT_SEED_ROOT.to_string(),
            out: None,
            include_skippable: false,
            load: ggtts_load::Request {
                engine: "qwen3-tts".into(),
                ..Default::default()
            },
        };
        let render_only = |flag: &str| -> Result<(), String> {
            if command == "render" {
                Ok(())
            } else {
                Err(format!("{flag} is for render"))
            }
        };
        let mut book = None;
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--offline" => {
                    render_only("--offline")?;
                    o.load.offline = true;
                }
                "--include-skippable" => {
                    render_only("--include-skippable")?;
                    o.include_skippable = true;
                }
                flag @ ("--engine" | "--seed-root" | "--out" | "--model" | "--codec" | "--voice" | "--speaker"
                | "--device" | "--threads") => {
                    let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?.clone();
                    if !matches!(flag, "--engine" | "--seed-root") {
                        render_only(flag)?;
                    }
                    match flag {
                        "--engine" => o.load.engine = value,
                        "--seed-root" => o.seed_root = value,
                        "--out" => o.out = Some(PathBuf::from(value)),
                        "--model" => o.load.model = Some(PathBuf::from(value)),
                        "--codec" => o.load.codec = Some(PathBuf::from(value)),
                        "--voice" => o.load.voice = Some(PathBuf::from(value)),
                        "--speaker" => o.load.speaker = Some(value),
                        "--device" => o.load.device = Some(value),
                        _ => {
                            let n: i32 = value
                                .parse()
                                .map_err(|_| format!("--threads must be a number, not {value}"))?;
                            if n < 1 {
                                return Err("--threads must be at least 1".into());
                            }
                            o.load.threads = Some(n);
                        }
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
    let engine = o.load.engine;
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

fn render(o: Options) -> ExitCode {
    let Some(out_dir) = o.out else {
        return fail("render needs --out DIR");
    };
    let req = o.load;
    match render_command_with(Path::new(&o.book), &out_dir, &o.seed_root, o.include_skippable, || {
        load_engine(req)
    }) {
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
        return fail(&format!("export needs DIR and --out BOOK.flac\n{USAGE}"));
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
