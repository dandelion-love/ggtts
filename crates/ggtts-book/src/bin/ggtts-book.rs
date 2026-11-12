//! ggtts-book: EPUB books for ggtts.
//!
//! ```text
//! ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
//!                                   print the chapter model, and each chapter's planned chunks and seeds, as JSON
//! ggtts-book --version
//! ```
//!
//! `--engine` names the engine whose profile the chunks are planned with (its chunk ceiling and split marks); the
//! default is `qwen3-tts`, ggtts-render's default engine. `--seed-root` is the root every chunk's seed is derived
//! from; the default is ggtts-render's default seed, `20260905`.
//!
//! Exit 0 on success; 2 when the book is refused (the reason, with its code, on stderr) or the command line is wrong.

use std::path::Path;
use std::process::ExitCode;

use ggtts_book::plan::{engine_profile, plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::Book;
use serde_json::json;

const USAGE: &str =
    "usage: ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]\n       ggtts-book --version";

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
        Some("inspect") => inspect(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn inspect(args: &[String]) -> ExitCode {
    let mut book_path = None;
    let mut engine = "qwen3-tts".to_string();
    let mut seed_root = DEFAULT_SEED_ROOT.to_string();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--engine" | "--seed-root" => {
                let Some(value) = it.next() else {
                    return fail(&format!("{arg} needs a value\n{USAGE}"));
                };
                if arg == "--engine" {
                    engine = value.clone();
                } else {
                    seed_root = value.clone();
                }
            }
            a if a.starts_with("--") => return fail(&format!("unknown option {a}\n{USAGE}")),
            _ if book_path.is_none() => book_path = Some(arg.clone()),
            _ => return fail(&format!("one book at a time\n{USAGE}")),
        }
    }
    let Some(book_path) = book_path else {
        return fail(USAGE);
    };
    let profile = match engine_profile(&engine) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let book = match Book::open(Path::new(&book_path)) {
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
