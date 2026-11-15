//! AC6: the book renders only through the core chunk step. No source file of `ggtts-book` calls an engine's
//! `generate`, `generate_traced`, `generate_hooked` or `decode` itself: every chunk goes through
//! `ggtts_core::pipeline::render_chunk`, so the book gets every capability the step carries.

use std::path::Path;

const ENGINE_CALLS: [&str; 4] = [".generate(", ".generate_traced(", ".generate_hooked(", ".decode("];

fn sources(dir: &Path, found: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            sources(&p, found);
        } else if p.extension().is_some_and(|x| x == "rs") {
            found.push(p);
        }
    }
}

/// The engine calls in `text`, outside comments: (line number, line).
fn engine_calls(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| {
            let code = l.split("//").next().unwrap_or("");
            ENGINE_CALLS.iter().any(|c| code.contains(c))
        })
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect()
}

#[test]
fn no_book_source_calls_the_engine_directly() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    assert!(files.len() > 5, "the scan found the crate's sources");
    let mut calls = Vec::new();
    for f in &files {
        for (line, code) in engine_calls(&std::fs::read_to_string(f).unwrap()) {
            calls.push(format!("{}:{line}: {code}", f.display()));
        }
    }
    assert!(
        calls.is_empty(),
        "render through ggtts_core::pipeline::render_chunk:\n{}",
        calls.join("\n")
    );
}

/// The scan sees a direct call: the book's chunk renderer as it was before it moved to the core step.
#[test]
fn the_scan_sees_a_direct_engine_call() {
    let before = "    let generated = engine.generate(&req).map_err(|e| e.to_string())?;\n    \
                  let pcm = engine.decode(&generated.codes).map_err(|e| e.to_string())?;\n    \
                  // engine.generate(&req) in a comment is not a call\n";
    assert_eq!(engine_calls(before).len(), 2);
}
