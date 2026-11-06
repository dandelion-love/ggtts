//! ggtts-render: text in, FLAC and a provenance manifest out.
//!
//! In this skeleton it only reports its version and the ggml pin. Rendering arrives with the pipeline.

use std::process::ExitCode;

const BINARY: &str = "ggtts-render";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("--version") | Some("-V") => {
            for line in ggtts_core::version_lines(BINARY) {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Some("-h") | Some("--help") => {
            println!("usage: {BINARY} [--version]");
            println!();
            println!("This build is a skeleton: it reports its version and the ggml pin, and renders nothing yet.");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("error: invalid argument: {other}");
            eprintln!("usage: {BINARY} [--version]");
            ExitCode::from(2)
        }
    }
}
