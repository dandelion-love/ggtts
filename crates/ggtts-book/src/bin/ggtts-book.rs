//! ggtts-book: EPUB books for ggtts.
//!
//! ```text
//! ggtts-book inspect BOOK.epub     print the chapter model as JSON
//! ggtts-book --version
//! ```
//!
//! Exit 0 on success; 2 when the book is refused (the reason, with its code, on stderr) or the command line is wrong.

use std::path::Path;
use std::process::ExitCode;

use ggtts_book::Book;

const USAGE: &str = "usage: ggtts-book inspect BOOK.epub\n       ggtts-book --version";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["--version"] => {
            for line in ggtts_core::version_lines("ggtts-book") {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        ["--help"] | ["-h"] => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        ["inspect", book] => match Book::open(Path::new(book)) {
            Ok(book) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&book.to_json()).expect("JSON values serialize")
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("ggtts-book: {e}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
