//! The Rust side of the text tokenizer identity check: `tokenize <talker.gguf> <records.txt>`. Reads records as
//! `reference/tokenize_ref.cpp` does (one per line, `\n` `\r` `\t` `\\` unescaped, other bytes passed through) and
//! prints, per record, `i <TAB> encode <TAB> encode_for_tts <TAB> encode_instruct <TAB> encode_reference_for_tts`.
use std::io::Write;
use std::path::Path;

use ggtts_qwen3_tts::gguf::Gguf;
use ggtts_qwen3_tts::tokenizer::Tokenizer;

fn unescape(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == b'\\' && i + 1 < s.len() {
            let c = match s[i + 1] {
                b'n' => Some(b'\n'),
                b'r' => Some(b'\r'),
                b't' => Some(b'\t'),
                b'\\' => Some(b'\\'),
                _ => None,
            };
            if let Some(c) = c {
                out.push(c);
                i += 2;
                continue;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

fn field(out: &mut String, ids: &[i32]) {
    out.push('\t');
    let s: Vec<String> = ids.iter().map(i32::to_string).collect();
    out.push_str(&s.join(" "));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tokenize <talker.gguf> <records.txt>");
        std::process::exit(2);
    }
    let g = Gguf::open(Path::new(&args[1])).unwrap();
    let tok = Tokenizer::load(&g).unwrap();
    let data = std::fs::read(&args[2]).unwrap();
    // std::getline semantics: split on '\n', no record after a final newline.
    let mut records: Vec<&[u8]> = data.split(|&b| b == b'\n').collect();
    if data.ends_with(b"\n") || data.is_empty() {
        records.pop();
    }
    let mut stdout = std::io::stdout().lock();
    for (i, rec) in records.iter().enumerate() {
        let text = unescape(rec);
        let mut line = i.to_string();
        field(&mut line, &tok.encode_bytes(&text).unwrap());
        field(&mut line, &tok.encode_for_tts(&text).unwrap());
        field(&mut line, &tok.encode_instruct(&text).unwrap());
        field(&mut line, &tok.encode_reference_for_tts(&text).unwrap());
        writeln!(stdout, "{line}").unwrap();
    }
}
