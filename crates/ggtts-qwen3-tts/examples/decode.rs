//! The Rust side of the decoder identity check: `decode <codec.gguf> <codes.json> <out.f32> [threads] [plant]`.
//! Reads the reference CLI's `--dump-generated-codes` format and writes the samples as raw little-endian f32, as
//! `reference/decode_ref.cpp` does.
use std::path::Path;
use std::time::Instant;

use ggtts_qwen3_tts::codec::{Decoder, Plant};
use ggtts_qwen3_tts::fixtures::read_codes_json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: decode <codec.gguf> <codes.json> <out.f32> [threads] [plant]");
        std::process::exit(2);
    }
    let threads = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let plant = match args.get(5).map(String::as_str) {
        None | Some("none") => Plant::None,
        Some("rope-normal") => Plant::RopeNormal,
        Some("snake-no-epsilon") => Plant::SnakeNoEpsilon,
        Some("window-off-by-one") => Plant::WindowOffByOne,
        Some(p) => panic!("unknown plant {p}"),
    };
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let (_, codes) = read_codes_json(&text).unwrap();
    let mut dec = Decoder::load_planted(Path::new(&args[1]), threads, plant).unwrap();
    let t = Instant::now();
    let pcm = dec.decode(&codes).unwrap();
    let ms = t.elapsed().as_millis();
    let bytes: Vec<u8> = pcm.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&args[3], bytes).unwrap();
    eprintln!(
        "decode: {} frames -> {} samples, {ms} ms",
        codes.len() / 16,
        pcm.samples.len()
    );
}
