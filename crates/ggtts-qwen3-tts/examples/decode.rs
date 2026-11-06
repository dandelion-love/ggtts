//! The Rust side of the decoder identity check: `decode <codec.gguf> <codes.json> <out.f32> [threads] [plant]`.
//! Reads the reference CLI's `--dump-generated-codes` format and writes the samples as raw little-endian f32, as
//! `reference/decode_ref.cpp` does. A plant other than `none` needs the `plant` feature.
use std::path::Path;
use std::time::Instant;

use ggtts_qwen3_tts::codec::Decoder;
use ggtts_qwen3_tts::fixtures::read_codes_json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: decode <codec.gguf> <codes.json> <out.f32> [threads] [plant]");
        std::process::exit(2);
    }
    let threads = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let (_, codes) = read_codes_json(&text).unwrap();
    let mut dec = load(Path::new(&args[1]), threads, args.get(5).map_or("none", String::as_str));
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

#[cfg(feature = "plant")]
fn load(path: &Path, threads: i32, plant: &str) -> Decoder {
    use ggtts_qwen3_tts::codec::Plant;
    use ggtts_qwen3_tts::cpu::Device;
    let plant = Plant::parse(plant).unwrap_or_else(|| panic!("unknown plant {plant}"));
    Decoder::load_planted(path, threads, Device::Cpu, plant).unwrap()
}

#[cfg(not(feature = "plant"))]
fn load(path: &Path, threads: i32, plant: &str) -> Decoder {
    assert!(plant == "none", "planting {plant} needs the plant feature");
    Decoder::load(path, threads).unwrap()
}
