//! The Rust side of the decoder identity check:
//! `decode <codec.gguf> <codes.json> <out.f32> [threads] [plant] [cpu|cuda]`.
//! Reads the reference CLI's `--dump-generated-codes` format, or one line of a `ggtts-render --codes` file
//! (`docs/codes.md`), and writes the samples as raw little-endian f32, as `reference/decode_ref.cpp` does. A plant
//! other than `none` needs the `plant` feature; `cuda` decodes on the first CUDA device and needs the `cuda` feature.
use std::path::Path;
use std::time::Instant;

use ggtts_qwen3_tts::codec::Decoder;
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::read_codes_json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: decode <codec.gguf> <codes.json> <out.f32> [threads] [plant] [cpu|cuda]");
        std::process::exit(2);
    }
    let threads = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let (_, codes) = read_codes_json(&text).unwrap();
    let device = match args.get(6).map_or("cpu", String::as_str) {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        other => {
            eprintln!("decode: the device is cpu or cuda, not {other}");
            std::process::exit(2);
        }
    };
    let mut dec = load(
        Path::new(&args[1]),
        threads,
        args.get(5).map_or("none", String::as_str),
        device,
    );
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
fn load(path: &Path, threads: i32, plant: &str, device: Device) -> Decoder {
    use ggtts_qwen3_tts::codec::Plant;
    let plant = Plant::parse(plant).unwrap_or_else(|| panic!("unknown plant {plant}"));
    Decoder::load_planted(path, threads, device, plant).unwrap()
}

#[cfg(not(feature = "plant"))]
fn load(path: &Path, threads: i32, plant: &str, device: Device) -> Decoder {
    assert!(plant == "none", "planting {plant} needs the plant feature");
    Decoder::load_on(path, threads, device).unwrap()
}
