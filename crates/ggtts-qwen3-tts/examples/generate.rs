//! Text to codec tokens: `generate <talker.gguf> <text> <out.codes.json> [seed] [threads] [plant]`, speaker
//! `hk_cantonese_speaker`, language `zh`, at most 1200 frames, the reference CLI's sampler defaults. Writes the
//! reference CLI's `--dump-generated-codes` format, so the two files compare byte for byte. A plant other than `none`
//! needs the `plant` feature.
use std::path::Path;
use std::time::Instant;

use ggtts_qwen3_tts::talker::{language_id, Sampling, Talker};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: generate <talker.gguf> <text> <out.codes.json> [seed] [threads] [plant]");
        std::process::exit(2);
    }
    let seed = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(20260905);
    let threads = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(4);
    let t = Instant::now();
    let mut talker = load(Path::new(&args[1]), threads, args.get(6).map_or("none", String::as_str));
    let load_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let generation = talker
        .generate(
            &args[2],
            "hk_cantonese_speaker",
            "",
            language_id("zh").unwrap(),
            seed,
            1200,
            Sampling::default(),
        )
        .unwrap();
    let gen_ms = t.elapsed().as_millis();
    let frames = generation.codes.len() / 16;
    let mut s = format!("{{\n  \"frames\": {frames},\n  \"codebooks\": 16,\n  \"codes\": [");
    for (i, c) in generation.codes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        if i % 16 == 0 {
            s.push_str("\n    ");
        }
        s.push_str(&c.to_string());
    }
    s.push_str("\n  ]\n}\n");
    std::fs::write(&args[3], s).unwrap();
    eprintln!(
        "generate: {frames} frames, stop {:?}, load {load_ms} ms, generate {gen_ms} ms",
        generation.stop
    );
}

#[cfg(feature = "plant")]
fn load(path: &Path, threads: i32, plant: &str) -> Talker {
    use ggtts_qwen3_tts::cpu::Device;
    use ggtts_qwen3_tts::talker::Plant;
    let plant = Plant::parse(plant).unwrap_or_else(|| panic!("unknown plant {plant}"));
    Talker::load_planted(path, threads, Device::Cpu, plant).unwrap()
}

#[cfg(not(feature = "plant"))]
fn load(path: &Path, threads: i32, plant: &str) -> Talker {
    assert!(plant == "none", "planting {plant} needs the plant feature");
    Talker::load(path, threads).unwrap()
}
