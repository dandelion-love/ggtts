//! Text to codec tokens: `generate <talker.gguf> <text> <out.codes.json> [seed] [threads] [plant] [settings]`, speaker
//! `hk_cantonese_speaker`, language `zh`, at most 1200 frames, the reference CLI's sampler defaults. Writes the
//! reference CLI's `--dump-generated-codes` format, so the two files compare byte for byte. A plant other than `none`
//! needs the `plant` feature.
//!
//! The settings, anywhere after the three paths, change one of those, with the reference CLI's flags where it has one:
//! `--temperature X`, `--top-k N`, `--top-p X`, `--repetition-penalty X`, `--max-tokens N`, `-l LANG`, and ggtts's own
//! `--subtalker-temperature X`, `--subtalker-top-k N`, `--subtalker-top-p X` (the reference has none: its code
//! predictor samples with the talker's). `crates/ggtts-qwen3-tts/scripts/sampler_parity.sh` compares them.
use std::path::Path;
use std::time::Instant;

use ggtts_qwen3_tts::talker::{language_id, Sampling, Talker};

fn main() {
    let all: Vec<String> = std::env::args().collect();
    if all.len() < 4 {
        eprintln!("usage: generate <talker.gguf> <text> <out.codes.json> [seed] [threads] [plant] [settings]");
        std::process::exit(2);
    }
    let mut sampling = Sampling::default();
    let (mut max_frames, mut lang) = (1200, "zh".to_string());
    let mut args: Vec<String> = all[..4].to_vec();
    let mut rest = all[4..].iter();
    while let Some(a) = rest.next() {
        let mut value = || rest.next().unwrap_or_else(|| panic!("{a} needs a value")).clone();
        match a.as_str() {
            "--temperature" => sampling.temperature = value().parse().unwrap(),
            "--top-k" => sampling.top_k = value().parse().unwrap(),
            "--top-p" => sampling.top_p = value().parse().unwrap(),
            "--repetition-penalty" => sampling.repetition_penalty = value().parse().unwrap(),
            "--subtalker-temperature" => sampling.subtalker_temperature = Some(value().parse().unwrap()),
            "--subtalker-top-k" => sampling.subtalker_top_k = Some(value().parse().unwrap()),
            "--subtalker-top-p" => sampling.subtalker_top_p = Some(value().parse().unwrap()),
            "--max-tokens" => max_frames = value().parse().unwrap(),
            "-l" => lang = value(),
            other if other.starts_with('-') => panic!("unknown setting {other}"),
            _ => args.push(a.clone()),
        }
    }
    let seed = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(20260905);
    let threads = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(4);
    let t = Instant::now();
    let mut talker = load(Path::new(&args[1]), threads, args.get(6).map_or("none", String::as_str));
    let load_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let language = language_id(&lang).unwrap_or_else(|| panic!("unknown language {lang}"));
    let generation = talker
        .generate(
            &args[2],
            None,
            "hk_cantonese_speaker",
            language,
            seed,
            max_frames,
            sampling,
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
