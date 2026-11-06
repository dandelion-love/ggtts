//! End to end on CPU: text to FLAC through the [`Engine`] trait.
//!
//! ```text
//! render --talker T.gguf --codec C.gguf --out x.flac [--threads N] (--text "..." [--seed S] | --story story.txt)
//! ```
//!
//! `--text` renders one request at one seed (default 20260905), as the reference CLI renders `-t`. `--story` reads a
//! story file (title line, then one paragraph per line) and renders it on the core's story path: chunked at the
//! profile's ceiling, a seed per chunk, judged, trimmed, joined and levelled. Timings go to stderr.
use std::path::PathBuf;
use std::time::Instant;

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::flac::write_flac;
use ggtts_core::pipeline::{plan_story, render_story, RenderOptions};
use ggtts_core::text::read_text;
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut talker, mut codec, mut out, mut text, mut story) = (None, None, None, None, None);
    let (mut seed, mut threads) = (20260905u64, 4i32);
    while let Some(a) = args.next() {
        let mut v = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--talker" => talker = Some(PathBuf::from(v())),
            "--codec" => codec = Some(PathBuf::from(v())),
            "--out" => out = Some(PathBuf::from(v())),
            "--text" => text = Some(v()),
            "--story" => story = Some(PathBuf::from(v())),
            "--seed" => seed = v().parse().unwrap(),
            "--threads" => threads = v().parse().unwrap(),
            _ => panic!("unknown argument {a}"),
        }
    }
    let (talker, codec, out) = (talker.expect("--talker"), codec.expect("--codec"), out.expect("--out"));
    let t0 = Instant::now();
    let mut engine = Qwen3Tts::load(&talker, &codec, threads, Voice::default()).unwrap();
    let load_ms = t0.elapsed().as_millis();
    let t1 = Instant::now();
    let (samples, rate) = if let Some(text) = text {
        let req = ChunkRequest {
            text: &text,
            seed,
            chunk: 0,
            draw: 0,
        };
        let g = engine.generate(&req).unwrap();
        let gen_ms = t1.elapsed().as_millis();
        let t2 = Instant::now();
        let pcm = engine.decode(&g.codes).unwrap();
        eprintln!(
            "render: {} frames, stop {:?}, generate {gen_ms} ms, decode {} ms",
            g.codes.n_frames(),
            g.stop,
            t2.elapsed().as_millis()
        );
        (pcm.samples, pcm.sample_rate)
    } else {
        let story = story.expect("--text or --story");
        let contents = std::fs::read_to_string(&story).unwrap();
        let (title, paragraphs) = read_text(&contents);
        let profile = *engine.profile();
        let chunks = plan_story(&profile, &paragraphs);
        let read = render_story(&mut engine, &title, &chunks, &RenderOptions::for_profile(&profile)).unwrap();
        eprintln!(
            "render: story {title:?}, {} chunks, {} first-draw defects, {:.2} s of audio",
            read.chunks.len(),
            read.first_draw_defects(),
            read.seconds()
        );
        (read.samples, read.sample_rate)
    };
    let render_ms = t1.elapsed().as_millis();
    let path = write_flac(&out, &samples, rate).unwrap();
    eprintln!(
        "render: wrote {} ({} samples at {rate} Hz), load {load_ms} ms, render {render_ms} ms, total {} ms",
        path.display(),
        samples.len(),
        t0.elapsed().as_millis()
    );
}
