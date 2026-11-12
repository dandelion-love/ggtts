//! head_rows_check: can Kitten-TTS 2's output head be computed for the samplable tokens only, without changing a draw?
//!
//! ```text
//! cargo run --release -p ggtts-kitten --features ggml --example head_rows_check -- --model LM.gguf --codec S3GEN.gguf \
//!     --voice VOICE.json --lines crates/ggtts-kitten/fixtures/head_rows_lines.txt [--threads 4] [--seed N] [--json FILE]
//! ```
//!
//! A step keeps a frame only for an audio id, and stops on the two end ids. But the sampler draws from whatever stays
//! finite after its penalties, temperature, top-k and top-p, over the whole vocabulary. Computing the head for the
//! allowed ids alone is exact for a step only if no other id is still finite there: then top-k's cutoff, top-p's mass
//! and the draw are all the same. This runs each line at one seed and counts, at every step, the other ids still
//! finite after the filters. It changes nothing: the scores are only read (`generate_ids_observed`).

use ggtts_kitten::backend::Device;
use ggtts_kitten::engine::KittenEngine;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let mut a: BTreeMap<String, String> = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        a.insert(k.trim_start_matches("--").to_string(), v);
    }
    let need = |k: &str| PathBuf::from(a.get(k).unwrap_or_else(|| panic!("--{k} is required")));
    let threads: usize = a.get("threads").map_or(4, |v| v.parse().expect("--threads N"));
    let seed: u64 = a.get("seed").map_or(20_260_905, |v| v.parse().expect("--seed N"));
    let lines: Vec<String> = std::fs::read_to_string(need("lines"))
        .expect("read --lines")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    let mut eng =
        KittenEngine::open(&need("model"), &need("codec"), &need("voice"), threads, Device::Cpu).expect("open");
    let tm = eng.tm;
    let (lo, hi) = (tm.audio_base, tm.audio_base + tm.audio_count);
    let allowed =
        |i: usize| (lo as usize..hi as usize).contains(&i) || i == tm.speech_end as usize || i == tm.stop as usize;
    println!(
        "head_rows_check: {} lines, seed {seed}, {threads} threads; allowed: audio ids {lo}..{hi} ({}), speech_end {}, stop {}",
        lines.len(),
        tm.audio_count,
        tm.speech_end,
        tm.stop
    );
    let mut records = Vec::new();
    let (mut all_steps, mut all_bad_steps, mut all_bad) = (0usize, 0usize, 0usize);
    let mut ids: BTreeMap<usize, usize> = BTreeMap::new();
    for (n, line) in lines.iter().enumerate() {
        let (mut steps, mut bad_steps, mut bad, mut finite) = (0usize, 0usize, 0usize, 0usize);
        let mut local: BTreeMap<usize, usize> = BTreeMap::new();
        let t0 = std::time::Instant::now();
        let (tokens, stop) = eng
            .generate_ids_observed(line, seed, &mut |scores: &[f32]| {
                steps += 1;
                let mut here = 0;
                for (i, v) in scores.iter().enumerate() {
                    if v.is_finite() {
                        finite += 1;
                        if !allowed(i) {
                            here += 1;
                            *local.entry(i).or_default() += 1;
                        }
                    }
                }
                if here > 0 {
                    bad_steps += 1;
                    bad += here;
                }
            })
            .expect("generate");
        let secs = t0.elapsed().as_secs_f64();
        let hash = tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(0x100_0000_01b3)
        });
        let worst: Vec<String> = {
            let mut v: Vec<_> = local.iter().collect();
            v.sort_by(|x, y| y.1.cmp(x.1));
            v.iter().take(5).map(|(i, c)| format!("{i}x{c}")).collect()
        };
        println!(
            "line {:2}: {steps:4} steps, {:.1} finite a step, {bad_steps} steps with another id finite ({bad} in all) {} | {} tokens, {stop:?}, hash {hash:016x}, {secs:.1} s",
            n + 1,
            finite as f64 / steps.max(1) as f64,
            if worst.is_empty() { String::new() } else { format!("[{}]", worst.join(" ")) },
            tokens.len()
        );
        for (i, c) in local {
            *ids.entry(i).or_default() += c;
        }
        all_steps += steps;
        all_bad_steps += bad_steps;
        all_bad += bad;
        records.push(format!(
            "{{\"line\":{},\"steps\":{steps},\"other_finite_steps\":{bad_steps},\"other_finite\":{bad},\"tokens\":{},\"stop\":\"{stop:?}\",\"hash\":\"{hash:016x}\"}}",
            n + 1,
            tokens.len()
        ));
    }
    println!(
        "all: {all_steps} steps; {all_bad_steps} with an id outside the allowed set still finite ({all_bad} in all); distinct ids {}",
        ids.len()
    );
    if !ids.is_empty() {
        let mut v: Vec<_> = ids.iter().collect();
        v.sort_by(|x, y| y.1.cmp(x.1));
        println!(
            "most often: {}",
            v.iter()
                .take(15)
                .map(|(i, c)| format!("{i}x{c}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    if let Some(path) = a.get("json") {
        std::fs::write(
            path,
            format!(
                "{{\"seed\":{seed},\"steps\":{all_steps},\"other_finite_steps\":{all_bad_steps},\"other_finite\":{all_bad},\"lines\":[{}]}}\n",
                records.join(",")
            ),
        )
        .expect("write --json");
        println!("wrote {path}");
    }
}
