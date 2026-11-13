//! head_rows_check: can Kitten-TTS 2's output head be computed for the samplable tokens only, without changing a draw?
//!
//! ```text
//! cargo run --release -p ggtts-kitten --features ggml --example head_rows_check -- --model LM.gguf --codec S3GEN.gguf \
//!     --voice VOICE.json --lines crates/ggtts-kitten/fixtures/head_rows_lines.txt [--threads 4] [--seed N] [--json FILE] \
//!     [--head full|restricted]
//! ```
//!
//! A step keeps a frame only for an audio id, and stops on the two end ids. But the sampler draws over the whole
//! vocabulary, as both references do, so computing the head for the allowed ids alone is exact for a step only if no
//! other id is in its top-k set: then top-k's cutoff, top-p's mass and the draw are all the same.
//!
//! This runs each line at one seed with the whole head (it sets `GGTTS_KITTEN_FULL_HEAD=1`) and, at every step, takes
//! the logits as the model gave them, applies the sampler's penalties and temperature but not top-k or top-p, and
//! measures the **margin**: the k-th largest score minus the largest score of any other id. A positive margin means no
//! other id reached the top-k set at that step. It also counts the other ids still finite after all the filters, the
//! weaker check. It changes nothing: the scores are only read (`generate_ids_observed`).

use ggtts_kitten::backend::Device;
use ggtts_kitten::engine::{KittenEngine, Step};
use ggtts_kitten::sampler::{self, Settings};
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
    // The margin is only meaningful against the whole head. `--head restricted` runs the engine's restricted head
    // instead, to compare its tokens (hashes) with a whole-head run; its margins are then not meaningful.
    let restricted = a.get("head").is_some_and(|v| v == "restricted");
    std::env::set_var("GGTTS_KITTEN_FULL_HEAD", "1");
    let mut eng =
        KittenEngine::open(&need("model"), &need("codec"), &need("voice"), threads, Device::Cpu).expect("open");
    let tm = eng.tm;
    let k = eng.settings.top_k;
    // The rows the restricted head would compute at these settings (with the variable unset).
    let head = {
        std::env::remove_var("GGTTS_KITTEN_FULL_HEAD");
        let r = eng.head_rows();
        if !restricted {
            std::env::set_var("GGTTS_KITTEN_FULL_HEAD", "1");
        }
        r
    };
    // "Allowed" is what the restricted head computes: an id outside it in the top-k set is what would break it.
    let rows = head
        .clone()
        .expect("the restricted head is off at these sampler settings; nothing to check");
    let allowed = |i: usize| rows.contains(&i);
    println!(
        "head_rows_check: {} lines, seed {seed}, {threads} threads, top_k {k}, top_p {}; the head's rows {}..{} ({} rows; audio ids from {}, speech_end {}, stop {})",
        lines.len(),
        eng.settings.top_p,
        rows.start,
        rows.end,
        rows.len(),
        tm.audio_base,
        tm.speech_end,
        tm.stop
    );
    let num = |v: f32| {
        if v.is_finite() {
            format!("{v:.4}")
        } else {
            "null".into()
        }
    };
    let mut records = Vec::new();
    let (mut all_steps, mut all_inside, mut all_finite_steps) = (0usize, 0usize, 0usize);
    let mut all_min = f32::INFINITY;
    for (n, line) in lines.iter().enumerate() {
        let (mut steps, mut inside, mut finite_steps) = (0usize, 0usize, 0usize);
        let mut min_margin = f32::INFINITY;
        let t0 = std::time::Instant::now();
        let mut observe = |s: &Step<'_>| {
            steps += 1;
            // Penalties and temperature, as the draw sees them, but neither top-k nor top-p.
            let mut p = s.raw.to_vec();
            let only = Settings {
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                ..s.settings
            };
            sampler::process(&mut p, s.history, s.prompt_len, s.ids, &only).expect("penalties");
            let mut sorted = p.clone();
            sorted.sort_unstable_by(|x, y| y.total_cmp(x));
            let kth = sorted[s.settings.top_k.max(1) - 1];
            let (best_id, best_other) = p.iter().enumerate().filter(|(i, _)| !allowed(*i)).fold(
                (usize::MAX, f32::NEG_INFINITY),
                |b, (i, v)| if *v > b.1 { (i, *v) } else { b },
            );
            let margin = kth - best_other;
            min_margin = min_margin.min(margin);
            if margin <= 0.0 {
                inside += 1;
                let in_rows = head.as_ref().is_some_and(|r| r.contains(&best_id));
                println!(
                    "  step {}: id {best_id} scored {best_other:.3} against the top-{}'s {kth:.3} (margin {margin:.3}); the head's rows {} it",
                    steps - 1,
                    s.settings.top_k,
                    if in_rows { "cover" } else { "do NOT cover" }
                );
            }
            if s.filtered.iter().enumerate().any(|(i, v)| v.is_finite() && !allowed(i)) {
                finite_steps += 1;
            }
        };
        let (tokens, stop) = eng
            .generate_ids_observed(line, seed, Some(&mut observe))
            .expect("generate");
        let secs = t0.elapsed().as_secs_f64();
        let hash = tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(0x100_0000_01b3)
        });
        println!(
            "line {:2}: {steps:4} steps, smallest margin {min_margin:7.3}, {inside} steps with another id in the top-{k}, {finite_steps} with one finite after the filters | {} tokens, {stop:?}, hash {hash:016x}, {secs:.1} s",
            n + 1,
            tokens.len()
        );
        all_steps += steps;
        all_inside += inside;
        all_finite_steps += finite_steps;
        all_min = all_min.min(min_margin);
        records.push(format!(
            "{{\"line\":{},\"steps\":{steps},\"min_margin\":{},\"other_in_top_k_steps\":{inside},\"other_finite_steps\":{finite_steps},\"tokens\":{},\"stop\":\"{stop:?}\",\"hash\":\"{hash:016x}\"}}",
            n + 1,
            num(min_margin),
            tokens.len()
        ));
    }
    println!(
        "all: {all_steps} steps; smallest margin {all_min:.3}; {all_inside} steps with another id in the top-{k}; {all_finite_steps} with one finite after the filters"
    );
    if let Some(path) = a.get("json") {
        std::fs::write(
            path,
            format!(
                "{{\"seed\":{seed},\"top_k\":{k},\"steps\":{all_steps},\"min_margin\":{},\"other_in_top_k_steps\":{all_inside},\"other_finite_steps\":{all_finite_steps},\"lines\":[{}]}}\n",
                num(all_min),
                records.join(",")
            ),
        )
        .expect("write --json");
        println!(
            "wrote {}",
            std::path::Path::new(path)
                .file_name()
                .map_or(String::new(), |f| f.to_string_lossy().into_owned())
        );
    }
}
