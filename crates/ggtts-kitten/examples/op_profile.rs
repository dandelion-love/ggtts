//! op_profile: where a Kitten-TTS 2 generation step's time goes, per ggml op, in the engine's own graph.
//!
//! ```text
//! cargo run --release -p ggtts-kitten --features ggml --example op_profile -- --model LM.gguf --codec S3GEN.gguf \
//!     --voice VOICE.json --lines FILE [--count 4] [--threads 4] [--json FILE]
//! ```
//!
//! It sets the scheduler's evaluation callback to observe every node, so the scheduler computes each node on its own
//! and calls back before and after it; the time between the two is that node's. Matrix products are keyed by their
//! weight's type and shape (the output head is the one with 158,241 rows), other ops by name. The prefill (a product
//! over more than one token) is kept apart from the one-token steps a generation is made of.
//!
//! Observing every node costs a fixed amount per node (each becomes its own split), so small ops read high and the
//! total is above an unobserved step's. Products are still each timed from their start to their end, which is what
//! comparing two weight types needs.

use ggtts_ggml_sys as g;
use ggtts_kitten::backend::Device;
use ggtts_kitten::engine::KittenEngine;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default)]
struct Profile {
    started: Option<Instant>,
    /// key -> (ns, count)
    ops: BTreeMap<String, (u128, u64)>,
}

static PROFILE: Mutex<Option<Profile>> = Mutex::new(None);

unsafe fn key(t: *mut g::ggml_tensor) -> String {
    let op = CStr::from_ptr(g::ggml_op_desc(t)).to_string_lossy().into_owned();
    if (*t).op == g::GGML_OP_MUL_MAT {
        let w = (*t).src[0];
        let x = (*t).src[1];
        let ty = CStr::from_ptr(g::ggml_type_name((*w).type_)).to_string_lossy();
        let phase = if (*x).ne[1] > 1 { "prefill " } else { "" };
        let what = if (*w).ne[1] == 158_241 {
            "head".to_string()
        } else {
            format!("{}x{}", (*w).ne[0], (*w).ne[1])
        };
        format!("{phase}MUL_MAT {ty} {what}")
    } else {
        let phase = if (*t).ne[1] > 1 && (*t).ne[2] == 1 && (*t).ne[0] == 2048 {
            "prefill "
        } else {
            ""
        };
        format!("{phase}{op}")
    }
}

unsafe extern "C" fn observe(t: *mut g::ggml_tensor, ask: bool, _user: *mut std::os::raw::c_void) -> bool {
    let mut guard = PROFILE.lock().unwrap();
    let p = guard.get_or_insert_with(Profile::default);
    if ask {
        p.started = Some(Instant::now());
        return true;
    }
    if let Some(t0) = p.started.take() {
        let e = p.ops.entry(key(t)).or_default();
        e.0 += t0.elapsed().as_nanos();
        e.1 += 1;
    }
    true
}

fn main() {
    let mut a: BTreeMap<String, String> = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        a.insert(k.trim_start_matches("--").to_string(), v);
    }
    let need = |k: &str| PathBuf::from(a.get(k).unwrap_or_else(|| panic!("--{k} is required")));
    let threads: usize = a.get("threads").map_or(4, |v| v.parse().expect("--threads N"));
    let count: usize = a.get("count").map_or(4, |v| v.parse().expect("--count N"));
    let lines: Vec<String> = std::fs::read_to_string(need("lines"))
        .expect("read --lines")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(count)
        .map(String::from)
        .collect();
    let mut eng =
        KittenEngine::open(&need("model"), &need("codec"), &need("voice"), threads, Device::Cpu).expect("open");
    unsafe { g::ggml_backend_sched_set_eval_callback(eng.model.be.sched, Some(observe), std::ptr::null_mut()) };
    let mut steps = 0usize;
    let t0 = Instant::now();
    for line in &lines {
        let (tokens, _) = eng.generate_ids(line, 20_260_905).expect("generate");
        steps += tokens.len();
    }
    let wall = t0.elapsed().as_secs_f64();
    let p = PROFILE.lock().unwrap().take().unwrap_or_default();
    let mut rows: Vec<(String, f64, u64)> = p
        .ops
        .into_iter()
        .map(|(k, (ns, n))| (k, ns as f64 / 1e6 / steps as f64, n))
        .collect();
    rows.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let step_rows: Vec<_> = rows.iter().filter(|r| !r.0.starts_with("prefill")).collect();
    let total: f64 = step_rows.iter().map(|r| r.1).sum();
    let mm: f64 = step_rows
        .iter()
        .filter(|r| r.0.starts_with("MUL_MAT") && !r.0.ends_with("head"))
        .map(|r| r.1)
        .sum();
    let head: f64 = step_rows.iter().filter(|r| r.0.ends_with("head")).map(|r| r.1).sum();
    println!(
        "op_profile: {} lines, {steps} steps, {threads} threads, {wall:.1} s observed ({:.1} ms a step with the observer)",
        lines.len(),
        wall * 1e3 / steps as f64
    );
    println!(
        "per step: observed nodes {total:.2} ms = layer products {mm:.2} + head {head:.2} + other ops {:.2}",
        total - mm - head
    );
    for (k, ms, n) in &rows {
        println!("  {ms:8.3} ms/step  {n:6} nodes  {k}");
    }
    if let Some(path) = a.get("json") {
        let body: Vec<String> = rows
            .iter()
            .map(|(k, ms, n)| format!("{{\"key\":\"{k}\",\"ms_per_step\":{ms:.4},\"nodes\":{n}}}"))
            .collect();
        std::fs::write(
            path,
            format!(
                "{{\"steps\":{steps},\"threads\":{threads},\"layer_products_ms\":{mm:.4},\"head_ms\":{head:.4},\"observed_ms\":{total:.4},\"ops\":[{}]}}\n",
                body.join(",")
            ),
        )
        .expect("write --json");
    }
}
