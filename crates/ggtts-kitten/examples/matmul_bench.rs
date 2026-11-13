//! matmul_bench: one-token matrix products at Kitten-TTS 2's LM shapes, per storage type, timed on this machine.
//!
//! ```text
//! cargo run --release -p ggtts-kitten --features ggml --example matmul_bench -- [--threads 4] [--iters 20] [--json FILE]
//! ```
//!
//! For each type (padded `TQ1_0`, padded `TQ2_0`, `Q4_0`) and each of the LM's seven matrix shapes, it builds one ggml
//! graph of 28 independent products, one per layer, each with its own weights, as a generation step reads them. It
//! times the whole graph and reports the median, so a type's per-step cost is the sum of its seven shapes. The weights
//! are random exact-ternary groups (one scale per 128 weights, values -s, 0 or +s), packed by `Layout::pack_row`, the
//! converter's own packer, so every byte is in the format the engine loads. Activations are random f32; ggml quantizes
//! them as the engine's products do (`Q8_0` for `Q4_0`, `Q8_K` for the padded types). It then times the output head:
//! the 158,241 by 2048 token table, in bf16 as shipped, and in f16 for comparison.
//!
//! The times are of products alone. A real step adds attention, norms and the graph's other ops, and builds its graph
//! once per step through the scheduler, so the sum here is a lower bound on the LM step.

use ggtts_ggml_sys as g;
use ggtts_kitten::ternary::{Layout, Rule, GROUP};
use std::time::Instant;

const LAYERS: usize = 28;
/// (name, input columns, output rows), as the GGUF stores them.
const SHAPES: [(&str, usize, usize); 7] = [
    ("q_proj", 2048, 2048),
    ("k_proj", 2048, 1024),
    ("v_proj", 2048, 1024),
    ("o_proj", 2048, 2048),
    ("gate_proj", 2048, 6144),
    ("up_proj", 2048, 6144),
    ("down_proj", 6144, 2048),
];
const VOCAB: usize = 158_241;
const EMBD: usize = 2048;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn ggml_type(layout: Layout) -> g::ggml_type {
    match layout {
        Layout::Q4 => g::GGML_TYPE_Q4_0,
        Layout::Tq1Pad | Layout::Tq1 => g::GGML_TYPE_TQ1_0,
        Layout::Tq2Pad => g::GGML_TYPE_TQ2_0,
    }
}

/// One matrix of exact-ternary groups, packed as `layout` stores it.
fn packed_matrix(layout: Layout, cols: usize, rows: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * layout.row_bytes(cols));
    let mut row = vec![0f32; cols];
    for _ in 0..rows {
        for grp in row.chunks_mut(GROUP) {
            // The packer stores each group's scale as f16 and refuses one that is not exact there.
            let s = unsafe { g::ggml_fp16_to_fp32(g::ggml_fp32_to_fp16(0.01 + 0.04 * rng.unit())) };
            for v in grp.iter_mut() {
                *v = match rng.next() % 3 {
                    0 => -s,
                    1 => 0.0,
                    _ => s,
                };
            }
        }
        layout
            .pack_row(&row, Rule::KITTEN, true, &mut out)
            .expect("exact ternary packs");
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time `iters` computes of one graph holding `mats` products (weights `ty`, `ne0` by `rows`, one f32 column each).
unsafe fn time_products(
    ty: g::ggml_type,
    ne0: usize,
    rows: usize,
    weights: &[Vec<u8>],
    threads: i32,
    iters: usize,
    rng: &mut Rng,
) -> f64 {
    let mut bytes = 0usize;
    for w in weights {
        bytes += w.len();
    }
    let ctx = g::ggml_init(g::ggml_init_params {
        mem_size: bytes
            + weights.len() * (ne0 * 4 + rows * 4 + 4 * g::ggml_tensor_overhead())
            + g::ggml_graph_overhead()
            + (64 << 20),
        mem_buffer: std::ptr::null_mut(),
        no_alloc: false,
    });
    assert!(!ctx.is_null(), "ggml_init");
    let gf = g::ggml_new_graph(ctx);
    for w in weights {
        let wt = g::ggml_new_tensor_2d(ctx, ty, ne0 as i64, rows as i64);
        assert_eq!(g::ggml_nbytes(wt), w.len(), "packed size");
        std::ptr::copy_nonoverlapping(w.as_ptr(), (*wt).data.cast::<u8>(), w.len());
        let xt = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, ne0 as i64, 1);
        let x = std::slice::from_raw_parts_mut((*xt).data.cast::<f32>(), ne0);
        for v in x.iter_mut() {
            *v = rng.unit() * 2.0 - 1.0;
        }
        g::ggml_build_forward_expand(gf, g::ggml_mul_mat(ctx, wt, xt));
    }
    // One warm-up compute, then the timed ones.
    assert_eq!(g::ggml_graph_compute_with_ctx(ctx, gf, threads), g::GGML_STATUS_SUCCESS);
    let mut t = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        assert_eq!(g::ggml_graph_compute_with_ctx(ctx, gf, threads), g::GGML_STATUS_SUCCESS);
        t.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    g::ggml_free(ctx);
    median(t)
}

fn main() {
    let mut threads = 4i32;
    let mut iters = 20usize;
    let mut json: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()).expect("--threads N"),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).expect("--iters N"),
            "--json" => json = args.next(),
            _ => panic!("unknown argument {a}; see the file's header"),
        }
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut records = Vec::new();
    println!("matmul_bench: {LAYERS} layers, {threads} threads, median of {iters}, ms per step (all {LAYERS} layers)");
    for layout in [Layout::Tq1Pad, Layout::Tq2Pad, Layout::Q4] {
        let mut total = 0.0;
        let mut line = format!("{:8}", layout.name());
        for (name, cols, rows) in SHAPES {
            let weights: Vec<Vec<u8>> = (0..LAYERS)
                .map(|_| packed_matrix(layout, cols, rows, &mut rng))
                .collect();
            let ms = unsafe {
                time_products(
                    ggml_type(layout),
                    layout.stored_cols(cols),
                    rows,
                    &weights,
                    threads,
                    iters,
                    &mut rng,
                )
            };
            total += ms;
            line += &format!(" {name} {ms:.2}");
            records.push(format!(
                "{{\"type\":\"{}\",\"shape\":\"{name}\",\"cols\":{cols},\"rows\":{rows},\"layers\":{LAYERS},\"ms_per_step\":{ms:.4}}}",
                layout.name()
            ));
        }
        println!("{line} | total {total:.2}");
        records.push(format!(
            "{{\"type\":\"{}\",\"shape\":\"all\",\"ms_per_step\":{total:.4}}}",
            layout.name()
        ));
    }
    // The output head: the token table as a matrix, once per step.
    for (label, ty) in [("bf16", g::GGML_TYPE_BF16), ("f16", g::GGML_TYPE_F16)] {
        let mut w = vec![0u8; VOCAB * EMBD * 2];
        for c in w.chunks_mut(2) {
            // A small random value in either format: the top byte of an f32 is a valid bf16, and f16 is built directly.
            let v = (rng.unit() - 0.5) * 0.1;
            let b = if ty == g::GGML_TYPE_BF16 {
                (v.to_bits() >> 16) as u16
            } else {
                unsafe { g::ggml_fp32_to_fp16(v) }
            };
            c.copy_from_slice(&b.to_le_bytes());
        }
        let ms = unsafe { time_products(ty, EMBD, VOCAB, &[w], threads, iters.max(5) / 2, &mut rng) };
        println!("head {label:5} {ms:.2}");
        records.push(format!(
            "{{\"type\":\"head-{label}\",\"shape\":\"head\",\"cols\":{EMBD},\"rows\":{VOCAB},\"ms_per_step\":{ms:.4}}}"
        ));
    }
    if let Some(path) = json {
        std::fs::write(&path, format!("[{}]\n", records.join(","))).expect("write --json");
        println!("wrote {path}");
    }
}
