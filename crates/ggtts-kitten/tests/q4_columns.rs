//! ggml's CPU `Q4_0` product, one column against many, on synthetic ternary weights (no model needed).
//!
//! The single-token check (DESIGN.md, level 2, "single-token path") found the ternary arm's one-token forward
//! differing from the batched one by rel L2 `1.4e-3` on the CPU, while the f32 arm agreed to `1.3e-7` (ggtts#21).
//! These tests separate the two possible causes:
//!
//! 1. the kernel: a one-column product computed differently from a many-column one. At the pin, ggml's CPU
//!    `mul_mat` quantizes every column of the activations to `Q8_0` with the same `from_float`, and computes each
//!    output with the same `vec_dot` whatever the column count (`ggml-cpu/ggml-cpu.c:1322-1356`, then `:1243`);
//!    the `llamafile_sgemm` branch (`:1295`, `:1366`) is compiled only with `GGML_LLAMAFILE`, which is off in
//!    this build. So the products should agree bit for bit, and the first test checks that.
//! 2. amplification: `Q8_0` rounds each activation to one of 255 levels of its block's absmax. An input that moves
//!    by `1e-7` moves almost no output, except where it carries an activation across a rounding boundary, and there
//!    the output jumps by a whole level (about `absmax / 127` times the weight). The second test measures that gain.
#![cfg(feature = "ggml")]

use ggtts_ggml_sys as g;
use ggtts_kitten::ternary::{pack_row_q4, Rule};

const ROWS: usize = 256;
const COLS: usize = 2048;
const NCOL: usize = 16;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Ternary rows with an fp16-exact scale per 128 weights, packed as this crate's converter packs them.
fn weights(rng: &mut Lcg) -> Vec<u8> {
    let mut packed = Vec::new();
    for _ in 0..ROWS {
        let row: Vec<f32> = (0..COLS)
            .map(|c| {
                let scale = (1 + (c / 128) % 7) as f32 / 64.0;
                let u = rng.next();
                if u < 0.3 {
                    -scale
                } else if u < 0.7 {
                    0.0
                } else {
                    scale
                }
            })
            .collect();
        pack_row_q4(&row, Rule::KITTEN, true, &mut packed).expect("exact ternary");
    }
    packed
}

/// Activations with a heavy tail, as a transformer's are.
fn activations(rng: &mut Lcg) -> Vec<f32> {
    (0..COLS * NCOL)
        .map(|_| {
            let u = rng.next() * 2.0 - 1.0;
            (u * u * u * 8.0) as f32
        })
        .collect()
}

/// `W x` for all NCOL columns at once, and for each column alone, from one graph on the CPU.
fn products(w: &[u8], x: &[f32], threads: i32) -> (Vec<f32>, Vec<Vec<f32>>) {
    products_of(g::GGML_TYPE_Q4_0, w, x, threads)
}

/// The same with the matrix in any type ggml's CPU multiplies (`w` is its raw bytes).
fn products_of(ty: g::ggml_type, w: &[u8], x: &[f32], threads: i32) -> (Vec<f32>, Vec<Vec<f32>>) {
    unsafe {
        let ctx = g::ggml_init(g::ggml_init_params {
            mem_size: 64 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        });
        assert!(!ctx.is_null());
        let wt = g::ggml_new_tensor_2d(ctx, ty, COLS as i64, ROWS as i64);
        assert_eq!(g::ggml_nbytes(wt), w.len());
        std::ptr::copy_nonoverlapping(w.as_ptr(), (*wt).data.cast::<u8>(), w.len());
        let xt = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, COLS as i64, NCOL as i64);
        std::ptr::copy_nonoverlapping(x.as_ptr(), (*xt).data.cast::<f32>(), x.len());
        let gf = g::ggml_new_graph(ctx);
        let all = g::ggml_mul_mat(ctx, wt, xt);
        g::ggml_build_forward_expand(gf, all);
        let nb1 = (*xt).nb[1];
        let ones: Vec<_> = (0..NCOL)
            .map(|j| {
                let xj = g::ggml_view_2d(ctx, xt, COLS as i64, 1, nb1, j * nb1);
                let y = g::ggml_mul_mat(ctx, wt, xj);
                g::ggml_build_forward_expand(gf, y);
                y
            })
            .collect();
        assert_eq!(g::ggml_graph_compute_with_ctx(ctx, gf, threads), g::GGML_STATUS_SUCCESS);
        let read = |t: *mut g::ggml_tensor, n: usize| std::slice::from_raw_parts((*t).data.cast::<f32>(), n).to_vec();
        let a = read(all, ROWS * NCOL);
        let o = ones.iter().map(|&t| read(t, ROWS)).collect();
        g::ggml_free(ctx);
        (a, o)
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        n += (f64::from(*x) - f64::from(*y)).powi(2);
        d += f64::from(*y).powi(2);
    }
    (n / d).sqrt()
}

#[test]
fn one_column_and_many_columns_are_bit_identical() {
    let mut rng = Lcg(42);
    let w = weights(&mut rng);
    let x = activations(&mut rng);
    for threads in [1, 4] {
        let (all, ones) = products(&w, &x, threads);
        for (j, one) in ones.iter().enumerate() {
            let many = &all[j * ROWS..(j + 1) * ROWS];
            let differ = one.iter().zip(many).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(
                differ, 0,
                "{threads} threads, column {j}: {differ} of {ROWS} outputs differ"
            );
        }
    }
}

#[test]
fn q8_activation_rounding_amplifies_a_tiny_input_difference() {
    let mut rng = Lcg(7);
    let w = weights(&mut rng);
    let x = activations(&mut rng);
    // The same weights as exact f32, the control: no activation rounding.
    let wf: Vec<u8> = w
        .chunks(COLS / 32 * 18)
        .flat_map(|r| ggtts_kitten::ternary::dequantize_row_q4(r, COLS).expect("dequantize"))
        .flat_map(f32::to_le_bytes)
        .collect();
    // A relative perturbation of 1e-6 per activation. (At 1e-7, the f32 arm's batched-against-single difference,
    // a matrix this small crosses about one rounding boundary in expectation; 28 layers of 328 positions cross many.)
    let xp: Vec<f32> = x
        .iter()
        .map(|v| v * (1.0 + 1.0e-6 * (rng.next() as f32 * 2.0 - 1.0)))
        .collect();
    let input = rel_l2(&xp, &x);
    let gain = |ty, w: &[u8]| {
        let (a, _) = products_of(ty, w, &x, 4);
        let (b, _) = products_of(ty, w, &xp, 4);
        rel_l2(&b, &a) / input
    };
    let q4 = gain(g::GGML_TYPE_Q4_0, &w);
    let f32g = gain(g::GGML_TYPE_F32, &wf);
    eprintln!("input rel L2 {input:.2e}; output gain: Q4_0 (Q8_0 activations) {q4:.1}, f32 {f32g:.2}");
    // Through an exact product the output moves by about as much as the input. Through Q8_0 activations it moves by
    // whole rounding levels wherever an activation crosses a boundary.
    assert!(
        f32g < 3.0,
        "the f32 control should pass the perturbation through: gain {f32g:.2}"
    );
    assert!(
        q4 > 10.0 * f32g,
        "expected Q8_0 rounding to amplify it: Q4_0 {q4:.1}, f32 {f32g:.2}"
    );
}
