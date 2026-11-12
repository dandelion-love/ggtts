//! ggml's CPU product on padded `TQ1_0` (`tq1_0-pad128`), on synthetic ternary weights (no model needed).
//!
//! The padded layouts store each 128-weight group in the first half of a 256-weight block, and the model pads every
//! activation the same way (`[128, n / 128, T]` padded to `[256, n / 128, T]` with zeros). These tests check what
//! that arrangement claims, with bars written before running them:
//!
//! 1. padded `TQ1_0` and padded `TQ2_0` hold the same trits and scales and both take `Q8_K` activations, so their
//!    products agree bit for bit;
//! 2. the padding half contributes exactly 0: with zero activations there, nonzero trits planted in the padding
//!    half leave the product bit-identical;
//! 3. the control for 2: the same planted trits with a nonzero activation in the padding half do change the
//!    product, so the comparison in 2 can see the padding;
//! 4. against the unpadded product in f64 (`W x` on the source values), the padded product is within the `Q8_K`
//!    activation rounding, rel L2 below `1e-2`, and a sign-flipped row is far outside it.
#![cfg(feature = "ggml")]

use ggtts_ggml_sys as g;
use ggtts_kitten::ternary::{pack_row, pack_row_tq1, tq1_decode, tq1_encode, Rule, TQ1_0_BLOCK_BYTES};

const ROWS: usize = 64;
const COLS: usize = 1024;
const NCOL: usize = 3;
const GROUP: usize = 128;

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

/// Ternary rows with a different fp16-exact scale in every 128-weight group, as the trained matrices have.
fn source(rng: &mut Lcg) -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|i| {
            let (r, c) = (i / COLS, i % COLS);
            let scale = (1 + (r + c / GROUP) % 11) as f32 / 128.0;
            let u = rng.next();
            if c % GROUP == 0 {
                scale // the absmax is the scale
            } else if u < 0.3 {
                -scale
            } else if u < 0.7 {
                0.0
            } else {
                scale
            }
        })
        .collect()
}

fn activations(rng: &mut Lcg) -> Vec<f32> {
    (0..COLS * NCOL)
        .map(|_| {
            let u = rng.next() * 2.0 - 1.0;
            (u * u * u * 8.0) as f32
        })
        .collect()
}

/// Each column's 128-groups followed by 128 copies of `pad` (zero in the model).
fn padded(x: &[f32], pad: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * x.len());
    for chunk in x.chunks(GROUP) {
        out.extend_from_slice(chunk);
        out.extend(std::iter::repeat_n(pad, GROUP));
    }
    out
}

fn product(ty: g::ggml_type, w: &[u8], xp: &[f32]) -> Vec<f32> {
    unsafe {
        let ctx = g::ggml_init(g::ggml_init_params {
            mem_size: 32 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        });
        assert!(!ctx.is_null());
        let wt = g::ggml_new_tensor_2d(ctx, ty, 2 * COLS as i64, ROWS as i64);
        assert_eq!(g::ggml_nbytes(wt), w.len());
        std::ptr::copy_nonoverlapping(w.as_ptr(), (*wt).data.cast::<u8>(), w.len());
        let xt = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, 2 * COLS as i64, NCOL as i64);
        std::ptr::copy_nonoverlapping(xp.as_ptr(), (*xt).data.cast::<f32>(), xp.len());
        let gf = g::ggml_new_graph(ctx);
        let y = g::ggml_mul_mat(ctx, wt, xt);
        g::ggml_build_forward_expand(gf, y);
        assert_eq!(g::ggml_graph_compute_with_ctx(ctx, gf, 2), g::GGML_STATUS_SUCCESS);
        let out = std::slice::from_raw_parts((*y).data.cast::<f32>(), ROWS * NCOL).to_vec();
        g::ggml_free(ctx);
        out
    }
}

fn pack(w: &[f32], tq1: bool) -> Vec<u8> {
    let mut out = Vec::new();
    for row in w.chunks(COLS) {
        if tq1 {
            pack_row_tq1(row, Rule::KITTEN, true, &mut out).unwrap();
        } else {
            pack_row(row, Rule::KITTEN, true, &mut out).unwrap();
        }
    }
    out
}

/// The same TQ1 bytes with every padding trit set to +1 or -1 (alternating by block), re-encoded canonically.
fn with_padding_trits(tq1: &[u8]) -> Vec<u8> {
    let mut out = tq1.to_vec();
    for (b, blk) in out.chunks_exact_mut(TQ1_0_BLOCK_BYTES).enumerate() {
        let mut t = tq1_decode(&blk[..52]).unwrap();
        for (j, x) in t[GROUP..].iter_mut().enumerate() {
            *x = if (b + j) % 2 == 0 { 1 } else { -1 };
        }
        blk[..52].copy_from_slice(&tq1_encode(&t));
    }
    out
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        n += (f64::from(*x) - y).powi(2);
        d += y.powi(2);
    }
    (n / d).sqrt()
}

fn reference(w: &[f32], x: &[f32]) -> Vec<f64> {
    let mut y = vec![0f64; ROWS * NCOL];
    for c in 0..NCOL {
        for r in 0..ROWS {
            y[c * ROWS + r] = (0..COLS)
                .map(|k| f64::from(w[r * COLS + k]) * f64::from(x[c * COLS + k]))
                .sum();
        }
    }
    y
}

#[test]
fn padded_tq1_equals_padded_tq2_bit_for_bit() {
    let mut rng = Lcg(7);
    let w = source(&mut rng);
    let x = padded(&activations(&mut rng), 0.0);
    let y1 = product(g::GGML_TYPE_TQ1_0, &pack(&w, true), &x);
    let y2 = product(g::GGML_TYPE_TQ2_0, &pack(&w, false), &x);
    assert!(bits_equal(&y1, &y2), "padded TQ1_0 and padded TQ2_0 products differ");
}

#[test]
fn the_padding_half_contributes_exactly_zero_and_the_control_sees_it() {
    let mut rng = Lcg(11);
    let w = source(&mut rng);
    let xs = activations(&mut rng);
    let clean = pack(&w, true);
    let planted = with_padding_trits(&clean);
    assert_ne!(clean, planted);
    let x0 = padded(&xs, 0.0);
    let y = product(g::GGML_TYPE_TQ1_0, &clean, &x0);
    assert!(
        bits_equal(&y, &product(g::GGML_TYPE_TQ1_0, &planted, &x0)),
        "nonzero padding trits changed the product with zero padding activations"
    );
    // The control: the same planted trits against a nonzero padding activation must move the product.
    let x1 = padded(&xs, 0.5);
    assert!(
        !bits_equal(&y, &product(g::GGML_TYPE_TQ1_0, &planted, &x1)),
        "the comparison cannot see the padding half"
    );
    // Not asserted: the clean matrix against that nonzero padding is not bit-identical either, though its padding
    // trits are zero, because  takes one activation scale per 256 values and the padding enters that absmax.
    // That is why the padding must be zero, which  gives.
}

#[test]
fn the_padded_product_matches_the_unpadded_one_within_activation_rounding() {
    let mut rng = Lcg(13);
    let w = source(&mut rng);
    let xs = activations(&mut rng);
    let want = reference(&w, &xs);
    let y = product(g::GGML_TYPE_TQ1_0, &pack(&w, true), &padded(&xs, 0.0));
    let e = rel_l2(&y, &want);
    eprintln!("padded TQ1_0 against the unpadded f64 product: rel L2 {e:.3e}");
    assert!(e < 1e-2, "rel L2 {e}");
    // The control: one row's signs flipped in the source is far outside the bar.
    let mut bad = w.clone();
    for v in &mut bad[5 * COLS..6 * COLS] {
        *v = -*v;
    }
    let y_bad = product(g::GGML_TYPE_TQ1_0, &pack(&bad, true), &padded(&xs, 0.0));
    let e_bad = rel_l2(&y_bad, &want);
    eprintln!("with one row's signs flipped: rel L2 {e_bad:.3e}");
    assert!(e_bad > 1e-1, "rel L2 {e_bad}");
}
