//! Kitten's ternary rule, and its storage as ggml `TQ2_0` on zero-padded 128-weight groups.
//!
//! The rule (DESIGN.md, "The ternary type"): one scale per `GROUP` consecutive weights along a row (the input
//! dimension), the scale is the group's absmax, and a weight's trit is `round_half_away_from_zero(w / absmax)`.
//!
//! The storage: a `TQ2_0` block is 256 weights, 64 code bytes and one fp16 scale. Its first 32 code bytes code
//! weights 0..128 and its last 32 code weights 128..256 (ggml-quants.c `quantize_row_tq2_0_ref`: byte `j + m`, bit
//! pair `n`, weight `j / 32 * 128 + n * 32 + m`, code `trit + 1`). One Kitten group goes in the first half of a
//! block, zeros in the second, and the block's scale is the group's scale. A row of `in` weights becomes a `TQ2_0`
//! row of `2 * in`.

use crate::safetensors::{f16_to_f32, f32_to_f16_exact};

/// Kitten's group size.
pub const GROUP: usize = 128;
/// `QK_K`.
pub const BLOCK: usize = 256;
/// `sizeof(block_tq2_0)`: 64 code bytes and an fp16 scale.
pub const TQ2_0_BLOCK_BYTES: usize = 66;
/// Stock ggml TQ1_0 packs 256 values and one fp16 scale in 54 bytes.
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// A half of zero codes: four fields of code 1 per byte.
const ZERO_CODES: u8 = 0b0101_0101;

/// How the scale of a group is taken. Only `AbsMax` is Kitten's; `AbsMean` exists to plant a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleRule {
    AbsMax,
    AbsMean,
}

/// The rule's parameters. `Rule::KITTEN` is the only correct one; the others are planted faults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub group: usize,
    pub scale: ScaleRule,
}

impl Rule {
    pub const KITTEN: Rule = Rule {
        group: GROUP,
        scale: ScaleRule::AbsMax,
    };
}

/// C `lroundf`: halves away from zero.
fn round_half_away(x: f32) -> f32 {
    (x.abs() + 0.5).floor().copysign(x)
}

/// Quantize one row under `rule`: a trit for every weight, and per 128-weight storage group the scale its group
/// was given. With a group larger than 128 (a planted fault) each storage group carries the larger group's scale.
pub fn quantize_row(row: &[f32], rule: Rule) -> (Vec<i8>, Vec<f32>) {
    assert!(
        rule.group % GROUP == 0 && row.len() % rule.group == 0,
        "group must be a multiple of 128 dividing the row"
    );
    let mut trits = vec![0i8; row.len()];
    let mut scales = Vec::with_capacity(row.len() / GROUP);
    for (gi, g) in row.chunks(rule.group).enumerate() {
        let s = match rule.scale {
            ScaleRule::AbsMax => g.iter().fold(0f32, |m, v| m.max(v.abs())),
            ScaleRule::AbsMean => g.iter().map(|v| v.abs()).sum::<f32>() / g.len() as f32,
        };
        let inv = if s == 0.0 { 0.0 } else { 1.0 / s };
        for (j, v) in g.iter().enumerate() {
            trits[gi * rule.group + j] = round_half_away(v * inv).clamp(-1.0, 1.0) as i8;
        }
        scales.extend(std::iter::repeat_n(s, rule.group / GROUP));
    }
    (trits, scales)
}

/// Why a row is not exactly representable.
#[derive(Debug, Clone, PartialEq)]
pub enum Inexact {
    /// A weight is not `trit * scale` for its group's scale.
    NotTernary { col: usize, value: f32, scale: f32 },
    /// A scale has no exact fp16.
    ScaleNotF16 { group: usize, scale: f32 },
    /// The lossy 256-weight TQ1 layout is read, never written (`tq1_0-pad128` is the exact one).
    UnsupportedTq1Export,
}

/// Pack one row into padded `TQ2_0` blocks (`row.len() / 128` blocks). In strict mode a row that the trits and
/// fp16 scales do not reproduce value for value is refused; otherwise it is packed as the rule says, faults and
/// all, which is how a planted fault reaches the readback check.
pub fn pack_row(row: &[f32], rule: Rule, strict: bool, out: &mut Vec<u8>) -> Result<(), Inexact> {
    let (trits, scales) = quantize_row(row, rule);
    for (g, chunk) in trits.chunks(GROUP).enumerate() {
        let s = scales[g];
        let h = match f32_to_f16_exact(s) {
            Some(h) => h,
            None if strict => return Err(Inexact::ScaleNotF16 { group: g, scale: s }),
            None => half_round(s),
        };
        if strict {
            for (j, &t) in chunk.iter().enumerate() {
                let v = row[g * GROUP + j];
                if f32::from(t) * s != v {
                    return Err(Inexact::NotTernary {
                        col: g * GROUP + j,
                        value: v,
                        scale: s,
                    });
                }
            }
        }
        let mut qs = [ZERO_CODES; 64];
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                q |= ((chunk[n * 32 + m] + 1) as u8 & 3) << (2 * n);
            }
            qs[m] = q;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(())
}

/// Round-to-nearest-even f32 to f16, for non-strict (planted) packing only.
fn half_round(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let man = bits & 0x7f_ffff;
    let mut h = (u32::from(sign) | ((exp as u32) << 10) | (man >> 13)) as u16;
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    h
}

/// One 128-weight group read back from a padded block: its trits, its scale, and whether the padding half is all
/// zero codes.
pub struct Group {
    pub trits: [i8; GROUP],
    pub scale: f32,
    pub padding_is_zero: bool,
}

/// Unpack block `b` of a padded `TQ2_0` row.
pub fn unpack_group(row: &[u8], b: usize) -> Result<Group, String> {
    let blk = &row[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
    let mut trits = [0i8; GROUP];
    for m in 0..32 {
        for n in 0..4 {
            let code = (blk[m] >> (2 * n)) & 3;
            if code == 3 {
                return Err(format!("block {b}: code 3 is not a trit"));
            }
            trits[n * 32 + m] = code as i8 - 1;
        }
    }
    Ok(Group {
        trits,
        scale: f16_to_f32(u16::from_le_bytes([blk[64], blk[65]])),
        padding_is_zero: blk[32..64].iter().all(|&q| q == ZERO_CODES),
    })
}

/// Dequantize a padded `TQ2_0` row back to its `n` weights.
pub fn dequantize_row(row: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(n);
    for b in 0..n / GROUP {
        let g = unpack_group(row, b)?;
        out.extend(g.trits.iter().map(|&t| f32::from(t) * g.scale));
    }
    Ok(out)
}

/// The readback check for one row: every trit and every scale read back equals what Kitten's rule gives on the
/// source row (scales compared as f32 bits), every weight read back equals the source's value, and every padding
/// half is zero. Returns the first mismatch.
pub fn check_row(src: &[f32], packed: &[u8]) -> Result<(), String> {
    let (trits, scales) = quantize_row(src, Rule::KITTEN);
    for (b, &scale) in scales.iter().enumerate() {
        let g = unpack_group(packed, b)?;
        if !g.padding_is_zero {
            return Err(format!("group {b}: the padding half is not zero"));
        }
        if g.scale.to_bits() != scale.to_bits() {
            return Err(format!("group {b}: scale {} read back, source absmax {scale}", g.scale));
        }
        for j in 0..GROUP {
            let col = b * GROUP + j;
            if g.trits[j] != trits[col] {
                return Err(format!(
                    "col {col}: trit {} read back, source {}",
                    g.trits[j], trits[col]
                ));
            }
            // Value equality: -0.0 == 0.0, the one thing a trit cannot carry (DESIGN.md, parity level 1).
            let v = f32::from(g.trits[j]) * g.scale;
            if v != src[col] {
                return Err(format!("col {col}: {v} read back, source {}", src[col]));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::safetensors::bf16_to_f32;

    /// A deterministic ternary row in Kitten's form: per group a bf16 scale, weights in {-s, -0, 0, +s}.
    pub(crate) fn kitten_row(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as u32
        };
        let mut row = Vec::with_capacity(n);
        for _ in 0..n / GROUP {
            let s = bf16_to_f32(0x3c00 + (next() % 0x300) as u16); // about 0.0078 to 0.06
            for j in 0..GROUP {
                let v = match next() % 4 {
                    0 => -s,
                    1 => s,
                    2 => -0.0,
                    _ => 0.0,
                };
                row.push(if j == 0 { s } else { v }); // the absmax is the scale
            }
        }
        row
    }

    #[test]
    fn layout_matches_ggml() {
        // Weight 0 sits in byte 0 bits 0-1, weight 32 in byte 0 bits 2-3, weight 1 in byte 1 bits 0-1.
        let mut row = vec![0f32; GROUP];
        row[0] = 1.0;
        row[32] = -1.0;
        row[1] = 1.0;
        let mut out = Vec::new();
        pack_row(&row, Rule::KITTEN, true, &mut out).unwrap();
        assert_eq!(out.len(), TQ2_0_BLOCK_BYTES);
        assert_eq!(out[0], 0b0101_0010);
        assert_eq!(out[0] & 3, 2);
        assert_eq!((out[0] >> 2) & 3, 0);
        assert_eq!(out[1] & 3, 2);
        assert!(out[32..64].iter().all(|&q| q == ZERO_CODES));
        assert_eq!(u16::from_le_bytes([out[64], out[65]]), 0x3c00);
    }

    #[test]
    fn kitten_rows_pack_exactly_and_read_back() {
        for seed in 0..20 {
            let row = kitten_row(2048, seed);
            let mut out = Vec::new();
            pack_row(&row, Rule::KITTEN, true, &mut out).unwrap();
            check_row(&row, &out).unwrap();
            let back = dequantize_row(&out, row.len()).unwrap();
            assert!(back.iter().zip(&row).all(|(a, b)| a == b));
        }
    }

    #[test]
    fn planted_group_256_is_caught() {
        let row = kitten_row(2048, 7);
        let wrong = Rule {
            group: 256,
            scale: ScaleRule::AbsMax,
        };
        assert!(matches!(
            pack_row(&row, wrong, true, &mut Vec::new()),
            Err(Inexact::NotTernary { .. })
        ));
        let mut out = Vec::new();
        pack_row(&row, wrong, false, &mut out).unwrap();
        assert!(check_row(&row, &out).is_err(), "a group of 256 must fail the readback");
    }

    #[test]
    fn planted_absmean_is_caught() {
        let row = kitten_row(2048, 11);
        let wrong = Rule {
            group: GROUP,
            scale: ScaleRule::AbsMean,
        };
        let mut out = Vec::new();
        pack_row(&row, wrong, false, &mut out).unwrap();
        assert!(
            check_row(&row, &out).is_err(),
            "a scale from the mean must fail the readback"
        );
    }

    #[test]
    fn a_non_ternary_row_is_refused() {
        let mut row = kitten_row(256, 3);
        row[5] = row[0] * 0.5;
        assert!(matches!(
            pack_row(&row, Rule::KITTEN, true, &mut Vec::new()),
            Err(Inexact::NotTernary { col: 5, .. })
        ));
    }

    #[test]
    fn a_scale_without_an_exact_half_is_refused() {
        let mut row = vec![0f32; GROUP];
        row[0] = 1.0 + f32::EPSILON;
        assert!(matches!(
            pack_row(&row, Rule::KITTEN, true, &mut Vec::new()),
            Err(Inexact::ScaleNotF16 { .. })
        ));
    }

    #[test]
    fn a_corrupted_byte_is_caught() {
        let row = kitten_row(1024, 5);
        let mut out = Vec::new();
        pack_row(&row, Rule::KITTEN, true, &mut out).unwrap();
        out[TQ2_0_BLOCK_BYTES + 3] ^= 0b11;
        assert!(check_row(&row, &out).is_err());
    }
}

/// `sizeof(block_q4_0)`: an fp16 scale and 16 bytes of 4-bit codes for 32 weights.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// How the ternary matrices are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Experimental stock TQ1_0, one scale per 256 values. Read only: two Kitten groups share a scale, so it is lossy
    /// for this model, and the converter does not write it.
    Tq1,
    /// Stock `TQ1_0`, one Kitten group per 256-weight block with zeros in the second half (`Q8_K` activations), as
    /// [`Layout::Tq2Pad`] does with `TQ2_0`. Exact: the block's scale is its group's.
    Tq1Pad,
    /// `TQ2_0`, one Kitten group per 256-weight block with zeros in the second half (`Q8_K` activations).
    Tq2Pad,
    /// Stock `Q4_0`: each 32-weight block carries its group's scale as `d` and codes 7, 8, 9 for -1, 0, +1, which is
    /// exact; activations are `Q8_0`, quantized per 32.
    Q4,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::Tq1 => "tq1_0-256-experimental",
            Layout::Tq1Pad => "tq1_0-pad128",
            Layout::Tq2Pad => "tq2_0-pad128",
            Layout::Q4 => "q4_0-g128",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "tq1_0-256-experimental" => Some(Layout::Tq1),
            "tq1_0-pad128" | "tq1_0" => Some(Layout::Tq1Pad),
            "tq2_0-pad128" | "tq2_0" => Some(Layout::Tq2Pad),
            "q4_0-g128" | "q4_0" => Some(Layout::Q4),
            _ => None,
        }
    }
    /// True for the layouts that store each 128-weight group in its own 256-weight block, so every activation
    /// entering a ternary matrix is padded the same way at run time.
    pub fn padded(self) -> bool {
        matches!(self, Layout::Tq2Pad | Layout::Tq1Pad)
    }
    /// The ggml type the layout's matrices are stored as.
    pub fn ggml_type(self) -> crate::gguf::GgmlType {
        match self {
            Layout::Tq1 | Layout::Tq1Pad => crate::gguf::GgmlType::TQ1_0,
            Layout::Tq2Pad => crate::gguf::GgmlType::TQ2_0,
            Layout::Q4 => crate::gguf::GgmlType::Q4_0,
        }
    }
    /// The stored row length (`ne[0]`) for `cols` weights.
    pub fn stored_cols(self, cols: usize) -> usize {
        match self {
            Layout::Tq1 => cols,
            Layout::Tq1Pad | Layout::Tq2Pad => 2 * cols,
            Layout::Q4 => cols,
        }
    }
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            Layout::Tq1 => cols / 256 * TQ1_0_BLOCK_BYTES,
            Layout::Tq1Pad => cols / GROUP * TQ1_0_BLOCK_BYTES,
            Layout::Tq2Pad => cols / GROUP * TQ2_0_BLOCK_BYTES,
            Layout::Q4 => cols / 32 * Q4_0_BLOCK_BYTES,
        }
    }
    pub fn pack_row(self, row: &[f32], rule: Rule, strict: bool, out: &mut Vec<u8>) -> Result<(), Inexact> {
        match self {
            Layout::Tq1 => Err(Inexact::UnsupportedTq1Export),
            Layout::Tq1Pad => pack_row_tq1(row, rule, strict, out),
            Layout::Tq2Pad => pack_row(row, rule, strict, out),
            Layout::Q4 => pack_row_q4(row, rule, strict, out),
        }
    }
    pub fn check_row(self, src: &[f32], packed: &[u8]) -> Result<(), String> {
        match self {
            Layout::Tq1 => Err("experimental TQ1 source readback is not supported".into()),
            Layout::Tq1Pad => check_row_tq1(src, packed),
            Layout::Tq2Pad => check_row(src, packed),
            Layout::Q4 => check_row_q4(src, packed),
        }
    }
    pub fn dequantize_row(self, packed: &[u8], n: usize) -> Result<Vec<f32>, String> {
        match self {
            Layout::Tq1 => dequantize_tq1(packed, n),
            Layout::Tq1Pad => dequantize_row_tq1(packed, n),
            Layout::Tq2Pad => dequantize_row(packed, n),
            Layout::Q4 => dequantize_row_q4(packed, n),
        }
    }
}

/// Decode ggml TQ1_0 blocks using the pinned ggml layout.
pub fn dequantize_tq1(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let expected = (n / 256)
        .checked_mul(TQ1_0_BLOCK_BYTES)
        .ok_or("TQ1 row size overflow")?;
    if n == 0 || n % 256 != 0 || data.len() != expected {
        return Err("bad TQ1 row size".into());
    }
    let mut out = Vec::with_capacity(n);
    const POW: [u8; 5] = [1, 3, 9, 27, 81];
    for block in data.chunks_exact(TQ1_0_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
        if !d.is_finite() || d < 0.0 {
            return Err("invalid TQ1 scale".into());
        }
        for &p in &POW {
            for &byte in &block[..32] {
                let q = byte.wrapping_mul(p);
                let trit = ((u16::from(q) * 3) >> 8) as i16 - 1;
                out.push(f32::from(trit) * d);
            }
        }
        for &p in &POW {
            for &byte in &block[32..48] {
                let q = byte.wrapping_mul(p);
                let trit = ((u16::from(q) * 3) >> 8) as i16 - 1;
                out.push(f32::from(trit) * d);
            }
        }
        for &p in &POW[..4] {
            for &byte in &block[48..52] {
                let q = byte.wrapping_mul(p);
                let trit = ((u16::from(q) * 3) >> 8) as i16 - 1;
                out.push(f32::from(trit) * d);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tq1_reference_tests {
    use super::*;

    #[test]
    fn synthetic_ggml_reference_and_corruption_control() {
        // Fixture contains 64 synthetic blocks and their float32 reference dequantization.
        let fixture = include_bytes!("../tests/fixtures/tq1-reference.bin");
        assert_eq!(fixture.len(), 64 * (54 + 1024));
        let mut detected = 0;
        let mut code_detected = 0;
        for sample in fixture.chunks_exact(54 + 1024) {
            let (bytes, values) = sample.split_at(54);
            let actual = dequantize_tq1(bytes, 256).unwrap();
            let expected: Vec<f32> = values
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                .collect();
            assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
            let mut bad = bytes.to_vec();
            bad[53] ^= 1;
            if let Ok(altered) = dequantize_tq1(&bad, 256) {
                if altered.iter().zip(&expected).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    detected += 1;
                }
            } else {
                detected += 1;
            }
            let mut bad_code = bytes.to_vec();
            bad_code[0] ^= 0xff;
            let altered = dequantize_tq1(&bad_code, 256).unwrap();
            if altered.iter().zip(&expected).any(|(a, b)| a.to_bits() != b.to_bits()) {
                code_detected += 1;
            }
        }
        assert!(detected > 0, "planted scale corruption must be detected");
        assert!(code_detected > 0, "planted code corruption must be detected");
    }

    #[test]
    fn rejects_bad_length_dimension_and_scale() {
        let block = [0u8; TQ1_0_BLOCK_BYTES];
        assert!(dequantize_tq1(&block[..53], 256).is_err());
        assert!(dequantize_tq1(&block, 128).is_err());
        assert!(dequantize_tq1(&[], 0).is_err());
        let mut bad = block;
        bad[52..54].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(dequantize_tq1(&bad, 256).is_err());
        bad[52..54].copy_from_slice(&0xbc00u16.to_le_bytes());
        assert!(dequantize_tq1(&bad, 256).is_err());
    }
}

/// Pack one row as `Q4_0` (ggml-quants.c `block_q4_0`: `d`, then byte `j` holds weight `j` in its low nibble and
/// weight `j + 16` in its high nibble; the value is `(q - 8) * d`).
pub fn pack_row_q4(row: &[f32], rule: Rule, strict: bool, out: &mut Vec<u8>) -> Result<(), Inexact> {
    let (trits, scales) = quantize_row(row, rule);
    for (g, s) in scales.iter().enumerate() {
        let h = match f32_to_f16_exact(*s) {
            Some(h) => h,
            None if strict => return Err(Inexact::ScaleNotF16 { group: g, scale: *s }),
            None => half_round(*s),
        };
        for blk in 0..GROUP / 32 {
            let base = g * GROUP + blk * 32;
            if strict {
                for j in 0..32 {
                    if f32::from(trits[base + j]) * s != row[base + j] {
                        return Err(Inexact::NotTernary {
                            col: base + j,
                            value: row[base + j],
                            scale: *s,
                        });
                    }
                }
            }
            out.extend_from_slice(&h.to_le_bytes());
            for j in 0..16 {
                let lo = (trits[base + j] + 8) as u8;
                let hi = (trits[base + j + 16] + 8) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    Ok(())
}

/// Dequantize a `Q4_0` row.
pub fn dequantize_row_q4(packed: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; n];
    for b in 0..n / 32 {
        let blk = &packed[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..16 {
            out[b * 32 + j] = (f32::from(blk[2 + j] & 15) - 8.0) * d;
            out[b * 32 + j + 16] = (f32::from(blk[2 + j] >> 4) - 8.0) * d;
        }
    }
    Ok(out)
}

/// The readback check for a `Q4_0` row: every code is 7, 8 or 9 and equals the source's trit, every block's scale
/// equals its group's absmax bit for bit, and every value equals the source's.
pub fn check_row_q4(src: &[f32], packed: &[u8]) -> Result<(), String> {
    let (trits, scales) = quantize_row(src, Rule::KITTEN);
    for b in 0..src.len() / 32 {
        let blk = &packed[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let s = scales[b * 32 / GROUP];
        if d.to_bits() != s.to_bits() {
            return Err(format!("block {b}: scale {d} read back, source absmax {s}"));
        }
        for j in 0..32 {
            let byte = blk[2 + j % 16];
            let q = if j < 16 { byte & 15 } else { byte >> 4 };
            if !(7..=9).contains(&q) {
                return Err(format!("block {b}: code {q} is not a trit"));
            }
            let col = b * 32 + j;
            let t = q as i8 - 8;
            if t != trits[col] || f32::from(t) * d != src[col] {
                return Err(format!(
                    "col {col}: {} read back, source {}",
                    f32::from(t) * d,
                    src[col]
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod q4_tests {
    use super::tests::kitten_row;
    use super::*;

    #[test]
    fn q4_packs_exactly_and_plants_are_caught() {
        let row = kitten_row(2048, 3);
        let mut out = Vec::new();
        pack_row_q4(&row, Rule::KITTEN, true, &mut out).unwrap();
        assert_eq!(out.len(), 2048 / 32 * Q4_0_BLOCK_BYTES);
        check_row_q4(&row, &out).unwrap();
        assert!(dequantize_row_q4(&out, 2048)
            .unwrap()
            .iter()
            .zip(&row)
            .all(|(a, b)| a == b));
        let mut bad = Vec::new();
        pack_row_q4(
            &row,
            Rule {
                group: 256,
                scale: ScaleRule::AbsMax,
            },
            false,
            &mut bad,
        )
        .unwrap();
        assert!(check_row_q4(&row, &bad).is_err());
    }
}

/// Write ggml's TQ1_0 byte for `trits` (first trit most significant), as `quantize_row_tq1_0_ref` does: the base-3
/// number of the codes `trit + 1`, multiplied by 3 once more for a four-trit field, scaled to a byte with a ceiling
/// division by 243.
fn tq1_byte(trits: impl Iterator<Item = i8>, four: bool) -> u8 {
    let mut q: u16 = 0;
    for t in trits {
        q = q * 3 + (t + 1) as u16;
    }
    if four {
        q *= 3;
    }
    (q * 256).div_ceil(243) as u8
}

/// Encode 256 trits as one TQ1_0 block's 52 code bytes, in ggml's order: `qs[m]` (m < 32) holds weights
/// `m + 32 n`, `qs[32 + m]` (m < 16) weights `160 + m + 16 n`, both for n in 0..5; `qh[m]` weights `240 + m + 4 n`
/// for n in 0..4.
pub fn tq1_encode(t: &[i8; 256]) -> [u8; 52] {
    let mut b = [0u8; 52];
    for m in 0..32 {
        b[m] = tq1_byte((0..5).map(|n| t[m + 32 * n]), false);
    }
    for m in 0..16 {
        b[32 + m] = tq1_byte((0..5).map(|n| t[160 + m + 16 * n]), false);
    }
    for m in 0..4 {
        b[48 + m] = tq1_byte((0..4).map(|n| t[240 + m + 4 * n]), true);
    }
    b
}

/// Decode one TQ1_0 block's 52 code bytes to 256 trits, as ggml's dequantization does, and refuse a block that is
/// not the canonical encoding of the trits it decodes to (ggml would read such a byte as some trits, silently).
pub fn tq1_decode(code: &[u8]) -> Result<[i8; 256], String> {
    const POW: [u8; 5] = [1, 3, 9, 27, 81];
    let digit = |byte: u8, p: u8| ((u16::from(byte.wrapping_mul(p)) * 3) >> 8) as i8 - 1;
    let mut t = [0i8; 256];
    for (n, &p) in POW.iter().enumerate() {
        for m in 0..32 {
            t[m + 32 * n] = digit(code[m], p);
        }
        for m in 0..16 {
            t[160 + m + 16 * n] = digit(code[32 + m], p);
        }
    }
    for (n, &p) in POW[..4].iter().enumerate() {
        for m in 0..4 {
            t[240 + m + 4 * n] = digit(code[48 + m], p);
        }
    }
    if tq1_encode(&t)[..] != code[..52] {
        return Err("a TQ1 code byte is not the encoding of its trits".into());
    }
    Ok(t)
}

/// Pack one row into padded `TQ1_0` blocks (`row.len() / 128` blocks of 54 bytes): a group's 128 trits, then 128
/// zero trits, then the group's scale as fp16. Strict and planted modes as [`pack_row`].
pub fn pack_row_tq1(row: &[f32], rule: Rule, strict: bool, out: &mut Vec<u8>) -> Result<(), Inexact> {
    let (trits, scales) = quantize_row(row, rule);
    for (g, chunk) in trits.chunks(GROUP).enumerate() {
        let s = scales[g];
        let h = match f32_to_f16_exact(s) {
            Some(h) => h,
            None if strict => return Err(Inexact::ScaleNotF16 { group: g, scale: s }),
            None => half_round(s),
        };
        if strict {
            for (j, &t) in chunk.iter().enumerate() {
                let v = row[g * GROUP + j];
                if f32::from(t) * s != v {
                    return Err(Inexact::NotTernary {
                        col: g * GROUP + j,
                        value: v,
                        scale: s,
                    });
                }
            }
        }
        let mut block = [0i8; 256];
        block[..GROUP].copy_from_slice(chunk);
        out.extend_from_slice(&tq1_encode(&block));
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(())
}

/// Dequantize a padded `TQ1_0` row back to its `n` weights, through the same decoder as the 256-weight layout
/// ([`dequantize_tq1`]), refusing a block whose padding half is not zero.
pub fn dequantize_row_tq1(packed: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if n == 0 || n % GROUP != 0 || packed.len() != n / GROUP * TQ1_0_BLOCK_BYTES {
        return Err("bad padded TQ1 row size".into());
    }
    let mut out = Vec::with_capacity(n);
    for (b, blk) in packed.chunks_exact(TQ1_0_BLOCK_BYTES).enumerate() {
        let v = dequantize_tq1(blk, 256)?;
        if v[GROUP..].iter().any(|&x| x != 0.0) {
            return Err(format!("block {b}: the padding half is not zero"));
        }
        out.extend_from_slice(&v[..GROUP]);
    }
    Ok(out)
}

/// The readback check for a padded `TQ1_0` row, as [`check_row`] is for padded `TQ2_0`: every block's code bytes
/// are canonical, its padding half is 128 zero trits, its scale equals its group's absmax bit for bit, every trit
/// equals the source's, and every weight read back (through [`dequantize_tq1`], the decoder the runtime uses)
/// equals the source's value.
pub fn check_row_tq1(src: &[f32], packed: &[u8]) -> Result<(), String> {
    if packed.len() != src.len() / GROUP * TQ1_0_BLOCK_BYTES {
        return Err("bad padded TQ1 row size".into());
    }
    let (trits, scales) = quantize_row(src, Rule::KITTEN);
    for (b, &scale) in scales.iter().enumerate() {
        let blk = &packed[b * TQ1_0_BLOCK_BYTES..(b + 1) * TQ1_0_BLOCK_BYTES];
        let t = tq1_decode(blk).map_err(|e| format!("group {b}: {e}"))?;
        if t[GROUP..].iter().any(|&x| x != 0) {
            return Err(format!("group {b}: the padding half is not zero"));
        }
        let d = f16_to_f32(u16::from_le_bytes([blk[52], blk[53]]));
        if d.to_bits() != scale.to_bits() {
            return Err(format!("group {b}: scale {d} read back, source absmax {scale}"));
        }
        let values = dequantize_tq1(blk, 256).map_err(|e| format!("group {b}: {e}"))?;
        for j in 0..GROUP {
            let col = b * GROUP + j;
            if t[j] != trits[col] {
                return Err(format!("col {col}: trit {} read back, source {}", t[j], trits[col]));
            }
            // Value equality: -0.0 == 0.0, the one thing a trit cannot carry (DESIGN.md, parity level 1).
            if values[j] != src[col] || values[j] != f32::from(t[j]) * d {
                return Err(format!("col {col}: {} read back, source {}", values[j], src[col]));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tq1_pad_tests {
    use super::tests::kitten_row;
    use super::*;

    #[test]
    fn the_encoder_is_the_inverse_of_ggmls_decoder_on_the_reference_fixture() {
        // The fixture's blocks were written by ggml; decoding then encoding each must give its bytes back.
        let fixture = include_bytes!("../tests/fixtures/tq1-reference.bin");
        for sample in fixture.chunks_exact(TQ1_0_BLOCK_BYTES + 1024) {
            let (bytes, values) = sample.split_at(TQ1_0_BLOCK_BYTES);
            let t = tq1_decode(bytes).unwrap();
            assert_eq!(tq1_encode(&t)[..], bytes[..52]);
            let d = f16_to_f32(u16::from_le_bytes([bytes[52], bytes[53]]));
            for (i, v) in values.chunks_exact(4).enumerate() {
                let want = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                assert_eq!((f32::from(t[i]) * d).to_bits(), want.to_bits(), "weight {i}");
            }
        }
    }

    #[test]
    fn every_trit_pattern_round_trips_at_every_position() {
        // Each of the 52 code bytes at every one of its 3^5 (or 3^4) values.
        for m in 0..52 {
            let (first, step, n) = match m {
                0..=31 => (m, 32, 5),
                32..=47 => (160 + m - 32, 16, 5),
                _ => (240 + m - 48, 4, 4),
            };
            for v in 0..3u32.pow(n) {
                let mut t = [0i8; 256];
                let mut x = v;
                for k in (0..n as usize).rev() {
                    t[first + step * k] = (x % 3) as i8 - 1;
                    x /= 3;
                }
                let code = tq1_encode(&t);
                assert_eq!(tq1_decode(&code).unwrap(), t, "byte {m} value {v}");
                let deq = dequantize_tq1(&[&code[..], &0x3c00u16.to_le_bytes()[..]].concat(), 256).unwrap();
                assert!(
                    deq.iter().zip(&t).all(|(a, &b)| *a == f32::from(b)),
                    "byte {m} value {v}"
                );
            }
        }
    }

    #[test]
    fn kitten_rows_of_odd_lengths_pack_exactly_and_read_back() {
        for (seed, n) in [(0, 128), (1, 384), (2, 2048), (3, 6144), (4, 1152)] {
            let row = kitten_row(n, seed);
            let mut out = Vec::new();
            pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
            assert_eq!(out.len(), n / GROUP * TQ1_0_BLOCK_BYTES);
            assert_eq!(Layout::Tq1Pad.row_bytes(n), out.len());
            check_row_tq1(&row, &out).unwrap();
            let back = dequantize_row_tq1(&out, n).unwrap();
            assert!(back.iter().zip(&row).all(|(a, b)| a == b));
        }
    }

    #[test]
    fn a_zero_group_and_an_all_negative_group_pack() {
        let mut row = vec![0f32; 2 * GROUP];
        row[GROUP..].fill(-0.5);
        let mut out = Vec::new();
        pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
        check_row_tq1(&row, &out).unwrap();
    }

    #[test]
    fn a_flipped_trit_is_red() {
        let row = kitten_row(1024, 5);
        let mut out = Vec::new();
        pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
        // Weight 3 of group 2, changed to another trit, the block re-encoded canonically.
        let b = 2 * TQ1_0_BLOCK_BYTES;
        let mut t = tq1_decode(&out[b..b + 52]).unwrap();
        t[3] = if t[3] == 1 { 0 } else { 1 };
        out[b..b + 52].copy_from_slice(&tq1_encode(&t));
        let e = check_row_tq1(&row, &out).unwrap_err();
        assert!(e.contains("col 259"), "{e}");
    }

    #[test]
    fn a_wrong_scale_is_red() {
        let row = kitten_row(1024, 6);
        let mut out = Vec::new();
        pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
        out[3 * TQ1_0_BLOCK_BYTES + 52] ^= 1; // one f16 ulp
        let e = check_row_tq1(&row, &out).unwrap_err();
        assert!(e.contains("group 3: scale"), "{e}");
    }

    #[test]
    fn a_nonzero_padding_trit_is_red_in_the_check_and_in_the_reader() {
        let row = kitten_row(512, 8);
        let mut out = Vec::new();
        pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
        let b = TQ1_0_BLOCK_BYTES;
        let mut t = tq1_decode(&out[b..b + 52]).unwrap();
        t[200] = 1;
        out[b..b + 52].copy_from_slice(&tq1_encode(&t));
        assert!(check_row_tq1(&row, &out).unwrap_err().contains("padding"));
        assert!(dequantize_row_tq1(&out, 512).unwrap_err().contains("padding"));
    }

    #[test]
    fn a_non_canonical_code_byte_is_red() {
        // Byte 1 decodes (to five trits of -1) but the encoder writes those as 0: only 243 of the 256 byte values
        // are codes.
        let row = kitten_row(256, 9);
        let mut out = Vec::new();
        pack_row_tq1(&row, Rule::KITTEN, true, &mut out).unwrap();
        out[0] = 1;
        assert!(check_row_tq1(&row, &out).is_err());
    }

    #[test]
    fn planted_group_256_and_absmean_are_red() {
        let row = kitten_row(2048, 7);
        let g256 = Rule {
            group: 256,
            scale: ScaleRule::AbsMax,
        };
        assert!(matches!(
            pack_row_tq1(&row, g256, true, &mut Vec::new()),
            Err(Inexact::NotTernary { .. })
        ));
        for wrong in [
            g256,
            Rule {
                group: GROUP,
                scale: ScaleRule::AbsMean,
            },
        ] {
            let mut out = Vec::new();
            pack_row_tq1(&row, wrong, false, &mut out).unwrap();
            assert!(check_row_tq1(&row, &out).is_err(), "{wrong:?} must fail the readback");
        }
    }

    #[test]
    fn a_non_ternary_row_and_an_inexact_scale_are_refused() {
        let mut row = kitten_row(256, 3);
        row[5] = row[0] * 0.5;
        assert!(matches!(
            pack_row_tq1(&row, Rule::KITTEN, true, &mut Vec::new()),
            Err(Inexact::NotTernary { col: 5, .. })
        ));
        let mut row = vec![0f32; GROUP];
        row[0] = 1.0 + f32::EPSILON;
        assert!(matches!(
            pack_row_tq1(&row, Rule::KITTEN, true, &mut Vec::new()),
            Err(Inexact::ScaleNotF16 { .. })
        ));
    }

    #[test]
    fn the_layout_names_parse() {
        assert_eq!(Layout::parse("tq1_0"), Some(Layout::Tq1Pad));
        assert_eq!(Layout::parse("tq1_0-pad128"), Some(Layout::Tq1Pad));
        assert_eq!(Layout::parse("tq1_0-256-experimental"), Some(Layout::Tq1));
        assert_eq!(Layout::Tq1Pad.stored_cols(1024), 2048);
        assert!(Layout::Tq1Pad.padded() && Layout::Tq2Pad.padded());
        assert!(!Layout::Tq1.padded() && !Layout::Q4.padded());
    }
}
