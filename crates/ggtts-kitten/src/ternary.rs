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
    /// `TQ2_0`, one Kitten group per 256-weight block with zeros in the second half (`Q8_K` activations).
    Tq2Pad,
    /// Stock `Q4_0`: each 32-weight block carries its group's scale as `d` and codes 7, 8, 9 for -1, 0, +1, which is
    /// exact; activations are `Q8_0`, quantized per 32.
    Q4,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::Tq2Pad => "tq2_0-pad128",
            Layout::Q4 => "q4_0-g128",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "tq2_0-pad128" | "tq2_0" => Some(Layout::Tq2Pad),
            "q4_0-g128" | "q4_0" => Some(Layout::Q4),
            _ => None,
        }
    }
    /// The stored row length (`ne[0]`) for `cols` weights.
    pub fn stored_cols(self, cols: usize) -> usize {
        match self {
            Layout::Tq2Pad => 2 * cols,
            Layout::Q4 => cols,
        }
    }
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            Layout::Tq2Pad => cols / GROUP * TQ2_0_BLOCK_BYTES,
            Layout::Q4 => cols / 32 * Q4_0_BLOCK_BYTES,
        }
    }
    pub fn pack_row(self, row: &[f32], rule: Rule, strict: bool, out: &mut Vec<u8>) -> Result<(), Inexact> {
        match self {
            Layout::Tq2Pad => pack_row(row, rule, strict, out),
            Layout::Q4 => pack_row_q4(row, rule, strict, out),
        }
    }
    pub fn check_row(self, src: &[f32], packed: &[u8]) -> Result<(), String> {
        match self {
            Layout::Tq2Pad => check_row(src, packed),
            Layout::Q4 => check_row_q4(src, packed),
        }
    }
    pub fn dequantize_row(self, packed: &[u8], n: usize) -> Result<Vec<f32>, String> {
        match self {
            Layout::Tq2Pad => dequantize_row(packed, n),
            Layout::Q4 => dequantize_row_q4(packed, n),
        }
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
