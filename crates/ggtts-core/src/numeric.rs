//! Float arithmetic that must match the reference to the last bit.
//!
//! The reference measures loudness with NumPy, whose sum over a contiguous array is a pairwise sum with eight
//! accumulators in blocks of up to 128 elements (`pairwise_sum` in NumPy's `loops_utils.h.src`). A plain left-to-
//! right sum rounds differently, and the level step's gain, and so every sample of a read, would move by an ulp.
//! These functions reproduce that order. Transcendentals come from the `libm` crate, so they are the same on every
//! platform rather than whatever the platform's C library provides.

const PW_BLOCKSIZE: usize = 128;

macro_rules! pairwise {
    ($name:ident, $t:ty) => {
        /// A pairwise sum in NumPy's order, accumulated in the element type.
        pub fn $name(a: &[$t]) -> $t {
            let n = a.len();
            if n < 8 {
                let mut res: $t = -0.0;
                for &x in a {
                    res += x;
                }
                res
            } else if n <= PW_BLOCKSIZE {
                let mut r: [$t; 8] = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
                let mut i = 8;
                let end = n - (n % 8);
                while i < end {
                    for (rj, &x) in r.iter_mut().zip(&a[i..i + 8]) {
                        *rj += x;
                    }
                    i += 8;
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                while i < n {
                    res += a[i];
                    i += 1;
                }
                res
            } else {
                let mut n2 = n / 2;
                n2 -= n2 % 8;
                $name(&a[..n2]) + $name(&a[n2..])
            }
        }
    };
}

pairwise!(pairwise_sum_f64, f64);
pairwise!(pairwise_sum_f32, f32);

/// `np.mean` of a non-empty float64 array: the reduction (identity 0 plus the pairwise sum) over the count.
pub fn mean_f64(a: &[f64]) -> f64 {
    (0.0 + pairwise_sum_f64(a)) / a.len() as f64
}

/// `np.mean` of a non-empty float32 array: summed in float32, divided in float64 (the count is an `intp`, which
/// promotes the division), and rounded back to float32.
pub fn mean_f32(a: &[f32]) -> f32 {
    let s: f32 = 0.0 + pairwise_sum_f32(a);
    (s as f64 / a.len() as f64) as f32
}

/// Base-10 logarithm, the same on every platform.
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// `x` to the power `y`, the same on every platform.
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Round half to even, as the reference's `round()` does for a float with no digits given.
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_differs_from_sequential_where_numpy_does() {
        // 0.1 summed 1000 times: NumPy 2.4's np.sum gives 100.00000000000001 (pairwise); a sequential loop
        // gives 99.9999999999986.
        let a = vec![0.1f64; 1000];
        assert_eq!(pairwise_sum_f64(&a), 100.00000000000001);
        let seq: f64 = a.iter().sum();
        assert_ne!(seq, pairwise_sum_f64(&a));
    }
}
