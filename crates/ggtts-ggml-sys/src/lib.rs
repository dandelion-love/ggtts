//! Raw, unsafe bindings to ggml at [`GGML_COMMIT`], with ggml built from source by CMake (CPU, static).
//!
//! Everything is behind the `ggml` feature; without it the crate holds only the pin. Nothing here is safe: a
//! tensor is a raw `*mut ggml_tensor`, and a failed `GGML_ASSERT` aborts the process rather than returning an
//! error, so callers validate before they call. See `README.md` for the build and for what in ggml's C API fights
//! Rust.

/// The ggml commit the bindings were generated from and the submodule is pinned to.
pub const GGML_COMMIT: &str = "af97976c7810cdabb1863172f31c432dab767de7";

#[cfg(feature = "ggml")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all,
    clippy::pedantic
)]
mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "ggml")]
pub use raw::*;

/// `GGML_PAD(x, n)`: a function-like macro, which bindgen does not translate. `n` must be a power of two.
pub const fn pad_to(x: i64, n: i64) -> i64 {
    (x + n - 1) & !(n - 1)
}

/// `GGML_ROPE_TYPE_NEOX`, `GGML_ROPE_TYPE_MROPE`: rope modes are `#define`d ints, re-stated here so code that
/// does not enable `ggml` can still name them in tests.
pub mod rope_mode {
    pub const NORMAL: i32 = 0;
    pub const NEOX: i32 = 2;
    pub const MROPE: i32 = 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_matches_the_macro() {
        assert_eq!(pad_to(0, 32), 0);
        assert_eq!(pad_to(1, 32), 32);
        assert_eq!(pad_to(256, 256), 256);
        assert_eq!(pad_to(257, 256), 512);
    }

    #[test]
    fn pin_is_a_full_sha() {
        assert_eq!(GGML_COMMIT.len(), 40);
        assert!(GGML_COMMIT.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[cfg(feature = "ggml")]
    #[test]
    fn rope_modes_match_the_header() {
        assert_eq!(rope_mode::NEOX, GGML_ROPE_TYPE_NEOX as i32);
        assert_eq!(rope_mode::MROPE, GGML_ROPE_TYPE_MROPE as i32);
    }

    #[cfg(feature = "ggml")]
    #[test]
    fn links_and_runs_one_op() {
        // A 2x2 add through a no_alloc=false context: proves the archives link and the bindings' layouts are sane.
        unsafe {
            let params = ggml_init_params {
                mem_size: 1 << 20,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null());
            let a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
            let b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
            let pa = (*a).data as *mut f32;
            let pb = (*b).data as *mut f32;
            for i in 0..4 {
                *pa.add(i) = i as f32;
                *pb.add(i) = 10.0;
            }
            let c = ggml_add(ctx, a, b);
            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            ggml_graph_compute_with_ctx(ctx, gf, 1);
            let pc = (*c).data as *const f32;
            let got: Vec<f32> = (0..4).map(|i| *pc.add(i)).collect();
            assert_eq!(got, vec![10.0, 11.0, 12.0, 13.0]);
            ggml_free(ctx);
        }
    }
}
