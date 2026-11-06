# Third-party sources

What ggtts takes from other projects, and under which licence.

| Source | Licence | Used as | What is taken |
|---|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) at `af97976c7810cdabb1863172f31c432dab767de7` | MIT | pinned dependency (CI compiles it; linking arrives with the ggml engine). The layer-0 spike links it as the submodule `spike/layer0/ggml-sys/ggml` | nothing copied |
| [`libm`](https://crates.io/crates/libm) crate (a port of musl's libm) | MIT OR Apache-2.0 | dependency of `ggtts-core` and `ggtts-fake` | `log10`, `pow` and `sin`, so float results do not depend on the platform's C library |
| [NumPy](https://github.com/numpy/numpy) 2.4 | BSD-3-Clause | reference | the summation order of `pairwise_sum` (`numpy/_core/src/umath/loops_utils.h.src`): eight accumulators in blocks of up to 128 elements, halves split on a multiple of 8. Reimplemented in `crates/ggtts-core/src/numeric.rs`, no code copied, so the level step's float results match the reference's |
| [`claxon`](https://crates.io/crates/claxon) | Apache-2.0 | test dependency only | decodes the FLAC the core writes, as an independent check |
| [`serde_json`](https://crates.io/crates/serde_json) | MIT OR Apache-2.0 | test dependency only | reads the fixtures |
| [qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) at `16bb5af` | MIT | reference runtime | `spike/layer0/reference/layer0_ref.cpp` transcribes layer 0 of the talker graph (`src/transformer/transformer_graph_talker.cpp:39-172`) and its prefill inputs (`src/transformer/transformer_runtime.cpp:101-205`) for a parity spike; `spike/layer0/layer0/src/main.rs` builds the same graph in Rust. The sampler and the full graph port arrive with the Qwen3-TTS engine |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) at `cd959bae` | MIT | reference code only, never linked | the command-line conventions (`-hf`, the `*_ARG_*` environment mirror); no code yet |
