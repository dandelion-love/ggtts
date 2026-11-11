# Experimental Kitten TQ1 CPU runtime

The Kitten reader accepts the `tq1_0-256-experimental` GGUF layout and ggml type `TQ1_0` (34) on CPU. Its 256-value blocks are a lossy conversion of the original 128-value training groups. This path reads existing models; the Kitten converter does not export or verify TQ1. On CUDA the matrices are expanded to checked f16 at load (see below); there is no native TQ1 CUDA kernel. Existing Q4 and padded TQ2 formats remain separate.

The parser refuses unsupported layout/type, invalid row block sizes, invalid scales, overflows and truncated tensors. Before constructing ggml tensors, the TQ1 loader checks required model tensor types and shapes against the declared hyperparameters. This does not establish that generated speech is complete or natural. The runtime comparison, omitted-text controls, baseline-first protocol, owner listening and release decision remain separate work.

## CUDA

On CUDA a TQ1 matrix is not run packed. As for the Q4 and padded TQ2 layouts, each matrix is dequantized at load to f16, every row is converted back and compared bit for bit, and a value that does not survive is refused rather than rounded (`ternary_as_f16` in `crates/ggtts-kitten/src/model.rs`). Device memory therefore holds two bytes a weight, not TQ1's 54 bytes per 256 weights: the packed format saves file size and CPU memory, not CUDA memory. With `--weights f32` the matrices are held as f32 on either device. Unit tests cover which representation each device and weights mode selects, the resulting dimensions, the f16 values against the reference fixture, and the refusals.

`crates/ggtts-kitten/scripts/cuda_probe.sh` with `TQ1_GGUF` set checks a TQ1 model on CUDA in both weights modes. No PyTorch recording exists for a TQ1 model, so the reference is this engine's CPU f32 arm of the same GGUF, written by `ggtts-kitten parity --write-ref` and held to the existing bars; a copy with one TQ1 matrix zeroed must fail. This establishes that CUDA computes what the CPU f32 arm computes. It says nothing about speech quality, and the packed CPU kernels are a different numeric path: sampled token sequences from the CPU's packed arm and from CUDA differ for the same model and seed ([#152](https://github.com/dandelion-love/ggtts/issues/152)).

## Reference fixture

`crates/ggtts-kitten/tests/fixtures/tq1-reference.bin` contains 64 synthetic 256-value blocks and their reference float32 outputs. SHA-256: `e74769c5c1992b867bb9371e9365c2a03350fd664b260860d0b34acaddd1bc9a`. The archived generator used NumPy `default_rng(20261107)` uniform values in [-2, 2], with an all-zero first row and a repeating -0.75/0/+0.75 second row. It sent float32 values through `quantize_row_tq1_0_ref` and `dequantize_row_tq1_0` from a locally built ggml library. The original library was not hashed and the NumPy version was not recorded, so exact regeneration of the original fixture is unproven.

The outputs were independently checked against a clean checkout of the ggml pin `af97976c7810cdabb1863172f31c432dab767de7`, built by the CPU `ggml` feature test. On Linux, with `GGML_SRC` pointing at that checkout, run `cargo test -p ggtts-kitten --features ggml --locked`, then compile the verifier against the resulting `target/debug/build/ggtts-ggml-sys-*/out/lib/libggml-base.a`:

```sh
gcc -fopenmp crates/ggtts-kitten/tests/fixtures/verify_tq1_reference.c "$GGML_BASE_A" -lm -lstdc++ -lpthread -o /tmp/verify-tq1-111
/tmp/verify-tq1-111 crates/ggtts-kitten/tests/fixtures/tq1-reference.bin
```

The observed archive SHA-256 was `4b5314f0713f41464374a1d7401535323a8d3dca656a47d2669016453331f214`; the verifier reported `blocks=64 code-controls=64 scale-controls=63`. This checks every stored block against the pinned dequantizer and detects planted changes; it does not establish the original generator's library identity.

Tests decode every supplied block, compare float32 bits, and plant corrupted scales and packed codes; malformed lengths and scales are rejected. A passing fixture test is not a real-model inference or listening result.
