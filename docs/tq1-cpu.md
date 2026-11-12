# Experimental Kitten TQ1 CPU runtime

The Kitten reader accepts the `tq1_0-256-experimental` GGUF layout and ggml type `TQ1_0` (34) on CPU. Its 256-value blocks are a lossy conversion of the original 128-value training groups. This path reads existing models; the Kitten converter does not write this layout. It writes `tq1_0-pad128` instead (`--layout tq1_0`): the same ggml type with one 128-value group and 128 zero trits per block, which is exact and verified by the readback check (`crates/ggtts-kitten/DESIGN.md`, "`TQ1_0` on zero-padded groups"). The rest of this page is about the 256-value layout. On CUDA the matrices are expanded to checked f16 at load (see below); there is no native TQ1 CUDA kernel. Existing Q4 and padded TQ2 formats remain separate.

The parser refuses unsupported layout/type, invalid row block sizes, invalid scales, overflows and truncated tensors. Before constructing ggml tensors, the TQ1 loader checks required model tensor types and shapes against the declared hyperparameters. This does not establish that generated speech is complete or natural. The runtime comparison, omitted-text controls, baseline-first protocol, owner listening and release decision remain separate work.

## CUDA

On CUDA a TQ1 matrix is not run packed. As for the Q4 and padded TQ2 layouts, each matrix is dequantized at load to f16, every row is converted back and compared bit for bit, and a value that does not survive is refused rather than rounded (`ternary_as_f16` in `crates/ggtts-kitten/src/model.rs`). Device memory therefore holds two bytes a weight, not TQ1's 54 bytes per 256 weights: the packed format saves file size and CPU memory, not CUDA memory. With `--weights f32` the matrices are held as f32 on either device. Unit tests cover which representation each device and weights mode selects, the resulting dimensions, the f16 values against the reference fixture, and the refusals.

`crates/ggtts-kitten/scripts/cuda_probe.sh` with `TQ1_GGUF` set checks a TQ1 model on CUDA in both weights modes. No PyTorch recording exists for a TQ1 model, so the reference is this engine's CPU f32 arm of the same GGUF, written by `ggtts-kitten parity --write-ref` and held to the existing bars; a copy with one TQ1 matrix zeroed must fail. This establishes that CUDA computes what the CPU f32 arm computes. It says nothing about speech quality, and the packed CPU kernels are a different numeric path: sampled token sequences from the CPU's packed arm and from CUDA differ for the same model and seed ([#152](https://github.com/dandelion-love/ggtts/issues/152)).

## The CPU's packed path, and why it is not CUDA's

On the CPU a TQ1 matrix stays packed, and ggml multiplies it with `ggml_vec_dot_tq1_0_q8_K`, which first converts the
activations to Q8_K: 8-bit integers with one scale per block of 256 values. The weights are read exactly; the
difference is the rounding of the activations. Per matrix product, on seven ternary matrices of one model, the
packed result equals the exact f32 product of the rounded activations to within 2e-7 (relative L2), and differs from
the exact product of the unrounded ones by 3e-3 to 3e-2, the same to five figures as that rounding alone. The
SIMD and generic kernels agree to about 1e-7. CUDA holds the same weights as exact f16 and keeps float activations,
and so does `--weights f32` on either device.

So the CPU's packed path and CUDA give different audio for the same model and seed
([#152](https://github.com/dandelion-love/ggtts/issues/152)). Against the CPU f32 arm of one model, the packed arm
reads RMSE 0.0547, minimum cosine 0.9998324 and argmax 31 of 32 on the recorded prompt: below the ternary bar of
0.9999. On 45 lines the sampled token sequences part after a median of 15 tokens. Speech-recognition error rates
on those lines did not differ measurably (paired difference, packed minus f32, -0.008 [-0.058, +0.050] with
Qwen3-ASR); that interval does not exclude about five points either way, and nobody has listened to either set.

**The probe reports this arm and does not judge it.** With `TQ1_GGUF` set, `cuda_probe.sh` runs the packed arm
against the CPU f32 reference, and again on the planted copy, and prints both as INFO lines. No bar decides them:
the arm is below the ternary bar by design, and a bar set at today's numbers would only record them. The probe
fails on this arm only when the instrument itself fails: the arm printed no numbers, or its planted run's worst
hidden rel L2 is not larger than its clean run's, so its report would not show a fault. Measured on the CPU at the
change that added it: clean RMSE 0.0547, minimum cosine 0.9998324, argmax 31/32, worst hidden rel L2 1.393e-2;
planted RMSE 1.0916, minimum cosine 0.9673624, argmax 17/32, worst hidden rel L2 5.655e-1. With the plant made a
no-op (the copy left unzeroed), both runs read the same and the probe fails, as it must.
`DEVICE=cpu STEPS=tq1` runs only this arm and its reference, without a GPU.

## Reference fixture

`crates/ggtts-kitten/tests/fixtures/tq1-reference.bin` contains 64 synthetic 256-value blocks and their reference float32 outputs. SHA-256: `e74769c5c1992b867bb9371e9365c2a03350fd664b260860d0b34acaddd1bc9a`. The archived generator used NumPy `default_rng(20261107)` uniform values in [-2, 2], with an all-zero first row and a repeating -0.75/0/+0.75 second row. It sent float32 values through `quantize_row_tq1_0_ref` and `dequantize_row_tq1_0` from a locally built ggml library. The original library was not hashed and the NumPy version was not recorded, so exact regeneration of the original fixture is unproven.

The outputs were independently checked against a clean checkout of the ggml pin `af97976c7810cdabb1863172f31c432dab767de7`, built by the CPU `ggml` feature test. On Linux, with `GGML_SRC` pointing at that checkout, run `cargo test -p ggtts-kitten --features ggml --locked`, then compile the verifier against the resulting `target/debug/build/ggtts-ggml-sys-*/out/lib/libggml-base.a`:

```sh
gcc -fopenmp crates/ggtts-kitten/tests/fixtures/verify_tq1_reference.c "$GGML_BASE_A" -lm -lstdc++ -lpthread -o /tmp/verify-tq1-111
/tmp/verify-tq1-111 crates/ggtts-kitten/tests/fixtures/tq1-reference.bin
```

The observed archive SHA-256 was `4b5314f0713f41464374a1d7401535323a8d3dca656a47d2669016453331f214`; the verifier reported `blocks=64 code-controls=64 scale-controls=63`. This checks every stored block against the pinned dequantizer and detects planted changes; it does not establish the original generator's library identity.

Tests decode every supplied block, compare float32 bits, and plant corrupted scales and packed codes; malformed lengths and scales are rejected. A passing fixture test is not a real-model inference or listening result.
