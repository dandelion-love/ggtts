# Experimental Kitten TQ1 CPU runtime

The Kitten reader accepts the `tq1_0-256-experimental` GGUF layout and ggml type `TQ1_0` (34) on CPU. Its 256-value blocks are a lossy conversion of the original 128-value training groups. This path reads existing models; the Kitten converter does not export or verify TQ1. TQ1 on CUDA is explicitly refused until separately validated. Existing Q4 and padded TQ2 formats remain separate.

The parser refuses unsupported layout/type, invalid row block sizes, invalid scales, overflows and truncated tensors. Before constructing ggml tensors, the TQ1 loader checks required model tensor types and shapes against the declared hyperparameters. This does not establish that generated speech is complete or natural. The runtime comparison, omitted-text controls, baseline-first protocol, owner listening and release decision remain separate work.

## Reference fixture

`crates/ggtts-kitten/tests/fixtures/tq1-reference.bin` contains 64 synthetic 256-value blocks and their reference float32 outputs. SHA-256: `e74769c5c1992b867bb9371e9365c2a03350fd664b260860d0b34acaddd1bc9a`. The archived generator used NumPy `default_rng(20261107)` uniform values in [-2, 2], with an all-zero first row and a repeating -0.75/0/+0.75 second row. It sent float32 values through `quantize_row_tq1_0_ref` and `dequantize_row_tq1_0` from a locally built ggml library. The recorded build path was a ggml-sys build, but the library itself was not hashed and the NumPy version was not recorded. Consequently this fixture is useful independent numerical evidence but is not yet a verified pinned-ggml regeneration. The C helper and generator beside the fixture document the procedure; recompile against the pinned ggml revision and compare the fixture hash before claiming that provenance.

Tests decode every supplied block, compare float32 bits, and plant corrupted scales and packed codes; malformed lengths and scales are rejected. A passing fixture test is not a real-model inference or listening result.
