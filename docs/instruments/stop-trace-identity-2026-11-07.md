# Stop-trace identity on real Qwen

The ignored `stop_trace_tensor_hook_preserves_real_tokens_and_samples` test exercises the actual in-process
`StopTrace` tensor hook together with the event trace. It compares every codec token and decoded float sample
with an uninstrumented generation on the same backend and verifies that both traces contain data. The test
does not measure timings or external artifact effects.

On 2026-11-07, the test used the Qwen v0.0.1 talker SHA-256
`33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67` and codec SHA-256
`7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824`, checked against the
[published revision](https://huggingface.co/api/models/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/revision/v0.0.1?blobs=true).
Both runs used the `beach` fixture, seed 20260905, four CPU threads, rustc 1.88.0 and ggml source at
`af97976c7810cdabb1863172f31c432dab767de7`.

| Backend | Result within backend | Frames / samples | Token bytes SHA-256 | Float sample bytes SHA-256 |
|---|---|---|---|---|
| CPU, Intel i9-12900K | identical | 83 / 159,360 | `0f4389ea2832140f1787ce87550b3b09b22e02e2292ce945b6f0b493a2a129ed` | `a3750e0aa1ae567c534de98482562a8bddaafa1e0179c079e9a8c14926cacbde` |
| CUDA backend, RTX 3080 Ti, driver 610.88 | identical | 93 / 178,560 | `847038f1a2a52eea80b8ea7361a3e1e1e1b761fefbc082e4e1f99d706d1a87ac` | `604172ec0ba866a89c968eeef91dfad42267271ff6349523d7fe8e3b2c2b09f2` |

These are **within-backend** comparisons. The CPU and CUDA outputs differ and this check makes no cross-backend
parity claim. The CUDA scheduler includes a CPU fallback; this test did not independently measure which graph
operations ran on the GPU. Each run used existing prebuilt ggml static libraries; their source-to-binary build
provenance was not independently established. The CUDA `libggml.a` and `ggml-cuda/libggml-cuda.a` SHA-256 values
were `4861ca75b0d41b46ee5585c056860c0506483f1b5710ce9134ecc3c0a16cf493` and
`022eda66414ca16d75b466e77fda82eb004c5efbad2d57c124e168ed6c69d04f`; the CPU `libggml.a` was
`e1989b25dd858f2e4a6008a4dc7c4c9b50321cad34c3e8e6136df26462cc87a8`.

To repeat, set `GGML_SRC` to the pinned ggml source, `GGML_LIB_DIR` to the corresponding prebuilt CPU or CUDA
libraries, and `GGTTS_TALKER` and `GGTTS_CODEC` to the pinned files. Then run once for each backend:

```sh
GGTTS_DEVICE=cpu cargo test -p ggtts-qwen3-tts --features ggml,plant --release --test frame_hook \
  stop_trace_tensor_hook_preserves_real_tokens_and_samples -- --ignored --nocapture
GGTTS_DEVICE=cuda cargo test -p ggtts-qwen3-tts --features cuda,plant --release --test frame_hook \
  stop_trace_tensor_hook_preserves_real_tokens_and_samples -- --ignored --nocapture
```

The CUDA command also needs the appropriate CUDA runtime libraries on the host. Neither test pins or verifies the
model files itself; the operator must verify the hashes before running it. This is one fixture and one tensor hook,
not the full per-instrument promotion proof in #34.
