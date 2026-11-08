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

## Shipped event instruments on the same real model

`scripts/qwen_event_instrument_probe.py` compares a traced baseline story with a second story render that also
runs every shipped event instrument as required. The probe verifies the published talker and codec hashes before
starting, requires every instrument report OK, and compares FLAC bytes, story draw seeds, and the recorded
codebook-0 token sequence. Event instruments execute **after** generation. The trace only carries codebook 0;
this is not an all-codebook comparison for every instrument.

On the same host and prebuilt libraries described above, both runs used the story seed `1689311478` and ran all
seven catalogue entries. Within each backend, the baseline and instrumented FLAC and recorded codebook-0 sequence
were identical:

| Backend | Trace steps | Codebook-0 sequence SHA-256 | FLAC SHA-256 | Renderer SHA-256 |
|---|---|---|---|---|
| CPU | 86 | `e081e798c14082e9f8eb55dfce089a279150d4750dee9349f5b7fb701f444840` | `915b19b5fc2ea864b6c90c12ef255c8109c7aa348d9e60e037289c39bb34f2a5` | `6c51dc22164773dcdb0aa09e56dde1778a1b1d493aa55eea0058696034d5b296` |
| CUDA | 81 | `670ea8876aa6d9e25f163f74e30b8f8d84feafb3bbcba5abd196035bf29cc496` | `83eef46e742a83a41864326ab8fdfcc6e193139c42bb82148520c5b21ee547bb` | `5df7bc2ab48f2d147d1386ea0b91c1f496391dfb48fb257f2f811dcb4a36937f` |

The sequence hash is over the compact JSON list of `(chunk, frame, token)` tuples, not a hash of all Qwen
codebooks. Both reports were `ggtts-report/2` and complete. This is still one story per backend; it does not
measure the cost of instrumentation, cover tensor hooks other than `StopTrace`, or prove external artifacts
unchanged. The probe's unit tests independently plant a different FLAC hash, token and seed, and check refusal
of an incorrect model before creating the work directory.

Build `ggtts-render` and `ggtts-instrument-selftest` from the same checkout, then run the following for each
backend with a **new** work directory outside the checkout and the model paths supplied:

```sh
python3 scripts/qwen_event_instrument_probe.py --render /path/to/ggtts-render \
  --talker /path/to/v0.0.1-talker.gguf --codec /path/to/codec.gguf \
  --device cpu --work /tmp/ggtts-qwen-cpu-proof
```

Use `--device cuda` and an independently built CUDA renderer for the CUDA run. The selftest executable must be
beside the renderer, and Python must be available for the script instruments. The output directory contains
generated audio and logs; keep it outside the repository. This codebook-0 probe is not the promotion gate.

## Full-codebook per-instrument proof and gate

The promotion gate invokes a separate local test, `each_event_instrument_preserves_real_qwen_tokens_and_samples`.
It verifies the published model hashes, compares hashes of **all** generated codec-token vectors and decoded float
samples with tracing off and on, then runs each catalogue event instrument separately as required and verifies the
audio is unchanged. Event instruments run after generation; the test does not imply they were in-process hooks.
The gate also runs the real `StopTrace` identity test above and a test-only hook-boundary logit-corruption plant.
The latter is simulated corruption when an observer requests logits; the public Rust hook receives an immutable
view, so this is not a claim that safe Rust can mutate it. With the simulated plant off, the observer was identical;
with it on, the token comparison detected the corruption.

An end-to-end invocation of the **instrument proof component**, using the same model hashes and host described above,
passed its CPU and CUDA build, per-instrument, stop-trace and logit-plant steps. The story's derived seed was
`1519829812`. Each backend reported all seven event instruments OK and each left the FLAC unchanged:

| Backend | Full-token vector SHA-256 | Decoded float-sample SHA-256 | Logit plant |
|---|---|---|---|
| CPU | `c1116a860d3d5a566a4a65f6712626da1db64ab6394e3939c0ab3d58bf1c2445` | `ed5daaf2dc72ef7e1200da047acd35b296e9e40c9c916696009415a57d9aaa46` | detected; quiet observer identical |
| CUDA | `ec1cc2ff38d158bba181eefaa713c3a1a163b87a3f8bb4bf8cc89afefe6034f1` | `de930207a376b8db6c95d68bdb50760d4bf6af00cb6873398c35e1edc4196940` | detected; quiet observer identical |

Hashes cover framed token-vector bytes, including draw seed, codebook count and vector length, and little-endian
float bit patterns. They compare runs **within** a backend, not between backends. The CPU proof component built
ggml from the pinned checkout; CUDA used the prebuilt libraries whose provenance limit is noted above. Logs remain
in `/tmp/ggtts-instrument-gate-UsvPnk` on the test machine. This invocation did not rerun the other promotion-gate
engine parity probes, so it is not a complete promotion pass or a substitute for the final exact-commit gate run.
