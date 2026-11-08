# Real Qwen CPU trace and hook checks

This records a partial check for #34, not the complete promotion gate. On 2026-11-07, in WSL on an Intel i9-12900K,
at ggtts `2e7eca7259339062cb650a9c296a18ae7c975324`, the existing `frame_hook` tests ran against the public Qwen
v0.0.1 talker and codec. SHA-256 of the talker was
`33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67`; the codec was
`7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824`. Both match the [public v0.0.1 revision metadata](https://huggingface.co/api/models/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/revision/v0.0.1?blobs=true).
The ggml source was at `af97976c7810cdabb1863172f31c432dab767de7`; rustc was 1.88.0. The CPU static libraries
were supplied from an existing external build; their source-to-binary build provenance was not independently proven.
Their SHA-256 values for `libggml.a`, `libggml-cpu.a`, and `libggml-base.a` were respectively
`e1989b25dd858f2e4a6008a4dc7c4c9b50321cad34c3e8e6136df26462cc87a8`,
`75771bdac3835026ecf22bfaf1ba9f40aa720b772eced10d260d50826f3d98bf`, and
`a38537d4e960f326946564bf3caa4234cb854254d313d1cd65d11745b1738d2c`.

- `event_trace_emits_steps_without_changing_tokens_or_audio`: **pass**, one `beach` fixture at seed 20260905,
  four threads. The test compared the complete generated codec-token vectors and decoded float samples on the
  same CPU engine, with and without event tracing, and checked the trace carried step stop probabilities.
- `a_quiet_hook_changes_no_token_and_a_drawing_hook_is_red`: **pass**, eight CPU golden fixtures. Every quiet
  frame hook matched the reference tokens and saw the returned frames; each planted hook that drew from the
  sampler changed the tokens, and the same plant without the hook remained identical. Runtime 142.79 s.

Both tests are ignored by default because they need the model files. The commands run were equivalent to:

```sh
export GGML_SRC=/path/to/pinned/ggml GGML_LIB_DIR=/path/to/cpu/ggml/src
export GGTTS_TALKER=/path/to/v0.0.1-talker.gguf GGTTS_CODEC=/path/to/codec.gguf
cargo test -p ggtts-qwen3-tts --features ggml,plant --release --test frame_hook \
  event_trace_emits_steps_without_changing_tokens_or_audio -- --ignored --nocapture
cargo test -p ggtts-qwen3-tts --features ggml,plant --release --test frame_hook \
  a_quiet_hook_changes_no_token_and_a_drawing_hook_is_red -- --ignored --nocapture
```

**Limits:** This does not run every shipped instrument, test every tensor hook, or establish GPU behavior. The plant
alters the sampler stream, not a decoded sample or logit. It does not test timing, external artifacts, or quality.
The gate on #34 must not be marked complete from this record alone; the published model files and the external
libraries must be checked on the machine running it.
