# Template: a Rust tensor hook

An in-process instrument. It runs inside the engine, once per generated frame, and reads values the event stream
does not carry. This one reads the stop probability (`p_stop`) and reports **stop pressure early in each draw**:

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `p_stop_early_max` | 1 | lower | The highest `p_stop` in the first half of any draw |
| `early_stop_frames` | 1 | lower | Frames in a draw's first half with `p_stop` at or above the threshold (0.5) |

## What a tensor hook may and may not do

- **It asks for what it reads.** `wants()` returns `p_stop` only, so the engine computes the stop probability and
  nothing else. With no hook registered, it computes nothing for hooks at all.
- **It is read-only by convention.** It gets shared references, so safe Rust cannot write through them. That is not
  a proof. The enforcement is the identity check: the same seed must render byte-identically with and without the
  hook. `tests/selftest.rs` runs that check for this hook on the fake engine, in CI.
- **It has no isolation.**
  - A **panic** is caught: the hook is switched off and the failure recorded, and the render goes on unchanged.
  - A **hang** hangs the render, since there is no timeout.
  - An **abort** kills the process.

  `crates/ggtts-core/tests/tensor_hooks.rs` tests all three. If you need a timeout or a crash boundary, write an
  event instrument instead.
- **It is compiled only with its feature.** The hook exists only with this crate's `enabled` feature. A binary
  includes it by depending on this crate with that feature, and a build that does not ask has no tensor hook.

```
cargo test -p ggtts-instrument-tensor-template
```

**Self-test** (`tests/selftest.rs`):
- *Planted:* a draw whose second frame has `p_stop` 0.95; `early_stop_frames` reads 1 and the maximum 0.95.
- *No-op:* a clean draw reads zero early frames, and two renders of the same story read the same.
- *Identity:* the fake engine renders the same tokens and audio with and without the hook.

The worked tensor example is `ggtts_core::hooks::StopTrace`: the per-step stop-probability trace, which the Qwen3-TTS
engine's `--trace` now runs on this API.
