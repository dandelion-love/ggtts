# Contributing to ggtts

Thank you for looking. ggtts is young and its design is still under review, so open an issue before a large change.

## Ground rules

* **Parity is the contract.** A change that alters the codec tokens for any golden fixture, or the result of any
  pipeline case, is a behaviour change, even if it sounds the same. Either it is a bug, or it is a deliberate new
  protocol version with re-recorded fixtures, and the pull request must say which.
* **Every test must be able to fail.** A new check comes with a demonstration that it goes red on a planted defect,
  and reads a no-op as no change.
* **Instruments never change output.** Code behind `--trace` reads values the engine already has. It must not draw
  from the sampler's random stream or reorder computation.
* **Refused settings stay refused.** Some settings are known to stop generation from terminating, for example
  greedy decoding of the main sampler on Qwen3-TTS, or a disabled depth sampler. They need an explicit override,
  and the manifest records it. Do not lift a refusal without measurements.
* **ggml is pinned, not patched.** The pin is a full commit SHA. A change to ggml goes upstream, and moving the pin
  re-checks every golden fixture.
* **Audio output is FLAC.** Lossy formats change the high-frequency detail that listening tests mark.

## Fixtures and data

* Fixture text is written for the purpose by a language model, or is otherwise licensed for redistribution with its
  source recorded beside it. Do not add text from speech corpora whose terms forbid redistribution.
* No audio in the repository. Fixtures hold text, seeds, token ids, numbers and hashes.
* No personal data. Do not name real people as voices.

## Ported code

Code ported from another project keeps its licence notice, names the file, line range and commit it came from in a
comment, and is listed in `THIRD_PARTY.md` in the same pull request. llama.cpp is read as reference code only: ggtts
does not link against it.

## Workflow

Work meets on the `rc` branch before it reaches `main`.

1. Branch from `rc`, and open your pull request **into `rc`**. Never base a pull request on another feature branch:
   it would merge into that branch, not into `rc`.
2. `cargo fmt`, `cargo clippy --workspace -- -D warnings` and `cargo test --workspace` must pass locally before you
   push, and CI must be green on the pull request.
3. The pull request says what changed, why, which parity tests cover it, and what was not tested.
4. Commit subjects say what changed and what was found, not "fix" or "update".

**Promotion.** `main` moves only by one reviewed pull request from `rc`. Hosted runners have no GPU, so before that
pull request is merged, someone with a supported GPU runs `scripts/rc_gate.sh --ref origin/rc` and attaches the
`rc-gate.json` it writes. The gate repeats CI's CPU steps at that ref, and it will run the GPU parity steps once an
engine exists. Until then it records `"gpu": "not-run (no engine)"`, so a CPU pass is never read as a GPU result.
Release tags are cut from `main`. Pre-release tags (`vX.Y.Z-rc.N`) may be cut from `rc`.

By contributing you agree that your contribution is licensed under the MIT licence of this repository.
