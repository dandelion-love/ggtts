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
* **The default is the most useful option for the person using it.** When a flag or mode has several values, the
  default is the one most people running ggtts would want: levelled audio, a terminating sampler, the fast and correct
  backend. A mode that exists to measure or test the engine is opt-in and named for what it proves: a parity stream, a
  trace, an unlevelled read, a planted defect. Tests and probes ask for those modes by name, so a change to the default
  never weakens a check. When you add an option, the pull request says which value is the default and why it is the most
  useful one.
* **Every option has to earn its place.** Each flag or mode is one more thing a user must read past and a maintainer
  must keep working, and more options mean more confusion. There is no hard limit, but a new option needs a reason
  important enough to justify that distraction. Prefer a better default to a new switch. A mode a test cannot run
  without passes that bar (the parity stream exists because the engine's identity checks need it); a convenience
  variant of an existing mode usually does not. The pull request that adds an option names the need it serves.
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

CI runs the desktop release builds, archive smoke tests, Android build and host binding tests together on the
same-repository `rc` to `main` promotion PR. Other PRs run the specialized checks selected by
`scripts/select_specialized_ci.py` for their changed paths; pushes skip these four jobs. A skip is not a pass.
The Linux and Windows ggml jobs and hosted CUDA compile use the same selector on ordinary PRs; a packaging-only
workflow edit selects the archive tests, while changes to a specialized job select that job. Changes to the
selector, unknown jobs or workflow-wide configuration conservatively select all. On pushes, ggml and CUDA still
run; the four release and binding jobs are skipped. General checks still run, and the tag-triggered release
workflow still builds its archives. CUDA compilation on promotion is separate from GPU runtime parity; the local
promotion gate supplies the latter.

**Promotion.** `main` moves only by one reviewed pull request from `rc`. Hosted runners have no GPU, so before that
pull request is merged, someone with a supported GPU runs `scripts/rc_gate.sh --ref origin/rc --work /path/to/scratch`
and attaches the resulting `rc-gate.json` and the logs it names. The gate runs CPU checks and both engines' CUDA
probes; missing prerequisites make the result incomplete, never a pass. It requires Python 3, Rust at the pinned
version, CUDA and the model/reference inputs documented at the top of each engine's `scripts/cuda_probe.sh`.
Set their environment variables before running; the gate assigns separate `WORK` and `CARGO_TARGET_DIR` directories
under the persistent scratch directory. Use `--wait-file` for a busy GPU. The report identifies the commit, model
and fixture hashes, device, probe commands and logs. External CUDA library build provenance must be checked separately.
The probes compare each engine to its own reference; they do not prove CPU and GPU outputs identical, nor the
instrument no-change proof in #34. A pass applies only to the recorded commit.
Release tags are cut from `main`. Pre-release tags (`vX.Y.Z-rc.N`) may be cut from `rc`.

By contributing you agree that your contribution is licensed under the MIT licence of this repository.
