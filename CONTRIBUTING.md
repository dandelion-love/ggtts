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
* These identifiers are public and fine to use: the Hugging Face repositories `jmtl/hk-cantonese-cc0-qwen3-tts`
  and `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF`, their release tags, and the speaker names `hk_voice_343`,
  `hk_voice_777` and `hk_cantonese_speaker`.

## Instrument contributions

An instrument contribution names the hooks and input artifacts it reads and documents what its result can and cannot
establish. Supply a self-test with a planted defect of the kind it claims to detect and a no-op that reads as no
change. The self-test must actually run in the existing test suite, validate the emitted envelope against the current
report contract, and leave a log with the outcome. A crash, missing required input or unsupported measurement must
remain visible as a non-OK result; never replace a missing value with zero. For an event instrument, test a truncated
event stream as well. Out-of-process instruments must not modify their inputs; in-process tensor hooks are trusted
and have the narrower isolation described in the instrument documentation.

Version an instrument when its behavior, defaults or metric meaning changes; document whether existing comparisons
remain valid. The report schema's format version changes when a previously valid reader cannot safely read a new
envelope. Preserve the previous schema and make the current reader accept the previous version. Adding a metric to
the metrics list does not change the report format. Any version change must include compatibility and rejection tests.

The local promotion gate currently does not prove that every instrument leaves a real model's tokens and audio
unchanged; that proof is tracked in #34. A fake-engine test does not establish the same property for real decoder
or GPU execution. Token/audio equality also says nothing about timing or external artifacts.

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
The Linux and Windows ggml jobs use the same selector on ordinary PRs. Unknown jobs or workflow-wide
configuration conservatively select all remaining specialized checks. Selector changes run their dedicated
tests; other changed paths select their own jobs. Pushes skip specialized jobs; general checks still run.

Ordinary PRs do not compile the CUDA archive. The `cuda-compile` job builds and smokes it on a hosted runner
without a GPU, as `release.yml`'s `cuda-build` does, line for line (the selection job refuses a
difference), in two cases only: every pull request into `main` (the promotion), and a push to `rc` that changes a
crate's `build.rs`, anything in `crates/ggtts-ggml-sys/` (the ggml submodule pointer included), a `Cargo.toml`,
`Cargo.lock`, `scripts/package_cuda_release.sh` or a workflow. Push runs are never cancelled, so each push is judged
by its own range. The mandatory exact-head on-device gate
still runs both CUDA engines and instrument identity checks, but it uses external ggml libraries and does not build
the shipped CUDA archive. A skipped hosted compile is not a pass, and a passed one is not evidence of GPU runtime
behavior.

For the explicitly listed generic measurement and report paths in the selector, ordinary PRs run the standard
Linux and Windows CPU checks and, for CLI or runner edits, package smoke. They do not automatically build the CUDA,
Android or full release binaries. CPU release, Android and CUDA archive builds remain mandatory on the promotion PR; the
tag-triggered release builds the archives again. Selector tests exercise the
promotion and path decisions without building the specialized jobs on that PR.

**Promotion.** `main` moves only by one reviewed pull request from `rc`. Hosted runners have no GPU, so before that
pull request is merged, someone with a supported GPU runs `scripts/rc_gate.sh --ref origin/rc --work /path/to/scratch`
and attaches the resulting `rc-gate.json` and the logs it names. The gate runs CPU checks and both engines' CUDA
probes; missing prerequisites make the result incomplete, never a pass. That includes `TQ1_GGUF`, the Kitten probe's
experimental TQ1 model: without it the report's `gpu.kitten.tq1` reads "tq1: not run" and the result is incomplete.
The report records the TQ1 model's sha256 beside the expected value, `TQ1_GGUF_SHA256`; a different model fails, and
checks that passed with no expected value are incomplete. No public TQ1_0 model or converter command is published
yet, so the operator supplies both the file and its sha256 ([#153](https://github.com/dandelion-love/ggtts/issues/153)
tracks publishing one). It requires Python 3, Rust at the pinned
version, CUDA and the model/reference inputs documented at the top of each engine's `scripts/cuda_probe.sh`.

The gate checks its Python requirements before anything is built: every package pinned in
`instruments/examples/python/loudness-ends/requirements.txt` (NumPy, SciPy, librosa, soundfile), at the pinned
version, for the interpreter the instruments run under (`GGTTS_PYTHON`, else `python3`). A missing or different one
stops the gate with an incomplete result naming the package and the file. Install them into a virtual environment
with `python3 -m pip install -r instruments/examples/python/loudness-ends/requirements.txt` and put it first on
`PATH`, or point `GGTTS_PYTHON` at its interpreter.

**What a gate run may take from earlier evidence.** Two stages need not rerun on every release candidate:

* **The CPU stage** (fmt, clippy, build, test) is taken from CI when every required CI check run on the gated
  commit concluded `success`: `check (fmt, clippy)` and `build-test` on Ubuntu and Windows, read with `gh api`.
  Skipped, cancelled, pending, absent or unreadable is not a pass, and the gate then runs the stage itself. The
  report's `cpu_source` says which. `--cpu-stage` always runs it.
* **An engine's CUDA probe** is reused from the last passing report (`--previous`, by default the `--out` file it is
  about to overwrite) when nothing the probe reads changed since the commit it actually ran on: the engine crates
  it builds and their path dependencies (from `cargo metadata`), `crates/ggtts-ggml-sys` with the ggml submodule
  pin, `crates/ggtts-core`, `Cargo.lock`, `Cargo.toml`, `rust-toolchain.toml`, `schemas/`, `fixtures/`, its
  `cuda_probe.sh`, and the sha256 of every model and reference input it hashes (and, for Kitten, the TQ1 model and
  its expected sha256). The stage then reads `reused from <sha> (<report>)`, keeps the earlier log path, and counts
  as a pass. Reuse is refused, and the probe runs, when the earlier report is missing, unreadable, of another
  `schema`, or did not pass, when that probe did not pass in it, or when its commit is unknown here; `reuse_refused`
  says why. `--no-reuse` runs every probe. A reused probe proves what it proved on the commit it names, on that
  run's device: it is evidence that nothing it depends on has changed, not a new run. The instrument identity proof
  always runs. To undo this rule, pass `--no-reuse --cpu-stage`, or revert the change that added it.

The probe list is `PROBES` in `scripts/rc_gate.py`: adding a probe (for example one for `crates/ggtts-ffi`) is one
entry naming its script, the packages it builds and the inputs it reads, and makes that probe required for a pass.
Set their environment variables before running; the gate assigns separate `WORK` and `CARGO_TARGET_DIR` directories
under the persistent scratch directory. Use `--wait-file` for a busy GPU. The report identifies the commit, model
and fixture hashes, device, probe commands and logs. External CUDA library build provenance must be checked separately.
The probes compare each engine to its own reference; they do not prove CPU and GPU outputs identical. The gate also
requires the pinned Qwen v0.0.1 talker and codec to pass the instrument identity tests on CPU and CUDA: full codec
tokens and decoded samples, every shipped event instrument, the `StopTrace` hook, and a test-only hook-boundary logit
mutation that must be detected. The CPU check builds ggml from the pinned checkout; CUDA uses the Qwen reference's
prebuilt ggml libraries. Missing inputs, a wrong model hash, a failed check, or a filtered test that never ran cannot
pass. These checks do not establish timing or external-artifact identity. A pass applies only to the recorded commit.
Release tags are cut from `main`. Pre-release tags (`vX.Y.Z-rc.N`) may be cut from `rc`.

By contributing you agree that your contribution is licensed under the MIT licence of this repository.
