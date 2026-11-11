# Changelog

What changes for someone using ggtts, release by release: new options, changed file formats, changed output, and
new requirements. Internal refactors and test changes are not listed.

Each release has a section headed `## <tag> (<date published, UTC>)`. The release workflow publishes that section as
the release's notes, and refuses to publish a tag that has none (`scripts/publish_release.sh`). Add new entries under
**Unreleased**; at release the heading becomes the tag. A released section does not change afterwards:
`scripts/check_changelog.sh` holds each one to the hash recorded in `scripts/changelog-released.sha256`, so an entry
that a merge places under a released heading is caught.

## Unreleased

### Added

- **The release archives ship the C library and `ggtts.h`.** Until now the C ABI (and with it
  `ggtts_load_with` and `ggtts_abi_version`) could only be had by building from source. Every archive now has
  `include/ggtts.h` and the library in `lib/`: `libggtts_ffi.so` in the Linux CPU and CUDA archives, and
  `ggtts_ffi.dll` with its MSVC import library `ggtts_ffi.dll.lib` in the Windows archive. The library carries the
  same engines as `ggtts-render` and renders on the CPU, including in the CUDA archive, whose library is the CPU
  build and loads without the NVIDIA driver. It exports the C ABI only, not the JNI entry points. `BUILDINFO.json`
  lists it under `"libraries"`, and the Linux library is held to the same Ubuntu 22.04 glibc floor as the binaries.
  Where the files are and how to link them: the README's "The C library" section.

### Changed

- **Release notes are this file's section for the tag.** A release page now carries what changed in that release,
  followed by the build provenance (the commit and how to verify an archive); until now it carried only the
  provenance. A tag with no section here is not published.

### Documented

- **The same TQ1_0 Kitten model and seed give different audio on the CPU and on CUDA, and the README now says
  why** ([#152](https://github.com/dandelion-love/ggtts/issues/152)). On the CPU, ggml multiplies a packed TQ1_0
  matrix by first rounding the activations to 8-bit integers with one scale per block of 256 values (Q8_K); the
  weights are read exactly. CUDA holds the same weights as exact f16 and keeps the activations in floating point,
  as does `--weights f32` on either device. Measured on one model, the CPU's packed logits are within cosine
  0.99983 of the float reading of the same file, which is below the ternary parity bar (0.9999); sampled token
  sequences part after a median of 15 tokens. Speech-recognition error rates on 45 lines did not differ measurably
  between the two, and nobody has compared them by ear. This is not new in this release: it has been so since
  TQ1_0 loaded on both devices (v0.1.0-rc.8).
- **Which archive to download**, and that the same seed on the Windows and Linux CPU archives is not byte-identical
  audio, are in the README.

## v0.1.0-rc.10 (2026-11-11)

The command-line tools render the same as in v0.1.0-rc.9; this release changes the C ABI and the Kotlin binding.

### Added

- **The C and Kotlin binding take a Qwen3-TTS speaker and style instruction.** The C-ABI is now version 2:
  `GGTTS_ABI_VERSION` in `ggtts.h` and `ggtts_abi_version()` in the library. `ggtts_load_with` is `ggtts_load` plus
  a `ggtts_settings` struct (`size`, `speaker`, `instruction`; start from `GGTTS_SETTINGS_INIT`), and in Kotlin
  `Ggtts.load` takes `speaker` and `instruction` as nullable named parameters. NULL or empty is no setting, and the
  output is then what `ggtts_load` gives, sample for sample. An unknown speaker (the message lists the model's
  speakers) and an instruction on a checkpoint that takes none are refused while loading with `GGTTS_ERR_ENGINE`
  (a `GgttsException` in Kotlin); either setting on another engine is `GGTTS_ERR_ARG`. The existing functions keep
  their signatures and behaviour, and the five-parameter Kotlin `Ggtts.load` stays in the compiled class, so code
  built against it runs without recompiling. Without a speaker the binding still uses `hk_cantonese_speaker`, so the
  v0.0.5 talker (`hk_voice_343`, `hk_voice_777`) needs one; resolving the default model and its default speaker in
  the binding is [#164](https://github.com/dandelion-love/ggtts/issues/164).

Pull requests: [#165](https://github.com/dandelion-love/ggtts/pull/165), promoted in
[#166](https://github.com/dandelion-love/ggtts/pull/166).

## v0.1.0-rc.9 (2026-11-11)

Superseded by v0.1.0-rc.10, which renders the same from the command line.

### Fixed

- **The Linux CPU archive starts on Ubuntu 22.04 again.** The archives of v0.1.0-rc.7 and rc.8 were built on Ubuntu
  24.04 and needed glibc 2.39 (`GLIBC_2.38`/`GLIBC_2.39` not found on older systems,
  [#85](https://github.com/dandelion-love/ggtts/issues/85)). The archive is now built on Ubuntu 22.04, and the
  release refuses a Linux CPU binary that needs more than Ubuntu 22.04 ships: glibc 2.35, libstdc++ `GLIBCXX_3.4.30`
  and `CXXABI_1.3.13`, libgcc_s `GCC_12.0.0`, libgomp `GOMP_5.1`/`OMP_5.1` (`scripts/check_glibc_floor.py`). The
  CUDA archive was already built on Ubuntu 22.04 and is unchanged.

### Added

- **`ggtts-render --instruction TEXT`** passes a natural-language style instruction ("Speak slowly and softly.") to a
  Qwen3-TTS checkpoint that takes one: `custom_voice` at 1.7B, or `voice_design`. It is placed before the prompt as
  the reference runtime places it, and the tokens match the reference's on CPU. A `base` checkpoint, a 0.6B
  `custom_voice` checkpoint and the other engines refuse it with an error naming the checkpoint type; an empty
  instruction is no instruction, and renders exactly as before. In the library it is
  `ChunkRequest::instruction` and `RenderOptions::instruction`. A story trace's `chunk_begin` records it as
  `instruction` when there is one; a strict `ggtts-trace/2` reader built against the older schema must allow that
  field. Code that builds a `ChunkRequest` must now set `instruction` (`None` for none). In this release the C and
  Kotlin binding does not carry an instruction; v0.1.0-rc.10 adds it.
- **A render with no model downloads the default one.** `ggtts-render --text "..." --out x.flac` with no `--model`
  renders with Qwen3-TTS on the talker and codec of `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` v0.0.5 (2.33 GB, pinned
  by commit and sha256) and the speaker `hk_voice_343`. The first run says what it will download, from where, how
  large and where to (at every log level), reports progress every 5 s, then downloads over HTTPS only into a
  per-user cache and verifies each file before use; later runs download nothing. A download that receives nothing
  for 60 s fails. `GGTTS_CACHE_DIR` moves the cache. `--offline` never downloads, and names any missing file and its URL.
  A path given with `--model` or `--codec` is used as it is, never downloaded over or held to a pinned hash, and may
  be mixed with a default file. Until now a render without `--model` was an error. See "Default model files" in
  the README.
- **Every report records the model files** under a new `models` list: role, path, sha256, whether the file came
  from the cache or a given path, and a cached file's URL. It is in the timing report, the duration report, the
  instrument `report.json` (a render's, not `ggtts-measure`'s) and `run.log`, and the trace header, which now also
  fills `model_sha256` and `codec_sha256`. Every render prints the same at the default log level. A given file is
  hashed once per run. The formats keep their versions (`ggtts-timing/1`, `ggtts-duration/1`, `ggtts-report/2`,
  `ggtts-trace/2`); a document without `models` is still valid.
- **ggtts-render links an HTTPS client** (ureq with rustls and its bundled Mozilla roots; no OpenSSL or other
  system TLS library). `licenses/rust-crates.txt` lists the crates it adds.
- **The Kitten tools ship in every archive.** `ggtts-kitten-convert` (the language model and decoder GGUFs from the
  published model) and `ggtts-kitten` (`prepare`, a reference voice from a clip) sit beside `ggtts-render`, so a
  Kitten render needs only an archive and the upstream model files, and no Python. A Kitten GGUF converted before
  the tokenizer was embedded is refused with the command that converts it again; every refusal on that path names
  the file it expected and the shipped command that makes it. The archive carries the Kitten guide as
  `README-kitten.md`, with the model's pass-through terms, and the licences of the sources the engine ports.
  In the CUDA archive the Kitten tools are CPU builds: they need no GPU, and a CUDA-linked `ggtts-kitten` would
  have carried a second copy of the CUDA kernels.
- **Every binary's `--version` prints the ggtts commit** it was built from, between the tag and the ggml pin; the
  release refuses an archive whose binaries disagree with `BUILDINFO.json`'s `ggtts_commit`.

Pull requests: [#156](https://github.com/dandelion-love/ggtts/pull/156),
[#157](https://github.com/dandelion-love/ggtts/pull/157), [#159](https://github.com/dandelion-love/ggtts/pull/159),
[#162](https://github.com/dandelion-love/ggtts/pull/162), promoted in
[#163](https://github.com/dandelion-love/ggtts/pull/163).

## v0.1.0-rc.8 (2026-11-11)

Superseded by v0.1.0-rc.9: this release's Linux CPU archive does not start on Ubuntu 22.04 (glibc 2.35), and its
archives cannot render Kitten-TTS 2 models without building the Kitten tools from source.

### Read this before upgrading from v0.1.0-rc.7

- **A story can render differently at the same seed.** The story splitter used to drop short fragments and some
  punctuation; it now keeps every non-whitespace character of the input, in order. A story whose chunks change gets
  different audio from rc.7 at the same seed; a story whose chunks are unchanged keeps its chunks and seeds.
- **Invalid input is refused before anything is rendered.** A request with nothing to say (for example only
  punctuation) is now an error, reported before the model is loaded or an output file is created.
- **Traces are `ggtts-trace/2`.** `ggtts-render --trace` writes `ggtts-trace/2`; this release still reads
  `ggtts-trace/1`. A reader that accepts only `/1` must be updated. The header now records the trace level
  (`--trace-level 1|2`) and what `p_stop` means.
- **Reports are `ggtts-report/2`.** Instrument envelopes and the merged report are written as `ggtts-report/2`
  (`schemas/ggtts-report-2.schema.json`). `ggtts-report/1` envelopes are still read and are normalized to `/2`; a
  reader that accepts only `/1` cannot read the new reports.
- **`p_stop` means something else for Qwen3-TTS.** It is now the end-of-speech share of the weights the sampler
  actually draws from (after temperature, top-k and top-p), labelled `qwen-sampler-weights/1` in the trace header
  and in each stop-hook record. Values from rc.7 traces and reports are not comparable with new ones; an old,
  unlabelled trace still reads, with its `p_stop` meaning reported as `unknown`. Strict readers of the stop-hook
  JSONL must allow the new `p_stop_semantics` field.
- **The x86-64 CPU archives need AVX2, FMA, F16C and BMI2**, and an operating system that enables AVX state. They
  do not need AVX-512, and they no longer depend on the instruction set of the machine that built them.

### Added

- **`ggtts-render --speaker NAME`** selects a named speaker of a Qwen3-TTS model, for example
  `--speaker hk_voice_343` or `--speaker hk_voice_777` with `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` v0.0.5. Without
  it the default is still `hk_cantonese_speaker`. A name the model does not have is an error that lists the
  model's speaker names. The option is refused for other engines.
- **`ggtts-measure --audio-state raw|trimmed|unknown`** records what you know about the audio you hand it; the
  default is `unknown`. Reports no longer claim `trimmed` for audio whose state was never stated.
- **Timing in traces and a `render-timing` instrument**: the time to first audio, the render's real-time factor and
  per-stage totals, from a streaming story trace. `ggtts-render --timing-report FILE` writes a `ggtts-timing/1` report
  (`docs/timing-and-logging.md`).
- **A `pitch-range` instrument**: the median and range of voiced pitch. **It needs Node.js** on the machine that
  runs it; its detector (Pitchy and fft.js) is bundled in the archive, so it needs no npm install and no network.
  Without Node.js the instrument is skipped, and a run that requires it is incomplete.
- **A `loudness-ends` instrument**: the loud and soft ends of the speech level. It needs the Python environment in
  its README (NumPy, SciPy, librosa, soundfile).
- **Two optional speech-recognition instruments, `asr-sensevoice` and `asr-qwen`**: character error rate of a
  render against a reference text, with a model you supply. They run on the CPU by default under `ggtts-measure`,
  and on CUDA when asked (`--instrument-device`).
- **Kitten-TTS 2 models in the experimental `TQ1_0` layout load on the CPU and on CUDA** (`docs/tq1-cpu.md`). On
  CUDA each matrix is expanded at load to f16, checked value for value. The CPU's packed path and CUDA are different
  numeric paths, so the same model and seed do not give the same tokens on the two
  ([#152](https://github.com/dandelion-love/ggtts/issues/152)). This changelog as published with rc.8 said they
  were refused on CUDA; the rc.8 code already loaded them there.
- **Licence notices in every archive**: `THIRD_PARTY.md`; `licenses/ggml-LICENSE.txt` for the statically linked
  ggml; `licenses/rust-crates.txt`, generated from the dependency tree, with the licence texts of every Rust crate
  linked into the two binaries; and the licence file beside each bundled third-party source (Pitchy, fft.js, the
  speech-core selection). The CUDA and cuBLAS libraries are not bundled and are not covered.

Pull requests: [#86](https://github.com/dandelion-love/ggtts/pull/86),
[#90](https://github.com/dandelion-love/ggtts/pull/90), [#92](https://github.com/dandelion-love/ggtts/pull/92),
[#93](https://github.com/dandelion-love/ggtts/pull/93), [#95](https://github.com/dandelion-love/ggtts/pull/95),
[#96](https://github.com/dandelion-love/ggtts/pull/96), [#97](https://github.com/dandelion-love/ggtts/pull/97),
[#98](https://github.com/dandelion-love/ggtts/pull/98), [#99](https://github.com/dandelion-love/ggtts/pull/99),
[#100](https://github.com/dandelion-love/ggtts/pull/100), [#103](https://github.com/dandelion-love/ggtts/pull/103),
[#114](https://github.com/dandelion-love/ggtts/pull/114), [#115](https://github.com/dandelion-love/ggtts/pull/115),
[#117](https://github.com/dandelion-love/ggtts/pull/117), [#122](https://github.com/dandelion-love/ggtts/pull/122),
[#123](https://github.com/dandelion-love/ggtts/pull/123), [#125](https://github.com/dandelion-love/ggtts/pull/125),
[#126](https://github.com/dandelion-love/ggtts/pull/126), [#128](https://github.com/dandelion-love/ggtts/pull/128),
[#129](https://github.com/dandelion-love/ggtts/pull/129), [#133](https://github.com/dandelion-love/ggtts/pull/133),
[#134](https://github.com/dandelion-love/ggtts/pull/134), [#135](https://github.com/dandelion-love/ggtts/pull/135),
[#136](https://github.com/dandelion-love/ggtts/pull/136), [#138](https://github.com/dandelion-love/ggtts/pull/138),
[#141](https://github.com/dandelion-love/ggtts/pull/141), [#142](https://github.com/dandelion-love/ggtts/pull/142),
[#144](https://github.com/dandelion-love/ggtts/pull/144), [#146](https://github.com/dandelion-love/ggtts/pull/146),
[#148](https://github.com/dandelion-love/ggtts/pull/148), [#149](https://github.com/dandelion-love/ggtts/pull/149),
and [#151](https://github.com/dandelion-love/ggtts/pull/151), promoted in
[#131](https://github.com/dandelion-love/ggtts/pull/131). #102, #105, #109 and #116 were merged into feature branches
and reached rc through the pull requests above.

## v0.1.0-rc.7

Released before this changelog existed; the tree at the tag has no CHANGELOG.md. It reached `main` through
[#84](https://github.com/dandelion-love/ggtts/pull/84). Its release page is no longer published; the tag
`v0.1.0-rc.7` remains.
