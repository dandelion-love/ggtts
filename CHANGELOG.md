# Changelog

What changes for someone using ggtts, release by release: new options, changed file formats, changed output, and
new requirements. Internal refactors and test changes are not listed.

## Unreleased: the next release candidate, after v0.1.0-rc.7

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

### Fixed

- **The Linux CPU archive starts on Ubuntu 22.04 again.** The archives of v0.1.0-rc.7 and rc.8 were built on Ubuntu
  24.04 and needed glibc 2.39 (`GLIBC_2.38`/`GLIBC_2.39` not found on older systems,
  [#85](https://github.com/dandelion-love/ggtts/issues/85)). The archive is now built on Ubuntu 22.04, and the
  release refuses a Linux CPU binary that needs more than Ubuntu 22.04 ships: glibc 2.35, libstdc++ `GLIBCXX_3.4.30`
  and `CXXABI_1.3.13`, libgcc_s `GCC_12.0.0`, libgomp `GOMP_5.1`/`OMP_5.1` (`scripts/check_glibc_floor.py`). The
  CUDA archive was already built on Ubuntu 22.04 and is unchanged.

### Added

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
- **Kitten-TTS 2 models in the experimental `TQ1_0` layout load on the CPU** (`docs/tq1-cpu.md`). They are refused
  on CUDA.
- **Licence notices in every archive**: `THIRD_PARTY.md`; `licenses/ggml-LICENSE.txt` for the statically linked
  ggml; `licenses/rust-crates.txt`, generated from the dependency tree, with the licence texts of every Rust crate
  linked into the two binaries; and the licence file beside each bundled third-party source (Pitchy, fft.js, the
  speech-core selection). The CUDA and cuBLAS libraries are not bundled and are not covered.

## v0.1.0-rc.7

The first release candidate with a changelog entry; see its
[release page](https://github.com/dandelion-love/ggtts/releases/tag/v0.1.0-rc.7).
