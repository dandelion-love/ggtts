# ggtts
An instrumented text-to-speech engine on ggml, written in Rust: one engine per model architecture, a pluggable compute stack, and token-exact parity tests

> **Status: release candidate.** Qwen3-TTS and Kitten-TTS 2 render with ggml on CPU and, in the Linux CUDA build,
> on supported NVIDIA GPUs. Release archives include `ggtts-render` and `ggtts-measure`; model files are separate.

## What it is

ggtts turns text into speech with neural codec language models (Qwen3-TTS first), running on
[ggml](https://github.com/ggml-org/ggml). It is built around three ideas.

**Two plug points, kept apart.**

* **The architecture.** Each model family is one engine behind a shared interface: text and a seed in, codec tokens
  out, and audio from the tokens. A new family is a new engine crate. It is not a change to the core.
* **The compute stack.** Each family's forward pass runs on a backend chosen at build time or at run time. A backend
  is the *model runner* for that family: it loads the weights, runs a step and returns logits. ggml is the first
  backend and the reference. Pluggability sits at that level, never at single tensor operations. The sampler and
  the seed stay above the backend, so a seed means the same thing on every backend.

**A pipeline for long text.** Long text is split into chunks with a deterministic rule. Each chunk gets its own seed,
derived from the text and its position. Each chunk is rendered, judged, trimmed and joined, and the read is levelled
and written as FLAC with a provenance manifest. The defaults are a fixed, versioned protocol: a render can be
reproduced from its command line and its manifest. The core holds the rules and no protocol number: the chunk
ceilings, the pause table, the judge bars and the level target are a profile the engine declares, named
`{protocol}/{family}`, for example `story-v1/qwen3-tts`.

**Instruments, opt in.** For a story, `--trace FILE` writes the complete `ggtts-trace/2` event stream as JSON
Lines, including per-step values the engine supplies (Qwen supplies the token and stop probability). Kitten does not
yet emit per-step events. For `--text`, Qwen's `--trace` remains its older step-only diagnostic and is not a complete
event stream for instruments. With tracing off nothing is computed for it. With it on, the output must not change: an
instrument never draws from the sampler's random stream. The hook points, the event order and the two instrument
tiers are in [docs/instruments/hooks.md](docs/instruments/hooks.md), and the stream's JSON schema is
[schemas/ggtts-trace-2.schema.json](schemas/ggtts-trace-2.schema.json). Current readers also accept v1; older
v1-only readers cannot read v2.

**Running instruments.** `ggtts-render --instrument NAME` (or `--instruments FILE.yaml`) runs instruments on the
finished render, each in its own process with its own log, a timeout and a crash boundary; the template config
`instruments/instruments.yaml` lists every shipped one. The Python instruments are scripts, run under
`GGTTS_PYTHON` if set, else `python3` (`python` on Windows), and need Python 3.8 or later on `PATH`. They are found in the `instruments/` directory:
next to `ggtts-render` in a release archive, or the repository's own when run from a checkout. Set
`GGTTS_INSTRUMENTS_DIR` to use another. If none is found, the instrument is reported failed with that reason and
the render's audio is still written. Writing one: [instruments/README.md](instruments/README.md).
When measuring existing audio with `ggtts-measure`, choose a new `--report` directory: an existing directory is
refused so report files cannot overwrite an input through a path, symlink or hard link.

## Binaries

| Binary | Job |
|---|---|
| `ggtts-render` | Render text or a story to FLAC and/or a live PCM stream |
| `ggtts-measure` | Run instruments on existing audio |
| `ggtts-kitten` and `ggtts-kitten-convert` | Source-built Kitten voice preparation and GGUF conversion tools |

The release archives ship `ggtts-render` and `ggtts-measure`; the Kitten preparation and conversion tools are built
from source. `ggtts-render --version` lists the included engines and backends as well as the ggml pin.

### Streaming: which stream

- **To listen: `ggtts-render --stream FILE|-`.** The audio leaves chunk by chunk while the render runs, levelled with
  one gain set from its first 5 s of audio, which are held back until then. Its loudness is close to the FLAC's
  but not equal, because the FLAC's gain is computed over the whole read. `--stream-level-seconds S` trades the
  two: more seconds give loudness closer to the FLAC's, fewer give earlier first audio, and `0` takes the first
  chunk's gain alone, which adds no latency but goes wrong when the first chunk is unlike the rest.
- **For parity checks: `--stream FILE --stream-raw`.** The pieces before the level step, bit-exact to the batch read
  before its level step. The FLAC is this stream times the gain printed at the end. Tests and `stream_probe.sh` ask
  for it by name.

### Default model files: what ggtts downloads, and when

With no `--model`, `ggtts-render` renders with the Qwen3-TTS engine on its default files, so text is all a render
needs:

```bash
ggtts-render --text "我哋聽日一齊去海邊睇日落。" --out x.flac
```

The first such run downloads the two files below (2.33 GB in all) into a per-user cache, checks each against its
pinned sha256, and renders; later runs download nothing. Before it starts, the run says what it will download, from
where, how large, and where it will be stored. The default speaker is `hk_voice_343`; the model's other speaker is
`hk_voice_777`.

| Role | File | Size | sha256 | Licence |
|---|---|---|---|---|
| talker | `qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf` | 2,042,834,144 bytes | `4eb27e5a3434d15ca818814b8e4f0f1941716654afb8543866732b62f9227041` | Apache-2.0 |
| codec | `qwen-tokenizer-12hz-q8_0.gguf` | 291,150,624 bytes | `7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824` | Apache-2.0 |

Both come from [jmtl/hk-cantonese-cc0-qwen3-tts-GGUF](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF)
at its `v0.0.5` tag, fetched by the exact commit `e6a823291c0189031086865033d0f5e4ad02a8ac` so the files cannot
change under a pinned hash. "Latest" means the newest release this build of ggtts knows about; a newer release
arrives with a new ggtts. The table is `ggtts_models::QWEN3_TTS` in `crates/ggtts-models`.

**Licences of the files ggtts can fetch.** The repository's card gives the licence of the weights as "Apache-2.0,
matching Qwen3-TTS 12Hz 1.7B, which these weights derive from", and describes the codec as an unmodified conversion
of `Qwen/Qwen3-TTS-Tokenizer-12Hz` (Apache-2.0). The card also says the voice is one pseudonymous Common Voice
contributor's, dedicated under CC-0, and sets out responsible-use terms (the voice is a real person's: no
impersonation, and label synthetic audio); read the card before using the voice. These licences are the models',
not this repository's.

**A path you give always wins.** `--model` and `--codec` name files to use as they are: they are never downloaded
over and are not required to match a pinned hash, so ggtts renders unreleased models for experiments. A mix is
allowed: `--model my-talker.gguf` with no `--codec` uses your talker and the default codec. A given talker keeps
the legacy speaker default (see `--speaker` below). An engine with no default files (Kitten-TTS 2, for now) needs
its paths.

**Where the cache is.** `$XDG_CACHE_HOME/ggtts` (else `~/.cache/ggtts`) on Linux, `~/Library/Caches/ggtts` on macOS,
and `%LOCALAPPDATA%\ggtts\cache` on Windows. `GGTTS_CACHE_DIR` overrides it. Each file is stored at
`models/<sha256>/<file name>`, so a file from another release never takes its place. Deleting the directory is
safe; the next default render downloads again.

**How a file is checked.** A download goes to a temporary name beside its final one and is hashed as it arrives.
Only when its size and sha256 match the table is it renamed into place; on a mismatch the run fails naming the file
and both hashes, and on an interruption it fails naming how much arrived. Either way the partial file is deleted.
Hashing 2 GB on every render would cost seconds, so after a file is verified a sidecar (`<file name>.verified`)
records its hash, size and modification time, and later runs trust the file while all three still match. Anything
else (a missing or edited sidecar, a changed size or time) makes the run hash the file again, and a file that no
longer matches is removed and downloaded again. The trade-off: an edit that keeps both the size and the
modification time goes unnoticed until the sidecar is deleted. Two renders starting at once each download to their
own temporary name and the rename is atomic, so a race costs a second download, never a corrupt file. A temporary
file left by a killed process is removed by the next download of that file once it is ten minutes old.

**Offline.** `--offline` never downloads. A default file that is not already in the cache is then an error that
names the file and its URL, so it can be fetched by other means and placed in the cache, or given with
`--model`/`--codec`.

**What the report records.** `--timing-report FILE` lists every model file the render used under `models`: its
role, path, sha256, `source` (`cache` or `path`) and, for a cached file, the URL it came from. A cached file's hash
is the pinned one it was verified against; a given file is hashed when the report is written.

### Render and listen: Qwen and Kitten

These Bash examples use the `ggtts-render` from an unpacked [release](https://github.com/dandelion-love/ggtts/releases)
and `ffplay` from FFmpeg (on Ubuntu/WSL,
install the `ffmpeg` package). Replace the paths below with your own files. Releases do **not** include model
weights, a prepared Kitten voice, or `ffplay`. The examples were checked on Linux/WSL; use a Bash shell for the
binary pipelines rather than a shell that converts piped output to text.
On Linux/WSL, use the CUDA archive and `DEVICE=cuda` for a supported NVIDIA GPU with the CUDA runtime installed;
set `DEVICE=cpu` for a CPU archive. On Windows use the CPU archive and its `ggtts-render.exe`.
Both Linux archives need glibc 2.35 or newer (Ubuntu 22.04 or later), with the matching libstdc++, libgcc_s and
libgomp (see [Building](#building)). The CPU archives of v0.1.0-rc.7 and rc.8 needed glibc 2.39 and did not
start on Ubuntu 22.04 ([#85](https://github.com/dandelion-love/ggtts/issues/85)). The Linux CUDA archive was tested on Ubuntu 22.04;
it needs the NVIDIA driver and CUDA runtime libraries even when its CPU backend is selected.

`--engine` chooses the model architecture. `--model` is its language-model GGUF and `--codec` is its **audio
decoder GGUF**, not an output file format or an audio encoder selector. Qwen needs the talker and its matching
12 Hz tokenizer/codec; Kitten needs its LM and S3Gen GGUFs plus a prepared reference-voice JSON. The Qwen model
and codec used by this project are in [jmtl/hk-cantonese-cc0-qwen3-tts-GGUF](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF)
and described in the [Qwen README](https://github.com/dandelion-love/ggtts/blob/main/crates/ggtts-qwen3-tts/README.md).
To convert Kitten and prepare a voice, follow the [Kitten README](https://github.com/dandelion-love/ggtts/blob/main/crates/ggtts-kitten/README.md#convert-prepare-render).
Those Kitten tools are source-built and are not in the release archive. The models' own licences apply.

For a Qwen model with named speakers, select the name in its model card with `--speaker NAME`. The Cantonese
v0.0.5 model uses `--speaker hk_voice_343` (or `hk_voice_777` for its other voice). Without the option, the
default model uses `hk_voice_343`, and a talker given with `--model` uses the legacy `hk_cantonese_speaker` name,
as for v0.0.1. An unknown or absent speaker is an error; ggtts does
not choose an arbitrary voice. `--speaker` applies only to Qwen, while `--voice` supplies Kitten's reference JSON.

```bash
# Run from the extracted archive; edit these paths and DEVICE for your machine.
GGTTS=./ggtts-render                    # on Windows: ./ggtts-render.exe
DEVICE=cuda                            # or cpu; Windows release is CPU-only
QWEN_MODEL=/path/to/qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf
QWEN_CODEC=/path/to/qwen-tokenizer-12hz-q8_0.gguf
KITTEN_MODEL=/path/to/kitten-lm.gguf
KITTEN_CODEC=/path/to/kitten-s3gen.gguf
KITTEN_VOICE=/path/to/prepared-voice.json

# First line is the title; following non-empty lines are paragraphs.
cat > qwen-story.txt <<'EOF'
Morning walk
今日天氣好好，我哋一齊去公園行下。
行完之後，我哋去飲杯茶。
EOF
cat > kitten-story.txt <<'EOF'
The lighthouse
The little lighthouse kept its lamp burning all through the stormy night.
In the morning, the fishing boats returned to the harbour.
EOF
```

**FLAC file plus live WAV playback.** `--out` writes a completed, lossless FLAC file; `--stream -` sends the live
stream to standard output. Diagnostics stay on standard error, so they do not corrupt the pipe. The live stream
and the completed FLAC can have different loudness, as described above.

```bash
# Qwen: talker + tokenizer/codec GGUF.
"$GGTTS" --engine qwen3-tts --device "$DEVICE" --model "$QWEN_MODEL" --codec "$QWEN_CODEC" \
  --story qwen-story.txt --out qwen.flac --stream - --stream-format wav \
  | ffplay -nodisp -autoexit -i -

# Kitten: LM + S3Gen GGUF + prepared reference voice.
"$GGTTS" --engine kitten-tts-2 --device "$DEVICE" --model "$KITTEN_MODEL" --codec "$KITTEN_CODEC" \
  --voice "$KITTEN_VOICE" --story kitten-story.txt --out kitten.flac --stream - --stream-format wav \
  | ffplay -nodisp -autoexit -i -

# Listen again after rendering.
ffplay -nodisp -autoexit qwen.flac
ffplay -nodisp -autoexit kitten.flac
```

To save the live WAV instead of piping it to a player, replace `--stream -` with `--stream qwen-live.wav`
or `--stream kitten-live.wav` and remove the pipe. The stream uses a WAV header with unknown length; some players
require a finalized WAV header. `--out` always writes FLAC and forces a `.flac` suffix, regardless of the suffix
given. Omit `--out` if only the stream is needed.

**Raw 16-bit PCM playback.** The other stream format is `s16le` (`raw` is an alias). It starts with the text line
`ggtts-pcm s16le rate=24000 channels=1` and a newline, followed by little-endian mono PCM. The models in these
examples output 24 kHz. The helper below removes and validates that header before sending the bytes to `ffplay`;
it refuses a different rate rather than playing it at the wrong speed. It needs Python 3.8+.

```bash
strip_ggtts_pcm_header() {
  python3 -u -c '
import sys
if sys.platform == "win32":
    import msvcrt, os
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
source = sys.stdin.buffer
if source.readline() != b"ggtts-pcm s16le rate=24000 channels=1\n":
    raise SystemExit("Expected a 24 kHz mono ggtts PCM stream")
while True:
    block = source.read1(65536)
    if not block:
        break
    sys.stdout.buffer.write(block)
'
}

"$GGTTS" --engine qwen3-tts --device "$DEVICE" --model "$QWEN_MODEL" --codec "$QWEN_CODEC" \
  --story qwen-story.txt --stream - --stream-format s16le \
  | strip_ggtts_pcm_header | ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 -i -

"$GGTTS" --engine kitten-tts-2 --device "$DEVICE" --model "$KITTEN_MODEL" --codec "$KITTEN_CODEC" \
  --voice "$KITTEN_VOICE" --story kitten-story.txt --stream - --stream-format s16le \
  | strip_ggtts_pcm_header | ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 -i -
```

To save that stream, use `--stream qwen.pcm` or `--stream kitten.pcm` instead of `--stream -` and omit the pipe;
the saved file **includes the header**, so strip it before treating it as plain PCM. For a different model's sample
rate, read its header and update both the helper and `ffplay -ar`. `--stream-format raw` selects the same container;
it is distinct from `--stream-raw`, which changes the levelled-versus-parity stream described above.

`--story` emits completed chunks as they become available, subject to the five-second levelling hold. A short
story may therefore be held until it finishes. `--text "..."` can replace `--story FILE` for one line, but the
stream receives that line only after the engine has generated and decoded it. `--stream-level-seconds 0` reduces
the story hold at the cost of potentially uneven loudness. The stream formats are WAV and s16le; FLAC is the
completed-file format. There is no native MP3, AAC or Opus encoder selection in `ggtts-render`.

## How parity is tested

The tests compare outputs, not impressions. They form a ladder:

1. **Codec tokens are identical.** For the same GGUF, text, seed and hardware class, the ggml backend must produce
   exactly the tokens a reference runtime produced. A mismatch reports the first chunk, frame and codebook that
   differ. A fixture recorded on a different model, ggml commit or hardware class is refused rather than compared.
2. **Decoder audio is identical** for identical tokens, or within a tolerance that is stated and justified where a
   kernel's reduction order differs.
3. **The pipeline matches its specification case by case.** Chunk boundaries, seeds, judge verdicts, trim bounds,
   gaps and levels are checked against recorded cases, using a fake engine with scripted output. A planted change,
   such as moving the chunk ceiling by one, must turn the suite red.
4. **Distributions, only where identity cannot hold**: across GPU models, or on a backend other than ggml. Such a
   backend declares its parity level as "distribution", and every render records the backend and its level.

The reference runtime is [qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) (MIT), which builds on ggml
commit `af97976` (v0.16.0+7). ggtts pins the same commit, because identical tokens need identical kernels.

Fixture text in this repository is written by a language model for the purpose. Fixtures hold text, seeds, token ids
and hashes, never audio.

## Building

```sh
cargo build
cargo test
cargo test -p ggtts-core --test pipeline_cases -- --nocapture   # the pipeline cases, with their summary
cargo run -p ggtts-cli --bin ggtts-render -- --version
```

`rust-toolchain.toml` pins the Rust toolchain. CI runs `cargo fmt`, `clippy -D warnings`, the tests on Linux and
Windows, and a planted change (the story chunk ceiling set to 44 must turn the pipeline cases red). Ordinary PRs do not
build the CUDA archive; the promotion PR into `main` builds and smokes it without a GPU, and the mandatory promotion gate exercises CUDA on a real GPU. A tag `v*` builds release archives named with the ggml pin, each
with a `BUILDINFO.json` and a build provenance attestation (`gh attestation verify <file> -R dandelion-love/ggtts`).

The `x86_64` CPU release archives require AVX2, FMA, F16C and BMI2 CPU support, plus operating-system support for AVX state. They do not require AVX-512. Other x86-64 CPUs are not covered by this release baseline.

The Linux archives' floor is Ubuntu 22.04: glibc 2.35, and the GCC 12 runtime libraries it ships (libstdc++ up to
`GLIBCXX_3.4.30` and `CXXABI_1.3.13`, libgcc_s `GCC_12.0.0`, libgomp `GOMP_5.1`/`OMP_5.1`). They link these
dynamically and bundle none of them. The CPU archive is built on GitHub's `ubuntu-22.04` runner and the CUDA archive
in an Ubuntu 22.04 container, because a binary needs the glibc it was linked against or newer.
`scripts/check_glibc_floor.py` reads the symbol versions each binary of the CPU archive needs
(`readelf --version-info`) and fails the release, and CI's release build, when any is above the floor. Older
systems (Ubuntu 20.04, glibc 2.31) are not supported.

Linux releases also include a separate `x86_64-linux-cuda` archive. It includes CUDA code for NVIDIA compute
capabilities 8.6 and 8.9; other GPU architectures are not covered by the tested release. It requires an NVIDIA
driver and the CUDA 12.6 runtime and cuBLAS libraries installed on the host and discoverable by the dynamic loader;
these libraries are not bundled. Never add CUDA's `stubs` directory to the runtime library path. NVIDIA documents
the [driver requirements](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html) and
[GPU compute capabilities](https://developer.nvidia.com/cuda/gpus). Run with `--device cuda`. Windows releases are
CPU-only. The CUDA release does not include NCCL for multi-GPU communication. The hosted release smoke test exercises
the packaged fake engine without a GPU; actual GPU inference
requires a separate test on a GPU-equipped host.

An experimental [Windows CUDA source build and local packaging guide](https://github.com/dandelion-love/ggtts/blob/rc/docs/windows-cuda-build.md)
is available; it is not yet a released or GPU-validated archive.

## The pipeline cases

`fixtures/pipeline/*.json` hold the reference implementation's recorded cases: text, numbers, decisions and hashes,
with synthetic signals described by their generator, never as samples. `out-of-scope.json` names every recorded
case the core does not implement and why, and the test summary lists them, so what is not covered is visible.

## The Kitten-TTS 2 engine

`crates/ggtts-kitten` runs [Kitten-TTS 2](https://huggingface.co/KittenML/kitten-tts-2) on ggml, on the CPU or CUDA:
a 1.7B speech LM with ternary (BitNet-style) transformer weights, Chatterbox-Turbo's S3Gen decoder, and voice
cloning from a reference clip, with no Python at run time. `ggtts-render` selects it from the GGUF's
architecture (`kitten-tts-2`) and takes `--codec` for the decoder GGUF and `--voice` for a prepared reference.

It reports the parity level `distribution` against Kitten's PyTorch path, with deterministic checks under it: the
ternary weights survive conversion exactly; LM logits are within stated bars of float32 PyTorch; the S3Gen decode
is at 89 dB SNR with recorded noise. It is `exact` against itself: the same inputs and seed give a byte-identical
FLAC. What it needs, how to convert and render, and the measurements: [`crates/ggtts-kitten/README.md`](crates/ggtts-kitten/README.md)
and [`DESIGN.md`](crates/ggtts-kitten/DESIGN.md).

**The model's licence is not this repository's.** Kitten-TTS 2's weights are under the Stellon Labs Community
License, and their outputs, including audio rendered here, must not be used to train any other model. No weights,
GGUFs or rendered audio are kept in this repository.

## Upgrading

What changed between releases, and what to check before upgrading, is in [`CHANGELOG.md`](CHANGELOG.md).

## Licence

MIT. Third-party sources and what is taken from each are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
