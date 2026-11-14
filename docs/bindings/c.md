# ggtts from C

The C library is `libggtts_ffi` with the header `ggtts.h`. It loads a text-to-speech engine, turns UTF-8 text into
mono 32-bit float samples, and hands them to you. Any language that can call C can use it.

This page covers the C ABI as of version 3 (`GGTTS_ABI_VERSION 3`). [All bindings](README.md) compares it with
Kotlin and Python.

## Get it

**From a release.** Releases after v0.1.0-rc.10 carry the library and the header in every archive on the
[releases page](https://github.com/dandelion-love/ggtts/releases). (v0.1.0-rc.10 and earlier have no `lib/`.)

| Archive | Library, in `lib/` | Header |
|---|---|---|
| `ggtts-<tag>-x86_64-linux-cpu-ggml-<pin>.tar.gz` | `libggtts_ffi.so` | `include/ggtts.h` |
| `ggtts-<tag>-x86_64-linux-cuda-ggml-<pin>.tar.gz` | `libggtts_ffi.so` (the CPU build: see [CUDA](#cuda)) | `include/ggtts.h` |
| `ggtts-<tag>-x86_64-windows-cpu-ggml-<pin>.zip` | `ggtts_ffi.dll`, and `ggtts_ffi.dll.lib`, the import library you link against | `include/ggtts.h` |

Each archive's `BUILDINFO.json` lists the library under `"libraries"`. Check an archive before you use it:
`gh attestation verify <file> -R dandelion-love/ggtts`.

**From source.** You need the Rust toolchain that `rust-toolchain.toml` names, CMake and a C++ compiler.

```sh
git clone https://github.com/dandelion-love/ggtts && cd ggtts
git submodule update --init crates/ggtts-ggml-sys/ggml
cargo build --release --locked -p ggtts-ffi --features ggml     # target/release/libggtts_ffi.so
```

The header is `crates/ggtts-ffi/include/ggtts.h`. Without `--features ggml` the library has only the `fake` test
engine. On Windows, build as the release does, which generates the ggml bindings and needs libclang
(`LIBCLANG_PATH`): `cargo build --release --locked -p ggtts-cli -p ggtts-kitten -p ggtts-ffi --features
ggtts-cli/bindgen,ggtts-ffi/ggml`.

## Requirements

* **Linux:** x86-64 with AVX2, FMA, F16C and BMI2. glibc 2.35 or newer (Ubuntu 22.04 or later), libstdc++,
  libgcc_s and libgomp. A minimal system, such as a slim container image, may lack libgomp: install `libgomp1`
  (`apt-get install libgomp1` on Debian and Ubuntu).
* **Windows:** x86-64 with the same CPU features, and the Microsoft Visual C++ Redistributable for x64, which the
  archive does not include. Read from the imports of `ggtts_ffi.dll` as CI's release build makes it (the release's
  build command, by `scripts/check_pe_imports.py`): beyond Windows' own libraries, it needs `VCRUNTIME140.dll`,
  `VCRUNTIME140_1.dll`, `MSVCP140.dll` and `VCOMP140.dll` (OpenMP), all from that redistributable. MinGW is not
  tested.
* **Disk and network**, for Qwen3-TTS's default model: 2.33 GB, downloaded once, only if you allow it (below).

## Build your program

Linux. The library has no soname, so keep an rpath to `lib/`, or copy `libggtts_ffi.so` beside your program and link
with `-Wl,-rpath,'$ORIGIN'`:

```sh
GGTTS=/path/to/the/unpacked/archive
cc -I"$GGTTS/include" first_render.c -L"$GGTTS/lib" -lggtts_ffi -Wl,-rpath,"$GGTTS/lib" -o first_render
```

Windows. Link the import library, and put `ggtts_ffi.dll` beside your `.exe` (or `lib\` on `PATH`) when it runs:

```bat
cl /I %GGTTS%\include first_render.c %GGTTS%\lib\ggtts_ffi.dll.lib
rem or, with LLVM: clang -I %GGTTS%\include first_render.c %GGTTS%\lib\ggtts_ffi.dll.lib -o first_render.exe
```

## A first render

This program loads Qwen3-TTS, renders one line and writes `first_render.wav`. Run it with no arguments. The first run
downloads the default model (2.33 GB) into the cache and checks its sha256; later runs download nothing.

<!-- example: examples/c/first_render.c -->
```c
/* first_render.c: load an engine, render one line, write it to a WAV file.
 *
 *   ./first_render                                  Qwen3-TTS with its default model (downloaded once, 2.33 GB)
 *   ./first_render qwen3-tts talker.gguf codec.gguf  Qwen3-TTS with model files you already have
 *   ./first_render kitten-tts-2 lm.gguf s3gen.gguf voice.json
 *   ./first_render fake                             the test engine: no files, a tone instead of speech
 *
 * It writes first_render.wav. */
#include <stdio.h>
#include <stdlib.h>

#include "ggtts.h"

/* Write mono float samples as a 16-bit PCM WAV file. Returns 0 on success. */
static int write_wav(const char *path, const float *samples, size_t len, uint32_t rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data = (uint32_t)(len * 2);
    unsigned char h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                           16, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 16, 0,
                           'd', 'a', 't', 'a', 0, 0, 0, 0};
    uint32_t fields[4][2] = {{4, 36 + data}, {24, rate}, {28, rate * 2}, {40, data}};
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++) h[fields[i][0] + b] = (unsigned char)(fields[i][1] >> (8 * b));
    fwrite(h, 1, sizeof h, f);
    for (size_t i = 0; i < len; i++) {
        float x = samples[i] > 1.0f ? 1.0f : samples[i] < -1.0f ? -1.0f : samples[i];
        int16_t s = (int16_t)(x * 32767.0f);
        unsigned char le[2] = {(unsigned char)(s & 0xff), (unsigned char)((s >> 8) & 0xff)};
        fwrite(le, 1, 2, f);
    }
    return fclose(f) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *engine = argc > 1 ? argv[1] : "qwen3-tts";
    const char *model = argc > 2 ? argv[2] : NULL; /* NULL: the default model (qwen3-tts only) */
    const char *codec = argc > 3 ? argv[3] : NULL;
    const char *voice = argc > 4 ? argv[4] : NULL; /* kitten-tts-2's prepared reference voice */

    /* The settings below are ABI version 3. A library older than this header refuses them, so say so plainly. */
    if (ggtts_abi_version() < 3) {
        fprintf(stderr, "this libggtts_ffi is ABI version %u; this program needs 3\n", (unsigned)ggtts_abi_version());
        return 1;
    }

    ggtts_settings settings = GGTTS_SETTINGS_INIT; /* every field unset: CPU, no speaker, no download */
    settings.allow_download = 1;                   /* fetch the default model files if the cache lacks them */

    uint64_t tts = 0;
    if (ggtts_load_with(engine, model, codec, voice, 4, &settings, &tts) != GGTTS_OK) {
        fprintf(stderr, "load failed: %s\n", ggtts_last_error());
        return 1;
    }

    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    /* UTF-8 text; one paragraph per line. 20260905 is the seed the other bindings use by default. */
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82"; /* 今日天氣好好。 */
    if (ggtts_render(tts, text, 20260905, &pcm, &len, &rate) != GGTTS_OK) {
        fprintf(stderr, "render failed: %s\n", ggtts_last_error());
        ggtts_free(tts);
        return 1;
    }

    int written = write_wav("first_render.wav", pcm, len, rate);
    printf("%lu samples at %u Hz (%.2f s)%s\n", (unsigned long)len, (unsigned)rate, (double)len / rate,
           written == 0 ? ", written to first_render.wav" : "; could not write first_render.wav");

    ggtts_pcm_free(pcm, len); /* the buffer is the library's: free it with ggtts_pcm_free, never free() */
    ggtts_free(tts);
    return written == 0 ? 0 : 1;
}
```

The text is written as escaped UTF-8 bytes so any compiler reads it the same way. If you write Chinese text directly
in your source, save the file as UTF-8 (and compile with `/utf-8` under MSVC).

## The functions

Every `int32_t` function returns `GGTTS_OK` (0) or a negative [error code](#errors). The message for the last error on
the calling thread is `ggtts_last_error()`.

| Function | What it does |
|---|---|
| `ggtts_load_with(engine, model, codec, voice, threads, &settings, &handle)` | Load an engine and write its handle. Use this one. |
| `ggtts_load(engine, model, codec, voice, threads, &handle)` | Version 1: `ggtts_load_with` with no settings. It never uses the default model, so it needs paths. |
| `ggtts_abi_version()` | The ABI version of the library actually loaded. |
| `ggtts_render(handle, text, seed, &pcm, &len, &rate)` | Render the whole text. `pcm` gets `len` samples at `rate` Hz. |
| `ggtts_render_stream(handle, text, seed, on_piece, user)` | Render, handing each piece to `on_piece` as soon as it is ready. |
| `ggtts_pcm_free(pcm, len)` | Free a buffer `ggtts_render` returned. |
| `ggtts_free(handle)` | Release the engine. |
| `ggtts_last_error()` | This thread's last error message. Empty after a success. |
| `ggtts_load_opts(json, &handle)` | Version 4: load from an [options document](#options-documents). |
| `ggtts_render_opts(handle, text, json, on_piece, user, &pcm, &len, &rate)` | Version 4: render, whole or streamed, with an options document. |

## Loading

`ggtts_load_with(engine, model, codec, voice, threads, settings, &handle)`:

| Argument | Meaning |
|---|---|
| `engine` | `"qwen3-tts"`, `"kitten-tts-2"`, or `"fake"` (a test engine that needs no files). |
| `model` | The model GGUF: Qwen3-TTS's talker, or Kitten-TTS 2's LM. NULL for Qwen3-TTS means the default model (below). |
| `codec` | The audio decoder GGUF: Qwen3-TTS's 12 Hz codec, or Kitten-TTS 2's S3Gen GGUF. NULL for Qwen3-TTS means the default codec. |
| `voice` | Kitten-TTS 2's prepared reference voice (a JSON file). NULL for the other engines. |
| `threads` | CPU threads. 0 or less means 4. Before each graph, ggtts lowers the count to the CPUs the process may use at that moment, re-read at most every 100 ms, and says so once on stderr. On Windows only the processor count is seen, not affinity or job objects. |
| `settings` | A `ggtts_settings`, or NULL for none. |

A file that is missing or is not a GGUF is refused before any engine loads.

Start the settings from `GGTTS_SETTINGS_INIT`, so every field you do not set is NULL or 0. Strings are UTF-8, owned
by you, and read only during the call. A NULL or empty string is no setting.

| Field | Meaning |
|---|---|
| `size` | `sizeof(ggtts_settings)`. `GGTTS_SETTINGS_INIT` sets it. |
| `speaker` | A Qwen3-TTS speaker the model has, such as `"hk_voice_777"`. Case is ignored. |
| `instruction` | A Qwen3-TTS style instruction for every chunk, such as `"Speak slowly and softly."` |
| `device` | `"cpu"` (the default) or `"cuda"`. See [CUDA](#cuda). |
| `cache_dir` | Where the default model files are cached. NULL: the per-user cache (below). |
| `allow_download` | `1`: download a default model file the cache lacks. `0` (the default): never download. |

The settings hold for the engine's life: every render on the handle uses them.

`GGTTS_KITTEN_FULL_HEAD=1` (exactly `1`) makes Kitten-TTS 2 compute its whole output head instead of the rows a step can
keep (see the crate README). It is a diagnostic switch read from the process environment only; no binding setting
carries it.

### The default model (Qwen3-TTS)

With engine `"qwen3-tts"` and `model` or `codec` NULL, the library uses the same default files as `ggtts-render`:
the v0.0.5 talker and codec of
[jmtl/hk-cantonese-cc0-qwen3-tts-GGUF](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF), 2.33 GB in all,
each checked against a pinned sha256. A path you give is used as it is, and you may give one and not the other.
Read the model's card before you use its voice; its licence is the model's, not this repository's.

* **Downloading is off unless you turn it on.** With `allow_download = 0`, a default file missing from the cache is
  `GGTTS_ERR_ENGINE`, and the message names the file and its URL. With `allow_download = 1`, the missing files are
  downloaded over HTTPS inside the call. The call blocks until they are verified and reports no progress. Ask your
  user before a 2.33 GB download.
* **The cache.** `$XDG_CACHE_HOME/ggtts` (else `~/.cache/ggtts`) on Linux, `~/Library/Caches/ggtts` on macOS, and
  `%LOCALAPPDATA%\ggtts\cache` on Windows. The environment variable `GGTTS_CACHE_DIR`, or the `cache_dir` setting,
  moves it. Each file is at `models/<sha256>/<file name>` under it. The README's
  [default model files](../../README.md#default-model-files-what-ggtts-downloads-and-when) section lists the files,
  their hashes and how a download is checked.
* **The speaker.** The default model's speakers are `hk_voice_343` (used when you give none) and `hk_voice_777`. A
  model you give by path uses `hk_cantonese_speaker` when you give none, so a talker without that speaker needs
  `speaker` set. An unknown speaker is `GGTTS_ERR_ENGINE`, and the message lists the model's speakers.

`ggtts_load`, NULL settings and a version-2 struct never use the default model: with no model they are
`GGTTS_ERR_ARG`.

### Kitten-TTS 2

Kitten-TTS 2 needs all three files: `model` (the LM GGUF), `codec` (the S3Gen GGUF) and `voice` (a reference voice
made by `ggtts-kitten prepare`). It has no default model, and no ggtts-ready Kitten files are published: make them
once with `ggtts-kitten-convert` and `ggtts-kitten`, which the release archives include, as
[the Kitten README](../../crates/ggtts-kitten/README.md) shows. It takes no speaker and no instruction: either one is
`GGTTS_ERR_ARG`. Its licence is the model's: audio rendered with it must not be used to train other models.

```sh
./first_render kitten-tts-2 kitten-lm.gguf kitten-s3gen.gguf prepared-voice.json
```

### CUDA

`device = "cuda"` renders on an NVIDIA GPU, but only in a library built with the `cuda` feature:

```sh
cargo build --release --locked -p ggtts-ffi --features cuda
```

**The library in the release archives is the CPU build, the CUDA archive's included.** It loads without the NVIDIA
driver, and it refuses `"cuda"` with `GGTTS_ERR_ENGINE` before reading any file. A CUDA build on a machine with no
usable GPU refuses `"cuda"` with `GGTTS_ERR_ENGINE` while loading. Any other device name is `GGTTS_ERR_ARG`, and the
fake engine takes `"cpu"` only. Kitten-TTS 2 takes the setting too, but its CUDA path through this library is not yet
tested. Building it needs the CUDA toolkit, and running it needs the NVIDIA driver and the CUDA runtime and cuBLAS
libraries it was built against. The README's [Building](../../README.md#building) section lists the GPUs the
release's CUDA build supports.

## Rendering

`ggtts_render(handle, text, seed, &pcm, &len, &rate)` renders the whole text before it returns.

* **Text** is UTF-8, one paragraph per line. Empty lines are skipped. Text with nothing to say (empty, or only
  punctuation) is `GGTTS_ERR_ARG`, refused before any rendering.
* **The render** splits the text into chunks, renders each, checks it, trims it, joins them and levels the whole read,
  as `ggtts-render --story` does.
* **The seed** fixes the random draws. The same engine, model, text and seed give the same samples on the same
  platform. Kotlin and Python default to 20260905; in C you always pass one.
* **The result** is `len` mono float samples at `rate` Hz (24000 for the models above). Free it with
  `ggtts_pcm_free(pcm, len)`, never with `free()`. It stays valid after `ggtts_free(handle)`.

## Streaming and stopping

`ggtts_render_stream(handle, text, seed, on_piece, user)` renders the same text and calls
`on_piece(user, samples, len, rate)` once for each kept chunk, in order, so playback can start before the render ends.

* The callback runs on the thread that called `ggtts_render_stream`.
* `samples` belongs to the library and is valid only during the call. Copy it, or hand it to your audio device,
  before you return.
* Return 0 to go on. Return anything else to stop: the render stops and `ggtts_render_stream` returns
  `GGTTS_ERR_STOPPED`.
* The pieces are the render's samples *before* its final level step. Joined, they have the render's length, but not
  its exact values.

This program streams, stops after two pieces, tries a render from inside the callback, and shows three errors:

<!-- example: examples/c/stream.c -->
```c
/* stream.c: render piece by piece, stop early, and see the threading rules and error codes at work.
 *
 *   ./stream            Qwen3-TTS with its default model (it must already be in the cache: this never downloads)
 *   ./stream fake       the test engine
 *   ./stream qwen3-tts talker.gguf codec.gguf
 */
#include <stdio.h>

#include "ggtts.h"

struct listener {
    uint64_t tts;      /* the engine, so the callback can try to call back into it */
    int pieces;        /* pieces received so far */
    int stop_after;    /* ask to stop after this many pieces */
    int32_t reentered; /* what a render from inside the callback returned */
};

/* Called once per kept chunk, in order, on the thread that called ggtts_render_stream. The samples belong to the
 * library and are valid only during this call: copy them (or hand them to an audio device) before returning. */
static int32_t on_piece(void *user, const float *samples, size_t len, uint32_t rate) {
    struct listener *l = user;
    (void)samples;
    l->pieces++;
    printf("piece %d: %lu samples at %u Hz\n", l->pieces, (unsigned long)len, (unsigned)rate);

    /* One render at a time per engine: calling back into it from here is refused, never a deadlock. */
    float *pcm = NULL;
    size_t n = 0;
    uint32_t r = 0;
    l->reentered = ggtts_render(l->tts, "x", 1, &pcm, &n, &r);

    return l->pieces >= l->stop_after; /* 0: go on; anything else: stop */
}

int main(int argc, char **argv) {
    const char *engine = argc > 1 ? argv[1] : "qwen3-tts";
    const char *model = argc > 2 ? argv[2] : NULL;
    const char *codec = argc > 3 ? argv[3] : NULL;

    ggtts_settings settings = GGTTS_SETTINGS_INIT; /* allow_download stays 0: a missing default file is an error */
    uint64_t tts = 0;
    int32_t code = ggtts_load_with(engine, model, codec, NULL, 4, &settings, &tts);
    if (code != GGTTS_OK) {
        /* GGTTS_ERR_ENGINE here names a missing default file and its URL: run first_render once, or pass paths. */
        fprintf(stderr, "load failed (%d): %s\n", (int)code, ggtts_last_error());
        return 1;
    }

    /* Three paragraphs, so three or more pieces: 今日天氣好好。 / 我哋去海邊行下。 / 行完之後，去食飯。 */
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82\n"
                       "\xe6\x88\x91\xe5\x93\x8b\xe5\x8e\xbb\xe6\xb5\xb7\xe9\x82\x8a\xe8\xa1\x8c\xe4\xb8\x8b\xe3\x80\x82\n"
                       "\xe8\xa1\x8c\xe5\xae\x8c\xe4\xb9\x8b\xe5\xbe\x8c\xef\xbc\x8c\xe5\x8e\xbb\xe9\xa3\x9f\xe9\xa3\xaf"
                       "\xe3\x80\x82";
    struct listener l = {tts, 0, 2, GGTTS_OK};
    code = ggtts_render_stream(tts, text, 20260905, on_piece, &l);
    int ok = 1;
    if (code != GGTTS_ERR_STOPPED) {
        fprintf(stderr, "expected GGTTS_ERR_STOPPED after %d pieces, got %d: %s\n", l.stop_after, (int)code,
                ggtts_last_error());
        ok = 0;
    }
    if (l.reentered != GGTTS_ERR_BUSY) {
        fprintf(stderr, "expected GGTTS_ERR_BUSY from the callback, got %d\n", (int)l.reentered);
        ok = 0;
    }
    printf("stopped after %d pieces (code %d); a render from the callback returned %d (GGTTS_ERR_BUSY)\n", l.pieces,
           (int)code, (int)l.reentered);

    /* Text with nothing to say is an argument error, refused before rendering. */
    code = ggtts_render_stream(tts, "\xef\xbc\x9f\xef\xbc\x81", 1, on_piece, &l); /* "？！" */
    printf("punctuation only: %d (GGTTS_ERR_ARG): %s\n", (int)code, ggtts_last_error());
    ok &= code == GGTTS_ERR_ARG;

    /* A freed handle is never reused: every later call with it is GGTTS_ERR_HANDLE, not a crash. */
    ggtts_free(tts);
    code = ggtts_render_stream(tts, text, 1, on_piece, &l);
    printf("after ggtts_free: %d (GGTTS_ERR_HANDLE): %s\n", (int)code, ggtts_last_error());
    ok &= code == GGTTS_ERR_HANDLE;
    return ok ? 0 : 1;
}
```

## Options documents

Version 4 takes every option of the shared options table as a JSON document, so an option added to the table reaches C
with no new field, function or ABI version. The schema is `schemas/ggtts-options-1.schema.json`, and
[docs/options.md](../options.md) describes the sections. **`ggtts_settings` is frozen at version 3**: no field is ever
added to it, and the struct functions keep working unchanged. A document and a struct never meet in one call.

`ggtts_load_opts(json, &handle)` takes the `load` section, and `render` defaults that every render on the handle
starts from:

```json
{"load": {"engine": "qwen3-tts", "speaker": "hk_voice_777", "offline": true},
 "render": {"instruction": "Speak slowly and softly."}}
```

A document with no `model` loads the engine's default files, as `ggtts-render` does. A missing one is downloaded
unless `"offline": true`, which is the struct's `allow_download = 0`.

`ggtts_render_opts(handle, text, json, on_piece, user, &pcm, &len, &rate)` renders `text` with a render document,
laid over the load's defaults. NULL is the empty document.

* **A whole read** has no `output.stream`: `on_piece` must be NULL, and `pcm`, `len` and `rate` get the samples, as
  `ggtts_render` gives them.
* **A stream** is `{"output": {"stream": "raw"}}`: each kept chunk goes to `on_piece` before the level step, as
  `ggtts_render_stream` hands it. `on_piece` is required, and `pcm`, `len` and `rate` may be NULL.
* **What a render document takes:**
  * `render`: `seed_salt` (`ggtts_render`'s seed, as a string), `instruction`, `steer` with `steer_scale` and
    `steer_layers`, `redraw` and `edge_s`.
  * `observe`: `trace` with `trace_level`, a trace file as `ggtts-render --trace` writes it.
* **What it refuses:**
  * any `load` key (they are fixed when the engine loads);
  * `render.text`, `render.story` and `render.seed`: the text is the argument, and the seed is `seed_salt`;
  * `render.measure_timings`, the reports and logging: the C ABI returns PCM, not timings or files;
  * `output.out` and `output.stream_format`;
  * the book options.
  * A levelled stream (`"levelled"`, `stream_raw` and `stream_level_seconds`) is not taken yet.

An unknown key, a value of the wrong type, a malformed document, a stream with no `on_piece`, a whole read with one,
and an option the engine does not support (steering on Kitten-TTS 2, say) are each `GGTTS_ERR_ARG`, refused before
anything is rendered, and the message names the option.

<!-- example: examples/c/options.c -->
```c
/* options.c: load and render with options documents, the JSON form of every option (C-ABI version 4).
 *
 *   ./options           the test engine
 *   ./options fake      the same
 */
#include <stdio.h>

#include "ggtts.h"

struct counter {
    int pieces;
};

static int32_t count(void *user, const float *samples, size_t len, uint32_t rate) {
    struct counter *c = user;
    (void)samples;
    (void)len;
    (void)rate;
    c->pieces++;
    return 0;
}

int main(int argc, char **argv) {
    (void)argv;
    if (argc > 1 && argv[1][0] != 'f') {
        fprintf(stderr, "this example runs on the fake engine only\n");
        return 1;
    }
    if (ggtts_abi_version() < 4) {
        fprintf(stderr, "this library is C-ABI version %u: the options documents need 4\n", ggtts_abi_version());
        return 1;
    }
    /* The load document: the load section, and render defaults every render on the handle starts from. */
    uint64_t tts = 0;
    int32_t code = ggtts_load_opts("{\"load\": {\"engine\": \"fake\"}, \"render\": {\"seed_salt\": \"7\"}}", &tts);
    if (code != GGTTS_OK) {
        fprintf(stderr, "load failed (%d): %s\n", (int)code, ggtts_last_error());
        return 1;
    }
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82"; /* 今日天氣好好。 */
    int ok = 1;

    /* A whole read: no output.stream, so on_piece is NULL and the samples come back in pcm. NULL is the empty
     * document: the load's defaults as they are. */
    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    code = ggtts_render_opts(tts, text, "{\"render\": {\"edge_s\": 0.2}}", NULL, NULL, &pcm, &len, &rate);
    printf("whole read: %d, %lu samples at %u Hz\n", (int)code, (unsigned long)len, (unsigned)rate);
    ok &= code == GGTTS_OK && len > 0;
    ggtts_pcm_free(pcm, len);

    /* A stream: output.stream "raw" hands each kept chunk to on_piece before the level step; pcm may be NULL. */
    struct counter c = {0};
    code = ggtts_render_opts(tts, text, "{\"output\": {\"stream\": \"raw\"}}", count, &c, NULL, NULL, NULL);
    printf("stream: %d, %d pieces\n", (int)code, c.pieces);
    ok &= code == GGTTS_OK && c.pieces > 0;

    /* Refused before anything is rendered, each GGTTS_ERR_ARG, the message naming the option. */
    const char *refused[] = {
        "{\"render\": {\"no_such_option\": 1}}",                 /* not an option */
        "{\"load\": {\"threads\": 2}}",                         /* a load option, in a render document */
        "{\"render\": {\"steer\": \"steer.safetensors\"}}",     /* the fake engine cannot steer */
        "{\"render\": ",                                        /* not JSON */
    };
    for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
        code = ggtts_render_opts(tts, text, refused[i], NULL, NULL, &pcm, &len, &rate);
        printf("refused: %d: %s\n", (int)code, ggtts_last_error());
        ok &= code == GGTTS_ERR_ARG;
    }
    ggtts_free(tts);
    return ok ? 0 : 1;
}
```

## Errors

Every status is one of these. `ggtts_last_error()` says which case it is. A refused load writes no handle.

| Code | Value | What happened | What to do |
|---|---|---|---|
| `GGTTS_OK` | 0 | Success. | Nothing. |
| `GGTTS_ERR_ARG` | -1 | An argument is wrong: a NULL or non-UTF-8 string, a NULL output pointer, empty or punctuation-only text, a missing `model`, `codec` or `voice` the engine needs, a speaker or instruction for an engine that takes none, an unknown device, `allow_download` other than 0 or 1, or a `settings.size` this library does not know. | Fix the argument the message names. An unknown size means your header is newer than the library: check `ggtts_abi_version()`. |
| `GGTTS_ERR_HANDLE` | -2 | No engine has this handle: it is 0, was never loaded, was freed, or was dropped after a panic. | Load an engine again. Do not use a handle after `ggtts_free`. |
| `GGTTS_ERR_ENGINE` | -3 | The engine refused: an unknown engine name, a missing file or one that is not a GGUF, an unknown speaker, an instruction the checkpoint takes none of, a default model file missing with downloading off, a failed or mismatched download, `"cuda"` in a library without CUDA or on a machine without a GPU, an engine this build does not have, or a failed render. | Read the message. For a missing default file, set `allow_download = 1`, put the file in the cache, or pass paths. For an unknown speaker, pick one the message lists. For `"cuda"`, use `"cpu"` or a CUDA build. |
| `GGTTS_ERR_PANIC` | -4 | A bug inside ggtts, caught at the boundary. The engine is dropped, so its handle now gives `GGTTS_ERR_HANDLE`. | Load again. Please report it as an issue, with the message. |
| `GGTTS_ERR_STOPPED` | -5 | Your stream callback returned non-zero. | Nothing: you asked for it. |
| `GGTTS_ERR_BUSY` | -6 | The engine is already rendering, on another thread or in its own stream callback. | Wait for that render, or load a second engine. |

Only unwinding panics are caught. An abort, a stack overflow or a crash in C code still ends the process.

## Threads

* **One render at a time per engine.** A second call on an engine that is rendering returns `GGTTS_ERR_BUSY` at
  once. It never waits and never deadlocks, including a call from the engine's own stream callback.
* **Different engines render at the same time**, on different threads. A second handle is a second loaded model, with
  its own memory.
* **`ggtts_last_error()` is per thread.** Read it on the thread that got the error, before that thread's next call.
* **`ggtts_free` never waits.** Called during a render (from another thread, or from the render's own callback), it
  returns at once. The render runs to its end, and the engine is dropped when it returns.

## Lifetime

| You get | Valid until | Free it with |
|---|---|---|
| A handle from `ggtts_load_with` | `ggtts_free(handle)`. Ids are never reused. | `ggtts_free` |
| `pcm` from `ggtts_render` | You free it, even after `ggtts_free(handle)`. | `ggtts_pcm_free(pcm, len)`, once |
| `samples` in a stream callback | The callback returns. | Nothing: the library frees it. |
| The string from `ggtts_last_error()` | The calling thread's next ggtts call. | Nothing. |

Strings and the settings struct you pass are read only during the call; you may free them after it returns.

## Versions

`GGTTS_ABI_VERSION` in the header and `ggtts_abi_version()` in the library go up by one whenever a function, a
struct, a struct field or a code is added. A function never changes its signature or behaviour.

| Version | Added |
|---|---|
| 1 | `ggtts_load`, `ggtts_render`, `ggtts_render_stream`, `ggtts_pcm_free`, `ggtts_free`, `ggtts_last_error` |
| 2 | `ggtts_load_with`, `ggtts_settings` (speaker, instruction), `ggtts_abi_version` |
| 3 | `ggtts_settings.device`, `.cache_dir` and `.allow_download`: the device and the default model |
| 4 | `ggtts_load_opts` and `ggtts_render_opts`: [options documents](#options-documents). `ggtts_settings` is frozen at version 3. |

A program built against the version-3 header checks `ggtts_abi_version() >= 3` before passing the longer struct. A
version-2 library refuses it with `GGTTS_ERR_ARG`, so the check gives a clearer message, not safety. A version-3
library loads a version-2 struct exactly as before.

## What it does not do

* Per-render speakers: the speaker is fixed when the engine loads. Load another engine to change it. (An instruction
  can change per render, in a render document.)
* Write audio files. You get PCM; write it as you like (the example writes WAV).
* Report download progress. `ggtts-render` and Python do.
* The JNI entry points the Kotlin binding needs: the archive's library exports the C ABI only. See
  [Kotlin and Android](kotlin-android.md).
