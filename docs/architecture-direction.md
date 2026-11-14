# Architecture direction: one render surface

Status: adopted by the owner on 2026-11-12, ahead of the work that implements it
([#263](https://github.com/dandelion-love/ggtts/issues/263)). It applies to every epic and story from now on, not
only to that epic.

## The rule

A capability is defined once, as data with a schema, on one shared surface. Front ends pass that value through; they
do not each wire the capability by hand. A conformance test lists every capability and fails when a front end neither
carries it nor explicitly refuses it.

The shared surface has two public pieces in `ggtts-core::pipeline`:

* **`render_plan`**, the entry point: renders a whole plan (draws, judge, trim, join, level) with the options value,
  observers and an output mode, in any combination;
* **`render_chunk`**, the step `render_plan` is built from: one planned chunk, its draws, judge, redraw and trim, for
  a caller that keeps its own plan, seeds, storage and join (a book).

**Checkable form.** Nothing outside `ggtts-core::pipeline`, the engine crates' libraries and tests calls
`Engine::generate`, `generate_traced`, `generate_hooked` or `decode` to render for a user. An engine that wraps another
engine and delegates to it (inside an `impl Engine` block) is not a render. A grep test in the conformance test
([#275](https://github.com/dandelion-love/ggtts/issues/275)) enforces it, with every exception named in the test. Before
[#278](https://github.com/dandelion-love/ggtts/issues/278) and #279 it would have flagged `ggtts-render --text` and
`ggtts-book`'s `render_chunk`; with them it flags `ggtts-kitten render` and `ggtts_core::trace::replay`.

Why: until now each capability was added to each front end separately, so each front end carries a different subset.
`ggtts-book` has no style instruction, no steering, no trace and no instruments; the bindings have no steering and no
trace; and each binding grew its own positional parameters and struct versions for the ones it does have.

## Capabilities and front ends

A **capability** is anything a caller chooses that changes, observes or delivers a render. Each is tagged with a
phase: **load** (fixed when the engine is opened) or **render** (per render).

* **load settings**: the engine, the model files, default-model download and offline mode, a reference voice, a
  named speaker, the device, the thread count;
* **render options**: the seed and seed salt, the style instruction, steering, redraws, the chunk ceiling, edge
  trimming, sampler settings;
* **observers**: the trace and its level, tensor hooks, instruments, timing and duration reports, logging;
* **output forms**: whole read or stream, levelled or raw, the container (FLAC, WAV, s16le, PCM).

A **front end** is anything a person or a program renders through:

| Front end | Where | Render path today |
|---|---|---|
| `ggtts-render --story` | `crates/ggtts-cli` | `render_plan`, with its stream mode as the output |
| `ggtts-render --text` | `crates/ggtts-cli` | `render_plan` on a one-chunk plan at `Given(seed)`, with no redraw, no trim and no level ([#278](https://github.com/dandelion-love/ggtts/issues/278)) |
| `ggtts-book render` | `crates/ggtts-book` | the core `render_chunk` per chunk at draw 0, its flags read by the shared options parser ([#279](https://github.com/dandelion-love/ggtts/issues/279)) |
| C ABI (`libggtts_ffi`, `ggtts.h`) | `crates/ggtts-ffi` | `render_story`, `render_story_streamed` |
| Kotlin (`ggtts.Ggtts`, over JNI) | `android/ggtts`, `crates/ggtts-ffi/src/jni.rs` | the C ABI |
| Python (`import ggtts`) | `crates/ggtts-py` | `render_story`, `render_story_streamed`, through the engine glue it shares with the C ABI (`ggtts_load::binding`, [#281](https://github.com/dandelion-love/ggtts/issues/281)) |
| Android demo and system TTS engine | `android/demo`, on `track/android` | Kotlin |
| `ggtts-kitten render` | `crates/ggtts-kitten`, shipped in the release archive | its own loop: one seed per line (`--seed` plus the line index), its own `--max-tokens`, Kitten only |

`ggtts-kitten render` is an engine parity tool that happens to ship. It is listed so its exemption is checkable: the
conformance table gives it a row with every capability `Refused` ("an engine parity tool; render with `ggtts-render
--engine kitten-tts-2`"), and the grep test names its file as an exception. `ggtts-measure` renders nothing.
`ggtts_core::trace::replay` and `replay_steered` regenerate recorded chunks through `render_chunk`
([#276](https://github.com/dandelion-love/ggtts/issues/276)), so they need no exemption.

## The state today

Read from the code on `rc` at 554f89e, `track/epub` at 7dd4837 and `track/android` at 1b7acba. Y carries it, N does not,
fixed means the front end hard-codes one value, and `?` is not confirmed from the code.

| Capability | render `--story` | render `--text` | book | C ABI | Kotlin | Python | Android app |
|---|---|---|---|---|---|---|---|
| engine, model and codec paths | Y | Y | Y | Y | Y | Y | Y |
| default model download, offline | Y | Y | Y | Y (settings v3) | Y | Y | fixed: never downloads |
| reference voice (Kitten) | Y | Y | Y | Y | Y | Y | Y |
| speaker | Y | Y | Y | Y (settings v2) | Y | Y | Y |
| device | Y | Y | Y | Y (settings v3) | N (documented: CPU only) | Y | N |
| threads | Y | Y | Y | Y | Y | Y | Y |
| seed | N: ignored ([#215](https://github.com/dandelion-love/ggtts/issues/215)) | Y: the sampler seed | own: `--seed-root` | Y: as the story's seed salt | as C | as C | as C |
| style instruction | Y | Y | Y (since [#279](https://github.com/dandelion-love/ggtts/issues/279)) | Y, fixed at load (v2) | Y, at load | Y, at load | N |
| steering | Y | Y | Y (since #279) | N | N | N | N |
| redraws, chunk ceiling, sampler | N (profile only) | N | N | N | N | N | N |
| edge trim | Y | N ([#180](https://github.com/dandelion-love/ggtts/issues/180)) | Y | Y | Y | Y | Y |
| trace and level | Y | qwen3-tts only, the engine's step file; accepted and ignored on kitten-tts-2 and fake | Y (since #279) | N | N | N | N |
| tensor hooks | N | N | N | N | N | N | N |
| instruments, report | Y | Y | N | N | N | N | N |
| timing and duration reports, log settings | Y | Y | N | N | N | N | ? |
| stream | Y: levelled by default, `--stream-raw` | one unlevelled piece; `--stream-raw` and `--stream-level-seconds` accepted and ignored | N | Y: unlevelled pieces only | as C | as C | as C |
| output | FLAC, WAV or s16le stream | same | FLAC per chapter, export | PCM | PCM | PCM, FLAC | played, saved |
| speech rate and pitch (Android `SynthesisRequest`) | n/a | n/a | n/a | n/a | n/a | n/a | ignored silently |

What the table shows beyond missing cells:

* **The same word means different things.** `seed` is a sampler seed on `--text`, ignored on `--story`, a seed salt
  in the bindings, a root in the book, and a base that each line adds its index to in `ggtts-kitten render`.
* **The same capability has different defaults.** `ggtts-render --stream` is levelled unless asked for raw; every
  binding streams the pieces before the level step. Since [#276](https://github.com/dandelion-love/ggtts/issues/276)
  the pipeline applies the levelled stream (`Output::Stream` with `StreamMode::Levelled`), so a binding can have it;
  the bindings still ask for `Raw`.
* **Options are accepted and dropped.** `--text` takes `--trace` on engines that write nothing, and the stream
  levelling flags, without effect.

**The render entry points.** Since [#276](https://github.com/dandelion-love/ggtts/issues/276), `ggtts-core` has one:
`render_plan(engine, chunks, seeds, judge, opts, observers, output)` (`crates/ggtts-core/src/pipeline.rs`), with any
observers (`Observers { events, hooks }`) and any output (`Output::Whole`, or `Output::Stream` raw or levelled). It is
the public step `render_chunk` per chunk (draws at a caller's `SeedSource`, a pluggable `Judge`, redraws, trim), then
join, output and level. `render_story`, `render_story_traced`, `render_story_hooked` and `render_story_streamed` are
wrappers over it. Before #276 they shared a private loop but each fixed its observers and output, so a caller could
not have hooks and a stream together. `ggtts-render --text` renders through `render_plan` too since
[#278](https://github.com/dandelion-love/ggtts/issues/278), with `RenderOptions::trim` and `RenderOptions::level` off,
and `ggtts-book` renders each chunk through `render_chunk` since
[#279](https://github.com/dandelion-love/ggtts/issues/279). The path that still diverges is outside them:
`ggtts-kitten render` calls the engine directly.

## What every epic and story states

An epic or story that adds or exposes a capability says, in its acceptance criteria:

1. **Where it enters the shared options value**: its name, type, default, phase (load or render) and every spelling
   (flag, environment variable, document key) in the schema's table. Until the value exists, the story says which
   #263 story it waits for, or that it is engine-internal.
2. **How each front end gets it or refuses it**: one line per front end in the table above. "Carried through the
   options value" or "refused: reason" are the only answers; "not yet" is a gap and is listed as one. The expected
   answers for the bindings: the trace is carried by the desktop bindings (C ABI, Python) and refused on Android;
   instruments are refused by every binding (they are out-of-process programs the CLI runs on a finished file).
3. **The conformance test that would catch a front end without it**: the case in the conformance test, and a planted
   front end that drops the option and turns it red.
4. **No per-front-end wiring.** A flag, a binding parameter, a struct field or an environment variable that exposes a
   capability on one front end only is a review finding, whoever wrote it. The fix is to put it on the shared surface.

An engine that cannot do something states it in its capability statement and is refused the same way everywhere,
before anything is rendered.

**A front end may expose a value of an option its binding already carries.** A screen that lets a user choose the
thread count, or a speaker, from what the binding already takes is not new wiring: it chooses a value, it does not add
a capability. Adding the option to the binding is.

## Environment variables

An environment variable that changes or observes a render is a spelling of an option, like a flag: it is a column in
the schema's table, read by the one parser, and recorded in provenance like any other option. That covers
`GGTTS_LOG_LEVEL`, `GGTTS_LOG_FORMAT` and `GGTTS_CACHE_DIR` today, which became table entries in
[#274](https://github.com/dandelion-love/ggtts/issues/274). `ggtts-render` reads all three through the table since
[#289](https://github.com/dandelion-love/ggtts/issues/289).

Until #274 lands, a new variable of this kind is allowed only with a filed issue that moves it into the table once the
table exists, linked from the pull request. The reason: the alternative, holding a diagnostic switch on an engine
until the table exists, would block engine work this direction does not mean to hold; the filed issue keeps it from
staying a silent, unrecorded switch.

Out of scope: variables that configure the machine or the build rather than a render (`CUDA_HOME`, `GGML_*` build
variables, ggml's and CUDA's own runtime variables such as `NVIDIA_TF32_OVERRIDE`), and those that locate a tool
(`GGTTS_PYTHON`, `GGTTS_INSTRUMENTS_DIR`) or serve tests.

## What is allowed meanwhile

"The first stories" are [#274](https://github.com/dandelion-love/ggtts/issues/274) (the options value) and
[#275](https://github.com/dandelion-love/ggtts/issues/275) (the conformance test), both merged on `rc`. Until then
these go ahead as before:

* bug fixes and review fixes;
* engine-internal work, including performance work, that adds no option;
* measurements, instruments that need no new option, and docs;
* front-end work that adds no capability wiring: build, packaging, lifecycle, UI, and choosing values of options the
  binding already carries.

What is held: a new flag, binding parameter or struct field that exposes a capability on one front end only. Such an
issue is reshaped to deliver through the shared surface and names the #263 story it waits for. A new environment
variable follows the rule above.

## Undoing this

Delete this file and the "Before you plan a feature" paragraph in `CONTRIBUTING.md`. Nothing in the code depends on
it. Comments that held or reshaped issues against #263 name this document; they can be answered by removing the hold.
