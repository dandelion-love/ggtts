# Architecture direction: one render surface

Status: adopted by the owner on 2026-11-12, ahead of the work that implements it
([#263](https://github.com/dandelion-love/ggtts/issues/263)). It applies to every epic and story from now on, not
only to that epic.

## The rule

A capability is defined once, as data with a schema, on one shared surface with one entry point. Front ends pass that
value through; they do not each wire the capability by hand. A conformance test lists every capability and fails when
a front end neither carries it nor explicitly refuses it.

Why: until now each capability was added to each front end separately, so each front end carries a different subset.
`ggtts-book` has no style instruction, no steering, no trace and no instruments; the bindings have no steering and no
trace; and each binding grew its own positional parameters and struct versions for the ones it does have.

## Capabilities and front ends

A **capability** is anything a caller chooses that changes, observes or delivers a render:

* **load settings**: the engine, the model files, default-model download and offline mode, a reference voice, a
  named speaker, the device, the thread count;
* **render options**: the seed and seed salt, the style instruction, steering, redraws, the chunk ceiling, edge
  trimming, sampler settings;
* **observers**: the trace and its level, tensor hooks, instruments, timing and duration reports, logging;
* **output forms**: whole read or stream, levelled or raw, the container (FLAC, WAV, s16le, PCM).

A **front end** is anything a person or a program renders through:

| Front end | Where | Render path today |
|---|---|---|
| `ggtts-render --story` | `crates/ggtts-cli` | `render_story_streamed` |
| `ggtts-render --text` | `crates/ggtts-cli` | its own `engine.generate` and `decode`, no trim, no level |
| `ggtts-book render` | `crates/ggtts-book`, on `track/epub` | its own `render_chunk`: `engine.generate` at draw 0, no judge |
| C ABI (`libggtts_ffi`, `ggtts.h`) | `crates/ggtts-ffi` | `render_story`, `render_story_streamed` |
| Kotlin (`ggtts.Ggtts`, over JNI) | `android/ggtts`, `crates/ggtts-ffi/src/jni.rs` | the C ABI |
| Python (`import ggtts`) | `crates/ggtts-py` | `render_story`, `render_story_streamed` (its own copy of the C ABI's glue) |
| Android demo and system TTS engine | `android/demo`, on `track/android` | Kotlin |

`ggtts-kitten` (an engine tool with its own generate loop) and `ggtts-measure` are engine and measurement tools, not
front ends; they stay outside this rule unless they start rendering for users.

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
| style instruction | Y | Y | N: hard-coded `None` ([#262](https://github.com/dandelion-love/ggtts/issues/262)) | Y, fixed at load (v2) | Y, at load | Y, at load | N |
| steering | Y | Y | N: hard-coded `None` | N | N | N | N |
| redraws, chunk ceiling, sampler | N (profile only) | N | N | N | N | N | N |
| edge trim | Y | N ([#180](https://github.com/dandelion-love/ggtts/issues/180)) | Y | Y | Y | Y | Y |
| trace and level | Y | Y (engine step file) | N | N | N | N | N |
| tensor hooks | N | N | N | N | N | N | N |
| instruments, report | Y | Y | N | N | N | N | N |
| timing and duration reports, log settings | Y | Y | N | N | N | N | ? |
| stream | Y: levelled by default, `--stream-raw` | Y | N | Y: unlevelled pieces only | as C | as C | as C |
| output | FLAC, WAV or s16le stream | same | FLAC per chapter, export | PCM | PCM | PCM, FLAC | played, saved |

Two things the table shows beyond missing cells. The same word means different things: `seed` is a sampler seed on
`--text`, ignored on `--story`, a seed salt in the bindings and a root in the book. And the same capability has
different defaults: `ggtts-render --stream` is levelled unless asked for raw, while every binding streams the pieces
before the level step.

**The render entry points.** `ggtts-core` exports four: `render_story`, `render_story_traced`, `render_story_hooked`
and `render_story_streamed` (`crates/ggtts-core/src/pipeline.rs`). They already share one private loop, `render`, but
each public one fixes which observers and which output it takes, so a caller cannot have hooks and a stream together.
The paths that actually diverge are outside them: `ggtts-render --text` and `ggtts-book`'s `render_chunk` each call
the engine directly, and so lose whatever the story path gains.

## What every epic and story states

An epic or story that adds or exposes a capability says, in its acceptance criteria:

1. **Where it enters the shared options value**: its name, type, default and section in the schema. Until the value
   exists ([#263](https://github.com/dandelion-love/ggtts/issues/263), its first story), the story says which 263
   story it waits for, or that it is engine-internal.
2. **How each front end gets it or refuses it**: one line per front end in the table above. "Carried through the
   options value" or "refused: reason" are the only answers; "not yet" is a gap and is listed as one.
3. **The conformance test that would catch a front end without it**: the case in the conformance test, and a planted
   front end that drops the option and turns it red.
4. **No per-front-end wiring.** A flag, a binding parameter or a struct field that exposes a capability on one front
   end only is a review finding, whoever wrote it. The fix is to put it on the shared surface.

An engine that cannot do something states it in its capability statement and is refused the same way everywhere,
before anything is rendered.

## What is allowed meanwhile

Until #263's first stories land, these go ahead as before:

* bug fixes and review fixes;
* engine-internal work, including performance work, that adds no option;
* measurements, instruments that need no new option, and docs;
* front-end work that adds no capability wiring: build, packaging, lifecycle, UI.

What is held: a new flag, binding parameter or struct field that exposes a capability on one front end only. Such an
issue is reshaped to deliver through the shared surface and names the #263 story it waits for.

## Undoing this

Delete this file and the "Before you plan a feature" paragraph in `CONTRIBUTING.md`. Nothing in the code depends on
it. Comments that held or reshaped issues against #263 name this document; they can be answered by removing the hold.
