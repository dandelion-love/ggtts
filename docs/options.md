# Options

Every option a ggtts front end takes is defined once, in `ggtts_core::options`: a value (`Options`) with a version and
five sections, and a table (`options::TABLE`) with one row per option. The schema,
[`schemas/ggtts-options-1.schema.json`](../schemas/ggtts-options-1.schema.json), is generated from the table, and a test
fails when the committed file differs (`GGTTS_BLESS=1` rewrites it). This is the first step of the
[one render surface](architecture-direction.md). `ggtts-render` reads its command line and its environment through it
([#289](https://github.com/dandelion-love/ggtts/issues/289)): `Options::from_args` over `Options::from_env`, a flag
winning over its variable. Its output, its `--help` and the words of its refusals did not change.

## The document

```json
{
  "version": 1,
  "load":    {"engine": "qwen3-tts", "device": "cuda", "threads": 8},
  "render":  {"text": "一句話。", "seed_salt": "take-2", "instruction": "Speak slowly and softly."},
  "observe": {"trace": "read.trace.jsonl", "trace_level": 2},
  "output":  {"out": "read.flac", "stream": "-", "stream_raw": true},
  "book":    {"seed_root": "book-1"}
}
```

An option left out is unset; what unset means is in its row's help (for example `device` unset is `cpu`, `redraw`
unset is the engine profile's). `Options::to_json` writes only what is set, and `from_json(to_json(x)) == x`.
`fixtures/options/` holds a document per section.

| Section | Phase | What it holds |
|---|---|---|
| `load` | load | what opens an engine: the engine, model, codec and voice files, the speaker, device, threads, offline and the model cache. `ggtts_load::Request` is this section. |
| `render` | render | what one render is asked to do: the text or story, the seed keys, the instruction, steering, redraws and edge trim. |
| `observe` | render | what watches it: the trace and its level, the timing and duration reports, logging, instruments and their report. |
| `output` | render | where the audio goes: the FLAC, the stream and its form. |
| `book` | render | what only a book render takes: the seed root, and whether to include skippable parts. |

Each row also carries every spelling of the option: its flags, its environment variable, and its name in the Python
binding, the C ABI and the Kotlin binding. A test reads each front end's source (`ggtts-render`'s flags,
`ggtts.load` and the render methods, `ggtts_settings` and `ggtts_load_with`, `Ggtts.load`) and fails on a spelling no
row has.

## Phases

A **load** option is fixed when the engine is opened; a **render** option is chosen per render. A render document
(`Options::from_render_json`: the options of one render on an engine already loaded) refuses a load option, for
example `load.device`, by name.

The style instruction is a render option. The bindings take it at load today, as the default for every render, and
the engine checks it there; moving it is a later step, and this table only records its phase.

## Seed keys

Three keys, three meanings; no key means two things.

| Key | What it is | Spelled |
|---|---|---|
| `render.seed` | the sampler seed of a one-chunk plan only | `ggtts-render --text --seed` |
| `render.seed_salt` | the story seed derivation's salt, appended to every chunk's seed key | the `seed` argument of every binding's render |
| `book.seed_root` | a book's root, from which every chunk's seed is derived | `ggtts-book --seed-root` |

## Refusals

`from_args`, `from_json`, `from_render_json` and `from_env` refuse, with a `Refusal { key, message }` whose `key` is the
schema name (`section.name`; an unknown flag by its spelling; `$` for a document that is not a JSON object):

* an unknown section, key or flag;
* a value of the wrong type or out of range (`"threads": "four"`, `"trace_level": 3`, a `--steer-scale` of `NaN`);
* an option given without the one it needs (`stream_raw` without `stream`, `trace_level` without `trace`,
  `steer_scale` or `steer_layers` without `steer`);
* a `version` other than 1;
* a load option in a render document.

## The environment rule

An environment variable is a spelling of an option, like a flag: `GGTTS_LOG_LEVEL` (`observe.log_level`),
`GGTTS_LOG_FORMAT` (`observe.log_format`) and `GGTTS_CACHE_DIR` (`load.cache_dir`) are columns of the table, read by
`Options::from_env`.

Every value the parsers set is marked with where it came from: a flag, a document or the environment.
`Options::layer` lays one value on another, so `from_env(..)?.layer(from_args(..)?)` is the environment with every
flag winning over it.

For an option the engine does not have (`Options::apply_engine_rule`):

* a value given **as a flag or in a document** is a request for this render, so it is refused;
* a value that comes **only from the environment** is ambient (exported in a shell, or set for another engine), so it
  is ignored, one warning line names the variable, and the record lists it under `ignored`.

Which options an engine has is its capability statement, a later step; until then the caller says.

## RenderOptions and the record

`Options::render_options(&profile)` fills the pipeline's `RenderOptions` from the render section: `redraw`,
`seed_salt`, `edge_s`, `measure_timings` (also on when a timing report is asked for), `instruction` and `steer`, with
the profile's defaults where the section leaves them unset. With nothing set it is `RenderOptions::for_profile`. A
steering file is read first, with `Options::read_steering`.

`options::record(&options, &engine_info)` is the provenance record: the engine, backend and parity; the instruction
(when not empty) and the steering record (`sha256`, `layers`, `scale`), as `ggtts-render`'s trace header and reports
write them; the seed keys given; the speaker given; the device; the trace level (0 when untraced); and the
environment values the engine ignored. `options::reported(&record)` is the part every `ggtts-render` report (timing,
duration, each instrument's inputs) and its trace header carry (`options::REPORTED`: the instruction and the steering),
so they are written from the record and cannot disagree with it.
