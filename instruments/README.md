# Instruments

## Device selection

`ggtts-render --device cuda` requests CUDA from every instrument through the generic
`GGTTS_INSTRUMENT_DEVICE` child environment. CPU-only instruments may ignore the
hint and must not report GPU use. The two ASR instruments honor it, fail visibly
when CUDA is unavailable or their model is not placed on CUDA, and report the
requested and actual device. For example:

```sh
ggtts-render --device cuda --model talker.gguf --codec codec.gguf --text "Hello" --out out.flac \
  --instrument asr-sensevoice,arg.model_dir=MODEL,arg.reference_file=REF \
  --instrument asr-qwen,arg.model_dir=MODEL2,arg.reference_file=REF
```

Use `--instrument-device cpu|cuda|auto` (or top-level YAML `device:`) to override
the renderer for the whole instrument chain. Standalone `ggtts-measure` uses the
same flag and defaults to CPU. A device-aware instrument can override its own
choice with e.g. `--instrument asr-qwen,arg.device=cpu`. `auto` chooses CUDA
only if available, otherwise CPU; an explicit `cuda` never silently falls back.
Device selection does not install GPU dependencies or prove GPU execution on a host.

An instrument measures a render without changing it. This folder holds what you need to write one.

| Path | What it is |
|---|---|
| `python/ggtts_instrument/` | The Python SDK, standard library only: read the event stream, read the audio, write the envelope |
| `templates/python/<hook>/` | A minimal instrument for each event hook point, with a README and a self-test |
| `examples/python/internal-pauses/` | A worked example: silences inside the speech of a rendered read |
| `examples/python/pitch-range/` | Voiced pitch range using the lab's pinned detector; requires Node.js |
| `examples/python/loudness-ends/` | Relative loud and soft speech ends; requires the documented pYIN Python environment |

The hook points and what each one carries are in [`docs/instruments/hooks.md`](../docs/instruments/hooks.md). The
envelope every instrument writes is `ggtts-report/2` (`schemas/ggtts-report-2.schema.json`); the runner also reads
`ggtts-report/1` and normalizes it into the current merged report. Older v1-only readers cannot read v2 reports.

For existing audio, `ggtts-measure --audio-state raw|trimmed|unknown` records the caller's knowledge; the default is
`unknown`. The flag does not modify the audio or verify the claim. `ggtts-render --text` reports `raw` and its story
pipeline reports `trimmed`. Event-only instruments use `n/a`. Standalone audio instruments without a runner report
`unknown`. A v1 `n/a` remains `n/a`; old reports do not prove an audio file was trimmed.

## Write a Python instrument

1. Copy the template for the hook you need. `step` reads per-frame values; `chunk` reads draws; `trim` reads the
   edges cut; `render` reads the totals and shows how to report a truncated stream; `audio` reads the rendered sound.
2. Change `NAME`, `VERSION` and `measure()`. Give every metric a unit and a direction (`higher`, `lower`, `neutral`
   or `n/a`). Write `None` for a value you could not compute, never 0.
3. Make its self-test plant the defect your instrument claims to detect, and check it is seen. Keep the no-op: the
   same render with nothing that matters changed must read as no change.
4. Run it: `python -m unittest test_instrument` in its folder.

The runner (ggtts#30) starts each instrument after the render as
`PROGRAM [--events E.jsonl] --audio OUT.flac --envelope ENV.json --name NAME [--KEY VALUE]...`.
`ggtts_instrument.run()` parses that, records every `--KEY VALUE` as the effective configuration, and writes a
`failed` envelope if `measure()` raises. A crash is always reported, never lost.

To measure audio you already have (a kept render, a release sample, a recording) without rendering, use
`ggtts-measure --audio IN.flac [--events E.jsonl] --report DIR` with the same instrument flags. It runs the same
runner with the same config, report and exit codes, loads no engine, and hands the instruments copies, so the given
file is never written.

## Tests

`cargo test -p ggtts-core --test python_instruments` renders a recorded pipeline case on the fake engine. It hands
the trace and the FLAC to every suite here, as the runner would: the SDK's tests, and each template's and the
example's self-test. Each suite must pass, run at least the tests it is known to have, and skip none. CI runs it in
the CPU build-and-test job. It needs Python 3.8 or later; `GGTTS_SKIP_PYTHON=1` skips it on purpose.

Without the cargo harness, each self-test runs on a small synthetic render, so Python alone is enough while you
write one.

