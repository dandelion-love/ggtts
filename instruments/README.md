# Instruments

An instrument measures a render without changing it. This folder holds what you need to write one.

| Path | What it is |
|---|---|
| `python/ggtts_instrument/` | The Python SDK, standard library only: read the event stream, read the audio, write the envelope |
| `templates/python/<hook>/` | A minimal instrument for each event hook point, with a README and a self-test |
| `examples/python/internal-pauses/` | A worked example: silences inside the speech of a rendered read |

The hook points and what each one carries are in [`docs/instruments/hooks.md`](../docs/instruments/hooks.md). The
envelope every instrument writes is `ggtts-report/1` (`schemas/ggtts-report-1.schema.json`).

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
