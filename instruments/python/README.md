# ggtts_instrument

The Python SDK for ggtts event instruments. Standard library only, Python 3.8 or later. It is a local package for
now; it will be published on PyPI later.

| Module | Use |
|---|---|
| `events` | `read_trace(path)` reads a whole `ggtts-trace/1` trace. `iter_events(source)` yields events as they arrive from a pipe or standard input (`-`). A stream without `end` reads as incomplete (`Trace.complete` is False), and an instrument reports it `truncated`. |
| `audio` | `read_audio(path)` returns mono float samples and the rate. It reads WAV (16-bit PCM or 32-bit float, streamed WAV included), and FLAC: any FLAC with `soundfile` installed, or without it the subset ggtts writes. |
| `envelope` | `Envelope` builds a `ggtts-report/1` envelope: identity, provenance, metrics, artifacts by hash. `check(dict)` lists every reason an envelope is invalid, and `write()` refuses to write one. |
| `cli` | `run(measure, name, version, hooks)` parses the runner's flags, calls `measure(ctx, env)`, and writes the envelope, as `failed` if `measure` raises. |
| `testing` | Self-test helpers: a render to measure, a clock-only change for the no-op, and running an instrument as the runner does. |

Tests: `python -m unittest discover -s tests` here, or the cargo harness described in `../README.md`.

The envelope rules here restate `schemas/ggtts-report-1.schema.json`. If the two ever disagree, the schema wins, and
`tests/test_sdk.py` should gain the case that showed it.
