# ggtts_instrument

The Python SDK for ggtts event instruments. Standard library only, Python 3.8 or later. It is a local package for
now; it will be published on PyPI later.

| Module | Use |
|---|---|
| `events` | `read_trace(path)` reads a whole `ggtts-trace/2` or `/1` trace. `iter_events(source)` yields events as they arrive from a pipe or standard input (`-`). A stream without `end` reads as incomplete (`Trace.complete` is False), and an instrument reports it `truncated`. |
| `audio` | `read_audio(path)` returns mono float samples and the rate. It reads WAV (16-bit PCM or 32-bit float, streamed WAV included), and FLAC: any FLAC with `soundfile` installed, or without it the subset ggtts writes. |
| `envelope` | `Envelope` builds a `ggtts-report/2` envelope: identity, provenance, metrics, artifacts by hash. `check(dict)` lists every reason an envelope is invalid, and `write()` refuses to write one. |
| `cli` | `run(measure, name, version, hooks)` parses the runner's flags, calls `measure(ctx, env)`, and writes the envelope, as `failed` if `measure` raises. |
| `testing` | Self-test helpers: a render to measure, a clock-only change for the no-op, and running an instrument as the runner does. |

When an instrument uses `run()` and `read_audio()`, the private envelope's
`provenance.config.ggtts_audio_decoders` records one `{kind, sha256}` per successful
decode. Kinds are `ggtts-python-wav`, `ggtts-python-flac`, and `libsndfile`.
The built-in readers hash the SDK decoder source; native decoding resolves the
module containing soundfile's bound open/read/close symbols on Linux or Windows;
all three must resolve to the same file, which is hashed. A matching
`decoder:<kind>` entry in `provenance.inputs`
holds each known hash. If module resolution is unsupported, ambiguous, or the
file cannot be read, the hash is null and no marker is added: this means unknown,
not the built-in decoder. No filesystem path is recorded. The hash identifies
the file read at measurement time; it is not an attestation of mapped memory or
protection against replacement while the process runs. The shareable exporter
removes configuration and paths, retaining input hashes only.

Tests: `python -m unittest discover -s tests` here, or the cargo harness described in `../README.md`.
The optional real native-decoder tests require `soundfile` and `numpy` and run
separately with `python -m unittest discover -s native_tests`. The default
standard-library suite does not exercise a native decoder.

The envelope rules here restate `schemas/ggtts-report-2.schema.json`; the SDK can also read v1. If the two ever disagree, the schema wins, and
`tests/test_sdk.py` should gain the case that showed it.
