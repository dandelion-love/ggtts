# Render timing

`render-timing` reads a streaming story trace and reports first-audio latency, total render time, real-time factor,
and total generation, decoding and judging time across all draws. Time to first audio is measured from immediately
after model loading to the first stream write after levelling; it is not the earlier chunk hand-off event. Render
time uses the same origin and includes streaming and levelling, but excludes story loading and planning (which run
before model loading), model loading itself, FLAC writing and instrument execution. Stage totals include redraws and are measured separately; trace overhead and
millisecond rounding mean they need not add up to total render time.

The normal configuration requires a complete v2 trace with all stage timings, first-write timing and a positive
audio duration. A missing timing produces `skipped`, so requiring the instrument makes the measurement incomplete.
It cannot calculate time to first audio when no stream was requested. A v1 trace remains readable but is not a
timing proof.

For a test that only needs the stage timings, set `arg.stage_only=true`. In that explicitly selected mode, first-audio
latency, render time and RTF are null and the report cannot support claims about them. The no-change promotion proof
uses that mode because it renders in-process without a CLI stream. It still runs the instrument against the actual
stage events and verifies that the audio remains unchanged.

Run `python -m unittest test_instrument` in this directory for the no-op and missing-input controls.
