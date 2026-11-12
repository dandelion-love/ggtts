---
type: Decision
title: Measure on the device through ggtts-render
description: On-device parity and the resource budget are measured with the command-line renderer run over adb, not through the app or the binding.
tags: [android, measurement]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:30:00-07:00 }
---
# Decision

The on-device measurements run `ggtts-render` from `adb shell`, with the same flags as on desktop. This covers the
token and sample parity, peak memory (`VmHWM`), the timing report, and thermals sampled alongside. The scripts are
in `scripts/android_on_device/`, and `scripts/android_parity.py` compares the results.

# Why

The CLI gives the same trace (`--trace-level 2`), raw stream (`--stream-raw`) and timing report on both machines,
so a phone run and a desktop run compare field for field. The app has no trace output, and its memory sits under
an app process the shell cannot read.

# Limits

- The binding's story path salts its seeds with the render's seed; the CLI's story path ignores `--seed`
  ([#215](https://github.com/dandelion-love/ggtts/issues/215)). So the app's draws differ from the CLI's for the
  same text. Their cost should be the same, but that is not measured.
- The app adds a JVM and an AudioTrack on top of the CLI's figures.

# Undo

Nothing depends on this choice beyond the record. A later measurement can run through the app.
