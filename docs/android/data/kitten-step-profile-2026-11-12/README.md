# Kitten-TTS 2 step profile and restricted head, desktop CPU (2026-11-12)

Raw outputs of the examples in `crates/ggtts-kitten/examples/`, at 4 threads:

- `desktop.json`: `matmul_bench`;
- `op-desktop-q4.json`, `op-desktop-tq1.json`: `op_profile` on three lines;
- `head-rows-q4.json`, `head-rows-tq1.json`, `head-rows-full.log`: `head_rows_check` with the full head (22 lines);
- `head-rows-restricted-*.json` and `.log`: the same with the restricted head;
- `headab-*`: the interleaved timing (full, restricted, restricted, full; four lines per model).

- `margin-*`: `head_rows_check` measuring the top-k margin against the audio and end ids only (the first range; it
  found the reference end marker, 158,238, inside the top-50 at 14 steps);
- `rows-full-*`, `rows-restricted-*`: the same against the rows the head now computes (6,571), with the whole head and
  with the restricted head (whose margins are not meaningful, so they read null);
- `phone-hd-*`: on the Galaxy Z Fold5, the released TQ1_0 LM with the whole head and the restricted head, interleaved
  (`scripts/android_on_device/head-measure.sh`);
- `phone-op-*`: `op_profile` on the phone (dominated by the observer's per-node cost there; see the finding);
- `phone-mb-*`: `matmul_bench` on the phone, the default and the armv8.6-a ggml builds, base, armv86, armv86, base.

The finding built on them is `docs/android/knowledge/research/kitten-step-profile.md`.
