# Kitten-TTS 2 step profile and restricted head, desktop CPU (2026-11-12)

Raw outputs of the examples in `crates/ggtts-kitten/examples/`, at 4 threads:

- `desktop.json`: `matmul_bench`;
- `op-desktop-q4.json`, `op-desktop-tq1.json`: `op_profile` on three lines;
- `head-rows-q4.json`, `head-rows-tq1.json`, `head-rows-full.log`: `head_rows_check` with the full head (22 lines);
- `head-rows-restricted-*.json` and `.log`: the same with the restricted head;
- `headab-*`: the interleaved timing (full, restricted, restricted, full; four lines per model).

- `v3-narrow-*`: `head_rows_check --head full --rows audio-and-ends`, the top-k margin against the audio ids and the
  two end ids only (the first range tried), with a line for every step where another id reached the top-50;
- `v3-rows-*`: `head_rows_check --head full`, the margin against the rows the head computes (6,571);
- `v3-restricted-*`: `head_rows_check --head restricted`, the restricted head's tokens (its margins are not meaningful
  and read null);
- `phone-head-measure.stdout`: the phone run's stdout, with the binaries' hashes and each run's raw-stream hash;
- the `v3-*` files are from `crates/ggtts-kitten/examples/head_rows_check.rs` as committed (f187749 and after);
  earlier versions' outputs were removed;
- `phone-hd-*`: on the Galaxy Z Fold5, the released TQ1_0 LM with the whole head and the restricted head, interleaved
  (`scripts/android_on_device/head-measure.sh`);
- `phone-op-*`: `op_profile` on the phone (dominated by the observer's per-node cost there; see the finding);
- `phone-mb-*`: `matmul_bench` on the phone, the default and the armv8.6-a ggml builds, base, armv86, armv86, base.

The finding built on them is `docs/android/knowledge/research/kitten-step-profile.md`.
