# Kitten-TTS 2 step profile and restricted head, desktop CPU (2026-11-12)

Raw outputs of the examples in `crates/ggtts-kitten/examples/`, at 4 threads:

- `desktop.json`: `matmul_bench`;
- `op-desktop-q4.json`, `op-desktop-tq1.json`: `op_profile` on three lines;
- `head-rows-q4.json`, `head-rows-tq1.json`, `head-rows-full.log`: `head_rows_check` with the full head (22 lines);
- `head-rows-restricted-*.json` and `.log`: the same with the restricted head;
- `headab-*`: the interleaved timing (full, restricted, restricted, full; four lines per model).

The finding built on them is `docs/android/knowledge/research/kitten-step-profile.md`.
