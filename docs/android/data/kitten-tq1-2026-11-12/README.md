# Kitten-TTS 2, Q4_0 against TQ1_0, on a Galaxy Z Fold5 (2026-11-12)

Raw files from `scripts/android_on_device/tq1-measure.sh`, run in the order q4-1, tq1-1, q4-2, tq1-2. Per run:

- `.log`: `ggtts-render`'s log (model hashes, first audio, load and render times);
- `.timing.json`: the timing report (`ggtts-timing/1`);
- `.mem`: exit code and peak `VmHWM` in KiB;
- `.rss`: the memory timeline from `run-mem.sh`: uptime in s, then VmRSS, VmHWM, RssAnon and RssFile in KiB, every 0.25 s.

The finding built on them is `docs/android/knowledge/research/kitten-memory-and-tq1.md`.
