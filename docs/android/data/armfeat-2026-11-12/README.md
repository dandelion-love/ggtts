# ggml's ARM features on a Galaxy Z Fold5 (2026-11-12)

Raw files from `scripts/android_on_device/armfeat-measure.sh`, in the order base-q4-1, armv86-q4-1, armv86-q4-2,
base-q4-2, base-tq1-1, armv86-tq1-1, armv86-tq1-2, base-tq1-2. `base` is ggml's default ARM build; `armv86` is
`GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16`. Per run: `.log`, `.timing.json`, `.mem` (peak `VmHWM` in KiB) and
`.rss` (uptime s, then VmRSS, VmHWM, RssAnon, RssFile in KiB, every 0.25 s).

The finding built on them is `docs/android/knowledge/research/arm-features-on-the-phone.md`.
