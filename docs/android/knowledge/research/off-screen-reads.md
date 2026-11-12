---
type: Research Finding
title: Off-screen reads run on the little cores, and stall with too many threads
description: Off screen, Android confines the app to three little cores even under a foreground service; with more ggml threads than allowed cores, rendering stalls.
tags: [android, performance, scheduling]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T10:24:00-07:00 }
---
# What was seen

On 2026-11-12, on the Galaxy Z Fold5 (Android 16), the demo started a Qwen3-TTS read under its foreground media
service, then its screen was closed (Back). The process kept running (about 320% CPU), but had no first chunk after
15 minutes; on screen the first chunk comes at about 50 s.

- **Where it ran:** `/proc/PID/cpuset` read `/moderate`. On this phone `top-app` may use CPUs 0-7, `foreground`
  0-6, and `background` 0-2. All four render threads ran on CPUs 0 to 2, the three Cortex-A510 little cores. The
  foreground service kept the process alive, but not on the fast cores.
- **Why it stalled:** four threads on three cores. ggml's threads wait for each other at every step, so the
  thread the kernel has parked holds up the other three.

# Measured with the CLI

One short Qwen3-TTS line, `ggtts-render` at four threads unless stated, pinned with `taskset`:

| CPUs | Threads | Result |
|---|---|---|
| all | 4 | RTF 4.6 (30 s) |
| 0-2 (little) | 3 | RTF 15.6 (87 s) |
| 0-2 (little) | 2 | RTF 19.3 (103 s) |
| 0-2 (little) | 4 | not done after 400 s, killed |

# With the engine fix

With the threads capped at the allowed CPUs before each compute (PR 240, issue 236), the same line pinned to CPUs
0-2 with four threads configured finished in 86 s (RTF 14.4), at the three-thread speed, instead of stalling.
Unpinned, it ran at RTF 4.3 (4.5 before). The audio was byte-identical in every run, capped or not.

# Consequences

- **Engine fix**, in PR 240 (for [#236](https://github.com/dandelion-love/ggtts/issues/236)): never run more threads than the CPUs
  the process may use, re-checked at each compute (`std::thread::available_parallelism`, which follows the
  affinity mask and so the cpuset). Off-screen reads then run at the little cores' speed, about 3 to 4 times
  slower, instead of stalling.
- **App decision:** keep the screen on while a read runs and the app is open, and say in the reading notification
  that off-screen reads are slower (PR 237, for [#222](https://github.com/dandelion-love/ggtts/issues/222)).
- **Limitation:** an app cannot choose its cpuset. A long read off screen will be slow on this phone, whatever the
  app does.
