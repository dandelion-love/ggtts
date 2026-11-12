---
type: Decision
title: Warn, do not refuse, on low memory
description: Below an engine's measured peak memory, the demo warns on the first press of Speak and loads on the second, because the peak counts model files Android can reclaim.
tags: [android, ux, memory]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T11:45:00-07:00 }
---
# Decision

Before a load, the demo compares the available memory (`ActivityManager.MemoryInfo.availMem`, and its `lowMemory`
flag) with the engine's measured peak: 3.00 GB for Qwen3-TTS, 3.78 GB for Kitten-TTS 2 (corrected 2026-11-12 from 2.93 and 3.70, a KiB slip). Below that, or when the
system reports low memory:
- the first press of Speak shows a warning and does not load;
- a second press with the same settings loads anyway;
- an engine already loaded is not checked again.

The screen also says each engine's size, memory and wait before anyone presses Speak (issue
[#223](https://github.com/dandelion-love/ggtts/issues/223)).

# Why a warning and not a refusal

The measured peak is the process's resident set, and that includes the memory-mapped model files, which Android can
drop and read back. Kitten-TTS 2 rendered on the phone with 3.2 GB "available" against its 3.7 GB peak. A refusal
at the peak would have blocked a read that works. A warning tells the user the risk (the system may stop the app)
and leaves the choice to them.

# Undo

Make the check refuse instead of warn: one condition in `DemoController.speak`, and `EngineCosts.memoryWarning`'s
wording.
