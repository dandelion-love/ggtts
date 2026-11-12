# Log

## 2026-11-12

- Kitten-TTS 2 on the phone with the released Cantonese TQ1_0 LM against the base Q4_0: 195 MB lower peak, 24% and 35%
  slower generation in two pairs. The 3.78 GB peak is two load-time spikes (whole-file reads and per-tensor copies), about 0.9 GB
  above what a render holds. Peak-memory figures corrected for a KiB slip (3.69 to 3.78 GB, 2.93 to 3.00 GB), here and
  in the demo's cost line. The landscape's unsourced "default on many phones" is removed.

- Review corrections. The benchmark's 440 ms and Kokoro's 8.2 s and 15.3 s are whole-utterance synthesis times,
  not time to first audio, and its "Android System TTS" names no vendor; the rows say so. Work still in open PRs
  (233, 234, 235, 237, 240) is marked by PR, not Done, and Integration is 2 until PR 235 lands. Each first-audio
  figure now cites one run. RHVoice's quote and what its listing names are corrected.

- Off-screen reads: Android confines the app to three little cores even under a foreground service, and four
  threads on three cores stall (#236 filed for the engine fix). Pattern 3 built (#222, PR 237).

- Pattern 1 built: ggtts as a system TTS engine (#221, PR 235). Integration was rated 3 (from 2) here; corrected
  above to 3 only once PR 235 lands.

- Added the low-memory decision (warn on the first Speak, load on the second) and the streaming measurements on the
  phone. Pattern 5 built: the screen states each engine's costs (#223, PR 234).

- Bundle started for the Android track. It holds:
  - five decisions: the integration branch, 16 KB alignment, user-supplied models with checked imports, streaming
    playback, and measuring through the CLI;
  - the on-device parity and budget finding, the landscape of other Android TTS with Play ratings read that day, our
    limitations with a stated rating, and the thread measurement;
  - the UX patterns worth copying, with issues #221, #222 and #223 filed for the ones not yet done.
