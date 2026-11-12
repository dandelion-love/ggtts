# Log

## 2026-11-12

- Pattern 1 is done: ggtts is a system TTS engine (#221). Integration is rated 3 (from 2), and the landscape row is
  updated.

- Added the low-memory decision (warn on the first Speak, load on the second) and the streaming measurements on the
  phone. Pattern 5 is now done: the screen states each engine's costs (#223).

- Bundle started for the Android track. It holds:
  - five decisions: the integration branch, 16 KB alignment, user-supplied models with checked imports, streaming
    playback, and measuring through the CLI;
  - the on-device parity and budget finding, the landscape of other Android TTS with Play ratings read that day, our
    limitations with a stated rating, and the thread measurement;
  - the UX patterns worth copying, with issues #221, #222 and #223 filed for the ones not yet done.
