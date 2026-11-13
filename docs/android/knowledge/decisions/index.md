# Decisions

* [Android work lands on one integration branch](/decisions/integration-branch.md) - every Android change goes into `track/android` by reviewed PR, and reaches `rc` as one PR
* [Native libraries are aligned for 16 KB pages](/decisions/16kb-page-alignment.md) - link with max-page-size 16384 and refuse a library aligned below it
* [No model ships; imports are checked](/decisions/models-are-brought-not-bundled.md) - the user supplies model files; the app checks type, size and published hashes on the way in
* [The demo streams audio chunk by chunk](/decisions/stream-playback.md) - play each chunk as it is ready, with Stop, instead of rendering the whole read first
* [Measure on the device through ggtts-render](/decisions/measure-through-the-cli.md) - parity and budget runs use the command-line renderer over adb, not the app
* [Warn, do not refuse, on low memory](/decisions/warn-on-low-memory.md) - below the measured peak, the first Speak warns and the second loads
