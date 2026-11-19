# Windows visible-controls integration driver

`visible_controls_test.dart` uses Flutter's integration-test binding to tap the
actual read and saved-read widgets. It uses real app services and writes only to
an explicitly supplied isolated saved-read library. It does not use desktop-wide
input automation. Run from the Flutter app directory on Windows:

```powershell
flutter pub get
flutter test integration_test/visible_controls_test.dart -d windows --no-pub
```

Provide the following environment variables before launching the test:

- `VISIBLE_RUN_DIR`: an empty, unique directory for test events and app-only PNGs.
- `VISIBLE_LIBRARY`: an isolated saved-read library directory.
- `VISIBLE_OPTIONS`: a settings JSON file loaded by the app's settings store.
- `VISIBLE_MODE`: `read`, `replay`, or `replay-only`.
- `VISIBLE_READ_ID_FILE`: the file to write on `read`, or read on replay.
- `VISIBLE_REPLAY_SECONDS`: duration of the saved FLAC for replay modes.
- `GGTTS_LIB` and `GGTTS_CACHE_DIR`: the intended engine and model cache paths.

The driver writes `process-pid.txt` at startup. Immediately before speaking or
replaying, it writes `app-pid.txt` and waits for a `capture-ready` file. An
external process-only audio recorder may attach to that PID and create the
marker after it is ready. The driver then taps the visible control and records
timestamped events, screenshots and the read ID. Supply a saved ID and library
for `replay-only`; a deliberately unavailable engine library proves that opening
Saved reads and playing the file do not require model preparation. In `read`
mode the test prepares offline, enters the fixed passage, speaks and saves.
Launch a fresh test process in `replay` mode to exercise the saved-read controls.

CUDA renders require the host's GPU reservation procedure. A process-output
capture can establish audio leaving this process; it cannot establish physical
speaker audibility or that anyone understood the final words. Test-created PNGs,
read text and audio can contain private material and are not repository fixtures.
