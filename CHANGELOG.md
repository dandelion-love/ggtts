# Changelog

What changes for someone using ggtts, release by release: new options, changed file formats, changed output, and
new requirements. Internal refactors and test changes are not listed.

Each release has a section headed `## <tag> (<date published, UTC>)`. The release workflow publishes that section as
the release's notes, and refuses to publish a tag that has none (`scripts/publish_release.sh`). Add new entries under
**Unreleased**; at release the heading becomes the tag. A released section does not change afterwards:
`scripts/check_changelog.sh` holds each one to the hash recorded in `scripts/changelog-released.sha256`, so an entry
that a merge places under a released heading is caught.

## Unreleased

### Added

- **A binding renders one chunk at an exact seed: `render.seed` in a render document** (the C ABI's
  `ggtts_render_opts`, Python's `options=`, Kotlin's `options`). The text is one chunk, drawn once at that seed and
  kept whole, as `ggtts-render --text TEXT --seed S` renders it, so the samples are the same, and a caller that loaded
  the engine once pays no process, model load or hash per chunk. `render.seed` was refused in a render document
  before; it is refused with `seed_salt` (the render call's own seed), `redraw` or `edge_s`, and in a load document,
  each under `render.seed`, and an empty text under `render.text`.

## v0.1.0-rc.13 (2026-11-15)

This pre-release is cut from `rc`. Its CUDA promotion gate (`scripts/rc_gate.sh`) has not run yet; the CPU checks in CI passed.

### Added

- **Qwen3-TTS's sampler settings, frame cap and language are options**: `--temperature`, `--top-k`, `--top-p`,
  `--repetition-penalty`, `--max-frames` and `--language` (`load.temperature` and the rest), with the reference CLI's
  meanings, and `--subtalker-temperature`, `--subtalker-top-k` and `--subtalker-top-p` for the code predictor, which
  otherwise samples with the talker's settings. Unset, each is the value every render had before (0.9, 50, 1.0, 1.05,
  1200 frames, `zh`), so a render without them is unchanged. A temperature at or below 0, or a top-k of 1 (talker or
  sub-talker), is greedy decoding and is refused, since greedy decoding does not terminate on this model. `ggtts-render`, `ggtts-book` and the load documents
  of the C ABI, Python (`options=`) and Kotlin (`options`) take them; Kitten-TTS 2 refuses each by its key. A render given any records them as `sampling` in its
  record, trace header, timing and duration reports, and a book's manifests, so a resume under other settings
  renders the chapter again. The capability statement (`ggtts_capabilities`) gains `sampler`.
- **A model file can be given as an open descriptor, `/proc/self/fd/<n>`.** Every front end's model, codec and
  voice paths take that form, and the engine reads through descriptor `n` without reopening the path. This is how an
  Android app hands over a file the user picked with the system picker: reopening the path fails with "Permission
  denied" for an app without a storage permission. Any other path opens as before. The Android demo uses it to load
  voices from a folder chosen once (⋮ "Choose a models folder…") in place, with no copy.
- **`--steer` takes an `https://` URL or `hf://<owner>/<repo>[@<revision>]/<path>`, downloaded once into the
  cache.** A published steering preset renders in one command, with no separate download. The file goes into the same per-user cache as the model files
  (`GGTTS_CACHE_DIR`, or `load.cache_dir`), at `urls/<sha256 of the URL>/<file name>`, and later runs reuse it.
  `URL#sha256=<hex>` pins it, refusing a file whose sha256 differs, and the log names the sha256 either way.
  `--offline` refuses a URL that is not cached yet, naming it. It is the options table's `render.steer`, so
  ggtts-book and every binding take the URL too. A local path is unchanged.
- **The Android demo and its system TTS engine carry the options document.** The app's request, its saved voices and
  its render calls are one options document, handed to the Kotlin binding as `options`. Any option the binding takes
  (a style instruction, steering, an edge, redraws, a levelled stream) now reaches the app and the system engine with
  no new field in either. The screen's choices are values in that document: the files, the speaker and the threads.
  The app still streams the raw pieces unless the document's `output` section says otherwise, and it still never
  downloads. Voices saved by an earlier version load as before and are saved as documents from then on. A saved voice
  the binding refuses is reported unusable rather than dropped. The system engine logs one line when a client asks
  for a speech rate or pitch, since ggtts has neither.

- **A stream through an options document is levelled by default, as `ggtts-render --stream` is.** The C ABI's
  `ggtts_render_opts` with a callback, Python's `render_stream` and `stream` with `options=`, and Kotlin's
  `renderStream` with `options` now hand each piece levelled by one gain from the first 5 s of audio
  (`output.stream_level_seconds` changes the hold). That is the stream to play: the same text and seed give
  `ggtts-render --stream`'s samples. `{"output": {"stream": "raw"}}` or `"stream_raw": true` gives the pieces before
  the level step, as before. Nothing existing changes: `ggtts_render_stream`, and the Python and Kotlin stream calls
  without options, still stream the unlevelled pieces, and a raw document gives the same pieces it gave. In C, a
  callback with no `output.stream`, which was refused, now streams levelled.
- **Python and Kotlin read the options schema and an engine's capabilities from the library**, as the C ABI does
  since version 5. Python: `ggtts.options_schema()` and `TTS.capabilities()`, each a dict. Kotlin:
  `Ggtts.optionsSchema()` and `Ggtts.capabilities()`, each JSON. Both come from the same shared pieces as
  `ggtts_options_schema` and `ggtts_capabilities`, so every front end states the same thing, and neither binding ships
  a schema file. A refused option's key was already on Python's `OptionError.key` and Kotlin's `GgttsException.key`.
- **Kotlin takes every option through one parameter, `options`.** `Ggtts.load(options = ...)` takes a load document,
  and `render(text, options = ...)` and `renderStream(text, options = ...)` a render document laid over the load's
  defaults: steering, `edge_s`, `redraw` and a per-render instruction. It is the C ABI's document, as a JSON string
  over JNI. A refusal throws `GgttsException` with `ERR_ARG`, and the new `GgttsException.key` is the refused
  option's schema key, read from `ggtts_last_error_key`; the trace (no file sink a phone user can read) and
  `load.device` (CPU only) are refused by key. The `load` parameters keep working beside a document, and a value
  given both ways is refused, naming both. `render` and `renderStream` take `seed: Long? = null` (the seed salt,
  `DEFAULT_SEED` when not given) and `load` takes nullable `engine`, `threads` and `allowDownload` with the same
  defaults as before; the old signatures stay in the class, hidden from source, so compiled code still links. A call
  without `options` renders the same samples as before.
- **Python takes every option through one argument, `options=`.** `ggtts.load(options=...)` takes a load document
  (the `load` section, `cache_dir` included, and `render` defaults), and `render`, `render_stream` and `stream` take a
  render document laid over those defaults: steering, `edge_s`, `redraw`, a per-render instruction and a trace. A
  document is a dict or a JSON string, read and checked by the C ABI's own document code. A refused option is the new
  `ggtts.OptionError`, a `GgttsError` and a `ValueError`, whose `key` is the option's schema key (`render.steer`); a
  value given both as an argument and in the document is refused, naming both. `seed` on a render is documented as
  the seed salt it already was (`render.seed_salt`). The arguments `engine`, `device` and `offline` now default to
  None (not given), which loads exactly what their old defaults did, and a setting refused by its key (a speaker on
  an engine that has none) is now an `OptionError`, still a `GgttsError` with the same code and message. A call
  without `options` renders the same samples as before.
- **`--kitten-full-head` (`load.kitten_full_head`) is an option of the table**, with `GGTTS_KITTEN_FULL_HEAD` as its
  environment spelling, for `ggtts-render` and `ggtts-book render`. It makes Kitten-TTS 2 compute its whole output
  head, as `GGTTS_KITTEN_FULL_HEAD=1` did, and the audio is unchanged by the move. When it is on, the trace header and
  the timing and duration reports carry `"kitten_full_head": true` (added within ggtts-trace/2, ggtts-timing/1 and
  ggtts-duration/1), and a book's chapter manifests record it, so a resume with the other head is refused. On qwen3-tts
  and fake, the variable exported in the environment is ignored with one warning and listed as ignored in the record;
  the flag, or the key in an options document, is refused. Through `ggtts-render` and `ggtts-book` the variable takes
  the table's boolean spellings (`1`, `true`, `0`, `false`) and refuses any other value; the bindings take no option
  for it, and their engine still reads exactly `1` from the environment. A C ABI load document that gives
  `load.kitten_full_head` is refused. The C ABI capability statement (`ggtts_capabilities`) gains `full_head`.

- **The C ABI states its options and an engine's capabilities, and names a refused option (version 5).**
  `ggtts_options_schema()` returns the options schema built into the library (`schemas/ggtts-options-1.schema.json`
  byte for byte). `ggtts_capabilities(handle, &json)` returns a loaded engine's capability statement as JSON
  (`ggtts-capabilities/1`: instruction, steering, devices, hooks, step events, and the model's `speaker_names`),
  freed with `ggtts_string_free`. `ggtts_last_error_key()` is the schema key of the option a `GGTTS_ERR_ARG`
  refused (`render.steer`, or an unknown key itself), and NULL for any other error. Nothing existing changes.
  `Engine::speakers()` and `ggtts_core::capabilities::statement` are the shared pieces behind it.
- **`--judge asr-qwen,model=DIR` redraws a chunk whose read-back lost or repeated text**, for `ggtts-render
  --story` and `ggtts-book render`, from one shared option (`render.judge`). An ASR (a local Qwen3-ASR, run as
  one Python process per render) reads every draw the duration judge keeps back; a draw that lost a run of 4
  characters or more (`min_run=N`), or said one twice, is rejected and redrawn at the chunk's next seed, up to
  `--redraw`. CJK characters (folded to simplified) and numbers (by value) are compared. An ASR that dies or misses its deadline
  (`timeout=S`, 120 s per draw; `start_timeout=S`, 600 s to load) stops the render, which a rerun resumes. The trace records each reading in a new
  `read` event, and a book's chapter manifest in each chunk's `reads`. Off by default. It needs `--redraw N`, and is
  refused before anything is drawn when its ASR cannot start; `--text` and the bindings refuse it. When the ASR's Python lacks a package, the
  refusal names the missing module (`the model did not load (ModuleNotFoundError: qwen_asr)`). A trace with
  text verdicts replays, taking them as recorded.
- **A text judge in the `ggtts-core` library: `textjudge::TextJudge`.** It judges a draw for duration as before,
  then reads it back with a recogniser the caller supplies (`textjudge::Recogniser`) and rejects a draw that lost
  text (`lost-text`: a run of 4 characters or more, a quarter of a chunk of 8 or more, or 8 in a row lost or
  misread) or said a stretch twice (`repeated-text`, found through a misread character or a filler at the join).
  Both texts are compared as CJK characters folded to simplified with OpenCC's character table, vendored in the
  crate, and each number (written out, in digits or mixed, as in 3萬) as one token for its value, so 三百五十六
  read back as 356 is not lost text and a misheard number is one slip. A single misread or missing
  character is counted, not rejected. Known limits: the fold is many-to-one, so a swap between two traditional
  characters with one simplified form reads as correct; a chunk of fewer than 4 compared characters is never
  rejected; text with no CJK characters or numbers has nothing to compare; a number of any length weighs one token,
  so a dropped number, however long (a phone number), is one lost token and the draw is kept, and a number said
  twice reads as one misread number. No front end uses it yet, and every render is unchanged; `ggtts-trace/2` allows the two new verdicts.
- **The C ABI takes options documents (ABI version 4): `ggtts_load_opts(json, &handle)` and
  `ggtts_render_opts(handle, text, json, on_piece, user, &pcm, &len, &rate)`.** Every option of the shared options
  table reaches C as JSON, so a new option needs no new struct field, function or ABI version. A render document
  takes the seed salt, an instruction, steering, redraws, the edge trim and a trace file; `{"output": {"stream":
  "raw"}}` streams to `on_piece`. A malformed document, an unknown key, a load key in a render document and an option
  the engine does not support are `GGTTS_ERR_ARG` before anything renders. **`ggtts_settings` is frozen at version
  3**; the struct functions keep working unchanged and now render through the same path, sample for sample. See
  [docs/bindings/c.md](docs/bindings/c.md#options-documents).
- **One render entry point in the `ggtts-core` library: `pipeline::render_plan`.** It renders a plan with any trace
  level, tensor hooks and output (the whole read, or a stream raw or levelled) in one call, so combinations that had
  no function before, such as hooks with a stream, now work. It is built from a public step, `pipeline::render_chunk`,
  which renders one chunk (draws at the caller's seeds, the judge, redraws, edge trim) for a caller with its own plan.
  `render_story`, `render_story_traced`, `render_story_hooked` and `render_story_streamed` keep working unchanged, and
  no output changes. `RenderOptions` gains `trim` and `level` (both on by default, as before); code that builds it
  field by field instead of from `RenderOptions::for_profile` must set them. `ggtts-render --text` now renders
  through the same entry point, with both off, and its output is unchanged.
- **`ggtts-book render` takes `--instruction`, `--steer` (with `--steer-scale` and `--steer-layers`) and `--trace`
  (with `--trace-level`)**, meaning what they mean in `ggtts-render`: its flags are read by the shared options table,
  and every chunk is rendered by the core chunk step. A book rendered without them is byte-identical to before. Each
  chapter manifest records the instruction and steering, and a resume under different ones renders the chapter again,
  naming the difference. A manifest without the field reads as "no instruction, no steering"; a manifest written by
  another ggtts build is refused on its engine record, as before, so the chapter is rendered again.
- **`ggtts-book render --redraw N` redraws a chunk the judge rejects**, up to N times at the chunk's seeds for draws
  1, 2, ..., through the same judge as a story. Unset is the engine profile's (0 for `qwen3-tts`), so a book renders
  as before unless asked; at 0 the output and manifests are byte-identical. With redraws, each chunk's manifest entry
  records its kept draw, its draw count and the kept draw's verdict, and a resume at a different `--redraw` renders
  again only the chunks whose outcome the new setting changes. `--redraw` is the shared options table's spelling of
  `render.redraw`, so `ggtts-render --story --redraw N` takes it too and its usage names it.
- **`ggtts-book render` runs instruments**: `--instrument`, `--instruments`, `--require`, `--instrument-device`,
  `--on-instrument-error`, `--instrument-timeout`, `--report` and `--run`, meaning what they mean in `ggtts-render`.
  They run on each chapter FLAC not yet measured with the same settings, with one report per chapter beside it
  (`NNN-<chapter id>.report/`), and the render report gains each chapter's `instruments` entry and the book's
  `measurement`. An ASR instrument is given each chapter's own text as its reference. A `--require`d instrument
  that is not ok, or a chapter left unmeasured under `--require`, exits 3 with every chapter written. An unknown
  instrument, or a `reference_file` given for a book, is refused before anything is rendered. The instrument
  timeout is per chapter. A chapter's kept measurement is reused only while its FLAC, the settings and each
  instrument's version and program (a script's bytes) are unchanged, and a chapter after an aborted one keeps it.
  Without the flags a book renders exactly as before. The instrument run is one library call,
  `ggtts_instruments::options`, which `ggtts-render` now makes too; its reports and exit codes are unchanged.

### Changed

- **`ggtts-render --text --trace FILE` writes the `ggtts-trace/2` event stream on every engine**, as `--story` does:
  the header, `chunk_begin`, the engine's `step`s, `chunk_end` and `end`. Before, it reached only qwen3-tts, which
  wrote its older step-only file, and kitten-tts-2 and fake accepted it and wrote nothing. A `--trace` path that
  cannot be written is now an error on every engine, reported after the model loads; on kitten-tts-2 and fake it used
  to pass silently. `--trace-level` is taken with `--text` too: it needs only `--trace`, and its refusal reads
  "--trace-level requires --trace" on `--story` as well (it was "requires a story and --trace").
- **`ggtts-render --text --stream` is levelled by default**, as `--story`'s is. `--stream-raw` gives the decoded draw
  exactly, which is what `--text --stream` wrote before. The `--text` FLAC is unchanged: the decoded draw, with no
  level step. A `--text` timing report's `stream.preparation_ns` is now a number (it was null), and at
  `--log-level info` `--text` writes the `stream`, `stream_handoff` and `trace_sink` lines a story writes.
- **`ggtts-render --text --stream-level-seconds S` is refused**: `--text` is one chunk, which the stream levels by its
  own gain whatever the hold, so the option changed nothing.
- **An option an engine does not support is refused in one wording everywhere, under its schema key, before anything
  is loaded or drawn.** Each engine states what it supports (`Engine::capabilities`, `ggtts_core::capabilities`), and
  `ggtts-render`, the C ABI, Python and Kotlin refuse with the same text. The old and new wording:
  - a speaker: `--speaker is only supported by the qwen3-tts engine` (ggtts-render) and `a speaker is only supported
    by the qwen3-tts engine, not fake` (the bindings) are now `load.speaker: a speaker is only supported by the
    qwen3-tts engine, not fake`;
  - an instruction: `--instruction is only supported by the qwen3-tts engine` (ggtts-render) and `the fake engine
    takes no style instruction; only qwen3-tts custom_voice (1.7B) and voice_design checkpoints do` (the bindings) are
    now `render.instruction: the fake engine takes no style instruction, which is only supported by the qwen3-tts
    engine's custom_voice (1.7B) and voice_design checkpoints`. A Qwen3-TTS checkpoint that takes none keeps its own
    explanation, now after `render.instruction: `;
  - steering: `--steer is only supported by the qwen3-tts engine, not fake` is now `render.steer: the fake engine
    cannot apply steering, which is only supported by the qwen3-tts engine`;
  - a tensor hook asking for a value the engine does not export is refused before the first draw:
    `hooks: the kitten-tts-2 engine exports nothing to tensor hooks, and a hook wants logits`.

  `ggtts-render --engine fake --device cuda` is now refused (`load.device: the fake engine runs on cpu, not cuda`), as
  the bindings already refused it; the fake engine had ignored the device. Renders that were accepted before are
  unchanged.
- **`ggtts-book` refuses the options of the shared table it does not take by name, with the reason**: `--text`,
  `--story`, `--seed`, `--stream` and its settings, the timing and duration reports, the log settings and the
  instruments. An engine name the table does not list is refused as `load.engine` rather than "unknown engine", and
  `--threads` also takes the table's spelling `-j`.
- **`ggtts-render --text --redraw N` is refused** (`--redraw (render.redraw): --text renders one draw and keeps it;
  redraws are a story's (--story)`). It was accepted and did nothing: `--text` renders one draw.
- **`ggtts-book`'s public Rust API** (for a caller of the library, not the command):
  - `book_render_options` is removed; a book's options are read with the shared table (`ggtts_book::options::parse`);
  - `render::render_chunk` returns a `manifest::ChunkOutput` (the chunk file's bytes and the draws it made, with
    the kept draw's verdict) instead of the bytes alone;
  - `ChapterPlan` has a public `redraw` field, so a struct literal must set it;
  - `ChapterOutcome::Written` has a `rejudged` field, so an exhaustive pattern must name it or use `..`.

### Changed

- **`ggtts-render` reads its flags, `GGTTS_LOG_LEVEL`, `GGTTS_LOG_FORMAT` and `GGTTS_CACHE_DIR` through the options
  table** (`ggtts_core::options`), a flag winning over its variable as before. Its output, `--help` and reports are
  unchanged, and the refusals it had words for keep them. A few command lines it used to accept, or refuse later, are
  now refused as they are read, before any model file is touched:
  - `--threads` below 1;
  - an `--engine` other than `auto`, `qwen3-tts`, `kitten-tts-2` or `fake`;
  - a `--device` other than `cpu` or `cuda` for every engine (kitten-tts-2 rendered on the CPU for any value but
    `cuda`, the fake engine ignored it, and a build without engines never reached the check);
  - an empty `--run`, or an empty path for `--out`, `--model`, `--codec`, `--voice`, `--story`, `--stream`, `--trace`,
    `--report`, the reports or `--instruments`;
  - an `--on-instrument-error`, `--instrument-device` or `--instrument-timeout` value the instruments would refuse,
    even with no instrument asked for.

  An empty `GGTTS_LOG_LEVEL` or `GGTTS_LOG_FORMAT` is now unset rather than an error. `--stream-format raw` is still
  s16le.
- **The Android app says what it is for before anything technical**: it reads text aloud in Cantonese or English,
  on the phone only, slowly, and is not for screen readers. A fresh install also lists the steps to a first read.
  A test fails if the app gains the internet permission while the screen says it has none.
- **The Android app offers voices, not an engine and its files.** It reads what each file in its models directory is
  from the file itself (a GGUF's architecture, a talker's speaker names, a voice JSON's fields), lists the voices the
  files make, and selects the first complete one, so Speak works on opening. Files that make no voice yet are named
  with what they lack. The engine, file pickers, speaker and threads are under "Advanced", closed by default, and the
  adb extras still set them.
- **The Android app's screen puts typing and Speak first.** The voice, a multi-line text box and a filled Speak button
  come right under the title; the purpose paragraphs move under "About this app" and the Android-TTS check under
  "Advanced". While a read runs, a progress bar and a seconds count show until the first sound, with plain wording
  ("Preparing the voice", "Writing the first sentence"). The app follows the phone's dark mode on Android 10 and later.
- **The Android app is laid out as a reading screen.** A voice card opens a sheet of voices, each with its language,
  engine, size, time to first sound and a Sample button. The text sits in a card with Paste, Clear and a character
  count, and a player is docked at the bottom: Read aloud, then the stage, a progress bar, the seconds and Stop. Import,
  About and Advanced are in a ⋮ menu. `scripts/android_voice_samples.sh` makes each voice's sample with the app itself.
- **The Android app has an icon:** 粵 in Noto Sans HK (SIL Open Font License) on the app's light blue, as an adaptive
  icon with a monochrome layer, rendered by `scripts/android_icon.py`.
- **The Android app and its text-to-speech engine are named "ggtts Text-to-Speech"** in the launcher and in the phone's
  text-to-speech settings, where they were "ggtts demo" and "ggtts (on-device, slow)". The speed warning is unchanged
  on the app's screen before a load.
- **A Kitten-TTS 2 model whose token map names ids outside its vocabulary is refused when it loads,** with an error
  naming the `kitten.token_map` field and both numbers. Such a file used to load and could fail or misbehave later.
  For code using the engine crate directly: output-head rows set outside the head are now an error from the forward
  pass, where they used to fall back to the whole head.

### Fixed

- **The Android demo recovers from playback failures.** If the phone's audio output fails to start, it is released
  instead of held; if it refuses audio partway through, the read shows an error instead of reporting a finished read;
  and an error such as running out of memory during playback leaves the screen usable instead of busy until restart.
- **Stop in the Android demo is immediate and safe at any moment.** It no longer holds the screen for up to a second
  while playback winds down, a Stop pressed while the last chunk plays is reported as stopped rather than as a
  finished read, and a Stop pressed just as a read starts no longer lets its first chunk play on.


### Fixed

- **A Qwen3-TTS model given by path speaks with no speaker named.** The speaker is the model's default:
  `hk_cantonese_speaker` when it has one, otherwise the first of its speakers in alphabetical order (`hk_voice_343`
  for the released talkers, the same as the default model). It was always `hk_cantonese_speaker`, which the released
  talkers do not have, so `ggtts-render --model`, the C API, Python and the Android app failed without a speaker.

## v0.1.0-rc.12 (2026-11-13)

### Added

- **An Android library (AAR) and a demo app, built in CI.** `android/` is a Gradle project, and
  `scripts/android_build.sh app OUT_DIR` builds it. It writes `ggtts.aar`, which holds the Kotlin binding,
  `libggtts_ffi.so` for arm64-v8a with `libc++_shared.so`, and under `META-INF/ggtts/` the licence notices and a build
  record (`BUILDINFO.json`: version, commit, ggml pin, features, ABI, minimum SDK). It also writes a one-screen demo
  app that speaks a line with either engine from model files you supply, and the library's unstripped copy for crash
  symbols. No model ships in either. The `android-arm64` CI job uploads all three. See
  [android/README.md](android/README.md).
- `licenses/android-crates.txt`: the licences of the Rust crates linked into the Android library, generated and
  checked by `scripts/third_party_notices.py --check-android`.
- **The Android demo plays a line as it renders.** Each chunk plays as soon as it is made, so first audio comes after
  one chunk instead of the whole line. Stop ends both the playback and the rendering, the screen shows progress, and a
  menu picks 2, 4, 6 or 8 CPU threads (4 by default).
- **The Android demo says what each engine costs before loading it:** the model size, the memory it peaked at on a
  phone, the wait for first audio, and that it is too slow for a screen reader. When free memory is below an engine's
  measured peak, the first Speak warns instead of loading; a second Speak loads it anyway.
- **ggtts can be Android's text-to-speech engine.** The demo app registers a system TTS engine (`TextToSpeechService`)
  that other apps can pick in the phone's settings: Hong Kong Cantonese with Qwen3-TTS and US English with Kitten-TTS 2,
  from the model files of the demo's last finished Speak. It streams each chunk to the platform as it is made, and
  declares its voices as very high latency (first audio takes tens of seconds), so it is not meant for screen readers.
- **An Android read outlives its screen.** Leave the demo during a read and it goes on in a foreground media-playback
  service, whose notification shows progress and has Stop; the app asks for the notification permission at the first
  Speak on Android 13 and later. The screen and the system TTS engine share one loaded engine, kept between reads: one
  of them reads at a time, the other is refused rather than load a second model, and Android's memory trims release it
  only when neither is reading.
- **`ggtts-book inspect` reads an EPUB into chapters.** A new crate and binary, `ggtts-book`, a layer on top of
  the engines. It reads the container, the package's metadata and spine, and the table of contents (the EPUB 3 nav
  document, else the EPUB 2 NCX, else the spine), and turns each chapter's XHTML into headings and paragraphs.
  Nested tables of contents keep their depth, a chapter split across files is one chapter, and front and back
  matter (cover, title page, copyright, contents, index, colophon) is kept and marked skippable. `inspect` prints
  the chapter model as JSON (`ggtts-book-chapters/1`). A DRM-protected book is refused, never decrypted or
  stripped; so are a file that is not a zip, a missing or malformed `container.xml`, and a spine item whose file
  is absent, each with its own error code. A book is read under limits: 64 MiB per entry, 256 MiB for the whole
  book (`book-too-large`), and 256 levels of element nesting, so a zip bomb or a deeply nested document is refused
  with a named error rather than exhausting memory or the stack. Entities count against that budget as they
  expand. `ggtts-book` is not yet in the release archives: build it with `cargo build --release -p ggtts-book`.
  See `docs/book.md`.
- **An exponent in a book is read.** `ggtts-book` drops a superscript only when it is a note marker: it links
  somewhere, or holds only digits or note symbols after punctuation or a space. `mc<sup>2</sup>` keeps its 2, and
  after a digit the exponent is marked so it does not run into the number (`10<sup>6</sup>` reads `10^6`).
- **A space an author wrote between CJK characters is kept.** `第一章 退潮` keeps its space; only a line break
  from the source's formatting between two CJK characters is dropped.
- **`ggtts-book inspect` plans every chunk and its seed.** Chapters are chunked by ggtts's own story chunker at
  the engine profile's ceiling (`--engine`, default `qwen3-tts`), and each chunk gets a seed derived from the
  seed root (`--seed-root`, default `20260905`), the book's identifier, the chapter, paragraph, chunk and draw,
  and the SHA-256 of its text (key `ggtts-book-seed/1`), so any chunk can be rendered alone. The library adds a
  per-chapter manifest (`ggtts-book-manifest/1`), written atomically and checksummed, from which a killed
  render resumes without redoing a finished chunk; a truncated, edited or mismatched manifest is refused and the
  chapter re-planned.
- **`ggtts-book render --engine fake` renders a book end to end.** Each chapter that is not front or back matter
  is rendered chunk by chunk at its planned seeds through one engine, stored per chunk with its manifest, then
  joined with the profile's pauses, levelled, and written as one FLAC per chapter. A stopped render resumes on
  the next run with the same `--out`, rendering only what was left. A resumed chapter can match an uninterrupted
  one byte for byte only on the same platform and build with the same `--threads`, and with an engine whose output
  does not depend on what it rendered earlier in the process. With Qwen3-TTS on CPU it was observed byte-identical
  at kill points 1, 4 and 6 of 13 chunks (see `docs/book.md`). `--engine fake` (no model) runs the whole path without a model.
- **`ggtts-book render` renders with the real engines, loaded once per book.** `--engine qwen3-tts` (now the
  default, as for `ggtts-render`) or `kitten-tts-2`, with `ggtts-render`'s model flags (`--model`, `--codec`,
  `--voice`, `--speaker`, `--device`, `--threads`, `--offline`) and its pinned default model files, in a build with
  the `ggml` feature. A new crate, `ggtts-load`, loads an engine by name with those rules. Every chapter manifest
  records the engine's provenance (family, backend, profile, model SHA-256s, speaker, ggtts build) and each
  chunk's render time; a resume with another engine re-renders the chapter rather than mixing two renders. An
  output directory holding another book is refused.
- **`ggtts-book export DIR --out BOOK.flac` joins a rendered book into one chaptered FLAC.** The chapters'
  own samples, 2 s apart, with a chapter mark per chapter as Vorbis comments (`CHAPTER001`,
  `CHAPTER001NAME`), which mpv and VLC read; written a block at a time, byte-identical for the same render. It
  exports exactly the chapters the last render planned, and checks each chapter file against the SHA-256 its
  manifest recorded. An unfinished render, a changed chapter file, or one that mixes seed roots, engines or books,
  is refused naming the chapter. Each chapter FLAC
  now carries `TITLE`, `ALBUM`, `ARTIST` and `TRACKNUMBER` tags, and `render --include-skippable` renders the
  front and back matter too.
- **`ggtts-book export --out BOOK.m4b` writes an audiobook for audiobook apps.** Lossless ALAC with a chapter
  track and a `chpl` chapter list, and the book's title and authors; the same samples as the `.flac` export. The
  container follows the extension; any other is refused.

### Changed

- **The Android demo's adb extras take only file names in its models directory.** `model`, `codec` and `voice` used to
  accept any path, from any app (the activity is exported), and a path used for a finished read was saved as the system
  TTS engine's model. A path, or a name with `/` or `..`, is now refused, and the whole intent with it, `speak`
  included. Only debug builds read these extras.
- **Kitten-TTS 2 computes its output head only for the tokens a step can keep** (from the first end token to the end of
  the vocabulary: the end, audio and following special tokens, 6,571 of the 158,241 rows), and gives every other token
  the lowest finite score, when the sampler's top-k is on and at most 50 and its top-p is below 1 (the defaults). On a
  test set of 22 lines with both language models no other token reached the top-k set and the tokens are identical; a
  step is 1.14x to 1.21x faster on desktop CPU and about 1.2x on a Galaxy Z Fold5. Outside those settings, or with
  `GGTTS_KITTEN_FULL_HEAD=1`, the whole head is computed. `ggtts-render`'s report names the head each render ran, in the
  engine's backend string (`output head rows A..B` or `whole output head`). The Python binding's `info` reads that
  string once, when the engine loads, and the C ABI's names the head only after a first render.
- **The Android demo states peak memory in bytes, as it compares them.** Its figures read the kernel's KiB as 1000
  bytes: Kitten-TTS 2 peaks at 3.78 GB, not 3.70, and Qwen3-TTS at 3.00 GB, not 2.93, so the low-memory warning now
  comes at the right point.
- **The CPU thread count is capped at the CPUs the process may use, checked before each graph.** Asking for more threads
  than CPUs no longer stalls: ggml's threads wait for each other, so on an Android phone whose app is moved to three
  cores when it leaves the screen, four threads did not finish a line in 400 s that three rendered in 87 s. The count is
  re-read at most every 100 ms; the first time it lowers your count, ggtts says so once on stderr. The output is the
  same at any count. On Linux and Android the cap follows affinity masks, cpusets and cgroup quotas; on Windows it sees
  only the number of logical processors, so affinity and job objects are not honoured there. `--threads`, and `threads`
  in the bindings, still mean the count asked for, and a value of 0 or less is treated as before. Kitten-TTS 2's bench
  JSON records the count asked for, not the count run.
- **Android builds are aligned for 16 KB pages.** `libggtts_ffi.so` and the Android `ggtts-render` link with
  `-z max-page-size=16384`. With 4 KB alignment, Android 15 and later flag the library on a device, and a 16 KB-page
  device refuses it. `scripts/android_build.sh` refuses a library aligned below 16 KB.
- `scripts/android_build.sh build` now ends by building the AAR and the demo app, so it needs a JDK 17 to 23 and the
  Android SDK as well as the NDK.

## v0.1.0-rc.11 (2026-11-12)

### Added

- **`ggtts-kitten-convert --layout tq1_0` writes an exact `TQ1_0` Kitten-TTS 2 GGUF**
  ([#200](https://github.com/dandelion-love/ggtts/issues/200)). The layout is `tq1_0-pad128`: each stock `TQ1_0`
  block holds one 128-weight group and 128 zero trits, with the group's own scale, so every trit and every scale of
  the source survives. The readback check verifies it as it does `q4_0` and `tq2_0`, and `--verify` checks an
  existing one. The 196 ternary matrices take 3.375 bits a weight, about 595 MB against 793 MB for `Q4_0`. `Q4_0`
  stays the default: on the CPU the padded `TQ1_0` product is bit-identical to padded `TQ2_0`'s, which misses the
  ternary parity bar that `Q4_0` passes. On CUDA every layout is held as the same exact f16. The engine and
  `ggtts-render` load the new layout on the CPU and on CUDA. Files in the older 256-weight `tq1_0-256-experimental`
  layout still load; that layout gives two 128-weight groups one scale, which is lossy for this model, and the
  converter refuses to write it.
- **Steering: `ggtts-render --steer FILE` adds vectors to the Qwen3-TTS talker's hidden state.** The file is
  safetensors with one F32 vector per talker layer, named `talker.layer.<i>`, each of the talker's hidden size.
  Each vector, times `--steer-scale X` (default 1.0), is added at the output of its layer, after the layer's residual
  add, at every position of the prompt and of every generated frame: what a PyTorch forward hook on the layer's
  output does. `--steer-layers 23,24,25` applies only those layers of the file (default: every layer in it). The
  code predictor is not steered. A render without `--steer` is the same render as before, and its graph has no
  steering node. A file that is not readable, a tensor with another name or dtype, a tensor named twice, a value that
  is not finite, a vector of the wrong length, a layer out of range, a layer list naming a layer the file does not
  hold or naming one twice, a scale that is not finite in f32 or makes a scaled value overflow f32, `--steer-scale` or
  `--steer-layers` without `--steer`, and `--steer` with any engine but `qwen3-tts`, are each refused before anything
  is rendered. Steering is a request input, not an instrument: it does not go through
  the hooks, and the trace-identity check still holds on a steered render. See [docs/steering.md](docs/steering.md).
- **A steered render says so in every record it writes.** The trace header, the timing and duration reports and the
  instruments' `report.json` gain an optional `steer` object: the steering file's sha256, the layers applied and the
  scale. An unsteered render writes none. The field is added within `ggtts-trace/2`, `ggtts-timing/1`,
  `ggtts-duration/1` and `ggtts-report/2`, as `instruction` and `models` were: a document without it is still valid.
  Replay refuses a steered trace unless it is given the same steering.
- **The C and Kotlin binding find the default model, and the C ABI can render on the GPU.** The C ABI is now
  version 3: `ggtts_settings` gains three trailing fields, `device`, `cache_dir` and `allow_download`
  (`GGTTS_SETTINGS_INIT` sets them to NULL, NULL and 0). With engine `qwen3-tts` and no model or codec path,
  `ggtts_load_with` loads the default model `ggtts-render` uses, from the same cache, verified against the same
  sha256, with its default speaker `hk_voice_343` when no speaker is given
  ([#164](https://github.com/dandelion-love/ggtts/issues/164)). It downloads a missing file only when
  `allow_download` is 1; otherwise the error (`GGTTS_ERR_ENGINE`) names the file and its URL. `device = "cuda"`
  renders Qwen3-TTS on the GPU in a library built with `ggtts-ffi`'s new `cuda` feature (Kitten-TTS 2 takes the same
  setting, but its CUDA path through the C ABI is not yet tested); a library
  without it refuses with `GGTTS_ERR_ENGINE` before reading any file, and an unknown device is `GGTTS_ERR_ARG`
  ([#167](https://github.com/dandelion-love/ggtts/issues/167)). The archives' C library is the CPU build, so it
  refuses `cuda`. In Kotlin, `Ggtts.load` takes `cacheDir` and `allowDownload`; it has no device, because Android has
  no GPU backend. Existing callers are unchanged: `ggtts_load`, NULL settings and a struct from the version-2 header
  load the paths they give, on the CPU, as before, and the seven-parameter Kotlin `Ggtts.load` stays in the compiled
  class and loads as it did. The library now links the same HTTPS client as `ggtts-render` (ureq with rustls), used
  only when a download is allowed.
- **The release archives ship the C library and `ggtts.h`.** Until now the C ABI (and with it
  `ggtts_load_with` and `ggtts_abi_version`) could only be had by building from source. Every archive now has
  `include/ggtts.h` and the library in `lib/`: `libggtts_ffi.so` in the Linux CPU and CUDA archives, and
  `ggtts_ffi.dll` with its MSVC import library `ggtts_ffi.dll.lib` in the Windows archive. The library carries the
  same engines as `ggtts-render` and renders on the CPU, including in the CUDA archive, whose library is the CPU
  build and loads without the NVIDIA driver. It exports the C ABI only, not the JNI entry points. `BUILDINFO.json`
  lists it under `"libraries"`, and the Linux library is held to the same Ubuntu 22.04 glibc floor as the binaries.
  Where the files are and how to link them: the README's "The C library" section.
- **The CUDA archive runs on T4, A100 and H100 class GPUs.** It carried code for compute capabilities 8.6 and 8.9
  only; it now carries machine code for 7.5, 8.0, 8.6, 8.9 and 9.0, and PTX for 9.0 so a newer GPU can compile it
  on first use. `BUILDINFO.json` lists them as `cuda_architectures` and the new `cuda_ptx`. The archive is larger;
  the README lists the supported GPUs.
- **Every archive is smoked before it is published, and the published release is checked.** The release renders one
  Qwen3-TTS line with `--speaker hk_voice_343` and the default model from the Linux CPU archive (on Ubuntu 22.04 and
  in a bare `ubuntu:22.04` container with only `libgomp1`) and the Windows archive, checks that an unknown speaker
  is refused and that `--version` names the tag's commit, and holds each archive to the sha256 its build recorded.
  Only then are the archives attested and published; afterwards the release downloads them and checks their sha256
  and attestations.
- **A Python binding: `import ggtts`.** `crates/ggtts-py` (PyO3, an abi3 wheel for Python 3.9 and later, built
  with maturin) loads Qwen3-TTS and Kitten-TTS 2 with the settings of `ggtts_load_with` (a speaker, a style
  instruction) and a device (`cpu` or `cuda`), and with no model loads the default model as `ggtts-render` does,
  with an `offline` mode. `TTS.render` returns float32 samples (a numpy array when numpy is installed, else an
  `array.array`, or the bytes) and their rate, `Audio.write_flac` writes FLAC with the library's writer, and
  `TTS.stream` (or `render_stream` with a callback) yields the chunks in order and can be stopped. An invalid
  argument or a refusal raises `ggtts.GgttsError` with the C-ABI's `code` and the `message`; a panic is caught and
  raised as `ERR_PANIC`, and the GIL is released while an engine loads and renders. It is not on PyPI.
- **Every release ships the Python wheels** beside the archives, attested like them:
  `ggtts-<version>-cp39-abi3-manylinux_2_35_x86_64.whl` (Linux x86_64; Ubuntu 22.04 / glibc 2.35 or later, the
  archives' floor) and `ggtts-<version>-cp39-abi3-win_amd64.whl` (Windows x86_64), one wheel each for every Python
  from 3.9. `<version>` is the tag's (`v0.1.0-rc.11` is `0.1.0rc11`), and `ggtts.VERSION` is the tag. Install one with
  `pip install <file>.whl`; numpy is optional. The wheel carries the engines (Qwen3-TTS and Kitten-TTS 2) and its
  licence notices in `ggtts/licenses/`; the Linux wheel bundles the GNU OpenMP runtime (libgomp) and its licence.
  Before publishing, each wheel is installed into a clean venv and renders a line, and on Linux one real Qwen3-TTS
  line from the default model.

### Changed

- **Release notes are this file's section for the tag.** A release page now carries what changed in that release,
  followed by the build provenance (the commit and how to verify an archive); until now it carried only the
  provenance. A tag with no section here is not published.

### Documented

- **The same TQ1_0 Kitten model and seed give different audio on the CPU and on CUDA, and the README now says
  why** ([#152](https://github.com/dandelion-love/ggtts/issues/152)). On the CPU, ggml multiplies a packed TQ1_0
  matrix by first rounding the activations to 8-bit integers with one scale per block of 256 values (Q8_K); the
  weights are read exactly. CUDA holds the same weights as exact f16 and keeps the activations in floating point,
  as does `--weights f32` on either device. Measured on one model, the CPU's packed logits are within cosine
  0.99983 of the float reading of the same file, which is below the ternary parity bar (0.9999); sampled token
  sequences part after a median of 15 tokens. Speech-recognition error rates on 45 lines did not differ measurably
  between the two, and nobody has compared them by ear. This is not new in this release: it has been so since
  TQ1_0 loaded on both devices (v0.1.0-rc.8).
- **Which archive to download**, and that the same seed on the Windows and Linux CPU archives is not byte-identical
  audio, are in the README.
- **How to use each binding: [docs/bindings](docs/bindings/README.md)**
  ([#192](https://github.com/dandelion-love/ggtts/issues/192)). One page each for the C library, Kotlin on Android
  and Python, and a table comparing them (for one, the C and Kotlin bindings download the default model only when
  asked, while Python downloads it unless `offline=True`). Each page shows how to get the binding and a first render
  that works, then covers loading options, streaming and stopping, every error code and what to do about it,
  threads, freeing, Kitten-TTS 2 and platform notes. Every code example is a file CI compiles and runs on the fake engine. The README's
  C library and Python sections, `docs/android/binding-api.md` and `crates/ggtts-py/README.md` now point there.
  There is still no published Android package: the Kotlin page says how to build the class and its library into an
  app.

## v0.1.0-rc.10 (2026-11-11)

The command-line tools render the same as in v0.1.0-rc.9; this release changes the C ABI and the Kotlin binding.

### Added

- **The C and Kotlin binding take a Qwen3-TTS speaker and style instruction.** The C-ABI is now version 2:
  `GGTTS_ABI_VERSION` in `ggtts.h` and `ggtts_abi_version()` in the library. `ggtts_load_with` is `ggtts_load` plus
  a `ggtts_settings` struct (`size`, `speaker`, `instruction`; start from `GGTTS_SETTINGS_INIT`), and in Kotlin
  `Ggtts.load` takes `speaker` and `instruction` as nullable named parameters. NULL or empty is no setting, and the
  output is then what `ggtts_load` gives, sample for sample. An unknown speaker (the message lists the model's
  speakers) and an instruction on a checkpoint that takes none are refused while loading with `GGTTS_ERR_ENGINE`
  (a `GgttsException` in Kotlin); either setting on another engine is `GGTTS_ERR_ARG`. The existing functions keep
  their signatures and behaviour, and the five-parameter Kotlin `Ggtts.load` stays in the compiled class, so code
  built against it runs without recompiling. Without a speaker the binding still uses `hk_cantonese_speaker`, so the
  v0.0.5 talker (`hk_voice_343`, `hk_voice_777`) needs one; resolving the default model and its default speaker in
  the binding is [#164](https://github.com/dandelion-love/ggtts/issues/164).

Pull requests: [#165](https://github.com/dandelion-love/ggtts/pull/165), promoted in
[#166](https://github.com/dandelion-love/ggtts/pull/166).

## v0.1.0-rc.9 (2026-11-11)

Superseded by v0.1.0-rc.10, which renders the same from the command line.

### Fixed

- **The Linux CPU archive starts on Ubuntu 22.04 again.** The archives of v0.1.0-rc.7 and rc.8 were built on Ubuntu
  24.04 and needed glibc 2.39 (`GLIBC_2.38`/`GLIBC_2.39` not found on older systems,
  [#85](https://github.com/dandelion-love/ggtts/issues/85)). The archive is now built on Ubuntu 22.04, and the
  release refuses a Linux CPU binary that needs more than Ubuntu 22.04 ships: glibc 2.35, libstdc++ `GLIBCXX_3.4.30`
  and `CXXABI_1.3.13`, libgcc_s `GCC_12.0.0`, libgomp `GOMP_5.1`/`OMP_5.1` (`scripts/check_glibc_floor.py`). The
  CUDA archive was already built on Ubuntu 22.04 and is unchanged.

### Added

- **`ggtts-render --instruction TEXT`** passes a natural-language style instruction ("Speak slowly and softly.") to a
  Qwen3-TTS checkpoint that takes one: `custom_voice` at 1.7B, or `voice_design`. It is placed before the prompt as
  the reference runtime places it, and the tokens match the reference's on CPU. A `base` checkpoint, a 0.6B
  `custom_voice` checkpoint and the other engines refuse it with an error naming the checkpoint type; an empty
  instruction is no instruction, and renders exactly as before. In the library it is
  `ChunkRequest::instruction` and `RenderOptions::instruction`. A story trace's `chunk_begin` records it as
  `instruction` when there is one; a strict `ggtts-trace/2` reader built against the older schema must allow that
  field. Code that builds a `ChunkRequest` must now set `instruction` (`None` for none). In this release the C and
  Kotlin binding does not carry an instruction; v0.1.0-rc.10 adds it.
- **A render with no model downloads the default one.** `ggtts-render --text "..." --out x.flac` with no `--model`
  renders with Qwen3-TTS on the talker and codec of `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` v0.0.5 (2.33 GB, pinned
  by commit and sha256) and the speaker `hk_voice_343`. The first run says what it will download, from where, how
  large and where to (at every log level), reports progress every 5 s, then downloads over HTTPS only into a
  per-user cache and verifies each file before use; later runs download nothing. A download that receives nothing
  for 60 s fails. `GGTTS_CACHE_DIR` moves the cache. `--offline` never downloads, and names any missing file and its URL.
  A path given with `--model` or `--codec` is used as it is, never downloaded over or held to a pinned hash, and may
  be mixed with a default file. Until now a render without `--model` was an error. See "Default model files" in
  the README.
- **Every report records the model files** under a new `models` list: role, path, sha256, whether the file came
  from the cache or a given path, and a cached file's URL. It is in the timing report, the duration report, the
  instrument `report.json` (a render's, not `ggtts-measure`'s) and `run.log`, and the trace header, which now also
  fills `model_sha256` and `codec_sha256`. Every render prints the same at the default log level. A given file is
  hashed once per run. The formats keep their versions (`ggtts-timing/1`, `ggtts-duration/1`, `ggtts-report/2`,
  `ggtts-trace/2`); a document without `models` is still valid.
- **ggtts-render links an HTTPS client** (ureq with rustls and its bundled Mozilla roots; no OpenSSL or other
  system TLS library). `licenses/rust-crates.txt` lists the crates it adds.
- **The Kitten tools ship in every archive.** `ggtts-kitten-convert` (the language model and decoder GGUFs from the
  published model) and `ggtts-kitten` (`prepare`, a reference voice from a clip) sit beside `ggtts-render`, so a
  Kitten render needs only an archive and the upstream model files, and no Python. A Kitten GGUF converted before
  the tokenizer was embedded is refused with the command that converts it again; every refusal on that path names
  the file it expected and the shipped command that makes it. The archive carries the Kitten guide as
  `README-kitten.md`, with the model's pass-through terms, and the licences of the sources the engine ports.
  In the CUDA archive the Kitten tools are CPU builds: they need no GPU, and a CUDA-linked `ggtts-kitten` would
  have carried a second copy of the CUDA kernels.
- **Every binary's `--version` prints the ggtts commit** it was built from, between the tag and the ggml pin; the
  release refuses an archive whose binaries disagree with `BUILDINFO.json`'s `ggtts_commit`.

Pull requests: [#156](https://github.com/dandelion-love/ggtts/pull/156),
[#157](https://github.com/dandelion-love/ggtts/pull/157), [#159](https://github.com/dandelion-love/ggtts/pull/159),
[#162](https://github.com/dandelion-love/ggtts/pull/162), promoted in
[#163](https://github.com/dandelion-love/ggtts/pull/163).

## v0.1.0-rc.8 (2026-11-11)

Superseded by v0.1.0-rc.9: this release's Linux CPU archive does not start on Ubuntu 22.04 (glibc 2.35), and its
archives cannot render Kitten-TTS 2 models without building the Kitten tools from source.

### Read this before upgrading from v0.1.0-rc.7

- **A story can render differently at the same seed.** The story splitter used to drop short fragments and some
  punctuation; it now keeps every non-whitespace character of the input, in order. A story whose chunks change gets
  different audio from rc.7 at the same seed; a story whose chunks are unchanged keeps its chunks and seeds.
- **Invalid input is refused before anything is rendered.** A request with nothing to say (for example only
  punctuation) is now an error, reported before the model is loaded or an output file is created.
- **Traces are `ggtts-trace/2`.** `ggtts-render --trace` writes `ggtts-trace/2`; this release still reads
  `ggtts-trace/1`. A reader that accepts only `/1` must be updated. The header now records the trace level
  (`--trace-level 1|2`) and what `p_stop` means.
- **Reports are `ggtts-report/2`.** Instrument envelopes and the merged report are written as `ggtts-report/2`
  (`schemas/ggtts-report-2.schema.json`). `ggtts-report/1` envelopes are still read and are normalized to `/2`; a
  reader that accepts only `/1` cannot read the new reports.
- **`p_stop` means something else for Qwen3-TTS.** It is now the end-of-speech share of the weights the sampler
  actually draws from (after temperature, top-k and top-p), labelled `qwen-sampler-weights/1` in the trace header
  and in each stop-hook record. Values from rc.7 traces and reports are not comparable with new ones; an old,
  unlabelled trace still reads, with its `p_stop` meaning reported as `unknown`. Strict readers of the stop-hook
  JSONL must allow the new `p_stop_semantics` field.
- **The x86-64 CPU archives need AVX2, FMA, F16C and BMI2**, and an operating system that enables AVX state. They
  do not need AVX-512, and they no longer depend on the instruction set of the machine that built them.

### Added

- **`ggtts-render --speaker NAME`** selects a named speaker of a Qwen3-TTS model, for example
  `--speaker hk_voice_343` or `--speaker hk_voice_777` with `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` v0.0.5. Without
  it the default is still `hk_cantonese_speaker`. A name the model does not have is an error that lists the
  model's speaker names. The option is refused for other engines.
- **`ggtts-measure --audio-state raw|trimmed|unknown`** records what you know about the audio you hand it; the
  default is `unknown`. Reports no longer claim `trimmed` for audio whose state was never stated.
- **Timing in traces and a `render-timing` instrument**: the time to first audio, the render's real-time factor and
  per-stage totals, from a streaming story trace. `ggtts-render --timing-report FILE` writes a `ggtts-timing/1` report
  (`docs/timing-and-logging.md`).
- **A `pitch-range` instrument**: the median and range of voiced pitch. **It needs Node.js** on the machine that
  runs it; its detector (Pitchy and fft.js) is bundled in the archive, so it needs no npm install and no network.
  Without Node.js the instrument is skipped, and a run that requires it is incomplete.
- **A `loudness-ends` instrument**: the loud and soft ends of the speech level. It needs the Python environment in
  its README (NumPy, SciPy, librosa, soundfile).
- **Two optional speech-recognition instruments, `asr-sensevoice` and `asr-qwen`**: character error rate of a
  render against a reference text, with a model you supply. They run on the CPU by default under `ggtts-measure`,
  and on CUDA when asked (`--instrument-device`).
- **Kitten-TTS 2 models in the experimental `TQ1_0` layout load on the CPU and on CUDA** (`docs/tq1-cpu.md`). On
  CUDA each matrix is expanded at load to f16, checked value for value. The CPU's packed path and CUDA are different
  numeric paths, so the same model and seed do not give the same tokens on the two
  ([#152](https://github.com/dandelion-love/ggtts/issues/152)). This changelog as published with rc.8 said they
  were refused on CUDA; the rc.8 code already loaded them there.
- **Licence notices in every archive**: `THIRD_PARTY.md`; `licenses/ggml-LICENSE.txt` for the statically linked
  ggml; `licenses/rust-crates.txt`, generated from the dependency tree, with the licence texts of every Rust crate
  linked into the two binaries; and the licence file beside each bundled third-party source (Pitchy, fft.js, the
  speech-core selection). The CUDA and cuBLAS libraries are not bundled and are not covered.

Pull requests: [#86](https://github.com/dandelion-love/ggtts/pull/86),
[#90](https://github.com/dandelion-love/ggtts/pull/90), [#92](https://github.com/dandelion-love/ggtts/pull/92),
[#93](https://github.com/dandelion-love/ggtts/pull/93), [#95](https://github.com/dandelion-love/ggtts/pull/95),
[#96](https://github.com/dandelion-love/ggtts/pull/96), [#97](https://github.com/dandelion-love/ggtts/pull/97),
[#98](https://github.com/dandelion-love/ggtts/pull/98), [#99](https://github.com/dandelion-love/ggtts/pull/99),
[#100](https://github.com/dandelion-love/ggtts/pull/100), [#103](https://github.com/dandelion-love/ggtts/pull/103),
[#114](https://github.com/dandelion-love/ggtts/pull/114), [#115](https://github.com/dandelion-love/ggtts/pull/115),
[#117](https://github.com/dandelion-love/ggtts/pull/117), [#122](https://github.com/dandelion-love/ggtts/pull/122),
[#123](https://github.com/dandelion-love/ggtts/pull/123), [#125](https://github.com/dandelion-love/ggtts/pull/125),
[#126](https://github.com/dandelion-love/ggtts/pull/126), [#128](https://github.com/dandelion-love/ggtts/pull/128),
[#129](https://github.com/dandelion-love/ggtts/pull/129), [#133](https://github.com/dandelion-love/ggtts/pull/133),
[#134](https://github.com/dandelion-love/ggtts/pull/134), [#135](https://github.com/dandelion-love/ggtts/pull/135),
[#136](https://github.com/dandelion-love/ggtts/pull/136), [#138](https://github.com/dandelion-love/ggtts/pull/138),
[#141](https://github.com/dandelion-love/ggtts/pull/141), [#142](https://github.com/dandelion-love/ggtts/pull/142),
[#144](https://github.com/dandelion-love/ggtts/pull/144), [#146](https://github.com/dandelion-love/ggtts/pull/146),
[#148](https://github.com/dandelion-love/ggtts/pull/148), [#149](https://github.com/dandelion-love/ggtts/pull/149),
and [#151](https://github.com/dandelion-love/ggtts/pull/151), promoted in
[#131](https://github.com/dandelion-love/ggtts/pull/131). #102, #105, #109 and #116 were merged into feature branches
and reached rc through the pull requests above.

## v0.1.0-rc.7

Released before this changelog existed; the tree at the tag has no CHANGELOG.md. It reached `main` through
[#84](https://github.com/dandelion-love/ggtts/pull/84). Its release page is no longer published; the tag
`v0.1.0-rc.7` remains.
