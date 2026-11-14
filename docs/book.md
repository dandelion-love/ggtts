# ggtts-book: EPUB books

`ggtts-book` reads an EPUB into chapters a ggtts engine can speak. It is a layer on top of the engines, in its own
crate (`crates/ggtts-book`), and links no engine and no ggml. Today it reads a book, plans every chunk with its seed, and renders
it chapter by chapter through the fake engine, resumably, into one FLAC per chapter. Rendering with a real engine,
chapter tags and a chaptered audiobook export are later steps.

```text
ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
                                 print the chapter model, and each chapter's planned chunks and seeds, as JSON
ggtts-book render BOOK.epub --out DIR [--engine ...] [--seed-root R] [model flags, see Rendering]
                                 render every chapter into DIR, resuming what an earlier run finished
ggtts-book export DIR --out BOOK.flac   one FLAC with a chapter mark per chapter, see Export
ggtts-book --version
```

`--engine` (for `inspect`, default `qwen3-tts`, as for `ggtts-render`) names the engine whose profile the chunks are planned with.
`--seed-root` (default `20260905`, `ggtts-render`'s default seed) is the root every seed is derived from.

Exit 0 on success. Exit 2 when the book is refused, with the reason on stderr, or when the command line is wrong.

**Use it only on books you have the right to convert.** The tool is neutral about what it reads, and it refuses
DRM-protected books rather than removing the protection.

## What it reads

1. **The container.** The file must be a zip. `META-INF/container.xml` names the package document (the OPF).
2. **The package.** Metadata (title, creators, language, and the identifier the package names as its unique
   identifier), the manifest and the spine, which gives the reading order.
3. **The table of contents.** The chapter source, in this order of preference:
   * the EPUB 3 navigation document's `toc` nav (nested lists keep their depth);
   * the EPUB 2 NCX (nested `navPoint`s keep their depth);
   * neither: one chapter per spine file.
   A table of contents that cannot be read, or lists nothing, is a warning, and the next source is used.
4. **The content.** Each spine file's XHTML, turned into headings and paragraphs.

## Chapters

* A table-of-contents entry starts a chapter where it points: at the start of a file, or at an element id inside one.
  Two chapters can share a file, as `part2.xhtml#c3` and `part2.xhtml#c4` do.
* A file the spine lists more than once (through two manifest ids) is read once; each repeat is a warning.
* A spine file the table of contents does not name **continues the chapter before it**, so a chapter split across two
  files is one chapter. The exceptions: the first file in the spine always starts a chapter, and an unnamed file that
  is itself front or back matter (below) starts its own.
* A chapter's `id` is the spine item's id, plus `#fragment` when it starts inside a file. It is the same on every read
  of the same book.
* A chapter's `title` is its table-of-contents label, else its first heading, else empty.
* A chapter with no text (an image-only cover, say) is dropped. If it came from a table-of-contents entry, a warning
  says so.

### Front and back matter are marked, not dropped

A chapter is `skippable` when the place it starts is marked as one of the `SKIPPABLE_TYPES`: cover, title page,
copyright page, table of contents, index, colophon, imprint, errata and the like. The mark can come from:

* the structural semantics (`epub:type`) on `html`, `body`, the wrappers that open the body (`section`, `div`, ...),
  or the element a fragment points at;
* a landmark in the navigation document, or a reference in the EPUB 2 `guide`;
* the navigation document itself;
* `linear="no"` on the spine item.

`skip_reason` names the first mark found, for example `epub:type copyright-page` or `guide title-page`.

## Text

* A block element (`p`, `div`, `li`, `h1` to `h6`, `blockquote`, table cells, the sectioning elements) is a block;
  `h1` to `h6` are headings with their level. `<br/>` ends a block, so each line of verse is its own paragraph.
* Not read: `head`, `script`, `style`, `noscript`, `template`, `svg`, `math`, ruby annotations (`rt`, `rp`), note
  references and the notes themselves (`epub:type` noteref, footnote, endnote, rearnote, or the `doc-*` roles),
  page-break markers, and a `<sup>` holding only a marker such as `12` or `*`.
* Character references, the HTML5 named entities (`&nbsp;`, `&mdash;`) and the internal entities a document's DTD
  declares (`<!ENTITY c "Company">`) are decoded. A DTD entity's value is taken as written, never expanded again,
  so nested entities cannot multiply; external (`SYSTEM`, `PUBLIC`) and parameter entities are never fetched or
  used. An entity nothing declares is left out, and the rest of the paragraph is read. The text entities expand to
  counts against the book's budget (see Limits).
* A `<sup>` is a note marker, and not read, when it links somewhere (an `<a href>` inside) or holds only digits or
  note symbols and does not follow a letter or digit directly (`wool.<sup>2</sup>`). Digits right after a letter
  or digit with no link are an exponent and are read: `mc<sup>2</sup>` reads as `mc2`. After a digit the exponent
  is marked with `^`, so `10<sup>6</sup>` reads as `10^6` rather than running into the number as `106`.
  The price: a note marker without a link straight after a word (`shown<sup>3</sup>`) is read too.
* Whitespace runs become one space. A run holding a line break between two CJK characters or full-width
  punctuation is dropped, so a line break in the middle of Chinese text does not become a pause; a space the
  author wrote there (`第一章 退潮`) is kept. Soft hyphens and zero-width spaces are removed.
* Content files must be UTF-8 and well-formed XML, nested no deeper than 256 elements.

## Limits

A book is read under resource limits (`Limits`, with `Book::read_with_limits` for other values):

| Limit | Default | Refusal |
|---|---|---|
| one entry, inflated | 64 MiB | `too-large` |
| every read of the book together, inflated, plus the text its entities expand to | 256 MiB | `book-too-large` |
| element nesting in any XML document | 256 levels | the document's own code (`content`, `package`, `container`), naming the depth |

Inflation stops at whatever the budget has left, whatever an entry's header claims. The depth limit keeps every
walk over a document's tree far from the end of the stack, on a 1 MiB Windows main thread too: a deeper document
is refused with a named error instead of crashing the process.

## Refusals

Each has its own error and a stable code, printed first:

| Code | When |
|---|---|
| `drm-protected` | `META-INF/encryption.xml` encrypts any file with an algorithm other than font obfuscation, or cannot be read; or a content file is encrypted with a zip password. The book is never decrypted or stripped. |
| `container` | `META-INF/container.xml` is missing or malformed, names no package document, or names one that is not in the book. |
| `package` | The package document is malformed, or has no spine or an empty one. |
| `missing-spine-item` | A spine item names an id the manifest does not list, or a file that is not in the book. |
| `content` | A spine file is not well-formed UTF-8 XHTML. |
| `not-zip` | The file is not a zip archive. |
| `too-large` | An entry is over 64 MiB uncompressed. |
| `book-too-large` | The book's reads together, with the text its entities expand to, pass 256 MiB: a zip bomb, an entity bomb, or one file read many times. |
| `zip`, `io` | The archive or the file cannot be read. |

Font obfuscation (`http://www.idpf.org/2008/embedding` and Adobe's) scrambles embedded fonts, not the text, so it
is not DRM and the book reads. It is accepted only on a font (`.otf`, `.ttf`, `.woff`, `.woff2`, `.ttc`, `.otc`):
the same algorithm on any other file is refused as `drm-protected`, because that file cannot be read as written.

## The JSON

`inspect` prints:

```json
{
  "format": "ggtts-book-chapters/1",
  "identifier": "urn:uuid:...",
  "title": "...",
  "authors": ["..."],
  "language": "en",
  "toc_source": "nav",
  "chapters": [
    {
      "index": 0,
      "id": "ch1",
      "title": "1. Low Tide",
      "depth": 1,
      "skippable": false,
      "skip_reason": null,
      "files": ["OEBPS/Text/ch1.xhtml"],
      "blocks": [
        {"kind": "heading", "level": 2, "text": "1. Low Tide"},
        {"kind": "paragraph", "text": "..."}
      ]
    }
  ],
  "warnings": []
}
```

A reader checks `format` before trusting the rest.

## The chunk plan

Each chapter's blocks (headings and paragraphs) are chunked by ggtts's own story chunker, `ggtts_core::text::plan`,
at the engine profile's `story_ceiling`, with the profile's split marks. There is no second chunker and no ceiling of
the book's own: Qwen3-TTS plans at its profile's ceiling, and an engine with another profile plans at its own. A
block with nothing to speak (a scene break such as `* * *`) has no chunks, and the other blocks keep their indices.

Only engines whose profile this binary is built with can plan: `qwen3-tts`, and `fake` (which renders on the Qwen3-TTS
profile). Kitten-TTS 2's profile is compiled only into ggml builds of its engine, so `--engine kitten-tts-2` is refused
rather than planned at a guessed ceiling.

## The seed per chunk

A chunk's seed is fixed by where it sits and what it says, so any one chunk can be rendered alone and gives the same
seed on every platform. The key is the version `ggtts-book-seed/1` followed by seven fields, each written as
`{byte length}:{value}` and joined by `|`:

| Field | Value |
|---|---|
| seed root | `--seed-root` |
| book id | the package's unique identifier, else `sha256:` and the SHA-256 of the package document |
| chapter id | the chapter's `id` |
| paragraph | the block's index in the chapter |
| chunk | the chunk's index within that block |
| draw | 0 for the first draw; a redraw is 1, 2, ... |
| text | the SHA-256 of the chunk's text, lower-case hex |

The seed is the first four bytes of the SHA-256 of that string, big-endian, the way ggtts reads its own story seeds.
The length prefix makes the encoding unambiguous: a `|` inside a chapter id cannot forge another field. Changing any
field changes the seed; changing the key's fields or encoding is a new key version, with new expected seeds committed
(`tests/plan.rs`, checked on Linux and Windows CI, and checked against Python's `hashlib` when committed).

Seeds make a chunk reproducible within ggtts's own guarantees for one engine, model, backend and platform (hardware class, OS and compiler; see the README). They
do not promise identical audio across hardware.

## The chapter manifest and resume

A render keeps one `manifest.json` per chapter directory, `ggtts-book-manifest/1`:

```json
{
  "manifest": {
    "format": "ggtts-book-manifest/1",
    "book_id": "...", "chapter_id": "ch1", "seed_root": "20260905",
    "profile": "story-v1/qwen3-tts", "ceiling": 45,
    "chunks": [
      {"index": 0, "paragraph": 0, "chunk": 0, "text": "...", "text_sha256": "...", "seed": 3621667395,
       "state": "done", "output": {"file": "chunk-00000.flac", "sha256": "..."}},
      {"index": 1, "paragraph": 1, "chunk": 0, "text": "...", "text_sha256": "...", "seed": 1234,
       "state": "planned", "output": null}
    ]
  },
  "sha256": "the SHA-256 of the manifest object's compact JSON"
}
```

It is written to a temporary file, flushed to disk and renamed over the old one, so a reader never sees half a file.
Each chunk's output is written the same way, and the manifest is saved after every chunk, so a run killed at any point
resumes at the first chunk not recorded as done.

A manifest is trusted only when it parses, its checksum matches, its format is known, and its book, chapter, seed
root, profile, ceiling and every chunk's text, hash and seed match the plan about to run. Otherwise it is refused, with
the reason (`NotJson`, `Checksum`, `Format`, `Mismatch` naming the field, `PlanChanged` naming the chunk), and the
chapter starts again from its plan: a wrong resume would splice another book's or another seed root's audio into this
one. A chunk recorded as done whose output file is missing, or no longer has its recorded SHA-256, is rendered again;
no other finished chunk is.

## Rendering

```text
ggtts-book render BOOK.epub --out DIR [--engine qwen3-tts|kitten-tts-2|fake] [--seed-root R]
                 [--model M.gguf] [--codec C.gguf] [--voice REF.json] [--speaker NAME]
                 [--device cpu|cuda] [--threads N] [--offline]
                 [--instruction TEXT] [--steer FILE [--steer-scale X] [--steer-layers L,L]]
                 [--trace FILE [--trace-level 1|2]] [--redraw N] [--include-skippable]
                 [--instrument NAME[,KEY=VALUE]...]... [--instruments FILE.yaml] [--require NAME]...
                 [--instrument-device cpu|cuda|auto] [--on-instrument-error continue|abort]
                 [--instrument-timeout S] [--report DIR] [--run ID]
```

**The flags are the shared options table's** (`ggtts_core::options`, described in `docs/options.md`), read by its
parser, so each means in a book what it means in `ggtts-render`. A book takes the load section (the engine and its
files), the style instruction (`--instruction`), steering (`--steer` and its scale and layers), the trace (`--trace`,
`--trace-level`), the instruments, `--out`, and the book's own `--seed-root` and `--include-skippable`. Any other
option of the table is refused by its flag and schema name with the reason, before anything is read or written:
`--text` and `--story` (a book renders its chapters), `--seed` (a book's seeds come from `--seed-root`), `--stream`
and its settings (a book is one FLAC per chapter), the timing and duration reports and the log settings. An option a
book does not
take that is set only in the environment (`GGTTS_LOG_LEVEL`, for example) is ambient and is dropped, not refused.

The instruction and the steering reach every chunk's request, as they reach a story's. An engine that cannot take
one refuses the request (the fake engine and Kitten-TTS 2 refuse steering, the fake engine an instruction), so the
render stops at the first chunk with the engine's reason and no chunk file is written. A steering file that does not
read is refused before the engine is loaded. `--trace` writes one ggtts-trace/2 stream for the chunks this run renders
(a header naming the engine, instruction and steering, then each chunk's events); chunks taken from an earlier run
are not in it. Chunk numbers in it restart at each chapter, and it replays with `ggtts_core::trace::replay`.

**Instruments** are `ggtts-render`'s, run by the same library call (`ggtts_instruments::options`) on each chapter
FLAC, after every chapter is rendered and the engine is released. A chapter is measured unless its report directory
holds a `measured.json` from an earlier run with the same chapter FLAC (by SHA-256) and the same instrument settings:
a resumed book measures the chapters an earlier, interrupted run never measured, and a re-run with other instruments
measures every chapter again. Each chapter is one render of the run, named by its directory (`NNN-<chapter id>`), with
its report beside its FLAC
(`NNN-<chapter id>.report/`: the envelopes, `report.json`, `.md`, `.html` and `run.log`), or in
`--report DIR/NNN-<chapter id>/`. The run is `--run`, else `book-<unix seconds>`, one for the whole book. The
render report gives each written chapter an `instruments` entry: its report directory, `measured` (`this run`,
`earlier run` or `no`, with the reason), each instrument's status, and why the measurement is incomplete. The book
gets a `measurement` of `complete` or `incomplete`. A `--require`d instrument that is not ok on any chapter, or a
chapter left unmeasured while an instrument is required, makes the measurement incomplete and the exit 3, with every
chapter FLAC written, as for `ggtts-render`. An unknown instrument is refused before the engine loads.

* **Reference text.** An instrument that compares the audio with a reference (`asr-qwen`, `asr-sensevoice`) is given
  each chapter's own text: the book writes it to `reference.txt` in the chapter's report directory and passes it as
  `arg.reference_file`. A `reference_file` given on the command line or in the instruments file is refused in a book,
  before anything is rendered: one file cannot be every chapter's text.
* **`--on-instrument-error abort`** skips the rest of the instruments on the chapter where one failed, and every
  chapter after it: they are reported `measured: no`, and are measured by the next run.
* **Timeouts are per instrument, per chapter.** `--instrument-timeout` (or an entry's `timeout_s`; 60 s when
  unset) bounds one instrument on one chapter FLAC. An ASR instrument on a long chapter needs more: set it.
* **No event stream.** The book's trace covers every chapter, so no chapter is handed one: an instrument that needs
  it is recorded as `skipped` with that reason (#341 would give each chapter its own).

The engine is `qwen3-tts` by default, as for `ggtts-render`, and the model flags mean what they mean there: a file
not given is the engine's pinned default, taken from the per-user cache (`GGTTS_CACHE_DIR` overrides where) and
downloaded the first time unless `--offline`; with the default talker and no `--speaker`, the default speaker is used.
The model engines need a build with the `ggml` feature (`cargo build --release -p ggtts-book --features ggml`, or
`cuda` for `--device cuda`). Without it they are refused with a named reason and nothing is downloaded, and
`--engine fake` (a tone per chunk, no model) still runs the whole path.

**The engine is loaded once for the whole book**, through the `ggtts-load` crate, which resolves and opens an engine
by the same rules `ggtts-render` uses. The book is read first, so a refused book costs no model load. Each chunk is
planned with the loaded engine's own profile (its ceiling and split marks).

For each chapter that is not front or back matter:

1. `DIR/NNN-<chapter id>/` holds the chapter's `manifest.json` and one `chunk-NNNNN.flac` per chunk. Each chunk is
   rendered by the core chunk step (`ggtts_core::pipeline::render_chunk`, the step every render is made of):
   generated at its planned seed (draw 0), decoded and judged. A draw the judge rejects (a runaway or a silent draw,
   by its seconds of speech per character) is redrawn at the chunk's seed for draw 1, 2, ..., up to `--redraw` times,
   as in a story; the first draw the judge passes is kept, else the last. The kept draw is trimmed at its edges by the
   profile's rule.
2. When every chunk is done, the chunks are read back from their files, joined with the profile's pauses (as the
   story path joins them), levelled by the story path's level step, and written as `DIR/NNN-<chapter id>.flac`.

Front and back matter, and a chapter with nothing to speak, are left out, with the reason in the report. The report
on stdout is JSON (`ggtts-book-render/1`): the engine's provenance, then per chapter `written` with its file and how
many chunks were rendered in this run and taken from an earlier one, or `skipped` with the reason.

### Provenance, and what a resume trusts

Every chapter manifest records what rendered it (`engine`): the engine family, backend and parity, the profile, each
model file's role and SHA-256, the speaker, and the ggtts version and commit. Each done chunk also records how long it
took to render (`render_ms`, generate to encoded FLAC), for reading back; timings are never checked.

A render that stops (a failed chunk, a killed process) keeps what it finished. Run the same command again with the
same `--out` to resume: only the chunks not recorded as done are rendered. A chunk file the manifest does not record
as done (a run killed between writing the file and saving the manifest), and any `.partial` file, is not trusted: that
chunk is rendered again. A resume with another engine, backend, model file, speaker or ggtts build does not splice two
renders into one chapter: the earlier manifest is refused (`for engine ...`) and the chapter is rendered again.

A render with an instruction or steering records them in every chapter manifest (`render`: the non-empty instruction,
and the steering as the trace header writes it, the file's SHA-256, the layers and the scale). A render with neither
writes no `render` field, so its manifests are the bytes they were before the field existed. A resume under a
different instruction or steering is refused the same way, naming the difference (`for instruction "calm", not
"brisk"`, `for steering ...`), and the chapter is rendered again. A manifest without the field, including one written
before it existed, is a render with no instruction and no steering: it resumes a run with neither, and is refused
for a run with either.

**Redraws.** `--redraw N` (the shared table's `render.redraw`; unset is the engine profile's, which is 0 for
`qwen3-tts`, so a book redraws only when asked, as a story does) is how many times a chunk the judge rejects is drawn
again. A render with redraws records `redraw` in each chapter manifest and, per done chunk, `kept_draw`, `draws` (the
number made) and `verdict` (the judge's verdict on the kept draw, `null` when it passed). A render at redraw 0 writes
none of them, so its manifests are the bytes they were before redraws existed. A changed `--redraw` does not refuse a
manifest. A done chunk is kept when a render at the new setting would end at the same draw: its kept draw passed the
judge and is within the new setting, or it is the new setting's last draw. Any other done chunk is judged again under
the new setting and rendered again, counted as `rejudged` in the report. So raising `--redraw` gives the chunks the
judge rejected their further draws, a chunk recorded without draws (rendered at redraw 0) is judged again once
redraws are on, and lowering it renders again the chunks kept at a draw the new setting does not reach.

**What a resume can match, and what has been observed.** Chapters are always joined from the stored chunk files, so
a resumed chapter can match an uninterrupted one byte for byte only when both of these hold:

* **The same platform and build, run the same way.** A platform is the hardware class, the operating system and the
  compiler (see the README): the Windows and Linux CPU builds do not give identical bytes for the same seed. The
  `--threads` setting must also be the same. The manifest enforces the engine, model, backend, speaker and ggtts
  build. It does not record the operating system, the compiler or `--threads`, so keeping those the same is up to
  you. The seed fixes a chunk's sampling, not the floating-point results of another platform.
* **An engine whose output for a chunk does not depend on what it rendered before in the same process.** A resumed
  run starts a new process, so its first chunk follows nothing, where in the uninterrupted run it followed the
  earlier chunks. The tests use an engine without such history and check every kill point. `ggtts-fake` on its own
  is not such an engine (its output alternates with the number of texts it has seen).

Neither condition is a guarantee of identical bytes; they are what a match needs. What has been observed: Qwen3-TTS
on CPU, on one Linux machine with the same build and `--threads`, a 13-chunk chapter killed after 1, 4 and 6 chunks.
Each resume exported byte-identically to the uninterrupted run, and every chunk file matched. That is evidence at
those three points, not a proof for every kill point, engine or device; CUDA was not tried.

The judge reads duration only: it cannot hear a chunk that dropped or repeated words at a normal pace. A judge that
listens (an ASR round trip) plugs into the same step.

`DIR/book.json` names the book rendered into `DIR` and lists the chapters the last render planned (`chapters`, the
directory names, in order). A `DIR` holding another book, or a non-empty `DIR` with no `book.json`, is refused before
anything is written, so one book never overwrites another's chunks. When a chapter file is written, its SHA-256 is
recorded in the chapter's manifest (`chapter_file`).

### Chapter files and their tags

Each chapter FLAC carries a VORBIS_COMMENT block: `TITLE` (the chapter's title), `ALBUM` (the book's title), one
`ARTIST` per author, `TRACKNUMBER` (its place among the chapters written), and `GGTTS_BOOK_ID`, `GGTTS_CHAPTER_ID`
and `GGTTS_SEED_ROOT`, which tie it to its render. Front and back matter is left out unless `render` is given
`--include-skippable`.

## Export: one chaptered audiobook file

```text
ggtts-book export DIR --out BOOK.flac|BOOK.m4b
```

`export` joins a finished render into one FLAC: the chapter files' own samples, in order, with 2 seconds of silence
between chapters (`CHAPTER_GAP_S`), and a chapter mark at the start of each, written as Vorbis comments:
`CHAPTER001=00:00:00.000` and `CHAPTER001NAME=<title>`, the convention ffmpeg-based players (mpv, VLC) read as chapters.
The book's title and authors are its `TITLE`, `ALBUM` and `ARTIST` tags. The file is written a block at a time, so a
book of many hours is never held in memory, and the same render always exports to the same bytes. It prints a JSON
report (`ggtts-book-export/1`) with each chapter's start.

`export` takes exactly the chapters `book.json` lists for the last render, so a directory left from an earlier render
with other options (front matter rendered with `--include-skippable`, then rendered without it) is not exported. It
refuses, naming the chapter: a directory with no `book.json`, or a `book.json` from before the chapter list; a listed
chapter with no directory; a chapter whose manifest has a chunk not done; a chapter directory with no chapter file; a
chapter file whose SHA-256 is not the one its manifest recorded (changed or replaced after the render); chapters
rendered with different seed roots or engines, or for another book; chapters at different sample rates. The container follows the extension of `--out`; anything other than `.flac` or `.m4b` is refused.

**`.m4b`, for audiobook apps.** The same samples as the `.flac` export, as ALAC (lossless) in ALAC's uncompressed
frame form, with the chapters twice: a QuickTime chapter text track (the form Apple's players read) and a Nero
`chpl` list (the form FFmpeg-based players read; it holds at most 255 chapters, and a longer book keeps only the
text track). The book's title and authors are `©nam`, `©alb` and `©ART`. FFmpeg reads the chapters and decodes
samples identical to the `.flac` export's (`tests/m4b.rs`, opt-in with `-- --ignored`).

### Not yet

A ggtts serve process that is already running, when one is reachable, will be used instead of loading a second copy
of the model; that needs a resident serve mode in ggtts first.

## Tests

`cargo test -p ggtts-book`. The fixture books are built in memory by the tests from text written for them, so no
book (and no binary file) is stored in the repository. They cover a nested table of contents, a chapter split across
two files, chapters that start at fragments, an NCX-only EPUB 2 book with a percent-escaped href, a spine-only book,
and one rejection per refusal above, each planted into an otherwise working book. `tests/plan.rs` and
`tests/manifest.rs` cover the plan, the committed seeds, every manifest refusal and resume, with a renderer that
records which chunks it was asked for. `tests/render.rs` and `tests/engine.rs` render the fixture through the fake engine:
end to end, killed and resumed, one engine load per book, provenance in every manifest, a resume with
another engine refused, and an output directory holding another book refused.

## Dependencies

`zip` (MIT) reads the container, with no default features: only stored and deflated entries, through `flate2`'s
pure-Rust backend (`miniz_oxide`), so no C library is linked. `quick-xml` (MIT) parses the XML. Every crate they bring
in is under a licence with an MIT option. `ggtts-book` is not yet in the release archives.
