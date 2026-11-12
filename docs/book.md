# ggtts-book: EPUB books

`ggtts-book` reads an EPUB into chapters a ggtts engine can speak. It is a layer on top of the engines, in its own
crate (`crates/ggtts-book`), and links no engine and no ggml. Today it reads a book, plans every chunk with its seed, and renders
it chapter by chapter through the fake engine, resumably, into one FLAC per chapter. Rendering with a real engine,
chapter tags and a chaptered audiobook export are later steps.

```text
ggtts-book inspect BOOK.epub [--engine qwen3-tts|fake] [--seed-root R]
                                 print the chapter model, and each chapter's planned chunks and seeds, as JSON
ggtts-book render BOOK.epub --out DIR --engine fake [--seed-root R]
                                 render every chapter into DIR, resuming what an earlier run finished
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
* Character references and the HTML5 named entities (`&nbsp;`, `&mdash;`) are decoded.
* Whitespace runs become one space. A space between two CJK characters or full-width punctuation is dropped, so a line
  break in the middle of Chinese text does not become a pause. Soft hyphens and zero-width spaces are removed.
* Content files must be UTF-8 and well-formed XML.

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
| `zip`, `io` | The archive or the file cannot be read. |

Font obfuscation (`http://www.idpf.org/2008/embedding` and Adobe's) scrambles embedded fonts, not the text, so it
is not DRM and the book reads.

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

Seeds make a chunk reproducible within ggtts's own guarantees for one engine, model, backend and hardware class. They
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

## Rendering (walking skeleton)

`ggtts-book render BOOK.epub --out DIR --engine fake` runs the whole path end to end with ggtts's fake engine, which
needs no model: chapters, seeds, manifests, resume and one FLAC per chapter. `--engine` must be named, and only
`fake` renders for now; a real engine (`qwen3-tts`, `kitten-tts-2`) is refused until its loader is wired into
`ggtts-book`. The engine is loaded once and used for every chunk of every chapter.

For each chapter that is not front or back matter:

1. `DIR/NNN-<chapter id>/` holds the chapter's `manifest.json` and one `chunk-NNNNN.flac` per chunk. Each chunk is
   generated at its planned seed (draw 0), decoded, and trimmed at its edges by the profile's rule.
2. When every chunk is done, the chunks are read back from their files, joined with the profile's pauses (as the
   story path joins them), levelled by the story path's level step, and written as `DIR/NNN-<chapter id>.flac`.

Front and back matter, and a chapter with nothing to speak, are left out, with the reason in the report. The report
on stdout is JSON (`ggtts-book-render/1`): per chapter, `written` with its file and how many chunks were rendered in
this run and taken from an earlier one, or `skipped` with the reason.

A render that stops (a failed chunk, a killed process) keeps what it finished. Run the same command again with the
same `--out` to resume: only the chunks not recorded as done are rendered, and because chapters are always joined from
the stored chunk files, a resumed chapter is byte-identical to one rendered in a single run. The same book and seed
root give byte-identical chapters (with the fake engine; a real engine's identity is bounded by its own guarantees
on one hardware class). Redraws of chunks the judge rejects are not done yet: every chunk is a first draw.

## Tests

`cargo test -p ggtts-book`. The fixture books are built in memory by the tests from text written for them, so no
book (and no binary file) is stored in the repository. They cover a nested table of contents, a chapter split across
two files, chapters that start at fragments, an NCX-only EPUB 2 book with a percent-escaped href, a spine-only book,
and one rejection per refusal above, each planted into an otherwise working book. `tests/plan.rs` and
`tests/manifest.rs` cover the plan, the committed seeds, every manifest refusal and resume, with a renderer that
records which chunks it was asked for.

## Dependencies

`zip` (MIT) reads the container, with no default features: only stored and deflated entries, through `flate2`'s
pure-Rust backend (`miniz_oxide`), so no C library is linked. `quick-xml` (MIT) parses the XML. Every crate they bring
in is under a licence with an MIT option. `ggtts-book` is not yet in the release archives.
