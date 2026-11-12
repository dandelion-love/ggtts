# ggtts-book: EPUB books

`ggtts-book` reads an EPUB into chapters a ggtts engine can speak. It is a layer on top of the engines, in its own
crate (`crates/ggtts-book`), and links no engine and no ggml. Today it reads and inspects books; rendering a book
through an engine, resuming a killed render and writing chapter audio are later steps.

```text
ggtts-book inspect BOOK.epub     print the chapter model as JSON
ggtts-book --version
```

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
  used. A paragraph holding an entity nothing declares keeps its text as written.
* A `<sup>` holding only digits or note symbols is treated as a note marker and not read. The trade-off: an
  exponent such as `mc<sup>2</sup>` loses its `2` (issue 214).
* Whitespace runs become one space. A space between two CJK characters or full-width punctuation is dropped, so a line
  break in the middle of Chinese text does not become a pause. Soft hyphens and zero-width spaces are removed.
* Content files must be UTF-8 and well-formed XML, nested no deeper than 256 elements.

## Limits

A book is read under resource limits (`Limits`, with `Book::read_with_limits` for other values):

| Limit | Default | Refusal |
|---|---|---|
| one entry, inflated | 64 MiB | `too-large` |
| every read of the book together, inflated | 256 MiB | `book-too-large` |
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
| `book-too-large` | The book's reads together inflate past 256 MiB: a zip bomb, or one file read many times. |
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

## Tests

`cargo test -p ggtts-book`. The fixture books are built in memory by the tests from text written for them, so no
book (and no binary file) is stored in the repository. They cover a nested table of contents, a chapter split across
two files, chapters that start at fragments, an NCX-only EPUB 2 book with a percent-escaped href, a spine-only book,
and one rejection per refusal above, each planted into an otherwise working book.

## Dependencies

`zip` (MIT) reads the container, with no default features: only stored and deflated entries, through `flate2`'s
pure-Rust backend (`miniz_oxide`), so no C library is linked. `quick-xml` (MIT) parses the XML. Every crate they bring
in is under a licence with an MIT option. `ggtts-book` is not yet in the release archives.
