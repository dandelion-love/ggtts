//! Story 1: the chapter model read from fixture books written for the tests.

mod common;

use common::*;
use ggtts_book::{Block, Book, TocSource};

fn read(epub: &Epub) -> Book {
    Book::read(epub.bytes()).unwrap_or_else(|e| panic!("the fixture should read: {e}"))
}

fn outline(book: &Book) -> Vec<(String, String, usize, bool)> {
    book.chapters
        .iter()
        .map(|c| (c.id.clone(), c.title.clone(), c.depth, c.skippable))
        .collect()
}

fn texts(book: &Book, id: &str) -> Vec<String> {
    let chapter = book
        .chapters
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("no chapter {id}"));
    chapter.blocks.iter().map(|b| b.text().to_string()).collect()
}

fn row(id: &str, title: &str, depth: usize, skippable: bool) -> (String, String, usize, bool) {
    (id.to_string(), title.to_string(), depth, skippable)
}

/// AC1: the container, the OPF and its metadata.
#[test]
fn metadata_comes_from_the_package() {
    let book = read(&nested());
    assert_eq!(book.title, "The Lighthouse Keeper's Almanac");
    assert_eq!(book.authors, ["Ada Test-Writer", "Bo Fixture"]);
    assert_eq!(book.language.as_deref(), Some("en"));
    // The identifier the package names as unique, not the first one listed.
    assert_eq!(
        book.identifier.as_deref(),
        Some("urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d")
    );
}

/// AC2 and AC4: chapters from the nav, nested entries keep their depth, a split chapter is one chapter, chapters
/// can start at fragments, and front and back matter are marked skippable rather than dropped.
#[test]
fn nested_nav_gives_chapters_in_reading_order() {
    let book = read(&nested());
    assert_eq!(book.toc_source, TocSource::Nav);
    assert_eq!(
        outline(&book),
        [
            // The cover is image-only: nothing to speak, so no chapter.
            row("titlepage", "The Lighthouse Keeper's Almanac", 0, true),
            row("copyright", "Copyright", 0, true),
            row("nav", "Contents", 0, true),
            row("part1", "Part One: The Shore", 0, false),
            row("ch1", "1. Low Tide", 1, false),
            row("ch2a", "2. The Second Lamp", 1, false),
            row("part2", "Part Two: The Light", 0, false),
            row("part2#c3", "3. Fog", 1, false),
            row("part2#c4", "4. 霧散", 1, false),
            row("colophon", "Colophon", 0, true),
        ]
    );
    let reasons: Vec<_> = book.chapters.iter().map(|c| c.skip_reason.clone()).collect();
    assert_eq!(reasons[0].as_deref(), Some("epub:type titlepage"));
    assert_eq!(reasons[1].as_deref(), Some("epub:type copyright-page"));
    assert_eq!(reasons[2].as_deref(), Some("epub:type toc"));
    assert_eq!(reasons[9].as_deref(), Some("epub:type colophon"));
    assert!(book.warnings.is_empty(), "{:?}", book.warnings);
}

/// AC2: the second file of a chapter split across two files belongs to that chapter.
#[test]
fn a_chapter_split_across_files_is_one_chapter() {
    let book = read(&nested());
    let ch2 = book.chapters.iter().find(|c| c.id == "ch2a").unwrap();
    assert_eq!(ch2.files, ["OEBPS/Text/ch2a.xhtml", "OEBPS/Text/ch2b.xhtml"]);
    assert_eq!(
        texts(&book, "ch2a"),
        [
            "2. The Second Lamp",
            "The second lamp had not been lit in years.",
            "She lit it anyway, and the harbour answered."
        ]
    );
}

/// AC3: headings and paragraphs, entities decoded, whitespace collapsed, script and style not read.
#[test]
fn xhtml_becomes_headings_and_paragraphs() {
    let book = read(&nested());
    let ch1 = book.chapters.iter().find(|c| c.id == "ch1").unwrap();
    assert_eq!(
        ch1.blocks,
        [
            Block::Heading {
                level: 2,
                text: "1. Low Tide".into()
            },
            Block::Paragraph {
                text: "The keeper counted the gulls before breakfast.".into()
            },
            Block::Paragraph {
                text: "There were seven of them\u{2014}one more than yesterday & one fewer than the day before.".into()
            },
        ]
    );
}

/// AC3: note references, note bodies and marker superscripts are not read; a <br/> ends a paragraph; the text
/// before a fragment's section stays in the chapter before it.
#[test]
fn notes_are_left_out_and_br_breaks() {
    let book = read(&nested());
    assert_eq!(texts(&book, "part2"), ["Part Two: The Light"]);
    assert_eq!(
        texts(&book, "part2#c3"),
        [
            "3. Fog",
            "The fog came in at noon, thick as wool.",
            "Line one of the song",
            "line two of the song"
        ]
    );
}

/// AC3: CJK text joined across a line break, ruby annotations not read.
#[test]
fn cjk_text_joins_across_lines_and_skips_ruby() {
    let book = read(&nested());
    assert_eq!(texts(&book, "part2#c4"), ["4. 霧散", "今日霧散咗，我哋可以出海喇。"]);
}

/// AC4 through a nav landmark alone: a chapter with text and no `epub:type` of its own is skippable because a
/// landmark names it.
#[test]
fn a_nav_landmark_alone_marks_a_chapter_skippable() {
    let epub = nested();
    let opf = epub
        .text("OEBPS/content.opf")
        .replace(
            "</manifest>",
            r#"<item id="index" href="Text/index.xhtml" media-type="application/xhtml+xml"/></manifest>"#,
        )
        .replace("</spine>", r#"<itemref idref="index"/></spine>"#);
    let nav = epub
        .text("OEBPS/Text/nav.xhtml")
        .replace(
            r#"<li><a href="colophon.xhtml">Colophon</a></li>"#,
            r#"<li><a href="colophon.xhtml">Colophon</a></li><li><a href="index.xhtml">Index</a></li>"#,
        )
        .replace(
            r#"<li><a epub:type="bodymatter" href="part1.xhtml">Start</a></li>"#,
            r#"<li><a epub:type="bodymatter" href="part1.xhtml">Start</a></li><li><a epub:type="index" href="index.xhtml">Index</a></li>"#,
        );
    let index = r#"<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>i</title></head><body><h1>Index</h1><p>lamps, 2; tides, 1</p></body></html>"#;
    let book = read(
        &epub
            .with("OEBPS/content.opf", &opf)
            .with("OEBPS/Text/nav.xhtml", &nav)
            .with("OEBPS/Text/index.xhtml", index),
    );
    let chapter = book
        .chapters
        .iter()
        .find(|c| c.id == "index")
        .expect("an index chapter");
    assert_eq!(texts(&book, "index"), ["Index", "lamps, 2; tides, 1"]);
    assert!(chapter.skippable);
    assert_eq!(chapter.skip_reason.as_deref(), Some("landmark index"));
}

/// AC2 and AC4 for EPUB 2: the NCX, a nested entry at a fragment, a percent-escaped href, the guide's title page.
#[test]
fn ncx_is_the_fallback_table_of_contents() {
    let book = read(&ncx_only());
    assert_eq!(book.toc_source, TocSource::Ncx);
    assert_eq!(book.identifier.as_deref(), Some("small-hours-0001"));
    assert_eq!(
        outline(&book),
        [
            row("title", "Small Hours", 0, true),
            row("one", "Chapter One", 0, false),
            row("one#s2", "A Quieter Section", 1, false),
            row("two", "Chapter Two", 0, false),
        ]
    );
    assert_eq!(book.chapters[0].skip_reason.as_deref(), Some("guide title-page"));
    assert_eq!(texts(&book, "one"), ["Chapter One", "The clock struck two."]);
    assert_eq!(texts(&book, "one#s2"), ["A Quieter Section", "Nobody answered."]);
    assert_eq!(book.chapters[3].files, ["book/chapter two.html"]);
}

/// AC2: with neither a nav nor an NCX, each spine file is a chapter titled by its first heading.
#[test]
fn spine_is_the_last_resort() {
    let book = read(&spine_only());
    assert_eq!(book.toc_source, TocSource::Spine);
    assert_eq!(
        outline(&book),
        [
            row("a", "The First Room", 0, false),
            row("b", "The Second Room", 0, false)
        ]
    );
}

/// A broken table of contents is a warning, not a refusal: the next source is used.
#[test]
fn an_unreadable_nav_falls_back_with_a_warning() {
    let epub = nested();
    // Out of the spine, so only the table of contents reads it (a spine item that is not well-formed is refused).
    let opf = epub.text("OEBPS/content.opf").replace(r#"<itemref idref="nav"/>"#, "");
    let book = read(
        &epub
            .with("OEBPS/content.opf", &opf)
            .with("OEBPS/Text/nav.xhtml", "<html><body><nav>"),
    );
    assert_eq!(book.toc_source, TocSource::Spine);
    assert!(
        book.warnings.iter().any(|w| w.contains("cannot be read")),
        "{:?}",
        book.warnings
    );
    assert!(book.chapters.iter().any(|c| c.id == "ch1"));
}

/// A table-of-contents entry that points at a missing anchor is reported, not silently dropped.
#[test]
fn a_dangling_toc_fragment_is_a_warning() {
    let epub = nested();
    let nav = epub.text("OEBPS/Text/nav.xhtml").replace("#c4", "#c9");
    let epub = epub.with("OEBPS/Text/nav.xhtml", &nav);
    let book = read(&epub);
    assert!(book.chapters.iter().all(|c| c.id != "part2#c4"));
    assert!(
        book.warnings
            .iter()
            .any(|w| w.contains("#c9") && w.contains("does not have")),
        "{:?}",
        book.warnings
    );
}

/// AC6's JSON shape, from the library.
#[test]
fn json_carries_the_model() {
    let json = read(&nested()).to_json();
    assert_eq!(json["format"], "ggtts-book-chapters/1");
    assert_eq!(json["toc_source"], "nav");
    assert_eq!(json["chapters"][4]["id"], "ch1");
    assert_eq!(
        json["chapters"][4]["blocks"][0],
        serde_json::json!({"kind": "heading", "level": 2, "text": "1. Low Tide"})
    );
    assert_eq!(json["chapters"][1]["skip_reason"], "epub:type copyright-page");
}
