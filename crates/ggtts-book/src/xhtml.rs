//! XHTML to text a TTS engine can read: headings and paragraphs in document order.
//!
//! * A block element (`p`, `div`, `li`, `h1`..`h6`, `blockquote`, table cells and the sectioning elements) ends the
//!   text before it and starts a new block. `<br/>` ends a block too, so a line of verse is its own paragraph.
//! * Not read: `head`, `script`, `style`, `noscript`, `template`, `svg`, `math`, ruby annotations (`rt`, `rp`),
//!   note references and the notes themselves (`epub:type` `noteref`, `footnote`, `endnote`, `rearnote` and their
//!   plurals, or the matching `doc-*` roles), page-break markers, and a `<sup>` that is a note marker: one that
//!   links somewhere, or holds only digits or note symbols and does not follow a letter or digit directly. An
//!   exponent (`mc<sup>2</sup>`, `10<sup>6</sup>`) is read.
//! * Whitespace runs become one space, and a run that holds a line break between two CJK characters (or full-width
//!   punctuation) is dropped, so a line break inside Chinese text does not become a pause. A space the author wrote
//!   between CJK characters (`第一章 退潮`) is kept. Soft hyphens and zero-width spaces are
//!   removed.

use ggtts_core::text::{is_cjk, is_strip_space};

use crate::dom::{Element, Node};

/// One block of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// A heading, `h1` to `h6`.
    Heading { level: u8, text: String },
    /// Any other block.
    Paragraph { text: String },
}

impl Block {
    pub fn text(&self) -> &str {
        match self {
            Block::Heading { text, .. } | Block::Paragraph { text } => text,
        }
    }
}

/// What a content document holds, in order: blocks, and the ids a chapter may start at.
#[derive(Debug, Clone, Default)]
pub(crate) struct Doc {
    /// The structural types declared on `html`, `body`, and the wrappers that open the body before any text.
    pub start_types: Vec<String>,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub(crate) enum Item {
    Anchor { id: String, types: Vec<String> },
    Block(Block),
}

const BLOCKS: &[&str] = &[
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "blockquote",
    "pre",
    "figure",
    "figcaption",
    "table",
    "tr",
    "td",
    "th",
    "caption",
    "section",
    "article",
    "aside",
    "header",
    "footer",
    "main",
    "nav",
    "address",
    "hr",
    "body",
];
const NOT_READ: &[&str] = &[
    "head", "script", "style", "noscript", "template", "svg", "math", "rt", "rp",
];
const NOTE_TYPES: &[&str] = &[
    "noteref",
    "note",
    "footnote",
    "endnote",
    "rearnote",
    "footnotes",
    "endnotes",
    "rearnotes",
    "pagebreak",
];
const NOTE_ROLES: &[&str] = &[
    "doc-noteref",
    "doc-footnote",
    "doc-endnote",
    "doc-endnotes",
    "doc-pagebreak",
];
const WRAPPERS: &[&str] = &["section", "article", "div", "nav", "header", "main"];

pub(crate) fn extract(root: &Element) -> Doc {
    let mut doc = Doc {
        start_types: root.epub_types(),
        items: Vec::new(),
    };
    if let Some(body) = root.child("body") {
        doc.start_types.extend(body.epub_types());
        let mut wrapper = body.elements().next();
        while let Some(w) = wrapper.filter(|w| WRAPPERS.contains(&w.local.as_str())) {
            doc.start_types.extend(w.epub_types());
            wrapper = w.elements().next();
        }
    }
    let mut state = State {
        doc: &mut doc,
        buf: String::new(),
        heading: None,
    };
    state.walk(root);
    state.flush();
    doc
}

struct State<'a> {
    doc: &'a mut Doc,
    buf: String,
    heading: Option<u8>,
}

impl State<'_> {
    /// Whether a `<sup>` is a note marker (not read) rather than part of the text (an exponent, read).
    ///
    /// It is a marker when it links somewhere (an `<a href>` inside), or when it holds only digits or note symbols
    /// and does not follow a letter or digit directly: `wool.<sup>2</sup>`, `<sup>*</sup>` at a paragraph's start.
    /// Digits right after a letter or digit, with no link, are an exponent and are read: `mc<sup>2</sup>`,
    /// `10<sup>6</sup>`. A word's note marker without a link (`shown<sup>3</sup>`) is read too; that is the price of
    /// keeping exponents, and books mostly link their notes.
    fn note_marker(&self, sup: &Element) -> bool {
        if sup.find_all("a").iter().any(|a| a.attr("href").is_some()) {
            return true;
        }
        if !is_marker(&sup.text()) {
            return false;
        }
        let before = self.buf.chars().next_back();
        !before.is_some_and(char::is_alphanumeric)
    }

    fn walk(&mut self, el: &Element) {
        let local = el.local.as_str();
        if NOT_READ.contains(&local)
            || el.epub_types().iter().any(|t| NOTE_TYPES.contains(&t.as_str()))
            || el.attr("role").is_some_and(|r| NOTE_ROLES.contains(&r))
            || (local == "sup" && self.note_marker(el))
        {
            return;
        }
        let heading = heading_level(local);
        let block = heading.is_some() || BLOCKS.contains(&local) || local == "br";
        if block {
            // The text before a block is finished before the block's own id is recorded, so it stays on the
            // near side of a chapter that starts here.
            self.flush();
        }
        if let Some(id) = el.attr("id") {
            self.doc.items.push(Item::Anchor {
                id: id.to_string(),
                types: el.epub_types(),
            });
        }
        if local == "br" {
            return;
        }
        if heading.is_some() {
            self.heading = heading;
        }
        for child in &el.children {
            match child {
                Node::Text(t) => self.buf.push_str(t),
                Node::Element(e) => self.walk(e),
            }
        }
        if block {
            self.flush();
            if heading.is_some() {
                self.heading = None;
            }
        }
    }

    fn flush(&mut self) {
        let text = normalize(&self.buf);
        self.buf.clear();
        if text.is_empty() {
            return;
        }
        // An anchor seen while this block's text was collected stays before the block, so a chapter that starts
        // at an id inside a paragraph begins with that whole paragraph.
        self.doc.items.push(Item::Block(match self.heading {
            Some(level) => Block::Heading { level, text },
            None => Block::Paragraph { text },
        }));
    }
}

fn heading_level(local: &str) -> Option<u8> {
    match local.as_bytes() {
        [b'h', d @ b'1'..=b'6'] => Some(d - b'0'),
        _ => None,
    }
}

/// A note marker: digits and the usual note symbols only.
fn is_marker(text: &str) -> bool {
    let t = normalize(text);
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit() || "*†‡§¶[](),-".contains(c))
}

fn wide(c: char) -> bool {
    is_cjk(c)
        || ('\u{3000}'..='\u{303f}').contains(&c)
        || ('\u{3040}'..='\u{30ff}').contains(&c)
        || ('\u{ff00}'..='\u{ffef}').contains(&c)
}

/// Collapse whitespace as described in the module documentation.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last: Option<char> = None;
    let mut pending_space = false;
    // Whether the pending run of whitespace holds a line break: source formatting, not a space the author wrote.
    let mut pending_break = false;
    for c in s.chars() {
        if c == '\u{ad}' || c == '\u{200b}' {
            continue;
        }
        if is_strip_space(c) {
            if last.is_some() {
                pending_space = true;
                pending_break |= c == '\n' || c == '\r';
            }
            continue;
        }
        if pending_space {
            if !(pending_break && last.is_some_and(wide) && wide(c)) {
                out.push(' ');
            }
            pending_space = false;
            pending_break = false;
        }
        out.push(c);
        last = Some(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_collapses_and_cjk_joins() {
        assert_eq!(normalize("  a \n\t b\u{a0}c  "), "a b c");
        assert_eq!(normalize("今日\n  好熱。\n我哋"), "今日好熱。我哋");
        assert_eq!(normalize("今日 hot 好"), "今日 hot 好");
        // A space the author wrote between CJK characters is kept (a chapter number and its title); only a line
        // break from the source's formatting is dropped.
        assert_eq!(normalize("第一章 退潮"), "第一章 退潮");
        assert_eq!(normalize("第一章\u{3000}退潮"), "第一章 退潮");
        assert_eq!(normalize("第一章 \n 退潮"), "第一章退潮");
        assert_eq!(normalize("hy\u{ad}phen\u{200b}ated"), "hyphenated");
    }

    fn paragraphs(body: &str) -> Vec<String> {
        let doc = format!("<html><body>{body}</body></html>");
        let root = crate::dom::parse(doc.as_bytes(), crate::archive::MAX_DEPTH).unwrap();
        extract(&root)
            .items
            .into_iter()
            .filter_map(|i| match i {
                Item::Block(b) => Some(b.text().to_string()),
                Item::Anchor { .. } => None,
            })
            .collect()
    }

    #[test]
    fn exponents_are_read_and_note_markers_are_not() {
        // An exponent, right after a letter or digit, with no link: read.
        assert_eq!(paragraphs("<p>E = mc<sup>2</sup>.</p>"), ["E = mc2."]);
        assert_eq!(paragraphs("<p>About 10<sup>6</sup> stars.</p>"), ["About 106 stars."]);
        // A marker after punctuation or a space, or at the start: not read.
        assert_eq!(
            paragraphs("<p>Thick as wool.<sup>2</sup> Then rain.</p>"),
            ["Thick as wool. Then rain."]
        );
        assert_eq!(paragraphs("<p><sup>*</sup>Starred.</p>"), ["Starred."]);
        // A superscript that links somewhere is a note marker, whatever precedes it.
        assert_eq!(
            paragraphs("<p>As shown<sup><a href=\"#n3\">3</a></sup> here.</p>"),
            ["As shown here."]
        );
        // Text that is not a marker is always read.
        assert_eq!(paragraphs("<p>The 1<sup>st</sup> keeper.</p>"), ["The 1st keeper."]);
    }

    #[test]
    fn markers() {
        assert!(is_marker(" 12 "));
        assert!(is_marker("*"));
        assert!(!is_marker("th"));
        assert!(!is_marker(""));
    }
}
