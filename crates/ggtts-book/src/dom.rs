//! A small element tree over quick-xml: enough for container.xml, the OPF, the NCX and XHTML.
//!
//! Books are small, so each document is read whole. The parser refuses what is not well-formed (a mismatched or
//! missing end tag, two root elements, no root) rather than guessing at it.

use quick_xml::escape::resolve_html5_entity;
use quick_xml::events::{BytesStart, BytesText, Event};
use quick_xml::reader::Reader;

#[derive(Debug, Clone)]
pub(crate) enum Node {
    Element(Element),
    Text(String),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Element {
    /// The name as written, with its prefix: `dc:title`.
    pub qname: String,
    /// The name without its prefix: `title`.
    pub local: String,
    /// Attributes by their name as written, values unescaped.
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

impl Element {
    /// The attribute with exactly this (unprefixed) name.
    pub(crate) fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// The structural semantics: the space-separated values of a prefixed `type` attribute (`epub:type`), lower-cased.
    pub(crate) fn epub_types(&self) -> Vec<String> {
        self.attrs
            .iter()
            .filter(|(k, _)| matches!(k.split_once(':'), Some((prefix, "type")) if prefix != "xmlns"))
            .flat_map(|(_, v)| v.split_whitespace().map(str::to_ascii_lowercase))
            .collect()
    }

    pub(crate) fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }

    pub(crate) fn child(&self, local: &str) -> Option<&Element> {
        self.elements().find(|e| e.local == local)
    }

    /// Every descendant element with this local name, in document order (this element included).
    pub(crate) fn descendants<'a>(&'a self, local: &str, out: &mut Vec<&'a Element>) {
        if self.local == local {
            out.push(self);
        }
        for e in self.elements() {
            e.descendants(local, out);
        }
    }

    pub(crate) fn find_all(&self, local: &str) -> Vec<&Element> {
        let mut out = Vec::new();
        self.descendants(local, &mut out);
        out
    }

    /// All the text inside, concatenated as written.
    pub(crate) fn text(&self) -> String {
        let mut out = String::new();
        self.text_into(&mut out);
        out
    }

    fn text_into(&self, out: &mut String) {
        for n in &self.children {
            match n {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => e.text_into(out),
            }
        }
    }
}

/// The most internal entities a document's DTD may declare, and the longest value one may have, in bytes.
const MAX_DTD_ENTITIES: usize = 256;
const MAX_DTD_ENTITY_BYTES: usize = 1024;

/// Parse one UTF-8 XML document into its root element, refusing elements nested deeper than `max_depth`.
///
/// The depth limit keeps every walk over the tree (they recurse once per level) far from the stack's end, on a
/// 1 MiB Windows main thread too. Internal entities a DTD declares (`<!ENTITY c "Company">`) are spoken as their
/// value, taken literally: a value is never expanded again, so nested entities cannot multiply (no "billion
/// laughs"), and external (`SYSTEM`, `PUBLIC`) and parameter entities are ignored, so nothing is fetched.
pub(crate) fn parse(bytes: &[u8], max_depth: usize) -> Result<Element, String> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    if bytes.starts_with(b"\xff\xfe") || bytes.starts_with(b"\xfe\xff") {
        return Err("UTF-16 text is not supported, only UTF-8".to_string());
    }
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8: {e}"))?;
    let mut reader = Reader::from_str(text);
    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    let mut entities: Vec<(String, String)> = Vec::new();
    let too_deep = || format!("elements are nested deeper than {max_depth} levels");
    loop {
        let event = reader
            .read_event()
            .map_err(|e| format!("not well-formed XML at byte {}: {e}", reader.error_position()))?;
        match event {
            Event::Start(e) => {
                if stack.len() >= max_depth {
                    return Err(too_deep());
                }
                stack.push(element(&e)?)
            }
            Event::Empty(e) => {
                if stack.len() >= max_depth {
                    return Err(too_deep());
                }
                attach(&mut stack, &mut root, Node::Element(element(&e)?))?
            }
            Event::End(_) => {
                let done = stack.pop().ok_or("an end tag with no start tag")?;
                attach(&mut stack, &mut root, Node::Element(done))?;
            }
            Event::DocType(d) => entities = dtd_entities(&String::from_utf8_lossy(&d)),
            Event::Text(t) => {
                if let Some(top) = stack.last_mut() {
                    top.children.push(Node::Text(text_of(&t, &entities)));
                }
            }
            Event::CData(c) => {
                if let Some(top) = stack.last_mut() {
                    top.children.push(Node::Text(String::from_utf8_lossy(&c).into_owned()));
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if let Some(open) = stack.last() {
        return Err(format!("not well-formed XML: <{}> is never closed", open.qname));
    }
    root.ok_or_else(|| "not well-formed XML: no root element".to_string())
}

fn attach(stack: &mut [Element], root: &mut Option<Element>, node: Node) -> Result<(), String> {
    match stack.last_mut() {
        Some(parent) => parent.children.push(node),
        None => {
            if root.is_some() {
                return Err("not well-formed XML: more than one root element".to_string());
            }
            if let Node::Element(e) = node {
                *root = Some(e);
            }
        }
    }
    Ok(())
}

fn element(start: &BytesStart<'_>) -> Result<Element, String> {
    let mut attrs = Vec::new();
    for attr in start.attributes() {
        let attr = attr.map_err(|e| format!("not well-formed XML: a bad attribute: {e}"))?;
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        let value = match attr.unescape_value() {
            Ok(v) => v.into_owned(),
            Err(_) => String::from_utf8_lossy(&attr.value).into_owned(),
        };
        attrs.push((key, value));
    }
    Ok(Element {
        qname: String::from_utf8_lossy(start.name().as_ref()).into_owned(),
        local: String::from_utf8_lossy(start.local_name().as_ref()).into_owned(),
        attrs,
        children: Vec::new(),
    })
}

/// Text with its character references, the HTML5 named entities and the DTD's internal entities resolved. An entity
/// nothing declares is kept as written rather than failing the book.
fn text_of(t: &BytesText<'_>, entities: &[(String, String)]) -> String {
    let resolve = |name: &str| -> Option<&str> {
        resolve_html5_entity(name).or_else(|| entities.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str()))
    };
    match t.unescape_with(resolve) {
        Ok(s) => s.into_owned(),
        Err(_) => String::from_utf8_lossy(t).into_owned(),
    }
}

/// The internal general entities declared in a DOCTYPE's internal subset: `<!ENTITY name "value">`. Parameter
/// entities (`%`) and external ones (`SYSTEM`, `PUBLIC`) are skipped; at most [`MAX_DTD_ENTITIES`], each value at
/// most [`MAX_DTD_ENTITY_BYTES`].
fn dtd_entities(doctype: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = doctype;
    while let Some(at) = rest.find("<!ENTITY") {
        rest = &rest[at + "<!ENTITY".len()..];
        let decl = rest.trim_start();
        if decl.starts_with('%') {
            continue;
        }
        let name_len = decl.find(char::is_whitespace).unwrap_or(decl.len());
        let (name, after) = decl.split_at(name_len);
        let after = after.trim_start();
        let Some(quote) = after.chars().next().filter(|q| *q == '"' || *q == '\'') else {
            continue; // SYSTEM, PUBLIC, or malformed: never fetched, never spoken
        };
        let Some(end) = after[1..].find(quote) else {
            break;
        };
        let value = &after[1..1 + end];
        if !name.is_empty() && value.len() <= MAX_DTD_ENTITY_BYTES && out.len() < MAX_DTD_ENTITIES {
            out.push((name.to_string(), value.to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = crate::archive::MAX_DEPTH;

    #[test]
    fn refuses_what_is_not_well_formed() {
        assert!(parse(b"<a><b></a>", D).is_err());
        assert!(parse(b"<a><b>", D).is_err());
        assert!(parse(b"<a/><b/>", D).is_err());
        assert!(parse(b"", D).is_err());
        assert!(parse(b"\xff\xfe<\x00a\x00/\x00>\x00", D).is_err());
        assert!(parse(b"<a>\xff</a>", D).is_err());
    }

    fn nested(depth: usize, empty_leaf: bool) -> String {
        let leaf = if empty_leaf { "<br/>" } else { "x" };
        format!("{}{leaf}{}", "<s>".repeat(depth), "</s>".repeat(depth))
    }

    #[test]
    fn nesting_is_limited_with_a_named_error() {
        assert!(parse(nested(4, false).as_bytes(), 4).is_ok());
        assert!(parse(nested(3, true).as_bytes(), 4).is_ok());
        let too_deep = "elements are nested deeper than 4 levels";
        assert_eq!(parse(nested(5, false).as_bytes(), 4).unwrap_err(), too_deep);
        assert_eq!(parse(nested(4, true).as_bytes(), 4).unwrap_err(), too_deep);
    }

    #[test]
    fn dtd_entities_are_spoken_literally_and_never_fetched_or_multiplied() {
        let doc = r#"<!DOCTYPE html [
  <!ENTITY c "Company">
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;">
  <!ENTITY % param "x">
  <!ENTITY ext SYSTEM "file:///etc/passwd">
]><p>&c; &lol2; &ext; &nbsp;</p>"#;
        let root = parse(doc.as_bytes(), D).unwrap();
        // &lol2; is its value as written, never expanded again; &ext; is not resolved, so the text stays literal.
        assert_eq!(root.text(), "&c; &lol2; &ext; &nbsp;");
        let doc = r#"<!DOCTYPE html [<!ENTITY c "Company"><!ENTITY lol2 "&lol;&lol;">]><p>&c; &lol2;&nbsp;!</p>"#;
        assert_eq!(parse(doc.as_bytes(), D).unwrap().text(), "Company &lol;&lol;\u{a0}!");
    }

    #[test]
    fn reads_entities_types_and_text() {
        let root = parse(
            "<p xmlns:epub='x' epub:type='Cover  titlepage'>a&nbsp;b &amp; c&#x4ECA;</p>".as_bytes(),
            D,
        )
        .unwrap();
        assert_eq!(root.epub_types(), ["cover", "titlepage"]);
        assert_eq!(root.text(), "a\u{a0}b & c今");
        assert_eq!(root.attr("type"), None);
    }
}
