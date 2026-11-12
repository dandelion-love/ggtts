//! The table of contents: the EPUB 3 nav document, else the EPUB 2 NCX. Each entry is a place a chapter starts.

use crate::archive::{resolve, Archive};
use crate::dom::{parse, Element};
use crate::error::BookError;
use crate::package::{Package, Pointer};
use crate::xhtml::normalize;

/// Where a book's chapters came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TocSource {
    /// The EPUB 3 navigation document's `toc` nav.
    Nav,
    /// The EPUB 2 NCX.
    Ncx,
    /// Neither: one chapter per spine file.
    Spine,
}

impl TocSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TocSource::Nav => "nav",
            TocSource::Ncx => "ncx",
            TocSource::Spine => "spine",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub title: String,
    pub file: String,
    pub fragment: Option<String>,
    /// 0 for a top-level entry, 1 for its children, and so on.
    pub depth: usize,
}

pub(crate) struct Toc {
    pub source: TocSource,
    pub entries: Vec<Entry>,
    pub landmarks: Vec<Pointer>,
}

/// Read the nav document, else the NCX. A table of contents that cannot be read, or lists nothing, is a warning
/// and the next source is tried; the spine is the last resort.
pub(crate) fn load(zip: &mut Archive, pkg: &Package, warnings: &mut Vec<String>) -> Result<Toc, BookError> {
    let mut landmarks = Vec::new();
    if let Some(nav) = pkg.manifest.iter().find(|m| m.properties.iter().any(|p| p == "nav")) {
        match read_doc(zip, nav.path.as_deref())? {
            Ok((path, root)) => {
                let (entries, marks) = nav_entries(&path, &root);
                landmarks = marks;
                if !entries.is_empty() {
                    return Ok(Toc {
                        source: TocSource::Nav,
                        entries,
                        landmarks,
                    });
                }
                warnings.push(format!("the navigation document {path} lists no table of contents"));
            }
            Err(reason) => warnings.push(format!("the navigation document {} cannot be read: {reason}", nav.href)),
        }
    }
    let ncx = pkg
        .spine_toc
        .as_deref()
        .and_then(|id| pkg.manifest.iter().find(|m| m.id == id))
        .or_else(|| pkg.manifest.iter().find(|m| m.media_type == "application/x-dtbncx+xml"));
    if let Some(ncx) = ncx {
        match read_doc(zip, ncx.path.as_deref())? {
            Ok((path, root)) => {
                let entries = ncx_entries(&path, &root);
                if !entries.is_empty() {
                    return Ok(Toc {
                        source: TocSource::Ncx,
                        entries,
                        landmarks,
                    });
                }
                warnings.push(format!("the NCX {path} lists no entries"));
            }
            Err(reason) => warnings.push(format!("the NCX {} cannot be read: {reason}", ncx.href)),
        }
    }
    Ok(Toc {
        source: TocSource::Spine,
        entries: Vec::new(),
        landmarks,
    })
}

fn read_doc(zip: &mut Archive, path: Option<&str>) -> Result<Result<(String, Element), String>, BookError> {
    let Some(path) = path else {
        return Ok(Err("its href is outside the book".to_string()));
    };
    let Some(bytes) = zip.read(path)? else {
        return Ok(Err("it is not in the book".to_string()));
    };
    Ok(parse(&bytes).map(|root| (path.to_string(), root)))
}

fn nav_entries(path: &str, root: &Element) -> (Vec<Entry>, Vec<Pointer>) {
    let navs = root.find_all("nav");
    let has = |nav: &&&Element, kind: &str| {
        nav.epub_types().iter().any(|t| t == kind)
            || nav.attr("role").is_some_and(|r| r.strip_prefix("doc-") == Some(kind))
    };
    let mut entries = Vec::new();
    if let Some(toc) = navs.iter().find(|n| has(n, "toc")) {
        if let Some(ol) = toc.find_all("ol").first() {
            walk_ol(path, ol, 0, &mut entries);
        }
    }
    let mut landmarks = Vec::new();
    if let Some(marks) = navs.iter().find(|n| has(n, "landmarks")) {
        for a in marks.find_all("a") {
            if let Some((file, fragment)) = a.attr("href").and_then(|h| resolve(path, h)) {
                landmarks.push(Pointer {
                    source: "landmark",
                    types: a.epub_types(),
                    file,
                    fragment,
                });
            }
        }
    }
    (entries, landmarks)
}

fn walk_ol(path: &str, ol: &Element, depth: usize, out: &mut Vec<Entry>) {
    for li in ol.elements().filter(|e| e.local == "li") {
        let label = li.elements().find(|e| e.local == "a" || e.local == "span");
        if let Some(a) = label.filter(|e| e.local == "a") {
            if let Some((file, fragment)) = a.attr("href").and_then(|h| resolve(path, h)) {
                out.push(Entry {
                    title: normalize(&a.text()),
                    file,
                    fragment,
                    depth,
                });
            }
        }
        for nested in li.elements().filter(|e| e.local == "ol") {
            walk_ol(path, nested, depth + 1, out);
        }
    }
}

fn ncx_entries(path: &str, root: &Element) -> Vec<Entry> {
    let mut out = Vec::new();
    if let Some(map) = root.find_all("navMap").first() {
        walk_points(path, map, 0, &mut out);
    }
    out
}

fn walk_points(path: &str, parent: &Element, depth: usize, out: &mut Vec<Entry>) {
    for point in parent.elements().filter(|e| e.local == "navPoint") {
        let title = point
            .child("navLabel")
            .map(|l| normalize(&l.text()))
            .unwrap_or_default();
        if let Some((file, fragment)) = point
            .child("content")
            .and_then(|c| c.attr("src"))
            .and_then(|s| resolve(path, s))
        {
            out.push(Entry {
                title,
                file,
                fragment,
                depth,
            });
        }
        walk_points(path, point, depth + 1, out);
    }
}
