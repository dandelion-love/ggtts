//! The chapter model: a book's metadata and its chapters in reading order.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{json, Value};

use crate::archive::Archive;
use crate::dom::parse;
use crate::error::BookError;
use crate::package::{check_drm, container, package, Pointer};
use crate::toc::{self, TocSource};
use crate::xhtml::{extract, Block, Item};

/// The structural types that mark front or back matter a listener usually skips.
pub const SKIPPABLE_TYPES: &[&str] = &[
    "cover",
    "titlepage",
    "title-page",
    "halftitlepage",
    "copyright-page",
    "copyright",
    "toc",
    "contents",
    "loi",
    "lot",
    "index",
    "colophon",
    "imprint",
    "errata",
    "landmarks",
    "page-list",
];

/// A book read for speaking.
#[derive(Debug, Clone, PartialEq)]
pub struct Book {
    /// The package's unique identifier (`dc:identifier` named by `unique-identifier`), if it has one.
    pub identifier: Option<String>,
    pub title: String,
    pub authors: Vec<String>,
    pub language: Option<String>,
    /// Where the chapters came from.
    pub toc_source: TocSource,
    pub chapters: Vec<Chapter>,
    /// Things that were odd but not fatal: a table-of-contents entry that points nowhere, say.
    pub warnings: Vec<String>,
}

/// One chapter, in reading order.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Stable across reads of the same book: the spine item's id, plus `#fragment` when the chapter starts inside
    /// a file.
    pub id: String,
    /// The table of contents' title, else the chapter's first heading, else empty.
    pub title: String,
    /// 0 for a top-level table-of-contents entry, 1 for its children, and so on.
    pub depth: usize,
    /// Front or back matter: kept, but marked so a renderer can leave it out.
    pub skippable: bool,
    /// Why it is skippable, for example `epub:type copyright-page` or `spine linear=no`.
    pub skip_reason: Option<String>,
    /// The content files the chapter reads from, in spine order.
    pub files: Vec<String>,
    pub blocks: Vec<Block>,
}

impl Book {
    /// Read an EPUB file.
    pub fn open(path: &Path) -> Result<Book, BookError> {
        let bytes = std::fs::read(path).map_err(|e| BookError::Io(format!("{}: {e}", path.display())))?;
        Book::read(bytes)
    }

    /// Read an EPUB from its bytes.
    ///
    /// The checks run in this order: zip, DRM, `container.xml`, the package, the spine, then the content.
    pub fn read(bytes: Vec<u8>) -> Result<Book, BookError> {
        let mut zip = Archive::new(bytes)?;
        check_drm(&mut zip)?;
        let opf_path = container(&mut zip)?;
        let opf = zip.read(&opf_path)?.unwrap_or_default();
        let pkg = package(&opf_path, &opf)?;
        let mut warnings = Vec::new();

        struct SpineFile {
            idref: String,
            path: String,
            linear: bool,
            is_nav: bool,
        }
        let mut spine = Vec::new();
        for r in &pkg.spine {
            let item = pkg
                .manifest
                .iter()
                .find(|m| m.id == r.idref)
                .ok_or_else(|| BookError::MissingSpineItem {
                    idref: r.idref.clone(),
                    href: None,
                })?;
            let path = item
                .path
                .clone()
                .filter(|p| zip.has(p))
                .ok_or_else(|| BookError::MissingSpineItem {
                    idref: r.idref.clone(),
                    href: Some(item.path.clone().unwrap_or_else(|| item.href.clone())),
                })?;
            spine.push(SpineFile {
                idref: r.idref.clone(),
                path,
                linear: r.linear,
                is_nav: item.properties.iter().any(|p| p == "nav"),
            });
        }

        let toc = toc::load(&mut zip, &pkg, &mut warnings)?;
        let in_spine: HashSet<&str> = spine.iter().map(|s| s.path.as_str()).collect();
        let mut starts: HashMap<(String, Option<String>), usize> = HashMap::new();
        for (i, entry) in toc.entries.iter().enumerate() {
            let key = (entry.file.clone(), entry.fragment.clone());
            if !in_spine.contains(entry.file.as_str()) {
                warnings.push(format!(
                    "table of contents entry {:?} points at {}, which is not in the spine",
                    entry.title, entry.file
                ));
            } else if let Entry::Vacant(slot) = starts.entry(key) {
                slot.insert(i);
            } else {
                warnings.push(format!(
                    "table of contents entry {:?} points at the same place as an earlier entry",
                    entry.title
                ));
            }
        }
        let pointers: Vec<&Pointer> = toc.landmarks.iter().chain(pkg.guide.iter()).collect();

        let mut chapters: Vec<Chapter> = Vec::new();
        let mut from_toc: Vec<Option<usize>> = Vec::new();
        let mut opened: HashSet<usize> = HashSet::new();
        for sf in &spine {
            let bytes = zip.read(&sf.path)?.unwrap_or_default();
            let root = parse(&bytes).map_err(|reason| BookError::Content {
                file: sf.path.clone(),
                reason,
            })?;
            let doc = extract(&root);
            let own = skip_reason(&doc.start_types, &pointers, &sf.path, None, sf.linear, sf.is_nav);
            let file_start = starts.get(&(sf.path.clone(), None)).copied();
            let every_file = toc.source == TocSource::Spine;
            if file_start.is_some() || every_file || chapters.is_empty() || own.is_some() {
                // A file the table of contents names starts a chapter. So does the first file, and an unnamed file
                // that is itself front or back matter. Any other unnamed file continues the chapter before it: the
                // second half of a chapter split across two files. With no table of contents, every file is a
                // chapter.
                chapters.push(new_chapter(&toc.entries, file_start, sf.idref.clone(), own));
                from_toc.push(file_start);
                opened.extend(file_start);
            }
            for item in doc.items {
                match item {
                    Item::Anchor { id, types } => {
                        let Some(&entry) = starts.get(&(sf.path.clone(), Some(id.clone()))) else {
                            continue;
                        };
                        let reason = skip_reason(&types, &pointers, &sf.path, Some(&id), sf.linear, false);
                        chapters.push(new_chapter(
                            &toc.entries,
                            Some(entry),
                            format!("{}#{id}", sf.idref),
                            reason,
                        ));
                        from_toc.push(Some(entry));
                        opened.insert(entry);
                    }
                    Item::Block(block) => {
                        let current = chapters.last_mut().expect("a chapter is open before any block");
                        current.blocks.push(block);
                    }
                }
                let current = chapters.last_mut().expect("a chapter is open");
                if current.files.last() != Some(&sf.path) {
                    current.files.push(sf.path.clone());
                }
            }
            let current = chapters.last_mut().expect("a chapter is open");
            if current.files.last() != Some(&sf.path) {
                current.files.push(sf.path.clone());
            }
        }

        for (i, entry) in toc.entries.iter().enumerate() {
            if starts.values().any(|&s| s == i) && !opened.contains(&i) {
                warnings.push(format!(
                    "table of contents entry {:?} points at #{} in {}, which that file does not have",
                    entry.title,
                    entry.fragment.as_deref().unwrap_or(""),
                    entry.file
                ));
            }
        }
        let mut kept = Vec::new();
        for (mut chapter, entry) in chapters.into_iter().zip(from_toc) {
            if chapter.blocks.is_empty() {
                if let Some(i) = entry {
                    warnings.push(format!(
                        "table of contents entry {:?} has no text of its own before the next chapter starts",
                        toc.entries[i].title
                    ));
                }
                continue;
            }
            if chapter.title.is_empty() {
                if let Some(Block::Heading { text, .. }) =
                    chapter.blocks.iter().find(|b| matches!(b, Block::Heading { .. }))
                {
                    chapter.title = text.clone();
                }
            }
            kept.push(chapter);
        }

        Ok(Book {
            identifier: pkg.identifier,
            title: pkg.title,
            authors: pkg.authors,
            language: pkg.language,
            toc_source: toc.source,
            chapters: kept,
            warnings,
        })
    }

    /// The chapter model as JSON, the shape `ggtts-book inspect` prints.
    pub fn to_json(&self) -> Value {
        json!({
            "format": crate::CHAPTERS_FORMAT,
            "identifier": self.identifier,
            "title": self.title,
            "authors": self.authors,
            "language": self.language,
            "toc_source": self.toc_source.as_str(),
            "chapters": self.chapters.iter().enumerate().map(|(i, c)| json!({
                "index": i,
                "id": c.id,
                "title": c.title,
                "depth": c.depth,
                "skippable": c.skippable,
                "skip_reason": c.skip_reason,
                "files": c.files,
                "blocks": c.blocks.iter().map(|b| match b {
                    Block::Heading { level, text } => json!({"kind": "heading", "level": level, "text": text}),
                    Block::Paragraph { text } => json!({"kind": "paragraph", "text": text}),
                }).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "warnings": self.warnings,
        })
    }
}

fn new_chapter(entries: &[toc::Entry], entry: Option<usize>, id: String, reason: Option<String>) -> Chapter {
    let (title, depth) = entry
        .map(|i| (entries[i].title.clone(), entries[i].depth))
        .unwrap_or_default();
    Chapter {
        id,
        title,
        depth,
        skippable: reason.is_some(),
        skip_reason: reason,
        files: Vec::new(),
        blocks: Vec::new(),
    }
}

/// The first reason a chapter starting here is front or back matter, or `None`.
fn skip_reason(
    types: &[String],
    pointers: &[&Pointer],
    file: &str,
    fragment: Option<&str>,
    linear: bool,
    is_nav: bool,
) -> Option<String> {
    if let Some(t) = types.iter().find(|t| SKIPPABLE_TYPES.contains(&t.as_str())) {
        return Some(format!("epub:type {t}"));
    }
    for p in pointers
        .iter()
        .filter(|p| p.file == file && p.fragment.as_deref() == fragment)
    {
        if let Some(t) = p.types.iter().find(|t| SKIPPABLE_TYPES.contains(&t.as_str())) {
            return Some(format!("{} {t}", p.source));
        }
    }
    if is_nav {
        return Some("the navigation document".to_string());
    }
    if !linear {
        return Some("spine linear=no".to_string());
    }
    None
}
