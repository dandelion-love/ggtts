//! `META-INF/encryption.xml`, `META-INF/container.xml` and the package document (the OPF).

use crate::archive::{resolve, Archive};
use crate::dom::Element;
use crate::error::BookError;
use crate::xhtml::normalize;

/// The font obfuscation algorithms. They scramble embedded fonts, not the text, and are not DRM.
const FONT_OBFUSCATION: &[&str] = &["http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"];
/// The file extensions of the fonts font obfuscation may apply to.
const FONT_EXTENSIONS: &[&str] = &["otf", "ttf", "woff", "woff2", "ttc", "otc"];

fn is_font(path: &str) -> bool {
    path.rsplit_once('.')
        .is_some_and(|(_, ext)| FONT_EXTENSIONS.iter().any(|f| f.eq_ignore_ascii_case(ext)))
}

/// Refuse a book whose `encryption.xml` encrypts anything other than a font obfuscated by a font-obfuscation
/// algorithm. A font-obfuscation algorithm on a file that is not a font is refused as DRM too, since that file cannot
/// be read as written. An `encryption.xml` that cannot be read is refused: a book that may be encrypted is not guessed
/// at.
pub(crate) fn check_drm(zip: &mut Archive) -> Result<(), BookError> {
    const FILE: &str = "META-INF/encryption.xml";
    let Some(bytes) = zip.read(FILE)? else {
        return Ok(());
    };
    let root = zip.parse(&bytes).map_err(|_| BookError::Drm {
        files: vec![format!("{FILE} (unreadable)")],
    })?;
    let mut encrypted = Vec::new();
    for data in root.find_all("EncryptedData") {
        let algorithm = data
            .child("EncryptionMethod")
            .and_then(|m| m.attr("Algorithm"))
            .unwrap_or("");
        for reference in data.find_all("CipherReference") {
            let uri = reference.attr("URI").unwrap_or("?");
            let path = resolve("", uri)
                .map(|(path, _)| path)
                .unwrap_or_else(|| uri.to_string());
            if FONT_OBFUSCATION.contains(&algorithm) && is_font(&path) {
                continue;
            }
            encrypted.push(path);
        }
    }
    if encrypted.is_empty() {
        Ok(())
    } else {
        Err(BookError::Drm { files: encrypted })
    }
}

/// The package document's path, from `META-INF/container.xml`.
pub(crate) fn container(zip: &mut Archive) -> Result<String, BookError> {
    let bytes = zip
        .read("META-INF/container.xml")?
        .ok_or_else(|| BookError::Container("missing".to_string()))?;
    let root = zip.parse(&bytes).map_err(BookError::Container)?;
    let rootfiles = root.find_all("rootfile");
    let chosen = rootfiles
        .iter()
        .find(|r| r.attr("media-type") == Some("application/oebps-package+xml"))
        .or_else(|| rootfiles.first())
        .ok_or_else(|| BookError::Container("names no package document (no <rootfile>)".to_string()))?;
    let full_path = chosen
        .attr("full-path")
        .ok_or_else(|| BookError::Container("its <rootfile> has no full-path".to_string()))?;
    let path = resolve("", full_path)
        .map(|(p, _)| p)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| BookError::Container(format!("names package document {full_path:?}, which is not a path")))?;
    if !zip.has(&path) {
        return Err(BookError::Container(format!(
            "names package document {path}, which is not in the book"
        )));
    }
    Ok(path)
}

pub(crate) struct ManifestItem {
    pub id: String,
    pub href: String,
    /// The archive path, or `None` when the href points outside the book.
    pub path: Option<String>,
    pub media_type: String,
    pub properties: Vec<String>,
}

pub(crate) struct SpineRef {
    pub idref: String,
    pub linear: bool,
}

/// A landmark or guide reference: structural types pointing at a place in the book.
#[derive(Debug, Clone)]
pub(crate) struct Pointer {
    /// Where it was declared: `landmark` or `guide`.
    pub source: &'static str,
    pub types: Vec<String>,
    pub file: String,
    pub fragment: Option<String>,
}

pub(crate) struct Package {
    pub identifier: Option<String>,
    pub title: String,
    pub authors: Vec<String>,
    pub language: Option<String>,
    pub manifest: Vec<ManifestItem>,
    pub spine: Vec<SpineRef>,
    pub spine_toc: Option<String>,
    pub guide: Vec<Pointer>,
}

pub(crate) fn package(path: &str, bytes: &[u8], zip: &mut Archive) -> Result<Package, BookError> {
    let error = |reason: String| BookError::Package {
        file: path.to_string(),
        reason,
    };
    let root = zip.parse(bytes).map_err(error)?;
    if root.local != "package" {
        return Err(error(format!("the root element is <{}>, not <package>", root.qname)));
    }
    let metadata = root.child("metadata");
    let texts = |local: &str| -> Vec<&Element> { metadata.map(|m| m.find_all(local)).unwrap_or_default() };
    let text = |e: &Element| normalize(&e.text());

    let identifiers = texts("identifier");
    let unique = root.attr("unique-identifier");
    let identifier = identifiers
        .iter()
        .find(|e| unique.is_some() && e.attr("id") == unique)
        .or_else(|| identifiers.first())
        .map(|e| text(e))
        .filter(|s| !s.is_empty());

    let manifest = root
        .child("manifest")
        .map(|m| {
            m.elements()
                .filter(|e| e.local == "item")
                .map(|e| {
                    let href = e.attr("href").unwrap_or("").to_string();
                    ManifestItem {
                        id: e.attr("id").unwrap_or("").to_string(),
                        path: resolve(path, &href).map(|(p, _)| p).filter(|p| !p.is_empty()),
                        href,
                        media_type: e.attr("media-type").unwrap_or("").to_string(),
                        properties: e
                            .attr("properties")
                            .unwrap_or("")
                            .split_whitespace()
                            .map(String::from)
                            .collect(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let spine_el = root
        .child("spine")
        .ok_or_else(|| error("it has no <spine>".to_string()))?;
    let spine: Vec<SpineRef> = spine_el
        .elements()
        .filter(|e| e.local == "itemref")
        .map(|e| SpineRef {
            idref: e.attr("idref").unwrap_or("").to_string(),
            linear: e.attr("linear") != Some("no"),
        })
        .collect();
    if spine.is_empty() {
        return Err(error("its <spine> is empty".to_string()));
    }

    let guide = root
        .child("guide")
        .map(|g| {
            g.elements()
                .filter(|e| e.local == "reference")
                .filter_map(|e| {
                    let (file, fragment) = resolve(path, e.attr("href")?)?;
                    Some(Pointer {
                        source: "guide",
                        types: e
                            .attr("type")
                            .unwrap_or("")
                            .split_whitespace()
                            .map(str::to_ascii_lowercase)
                            .collect(),
                        file,
                        fragment,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Package {
        identifier,
        title: texts("title").first().map(|e| text(e)).unwrap_or_default(),
        authors: texts("creator")
            .into_iter()
            .map(text)
            .filter(|s| !s.is_empty())
            .collect(),
        language: texts("language").first().map(|e| text(e)).filter(|s| !s.is_empty()),
        manifest,
        spine,
        spine_toc: spine_el.attr("toc").map(String::from),
        guide,
    })
}
