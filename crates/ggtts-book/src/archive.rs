//! The zip container, and the paths inside it.

use std::io::{Cursor, Read};

use crate::error::BookError;

/// The largest single entry ggtts-book reads by default, in bytes. A zip bomb is refused rather than inflated.
pub const MAX_ENTRY_BYTES: u64 = 64 << 20;
/// The most a book may inflate to in total by default, in bytes, every read counted. A book whose entries are each
/// under [`MAX_ENTRY_BYTES`] but which reads one large file many times is refused at this cap, not inflated.
pub const MAX_BOOK_BYTES: u64 = 256 << 20;
/// The deepest element nesting any document may have by default. The tree walks recurse once per level, so a deeper
/// document is refused with a named error instead of overflowing the stack.
pub const MAX_DEPTH: usize = 256;

/// The resource limits a book is read under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The largest single entry, inflated, in bytes.
    pub entry_bytes: u64,
    /// The most all reads of the book may inflate to together, in bytes.
    pub book_bytes: u64,
    /// The deepest element nesting in any XML document.
    pub depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            entry_bytes: MAX_ENTRY_BYTES,
            book_bytes: MAX_BOOK_BYTES,
            depth: MAX_DEPTH,
        }
    }
}

pub(crate) struct Archive {
    zip: zip::ZipArchive<Cursor<Vec<u8>>>,
    pub(crate) limits: Limits,
    /// Bytes inflated so far, every read counted.
    total: u64,
}

impl Archive {
    pub(crate) fn new(bytes: Vec<u8>, limits: Limits) -> Result<Self, BookError> {
        zip::ZipArchive::new(Cursor::new(bytes))
            .map(|zip| Self { zip, limits, total: 0 })
            .map_err(|e| BookError::NotZip(e.to_string()))
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.zip.index_for_name(name).is_some()
    }

    /// Parse an XML document from the book under the book's depth limit.
    pub(crate) fn parse(&self, bytes: &[u8]) -> Result<crate::dom::Element, String> {
        crate::dom::parse(bytes, self.limits.depth)
    }

    /// The entry's bytes, or `None` when the archive has no entry of that name. A zip-level encrypted entry is
    /// DRM, and is refused as such.
    pub(crate) fn read(&mut self, name: &str) -> Result<Option<Vec<u8>>, BookError> {
        let Some(index) = self.zip.index_for_name(name) else {
            return Ok(None);
        };
        let zip_error = |e: zip::result::ZipError| BookError::Zip {
            file: name.to_string(),
            reason: e.to_string(),
        };
        if self.zip.by_index_raw(index).map_err(zip_error)?.encrypted() {
            return Err(BookError::Drm {
                files: vec![name.to_string()],
            });
        }
        let Limits {
            entry_bytes,
            book_bytes,
            ..
        } = self.limits;
        let entry = self.zip.by_index(index).map_err(zip_error)?;
        if entry.size() > entry_bytes {
            return Err(BookError::TooLarge { file: name.to_string() });
        }
        // Never inflate past what the book's budget has left, whatever the entry's header claims.
        let budget = book_bytes.saturating_sub(self.total);
        let mut out = Vec::new();
        entry
            .take(entry_bytes.min(budget) + 1)
            .read_to_end(&mut out)
            .map_err(|e| BookError::Zip {
                file: name.to_string(),
                reason: e.to_string(),
            })?;
        if out.len() as u64 > entry_bytes {
            return Err(BookError::TooLarge { file: name.to_string() });
        }
        self.total += out.len() as u64;
        if self.total > book_bytes {
            return Err(BookError::BookTooLarge { limit: book_bytes });
        }
        Ok(Some(out))
    }
}

/// Resolve `href`, as written in `base_file`, to a path inside the archive and an optional fragment.
///
/// Percent-escapes are decoded, `.` and `..` are applied, and a leading `/` means the archive root. An href with a
/// scheme (`https://`, `mailto:`, `data:`), or one that climbs above the root, is not in the book: `None`.
pub(crate) fn resolve(base_file: &str, href: &str) -> Option<(String, Option<String>)> {
    let href = href.trim();
    if href.contains("://") || href.starts_with("mailto:") || href.starts_with("data:") {
        return None;
    }
    let (path, fragment) = match href.split_once('#') {
        Some((path, fragment)) => (path, Some(percent_decode(fragment)).filter(|f| !f.is_empty())),
        None => (href, None),
    };
    let path = percent_decode(path);
    let joined = if path.is_empty() {
        base_file.to_string()
    } else if let Some(rooted) = path.strip_prefix('/') {
        rooted.to_string()
    } else {
        format!("{}{}", dir_of(base_file), path)
    };
    Some((normalize_path(&joined)?, fragment))
}

fn dir_of(file: &str) -> &str {
    match file.rfind('/') {
        Some(i) => &file[..=i],
        None => "",
    }
}

fn normalize_path(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_rooted_escaped_and_fragment_hrefs() {
        assert_eq!(
            resolve("OEBPS/Text/nav.xhtml", "ch1.xhtml"),
            Some(("OEBPS/Text/ch1.xhtml".into(), None))
        );
        assert_eq!(
            resolve("OEBPS/Text/nav.xhtml", "../Text/part%202.xhtml#c3"),
            Some(("OEBPS/Text/part 2.xhtml".into(), Some("c3".into())))
        );
        assert_eq!(
            resolve("OEBPS/toc.ncx", "#top"),
            Some(("OEBPS/toc.ncx".into(), Some("top".into())))
        );
        assert_eq!(
            resolve("OEBPS/content.opf", "/x/y.xhtml"),
            Some(("x/y.xhtml".into(), None))
        );
        assert_eq!(resolve("a.opf", "../../escape.xhtml"), None);
        assert_eq!(resolve("a.opf", "https://example.org/x.xhtml"), None);
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%E4%BB%8A"), "今");
    }
}
