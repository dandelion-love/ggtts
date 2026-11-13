//! Why a book was refused. Every refusal has its own variant and a stable code, so a caller (and a test) can tell
//! a DRM-protected book from a broken one without matching on prose.

use std::fmt;

/// A book that could not be read, or must not be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookError {
    /// The file could not be read from disk.
    Io(String),
    /// The file is not a zip archive, so it is not an EPUB.
    NotZip(String),
    /// A zip entry could not be read (a damaged archive).
    Zip { file: String, reason: String },
    /// A zip entry inflates past the per-entry limit ([`Limits::entry_bytes`](crate::Limits::entry_bytes)).
    TooLarge { file: String },
    /// The book's reads together inflate past the book-wide limit ([`Limits::book_bytes`](crate::Limits::book_bytes)).
    BookTooLarge { limit: u64 },
    /// The book is DRM-protected: its content files are encrypted. ggtts-book never decrypts or strips DRM.
    Drm { files: Vec<String> },
    /// `META-INF/container.xml` is missing or malformed, or names a package document that is not in the book.
    Container(String),
    /// The package document (the OPF) is malformed, or has no spine.
    Package { file: String, reason: String },
    /// A spine item names a manifest id that does not exist, or a file that is not in the book.
    MissingSpineItem { idref: String, href: Option<String> },
    /// A content document in the spine is not well-formed XHTML.
    Content { file: String, reason: String },
}

impl BookError {
    /// A stable, machine-readable code for the refusal.
    pub fn code(&self) -> &'static str {
        match self {
            BookError::Io(_) => "io",
            BookError::NotZip(_) => "not-zip",
            BookError::Zip { .. } => "zip",
            BookError::TooLarge { .. } => "too-large",
            BookError::BookTooLarge { .. } => "book-too-large",
            BookError::Drm { .. } => "drm-protected",
            BookError::Container(_) => "container",
            BookError::Package { .. } => "package",
            BookError::MissingSpineItem { .. } => "missing-spine-item",
            BookError::Content { .. } => "content",
        }
    }
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        match self {
            BookError::Io(reason) => write!(f, "cannot read the book: {reason}"),
            BookError::NotZip(reason) => write!(f, "not an EPUB (not a zip archive): {reason}"),
            BookError::Zip { file, reason } => write!(f, "cannot read {file} from the archive: {reason}"),
            BookError::TooLarge { file } => write!(f, "{file} inflates past the per-entry limit"),
            BookError::BookTooLarge { limit } => write!(
                f,
                "the book inflates past {limit} bytes in total (a zip bomb, entities that expand to more text than \
                 the book holds, or one file read many times)"
            ),
            BookError::Drm { files } => write!(
                f,
                "the book is DRM-protected ({} encrypted: {}); ggtts-book does not decrypt or remove DRM",
                if files.len() == 1 { "one file is" } else { "files are" },
                files.join(", ")
            ),
            BookError::Container(reason) => write!(f, "META-INF/container.xml: {reason}"),
            BookError::Package { file, reason } => write!(f, "package document {file}: {reason}"),
            BookError::MissingSpineItem { idref, href: None } => {
                write!(f, "the spine names item {idref:?}, which the manifest does not list")
            }
            BookError::MissingSpineItem {
                idref,
                href: Some(href),
            } => {
                write!(
                    f,
                    "the spine names item {idref:?}, whose file {href} is not in the book"
                )
            }
            BookError::Content { file, reason } => write!(f, "content document {file}: {reason}"),
        }
    }
}

impl std::error::Error for BookError {}
