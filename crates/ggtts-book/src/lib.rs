//! ggtts-book: an EPUB book, read into chapters a TTS engine can speak.
//!
//! A layer on top of the ggtts engines, never inside them. It reads the zip container, `META-INF/container.xml`, the
//! package document's metadata, manifest and spine, and the table of contents (the EPUB 3 nav document, else the
//! EPUB 2 NCX, else one chapter per spine file), and turns each chapter's XHTML into headings and paragraphs.
//! Front and back matter are kept and marked [skippable](Chapter::skippable).
//!
//! A DRM-protected book is refused ([`BookError::Drm`]); it is never decrypted or stripped. So are a file that is
//! not a zip, a missing or malformed container, and a spine item whose file is absent. See `docs/book.md`.
//!
//! [`plan`] chunks each chapter with the engine's own chunker at its profile's ceiling and gives every chunk a
//! seed fixed by its place and text; [`manifest`] records a chapter's progress so a killed render resumes.

mod archive;
mod book;
mod dom;
mod error;
pub mod manifest;
mod package;
pub mod plan;
mod toc;
pub mod xhtml;

pub use archive::{Limits, MAX_BOOK_BYTES, MAX_DEPTH, MAX_ENTRY_BYTES};
pub use book::{Book, Chapter, SKIPPABLE_TYPES};
pub use error::BookError;
pub use toc::TocSource;
pub use xhtml::Block;

/// The `format` of the chapter model JSON `ggtts-book inspect` prints. A reader checks it before trusting the rest.
pub const CHAPTERS_FORMAT: &str = "ggtts-book-chapters/1";
