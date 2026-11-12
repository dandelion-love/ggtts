//! Story 1, AC5: each refusal has its own error. Every case is the working fixture with one defect planted, and
//! `the_unplanted_fixtures_read` shows the same fixture passing without it.

mod common;

use common::*;
use ggtts_book::{Book, BookError, Limits};

fn refused(epub: &Epub) -> BookError {
    match Book::read(epub.bytes()) {
        Ok(book) => panic!(
            "the book should be refused, but read as {} chapters",
            book.chapters.len()
        ),
        Err(e) => e,
    }
}

#[test]
fn the_unplanted_fixtures_read() {
    for epub in [nested(), ncx_only(), spine_only()] {
        Book::read(epub.bytes()).unwrap();
    }
}

/// Fan-out: fifty manifest items naming one file, all in the spine. The file is read and spoken once, with a
/// warning per repeat, so a small zip cannot be multiplied into a huge book.
#[test]
fn a_file_the_spine_lists_many_times_is_read_once() {
    let epub = nested();
    let mut opf = epub.text("OEBPS/content.opf");
    let items: String = (0..50)
        .map(|i| format!(r#"<item id="again{i}" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>"#))
        .collect();
    let refs: String = (0..50).map(|i| format!(r#"<itemref idref="again{i}"/>"#)).collect();
    opf = opf.replace("</manifest>", &format!("{items}</manifest>"));
    opf = opf.replace("</spine>", &format!("{refs}</spine>"));
    let book = Book::read(epub.with("OEBPS/content.opf", &opf).bytes()).unwrap();
    let spoken = book
        .chapters
        .iter()
        .flat_map(|c| &c.blocks)
        .filter(|b| b.text() == "The keeper counted the gulls before breakfast.")
        .count();
    assert_eq!(spoken, 1);
    assert_eq!(book.warnings.iter().filter(|w| w.contains("already lists")).count(), 50);
}

/// The book-wide budget: every read counts, and a book that inflates past the budget is refused with its own code,
/// even when each entry is under the per-entry cap.
#[test]
fn a_book_past_its_total_budget_is_refused() {
    let bytes = nested().bytes();
    let tight = Limits {
        book_bytes: 4_000,
        ..Limits::default()
    };
    let err = Book::read_with_limits(bytes.clone(), tight).unwrap_err();
    assert_eq!(err, BookError::BookTooLarge { limit: 4_000 });
    assert_eq!(err.code(), "book-too-large");
    // The same book under the defaults reads.
    Book::read_with_limits(bytes, Limits::default()).unwrap();
}

/// The per-entry cap has its own code.
#[test]
fn an_entry_past_its_cap_is_refused() {
    let tight = Limits {
        entry_bytes: 600,
        ..Limits::default()
    };
    let err = Book::read_with_limits(nested().bytes(), tight).unwrap_err();
    assert_eq!(err.code(), "too-large");
}

fn deep_chapter(depth: usize) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head><body><p>{}deep{}</p></body></html>"#,
        "<span>".repeat(depth),
        "</span>".repeat(depth)
    )
}

/// Deep nesting is refused with a named error, not a stack overflow: 50,000 nested spans, read on a thread with a
/// 1 MiB stack, the size of a Windows main thread.
#[test]
fn deep_nesting_is_refused_not_a_crash() {
    let epub = nested().with("OEBPS/Text/ch1.xhtml", &deep_chapter(50_000));
    let err = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || Book::read(epub.bytes()))
        .unwrap()
        .join()
        .expect("the reader must not crash")
        .unwrap_err();
    assert_eq!(err.code(), "content");
    assert!(err.to_string().contains("nested deeper than 256 levels"), "{err}");
}

/// Up to the limit, deep nesting reads, on the same 1 MiB stack.
#[test]
fn nesting_up_to_the_limit_reads_on_a_small_stack() {
    // html > body > p > 250 spans: 253 levels.
    let epub = nested().with("OEBPS/Text/ch1.xhtml", &deep_chapter(250));
    let book = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || Book::read(epub.bytes()))
        .unwrap()
        .join()
        .expect("the reader must not crash")
        .unwrap();
    let ch1 = book.chapters.iter().find(|c| c.id == "ch1").unwrap();
    assert_eq!(ch1.blocks[0].text(), "deep");
}

/// Font obfuscation is accepted only on a font. On a content document it is refused as DRM.
#[test]
fn font_obfuscation_on_a_content_file_is_refused() {
    let epub = nested().with(
        "META-INF/encryption.xml",
        &encryption_xml("OEBPS/Text/ch1.xhtml", IDPF_FONT_OBFUSCATION),
    );
    assert_eq!(
        refused(&epub),
        BookError::Drm {
            files: vec!["OEBPS/Text/ch1.xhtml".into()]
        }
    );
}

/// (a) DRM: encryption.xml naming a content document is refused, naming the file, and nothing is decrypted.
#[test]
fn a_drm_protected_book_is_refused() {
    let epub = nested().with("META-INF/encryption.xml", &encryption_xml("OEBPS/Text/ch1.xhtml", AES));
    let err = refused(&epub);
    assert_eq!(
        err,
        BookError::Drm {
            files: vec!["OEBPS/Text/ch1.xhtml".into()]
        }
    );
    assert_eq!(err.code(), "drm-protected");
    assert!(err.to_string().contains("does not decrypt or remove DRM"), "{err}");
}

/// A content file encrypted with a zip password is refused as DRM too, before anything tries to read it.
#[test]
fn a_zip_encrypted_content_file_is_refused() {
    let err = Book::read(nested().bytes_encrypting("OEBPS/Text/ch1.xhtml")).unwrap_err();
    assert_eq!(
        err,
        BookError::Drm {
            files: vec!["OEBPS/Text/ch1.xhtml".into()]
        }
    );
}

/// Font obfuscation scrambles an embedded font, not the text: it is not DRM, and the book reads.
#[test]
fn font_obfuscation_is_not_drm() {
    let epub = nested().with(
        "META-INF/encryption.xml",
        &encryption_xml("OEBPS/Fonts/serif.otf", IDPF_FONT_OBFUSCATION),
    );
    assert_eq!(Book::read(epub.bytes()).unwrap(), Book::read(nested().bytes()).unwrap());
}

/// An encryption.xml that cannot be read may hide encryption: refused, not guessed at.
#[test]
fn an_unreadable_encryption_xml_is_refused() {
    let err = refused(&nested().with("META-INF/encryption.xml", "<encryption><EncryptedData>"));
    assert_eq!(err.code(), "drm-protected");
}

/// (b) The container: missing, malformed, naming no package, or naming a package that is absent.
#[test]
fn a_missing_container_is_refused() {
    let err = refused(&nested().without("META-INF/container.xml"));
    assert_eq!(err, BookError::Container("missing".into()));
}

#[test]
fn a_malformed_container_is_refused() {
    let err = refused(&nested().with("META-INF/container.xml", "<container><rootfiles></container>"));
    assert_eq!(err.code(), "container");
    assert!(err.to_string().contains("not well-formed"), "{err}");
}

#[test]
fn a_container_naming_no_package_is_refused() {
    let err = refused(&nested().with("META-INF/container.xml", "<container><rootfiles/></container>"));
    assert_eq!(err.code(), "container");
    assert!(err.to_string().contains("no <rootfile>"), "{err}");
}

#[test]
fn a_container_naming_an_absent_package_is_refused() {
    let err = refused(&nested().without("OEBPS/content.opf"));
    assert_eq!(
        err,
        BookError::Container("names package document OEBPS/content.opf, which is not in the book".into())
    );
}

/// (c) A spine item whose file is absent, or whose id the manifest does not list.
#[test]
fn an_absent_spine_file_is_refused() {
    let err = refused(&nested().without("OEBPS/Text/ch2b.xhtml"));
    assert_eq!(
        err,
        BookError::MissingSpineItem {
            idref: "ch2b".into(),
            href: Some("OEBPS/Text/ch2b.xhtml".into())
        }
    );
}

#[test]
fn a_spine_id_missing_from_the_manifest_is_refused() {
    let epub = nested();
    let opf = epub
        .text("OEBPS/content.opf")
        .replace(r#"<itemref idref="ch1"/>"#, r#"<itemref idref="ch9"/>"#);
    let err = refused(&epub.with("OEBPS/content.opf", &opf));
    assert_eq!(
        err,
        BookError::MissingSpineItem {
            idref: "ch9".into(),
            href: None
        }
    );
}

/// (d) Not a zip at all.
#[test]
fn a_file_that_is_not_a_zip_is_refused() {
    let err = Book::read(b"this is not a book".to_vec()).unwrap_err();
    assert_eq!(err.code(), "not-zip");
}

/// A package with no spine, and a spine document that is not well-formed, each have their own error.
#[test]
fn a_package_without_a_spine_and_malformed_content_are_refused() {
    let epub = nested();
    let opf = epub.text("OEBPS/content.opf");
    let start = opf.find("<spine>").unwrap();
    let end = opf.find("</spine>").unwrap() + "</spine>".len();
    let no_spine = format!("{}{}", &opf[..start], &opf[end..]);
    let err = refused(&nested().with("OEBPS/content.opf", &no_spine));
    assert_eq!(err.code(), "package");

    let err = refused(&nested().with("OEBPS/Text/ch1.xhtml", "<html><body><p>unclosed</body></html>"));
    assert_eq!(err.code(), "content");
    assert!(err.to_string().contains("OEBPS/Text/ch1.xhtml"), "{err}");
}
