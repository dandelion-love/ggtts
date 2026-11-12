//! Fixture books, built in memory from text written for these tests (no text from any published book).
//!
//! A fixture is a list of (path, contents) pairs; [`Epub::bytes`] zips them, `mimetype` first and stored, the rest
//! deflated, as an EPUB is. Each rejection fixture is a working book with one defect planted, so the test that
//! refuses it also shows the unplanted book passing.

#![allow(dead_code)]

use std::io::Write;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

pub struct Epub {
    pub files: Vec<(String, Vec<u8>)>,
}

impl Epub {
    pub fn new(files: &[(&str, &str)]) -> Self {
        let mut all = vec![("mimetype".to_string(), b"application/epub+zip".to_vec())];
        all.extend(files.iter().map(|(p, c)| (p.to_string(), c.as_bytes().to_vec())));
        Epub { files: all }
    }

    pub fn with(mut self, path: &str, contents: &str) -> Self {
        self.files.retain(|(p, _)| p != path);
        self.files.push((path.to_string(), contents.as_bytes().to_vec()));
        self
    }

    pub fn text(&self, path: &str) -> String {
        let (_, bytes) = self
            .files
            .iter()
            .find(|(p, _)| p == path)
            .unwrap_or_else(|| panic!("no {path}"));
        String::from_utf8(bytes.clone()).unwrap()
    }

    pub fn without(mut self, path: &str) -> Self {
        self.files.retain(|(p, _)| p != path);
        self
    }

    /// The zip, with the named entry flagged as encrypted with a zip password (the archive-level kind, not
    /// encryption.xml): bit 0 of the general-purpose flags, set in its local header and its central directory record.
    pub fn bytes_encrypting(&self, encrypted: &str) -> Vec<u8> {
        let mut bytes = self.bytes();
        let name = encrypted.as_bytes();
        let u16_at = |b: &[u8], i: usize| usize::from(u16::from_le_bytes([b[i], b[i + 1]]));
        let mut flagged = 0;
        for i in 0..bytes.len().saturating_sub(46) {
            let (flags, name_len, name_at) = match &bytes[i..i + 4] {
                b"PK\x03\x04" => (i + 6, u16_at(&bytes, i + 26), i + 30),
                b"PK\x01\x02" => (i + 8, u16_at(&bytes, i + 28), i + 46),
                _ => continue,
            };
            if bytes.get(name_at..name_at + name_len) == Some(name) {
                bytes[flags] |= 1;
                flagged += 1;
            }
        }
        assert_eq!(flagged, 2, "the entry's local header and central record");
        bytes
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut zip = ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (path, contents) in &self.files {
            let method = if path == "mimetype" {
                CompressionMethod::Stored
            } else {
                CompressionMethod::Deflated
            };
            let options = SimpleFileOptions::default().compression_method(method);
            zip.start_file(path.as_str(), options).unwrap();
            zip.write_all(contents).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    /// Write the book into a fresh temporary directory and return its path.
    pub fn write_temp(&self, name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ggtts-book-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.epub"));
        std::fs::write(&path, self.bytes()).unwrap();
        path
    }
}

pub const CONTAINER: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#;

fn xhtml(body: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>x</title><style>p {{ margin: 0 }}</style></head>
{body}
</html>"#
    )
}

/// An EPUB 3 book with a nested table of contents, a chapter split across two files, two chapters that start at
/// fragments inside one file, and front and back matter.
///
/// Spine: cover (linear="no", image only), titlepage (not in the TOC), copyright, nav, part1, ch1, ch2a, ch2b
/// (not in the TOC: the second half of chapter 2), part2 (holding chapters 3 and 4 at #c3 and #c4), colophon.
pub fn nested() -> Epub {
    let opf = r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="isbn">978-0-00-000000-0</dc:identifier>
    <dc:identifier id="uid">urn:uuid:5b1d6f0e-1c1e-4b7a-9d8a-0f3e2a1b4c5d</dc:identifier>
    <dc:title>The Lighthouse Keeper&apos;s Almanac</dc:title>
    <dc:creator>Ada Test-Writer</dc:creator>
    <dc:creator>Bo Fixture</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="Text/nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="titlepage" href="Text/titlepage.xhtml" media-type="application/xhtml+xml"/>
    <item id="copyright" href="Text/copyright.xhtml" media-type="application/xhtml+xml"/>
    <item id="part1" href="Text/part1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2a" href="Text/ch2a.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2b" href="Text/ch2b.xhtml" media-type="application/xhtml+xml"/>
    <item id="part2" href="Text/part2.xhtml" media-type="application/xhtml+xml"/>
    <item id="colophon" href="Text/colophon.xhtml" media-type="application/xhtml+xml"/>
    <item id="css" href="Styles/book.css" media-type="text/css"/>
    <item id="font" href="Fonts/serif.otf" media-type="font/otf"/>
  </manifest>
  <spine>
    <itemref idref="cover" linear="no"/>
    <itemref idref="titlepage"/>
    <itemref idref="copyright"/>
    <itemref idref="nav"/>
    <itemref idref="part1"/>
    <itemref idref="ch1"/>
    <itemref idref="ch2a"/>
    <itemref idref="ch2b"/>
    <itemref idref="part2"/>
    <itemref idref="colophon"/>
  </spine>
</package>"#;
    let nav = xhtml(
        r#"<body epub:type="frontmatter">
<nav epub:type="toc" id="toc"><h1>Contents</h1>
<ol>
  <li><a href="copyright.xhtml">Copyright</a></li>
  <li><a href="part1.xhtml">Part One: The Shore</a>
    <ol>
      <li><a href="ch1.xhtml">1. Low Tide</a></li>
      <li><a href="ch2a.xhtml">2. The <em>Second</em> Lamp</a></li>
    </ol>
  </li>
  <li><a href="part2.xhtml">Part Two: The Light</a>
    <ol>
      <li><a href="part2.xhtml#c3">3. Fog</a></li>
      <li><a href="part2.xhtml#c4">4. 霧散</a></li>
    </ol>
  </li>
  <li><a href="colophon.xhtml">Colophon</a></li>
</ol>
</nav>
<nav epub:type="landmarks"><ol>
  <li><a epub:type="cover" href="cover.xhtml">Cover</a></li>
  <li><a epub:type="bodymatter" href="part1.xhtml">Start</a></li>
</ol></nav>
</body>"#,
    );
    Epub::new(&[
        ("META-INF/container.xml", CONTAINER),
        ("OEBPS/content.opf", opf),
        ("OEBPS/Text/nav.xhtml", &nav),
        (
            "OEBPS/Text/cover.xhtml",
            &xhtml(r#"<body epub:type="cover"><img src="../Images/cover.png" alt="A lighthouse"/></body>"#),
        ),
        (
            "OEBPS/Text/titlepage.xhtml",
            &xhtml(
                r#"<body epub:type="titlepage"><h1>The Lighthouse Keeper's Almanac</h1><p>Ada Test-Writer</p></body>"#,
            ),
        ),
        (
            "OEBPS/Text/copyright.xhtml",
            &xhtml(
                r#"<body><section epub:type="copyright-page"><p>Written for the ggtts-book tests.</p></section></body>"#,
            ),
        ),
        (
            "OEBPS/Text/part1.xhtml",
            &xhtml(
                r#"<body><h1>Part One: The Shore</h1><blockquote><p>Every light is a promise to a stranger.</p></blockquote></body>"#,
            ),
        ),
        (
            "OEBPS/Text/ch1.xhtml",
            &xhtml(
                r#"<body><section><h2>1. Low Tide</h2>
<p>The keeper   counted the
   gulls before breakfast.</p>
<p>There were <b>seven</b> of them&#8212;one more than yesterday&nbsp;&amp; one fewer than the day before.</p>
<script>document.write("never read")</script>
</section></body>"#,
            ),
        ),
        (
            "OEBPS/Text/ch2a.xhtml",
            &xhtml(r#"<body><h2>2. The Second Lamp</h2><p>The second lamp had not been lit in years.</p></body>"#),
        ),
        (
            "OEBPS/Text/ch2b.xhtml",
            &xhtml(r#"<body><p>She lit it anyway, and the harbour answered.</p></body>"#),
        ),
        (
            "OEBPS/Text/part2.xhtml",
            &xhtml(
                r##"<body><h1>Part Two: The Light</h1>
<section id="c3"><h2>3. Fog</h2>
<p>The fog came in at noon<a epub:type="noteref" href="#n1">1</a>, thick as wool.<sup>2</sup></p>
<p>Line one of the song<br/>line two of the song</p>
<aside epub:type="footnote" id="n1"><p>A note nobody should hear read aloud.</p></aside>
</section>
<section id="c4"><h2>4. 霧散</h2>
<p>今日<ruby>霧<rt>mou6</rt></ruby>散咗，
我哋可以出海喇。</p>
</section></body>"##,
            ),
        ),
        (
            "OEBPS/Text/colophon.xhtml",
            &xhtml(r#"<body><section epub:type="colophon"><p>Set in a serif typeface.</p></section></body>"#),
        ),
        ("OEBPS/Styles/book.css", "p { text-indent: 1em }"),
        ("OEBPS/Fonts/serif.otf", "scrambled font bytes"),
    ])
}

/// An EPUB 2 book whose only table of contents is an NCX, with a nested entry at a fragment, a percent-escaped
/// href, and a guide reference marking the title page.
pub fn ncx_only() -> Epub {
    let opf = r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>Small Hours</dc:title>
    <dc:creator opf:role="aut">C. Placeholder</dc:creator>
    <dc:identifier id="BookId">small-hours-0001</dc:identifier>
    <dc:language>en-GB</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="title" href="title.html" media-type="application/xhtml+xml"/>
    <item id="one" href="one.html" media-type="application/xhtml+xml"/>
    <item id="two" href="chapter%20two.html" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="title"/>
    <itemref idref="one"/>
    <itemref idref="two"/>
  </spine>
  <guide>
    <reference type="title-page" title="Title" href="title.html"/>
  </guide>
</package>"#;
    let ncx = r#"<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="small-hours-0001"/></head>
  <docTitle><text>Small Hours</text></docTitle>
  <navMap>
    <navPoint id="p1" playOrder="1"><navLabel><text>Chapter One</text></navLabel><content src="one.html"/>
      <navPoint id="p2" playOrder="2"><navLabel><text>A Quieter Section</text></navLabel><content src="one.html#s2"/></navPoint>
    </navPoint>
    <navPoint id="p3" playOrder="3"><navLabel><text>Chapter Two</text></navLabel><content src="chapter%20two.html"/></navPoint>
  </navMap>
</ncx>"#;
    let html = |body: &str| {
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head><body>{body}</body></html>"#
        )
    };
    Epub::new(&[
        (
            "META-INF/container.xml",
            &CONTAINER.replace("OEBPS/content.opf", "book/package.opf"),
        ),
        ("book/package.opf", opf),
        ("book/toc.ncx", ncx),
        ("book/title.html", &html("<h1>Small Hours</h1><p>C. Placeholder</p>")),
        (
            "book/one.html",
            &html(
                r#"<h2>Chapter One</h2><p>The clock struck two.</p><h3 id="s2">A Quieter Section</h3><p>Nobody answered.</p>"#,
            ),
        ),
        (
            "book/chapter two.html",
            &html("<h2>Chapter Two</h2><div>Morning, at last.</div>"),
        ),
    ])
}

/// A book with no nav document and no NCX: one chapter per spine file.
pub fn spine_only() -> Epub {
    let opf = r#"<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Two Rooms</dc:title></metadata>
  <manifest>
    <item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>
    <item id="b" href="b.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="a"/><itemref idref="b"/></spine>
</package>"#;
    Epub::new(&[
        ("META-INF/container.xml", CONTAINER),
        ("OEBPS/content.opf", opf),
        (
            "OEBPS/a.xhtml",
            &xhtml("<body><h1>The First Room</h1><p>It was empty.</p></body>"),
        ),
        (
            "OEBPS/b.xhtml",
            &xhtml("<body><h1>The Second Room</h1><p>It was not.</p></body>"),
        ),
    ])
}

/// `encryption.xml` encrypting the named content file with AES (DRM).
pub fn encryption_xml(uri: &str, algorithm: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#">
  <enc:EncryptedData>
    <enc:EncryptionMethod Algorithm="{algorithm}"/>
    <enc:CipherData><enc:CipherReference URI="{uri}"/></enc:CipherData>
  </enc:EncryptedData>
</encryption>"#
    )
}

pub const AES: &str = "http://www.w3.org/2001/04/xmlenc#aes128-cbc";
pub const IDPF_FONT_OBFUSCATION: &str = "http://www.idpf.org/2008/embedding";
