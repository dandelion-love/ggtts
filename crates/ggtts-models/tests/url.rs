//! A file named by URL (`Cache::fetch_url`): downloaded once into the cache and reused, refused offline when it is not
//! cached, checked against an optional `#sha256=` pin, and capped in size. A fake fetcher serves the bytes and counts
//! how often it is asked, so nothing here touches the network.

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ggtts_models::{Cache, Fetch, UrlSpec, URL_FILE_MAX};
use sha2::{Digest, Sha256};

const URL: &str = "https://huggingface.co/someone/presets/resolve/main/presets/calm.safetensors";

/// Serves `body` for every URL and counts the requests.
struct Served {
    body: Vec<u8>,
    opened: AtomicUsize,
}

impl Served {
    fn new(body: &[u8]) -> Self {
        Served {
            body: body.to_vec(),
            opened: AtomicUsize::new(0),
        }
    }
    fn opened(&self) -> usize {
        self.opened.load(Ordering::SeqCst)
    }
}

impl Fetch for Served {
    fn open(&self, _url: &str) -> Result<Box<dyn Read>, String> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(std::io::Cursor::new(self.body.clone())))
    }
}

/// Never ends: zeros for ever.
struct Endless;
impl Fetch for Endless {
    fn open(&self, _url: &str) -> Result<Box<dyn Read>, String> {
        Ok(Box::new(std::io::repeat(0)))
    }
}

/// A fetcher that fails the test if it is used.
struct NoNetwork;
impl Fetch for NoNetwork {
    fn open(&self, url: &str) -> Result<Box<dyn Read>, String> {
        panic!("fetched {url}")
    }
}

fn cache(name: &str) -> (Cache, PathBuf) {
    let root = std::env::temp_dir().join(format!("ggtts-url-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    (Cache::new(root.clone()), root)
}

fn spec(text: &str) -> UrlSpec {
    UrlSpec::parse(text).expect("a URL").expect("a valid URL")
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn a_url_is_parsed_with_its_pin_and_a_path_is_not_a_url() {
    let pin = "ab".repeat(32);
    assert_eq!(
        spec(&format!("{URL}#sha256={}", pin.to_uppercase())),
        UrlSpec {
            url: URL.into(),
            sha256: Some(pin)
        }
    );
    assert_eq!(spec(URL).sha256, None);
    // The hub shorthand is the Hub's resolve URL, at main unless a revision is given, with the same pin.
    let hub = spec(&format!(
        "hf://someone/presets/presets/calm.safetensors#sha256={}",
        "ab".repeat(32)
    ));
    assert_eq!((hub.url.as_str(), hub.sha256.is_some()), (URL, true));
    assert_eq!(
        spec("hf://someone/presets@0123abc/presets/calm.safetensors").url,
        "https://huggingface.co/someone/presets/resolve/0123abc/presets/calm.safetensors"
    );
    assert!(UrlSpec::parse("hf://someone/presets").unwrap().is_err());
    assert!(UrlSpec::parse("presets/calm.safetensors").is_none());
    assert!(UrlSpec::parse("/abs/calm.safetensors").is_none());
    let http = UrlSpec::parse("http://example.com/x.safetensors").unwrap().unwrap_err();
    assert!(http.contains("only https://"), "{http}");
    let bad = UrlSpec::parse(&format!("{URL}#sha256=xyz")).unwrap().unwrap_err();
    assert!(bad.contains("64 hex digits"), "{bad}");
}

/// AC1, AC5: the first call downloads, a later one reuses the cached file without asking the network.
#[test]
fn a_url_is_downloaded_once_and_then_reused() {
    let (cache, root) = cache("once");
    let served = Served::new(b"steering bytes");
    let mut said = Vec::new();
    let first = cache
        .fetch_url(&spec(URL), false, &served, &mut |m| said.push(m.to_string()))
        .unwrap();
    assert!(first.downloaded);
    assert_eq!(first.sha256, sha(b"steering bytes"));
    assert_eq!(std::fs::read(&first.path).unwrap(), b"steering bytes");
    assert!(first.path.starts_with(root.join("urls")), "{}", first.path.display());
    assert_eq!(first.path.file_name().unwrap(), "calm.safetensors");
    let second = cache
        .fetch_url(&spec(URL), false, &served, &mut |m| said.push(m.to_string()))
        .unwrap();
    assert_eq!(
        (second.path.clone(), second.sha256.clone(), second.downloaded),
        (first.path, first.sha256.clone(), false)
    );
    assert_eq!(served.opened(), 1, "the cached file was fetched again");
    // AC2: the sha256 is said either way.
    assert!(
        said.iter().filter(|m| m.contains(&first.sha256)).count() >= 2,
        "{said:?}"
    );
    // Offline, a cached URL is still served.
    assert!(cache.fetch_url(&spec(URL), true, &NoNetwork, &mut |_| {}).is_ok());
    let _ = std::fs::remove_dir_all(root);
}

/// AC1, AC5: offline, a URL not in the cache is refused, naming it, and nothing is fetched.
#[test]
fn offline_a_url_not_in_the_cache_is_refused_naming_it() {
    let (cache, root) = cache("offline");
    let e = cache.fetch_url(&spec(URL), true, &NoNetwork, &mut |_| {}).unwrap_err();
    assert!(
        e.contains(URL) && e.contains("offline mode does not download it"),
        "{e}"
    );
    assert!(!cache.path_for_url(&spec(URL)).exists());
    let _ = std::fs::remove_dir_all(root);
}

/// AC2, AC5: a pin refuses a download whose sha256 differs, and removes it; the right pin is taken; a cached copy that
/// differs from a pin is refused too.
#[test]
fn a_pin_refuses_a_file_whose_sha256_differs() {
    let (cache, root) = cache("pin");
    let served = Served::new(b"the real preset");
    let wrong = format!("{URL}#sha256={}", "0".repeat(64));
    let e = cache.fetch_url(&spec(&wrong), false, &served, &mut |_| {}).unwrap_err();
    assert!(e.contains("not the pinned") && e.contains("refused and removed"), "{e}");
    assert!(!cache.path_for_url(&spec(URL)).exists(), "a refused download was kept");
    let right = format!("{URL}#sha256={}", sha(b"the real preset"));
    let got = cache.fetch_url(&spec(&right), false, &served, &mut |_| {}).unwrap();
    assert_eq!(got.sha256, sha(b"the real preset"));
    // The same URL is one cache entry whatever the pin: a wrong pin against the cached copy is refused.
    let e = cache.fetch_url(&spec(&wrong), false, &served, &mut |_| {}).unwrap_err();
    assert!(e.contains("not the pinned") && e.contains("cached copy"), "{e}");
    let _ = std::fs::remove_dir_all(root);
}

/// A URL serving more than `URL_FILE_MAX` is refused and leaves nothing behind.
#[test]
fn a_url_serving_too_much_is_refused() {
    let (cache, root) = cache("max");
    let e = cache.fetch_url(&spec(URL), false, &Endless, &mut |_| {}).unwrap_err();
    assert!(e.contains("more than") && e.contains(&URL_FILE_MAX.to_string()), "{e}");
    let dir = cache.path_for_url(&spec(URL)).parent().unwrap().to_path_buf();
    let left: Vec<_> = std::fs::read_dir(&dir)
        .map(|d| d.flatten().collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "left behind: {left:?}");
    let _ = std::fs::remove_dir_all(root);
}

/// A cached copy that changed since it was downloaded (overwritten, or truncated) is refused, not served under the
/// URL; one that only had its time touched is still served.
#[test]
fn a_cached_copy_that_changed_since_its_download_is_refused() {
    let (cache, root) = cache("changed");
    let served = Served::new(b"the preset as published");
    let got = cache.fetch_url(&spec(URL), false, &served, &mut |_| {}).unwrap();
    // Touched, same bytes: served.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&got.path, b"the preset as published").unwrap();
    assert_eq!(
        cache
            .fetch_url(&spec(URL), true, &NoNetwork, &mut |_| {})
            .unwrap()
            .sha256,
        got.sha256
    );
    // Overwritten: refused, naming the cached copy.
    std::fs::write(&got.path, b"something else").unwrap();
    let e = cache.fetch_url(&spec(URL), true, &NoNetwork, &mut |_| {}).unwrap_err();
    assert!(
        e.contains("changed since it was downloaded") && e.contains("delete it"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(root);
}
