//! `resolve_steer`: a steering file named by URL is swapped for its cached copy, which the options then read as any
//! steering file; a path is left as it is; offline and a pin refuse as `render.steer`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use ggtts_core::options::Options;
use ggtts_models::{Cache, Fetch};

struct Served(Vec<u8>, AtomicUsize);
impl Fetch for Served {
    fn open(&self, _url: &str) -> Result<Box<dyn Read>, String> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(std::io::Cursor::new(self.0.clone())))
    }
}

const URL: &str = "https://huggingface.co/someone/presets/resolve/main/presets/calm.safetensors";

fn options(steer: &str, root: &Path, offline: bool) -> Options {
    let mut o = Options::default();
    o.render.steer = Some(PathBuf::from(steer));
    o.load.cache_dir = Some(root.to_path_buf());
    o.load.offline = offline;
    o
}

#[test]
fn a_url_resolves_to_its_cached_copy_and_reads_as_steering() {
    let root = std::env::temp_dir().join(format!("ggtts-steer-url-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let v = vec![0.25f32; 8];
    let bytes = ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v)]);
    let served = Served(bytes.clone(), AtomicUsize::new(0));
    let cache = || Ok(Cache::new(root.clone()));
    for _ in 0..2 {
        let mut o = options(URL, &root, false);
        ggtts_load::resolve_steer_with(&mut o, &cache, &served, &mut |_| {}).unwrap();
        let path = o.render.steer.clone().unwrap();
        assert!(path.starts_with(&root), "{}", path.display());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        o.read_steering().unwrap();
        assert!(o.render.steering.is_some());
    }
    assert_eq!(served.1.load(Ordering::SeqCst), 1, "downloaded twice");
    // A path is left as it is (AC3), and nothing is fetched for it.
    let mut o = options("presets/calm.safetensors", &root, true);
    ggtts_load::resolve_steer_with(&mut o, &cache, &served, &mut |_| unreachable!()).unwrap();
    assert_eq!(o.render.steer, Some(PathBuf::from("presets/calm.safetensors")));
    // Offline, an uncached URL is refused as render.steer, naming it.
    let other = "https://huggingface.co/someone/presets/resolve/main/presets/angry.safetensors";
    let mut o = options(other, &root, true);
    let r = ggtts_load::resolve_steer_with(&mut o, &cache, &served, &mut |_| {}).unwrap_err();
    assert_eq!(r.key, "render.steer");
    assert!(r.message.contains(other), "{}", r.message);
    // A pin that differs is refused as render.steer.
    let mut o = options(&format!("{URL}#sha256={}", "1".repeat(64)), &root, false);
    let r = ggtts_load::resolve_steer_with(&mut o, &cache, &served, &mut |_| {}).unwrap_err();
    assert_eq!(r.key, "render.steer");
    assert!(r.message.contains("not the pinned"), "{}", r.message);
    let _ = std::fs::remove_dir_all(root);
}
