//! The cache and the download, against a local HTTP server serving a small fake model. Nothing here touches the
//! network beyond 127.0.0.1.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ggtts_models::{resolve, sha256_file, Cache, Defaults, Http, Pinned, Source};
use sha2::{Digest, Sha256};

const MODEL: &[u8] = b"a small fake model, standing in for a GGUF talker of two gigabytes";
const CODEC: &[u8] = b"a smaller fake codec";

/// A local HTTP/1.1 server. `/model` and `/codec` serve the fake files, `/wrong` serves bytes of the model's length
/// that are not the model, and `/cut` promises the model's length and closes after half of it. Every request is
/// counted by path.
struct Server {
    base: String,
    hits: Arc<Mutex<HashMap<String, usize>>>,
}

impl Server {
    fn start() -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(Mutex::new(HashMap::new()));
        let counter = hits.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let counter = counter.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut request = String::new();
                    if reader.read_line(&mut request).is_err() {
                        return;
                    }
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                    }
                    let path = request.split_whitespace().nth(1).unwrap_or("").to_string();
                    *counter.lock().unwrap().entry(path.clone()).or_default() += 1;
                    let wrong = vec![b'x'; MODEL.len()];
                    let (status, declared, body): (&str, usize, &[u8]) = match path.as_str() {
                        "/model" => ("200 OK", MODEL.len(), MODEL),
                        "/codec" => ("200 OK", CODEC.len(), CODEC),
                        "/wrong" => ("200 OK", wrong.len(), &wrong),
                        "/cut" => ("200 OK", MODEL.len(), &MODEL[..MODEL.len() / 2]),
                        _ => ("404 Not Found", 0, b""),
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                    );
                    let _ = stream.write_all(body);
                    let _ = stream.flush();
                });
            }
        });
        Server { base, hits }
    }

    fn url(&self, path: &str) -> &'static str {
        Box::leak(format!("{}{path}", self.base).into_boxed_str())
    }

    fn hits(&self, path: &str) -> usize {
        self.hits.lock().unwrap().get(path).copied().unwrap_or(0)
    }
}

fn sha(bytes: &[u8]) -> &'static str {
    Box::leak(format!("{:x}", Sha256::digest(bytes)).into_boxed_str())
}

fn pinned(role: &'static str, file_name: &'static str, url: &'static str, content: &[u8]) -> Pinned {
    Pinned {
        role,
        file_name,
        url,
        size: content.len() as u64,
        sha256: sha(content),
        licence: "test",
    }
}

fn defaults(files: Vec<Pinned>) -> &'static Defaults {
    Box::leak(Box::new(Defaults {
        engine: "test",
        release: "a test release",
        files: Box::leak(files.into_boxed_slice()),
        speakers: &[],
        default_speaker: None,
    }))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-models-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every file under `dir`, relative to it.
fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path().strip_prefix(dir).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    out
}

fn run(
    d: &Defaults,
    given: &[Option<PathBuf>],
    cache: &Cache,
    offline: bool,
) -> (Result<Vec<ggtts_models::ModelFile>, String>, Vec<String>) {
    let mut said = Vec::new();
    let r = resolve(d, given, cache, offline, &Http::default(), &mut |m| {
        said.push(m.to_string())
    });
    (r, said)
}

#[test]
fn first_run_downloads_and_verifies_and_a_second_run_downloads_nothing() {
    let server = Server::start();
    let dir = scratch("first");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let (r, said) = run(d, &[None], &cache, false);
    let files = r.unwrap();
    assert_eq!(server.hits("/model"), 1);
    assert_eq!(std::fs::read(&files[0].path).unwrap(), MODEL);
    assert_eq!(files[0].source, Source::Cache);
    assert_eq!(files[0].sha256.as_deref(), Some(sha(MODEL)));
    // Said before it started: what, from where, how large, and where it goes.
    let announce = said
        .iter()
        .position(|m| m.starts_with("downloading 1 file(s)"))
        .unwrap();
    let detail = &said[announce + 1];
    assert!(
        detail.contains("model.gguf") && detail.contains(server.url("/model")),
        "{detail}"
    );
    assert!(detail.contains(&format!("{} bytes", MODEL.len())), "{detail}");
    assert!(detail.contains(&files[0].path.display().to_string()), "{detail}");
    assert!(said.iter().any(|m| m.contains("verified sha256")), "{said:?}");

    let (r, said) = run(d, &[None], &cache, false);
    assert_eq!(r.unwrap()[0].path, files[0].path);
    assert_eq!(server.hits("/model"), 1, "the second run downloaded again");
    assert!(said.is_empty(), "the second run should be silent: {said:?}");

    // Without its sidecar the file is hashed again, kept, and still not downloaded.
    let sidecar = files_under(&dir)
        .into_iter()
        .find(|f| f.ends_with(".verified"))
        .unwrap();
    std::fs::remove_file(dir.join(&sidecar)).unwrap();
    let (r, said) = run(d, &[None], &cache, false);
    r.unwrap();
    assert!(said.iter().any(|m| m.starts_with("verifying")), "{said:?}");
    assert!(dir.join(&sidecar).exists());
    assert_eq!(server.hits("/model"), 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_wrong_hash_is_refused_naming_the_file_and_both_hashes_and_nothing_is_left() {
    let server = Server::start();
    let dir = scratch("wrong");
    let cache = Cache::new(dir.join("cache"));
    // The pin is the model's; the server sends other bytes of the same length.
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/wrong"), MODEL)]);
    let (r, _) = run(d, &[None], &cache, false);
    let err = r.unwrap_err();
    assert!(err.contains("model.gguf"), "{err}");
    assert!(err.contains("sha256 mismatch"), "{err}");
    assert!(err.contains(sha(MODEL)), "{err}");
    assert!(err.contains(sha(&vec![b'x'; MODEL.len()])), "{err}");
    assert_eq!(files_under(&dir), Vec::<String>::new(), "a refused download left files");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_interrupted_download_leaves_the_cache_clean() {
    let server = Server::start();
    let dir = scratch("cut");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/cut"), MODEL)]);
    let (r, _) = run(d, &[None], &cache, false);
    let err = r.unwrap_err();
    assert!(err.contains("model.gguf") && err.contains("interrupted"), "{err}");
    assert_eq!(
        files_under(&dir),
        Vec::<String>::new(),
        "an interrupted download left files"
    );

    // A partial left by a killed process is removed by the next download, once stale.
    let good = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let entry = cache.path_for(&good.files[0]);
    std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
    let orphan = entry.with_file_name("model.gguf.partial-1-1");
    std::fs::write(&orphan, b"half").unwrap();
    let old = std::time::SystemTime::now() - ggtts_models::STALE_PARTIAL - std::time::Duration::from_secs(1);
    std::fs::File::options()
        .write(true)
        .open(&orphan)
        .unwrap()
        .set_modified(old)
        .unwrap();
    let (r, _) = run(good, &[None], &cache, false);
    r.unwrap();
    assert!(!orphan.exists(), "a stale partial survived a download");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn offline_refuses_to_download_and_names_the_file_and_its_url() {
    let server = Server::start();
    let dir = scratch("offline");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let (r, _) = run(d, &[None], &cache, true);
    let err = r.unwrap_err();
    assert!(
        err.contains("model.gguf") && err.contains(server.url("/model")),
        "{err}"
    );
    assert_eq!(server.hits("/model"), 0);
    assert_eq!(files_under(&dir), Vec::<String>::new());
    // Offline with the file in the cache is fine.
    run(d, &[None], &cache, false).0.unwrap();
    run(d, &[None], &cache, true).0.unwrap();
    assert_eq!(server.hits("/model"), 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_given_path_wins_and_is_not_held_to_the_table() {
    let server = Server::start();
    let dir = scratch("given");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let mine = dir.join("my-unreleased-talker.gguf");
    std::fs::write(&mine, b"not the pinned model at all").unwrap();
    let (r, said) = run(d, &[Some(mine.clone())], &cache, true);
    let files = r.unwrap();
    assert_eq!(files[0].path, mine);
    assert_eq!(files[0].source, Source::Path);
    assert_eq!(files[0].sha256, None, "a given path was hashed against the table");
    assert_eq!(server.hits("/model"), 0);
    assert!(said.is_empty(), "{said:?}");
    assert!(!dir.join("cache").exists(), "a given path touched the cache");
    // Its own hash goes in the report.
    let entry = files[0].report().unwrap();
    assert_eq!(entry["sha256"], sha256_file(&mine).unwrap());
    assert_eq!(entry["source"], "path");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_given_talker_mixes_with_the_default_codec() {
    let server = Server::start();
    let dir = scratch("mixed");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![
        pinned("talker", "model.gguf", server.url("/model"), MODEL),
        pinned("codec", "codec.gguf", server.url("/codec"), CODEC),
    ]);
    let mine = dir.join("local-talker.gguf");
    std::fs::write(&mine, b"local").unwrap();
    let (r, said) = run(d, &[Some(mine.clone()), None], &cache, false);
    let files = r.unwrap();
    assert_eq!((server.hits("/model"), server.hits("/codec")), (0, 1));
    assert!(said.iter().any(|m| m.starts_with("downloading 1 file(s)")), "{said:?}");
    assert_eq!((files[0].source, files[0].path.clone()), (Source::Path, mine));
    assert_eq!(files[1].source, Source::Cache);
    assert_eq!(std::fs::read(&files[1].path).unwrap(), CODEC);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_cached_file_that_no_longer_matches_is_downloaded_again() {
    let server = Server::start();
    let dir = scratch("damaged");
    let cache = Cache::new(dir.join("cache"));
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let path = run(d, &[None], &cache, false).0.unwrap()[0].path.clone();
    std::fs::write(&path, b"damaged").unwrap();
    // Offline, the damaged file is found, removed, and reported missing.
    let (r, said) = run(d, &[None], &cache, true);
    assert!(r.unwrap_err().contains("not in the cache"));
    assert!(said.iter().any(|m| m.contains("removing it")), "{said:?}");
    assert!(!path.exists());
    run(d, &[None], &cache, false).0.unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), MODEL);
    assert_eq!(server.hits("/model"), 2);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn two_processes_starting_at_once_leave_one_good_file() {
    let server = Server::start();
    let dir = scratch("race");
    let d = defaults(vec![pinned("talker", "model.gguf", server.url("/model"), MODEL)]);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let cache = Cache::new(dir.join("cache"));
            std::thread::spawn(move || resolve(d, &[None], &cache, false, &Http::default(), &mut |_| {}))
        })
        .collect();
    for t in threads {
        let files = t.join().unwrap().unwrap();
        assert_eq!(std::fs::read(&files[0].path).unwrap(), MODEL);
    }
    let left = files_under(&dir);
    assert_eq!(left.len(), 2, "only the file and its sidecar: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_cache_root_follows_the_override() {
    // The only test that reads the environment variable, so no other test races it.
    let dir = scratch("root");
    std::env::set_var(ggtts_models::CACHE_ENV, &dir);
    assert_eq!(Cache::default_root().unwrap(), dir);
    std::env::remove_var(ggtts_models::CACHE_ENV);
    let root = Cache::default_root().unwrap();
    assert!(root.is_absolute(), "{}", root.display());
    assert!(root.to_string_lossy().contains("ggtts"), "{}", root.display());
    let _ = std::fs::remove_dir_all(dir);
}
