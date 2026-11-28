//! The model files a render needs when it is not given them: a table pinned by sha256, a per-user cache, and a
//! download that is verified before anything uses it.
//!
//! * **The table** ([`QWEN3_TTS`]) names each default file of an engine by an exact URL at a fixed commit, its size,
//!   its sha256 and its licence. "Latest" means the newest release this build knows about; a newer release needs a
//!   new table, never a moving branch.
//! * **A given path wins.** [`resolve`] uses a file named on the command line as it is: it is never downloaded over
//!   and never compared with the table, so an unreleased model can be rendered. Its sha256 is computed once per run
//!   for the record ([`ModelFile::ensure_sha256`]), never compared with anything.
//! * **The cache** ([`Cache`]) holds each file at `models/<sha256>/<file name>` under one root, the platform's
//!   per-user cache directory or `GGTTS_CACHE_DIR`. A download goes to a temporary name in the same directory, is
//!   hashed as it arrives, and is renamed into place only when its size and sha256 match the table. On a mismatch
//!   or an interruption the partial file is deleted.
//! * **Re-verification is cheap.** After a file is verified, a sidecar (`<file name>.verified`) records its sha256,
//!   size and modification time. A later run trusts the file when the sidecar names the pinned hash and the size and
//!   modification time still match; otherwise it hashes the file again (a few seconds for 2 GB) and removes it if
//!   it does not match. The trade-off: an edit that keeps both the size and the modification time is not noticed
//!   until the sidecar is deleted. The cache is ggtts's own directory, so that edit would have to be deliberate.
//! * **Two processes at once** each download to their own temporary name, and the rename is atomic, so a lost race
//!   costs a second download, never a corrupt file.
//! * **Each download holds a separate guard** before staging appears until publication finishes: Unix uses
//!   `flock`; Windows opens the guard without sharing. Cleanup checks that guard before touching the partial or its
//!   staging name, including the Windows close-before-rename window. A dead writer's partial is removed on the next
//!   resolve. A crash before a partial appears or after publication can leave a small orphan guard, which is not
//!   removed automatically. Concurrent processes using older library versions do not honor these guards.
//! * **On local filesystems with working process locks**, active guarded downloads are not removed by another current
//!   process. Filesystems that do not implement those locking semantics are not covered by this guarantee.
//! * **A download that stops sending** fails after [`READ_IDLE`] without a byte, naming the file and how much arrived.
//!   Progress is reported every [`PROGRESS_EVERY`].

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

/// Overrides the cache root.
pub const CACHE_ENV: &str = "GGTTS_CACHE_DIR";

/// A download that receives nothing for this long has stalled, and fails.
pub const READ_IDLE: Duration = Duration::from_secs(60);

/// How often a download reports its progress.
pub const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// One default file of an engine, pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pinned {
    /// What the engine uses it as, for example `talker` or `codec`.
    pub role: &'static str,
    pub file_name: &'static str,
    /// The exact URL, at a fixed commit.
    pub url: &'static str,
    pub size: u64,
    /// Lower-case hex.
    pub sha256: &'static str,
    /// The file's licence, as its publisher states it.
    pub licence: &'static str,
}

/// One named release: the files of one engine, in the order the engine's command-line paths take them.
#[derive(Debug)]
pub struct Release {
    /// What `load.model` (`--model`) names it by. Never a file name: it has no path separator and no `.gguf`.
    pub name: &'static str,
    pub engine: &'static str,
    /// Where the files come from, for messages and the README.
    pub release: &'static str,
    pub files: &'static [Pinned],
    /// The speakers the default model has.
    pub speakers: &'static [&'static str],
    /// The speaker a render uses when it is given none and the model is this release's.
    pub default_speaker: Option<&'static str>,
}

impl Release {
    /// The size of every file, in bytes.
    pub fn size(&self) -> u64 {
        self.files.iter().map(|p| p.size).sum()
    }

    /// The files' licences, each once, in file order, joined with `, `.
    pub fn licences(&self) -> String {
        let mut seen: Vec<&str> = Vec::new();
        for p in self.files {
            if !seen.contains(&p.licence) {
                seen.push(p.licence);
            }
        }
        seen.join(", ")
    }
}

/// An engine's default files are a release: the one [`defaults_for`] gives.
pub type Defaults = Release;

/// Qwen3-TTS, `hk-cantonese-1.7b`: the talker and the 12 Hz codec of `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` v0.0.5
/// (commit `e6a823291c0189031086865033d0f5e4ad02a8ac`), on `Qwen3-TTS-12Hz-1.7B-Base`. Both files are Apache-2.0 by
/// the repository's card. The default speaker is `hk_voice_343`, the voice the card's own example renders. It is the
/// qwen3-tts engine's default.
pub const QWEN3_TTS: Release = Release {
    name: "hk-cantonese-1.7b",
    engine: "qwen3-tts",
    release: "jmtl/hk-cantonese-cc0-qwen3-tts-GGUF v0.0.5",
    files: &[
        Pinned {
            role: "talker",
            file_name: "qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf",
            url: "https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/resolve/e6a823291c0189031086865033d0f5e4ad02a8ac/qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf",
            size: 2_042_834_144,
            sha256: "4eb27e5a3434d15ca818814b8e4f0f1941716654afb8543866732b62f9227041",
            licence: "Apache-2.0",
        },
        Pinned {
            role: "codec",
            file_name: "qwen-tokenizer-12hz-q8_0.gguf",
            url: "https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/resolve/e6a823291c0189031086865033d0f5e4ad02a8ac/qwen-tokenizer-12hz-q8_0.gguf",
            size: 291_150_624,
            sha256: "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824",
            licence: "Apache-2.0",
        },
    ],
    speakers: &["hk_voice_343", "hk_voice_777"],
    default_speaker: Some("hk_voice_343"),
};

/// Every named release this build can download, by [`Release::name`]. A new release is one more entry here, pinned
/// like the others; it changes no engine's default unless [`ALL`] names it too.
pub const RELEASES: &[&Release] = &[&QWEN3_TTS];

/// Every engine that has default files: the release a render of that engine uses when it names no model. An engine
/// left out renders only from given paths or a named release.
pub const ALL: &[&Defaults] = &[&QWEN3_TTS];

/// The default files of `engine`, if it has any.
pub fn defaults_for(engine: &str) -> Option<&'static Defaults> {
    ALL.iter().copied().find(|d| d.engine == engine)
}

/// The release named `name`, if this build knows it.
pub fn named(name: &str) -> Option<&'static Release> {
    RELEASES.iter().copied().find(|r| r.name == name)
}

/// Whether `release` is its engine's default.
pub fn is_default(release: &Release) -> bool {
    defaults_for(release.engine).is_some_and(|d| d.name == release.name)
}

/// Every release name, in table order, joined with `, `: what a refusal of an unknown name lists.
pub fn known_names() -> String {
    RELEASES.iter().map(|r| r.name).collect::<Vec<_>>().join(", ")
}

/// Whether `text` reads as a release name rather than a file: no path separator, no drive or scheme colon, and no
/// `.gguf` or `.json` ending.
fn looks_like_a_name(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    !text.is_empty() && !text.contains(['/', '\\', ':']) && !lower.ends_with(".gguf") && !lower.ends_with(".json")
}

/// What a model setting (`load.model`, `--model`) selects, for an engine (`auto` for any): `Ok(Some(release))` when
/// it names a release, `Ok(None)` when it is a file path or not given. This is the one rule every front end applies:
///
/// * a value equal to a release's name is that release (a local file of that name is given as `./<name>`);
/// * a name of another engine's release is refused, naming both engines;
/// * a value that reads as a name (no path separator, no `.gguf` or `.json` ending), is no release, and is no
///   existing file is refused, listing the known names;
/// * anything else is a path, used as it is.
///
/// The refusal is the message alone; each front end prefixes it with its own spelling of the setting.
pub fn select(engine: &str, model: Option<&Path>) -> Result<Option<&'static Release>, String> {
    let Some(model) = model else {
        return Ok(None);
    };
    let Some(text) = model.to_str() else {
        return Ok(None);
    };
    if let Some(release) = named(text) {
        if engine != "auto" && engine != release.engine {
            return Err(format!(
                "{text} is a {} release, and the engine is {engine}",
                release.engine
            ));
        }
        return Ok(Some(release));
    }
    if looks_like_a_name(text) && !model.exists() {
        return Err(format!(
            "{text} is neither a known model nor a file; the known models are {}",
            known_names()
        ));
    }
    Ok(None)
}

/// One line per release for a listing (`ggtts-render --models`), tab-separated: name, engine, size, licence, whether
/// every file is in `cache` (checked as a render checks it: its verification record, else its hash), and `default`
/// for an engine's default.
pub fn listing(cache: &Cache) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for r in RELEASES {
        let mut cached = true;
        for p in r.files {
            if !cache.check(p, &mut |_| {})? {
                cached = false;
            }
        }
        lines.push(format!(
            "{}\t{}\t{}\t{}\t{}{}",
            r.name,
            r.engine,
            human(r.size()),
            r.licences(),
            if cached { "cached" } else { "not cached" },
            if is_default(r) { "\tdefault" } else { "" }
        ));
    }
    Ok(lines)
}

/// Where a model file came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The cache, verified against the table.
    Cache,
    /// A path given on the command line, used as it is.
    Path,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Cache => "cache",
            Source::Path => "path",
        }
    }
}

/// One model file a render uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub role: String,
    pub path: PathBuf,
    pub source: Source,
    /// Known without reading the file: the pinned hash of a cached file. `None` for a given path until it is hashed.
    pub sha256: Option<String>,
    /// Where a cached file was fetched from.
    pub url: Option<String>,
}

impl ModelFile {
    pub fn given(role: &str, path: PathBuf) -> Self {
        ModelFile {
            role: role.to_string(),
            path,
            source: Source::Path,
            sha256: None,
            url: None,
        }
    }

    /// Hashes a given file, once: returns how long that took, or `None` when the hash was already known (a cached
    /// file's is the pinned hash it was verified against).
    pub fn ensure_sha256(&mut self) -> Result<Option<Duration>, String> {
        if self.sha256.is_some() {
            return Ok(None);
        }
        let started = std::time::Instant::now();
        self.sha256 = Some(sha256_file(&self.path)?);
        Ok(Some(started.elapsed()))
    }

    /// The file's report entry: `role`, `path`, `sha256`, `source` and `url`. A given file not yet hashed is hashed
    /// now.
    pub fn report(&self) -> Result<serde_json::Value, String> {
        let sha256 = match &self.sha256 {
            Some(h) => h.clone(),
            None => sha256_file(&self.path)?,
        };
        Ok(serde_json::json!({
            "role": self.role, "path": self.path.to_string_lossy(), "sha256": sha256,
            "source": self.source.as_str(), "url": self.url,
        }))
    }
}

/// Opens a URL for reading. The real one is [`Http`]; tests can serve their own.
pub trait Fetch {
    fn open(&self, url: &str) -> Result<Box<dyn Read>, String>;
    /// Cooperative cancellation, checked before verified file publication.
    fn cancelled(&self) -> bool {
        false
    }
}

/// HTTPS over ureq with rustls. Redirects are followed (the hub redirects to its CDN), but never out of HTTPS: every
/// hop is checked. A status other than 2xx is an error, and so is [`READ_IDLE`] without a byte.
pub struct Http {
    agent: ureq::Agent,
}

impl Default for Http {
    fn default() -> Self {
        Http::new(true, READ_IDLE)
    }
}

impl Http {
    /// `https_only` false is for tests against a local plain-HTTP server; `read_idle` is how long a read may wait.
    pub fn new(https_only: bool, read_idle: Duration) -> Self {
        let agent = ureq::Agent::config_builder()
            .https_only(https_only)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_per_read(Some(read_idle))
            .user_agent(concat!("ggtts/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        Http { agent }
    }
}

impl Fetch for Http {
    fn open(&self, url: &str) -> Result<Box<dyn Read>, String> {
        let response = self.agent.get(url).call().map_err(|e| e.to_string())?;
        Ok(Box::new(response.into_body().into_reader()))
    }
}

/// The sha256 of a file, as lower-case hex.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    hash_file(path).map(|(_, h)| h)
}

fn hash_file(path: &Path) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let read = file.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        n += read as u64;
    }
    Ok((n, format!("{:x}", hasher.finalize())))
}

/// `bytes` for a person: `2.04 GB (2042834144 bytes)`.
pub fn human(bytes: u64) -> String {
    let (value, unit) = if bytes >= 1_000_000_000 {
        (bytes as f64 / 1e9, "GB")
    } else if bytes >= 1_000_000 {
        (bytes as f64 / 1e6, "MB")
    } else {
        return format!("{bytes} bytes");
    };
    format!("{value:.2} {unit} ({bytes} bytes)")
}

fn mtime_ns(meta: &fs::Metadata) -> Option<u128> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
}

/// A name no other download in this process or another one is using: the process id and a counter.
fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!(
        "{}-{}-{nanos}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// A separate lock survives staging and the Windows close-before-rename window.
/// Its name is unique to the partial and never reused.
struct DownloadGuard {
    file: Option<File>,
    path: PathBuf,
}

fn guard_path(partial: &Path) -> PathBuf {
    let mut name = partial.file_name().unwrap_or_default().to_os_string();
    name.push(".guard");
    partial.with_file_name(name)
}

impl DownloadGuard {
    fn create(partial: &Path) -> std::io::Result<Self> {
        let path = guard_path(partial);
        let mut options = fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0);
        }
        let file = options.open(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let error = std::io::Error::last_os_error();
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        }
        Ok(Self { file: Some(file), path })
    }
}

impl Drop for DownloadGuard {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

/// Hold the guard while deleting a partial. Old partials without a guard retain
/// the existing data-file lock check. The writer creates its guard before the
/// partial appears, so a missing guard cannot describe a new live writer.
fn remove_guarded_partial(path: &Path) -> bool {
    let mut base = path.to_path_buf();
    if path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with(".locking"))
    {
        let name = path.file_name().unwrap().to_string_lossy();
        base = path.with_file_name(name.strip_suffix(".locking").unwrap());
    }
    let guard = guard_path(&base);
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0);
    }
    let lock = match options.open(&guard) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return remove_if_abandoned(path),
        Err(_) => return false,
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return false;
        }
    }
    let removed = fs::remove_file(path).is_ok();
    drop(lock);
    // The data and its staging name share a guard. Do not remove the guard
    // while either still exists: another cleanup may be examining it.
    let mut staging = base.as_os_str().to_os_string();
    staging.push(".locking");
    if !base.exists() && !Path::new(&staging).exists() {
        let _ = fs::remove_file(guard);
    }
    removed
}

/// Creates a temporary download, locked against removal by any other process for as long as it is open.
#[cfg(unix)]
fn create_locked(path: &Path) -> std::io::Result<File> {
    use std::os::unix::io::AsRawFd;
    // Created and locked under a name no cleanup looks at, then renamed to its own: a cleanup can never find it
    // unlocked. The descriptor, and its lock, survive the rename.
    let mut staging = path.file_name().unwrap_or_default().to_os_string();
    staging.push(".locking");
    let staging = path.with_file_name(staging);
    let file = fs::OpenOptions::new().write(true).create_new(true).open(&staging)?;
    // SAFETY: flock on a file descriptor this function owns; the lock is released when the file is closed.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let e = std::io::Error::last_os_error();
        let _ = fs::remove_file(&staging);
        return Err(e);
    }
    if let Err(e) = fs::rename(&staging, path) {
        let _ = fs::remove_file(&staging);
        return Err(e);
    }
    Ok(file)
}

/// Creates a temporary download, locked against removal by any other process for as long as it is open.
#[cfg(not(unix))]
fn create_locked(path: &Path) -> std::io::Result<File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_SHARE_READ only: without FILE_SHARE_DELETE no other process can delete or rename it while it is open.
        options.share_mode(0x0000_0001);
    }
    options.open(path)
}

/// Removes `path` if no live process is writing it: on Unix when its writer's lock can be taken, on Windows when it
/// can be deleted (its writer holds it open without delete sharing).
fn remove_if_abandoned(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let Ok(file) = File::open(path) else {
            return false;
        };
        // SAFETY: as above, on a descriptor owned here.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return false;
        }
        fs::remove_file(path).is_ok()
    }
    #[cfg(not(unix))]
    {
        fs::remove_file(path).is_ok()
    }
}

/// Removes a temporary download when it goes out of scope, whatever the reason.
struct Partial(PathBuf);

impl Drop for Partial {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// The per-user cache of downloaded model files.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new(root: PathBuf) -> Self {
        Cache { root }
    }

    /// `GGTTS_CACHE_DIR`, else the platform's per-user cache: `$XDG_CACHE_HOME/ggtts` or `~/.cache/ggtts` on Linux,
    /// `~/Library/Caches/ggtts` on macOS, `%LOCALAPPDATA%\ggtts\cache` on Windows.
    pub fn default_root() -> Result<PathBuf, String> {
        let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
        if let Some(root) = var(CACHE_ENV) {
            return Ok(root);
        }
        if cfg!(windows) {
            return var("LOCALAPPDATA")
                .map(|d| d.join("ggtts").join("cache"))
                .ok_or_else(|| format!("no cache directory: set {CACHE_ENV} or LOCALAPPDATA"));
        }
        if cfg!(target_os = "macos") {
            return var("HOME")
                .map(|h| h.join("Library").join("Caches").join("ggtts"))
                .ok_or_else(|| format!("no cache directory: set {CACHE_ENV} or HOME"));
        }
        // The XDG base directory rule: a relative XDG_CACHE_HOME is ignored.
        if let Some(xdg) = var("XDG_CACHE_HOME").filter(|p| p.is_absolute()) {
            return Ok(xdg.join("ggtts"));
        }
        var("HOME")
            .map(|h| h.join(".cache").join("ggtts"))
            .ok_or_else(|| format!("no cache directory: set {CACHE_ENV}, XDG_CACHE_HOME or HOME"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where `pinned` lives in the cache, present or not.
    pub fn path_for(&self, pinned: &Pinned) -> PathBuf {
        self.root.join("models").join(pinned.sha256).join(pinned.file_name)
    }

    fn sidecar(path: &Path) -> PathBuf {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".verified");
        path.with_file_name(name)
    }

    fn sidecar_matches(path: &Path, meta: &fs::Metadata, pinned: &Pinned) -> bool {
        let Ok(text) = fs::read_to_string(Self::sidecar(path)) else {
            return false;
        };
        let Ok(record) = serde_json::from_str::<serde_json::Value>(&text) else {
            return false;
        };
        record["sha256"] == pinned.sha256
            && record["size"] == pinned.size
            && meta.len() == pinned.size
            && mtime_ns(meta).is_some_and(|m| record["mtime_ns"].as_str() == Some(m.to_string().as_str()))
    }

    fn write_sidecar(path: &Path, pinned: &Pinned) -> Result<(), String> {
        let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let Some(mtime) = mtime_ns(&meta) else {
            // No modification time on this filesystem: no sidecar, so every run hashes the file again.
            return Ok(());
        };
        let record = serde_json::json!({
            "format": "ggtts-cache-entry/1", "file": pinned.file_name, "sha256": pinned.sha256,
            "size": meta.len(), "mtime_ns": mtime.to_string(),
        });
        let sidecar = Self::sidecar(path);
        let tmp = sidecar.with_extension(format!("verified.tmp-{}", unique_suffix()));
        fs::write(&tmp, format!("{record}\n")).map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::rename(&tmp, &sidecar).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("{}: {e}", sidecar.display())
        })
    }

    /// Whether `pinned` is in the cache and matches the table. A file that does not match is removed, so it is
    /// downloaded again.
    pub fn check(&self, pinned: &Pinned, say: &mut dyn FnMut(&str)) -> Result<bool, String> {
        let path = self.path_for(pinned);
        let meta = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        if Self::sidecar_matches(&path, &meta, pinned) {
            return Ok(true);
        }
        say(&format!(
            "verifying {} ({}): no record of an earlier check matches it",
            path.display(),
            human(meta.len())
        ));
        let (size, hash) = hash_file(&path)?;
        if size == pinned.size && hash == pinned.sha256 {
            Self::write_sidecar(&path, pinned)?;
            return Ok(true);
        }
        say(&format!(
            "{}: {size} bytes with sha256 {hash}, expected {} bytes with sha256 {}; removing it",
            path.display(),
            pinned.size,
            pinned.sha256
        ));
        let _ = fs::remove_file(Self::sidecar(&path));
        fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(false)
    }

    /// Removes the temporary downloads of `pinned` whose writer has exited (see `create_locked`); returns how many.
    /// A live download is never touched, however old.
    pub fn remove_abandoned_partials(&self, pinned: &Pinned) -> usize {
        let path = self.path_for(pinned);
        let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
            return 0;
        };
        let prefix = format!("{}.partial-", name.to_string_lossy());
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with(&prefix) && !name.ends_with(".guard")
            })
            .filter(|e| remove_guarded_partial(&e.path()))
            .count()
    }

    /// Downloads `pinned` into the cache, verifies it and moves it into place.
    pub fn download(&self, pinned: &Pinned, fetch: &dyn Fetch, say: &mut dyn FnMut(&str)) -> Result<PathBuf, String> {
        let path = self.path_for(pinned);
        let dir = path.parent().expect("a cache path has a directory");
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        self.remove_abandoned_partials(pinned);
        let tmp = dir.join(format!("{}.partial-{}", pinned.file_name, unique_suffix()));
        let _guard = DownloadGuard::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let partial = Partial(tmp.clone());
        let mut out = Some(create_locked(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?);
        let name = pinned.file_name;
        let mut body = fetch
            .open(pinned.url)
            .map_err(|e| format!("{name}: downloading {} failed: {e}", pinned.url))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut n = 0u64;
        let started = std::time::Instant::now();
        let mut last_report = started;
        loop {
            let read = body.read(&mut buf).map_err(|e| {
                format!(
                    "{name}: download interrupted after {n} of {} bytes from {}: {e}; the partial file was removed",
                    pinned.size, pinned.url
                )
            })?;
            if read == 0 {
                break;
            }
            n += read as u64;
            if n > pinned.size {
                return Err(format!(
                    "{name}: {} sent more than the pinned {} bytes; the partial file was removed",
                    pinned.url, pinned.size
                ));
            }
            hasher.update(&buf[..read]);
            out.as_mut()
                .expect("open until the rename")
                .write_all(&buf[..read])
                .map_err(|e| format!("{}: {e}", tmp.display()))?;
            if last_report.elapsed() >= PROGRESS_EVERY {
                last_report = std::time::Instant::now();
                let seconds = started.elapsed().as_secs_f64().max(1e-3);
                say(&format!(
                    "{name}: {:.0} of {:.0} MB ({}%), {:.1} MB/s",
                    n as f64 / 1e6,
                    pinned.size as f64 / 1e6,
                    n * 100 / pinned.size.max(1),
                    n as f64 / 1e6 / seconds
                ));
            }
        }
        if n != pinned.size {
            return Err(format!(
                "{name}: download interrupted after {n} of {} bytes from {}; the partial file was removed",
                pinned.size, pinned.url
            ));
        }
        if fetch.cancelled() {
            return Err(format!("{name}: download cancelled; the partial file was removed"));
        }
        out.as_ref()
            .expect("open until the rename")
            .sync_all()
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        let got = format!("{:x}", hasher.finalize());
        if got != pinned.sha256 {
            return Err(format!(
                "{name}: sha256 mismatch: expected {}, got {got}; the download from {} was refused and removed",
                pinned.sha256, pinned.url
            ));
        }
        // On Windows a file open without delete sharing cannot be renamed, even by its own process, so it is closed
        // first; on Unix it stays open, and locked, through the rename.
        if cfg!(windows) {
            out = None;
        }
        if fetch.cancelled() {
            return Err(format!("{name}: download cancelled; the partial file was removed"));
        }
        let renamed = fs::rename(&tmp, &path);
        drop(out);
        if let Err(e) = renamed {
            // Another process may have put a verified copy there first (on Windows a file in use cannot be replaced).
            if !self.check(pinned, say)? {
                return Err(format!("{}: {e}", path.display()));
            }
        }
        drop(partial);
        Self::write_sidecar(&path, pinned)?;
        say(&format!(
            "{name}: verified sha256 {}, stored at {}",
            pinned.sha256,
            path.display()
        ));
        Ok(path)
    }
}

// ── files named by URL ──

/// The most a file named by URL may hold. A steering preset is a few kilobytes to a few megabytes; this stops a wrong
/// URL from filling the disk.
pub const URL_FILE_MAX: u64 = 256 << 20;

/// A file named by an `https://` URL, or by the Hugging Face Hub shorthand `hf://<owner>/<repo>[@<revision>]/<path>`
/// (the Hub's `resolve` URL, at `main` unless a revision is given), with an optional pin: `...#sha256=<64 hex>`. A
/// revision is one path segment (a commit, a tag, or a branch with no `/`): the shorthand is split at the first two
/// slashes, so a revision such as `refs/pr/3` needs the full `https://` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlSpec {
    /// The URL without the pin.
    pub url: String,
    /// The pinned sha256, lower-case hex.
    pub sha256: Option<String>,
}

impl UrlSpec {
    /// `Some` when `text` names a URL: `https://...` or `hf://owner/repo[@revision]/path`, optionally ending in
    /// `#sha256=<64 hex>`. `http://` is refused, as every download here is HTTPS; any other text is `None` (a path).
    pub fn parse(text: &str) -> Option<Result<UrlSpec, String>> {
        let lower = text.to_ascii_lowercase();
        if lower.starts_with("hf://") {
            return Some(Self::hub(text));
        }
        if lower.starts_with("http://") {
            return Some(Err(format!("{text}: only https:// URLs are downloaded")));
        }
        if !lower.starts_with("https://") {
            return None;
        }
        let (url, fragment) = match text.split_once('#') {
            Some((u, f)) => (u, Some(f)),
            None => (text, None),
        };
        let sha256 = match fragment {
            None => None,
            Some(f) => {
                let hex = f.strip_prefix("sha256=").unwrap_or("");
                if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Some(Err(format!("{text}: a pin is #sha256= and 64 hex digits, not #{f}")));
                }
                Some(hex.to_ascii_lowercase())
            }
        };
        Some(Ok(UrlSpec {
            url: url.to_string(),
            sha256,
        }))
    }

    /// `hf://owner/repo[@revision]/path[#sha256=...]` as the Hub's `resolve` URL.
    fn hub(text: &str) -> Result<UrlSpec, String> {
        let (body, fragment) = match text[5..].split_once('#') {
            Some((b, f)) => (b, Some(f)),
            None => (&text[5..], None),
        };
        let mut parts = body.splitn(3, '/');
        let (owner, repo, path) = match (parts.next(), parts.next(), parts.next()) {
            (Some(o), Some(r), Some(p)) if !o.is_empty() && !r.is_empty() && !p.is_empty() => (o, r, p),
            _ => return Err(format!("{text}: a hub file is hf://<owner>/<repo>[@<revision>]/<path>")),
        };
        let (repo, revision) = repo.split_once('@').unwrap_or((repo, "main"));
        if repo.is_empty() || revision.is_empty() {
            return Err(format!("{text}: a hub file is hf://<owner>/<repo>[@<revision>]/<path>"));
        }
        let url = format!("https://huggingface.co/{owner}/{repo}/resolve/{revision}/{path}");
        let full = match fragment {
            Some(f) => format!("{url}#{f}"),
            None => url,
        };
        match Self::parse(&full) {
            Some(r) => r,
            None => unreachable!("an https:// URL parses"),
        }
    }

    /// The file name the URL ends in, or `file` when it ends in none.
    fn file_name(&self) -> String {
        let path = self.url.split(['?', '#']).next().unwrap_or("");
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() || name.contains(['\\', ':']) || name == "." || name == ".." {
            "file".to_string()
        } else {
            name.to_string()
        }
    }
}

/// A file a URL named, in the cache: where it is and its sha256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlFile {
    pub path: PathBuf,
    pub sha256: String,
    /// Whether this call downloaded it (false: it was already in the cache).
    pub downloaded: bool,
}

impl Cache {
    /// Where a URL's file lives in the cache, present or not: `urls/<sha256 of the URL>/<file name>`, so one URL is
    /// one entry, whatever it serves.
    pub fn path_for_url(&self, spec: &UrlSpec) -> PathBuf {
        let key = format!("{:x}", Sha256::digest(spec.url.as_bytes()));
        self.root.join("urls").join(key).join(spec.file_name())
    }

    /// The sha256 of a cached URL file. With a sidecar that names this URL: its recorded sha256 when the size and
    /// modification time are the ones recorded, and otherwise the file is hashed again and must still be that
    /// sha256 (a cached copy that changed since it was downloaded is refused, never served as the URL's). With no
    /// sidecar for this URL, the file is hashed and the result recorded.
    fn url_entry(&self, spec: &UrlSpec, path: &Path) -> Result<Option<String>, String> {
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let record = fs::read_to_string(Self::sidecar(path))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .filter(|r| r["url"] == spec.url.as_str());
        let Some(record) = record else {
            let (_, sha) = hash_file(path)?;
            Self::write_url_sidecar(path, spec, &sha)?;
            return Ok(Some(sha));
        };
        let recorded = record["sha256"].as_str().unwrap_or("").to_string();
        let unchanged = record["size"] == meta.len()
            && mtime_ns(&meta).is_some_and(|m| record["mtime_ns"].as_str() == Some(m.to_string().as_str()));
        if unchanged {
            return Ok(Some(recorded));
        }
        let (_, sha) = hash_file(path)?;
        if sha != recorded {
            return Err(format!(
                "{}: the cached copy at {} changed since it was downloaded (sha256 {sha}, recorded {recorded}); \
                 delete it to download again",
                spec.url,
                path.display()
            ));
        }
        Self::write_url_sidecar(path, spec, &sha)?;
        Ok(Some(sha))
    }

    fn write_url_sidecar(path: &Path, spec: &UrlSpec, sha256: &str) -> Result<(), String> {
        let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let Some(mtime) = mtime_ns(&meta) else {
            return Ok(());
        };
        let record = serde_json::json!({
            "format": "ggtts-cache-url/1", "url": spec.url, "sha256": sha256,
            "size": meta.len(), "mtime_ns": mtime.to_string(),
        });
        let sidecar = Self::sidecar(path);
        let tmp = sidecar.with_extension(format!("verified.tmp-{}", unique_suffix()));
        fs::write(&tmp, format!("{record}\n")).map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::rename(&tmp, &sidecar).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("{}: {e}", sidecar.display())
        })
    }

    /// The file `spec` names, from the cache, downloading it first when it is not there (never when `offline`: then
    /// a missing file is an error naming the URL and where it would be). A URL is downloaded once and the cached file
    /// reused; a pin refuses a file whose sha256 differs, cached or downloaded, and a downloaded file that differs
    /// is removed. `say` gets what is downloaded and the sha256 either way.
    pub fn fetch_url(
        &self,
        spec: &UrlSpec,
        offline: bool,
        fetch: &dyn Fetch,
        say: &mut dyn FnMut(&str),
    ) -> Result<UrlFile, String> {
        let path = self.path_for_url(spec);
        let pinned = |sha: &str| match &spec.sha256 {
            Some(want) if want != sha => Err(format!("{}: sha256 {sha}, not the pinned {want}", spec.url)),
            _ => Ok(()),
        };
        if let Some(sha) = self.url_entry(spec, &path)? {
            pinned(&sha).map_err(|e| {
                format!(
                    "{e} (the cached copy at {}; delete it to download again)",
                    path.display()
                )
            })?;
            say(&format!("{}: sha256 {sha}, cached at {}", spec.url, path.display()));
            return Ok(UrlFile {
                path,
                sha256: sha,
                downloaded: false,
            });
        }
        if offline {
            return Err(format!(
                "{} is not in the cache ({}) and offline mode does not download it",
                spec.url,
                path.display()
            ));
        }
        let dir = path.parent().expect("a cache path has a directory");
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = spec.file_name();
        let tmp = dir.join(format!("{name}.partial-{}", unique_suffix()));
        let _guard = DownloadGuard::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let partial = Partial(tmp.clone());
        let mut out = Some(create_locked(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?);
        say(&format!("downloading {} into {}", spec.url, path.display()));
        let mut body = fetch
            .open(&spec.url)
            .map_err(|e| format!("downloading {} failed: {e}", spec.url))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut n = 0u64;
        loop {
            let read = body
                .read(&mut buf)
                .map_err(|e| format!("{}: download interrupted after {n} bytes: {e}", spec.url))?;
            if read == 0 {
                break;
            }
            n += read as u64;
            if n > URL_FILE_MAX {
                return Err(format!(
                    "{}: more than {} bytes; refused",
                    spec.url,
                    human(URL_FILE_MAX)
                ));
            }
            hasher.update(&buf[..read]);
            out.as_mut()
                .expect("open until the rename")
                .write_all(&buf[..read])
                .map_err(|e| format!("{}: {e}", tmp.display()))?;
        }
        out.as_ref()
            .expect("open until the rename")
            .sync_all()
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        let sha = format!("{:x}", hasher.finalize());
        pinned(&sha).map_err(|e| format!("{e}; the download was refused and removed"))?;
        if cfg!(windows) {
            out = None;
        }
        let renamed = fs::rename(&tmp, &path);
        drop(out);
        let sha = match renamed {
            Ok(()) => {
                Self::write_url_sidecar(&path, spec, &sha)?;
                sha
            }
            // Another process may have put its copy there first (on Windows a file in use cannot be replaced).
            Err(e) => match self.url_entry(spec, &path)? {
                Some(theirs) => {
                    pinned(&theirs)?;
                    theirs
                }
                None => return Err(format!("{}: {e}", path.display())),
            },
        };
        drop(partial);
        say(&format!(
            "{}: {}, sha256 {sha}, stored at {}",
            spec.url,
            human(n),
            path.display()
        ));
        Ok(UrlFile {
            path,
            sha256: sha,
            downloaded: true,
        })
    }
}

/// The files a render uses: `given[i]` when it is a path, else `defaults.files[i]` from the cache, downloading what
/// is missing first. `given` is in the order of `defaults.files`. Offline, a missing file is an error naming it and
/// its URL. `say` receives what is about to be downloaded, from where, how much and where it goes, before it starts.
pub fn resolve(
    defaults: &Defaults,
    given: &[Option<PathBuf>],
    cache: &Cache,
    offline: bool,
    fetch: &dyn Fetch,
    say: &mut dyn FnMut(&str),
) -> Result<Vec<ModelFile>, String> {
    if given.len() != defaults.files.len() {
        return Err(format!(
            "{}: {} paths for {} default files",
            defaults.engine,
            given.len(),
            defaults.files.len()
        ));
    }
    let mut missing = Vec::new();
    for (pinned, path) in defaults.files.iter().zip(given) {
        if path.is_some() {
            continue;
        }
        cache.remove_abandoned_partials(pinned);
        if !cache.check(pinned, say)? {
            missing.push(pinned);
        }
    }
    if !missing.is_empty() {
        if offline {
            let list: Vec<String> = missing
                .iter()
                .map(|p| {
                    format!(
                        "{} {} is not in the cache ({}) and offline mode does not download it from {}",
                        p.role,
                        p.file_name,
                        cache.path_for(p).display(),
                        p.url
                    )
                })
                .collect();
            return Err(list.join("; "));
        }
        let total: u64 = missing.iter().map(|p| p.size).sum();
        say(&format!(
            "downloading {} file(s) of {}, {} in all, into {}",
            missing.len(),
            defaults.release,
            human(total),
            cache.root().display()
        ));
        for p in &missing {
            say(&format!(
                "  {} {}, {}, from {} to {}",
                p.role,
                p.file_name,
                human(p.size),
                p.url,
                cache.path_for(p).display()
            ));
        }
        for p in missing {
            cache.download(p, fetch, say)?;
        }
    }
    Ok(defaults
        .files
        .iter()
        .zip(given)
        .map(|(pinned, path)| match path {
            Some(path) => ModelFile::given(pinned.role, path.clone()),
            None => ModelFile {
                role: pinned.role.to_string(),
                path: cache.path_for(pinned),
                source: Source::Cache,
                sha256: Some(pinned.sha256.to_string()),
                url: Some(pinned.url.to_string()),
            },
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_release_has_a_unique_name_that_is_no_file_name_and_every_default_is_a_release() {
        let mut names: Vec<&str> = RELEASES.iter().map(|r| r.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), RELEASES.len(), "two releases share a name");
        for r in RELEASES {
            assert!(looks_like_a_name(r.name), "{} reads as a file", r.name);
            assert_eq!(named(r.name).map(|n| n.name), Some(r.name));
            assert!(
                !r.files.is_empty() && r.size() > 0 && !r.licences().is_empty(),
                "{}",
                r.name
            );
        }
        for d in ALL {
            assert!(
                named(d.name).is_some(),
                "{}: a default that is not a named release",
                d.engine
            );
            assert!(is_default(d));
        }
        assert_eq!(defaults_for("qwen3-tts").map(|d| d.name), Some("hk-cantonese-1.7b"));
    }

    #[test]
    fn select_takes_a_name_a_path_or_nothing_and_refuses_the_rest_by_name() {
        let p = |s: &str| PathBuf::from(s);
        let name = |engine: &str, model: &str| select(engine, Some(&p(model))).map(|r| r.map(|r| r.name));
        assert!(matches!(select("auto", None), Ok(None)));
        for engine in ["auto", "qwen3-tts"] {
            let r = select(engine, Some(&p("hk-cantonese-1.7b"))).unwrap().unwrap();
            assert_eq!(r.name, "hk-cantonese-1.7b");
        }
        let other = select("kitten-tts-2", Some(&p("hk-cantonese-1.7b"))).unwrap_err();
        assert!(
            other.contains("is a qwen3-tts release") && other.contains("kitten-tts-2"),
            "{other}"
        );
        // Paths, existing or not, are paths: a separator, a .gguf or .json ending, or a drive.
        for path in [
            "./hk-cantonese-1.7b",
            "talker.gguf",
            "TALKER.GGUF",
            "voice.json",
            "C:x",
            "/no/such/m",
        ] {
            assert_eq!(name("auto", path), Ok(None), "{path}");
        }
        let unknown = select("auto", Some(&p("hk-cantonese-9b"))).unwrap_err();
        assert!(
            unknown.contains("hk-cantonese-9b") && unknown.contains(&known_names()),
            "{unknown}"
        );
        // An existing file with a name-like name is that file: the test runs in the crate's directory.
        assert!(Path::new("Cargo.toml").is_file());
        assert_eq!(name("auto", "Cargo.toml"), Ok(None));
    }

    #[test]
    fn the_listing_names_each_release_its_size_licence_cache_state_and_the_default() {
        let root = std::env::temp_dir().join(format!("ggtts-listing-{}", unique_suffix()));
        let lines = listing(&Cache::new(root.clone())).unwrap();
        assert_eq!(lines.len(), RELEASES.len());
        let first: Vec<&str> = lines[0].split('\t').collect();
        assert_eq!(
            first,
            [
                "hk-cantonese-1.7b",
                "qwen3-tts",
                "2.33 GB (2333984768 bytes)",
                "Apache-2.0",
                "not cached",
                "default"
            ]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn every_pinned_file_is_at_a_fixed_commit_with_a_full_hash() {
        for defaults in RELEASES {
            assert!(!defaults.files.is_empty(), "{}", defaults.engine);
            for p in defaults.files {
                let revision = p.url.split("/resolve/").nth(1).and_then(|r| r.split('/').next());
                let revision = revision.unwrap_or_else(|| panic!("{}: no revision in {}", p.file_name, p.url));
                assert!(
                    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{}: {revision} is not a commit",
                    p.file_name
                );
                assert!(p.url.starts_with("https://"), "{}", p.url);
                assert!(p.url.ends_with(&format!("/{}", p.file_name)), "{}", p.url);
                assert!(
                    p.sha256.len() == 64 && p.sha256.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                    "{}",
                    p.file_name
                );
                assert!(p.size > 0 && !p.licence.is_empty() && !p.role.is_empty());
            }
            if let Some(speaker) = defaults.default_speaker {
                assert!(defaults.speakers.contains(&speaker), "{}", defaults.engine);
            }
        }
        assert_eq!(defaults_for("qwen3-tts").map(|d| d.files.len()), Some(2));
        assert!(defaults_for("kitten-tts-2").is_none());
    }

    /// Subprocess participant: hold the guard while both the pre-lock staging
    /// name and a closed data file exist, simulating both vulnerable windows.
    #[test]
    fn guarded_partial_child() {
        let Ok(root) = std::env::var("GGTTS_TEST_GUARD_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let partial = root.join("model.gguf.partial-child");
        let _guard = DownloadGuard::create(&partial).unwrap();
        fs::write(root.join("model.gguf.partial-child.locking"), b"staging").unwrap();
        fs::write(&partial, b"closed but publishing").unwrap();
        fs::write(root.join("ready"), b"1").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !root.join("release").exists() {
            assert!(std::time::Instant::now() < deadline, "parent did not release child");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn cleanup_cannot_remove_live_staging_or_closed_partial_in_another_process() {
        let root = std::env::temp_dir().join(format!("ggtts-guard-{}", unique_suffix()));
        let cache = Cache::new(root.clone());
        let pinned = Pinned {
            role: "test",
            file_name: "model.gguf",
            url: "unused",
            size: 1,
            sha256: "test-hash",
            licence: "test",
        };
        let dir = cache.path_for(&pinned).parent().unwrap().to_path_buf();
        fs::create_dir_all(&dir).unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::guarded_partial_child", "--nocapture"])
            .env("GGTTS_TEST_GUARD_ROOT", &dir)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
        while !dir.join("ready").exists() {
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("child did not acquire its guard");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let partial = dir.join("model.gguf.partial-child");
        let stage = dir.join("model.gguf.partial-child.locking");
        let removed = cache.remove_abandoned_partials(&pinned);
        fs::write(dir.join("release"), b"1").unwrap();
        assert!(child.wait().unwrap().success());
        assert!(removed == 0, "cleanup removed a live writer's file");
        assert!(stage.exists() && partial.exists());
        assert_eq!(cache.remove_abandoned_partials(&pinned), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human(512), "512 bytes");
        assert_eq!(human(291_150_624), "291.15 MB (291150624 bytes)");
        assert_eq!(human(2_042_834_144), "2.04 GB (2042834144 bytes)");
    }
}
