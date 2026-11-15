//! Opening a model file that arrives as an already-open descriptor.
//!
//! Android gives an app a file the user picked (a document in a folder chosen with the system picker) as an open
//! descriptor, not a path it may open. Native code sees it as `/proc/self/fd/<n>`, but opening that path makes the
//! kernel check access to the file underneath, which an app without a storage permission may not read: the open fails
//! with "Permission denied" even though the descriptor itself is readable. So a path of exactly that form is taken as
//! the descriptor it names: duplicated, rewound, and never reopened by path. Any other path opens as usual.
//!
//! A duplicate shares its file offset with the original descriptor, so every reader here rewinds first and reads
//! sequentially; loads of one engine run one after another, never on the same descriptor at once.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// The descriptor a path names, if it is exactly `/proc/self/fd/<n>`.
pub fn fd_of(path: &Path) -> Option<i32> {
    let s = path.to_str()?;
    let n = s.strip_prefix("/proc/self/fd/")?;
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    n.parse().ok()
}

/// Opens `path` for reading at its start. `/proc/self/fd/<n>` is descriptor `n`, duplicated and rewound.
pub fn open(path: &Path) -> io::Result<File> {
    match fd_of(path) {
        Some(fd) => dup(fd),
        None => File::open(path),
    }
}

/// [`open`], then the whole file.
pub fn read(path: &Path) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    open(path)?.read_to_end(&mut out)?;
    Ok(out)
}

#[cfg(unix)]
fn dup(fd: i32) -> io::Result<File> {
    use std::os::fd::{BorrowedFd, OwnedFd};
    if fd < 0 {
        return Err(io::Error::from_raw_os_error(9)); // EBADF
    }
    // SAFETY: the borrow lasts only for the duplicating fcntl, which fails with EBADF if fd is not open.
    let owned: OwnedFd = unsafe { BorrowedFd::borrow_raw(fd) }.try_clone_to_owned()?;
    let mut f = File::from(owned);
    // A model is read at offsets, so a descriptor that cannot seek (a pipe, as some document providers give for a
    // file they stream from elsewhere) is refused here with what to do, never read wrongly.
    f.seek(SeekFrom::Start(0)).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "descriptor {fd} does not seek ({e}): its provider gives no seekable file, so copy the file into the \
                 app's own folder"
            ),
        )
    })?;
    Ok(f)
}

#[cfg(not(unix))]
fn dup(_fd: i32) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "/proc/self/fd paths exist only on Unix",
    ))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::PermissionsExt;

    fn temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ggtts-fdpath-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    #[test]
    fn only_an_exact_proc_self_fd_path_names_a_descriptor() {
        assert_eq!(fd_of(Path::new("/proc/self/fd/7")), Some(7));
        assert_eq!(fd_of(Path::new("/proc/self/fd/144")), Some(144));
        for p in [
            "/proc/self/fd/",
            "/proc/self/fd/7a",
            "/proc/self/fd/7/x",
            "/proc/1/fd/7",
            "proc/self/fd/7",
            "/data/x.gguf",
        ] {
            assert_eq!(fd_of(Path::new(p)), None, "{p}");
        }
    }

    /// Gives the file its mode back however the test ends, so a failed assertion leaves no mode-000 file behind.
    struct Restore(std::path::PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o600));
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// The case Android makes: the descriptor is readable, the file behind it is not. A mutant that reopens the path
    /// (`File::open` on `/proc/self/fd/<n>`) gets "Permission denied" here and fails the test. Root reads any file,
    /// so the test says so rather than passing for the wrong reason.
    #[test]
    fn a_descriptor_whose_file_may_not_be_opened_is_read_without_reopening() {
        let p = temp("model.gguf", b"GGUFsome bytes");
        let _restore = Restore(p.clone());
        let held = File::open(&p).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o000)).unwrap();
        let fd_path = std::path::PathBuf::from(format!("/proc/self/fd/{}", held.as_raw_fd()));
        let reopen = File::open(&fd_path);
        if reopen.is_ok() {
            eprintln!("skipped: this user reads a mode-000 file (root?), so the reopen cannot be told apart");
            return;
        }
        assert_eq!(reopen.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(read(&fd_path).unwrap(), b"GGUFsome bytes");
        // Read twice: each open rewinds, though the duplicates share one offset.
        let mut magic = [0u8; 4];
        open(&fd_path).unwrap().read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"GGUF");
        assert_eq!(read(&fd_path).unwrap(), b"GGUFsome bytes");
    }

    /// A descriptor that cannot seek (a pipe) fails with an error saying so and what to do, and is not read.
    #[test]
    fn a_descriptor_that_cannot_seek_is_refused_with_what_to_do() {
        let (reader, mut writer) = std::io::pipe().unwrap();
        writer.write_all(b"GGUFstreamed").unwrap();
        drop(writer);
        let fd_path = std::path::PathBuf::from(format!("/proc/self/fd/{}", reader.as_raw_fd()));
        let e = read(&fd_path).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("does not seek"), "{msg}");
        assert!(msg.contains("copy the file into the app's own folder"), "{msg}");
    }

    #[test]
    fn a_plain_path_opens_as_usual_and_a_closed_descriptor_fails() {
        let p = temp("plain.json", b"{}");
        assert_eq!(read(&p).unwrap(), b"{}");
        assert!(read(Path::new("/proc/self/fd/987654")).is_err());
    }
}
