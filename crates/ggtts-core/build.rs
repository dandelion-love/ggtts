//! Embeds the ggtts commit this build is made from, as `GGTTS_COMMIT`, for `--version`.
//!
//! The source is the one the release's BUILDINFO.json uses: `GITHUB_SHA`, which GitHub Actions sets for every step
//! of a workflow run. Outside Actions it is `git rev-parse HEAD` of the checkout; with neither (a source tarball),
//! `unknown`. A release build (`GGTTS_RELEASE_VERSION` set) refuses `unknown`.

use std::path::PathBuf;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn is_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn main() {
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");
    println!("cargo:rerun-if-env-changed=GGTTS_RELEASE_VERSION");
    let commit = std::env::var("GITHUB_SHA").ok().filter(|s| is_sha(s)).or_else(|| {
        // Rebuild when the checkout moves. HEAD lives in the worktree's own git directory; the branch it points at
        // and packed-refs live in the common one, which differs from it in a linked worktree.
        let dir = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from);
        let common = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"])
            .map(PathBuf::from)
            .or_else(|| dir.clone());
        if let (Some(dir), Some(common)) = (dir, common) {
            println!("cargo:rerun-if-changed={}", dir.join("HEAD").display());
            println!("cargo:rerun-if-changed={}", common.join("packed-refs").display());
            if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
                println!("cargo:rerun-if-changed={}", common.join(r).display());
            }
        }
        git(&["rev-parse", "HEAD"]).filter(|s| is_sha(s))
    });
    let release = std::env::var("GGTTS_RELEASE_VERSION").unwrap_or_default();
    if commit.is_none() && !release.is_empty() {
        panic!(
            "GGTTS_RELEASE_VERSION is {release:?} but the ggtts commit is unknown: set GITHUB_SHA or build from a git \
             checkout, so --version can name the commit BUILDINFO.json records"
        );
    }
    println!(
        "cargo:rustc-env=GGTTS_COMMIT={}",
        commit.as_deref().unwrap_or("unknown")
    );
}
