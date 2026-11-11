//! Embeds the ggtts commit this build is made from, as `GGTTS_COMMIT`, for `--version`.
//!
//! The source is the one the release's BUILDINFO.json uses: `GITHUB_SHA`, which GitHub Actions sets for every step
//! of a workflow run. Outside Actions it is `git rev-parse HEAD` of the checkout; with neither (a source tarball),
//! `unknown`.

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
    let from_env = std::env::var("GITHUB_SHA").ok().filter(|s| is_sha(s));
    let commit = from_env.clone().or_else(|| {
        // Rebuild when the checkout moves: HEAD itself, the branch it points at, and packed refs.
        if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from) {
            for f in ["HEAD", "packed-refs"] {
                println!("cargo:rerun-if-changed={}", dir.join(f).display());
            }
            if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
                println!("cargo:rerun-if-changed={}", dir.join(r).display());
            }
        }
        git(&["rev-parse", "HEAD"]).filter(|s| is_sha(s))
    });
    println!(
        "cargo:rustc-env=GGTTS_COMMIT={}",
        commit.as_deref().unwrap_or("unknown")
    );
}
