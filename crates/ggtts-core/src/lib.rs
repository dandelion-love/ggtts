//! ggtts core.
//!
//! The parts every engine shares: the engine interface ([`engine`]), the long-text story path ([`text`] for
//! reading and chunking, [`seed`] for the seed per chunk, [`audio`] for the judge and the trim, [`pipeline`] for
//! the redraw, the joins and the level), FLAC output ([`flac`]), in-process tensor hooks ([`hooks`]), and the
//! instrument events with the trace format, replay and the trace-identity check ([`trace`]), streaming output
//! ([`stream`], fed by [`pipeline::render_story_streamed`]), and the instruments' report envelope with its merger
//! ([`report`]), the steering a request can carry with its safetensors reader ([`steer`]), and the options value
//! every front end's options are defined in, with its table, schema and provenance record ([`options`]). The provenance manifest
//! comes next.
//!
//! The core holds the rules and no protocol number: the ceilings, the pause table, the judge bars and the level
//! target come from the [`profile::Profile`] the engine declares.
//!
//! The pipeline reproduces a reference implementation rule for rule, oddities included, and is checked against
//! its recorded cases in `fixtures/pipeline/` (`cargo test -p ggtts-core --test pipeline_cases`).

pub mod audio;
pub mod capabilities;
pub mod engine;
pub mod flac;
pub mod hooks;
pub mod logging;
pub mod numeric;
pub mod options;
pub mod pipeline;
pub mod profile;
pub mod report;
pub mod seed;
pub mod steer;
pub mod stream;
pub mod text;
pub mod textjudge;
pub mod trace;

/// The release tag embedded at build time, or the workspace version for ordinary local builds.
pub const VERSION: &str = match option_env!("GGTTS_RELEASE_VERSION") {
    Some(version) if !version.is_empty() => version,
    _ => env!("CARGO_PKG_VERSION"),
};

/// The ggtts commit this build is made from: `GITHUB_SHA` in a GitHub Actions build (the value BUILDINFO.json's
/// `ggtts_commit` records), else the checkout's `HEAD`, else `unknown` (see `build.rs`).
pub const SOURCE_COMMIT: &str = env!("GGTTS_COMMIT");

/// The ggtts repository, as `--version` prints it.
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

/// The ggml source this build is pinned to.
///
/// Exact token parity with the reference runtime needs the same ggml kernels, so the pin is a commit, not a
/// version range. It is the commit the reference runtime (qwen3-tts.cpp at `16bb5af`) builds against: ggml
/// v0.16.0 plus 7 commits. Moving it is a deliberate change that re-checks the parity fixtures.
pub mod ggml {
    /// Upstream repository.
    pub const REPOSITORY: &str = "https://github.com/ggml-org/ggml";
    /// Full commit SHA of the pinned ggml.
    pub const COMMIT: &str = "af97976c7810cdabb1863172f31c432dab767de7";
    /// Human-readable position of the pin relative to ggml's release tags.
    pub const DESCRIBE: &str = "v0.16.0+7";

    /// The first seven characters of [`COMMIT`], as printed in versions and artifact names.
    pub fn short() -> &'static str {
        &COMMIT[..7]
    }
}

/// One line per fact, as `--version` prints them: the binary and its release tag, the ggtts commit, the ggml pin.
pub fn version_lines(binary: &str) -> [String; 3] {
    [
        format!("{binary} {VERSION}"),
        format!("ggtts {SOURCE_COMMIT} ({REPOSITORY})"),
        format!("ggml {} ({}, {})", ggml::COMMIT, ggml::DESCRIBE, ggml::REPOSITORY),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_pin_is_a_full_commit_sha() {
        assert_eq!(ggml::COMMIT.len(), 40);
        assert!(ggml::COMMIT
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert_eq!(ggml::short(), "af97976");
    }

    #[test]
    fn version_names_binary_commit_and_pin() {
        let [first, second, third] = version_lines("ggtts-render");
        assert_eq!(first, format!("ggtts-render {VERSION}"));
        assert_eq!(second, format!("ggtts {SOURCE_COMMIT} ({REPOSITORY})"));
        assert!(third.contains(ggml::COMMIT));
        assert!(third.contains(ggml::DESCRIBE));
    }

    #[test]
    fn ggtts_commit_is_a_full_sha_or_unknown() {
        // build.rs writes `unknown` for a build with neither GITHUB_SHA nor git (a source tarball); a release build
        // refuses that, and the release smoke compares the printed commit with BUILDINFO's.
        assert!(
            SOURCE_COMMIT == "unknown"
                || (SOURCE_COMMIT.len() == 40
                    && SOURCE_COMMIT
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())),
            "GGTTS_COMMIT is {SOURCE_COMMIT:?}"
        );
        assert!(REPOSITORY.starts_with("https://"), "{REPOSITORY}");
    }
}
