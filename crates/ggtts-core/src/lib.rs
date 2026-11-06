//! ggtts core.
//!
//! The parts every engine shares: the engine interface ([`engine`]), the long-text story path ([`text`] for
//! reading and chunking, [`seed`] for the seed per chunk, [`audio`] for the judge and the trim, [`pipeline`] for
//! the redraw, the joins and the level), FLAC output ([`flac`]), and the instrument events with the trace format,
//! replay and the trace-identity check ([`trace`]), streaming output ([`stream`], fed by
//! [`pipeline::render_story_streamed`]), and the instruments' report envelope with its merger ([`report`]). The
//! provenance manifest comes next.
//!
//! The core holds the rules and no protocol number: the ceilings, the pause table, the judge bars and the level
//! target come from the [`profile::Profile`] the engine declares.
//!
//! The pipeline reproduces a reference implementation rule for rule, oddities included, and is checked against
//! its recorded cases in `fixtures/pipeline/` (`cargo test -p ggtts-core --test pipeline_cases`).

pub mod audio;
pub mod engine;
pub mod flac;
pub mod numeric;
pub mod pipeline;
pub mod profile;
pub mod report;
pub mod seed;
pub mod stream;
pub mod text;
pub mod trace;

/// The ggtts version, from the workspace manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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

/// One line per fact, as `--version` prints them.
pub fn version_lines(binary: &str) -> [String; 2] {
    [
        format!("{binary} {VERSION}"),
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
    fn version_names_binary_and_pin() {
        let [first, second] = version_lines("ggtts-render");
        assert_eq!(first, format!("ggtts-render {VERSION}"));
        assert!(second.contains(ggml::COMMIT));
        assert!(second.contains(ggml::DESCRIBE));
    }
}
