//! The protocol profile: every number the story path reads, declared by an engine.
//!
//! The core holds the rules (how to chunk, judge, trim, join and level) and no protocol number. The numbers that
//! tune those rules for one architecture (the chunk ceilings, the pause table, the judge bars, the level target)
//! live in a [`Profile`] that the engine declares through [`Engine::profile`](crate::engine::Engine::profile).
//! A profile is named `{protocol}/{family}`, for example `story-v1/qwen3-tts`. Changing a number in a published
//! profile is a new protocol version, with a changed set of pipeline cases.
//!
//! The fields are public and the type is `Copy`, so a test can derive a profile from another with struct update
//! syntax, for example a planted ceiling: `Profile { story_ceiling: 44, ..base }`.

/// The numbers of one protocol for one architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Profile {
    /// The profile's name, `{protocol}/{family}`, for provenance.
    pub name: &'static str,
    /// The protocol version these numbers belong to, for example `story-v1`.
    pub protocol: &'static str,

    // ── chunking (render spec sections 1 and 2) ──
    /// The story path's chunk ceiling, in code points.
    pub story_ceiling: usize,
    /// The passage path's ceiling. Not a mode: a caller passes it explicitly.
    pub passage_ceiling: usize,
    /// Historical profile minimum; retained for compatibility, but no longer drops short text.
    pub min_chunk_chars: usize,
    /// The clause marks the chunker splits on.
    pub split_marks: &'static [char],
    /// The joiner inserted when two clauses merge and the first carries no mark.
    pub unmarked_joiner: char,

    // ── redraw, trim and judge (sections 4 to 6) ──
    /// Redraws of a chunk the judge rejects, by default.
    pub redraw: u32,
    /// Silence kept at each edge of a chunk after trimming, in seconds.
    pub edge_s: f64,
    /// A frame is loud when its RMS is less than this many dB below the loudest frame's, strictly.
    pub top_db: f64,
    /// Above this many seconds of speech per character, a draw is a runaway (strict).
    pub runaway_spc: f64,
    /// Below this many seconds of speech per character, a draw is near-silent (strict).
    pub silent_spc: f64,

    // ── joins (section 7) ──
    /// The pause after a chunk, by the mark that ended it, in seconds.
    pub pauses: &'static [(&'static str, f64)],
    /// The pause after a chunk whose mark is not in [`pauses`](Self::pauses), or that has none, in seconds.
    pub default_pause_s: f64,
    /// Extra pause after a paragraph's last chunk, in seconds.
    pub para_extra_s: f64,

    // ── level ──
    /// The read's loudness target: RMS over the samples above [`level_floor`](Self::level_floor), in dBFS.
    pub target_rms_db: f64,
    /// Samples at or below this magnitude are not measured by the level step.
    pub level_floor: f32,
    /// The level step clips to this magnitude.
    pub clip: f64,
}

impl Profile {
    /// Use the first recognized punctuation in a retained multi-mark suffix;
    /// unknown suffixes use the default pause.
    pub fn pause_s(&self, punct: &str) -> f64 {
        if let Some((_, seconds)) = self.pauses.iter().find(|(mark, _)| *mark == punct) {
            return *seconds;
        }
        for ch in punct.chars() {
            if let Some((_, seconds)) = self.pauses.iter().find(|(mark, _)| {
                let mut chars = mark.chars();
                chars.next() == Some(ch) && chars.next().is_none()
            }) {
                return *seconds;
            }
        }
        self.default_pause_s
    }
}
