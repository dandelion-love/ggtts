//! The Qwen3-TTS engine.
//!
//! Always present: the engine's protocol profile, [`PROFILE`], the numbers the story path renders Qwen3-TTS output
//! with. They are the reference implementation's `story-v1` defaults, and the pipeline cases in `fixtures/pipeline/`
//! were recorded with them.
//!
//! With the `ggml` feature: the model over ggml, ported from qwen3-tts.cpp at `16bb5af` op for op, so that on the
//! CPU backend its outputs are bit-identical to the reference's. Each module names the reference file and lines it
//! follows. [`codec::Decoder`] turns codec tokens into samples.

use ggtts_core::profile::Profile;

#[cfg(feature = "ggml")]
pub mod codec;
#[cfg(feature = "ggml")]
pub mod cpu;
#[cfg(feature = "ggml")]
pub mod engine;
pub mod fixtures;
#[cfg(feature = "ggml")]
pub mod gguf;
#[cfg(feature = "ggml")]
pub mod talker;
pub mod tokenizer;

/// `story-v1/qwen3-tts`: the story path's numbers for Qwen3-TTS (render spec sections 1 to 7).
///
/// Every number is part of the protocol. Changing one is a new protocol version, with a changed set of pipeline
/// cases.
pub const PROFILE: Profile = Profile {
    name: "story-v1/qwen3-tts",
    protocol: "story-v1",
    story_ceiling: 45,
    passage_ceiling: 26,
    min_chunk_chars: 4,
    // ASCII punctuation, quotes and `……` are not split points.
    split_marks: &['，', '。', '、', '；', '：', '！', '？'],
    unmarked_joiner: '，',
    // Zero: every chunk is its first draw, defects and all.
    redraw: 0,
    edge_s: 0.12,
    top_db: 40.0,
    runaway_spc: 0.60,
    silent_spc: 0.07,
    pauses: &[
        ("，", 0.20),
        ("、", 0.16),
        ("：", 0.30),
        ("；", 0.32),
        ("。", 0.48),
        ("！", 0.48),
        ("？", 0.50),
    ],
    default_pause_s: 0.18,
    para_extra_s: 0.45,
    target_rms_db: -20.0,
    level_floor: 1e-3,
    clip: 0.999,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ggtts_core::text::chunk;

    #[test]
    fn merge_keeps_the_first_clause_mark() {
        let got = chunk(&PROFILE, "今日好熱。我哋去游水！", PROFILE.story_ceiling);
        assert_eq!(got, vec![("今日好熱。我哋去游水".to_string(), "！".to_string())]);
    }

    #[test]
    fn pause_table_reads_by_mark() {
        assert_eq!(PROFILE.pause_s("。"), 0.48);
        assert_eq!(PROFILE.pause_s("！"), 0.48);
        assert_eq!(PROFILE.pause_s("？"), 0.50);
        assert_eq!(PROFILE.pause_s(""), 0.18);
        assert_eq!(PROFILE.pause_s("."), 0.18);
    }
}
