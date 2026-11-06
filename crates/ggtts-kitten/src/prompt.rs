//! The prompt: Kitten's token layout and the order of a generation prompt (`kittenml/prompt.py:10-56`,
//! `cpp/tools/kitten-tts/sampling.h:22-35`).

use crate::gguf::Value;

/// Kitten's token layout, from the GGUF's metadata.
#[derive(Debug, Clone, Copy)]
pub struct TokenMap {
    pub audio_base: u32,
    pub audio_count: u32,
    pub speech_start: u32,
    pub speech_end: u32,
    pub text_start: u32,
    pub start: u32,
    pub stop: u32,
    pub final_seg: u32,
    pub ref_text_start: u32,
    pub ref_text_end: u32,
    pub ref_speech_start: u32,
    pub ref_speech_end: u32,
}

impl TokenMap {
    pub fn from_meta(meta: &std::collections::BTreeMap<String, Value>) -> Result<Self, String> {
        let get = |k: &str| {
            meta.get(&format!("kitten.token_map.{k}"))
                .and_then(Value::as_u32)
                .ok_or(format!("token_map.{k} missing"))
        };
        let opt = |k: &str| get(k).unwrap_or(0);
        Ok(TokenMap {
            audio_base: get("audio_id_base")?,
            audio_count: get("num_audio_tokens")?,
            speech_start: get("speech_start_id")?,
            speech_end: get("speech_end_id")?,
            text_start: get("text_start_id")?,
            start: get("start_id")?,
            stop: get("stop_id")?,
            final_seg: opt("final_seg_id"),
            ref_text_start: opt("reference_text_start_id"),
            ref_text_end: opt("reference_text_end_id"),
            ref_speech_start: opt("reference_speech_start_id"),
            ref_speech_end: opt("reference_speech_end_id"),
        })
    }

    /// The prompt for one chunk: the reference prefix (if any), the emotion control, the target text.
    pub fn prompt(
        &self,
        text: &[u32],
        ref_text: &[u32],
        ref_audio: &[u32],
        emotion: &[u32],
        reference: bool,
    ) -> Vec<u32> {
        let mut p = vec![self.start];
        if reference {
            let roles = self.ref_text_start != 0;
            p.push(if roles { self.ref_text_start } else { self.text_start });
            p.extend_from_slice(ref_text);
            if roles {
                p.push(self.ref_text_end);
            }
            p.push(if roles {
                self.ref_speech_start
            } else {
                self.speech_start
            });
            p.extend(ref_audio.iter().map(|&t| self.audio_base + t));
            p.push(if roles { self.ref_speech_end } else { self.speech_end });
        }
        p.extend_from_slice(emotion);
        p.push(self.text_start);
        p.extend_from_slice(text);
        if self.final_seg != 0 {
            p.push(self.final_seg);
        }
        p.push(self.speech_start);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_order_matches_kittenml() {
        let tm = TokenMap {
            audio_base: 1000,
            audio_count: 10,
            speech_start: 1,
            speech_end: 2,
            text_start: 3,
            start: 4,
            stop: 5,
            final_seg: 6,
            ref_text_start: 7,
            ref_text_end: 8,
            ref_speech_start: 9,
            ref_speech_end: 10,
        };
        let p = tm.prompt(&[50, 51], &[60], &[0, 3], &[70], true);
        assert_eq!(p, vec![4, 7, 60, 8, 9, 1000, 1003, 10, 70, 3, 50, 51, 6, 1]);
        assert_eq!(tm.prompt(&[50], &[], &[], &[], false), vec![4, 3, 50, 6, 1]);
    }
}
