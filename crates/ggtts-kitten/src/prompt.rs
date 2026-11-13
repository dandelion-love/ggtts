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
    /// The map in `meta`, refused unless every id it names exists in the model's `n_vocab` tokens ([`Self::checked`]).
    /// The vocabulary is a required argument so that no caller can take a map without the check.
    pub fn from_meta(meta: &std::collections::BTreeMap<String, Value>, n_vocab: usize) -> Result<Self, String> {
        let get = |k: &str| {
            meta.get(&format!("kitten.token_map.{k}"))
                .and_then(Value::as_u32)
                .ok_or(format!("token_map.{k} missing"))
        };
        let opt = |k: &str| get(k).unwrap_or(0);
        TokenMap {
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
        }
        .checked(n_vocab)
    }

    /// This map, if every id it names exists in a vocabulary of `n_vocab` tokens: the audio range
    /// `audio_id_base .. audio_id_base + num_audio_tokens` must end at or before `n_vocab`, and every id at or past
    /// none of it (an optional id left at 0 is absent, and not checked). The C++ reference refuses such a map when it
    /// loads; without this, an audio id past the vocabulary would index out of bounds in the sampler, and the
    /// restricted head would be asked for rows the head does not have.
    pub fn checked(self, n_vocab: usize) -> Result<Self, String> {
        let n = n_vocab as u64;
        let end = self.audio_base as u64 + self.audio_count as u64;
        if self.audio_count == 0 || end > n {
            return Err(format!(
                "kitten.token_map: audio_id_base {} + num_audio_tokens {} = {end}, which does not fit in the \
                 vocabulary of {n_vocab} tokens",
                self.audio_base, self.audio_count
            ));
        }
        let required = [
            ("speech_start_id", self.speech_start),
            ("speech_end_id", self.speech_end),
            ("text_start_id", self.text_start),
            ("start_id", self.start),
            ("stop_id", self.stop),
        ];
        let optional = [
            ("final_seg_id", self.final_seg),
            ("reference_text_start_id", self.ref_text_start),
            ("reference_text_end_id", self.ref_text_end),
            ("reference_speech_start_id", self.ref_speech_start),
            ("reference_speech_end_id", self.ref_speech_end),
        ];
        let ids = required.iter().chain(optional.iter().filter(|(_, id)| *id != 0));
        for (key, id) in ids {
            if *id as u64 >= n {
                return Err(format!(
                    "kitten.token_map.{key} {id} is at or past the end of the vocabulary of {n_vocab} tokens"
                ));
            }
        }
        Ok(self)
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

    /// The released LMs' map (as `sampler::tests::kitten_map`), in a vocabulary of 158,241 tokens.
    fn released() -> TokenMap {
        TokenMap {
            audio_base: 151_675,
            audio_count: 6561,
            speech_start: 151_669,
            speech_end: 151_670,
            text_start: 151_671,
            start: 151_673,
            stop: 151_674,
            final_seg: 158_236,
            ref_text_start: 158_239,
            ref_text_end: 158_240,
            ref_speech_start: 158_237,
            ref_speech_end: 158_238,
        }
    }

    fn meta(tm: &TokenMap) -> std::collections::BTreeMap<String, Value> {
        let keys = [
            ("audio_id_base", tm.audio_base),
            ("num_audio_tokens", tm.audio_count),
            ("speech_start_id", tm.speech_start),
            ("speech_end_id", tm.speech_end),
            ("text_start_id", tm.text_start),
            ("start_id", tm.start),
            ("stop_id", tm.stop),
            ("final_seg_id", tm.final_seg),
            ("reference_text_start_id", tm.ref_text_start),
            ("reference_text_end_id", tm.ref_text_end),
            ("reference_speech_start_id", tm.ref_speech_start),
            ("reference_speech_end_id", tm.ref_speech_end),
        ];
        keys.iter()
            .map(|(k, v)| (format!("kitten.token_map.{k}"), Value::U32(*v)))
            .collect()
    }

    #[test]
    fn reading_a_map_from_metadata_applies_the_vocabulary_check() {
        let ok = TokenMap::from_meta(&meta(&released()), 158_241).expect("the released map");
        assert_eq!((ok.audio_base, ok.stop, ok.ref_speech_end), (151_675, 151_674, 158_238));
        let e = TokenMap::from_meta(
            &meta(&TokenMap {
                audio_count: 6567,
                ..released()
            }),
            158_241,
        )
        .unwrap_err();
        assert!(e.contains("num_audio_tokens 6567"), "{e}");
        let e = TokenMap::from_meta(
            &meta(&TokenMap {
                stop: 158_241,
                ..released()
            }),
            158_241,
        )
        .unwrap_err();
        assert!(e.contains("kitten.token_map.stop_id 158241"), "{e}");
        // The same released map in a smaller vocabulary does not fit.
        let e = TokenMap::from_meta(&meta(&released()), 158_000).unwrap_err();
        assert!(e.contains("158000 tokens"), "{e}");
    }

    #[test]
    fn a_map_that_fits_its_vocabulary_is_accepted() {
        assert!(released().checked(158_241).is_ok());
        // The audio range may end exactly at the vocabulary's end.
        assert!(TokenMap {
            audio_count: 6566,
            ..released()
        }
        .checked(158_241)
        .is_ok());
        let tm = TokenMap {
            audio_base: 22,
            audio_count: 10,
            ..released()
        };
        let small = TokenMap {
            speech_start: 1,
            speech_end: 2,
            text_start: 3,
            start: 4,
            stop: 5,
            final_seg: 0,
            ref_text_start: 0,
            ref_text_end: 0,
            ref_speech_start: 0,
            ref_speech_end: 0,
            ..tm
        };
        assert!(small.checked(32).is_ok(), "optional ids left at 0 are absent");
    }

    #[test]
    fn an_audio_range_past_the_vocabulary_is_refused_naming_both_numbers() {
        let tm = TokenMap {
            audio_count: 6567,
            ..released()
        };
        let e = tm.checked(158_241).unwrap_err();
        assert!(
            e.contains("audio_id_base 151675") && e.contains("num_audio_tokens 6567"),
            "{e}"
        );
        assert!(e.contains("158242") && e.contains("158241 tokens"), "{e}");
        let e = TokenMap {
            audio_count: 0,
            ..released()
        }
        .checked(158_241)
        .unwrap_err();
        assert!(e.contains("num_audio_tokens 0"), "{e}");
        // u32 overflow is not a way past the check.
        let e = TokenMap {
            audio_base: u32::MAX,
            audio_count: 2,
            ..released()
        }
        .checked(158_241)
        .unwrap_err();
        assert!(e.contains("4294967297"), "{e}");
    }

    #[test]
    fn every_end_and_role_id_at_or_past_the_vocabulary_is_refused_by_name() {
        let n = 158_241usize;
        type Set = fn(&mut TokenMap, u32);
        let fields: [(&str, Set); 10] = [
            ("speech_start_id", |t, v| t.speech_start = v),
            ("speech_end_id", |t, v| t.speech_end = v),
            ("text_start_id", |t, v| t.text_start = v),
            ("start_id", |t, v| t.start = v),
            ("stop_id", |t, v| t.stop = v),
            ("final_seg_id", |t, v| t.final_seg = v),
            ("reference_text_start_id", |t, v| t.ref_text_start = v),
            ("reference_text_end_id", |t, v| t.ref_text_end = v),
            ("reference_speech_start_id", |t, v| t.ref_speech_start = v),
            ("reference_speech_end_id", |t, v| t.ref_speech_end = v),
        ];
        for (key, set) in fields {
            for id in [n as u32, n as u32 + 7] {
                let mut tm = released();
                set(&mut tm, id);
                let e = tm.checked(n).unwrap_err();
                assert!(
                    e.contains(&format!("kitten.token_map.{key} {id}")) && e.contains("158241 tokens"),
                    "{e}"
                );
            }
            let mut tm = released();
            set(&mut tm, n as u32 - 1);
            assert!(tm.checked(n).is_ok(), "{key} at the last id is inside the vocabulary");
        }
    }
}
