//! Text in: the story file, clause chunking at a ceiling, and the chunk plan (render spec sections 1 and 2).
//!
//! The corrected story path preserves short tails and consecutive punctuation; punctuation-only input is refused.
//!
//! The split marks, the joiner, the minimum length and the ceilings come from the [`Profile`].

use crate::profile::Profile;
use unicode_general_category::{get_general_category, GeneralCategory};

/// True when a character has Unicode punctuation category (P*).
fn punctuation(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

/// Punctuation without spoken content is not a model request.
pub fn validate_speech(text: &str) -> Result<(), &'static str> {
    let mut chars = text.chars().filter(|c| !is_strip_space(*c));
    if let Some(first) = chars.next() {
        if punctuation(first) && chars.all(punctuation) {
            return Err("punctuation-only input cannot be rendered as speech");
        }
    }
    Ok(())
}

/// Whitespace as the reference strips it: Unicode `White_Space`, plus the four ASCII separators U+001C..U+001F.
pub fn is_strip_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Strip leading and trailing whitespace, by [`is_strip_space`].
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_strip_space)
}

/// A story file: the first line is the title, every later non-empty line is a paragraph, each stripped.
/// Line endings `\r\n` and `\r` read as `\n`. No other normalisation.
pub fn read_text(contents: &str) -> (String, Vec<String>) {
    let normalised = contents.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = strip(&normalised).split('\n');
    let title = strip(lines.next().unwrap_or("")).to_string();
    let paragraphs = lines.map(strip).filter(|l| !l.is_empty()).map(str::to_string).collect();
    (title, paragraphs)
}

/// Split one paragraph into `(text, mark)` pairs no longer than `ceiling` code points (the mark not counted).
///
/// Clauses end at one of the profile's [`split_marks`](Profile::split_marks). Neighbouring clauses merge while the
/// merge, counting a one-character joiner, fits the ceiling; the first clause keeps its own mark as the joiner (or
/// the profile's [`unmarked_joiner`](Profile::unmarked_joiner) if it has none). A clause longer than the ceiling is
/// hard-cut, and the cut pieces carry no mark. Short tails are retained.
///
/// # Panics
/// If `ceiling` is zero.
pub fn chunk(profile: &Profile, text: &str, ceiling: usize) -> Result<Vec<(String, String)>, &'static str> {
    validate_speech(text)?;
    assert!(ceiling > 0, "the chunk ceiling must be at least 1");
    let mut parts: Vec<(String, String)> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if profile.split_marks.contains(&ch) {
            parts.push((strip(&buf).to_string(), ch.to_string()));
            buf.clear();
        } else {
            buf.push(ch);
        }
    }
    let tail = strip(&buf);
    if !tail.is_empty() {
        parts.push((tail.to_string(), String::new()));
    }

    let mut combined: Vec<(String, String)> = Vec::new();
    let mut leading = String::new();
    for (body, mark) in parts {
        if body.is_empty() {
            if let Some((_, previous_mark)) = combined.last_mut() {
                previous_mark.push_str(&mark);
            } else {
                leading.push_str(&mark);
            }
        } else {
            combined.push((format!("{leading}{body}"), mark));
            leading.clear();
        }
    }
    if !leading.is_empty() {
        combined.push((String::new(), leading));
    }
    let mut out: Vec<(String, String)> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    let mut cur_mark = String::new();
    for (part, mark) in combined {
        if part.is_empty() {
            out.push((part, mark));
            continue;
        }
        let part: Vec<char> = part.chars().collect();
        if cur.is_empty() {
            cur = part;
            cur_mark = mark;
        } else if cur.len() + part.len() + cur_mark.chars().count().max(1) <= ceiling {
            if cur_mark.is_empty() {
                cur.push(profile.unmarked_joiner);
            } else {
                cur.extend(cur_mark.chars());
            }
            cur.extend(part);
            cur_mark = mark;
        } else {
            out.push((cur.iter().collect(), cur_mark));
            cur = part;
            cur_mark = mark;
        }
        while cur.len() > ceiling {
            out.push((cur[..ceiling].iter().collect(), String::new()));
            cur = cur[ceiling..].to_vec();
        }
    }
    if !cur.is_empty() {
        out.push((cur.iter().collect(), cur_mark));
    }
    let mut speech: Vec<(String, String)> = Vec::new();
    for (body, mark) in out {
        if validate_speech(&format!("{body}{mark}")).is_err() {
            if let Some((_, prior_mark)) = speech.last_mut() {
                prior_mark.push_str(&body);
                prior_mark.push_str(&mark);
            } else {
                return Err("punctuation-only chunk cannot be rendered as speech");
            }
        } else {
            speech.push((body, mark));
        }
    }
    Ok(speech)
}

/// One chunk of a story, in reading order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedChunk {
    /// The paragraph's index in the story.
    pub para: usize,
    /// The text the engine reads: the chunk's text followed by its mark.
    pub text: String,
    /// The mark that ended the chunk, or empty for a hard cut or an unmarked end.
    pub punct: String,
    /// Whether this is the paragraph's last chunk.
    pub last_in_para: bool,
}

/// Chunk every paragraph at `ceiling`, by the profile's rules, and list the chunks in reading order.
pub fn plan(profile: &Profile, paragraphs: &[String], ceiling: usize) -> Result<Vec<PlannedChunk>, &'static str> {
    let mut out = Vec::new();
    for (para, text) in paragraphs.iter().enumerate() {
        let chunks = chunk(profile, text, ceiling)?;
        let n = chunks.len();
        for (ci, (body, punct)) in chunks.into_iter().enumerate() {
            out.push(PlannedChunk {
                para,
                text: format!("{body}{punct}"),
                punct,
                last_in_para: ci + 1 == n,
            });
        }
    }
    Ok(out)
}

/// Whether a code point counts as one CJK character: U+3400..U+9FFF or U+20000..U+2FFFF, inclusive.
pub fn is_cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c) || ('\u{20000}'..='\u{2ffff}').contains(&c)
}

/// The number of CJK characters in `s`, by [`is_cjk`].
pub fn cjk_len(s: &str) -> usize {
    s.chars().filter(|&c| is_cjk(c)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_matches_reference_whitespace() {
        assert_eq!(strip("\u{1c}\u{3000} 文字 \t\n"), "文字");
    }
}
