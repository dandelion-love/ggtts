//! Text in: the story file, clause chunking at a ceiling, and the chunk plan (render spec sections 1 and 2).
//!
//! The rules are reproduced as they are, oddities included, so that parity holds: a mark on an empty clause is
//! lost, a hard cut carries no mark, and a chunk under four code points is dropped. A fix is a new protocol version.

/// The clause marks the chunker splits on. ASCII punctuation, quotes and `……` are not split points.
pub const SPLIT: [char; 7] = ['，', '。', '、', '；', '：', '！', '？'];

/// A chunk whose text (without its mark) is shorter than this many code points is dropped.
pub const MIN_CHUNK_CHARS: usize = 4;

/// The joiner inserted when two clauses merge and the first carries no mark.
const UNMARKED_JOINER: char = '，';

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
/// Clauses end at a [`SPLIT`] mark. Neighbouring clauses merge while the merge, counting a one-character joiner,
/// fits the ceiling; the first clause keeps its own mark as the joiner (or `，` if it has none). A clause longer
/// than the ceiling is hard-cut, and the cut pieces carry no mark. Chunks under [`MIN_CHUNK_CHARS`] are dropped.
///
/// # Panics
/// If `ceiling` is zero.
pub fn chunk(text: &str, ceiling: usize) -> Vec<(String, String)> {
    assert!(ceiling > 0, "the chunk ceiling must be at least 1");
    let mut parts: Vec<(String, String)> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if SPLIT.contains(&ch) {
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

    let mut out: Vec<(String, String)> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    let mut cur_mark = String::new();
    for (part, mark) in parts {
        if part.is_empty() {
            continue;
        }
        let part: Vec<char> = part.chars().collect();
        if cur.is_empty() {
            cur = part;
            cur_mark = mark;
        } else if cur.len() + part.len() < ceiling {
            if cur_mark.is_empty() {
                cur.push(UNMARKED_JOINER);
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
    out.into_iter()
        .filter(|(c, _)| c.chars().count() >= MIN_CHUNK_CHARS)
        .collect()
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

/// Chunk every paragraph at `ceiling` and list the chunks in reading order.
pub fn plan(paragraphs: &[String], ceiling: usize) -> Vec<PlannedChunk> {
    let mut out = Vec::new();
    for (para, text) in paragraphs.iter().enumerate() {
        let chunks = chunk(text, ceiling);
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
    out
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
    fn merge_keeps_the_first_clause_mark() {
        let got = chunk("今日好熱。我哋去游水！", 45);
        assert_eq!(got, vec![("今日好熱。我哋去游水".to_string(), "！".to_string())]);
    }

    #[test]
    fn strip_matches_reference_whitespace() {
        assert_eq!(strip("\u{1c}\u{3000} 文字 \t\n"), "文字");
    }
}
