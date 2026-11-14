//! The text judge: a draw is read back by a recogniser and compared with the chunk's text, and a draw that lost a run
//! of text, or said one twice, is rejected like any judge rejection.
//!
//! The duration judge ([`crate::audio::judge`]) cannot see a skipped clause: a chunk that drops one gets shorter, not
//! longer, so it is never a runaway. This judge reads the draw back. It needs a recogniser ([`Recogniser`]); this
//! module has none of its own, so it is tested with a stand-in and any speech recogniser can be put behind it.
//!
//! **What is compared** ([`Fold::fold`]). A Cantonese recogniser often writes a traditional-Chinese sentence in
//! simplified characters, and a number as digits. Compared as written, every such character would count as lost and
//! good draws would be redrawn. So both texts become the same kind of tokens first:
//! * CJK characters ([`crate::text::is_cjk`]), folded to simplified with OpenCC's character table ([`Fold::t2s`]);
//! * numbers, written out (三百五十六, 兩萬), in digits (356, ３５６) or mixed (3萬, 1萬1千), as one token for their
//!   value, so 三百五十六 and 356 are the same token, and a misheard number or unit is one misread token, not several
//!   lost ones;
//! * nothing else: punctuation, spaces and Latin letters are left out.
//!
//! There is no fallback: a judge cannot be built without a [`Fold`], and a table that does not read is refused.
//!
//! **How a defect is found.** The tokens are aligned with affine gap costs (a substitution 2, a gap 1 to open and 2 per
//! token), so a dropped or added stretch stays one run even where characters repeat (好好好), and on ties a gap is
//! extended rather than broken. Then:
//! * *lost text* ([`Verdict::LostText`]) is any of: a run of at least [`TextJudge::min_run`] reference tokens with
//!   nothing in the transcript; a quarter or more of the reference lost ([`TextJudge::max_lost_rate`]), when the
//!   reference has at least 8 tokens and 2 are lost; or a run of [`TextJudge::damaged_run`] reference tokens each lost
//!   or misread (8 unless set): a garbled stretch.
//! * *repeated text* ([`Verdict::RepeatedText`]): a run of transcript tokens with nothing in the reference that holds
//!   at least `min_run` tokens of the reference, found by local alignment, so one misread token or a filler (呃) at
//!   the join does not hide a clause said twice.
//! * *wrong*: a single misread token. Counted, never a rejection on its own.
//!
//! **Known limits.**
//! * The fold is many-to-one: a swap between two traditional characters that share a simplified form (乾 and 幹 both
//!   fold to 干) reads as correct.
//! * A chunk with fewer than `min_run` tokens can never be rejected, even by an empty transcript: none of the bars can
//!   be reached. A chunk with no CJK characters and no numbers has no tokens and always passes.
//! * A repeated clause shorter than `min_run`, or one so misread that no `min_run` of its tokens survive in order,
//!   reads as extra text, not as a repeat.
//!
//! A draw the duration judge rejects keeps its verdict and is not read back. A recogniser that fails keeps the draw
//! (it cannot be judged), and the reading records the error.

use std::collections::HashMap;

use crate::audio::{judge_f32, Judgement, Verdict};
use crate::pipeline::Judge;
use crate::profile::Profile;
use crate::text::is_cjk;

/// OpenCC's traditional-to-simplified character table (`TSCharacters.txt`), vendored with its licence beside it.
const TS_CHARACTERS: &str = include_str!("../data/opencc/TSCharacters.txt");

/// The shortest run of lost or repeated tokens that rejects a draw, unless the caller sets another.
pub const DEFAULT_MIN_RUN: usize = 4;
/// The share of the reference lost that rejects a draw, unless the caller sets another.
pub const DEFAULT_MAX_LOST_RATE: f64 = 0.25;
/// The fewest reference tokens, and lost tokens, the lost-rate bar applies to: a short chunk is judged by its runs.
pub const LOST_RATE_MIN_TOKENS: usize = 8;
pub const LOST_RATE_MIN_LOST: usize = 2;

/// A script fold: each character to its simplified form.
#[derive(Debug, Clone)]
pub struct Fold {
    map: HashMap<char, char>,
}

/// The value of one written-out digit or zero, by character.
fn numeral_digit(c: char) -> Option<u32> {
    Some(match c {
        '零' | '〇' => 0,
        '一' => 1,
        '二' | '兩' | '两' => 2,
        '三' => 3,
        '四' => 4,
        '五' => 5,
        '六' => 6,
        '七' => 7,
        '八' => 8,
        '九' => 9,
        _ => return None,
    })
}

/// The multiplier of one written-out unit, by character.
fn numeral_unit(c: char) -> Option<u128> {
    Some(match c {
        '十' => 10,
        '百' => 100,
        '千' => 1_000,
        '萬' | '万' => 10_000,
        '億' | '亿' => 100_000_000,
        _ => return None,
    })
}

/// A digit written as a digit, ASCII or full-width.
fn digit(c: char) -> Option<u32> {
    match c {
        '0'..='9' => c.to_digit(10),
        '０'..='９' => Some(c as u32 - '０' as u32),
        _ => None,
    }
}

/// A run of numerals (digits ASCII or full-width, written-out digits, units) as the digits of its value. A run of
/// digits is one number (35, ３５). Without a unit the run is read digit by digit, as it is spoken (一九九七 is 1997,
/// 19九7 too); with units it is a number, a digit run counting as one digit place (3萬, 35萬, 1萬1千, 三百五十六).
fn numeral_value(run: &[char]) -> String {
    if !run.iter().any(|&c| numeral_unit(c).is_some()) {
        return run
            .iter()
            .filter_map(|&c| digit(c).or_else(|| numeral_digit(c)))
            .map(|d| char::from(b'0' + d as u8))
            .collect();
    }
    let (mut total, mut section, mut num) = (0u128, 0u128, 0u128);
    let mut in_digits = false;
    for &c in run {
        if let Some(d) = digit(c) {
            // Consecutive digits are one number: 35 before a unit is thirty-five of it.
            num = if in_digits {
                num.saturating_mul(10).saturating_add(u128::from(d))
            } else {
                u128::from(d)
            };
            in_digits = true;
            continue;
        }
        in_digits = false;
        if let Some(d) = numeral_digit(c) {
            num = u128::from(d);
        } else if let Some(u) = numeral_unit(c) {
            if u >= 10_000 {
                section = section.saturating_add(num);
                total = total.saturating_add(section.max(1).saturating_mul(u));
                section = 0;
            } else {
                section = section.saturating_add(num.max(1).saturating_mul(u));
            }
            num = 0;
        }
    }
    total.saturating_add(section).saturating_add(num).to_string()
}

/// The one token a number is compared as: a character in the Supplementary Private Use Area-A chosen by its value
/// (FNV-1a), so the same number is the same token in both texts and a number counts as one token, whatever its
/// length. Two different numbers share a token with odds of 1 in 65,534.
fn number_token(value: &str) -> char {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in value.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    char::from_u32(0xF_0000 + (h % 65_534) as u32).expect("a private-use code point")
}

/// The token for the number `text` names (written out, in digits, or mixed), as [`Fold::fold`] compares it.
pub fn number(text: &str) -> char {
    number_token(&numeral_value(&text.chars().collect::<Vec<_>>()))
}

impl Fold {
    /// A fold from an OpenCC character table: one line per character, `FROM<TAB>TO[ TO...]`, the first `TO` taken.
    /// Refused when a line does not read, or the table is empty.
    pub fn from_table(table: &str) -> Result<Self, String> {
        let mut map = HashMap::new();
        for (n, line) in table.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (from, to) = line
                .split_once('\t')
                .ok_or_else(|| format!("script fold table line {}: no tab", n + 1))?;
            let to = to.split(' ').next().unwrap_or_default();
            let (mut f, mut t) = (from.chars(), to.chars());
            match (f.next(), f.next(), t.next(), t.next()) {
                (Some(a), None, Some(b), None) => {
                    map.insert(a, b);
                }
                _ => {
                    return Err(format!(
                        "script fold table line {}: not one character to one: {line:?}",
                        n + 1
                    ))
                }
            }
        }
        if map.is_empty() {
            return Err("script fold table is empty".into());
        }
        Ok(Self { map })
    }

    /// Traditional to simplified, by OpenCC's character table.
    pub fn t2s() -> Result<Self, String> {
        Self::from_table(TS_CHARACTERS)
    }

    /// The tokens of `text` the judge compares: CJK characters folded, and each number (a run of written-out or
    /// ASCII or full-width digits and units, in any mix) as one token for its value ([`number`]). Punctuation, spaces
    /// and Latin letters are left out, and end a number.
    pub fn fold(&self, text: &str) -> Vec<char> {
        let mut out = Vec::new();
        let mut run: Vec<char> = Vec::new();
        for c in text.chars() {
            if digit(c).is_some() || numeral_digit(c).is_some() || numeral_unit(c).is_some() {
                run.push(c);
                continue;
            }
            if !run.is_empty() {
                out.push(number_token(&numeral_value(&run)));
                run.clear();
            }
            if is_cjk(c) {
                out.push(*self.map.get(&c).unwrap_or(&c));
            }
        }
        if !run.is_empty() {
            out.push(number_token(&numeral_value(&run)));
        }
        out
    }
}

/// Reads a draw back as text.
pub trait Recogniser {
    /// The text heard in `samples` at `sample_rate`. `text` is what the draw was asked to say, for a recogniser that
    /// uses it as context; the judge compares against it either way.
    fn recognise(&mut self, samples: &[f32], sample_rate: u32, text: &str) -> Result<String, String>;

    /// Why the recogniser cannot read any more draws (its process died, or stopped answering), or `None`. An `Err`
    /// from [`Recogniser::recognise`] with no fault is one draw it could not read; a fault ends the render
    /// ([`crate::pipeline::Judge::fault`]).
    fn fault(&self) -> Option<String> {
        None
    }
}

/// How a transcript compares with its reference, both as tokens ([`compare`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Comparison {
    /// Reference tokens compared.
    pub compared: usize,
    /// Reference tokens the transcript does not have.
    pub lost: usize,
    /// The longest run of them.
    pub longest_lost: usize,
    /// Reference tokens read as another token.
    pub wrong: usize,
    /// The longest run of reference tokens each lost or misread.
    pub longest_damaged: usize,
    /// Transcript tokens the reference does not have.
    pub extra: usize,
    /// The most reference tokens found, by local alignment, in one run of extra tokens: a stretch said twice.
    pub longest_repeat: usize,
}

/// Alignment costs, doubled to stay whole: a substitution, a gap's opening, and each token of a gap.
const SUB: u32 = 2;
const OPEN: u32 = 1;
const EXTEND: u32 = 2;
const INF: u32 = u32::MAX / 4;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Keep,
    Wrong,
    Lost,
    Extra,
}

/// Align `hyp` to `reference` with affine gap costs (Gotoh), extending a gap rather than breaking it on ties.
fn align(reference: &[char], hyp: &[char]) -> Vec<Op> {
    let (n, m) = (reference.len(), hyp.len());
    // m_[i][j]: best cost ending in a match or substitution; d[i][j]: ending in a lost token (a gap in the
    // transcript); x[i][j]: ending in an extra token.
    let mut m_ = vec![vec![INF; m + 1]; n + 1];
    let mut d = vec![vec![INF; m + 1]; n + 1];
    let mut x = vec![vec![INF; m + 1]; n + 1];
    m_[0][0] = 0;
    for (i, row) in d.iter_mut().enumerate().skip(1) {
        row[0] = OPEN + EXTEND * i as u32;
    }
    for (j, cell) in x[0].iter_mut().enumerate().skip(1) {
        *cell = OPEN + EXTEND * j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let s = if reference[i - 1] == hyp[j - 1] { 0 } else { SUB };
            m_[i][j] = m_[i - 1][j - 1].min(d[i - 1][j - 1]).min(x[i - 1][j - 1]) + s;
            d[i][j] = (d[i - 1][j] + EXTEND)
                .min(m_[i - 1][j] + OPEN + EXTEND)
                .min(x[i - 1][j] + OPEN + EXTEND);
            x[i][j] = (x[i][j - 1] + EXTEND)
                .min(m_[i][j - 1] + OPEN + EXTEND)
                .min(d[i][j - 1] + OPEN + EXTEND);
        }
    }
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        M,
        D,
        X,
    }
    let at = |s: State, i: usize, j: usize| match s {
        State::M => m_[i][j],
        State::D => d[i][j],
        State::X => x[i][j],
    };
    let (mut i, mut j) = (n, m);
    let end = [State::M, State::D, State::X];
    let mut state = *end.iter().min_by_key(|&&s| at(s, n, m)).expect("three states");
    let mut ops = Vec::with_capacity(n + m);
    while i > 0 || j > 0 {
        let here = at(state, i, j);
        match state {
            State::M => {
                ops.push(if reference[i - 1] == hyp[j - 1] {
                    Op::Keep
                } else {
                    Op::Wrong
                });
                let s = if reference[i - 1] == hyp[j - 1] { 0 } else { SUB };
                let prev = [State::M, State::D, State::X];
                state = *prev
                    .iter()
                    .find(|&&p| at(p, i - 1, j - 1) + s == here)
                    .expect("a predecessor");
                i -= 1;
                j -= 1;
            }
            State::D => {
                ops.push(Op::Lost);
                // Extend the gap first.
                state = if d[i - 1][j] + EXTEND == here {
                    State::D
                } else if m_[i - 1][j] + OPEN + EXTEND == here {
                    State::M
                } else {
                    State::X
                };
                i -= 1;
            }
            State::X => {
                ops.push(Op::Extra);
                state = if x[i][j - 1] + EXTEND == here {
                    State::X
                } else if m_[i][j - 1] + OPEN + EXTEND == here {
                    State::M
                } else {
                    State::D
                };
                j -= 1;
            }
        }
        if i == 0 && j > 0 {
            state = State::X;
        } else if j == 0 && i > 0 {
            state = State::D;
        }
    }
    ops.reverse();
    ops
}

/// The most tokens of `run` that match `reference` in one local alignment (Smith-Waterman: a match +2, a mismatch
/// or a gap -1 and -2), so a stretch of the reference said again with a misread token or a filler is still found.
fn repeated_in(reference: &[char], run: &[char]) -> usize {
    let (n, m) = (reference.len(), run.len());
    // (score, matched tokens) per cell; the best cell's matched count is the answer.
    let mut prev = vec![(0i32, 0usize); m + 1];
    let mut best = (0i32, 0usize);
    for i in 1..=n {
        let mut cur = vec![(0i32, 0usize); m + 1];
        for j in 1..=m {
            let (s, k) = prev[j - 1];
            let diag = if reference[i - 1] == run[j - 1] {
                (s + 2, k + 1)
            } else {
                (s - 1, k)
            };
            let up = (prev[j].0 - 2, prev[j].1);
            let left = (cur[j - 1].0 - 2, cur[j - 1].1);
            let mut c = (0, 0);
            for cand in [diag, up, left] {
                if cand.0 > c.0 || (cand.0 == c.0 && cand.0 > 0 && cand.1 > c.1) {
                    c = cand;
                }
            }
            cur[j] = c;
            if c.0 > best.0 || (c.0 == best.0 && c.1 > best.1) {
                best = c;
            }
        }
        prev = cur;
    }
    best.1
}

/// Align `hyp` to `reference` (both tokens, [`Fold::fold`]) and count what was lost, misread, garbled, added and
/// repeated.
pub fn compare(reference: &[char], hyp: &[char]) -> Comparison {
    let ops = align(reference, hyp);
    let mut c = Comparison {
        compared: reference.len(),
        ..Comparison::default()
    };
    let (mut lost_run, mut damaged_run) = (0usize, 0usize);
    // The extra run being read: where it starts in `hyp`, if one is open.
    let mut extra_from: Option<usize> = None;
    let mut j = 0;
    // An inserted run can slide along the transcript where the token leaving it equals the token entering it (the
    // alignment is as cheap either way), so the aligner may report a rotation of what was said twice. The repeat is
    // read at every place it can slide to, and the most found is kept.
    let repeat_at = |s: usize, e: usize| -> usize {
        let mut best = repeated_in(reference, &hyp[s..e]);
        let (mut a, mut b) = (s, e);
        while b < hyp.len() && hyp[a] == hyp[b] {
            a += 1;
            b += 1;
            best = best.max(repeated_in(reference, &hyp[a..b]));
        }
        let (mut a, mut b) = (s, e);
        while a > 0 && hyp[a - 1] == hyp[b - 1] {
            a -= 1;
            b -= 1;
            best = best.max(repeated_in(reference, &hyp[a..b]));
        }
        best
    };
    for op in ops {
        if op != Op::Lost {
            lost_run = 0;
        }
        if op == Op::Keep {
            damaged_run = 0;
        }
        if op != Op::Extra {
            if let Some(s) = extra_from.take() {
                c.longest_repeat = c.longest_repeat.max(repeat_at(s, j));
            }
        }
        match op {
            Op::Keep => j += 1,
            Op::Wrong => {
                c.wrong += 1;
                damaged_run += 1;
                j += 1;
            }
            Op::Lost => {
                c.lost += 1;
                lost_run += 1;
                damaged_run += 1;
                c.longest_lost = c.longest_lost.max(lost_run);
            }
            Op::Extra => {
                c.extra += 1;
                extra_from.get_or_insert(j);
                j += 1;
            }
        }
        c.longest_damaged = c.longest_damaged.max(damaged_run);
    }
    if let Some(s) = extra_from {
        c.longest_repeat = c.longest_repeat.max(repeat_at(s, j));
    }
    c
}

/// What the text judge read in one draw.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// The recogniser's transcript, as it came back. Empty when the recogniser failed.
    pub transcript: String,
    /// The comparison of the two texts' tokens.
    pub comparison: Comparison,
    /// The verdict this reading gave: [`Verdict::LostText`], [`Verdict::RepeatedText`] or none.
    pub verdict: Option<Verdict>,
    /// Why the draw could not be read, if the recogniser failed.
    pub error: Option<String>,
}

/// The duration judge, then a read-back by `R` for every draw it keeps.
pub struct TextJudge<R: Recogniser> {
    recogniser: R,
    fold: Fold,
    /// The shortest lost or repeated run that rejects a draw ([`DEFAULT_MIN_RUN`] unless set).
    pub min_run: usize,
    /// The share of the reference lost that rejects a draw ([`DEFAULT_MAX_LOST_RATE`] unless set), when the reference
    /// has at least [`LOST_RATE_MIN_TOKENS`] tokens and [`LOST_RATE_MIN_LOST`] are lost.
    pub max_lost_rate: f64,
    /// The run of lost or misread reference tokens that rejects a draw; `None` is twice [`TextJudge::min_run`].
    pub damaged_run: Option<usize>,
    last: Option<Reading>,
}

impl<R: Recogniser> TextJudge<R> {
    /// A text judge with `recogniser` and `fold`. There is no constructor without a fold: unfolded text is never
    /// compared.
    pub fn new(recogniser: R, fold: Fold) -> Self {
        Self {
            recogniser,
            fold,
            min_run: DEFAULT_MIN_RUN,
            max_lost_rate: DEFAULT_MAX_LOST_RATE,
            damaged_run: None,
            last: None,
        }
    }

    /// The reading of the last draw judged, or `None` when the duration judge rejected it and it was not read back.
    pub fn last_reading(&self) -> Option<&Reading> {
        self.last.as_ref()
    }

    /// The verdict a comparison gives under this judge's bars.
    pub fn verdict(&self, c: &Comparison) -> Option<Verdict> {
        let damaged_bar = self.damaged_run.unwrap_or(2 * self.min_run);
        let lost_rate = c.compared >= LOST_RATE_MIN_TOKENS
            && c.lost >= LOST_RATE_MIN_LOST
            && c.lost as f64 >= self.max_lost_rate * c.compared as f64;
        if c.longest_lost >= self.min_run || lost_rate || c.longest_damaged >= damaged_bar {
            Some(Verdict::LostText)
        } else if c.longest_repeat >= self.min_run {
            Some(Verdict::RepeatedText)
        } else {
            None
        }
    }

    /// Read `transcript` against `text`: the comparison and the verdict.
    pub fn read(&self, text: &str, transcript: &str) -> Reading {
        let comparison = compare(&self.fold.fold(text), &self.fold.fold(transcript));
        Reading {
            transcript: transcript.to_string(),
            verdict: self.verdict(&comparison),
            comparison,
            error: None,
        }
    }
}

impl Reading {
    /// The reading as the trace's `read` event and a book manifest carry it: the transcript, the counts, the verdict
    /// it gave, and the recogniser's error.
    pub fn to_json(&self) -> serde_json::Value {
        let c = &self.comparison;
        serde_json::json!({
            "transcript": self.transcript,
            "compared": c.compared,
            "lost": c.lost,
            "longest_lost": c.longest_lost,
            "wrong": c.wrong,
            "longest_damaged": c.longest_damaged,
            "extra": c.extra,
            "longest_repeat": c.longest_repeat,
            "read_verdict": self.verdict.map(Verdict::as_str),
            "error": self.error,
        })
    }
}

impl<R: Recogniser> Judge for TextJudge<R> {
    fn take_reading(&mut self) -> Option<serde_json::Value> {
        self.last.as_ref().map(Reading::to_json)
    }

    fn fault(&mut self) -> Option<String> {
        self.recogniser.fault()
    }

    fn judge(&mut self, profile: &Profile, samples: &[f32], sample_rate: u32, text: &str) -> Judgement {
        let duration = judge_f32(profile, samples, sample_rate, text);
        if duration.verdict.is_some() {
            self.last = None;
            return duration;
        }
        let reading = match self.recogniser.recognise(samples, sample_rate, text) {
            Ok(transcript) => self.read(text, &transcript),
            Err(e) => Reading {
                transcript: String::new(),
                comparison: Comparison::default(),
                verdict: None,
                error: Some(e),
            },
        };
        let verdict = reading.verdict;
        self.last = Some(reading);
        Judgement { verdict, ..duration }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn the_vendored_table_reads_and_folds_traditional_to_simplified() {
        let f = Fold::t2s().unwrap();
        assert_eq!(f.fold("說話為裡裏"), chars("说话为里里"));
        // Characters the table does not list (shared by both scripts, and Cantonese ones) are kept.
        assert_eq!(f.fold("嘅喺佢"), chars("嘅喺佢"));
        // Punctuation, spaces and Latin letters are left out; digits are kept.
        assert_eq!(f.fold("Hi, 你好！ 1 2"), vec!['你', '好', number("1"), number("2")]);
    }

    #[test]
    fn a_number_is_one_token_for_its_value_however_written() {
        let f = Fold::t2s().unwrap();
        // Each row: one value, written out, in digits (ASCII and full-width), and mixed with units.
        for forms in [
            &["三百五十六", "356", "３５６", "3百5十6"][..],
            &["兩萬", "两万", "20000", "2萬"],
            &["三萬", "30000", "3萬", "3万"],
            &["三十五萬", "350000", "35萬"],
            &["一萬一千", "11000", "1萬1千", "1万1千", "一萬1千"],
            &["三萬五千", "35000", "3萬5千"],
            &["一億五千萬", "150000000", "1億5千萬"],
            &["三百零五", "305", "3百零5"],
            &["十", "10"],
            &["十五", "15"],
            &["一九九七", "1997", "19九7"],
        ] {
            let tokens: Vec<Vec<char>> = forms.iter().map(|w| f.fold(w)).collect();
            assert!(tokens.iter().all(|t| t.len() == 1), "{forms:?}: {tokens:?}");
            assert!(tokens.iter().all(|t| *t == tokens[0]), "{forms:?}: {tokens:?}");
            assert_eq!(tokens[0][0], number(forms[0]));
        }
        // Different values are different tokens.
        assert_ne!(number("一萬"), number("一千"));
        assert_ne!(number("356"), number("365"));
        // In a sentence, the number is one token and the words around it stay.
        assert_eq!(f.fold("我有三百五十六蚊"), f.fold("我有356蚊"));
        assert_eq!(f.fold("我有三百五十六蚊").len(), 4);
    }

    #[test]
    fn a_table_that_does_not_read_is_refused() {
        assert!(Fold::from_table("").unwrap_err().contains("empty"));
        assert!(Fold::from_table("說说\n").unwrap_err().contains("no tab"));
        assert!(Fold::from_table("說話\t说话\n")
            .unwrap_err()
            .contains("not one character"));
        // Several targets: the first is taken.
        assert_eq!(Fold::from_table("乾\t干 乾\n").unwrap().fold("乾"), chars("干"));
    }

    #[test]
    fn compare_counts_lost_wrong_extra_damaged_and_repeated_runs() {
        let r = chars("今日天氣好好我哋去公園");
        assert_eq!(
            compare(&r, &r),
            Comparison {
                compared: r.len(),
                ..Comparison::default()
            }
        );
        let c = compare(&r, &chars("今日天氣好好去公園"));
        assert_eq!((c.lost, c.longest_lost, c.wrong, c.extra), (2, 2, 0, 0));
        let c = compare(&r, &chars("今日天氣好好我哋去公圓"));
        assert_eq!((c.lost, c.wrong, c.longest_damaged), (0, 1, 1));
        let c = compare(&r, &chars("今日天氣好好我哋去公園我哋去公園"));
        assert_eq!((c.extra, c.longest_repeat), (5, 5));
        // Extra text that holds no stretch of the reference is extra, not a repeat.
        let c = compare(&r, &chars("今日天氣好好我哋去公園唔該晒啦喂"));
        assert_eq!((c.extra, c.longest_repeat), (5, 0));
    }
}
