//! The text judge: a draw is read back by a recogniser and compared with the chunk's text, and a draw that lost a run
//! of text, or said one twice, is rejected like any judge rejection.
//!
//! The duration judge ([`crate::audio::judge`]) cannot see a skipped clause: a chunk that drops one gets shorter, not
//! longer, so it is never a runaway. This judge reads the draw back. It needs a recogniser ([`Recogniser`]); this
//! module has none of its own, so it is tested with a stand-in and any speech recogniser can be put behind it.
//!
//! **The script is folded before comparing.** A Cantonese recogniser often writes a traditional-Chinese sentence in
//! simplified characters. Compared unfolded, every such character would count as lost and good draws would be
//! redrawn. Both texts are folded to simplified characters with OpenCC's character table ([`Fold::t2s`]), and only
//! CJK characters ([`crate::text::is_cjk`]) are compared. There is no fallback: a judge cannot be built without a
//! [`Fold`], and a table that does not read is refused.
//!
//! **Known limit:** the fold is many-to-one. A swap between two traditional characters that share a simplified form
//! (乾 and 幹 both fold to 干) reads as correct.
//!
//! **How a run is found.** The folded reference and transcript are aligned by edit distance (a match, then a
//! substitution, then a deletion, then an insertion, on ties), and the alignment's runs are counted:
//! * *lost*: reference characters with nothing in the transcript (deletions). A run of at least
//!   [`TextJudge::min_run`] is [`Verdict::LostText`].
//! * *repeated*: transcript characters with nothing in the reference (insertions) whose run, at least `min_run` long,
//!   is a stretch of the reference: something said twice. [`Verdict::RepeatedText`].
//! * *wrong*: substitutions. Counted, never a rejection on their own: a recogniser's misreading of one character is
//!   not a defect of the draw.
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

/// The shortest run of lost or repeated characters that rejects a draw, unless the caller sets another.
pub const DEFAULT_MIN_RUN: usize = 4;

/// A script fold: each character to its simplified form.
#[derive(Debug, Clone)]
pub struct Fold {
    map: HashMap<char, char>,
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

    /// The CJK characters of `text`, folded. Everything else (punctuation, Latin, digits, space) is left out.
    pub fn fold(&self, text: &str) -> Vec<char> {
        text.chars()
            .filter(|&c| is_cjk(c))
            .map(|c| *self.map.get(&c).unwrap_or(&c))
            .collect()
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

/// How a transcript compares with its reference, both folded ([`compare`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Comparison {
    /// Reference characters the transcript does not have.
    pub lost: usize,
    /// The longest run of them.
    pub longest_lost: usize,
    /// Reference characters read as another character.
    pub wrong: usize,
    /// Transcript characters the reference does not have.
    pub extra: usize,
    /// The longest run of extra characters that is a stretch of the reference: said twice.
    pub longest_repeat: usize,
}

/// Align `hyp` to `reference` by edit distance and count what was lost, wrong, extra and repeated.
pub fn compare(reference: &[char], hyp: &[char]) -> Comparison {
    let (n, m) = (reference.len(), hyp.len());
    // d[i][j]: the distance between reference[..i] and hyp[..j].
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + usize::from(reference[i - 1] != hyp[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    // Walk back: a match or substitution first, then a deletion (lost), then an insertion (extra).
    #[derive(Clone, Copy, PartialEq)]
    enum Op {
        Keep,
        Wrong,
        Lost,
        Extra,
    }
    let mut ops = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(reference[i - 1] != hyp[j - 1]) {
            ops.push(if reference[i - 1] == hyp[j - 1] {
                Op::Keep
            } else {
                Op::Wrong
            });
            i -= 1;
            j -= 1;
        } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            ops.push(Op::Lost);
            i -= 1;
        } else {
            ops.push(Op::Extra);
            j -= 1;
        }
    }
    ops.reverse();
    let mut c = Comparison::default();
    let (mut lost_run, mut extra_run) = (0usize, Vec::new());
    let mut j = 0;
    let close_extra = |run: &mut Vec<char>, c: &mut Comparison| {
        if !run.is_empty() && is_stretch_of(reference, run) {
            c.longest_repeat = c.longest_repeat.max(run.len());
        }
        run.clear();
    };
    for op in ops {
        if op != Op::Lost {
            lost_run = 0;
        }
        if op != Op::Extra {
            close_extra(&mut extra_run, &mut c);
        }
        match op {
            Op::Keep => j += 1,
            Op::Wrong => {
                c.wrong += 1;
                j += 1;
            }
            Op::Lost => {
                c.lost += 1;
                lost_run += 1;
                c.longest_lost = c.longest_lost.max(lost_run);
            }
            Op::Extra => {
                c.extra += 1;
                extra_run.push(hyp[j]);
                j += 1;
            }
        }
    }
    close_extra(&mut extra_run, &mut c);
    c
}

/// `run` occurs in `reference` as a contiguous stretch.
fn is_stretch_of(reference: &[char], run: &[char]) -> bool {
    run.len() <= reference.len() && reference.windows(run.len()).any(|w| w == run)
}

/// What the text judge read in one draw.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// The recogniser's transcript, unfolded, as it came back. Empty when the recogniser failed.
    pub transcript: String,
    /// The comparison of the folded texts.
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
            last: None,
        }
    }

    /// The reading of the last draw judged, or `None` when the duration judge rejected it and it was not read back.
    pub fn last_reading(&self) -> Option<&Reading> {
        self.last.as_ref()
    }

    /// Read `transcript` against `text`: the comparison and the verdict.
    pub fn read(&self, text: &str, transcript: &str) -> Reading {
        let comparison = compare(&self.fold.fold(text), &self.fold.fold(transcript));
        let verdict = if comparison.longest_lost >= self.min_run {
            Some(Verdict::LostText)
        } else if comparison.longest_repeat >= self.min_run {
            Some(Verdict::RepeatedText)
        } else {
            None
        };
        Reading {
            transcript: transcript.to_string(),
            comparison,
            verdict,
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
            "lost": c.lost,
            "longest_lost": c.longest_lost,
            "wrong": c.wrong,
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
        // Only CJK is compared.
        assert_eq!(f.fold("Hi, 你好！ 1 2"), chars("你好"));
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
    fn compare_counts_lost_wrong_extra_and_repeated_runs() {
        let r = chars("今日天氣好好我哋去公園");
        assert_eq!(compare(&r, &r), Comparison::default());
        let c = compare(&r, &chars("今日天氣好好去公園"));
        assert_eq!((c.lost, c.longest_lost, c.wrong, c.extra), (2, 2, 0, 0));
        let c = compare(&r, &chars("今日天氣好好我哋去公圓"));
        assert_eq!((c.lost, c.wrong), (0, 1));
        let c = compare(&r, &chars("今日天氣好好我哋去公園我哋去公園"));
        assert_eq!((c.extra, c.longest_repeat), (5, 5));
        // Extra text that is not a stretch of the reference is extra, not a repeat.
        let c = compare(&r, &chars("今日天氣好好我哋去公園唔該晒啦喂"));
        assert_eq!((c.extra, c.longest_repeat), (5, 0));
    }
}
