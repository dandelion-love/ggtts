//! The text judge ([`ggtts_core::textjudge`]) through the core chunk step, on the fake engine, with a stand-in
//! recogniser: what it folds, what it counts, what it rejects, and what it leaves alone.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use ggtts_core::audio::Verdict;
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{render_chunk, DurationJudge, Given, Judge, KeptChunk, Observers, RenderOptions};
use ggtts_core::textjudge::{Fold, Reading, Recogniser, TextJudge, DEFAULT_MIN_RUN};
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;

/// A chunk of Hong Kong Cantonese in traditional characters.
const TEXT: &str = "今日天氣好好，我哋一齊去公園散步，順便買啲嘢食。";
/// The same, as a recogniser often writes it: simplified.
const TEXT_SIMPLIFIED: &str = "今日天气好好，我哋一齐去公园散步，顺便买啲嘢食。";

/// Answers each read-back with the next scripted transcript (the last repeats), counting the calls.
#[derive(Clone)]
struct Scripted {
    answers: Rc<RefCell<VecDeque<Result<String, String>>>>,
    calls: Rc<RefCell<usize>>,
}

impl Scripted {
    fn new(answers: &[Result<&str, &str>]) -> Self {
        Self {
            answers: Rc::new(RefCell::new(
                answers
                    .iter()
                    .map(|a| a.map(str::to_string).map_err(str::to_string))
                    .collect(),
            )),
            calls: Rc::new(RefCell::new(0)),
        }
    }

    fn calls(&self) -> usize {
        *self.calls.borrow()
    }
}

impl Recogniser for Scripted {
    fn recognise(&mut self, _: &[f32], _: u32, _: &str) -> Result<String, String> {
        *self.calls.borrow_mut() += 1;
        let mut a = self.answers.borrow_mut();
        if a.len() > 1 {
            a.pop_front().unwrap()
        } else {
            a.front().cloned().unwrap()
        }
    }
}

fn judge(answers: &[Result<&str, &str>]) -> (TextJudge<Scripted>, Scripted) {
    let r = Scripted::new(answers);
    (TextJudge::new(r.clone(), Fold::t2s().unwrap()), r)
}

/// Render `text` through the core step on the fake engine with `judge` and `redraw` redraws.
fn render(engine: &mut FakeEngine, text: &str, judge: &mut dyn Judge, redraw: u32) -> KeptChunk {
    let opts = RenderOptions {
        redraw,
        ..RenderOptions::for_profile(&ggtts_qwen3_tts::PROFILE)
    };
    let mut hooks = Hooks::new();
    render_chunk(
        engine,
        text,
        0,
        &Given(7),
        judge,
        &opts,
        &mut Observers::new(&mut NullEvents, &mut hooks),
    )
    .unwrap()
}

/// Judge one draw of `TEXT` read back as `transcript`: the kept verdict and the reading.
fn one(transcript: &str) -> (Option<Verdict>, Reading) {
    let (mut j, _) = judge(&[Ok(transcript)]);
    let kept = render(&mut FakeEngine::default(), TEXT, &mut j, 0);
    (kept.kept().judgement.verdict, j.take_reading().unwrap())
}

/// AC1, AC2: a transcript identical to the reference, or the reference written in simplified characters, scores zero
/// and keeps the draw; unfolded, the simplified one would have counted every changed character.
#[test]
fn the_reference_in_either_script_scores_zero() {
    for transcript in [TEXT, TEXT_SIMPLIFIED] {
        let (verdict, r) = one(transcript);
        assert_eq!(verdict, None, "{transcript}");
        assert_eq!(
            (r.comparison.lost, r.comparison.wrong, r.comparison.extra),
            (0, 0, 0),
            "{transcript}"
        );
        assert_eq!(r.transcript, transcript);
    }
    // The control: compared without the fold, the simplified transcript differs in 8 characters.
    let identity = Fold::from_table("a\ta\n").unwrap();
    let unfolded = TextJudge::new(Scripted::new(&[Ok("")]), identity).read(TEXT, TEXT_SIMPLIFIED);
    assert_eq!(unfolded.comparison.wrong, 5, "{unfolded:?}");
}

/// AC3: one wrong character and one dropped character are counted, and neither rejects the draw.
#[test]
fn one_wrong_or_dropped_character_is_counted_and_kept() {
    let (verdict, r) = one("今日天氣好好，我哋一齊去公圓散步，順便買啲嘢食。");
    assert_eq!((verdict, r.comparison.wrong, r.comparison.lost), (None, 1, 0));
    let (verdict, r) = one("今日天氣好好，我哋一齊去公園散步，順便買嘢食。");
    assert_eq!((verdict, r.comparison.lost, r.comparison.longest_lost), (None, 1, 1));
}

/// AC3: a dropped clause is rejected as lost text, and a clause said twice as repeated text.
#[test]
fn a_dropped_clause_is_lost_text_and_a_doubled_one_repeated_text() {
    let (verdict, r) = one("今日天氣好好，順便買啲嘢食。");
    assert_eq!(verdict, Some(Verdict::LostText));
    assert_eq!(r.comparison.longest_lost, 9, "{r:?}");
    let (verdict, r) = one("今日天氣好好，我哋一齊去公園散步，我哋一齊去公園散步，順便買啲嘢食。");
    assert_eq!(verdict, Some(Verdict::RepeatedText));
    assert_eq!(r.comparison.longest_repeat, 9, "{r:?}");
    // A run one short of the bar is counted, not rejected.
    let short: String = TEXT
        .chars()
        .take(7)
        .chain(TEXT.chars().skip(7 + DEFAULT_MIN_RUN - 1))
        .collect();
    let (verdict, r) = one(&short);
    assert_eq!(
        (verdict, r.comparison.longest_lost),
        (None, DEFAULT_MIN_RUN - 1),
        "{short}"
    );
}

/// AC3 and AC5: a draw rejected for lost text is redrawn at the next draw's seed, through the same step and redraws
/// as any judge rejection; the redraw read back whole is kept.
#[test]
fn a_draw_that_lost_text_is_redrawn_and_the_whole_one_kept() {
    let (mut j, r) = judge(&[Ok("今日天氣好好，順便買啲嘢食。"), Ok(TEXT)]);
    let kept = render(&mut FakeEngine::default(), TEXT, &mut j, 2);
    let verdicts: Vec<_> = kept.draws.iter().map(|d| d.judgement.verdict).collect();
    assert_eq!(verdicts, vec![Some(Verdict::LostText), None]);
    assert_eq!(kept.kept().draw, 1);
    assert_eq!(r.calls(), 2);
}

/// AC5: a draw the duration judge rejects keeps its verdict and is never read back; a recogniser that fails keeps the
/// draw and records why.
#[test]
fn a_duration_rejection_is_not_read_back_and_a_failed_read_keeps_the_draw() {
    // The fake speaks for 40 seconds: a runaway at 0.6 s per character.
    let script = [(TEXT.to_string(), vec![40.0])].into_iter().collect();
    let mut engine = FakeEngine::new(script, Default::default());
    let (mut j, r) = judge(&[Ok("")]);
    let kept = render(&mut engine, TEXT, &mut j, 0);
    assert_eq!(kept.kept().judgement.verdict, Some(Verdict::Runaway));
    assert_eq!(r.calls(), 0, "a runaway draw was read back");
    assert!(j.last_reading().is_none());

    let (mut j, _) = judge(&[Err("the recogniser crashed")]);
    let kept = render(&mut FakeEngine::default(), TEXT, &mut j, 2);
    assert_eq!((kept.draws.len(), kept.kept().judgement.verdict), (1, None));
    let reading = j.take_reading().unwrap();
    assert_eq!(reading.error.as_deref(), Some("the recogniser crashed"));
}

/// AC4: there is no judge without a fold, and a fold that does not read is refused, never replaced by an unfolded
/// comparison.
#[test]
fn a_fold_that_does_not_read_is_refused() {
    assert!(Fold::t2s().is_ok());
    for bad in ["", "no tab here", "兩個\t两个"] {
        assert!(Fold::from_table(bad).is_err(), "{bad:?}");
    }
}

/// The trace schema allows every verdict a judge can give, so a trace of a text-judged render validates.
#[test]
fn the_trace_schema_allows_every_verdict() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-trace-2.schema.json");
    let text = std::fs::read_to_string(path).unwrap();
    for v in [
        Verdict::Runaway,
        Verdict::Silent,
        Verdict::LostText,
        Verdict::RepeatedText,
    ] {
        // Exhaustive: a new verdict fails to compile here until it is listed.
        match v {
            Verdict::Runaway | Verdict::Silent | Verdict::LostText | Verdict::RepeatedText => {}
        }
        assert!(
            text.contains(&format!("\"{}\"", v.as_str())),
            "{} is not in the trace schema",
            v.as_str()
        );
    }
}

/// AC6: the duration judge is unchanged: the same draws and verdicts as before on the same render.
#[test]
fn the_duration_judge_keeps_what_the_text_judge_would_on_a_clean_read() {
    let a = render(&mut FakeEngine::default(), TEXT, &mut DurationJudge, 2);
    let (mut j, _) = judge(&[Ok(TEXT)]);
    let b = render(&mut FakeEngine::default(), TEXT, &mut j, 2);
    assert_eq!(a.samples, b.samples);
    assert_eq!(a.draws.len(), b.draws.len());
    assert_eq!(a.kept().judgement, b.kept().judgement);
}
