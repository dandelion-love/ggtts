//! Completed-chapter feedback must use verified final audio and refuse changed evidence.
mod common;
use ggtts_book::{
    manifest::Manifest,
    plan::{plan_book, DEFAULT_SEED_ROOT},
    render::{feedback_chapter, render_book, ChapterOutcome},
    Book,
};
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;

#[test]
fn page_refuses_clobber_and_shifted_spans() {
    let dir = std::env::temp_dir().join(format!("ggtts-feedback-adapter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let book = Book::read(common::nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let reports = render_book(&mut FakeEngine::default(), &book, &plans, &dir).unwrap();
    let report = reports
        .iter()
        .find(|r| {
            plans.iter().any(|p| p.chapter_id == r.id && p.chunks.len() >= 2)
                && matches!(r.outcome, ChapterOutcome::Written { .. })
        })
        .unwrap();
    let plan = plans.iter().find(|p| p.chapter_id == report.id).unwrap();
    let path = match &report.outcome {
        ChapterOutcome::Written { file, .. } => file,
        _ => unreachable!(),
    };
    let chapter_dir = dir.join(path.file_stem().unwrap());
    let page = dir.join("page.html");
    feedback_chapter(&chapter_dir, &page).unwrap();
    let before = std::fs::read(&page).unwrap();
    assert!(feedback_chapter(&chapter_dir, &page).is_err());
    assert_eq!(std::fs::read(&page).unwrap(), before);
    let mut m = Manifest::load(&chapter_dir, plan).unwrap();
    let spans = m.chapter_spans.as_mut().unwrap();
    spans.chunks[1].start += 1;
    spans.chunks[1].end += 1;
    m.save(&chapter_dir).unwrap();
    let rejected = dir.join("rejected.html");
    assert!(feedback_chapter(&chapter_dir, &rejected).is_err());
    assert!(!rejected.exists());
    assert!(!dir.join("rejected.html.provenance.json").exists());
    std::fs::remove_dir_all(dir).ok();
}
