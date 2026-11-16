//! Independent #477 oracle: persisted spans refer to the final levelled chapter.
mod common;
use common::nested;
use ggtts_book::manifest::{chunk_file, Manifest};
use ggtts_book::plan::{plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_book, ChapterOutcome};
use ggtts_book::Book;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::flac::to_i16;
use ggtts_core::pipeline::{gap_samples, level};
use ggtts_core::profile::Profile;
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;
use std::path::{Path, PathBuf};

fn decoded(path: &Path) -> (u32, Vec<i16>) {
    let mut f = claxon::FlacReader::open(path).unwrap();
    let rate = f.streaminfo().sample_rate;
    (rate, f.samples().map(|s| s.unwrap() as i16).collect())
}
fn fixture(name: &str) -> (PathBuf, ggtts_book::plan::ChapterPlan, PathBuf) {
    let dir = std::env::temp_dir().join(format!("ggtts-book-feedback-independent-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let reports = render_book(&mut FakeEngine::default(), &book, &plans, &dir).unwrap();
    let report = reports
        .iter()
        .find(|r| {
            plans.iter().any(|p| p.chapter_id == r.id && p.chunks.len() >= 2)
                && matches!(r.outcome, ChapterOutcome::Written { .. })
        })
        .expect("multi-chunk chapter");
    let plan = plans.into_iter().find(|p| p.chapter_id == report.id).unwrap();
    let file = match &report.outcome {
        ChapterOutcome::Written { file, .. } => file.clone(),
        _ => unreachable!(),
    };
    (dir, plan, file)
}
fn check_spans(final_pcm: &[i16], expected: &[i16], spans: &[(usize, usize)]) {
    for (i, &(start, end)) in spans.iter().enumerate() {
        assert!(start < end && end <= final_pcm.len());
        assert_eq!(&final_pcm[start..end], &expected[start..end], "chapter span {i}");
    }
}
#[test]
fn persisted_spans_match_levelled_chapter_and_detect_shift() {
    let (dir, plan, file) = fixture("spans");
    let chapter_dir = dir.join(file.file_stem().unwrap());
    let manifest = Manifest::load(&chapter_dir, &plan).unwrap();
    let spans = manifest.chapter_spans.as_ref().expect("completed chapter spans");
    let (rate, final_pcm) = decoded(&file);
    assert_eq!(spans.sample_rate, rate);
    assert_eq!(spans.sample_count, final_pcm.len() as u64);
    assert_eq!(spans.chunks.len(), plan.chunks.len());
    let mut joined = Vec::<f32>::new();
    let mut expected_spans = Vec::new();
    for (i, chunk) in plan.chunks.iter().enumerate() {
        let (chunk_rate, pcm) = decoded(&chapter_dir.join(chunk_file(i)));
        assert_eq!(chunk_rate, rate);
        let start = joined.len();
        joined.extend(pcm.into_iter().map(|v| v as f32 / 32768.0));
        expected_spans.push((start, joined.len()));
        if i + 1 < plan.chunks.len() {
            joined.resize(
                joined.len() + gap_samples(&PROFILE, &chunk.punct, chunk.last_in_para, rate),
                0.0,
            );
        }
    }
    level(&PROFILE, &mut joined);
    let expected: Vec<i16> = joined.iter().map(|&x| to_i16(x)).collect();
    let actual_spans: Vec<_> = spans
        .chunks
        .iter()
        .map(|s| (s.start as usize, s.end as usize))
        .collect();
    assert_eq!(actual_spans, expected_spans);
    check_spans(&final_pcm, &expected, &actual_spans);
    // A deliberate one-sample shift must fail the same span oracle.
    let mut planted = actual_spans.clone();
    planted[0].0 += 1;
    let red = std::panic::catch_unwind(|| assert_eq!(planted, expected_spans, "span boundary differs"));
    assert!(red.is_err(), "the same oracle must fail a shifted boundary");
    std::fs::remove_dir_all(dir).ok();
}

/// Change the duration of a rerendered chunk so stale later boundaries cannot pass.
struct Longer {
    fake: FakeEngine,
}
impl Engine for Longer {
    fn info(&self) -> &EngineInfo {
        self.fake.info()
    }
    fn profile(&self) -> &Profile {
        self.fake.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.fake.generate(req)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        let mut pcm = self.fake.decode(codes)?;
        let extra = pcm.samples.len() / 4;
        pcm.samples.extend(std::iter::repeat_n(0.3, extra));
        Ok(pcm)
    }
}
#[test]
fn resumed_replaced_chunk_recomputes_following_spans() {
    let (dir, plan, file) = fixture("resume");
    let chapter_dir = dir.join(file.file_stem().unwrap());
    let before = Manifest::load(&chapter_dir, &plan).unwrap();
    let old = before.chapter_spans.unwrap();
    std::fs::write(chapter_dir.join(chunk_file(0)), b"planted corruption").unwrap();
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let reports = render_book(
        &mut Longer {
            fake: FakeEngine::default(),
        },
        &book,
        &plans,
        &dir,
    )
    .unwrap();
    let report = reports.iter().find(|r| r.id == plan.chapter_id).unwrap();
    match &report.outcome {
        ChapterOutcome::Written { rendered, resumed, .. } => {
            assert_eq!(rendered, &vec![0]);
            assert_eq!(*resumed, plan.chunks.len() - 1);
        }
        _ => panic!("expected resumed chapter"),
    }
    let after = Manifest::load(&chapter_dir, &plan).unwrap().chapter_spans.unwrap();
    assert_ne!(old.chunks[0].end, after.chunks[0].end, "rerender must change duration");
    let delta = after.chunks[0].end as i64 - old.chunks[0].end as i64;
    assert_eq!(
        after.chunks[1].start as i64 - old.chunks[1].start as i64,
        delta,
        "later span must shift with replaced audio"
    );
    let (_, final_pcm) = decoded(&file);
    assert_eq!(after.sample_count, final_pcm.len() as u64);
    std::fs::remove_dir_all(dir).ok();
}

fn unbase64(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for b in s.bytes().take_while(|&b| b != b'=') {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("invalid base64"),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

#[test]
fn feedback_page_contains_exact_final_pcm_and_rejects_incomplete_evidence() {
    use ggtts_book::render::feedback_chapter;
    use ggtts_core::seed::{hex, sha256};
    let (dir, plan, file) = fixture("page");
    let chapter_dir = dir.join(file.file_stem().unwrap());
    let manifest_path = chapter_dir.join("manifest.json");
    let original = std::fs::read(&manifest_path).unwrap();
    let manifest = Manifest::load(&chapter_dir, &plan).unwrap();
    let spans = manifest.chapter_spans.unwrap();
    let (_, final_pcm) = decoded(&file);
    let page = dir.join("page.html");
    feedback_chapter(&chapter_dir, &page).unwrap();
    let html = std::fs::read_to_string(&page).unwrap();
    let marker = "<script id=\"feedback-input\" type=\"application/json\">";
    let input = html.split_once(marker).unwrap().1.split_once("</script>").unwrap().0;
    let input: serde_json::Value = serde_json::from_str(input).unwrap();
    let items = input["items"].as_array().unwrap();
    assert_eq!(items.len(), spans.chunks.len());
    for (item, span) in items.iter().zip(&spans.chunks) {
        let encoded = item["audio"]
            .as_str()
            .unwrap()
            .strip_prefix("data:audio/flac;base64,")
            .unwrap();
        let bytes = unbase64(encoded);
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(bytes)).unwrap();
        let actual: Vec<i16> = reader.samples().map(|s| s.unwrap() as i16).collect();
        assert_eq!(actual, final_pcm[span.start as usize..span.end as usize]);
    }
    let before = std::fs::read(&page).unwrap();
    assert!(feedback_chapter(&chapter_dir, &page).is_err(), "must refuse overwrite");
    assert_eq!(std::fs::read(&page).unwrap(), before);
    let mutate = |change: &dyn Fn(&mut serde_json::Value)| {
        let mut outer: serde_json::Value = serde_json::from_slice(&original).unwrap();
        change(&mut outer["manifest"]);
        outer["sha256"] = hex(&sha256(serde_json::to_string(&outer["manifest"]).unwrap().as_bytes())).into();
        std::fs::write(&manifest_path, serde_json::to_vec(&outer).unwrap()).unwrap();
    };
    let bad_page = dir.join("bad.html");
    mutate(&|body| {
        body["chapter_spans"]["chunks"][1][0] = (spans.chunks[1].start + 1).into();
    });
    assert!(
        feedback_chapter(&chapter_dir, &bad_page).is_err(),
        "shifted span must fail even with valid manifest checksum"
    );
    assert!(!bad_page.exists());
    mutate(&|body| {
        body["chunks"][0].as_object_mut().unwrap().remove("kept_stop");
    });
    assert!(
        feedback_chapter(&chapter_dir, &bad_page).is_err(),
        "missing stop must not be guessed"
    );
    assert!(!bad_page.exists());
    std::fs::write(&manifest_path, original).unwrap();
    std::fs::write(chapter_dir.join(chunk_file(0)), b"tampered chunk").unwrap();
    assert!(
        feedback_chapter(&chapter_dir, &bad_page).is_err(),
        "tampered chunk must fail"
    );
    assert!(!bad_page.exists());
    std::fs::remove_dir_all(dir).ok();
}
