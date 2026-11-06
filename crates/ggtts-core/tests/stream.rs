//! Chunk-level streaming: each chunk's piece arrives as soon as it is kept, the pieces are the batch read before
//! its level step sample for sample, and levelling them gives the batch read exactly.
//!
//! The parity check is shown to see what it rules out: a sample dropped at a seam, a chunk handed over before its
//! trim, and a missing gap each turn it red at the right sample. A one-chunk story, where the stream is a single
//! piece, reads as identical.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::{cases, Case};
use ggtts_core::audio::trim_edges_f32;
use ggtts_core::engine::Engine;
use ggtts_core::flac::to_i16;
use ggtts_core::pipeline::{gap_samples, level, render_story, render_story_streamed, Read, RenderError};
use ggtts_core::stream::{first_sample_diff, raw_header, Container, StreamWriter, UNKNOWN_LENGTH};
use ggtts_core::trace::{NullEvents, Recorder};
use ggtts_fake::FakeEngine;

struct Streamed {
    pieces: Vec<(usize, usize, Vec<f32>, u32)>,
    read: Read,
    engine: FakeEngine,
}

fn stream(case: &Case) -> Streamed {
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let mut pieces = Vec::new();
    let read = render_story_streamed(&mut engine, &case.story_id, &chunks, &opts, &mut NullEvents, &mut |p| {
        pieces.push((p.chunk, p.start, p.samples.to_vec(), p.sample_rate));
        Ok(())
    })
    .unwrap();
    Streamed { pieces, read, engine }
}

fn concat(pieces: &[(usize, usize, Vec<f32>, u32)]) -> Vec<f32> {
    pieces.iter().flat_map(|p| p.2.iter().copied()).collect()
}

/// The stream as the spec defines it, rebuilt from the engine's kept clips: each trimmed, then its gap.
fn expected_stream(case: &Case, s: &Streamed) -> Vec<f32> {
    let profile = *s.engine.profile();
    let chunks = case.chunks(&s.engine);
    let opts = case.opts(&s.engine);
    let rate = s.read.sample_rate;
    let mut out = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        let clip = &s.engine.kept[&c.text];
        let clip_rate = case
            .shape
            .get(&c.text)
            .and_then(|sh| sh.sample_rate)
            .unwrap_or(ggtts_fake::SAMPLE_RATE);
        let t = trim_edges_f32(&profile, clip, clip_rate, opts.edge_s);
        out.extend_from_slice(&clip[t.start_sample..t.end_sample]);
        if i + 1 < chunks.len() {
            out.resize(out.len() + gap_samples(&profile, &c.punct, c.last_in_para, rate), 0.0);
        }
    }
    out
}

fn levelled(case: &Case, s: &Streamed, mut samples: Vec<f32>) -> Vec<f32> {
    let gain = level(s.engine.profile(), &mut samples);
    assert_eq!(gain, s.read.gain, "{}: the stream's level gain is the batch's", case.id);
    samples
}

#[test]
fn the_stream_is_the_batch_read_before_its_level() {
    for case in cases() {
        let s = stream(&case);
        let mut batch_engine = case.engine();
        let chunks = case.chunks(&batch_engine);
        let batch_opts = case.opts(&batch_engine);
        let batch = render_story(&mut batch_engine, &case.story_id, &chunks, &batch_opts).unwrap();
        assert_eq!(
            s.read.samples, batch.samples,
            "{}: streaming changes the batch read",
            case.id
        );
        assert_eq!(s.read.chunks, batch.chunks, "{}", case.id);

        assert_eq!(s.pieces.len(), chunks.len(), "{}: one piece per chunk", case.id);
        let mut at = 0;
        for (n, (chunk, start, samples, rate)) in s.pieces.iter().enumerate() {
            assert_eq!(
                (*chunk, *start),
                (n, at),
                "{}: pieces are in order and contiguous",
                case.id
            );
            assert_eq!(*rate, s.read.sample_rate);
            assert_eq!(
                s.read.chunks[n].span.0, *start,
                "{}: a piece starts at its chunk",
                case.id
            );
            at += samples.len();
        }

        let got = concat(&s.pieces);
        assert_eq!(
            first_sample_diff(&got, &expected_stream(&case, &s)),
            None,
            "{}: stream vs spec",
            case.id
        );
        let lv = levelled(&case, &s, got);
        assert_eq!(
            first_sample_diff(&lv, &batch.samples),
            None,
            "{}: levelled stream vs batch",
            case.id
        );
    }
}

#[test]
fn each_piece_arrives_before_the_next_chunk_is_generated() {
    for case in cases() {
        let mut engine = case.engine();
        let chunks = case.chunks(&engine);
        let opts = case.opts(&engine);
        let mut rec = Recorder::new(1);
        render_story_streamed(&mut engine, &case.story_id, &chunks, &opts, &mut rec, &mut |_| Ok(())).unwrap();
        let order: Vec<(String, u64)> = rec
            .lines
            .iter()
            .filter(|l| l["ev"] == "chunk_begin" || l["ev"] == "emit")
            .map(|l| (l["ev"].as_str().unwrap().to_string(), l["chunk"].as_u64().unwrap()))
            .collect();
        for w in order.windows(2) {
            if w[1].0 == "chunk_begin" && w[1].1 > w[0].1 {
                assert_eq!(
                    w[0],
                    ("emit".to_string(), w[1].1 - 1),
                    "{}: chunk {} began first",
                    case.id,
                    w[1].1
                );
            }
        }
    }
}

#[test]
fn the_parity_check_sees_a_planted_seam_defect() {
    let case = cases()
        .into_iter()
        .find(|c| c.paragraphs.join("").chars().count() > 40)
        .unwrap();
    let s = stream(&case);
    assert!(s.pieces.len() >= 2, "{}: needs a seam", case.id);
    let want = expected_stream(&case, &s);
    let seam = s.pieces[1].1;

    // A sample dropped at the first seam.
    let mut dropped = concat(&s.pieces);
    dropped.remove(seam);
    let at = first_sample_diff(&dropped, &want).expect("a dropped sample is red");
    // The seam sits in kept edge silence, all zeros, so the shift shows where the second chunk's speech begins:
    // inside the second piece, never before the seam.
    let second_end = seam + s.pieces[1].2.len();
    assert!(
        (seam..second_end).contains(&at),
        "{}: named {at}, the second piece is {seam}..{second_end}",
        case.id
    );
    assert!(first_sample_diff(&levelled_any(&case, &s, dropped), &s.read.samples).is_some());

    // The first chunk handed over before its trim.
    let chunks = case.chunks(&s.engine);
    let mut untrimmed = s.engine.kept[&chunks[0].text].clone();
    untrimmed.extend_from_slice(&s.pieces[0].2[s.read.chunks[0].span.1..]);
    let mut planted = untrimmed;
    for p in &s.pieces[1..] {
        planted.extend_from_slice(&p.2);
    }
    // Both start in silence, so the first difference is where the trimmed chunk's speech begins.
    let at = first_sample_diff(&planted, &want).expect("an untrimmed chunk is red");
    assert!(
        at < s.read.chunks[0].span.1,
        "{}: named {at}, inside the first chunk",
        case.id
    );

    // The gap after the first chunk left out.
    let mut gapless = s.pieces[0].2[..s.read.chunks[0].span.1].to_vec();
    for p in &s.pieces[1..] {
        gapless.extend_from_slice(&p.2);
    }
    // The gap runs into the next chunk's edge silence, so the loss shows at that chunk's speech, past the gap's start.
    let at = first_sample_diff(&gapless, &want).expect("a missing gap is red");
    assert!(
        (s.read.chunks[0].span.1..second_end).contains(&at),
        "{}: named {at}",
        case.id
    );
}

/// Level without asserting the gain: for planted streams, whose gain may differ.
fn levelled_any(_case: &Case, s: &Streamed, mut samples: Vec<f32>) -> Vec<f32> {
    level(s.engine.profile(), &mut samples);
    samples
}

#[test]
fn a_one_piece_stream_reads_as_identical() {
    let mut one = cases().remove(0);
    let first = one.chunks(&one.engine())[0].text.clone();
    one.paragraphs = vec![first];
    let s = stream(&one);
    assert_eq!(s.pieces.len(), 1);
    assert_eq!(first_sample_diff(&concat(&s.pieces), &expected_stream(&one, &s)), None);
    assert_eq!(
        first_sample_diff(&levelled(&one, &s, concat(&s.pieces)), &s.read.samples),
        None
    );
}

/// A shared buffer, so a test can read the stream between writes, as a player would.
#[derive(Clone, Default)]
struct Shared(Rc<RefCell<Vec<u8>>>);

impl std::io::Write for Shared {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Read a WAV stream so far, as a player would: the header, then whole samples.
fn read_wav_so_far(bytes: &[u8]) -> (u32, Vec<i16>) {
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "PCM");
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "16-bit");
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), UNKNOWN_LENGTH);
    let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let data = &bytes[44..];
    (
        rate,
        data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect(),
    )
}

#[test]
fn a_reader_plays_the_stream_as_it_arrives() {
    for case in cases() {
        for container in [Container::Wav, Container::RawS16] {
            let buf = Shared::default();
            let mut w = StreamWriter::new(buf.clone(), container);
            let mut engine = case.engine();
            let chunks = case.chunks(&engine);
            let opts = case.opts(&engine);
            let mut seen = Vec::new();
            let mut reads = 0;
            render_story_streamed(&mut engine, &case.story_id, &chunks, &opts, &mut NullEvents, &mut |p| {
                w.write(p.samples, p.sample_rate).unwrap();
                seen.extend(p.samples.iter().map(|&x| to_i16(x)));
                // Read what has arrived so far: every sample written is already readable.
                let bytes = buf.0.borrow().clone();
                let (rate, got) = match container {
                    Container::Wav => read_wav_so_far(&bytes),
                    Container::RawS16 => {
                        let head = raw_header(p.sample_rate);
                        assert!(bytes.starts_with(head.as_bytes()));
                        let data = &bytes[head.len()..];
                        (
                            p.sample_rate,
                            data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect(),
                        )
                    }
                };
                assert_eq!(rate, p.sample_rate);
                assert_eq!(got, seen, "{}: the stream so far", case.id);
                reads += 1;
                Ok(())
            })
            .unwrap();
            assert_eq!(reads, chunks.len());
        }
    }
}

#[test]
fn a_failing_sink_stops_the_render_before_the_next_chunk() {
    let case = cases()
        .into_iter()
        .find(|c| c.paragraphs.join("").chars().count() > 40)
        .unwrap();
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    assert!(chunks.len() >= 2, "{}: needs two chunks", case.id);
    let opts = case.opts(&engine);
    let mut handed = 0;
    let got = render_story_streamed(
        &mut engine,
        &case.story_id,
        &chunks,
        &opts,
        &mut NullEvents,
        &mut |_| {
            handed += 1;
            Err("the reader went away".into())
        },
    );
    assert!(matches!(got, Err(RenderError::Sink(_))), "a refused piece is an error");
    assert_eq!(handed, 1, "no piece after the refused one");
    let generated: std::collections::HashSet<&str> = engine.calls.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(generated.len(), 1, "{}: only the first chunk was generated", case.id);
}
