//! CPU integration: feedback clips match the exact kept spans in the final levelled read.
use std::collections::HashMap;
use std::io::Cursor;

use ggtts_core::audio::trim_edges_f32;
use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::feedback::from_read;
use ggtts_core::flac::{encode_flac, to_i16};
use ggtts_core::pipeline::{render_story, RenderOptions, SeedSource, StorySeeds};
use ggtts_core::text::PlannedChunk;
use ggtts_fake::FakeEngine;
use serde_json::json;

const STORY_ID: &str = "feedback-integration";
const FIRST: &str = "中";
const SECOND: &str = "文";

fn rendered() -> (FakeEngine, RenderOptions, ggtts_core::pipeline::Read) {
    let mut script = HashMap::new();
    script.insert(FIRST.to_string(), vec![1.2, 0.25]);
    script.insert(SECOND.to_string(), vec![0.25]);
    let mut engine = FakeEngine::new(script, HashMap::new());
    let mut opts = RenderOptions::for_profile(engine.profile());
    opts.redraw = 1;
    let chunks = vec![
        PlannedChunk {
            para: 0,
            text: FIRST.into(),
            punct: String::new(),
            last_in_para: false,
        },
        PlannedChunk {
            para: 0,
            text: SECOND.into(),
            punct: ".".into(),
            last_in_para: true,
        },
    ];
    let read = render_story(&mut engine, STORY_ID, &chunks, &opts).unwrap();
    (engine, opts, read)
}

fn decode_base64(s: &str) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in s.bytes().take_while(|b| *b != b'=') {
        let v = ALPHABET.iter().position(|x| *x == b).expect("base64 alphabet") as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

fn flac_samples(bytes: Vec<u8>) -> (u32, Vec<i16>) {
    let mut reader = claxon::FlacReader::new(Cursor::new(bytes)).unwrap();
    let rate = reader.streaminfo().sample_rate;
    let samples = reader.samples().map(|x| x.unwrap() as i16).collect();
    (rate, samples)
}

fn encode_base64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(A[((n >> 18) & 63) as usize] as char);
        out.push(A[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            A[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            A[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn rejected_draw_flac(opts: &RenderOptions) -> Vec<u8> {
    let seeds = StorySeeds {
        story_id: STORY_ID,
        salt: &opts.seed_salt,
    };
    let mut engine = FakeEngine::new(HashMap::from([(FIRST.to_string(), vec![1.2])]), HashMap::new());
    let req = ChunkRequest {
        text: FIRST,
        seed: seeds.seed(0, FIRST, 0),
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let generated = engine.generate(&req).unwrap();
    let decoded = engine.decode(&generated.codes).unwrap();
    let trim = trim_edges_f32(engine.profile(), &decoded.samples, decoded.sample_rate, opts.edge_s);
    let pcm: Vec<i16> = decoded.samples[trim.start_sample..trim.end_sample]
        .iter()
        .map(|&x| to_i16(x))
        .collect();
    encode_flac(&pcm, decoded.sample_rate).unwrap()
}

fn assert_page_matches_output(
    read: &ggtts_core::pipeline::Read,
    bundle: &ggtts_core::feedback::FeedbackBundle,
    output: &[i16],
) {
    for (i, chunk) in read.chunks.iter().enumerate() {
        let uri = bundle.input["items"][i]["audio"]
            .as_str()
            .unwrap()
            .strip_prefix("data:audio/flac;base64,")
            .unwrap();
        let (_, clip) = flac_samples(decode_base64(uri));
        let record = &bundle.provenance["chunks"][i];
        let start = record["start"].as_u64().unwrap() as usize;
        let end = record["end"].as_u64().unwrap() as usize;
        assert_eq!((start, end), chunk.span, "provenance span for chunk {i}");
        assert!(
            clip == output[start..end],
            "page chunk {i} must match final output span"
        );
    }
}

#[test]
fn redrawn_chunks_are_kept_and_match_final_levelled_output_spans() {
    let (_engine, opts, read) = rendered();
    assert_eq!(read.chunks.len(), 2);
    assert_eq!(read.chunks[0].redraws, 1);
    assert!(read.chunks[0].first_draw_defect.is_some());
    assert_eq!(read.chunks[0].final_defect, None);
    assert_eq!(read.chunks[1].redraws, 0);
    assert!(
        read.gain.is_some_and(|g| (g - 1.0).abs() > 1e-6),
        "different fake chunk amplitudes exercise whole-read levelling"
    );

    let bundle = from_read(&read, "render-read", json!({"engine":"fake","profile":"qwen3-tts"})).unwrap();
    let final_pcm: Vec<i16> = read.samples.iter().map(|&s| to_i16(s)).collect();
    let (final_rate, output_pcm) = flac_samples(encode_flac(&final_pcm, read.sample_rate).unwrap());
    assert_eq!(final_rate, read.sample_rate);
    assert_eq!(output_pcm, final_pcm);

    for (i, chunk) in read.chunks.iter().enumerate() {
        let uri = bundle.input["items"][i]["audio"]
            .as_str()
            .unwrap()
            .strip_prefix("data:audio/flac;base64,")
            .unwrap();
        let (rate, clip) = flac_samples(decode_base64(uri));
        let (start, end) = chunk.span;
        assert_eq!(rate, read.sample_rate);
        assert_eq!(
            clip,
            output_pcm[start..end],
            "page chunk {i} must match final output span"
        );
        let record = &bundle.provenance["chunks"][i];
        assert_eq!(record["chunk"], i);
        assert_eq!(record["draw"], chunk.redraws);
        assert_eq!(record["seed"], chunk.seed);
        assert_eq!(record["start"], start);
        assert_eq!(record["end"], end);
        assert_eq!(bundle.input["items"][i]["text"], chunk.text);
    }

    let seeds = StorySeeds {
        story_id: STORY_ID,
        salt: &opts.seed_salt,
    };
    let mut rejected_engine = FakeEngine::new(HashMap::from([(FIRST.to_string(), vec![1.2])]), HashMap::new());
    let req = ChunkRequest {
        text: FIRST,
        seed: seeds.seed(0, FIRST, 0),
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let generated = rejected_engine.generate(&req).unwrap();
    let decoded = rejected_engine.decode(&generated.codes).unwrap();
    let trim = trim_edges_f32(
        rejected_engine.profile(),
        &decoded.samples,
        decoded.sample_rate,
        opts.edge_s,
    );
    let rejected_len = trim.end_sample - trim.start_sample;
    let uri = bundle.input["items"][0]["audio"]
        .as_str()
        .unwrap()
        .strip_prefix("data:audio/flac;base64,")
        .unwrap();
    let (_, kept_page) = flac_samples(decode_base64(uri));
    assert!(kept_page.len() < rejected_len, "page must omit rejected long draw");
}

#[test]
fn missing_or_conflicting_read_evidence_is_refused() {
    let (_, _, read) = rendered();
    assert!(
        from_read(&read, "render-read", json!({})).is_err(),
        "missing provenance"
    );
    let mut overlap = read.clone();
    overlap.chunks[1].span = overlap.chunks[0].span;
    assert!(
        from_read(&overlap, "render-read", json!({"engine":"fake"})).is_err(),
        "overlapping spans"
    );
    let mut bad_rate = read.clone();
    bad_rate.chunks[0].decoded_sample_rate += 1;
    assert!(
        from_read(&bad_rate, "render-read", json!({"engine":"fake"})).is_err(),
        "conflicting sample rate"
    );
    let mut no_samples = read;
    no_samples.samples.clear();
    assert!(
        from_read(&no_samples, "render-read", json!({"engine":"fake"})).is_err(),
        "missing final samples"
    );
}

#[test]
#[should_panic(expected = "page chunk 0 must match final output span")]
fn sample_identity_oracle_fails_when_rejected_draw_replaces_kept_audio() {
    let (_engine, opts, read) = rendered();
    let mut planted = from_read(&read, "render-read", json!({"engine":"fake"})).unwrap();
    let rejected = rejected_draw_flac(&opts);
    planted.input["items"][0]["audio"] = format!("data:audio/flac;base64,{}", encode_base64(&rejected)).into();
    let output: Vec<i16> = read.samples.iter().map(|&x| to_i16(x)).collect();
    assert_page_matches_output(&read, &planted, &output);
}
