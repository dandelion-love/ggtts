use ggtts_core::audio::Verdict;
use ggtts_core::engine::Stop;
use ggtts_core::feedback::{from_read, input_digest, render_html};
use ggtts_core::flac::to_i16;
use ggtts_core::pipeline::{ChunkRecord, Read};
use serde_json::json;
use std::io::Cursor;

fn fixture() -> Read {
    let chunk = |i, text: &str, span, draw, seed| ChunkRecord {
        i,
        text: text.into(),
        seed,
        redraws: draw,
        decoded_samples: 4,
        decoded_sample_rate: 24000,
        first_draw_defect: if draw > 0 { Some(Verdict::Silent) } else { None },
        final_defect: None,
        s_per_char: 1.0,
        first_s_per_char: 1.0,
        span,
        stop: Stop::Eos,
    };
    Read {
        chunks: vec![chunk(0, "中😀", (0, 2), 1, 42), chunk(1, "文", (3, 5), 0, 43)],
        samples: vec![0.25, -0.5, 0.0, 0.75, -1.0],
        sample_rate: 24000,
        gain: Some(0.5),
        frames_generated: 3,
        timings: None,
        stream: None,
    }
}
fn decode64(s: &str) -> Vec<u8> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut acc = 0u32;
    let mut bits = 0;
    let mut out = Vec::new();
    for c in s.bytes().take_while(|c| *c != b'=') {
        let v = alphabet.iter().position(|x| *x == c).unwrap() as u32;
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
#[test]
fn page_audio_matches_final_output_samples_and_kept_draw() {
    let read = fixture();
    let bundle = from_read(&read, "read-1", json!({"model_sha256":"known"})).unwrap();
    let items = bundle.input["items"].as_array().unwrap();
    for (i, item) in items.iter().enumerate() {
        let data = item["audio"]
            .as_str()
            .unwrap()
            .strip_prefix("data:audio/flac;base64,")
            .unwrap();
        let bytes = decode64(data);
        let mut flac = claxon::FlacReader::new(Cursor::new(bytes)).unwrap();
        let actual: Vec<i16> = flac.samples().map(|x| x.unwrap() as i16).collect();
        let (a, b) = read.chunks[i].span;
        let expected: Vec<i16> = read.samples[a..b].iter().map(|v| to_i16(*v)).collect();
        assert_eq!(actual, expected);
    }
    assert_eq!(bundle.provenance["chunks"][0]["draw"], 1);
    assert_eq!(bundle.provenance["chunks"][0]["seed"], 42);
    assert_eq!(bundle.provenance["chunks"][1]["start"], 3);
    assert_eq!(bundle.provenance["input_sha256"], input_digest(&bundle.input).unwrap());
    assert!(render_html(&bundle.input).unwrap().contains("feedback-input-sha256"));
}
#[test]
fn rejects_invalid_evidence() {
    let read = fixture();
    assert!(from_read(&read, "bad id", json!({"model":"x"})).is_err());
    assert!(from_read(&read, "ok", json!({})).is_err());
    let mut bad = read.clone();
    bad.chunks[1].span = (1, 5);
    assert!(from_read(&bad, "ok", json!({"model":"x"})).is_err());
    let mut bad = read.clone();
    bad.chunks[0].decoded_sample_rate = 16000;
    assert!(from_read(&bad, "ok", json!({"model":"x"})).is_err());
    let mut bad = read.clone();
    bad.chunks[1].span = (100, 101);
    assert!(from_read(&bad, "ok", json!({"model":"x"})).is_err());
    let mut bad = read.clone();
    bad.samples[2] = 0.5;
    assert!(from_read(&bad, "ok", json!({"model":"x"})).is_err());
    assert!(render_html(&json!({"format":"ggtts-feedback-input/1","id":"x","items":[{"id":"a","text":"x","audio":"https://example.invalid"}]})).is_err());
    let mut bad = read;
    bad.samples[0] = f32::NAN;
    assert!(from_read(&bad, "ok", json!({"model":"x"})).is_err());
}
#[test]
fn canonical_digest_matches_python_for_unicode_and_markup() {
    let input = json!({"format":"ggtts-feedback-input/1","id":"a","items":[{"id":"x","text":"中😀</script>","audio":"data:audio/flac;base64,AA=="}]});
    assert_eq!(
        input_digest(&input).unwrap(),
        "581b03d52a58ea7bf796a1d07f0baa5d74f54734c3b02afa750c412210bc3a6a"
    );
}
