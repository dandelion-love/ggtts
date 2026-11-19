//! Black-box feedback-page tests on ggtts-render's CPU fake engine.
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ggtts-feedback-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn render(dir: &Path, extra: &[&str]) -> Output {
    let story = dir.join("story.txt");
    std::fs::write(&story, "A short read\n\u{4e2d}\n\u{6587}").unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ggtts-render"));
    cmd.args(["--engine", "fake", "--story"]).arg(story).args(extra);
    cmd.output().unwrap()
}

fn data_uri_to_bytes(uri: &str) -> Vec<u8> {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let encoded = uri.strip_prefix("data:audio/flac;base64,").expect("FLAC data URI");
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in encoded.bytes().take_while(|b| *b != b'=') {
        let v = A.iter().position(|x| *x == b).expect("base64 alphabet") as u32;
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

fn decode_flac(bytes: Vec<u8>) -> (u32, Vec<i16>) {
    let mut reader = claxon::FlacReader::new(Cursor::new(bytes)).unwrap();
    let rate = reader.streaminfo().sample_rate;
    let samples = reader.samples().map(|s| s.unwrap() as i16).collect();
    (rate, samples)
}

fn input_from_page(html: &str) -> Value {
    const START: &str = "<script id=\"feedback-input\" type=\"application/json\">";
    let start = html.find(START).unwrap() + START.len();
    let end = html[start..].find("</script>").unwrap() + start;
    serde_json::from_str(&html[start..end]).unwrap()
}

#[test]
fn cli_page_chunks_are_exact_spans_of_the_written_levelled_flac() {
    let dir = scratch("spans");
    let audio = dir.join("final.flac");
    let page = dir.join("feedback.html");
    let output = render(
        &dir,
        &[
            "--out",
            audio.to_str().unwrap(),
            "--feedback-page",
            page.to_str().unwrap(),
            "--redraw",
            "1",
            "--log-level",
            "error",
        ],
    );
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let companion = PathBuf::from(format!("{}.provenance.json", page.display()));
    assert!(audio.is_file() && page.is_file() && companion.is_file());

    let input = input_from_page(&std::fs::read_to_string(&page).unwrap());
    let provenance: Value = serde_json::from_slice(&std::fs::read(companion).unwrap()).unwrap();
    assert_eq!(input["format"], "ggtts-feedback-input/1");
    assert_eq!(provenance["format"], "ggtts-feedback-provenance/1");
    assert_eq!(provenance["input_sha256"].as_str().unwrap().len(), 64);

    let (rate, output_pcm) = decode_flac(std::fs::read(&audio).unwrap());
    assert_eq!(rate, provenance["sample_rate"].as_u64().unwrap() as u32);
    assert_eq!(
        output_pcm.len(),
        provenance["output_samples"].as_u64().unwrap() as usize
    );
    let items = input["items"].as_array().unwrap();
    let records = provenance["chunks"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(records.len(), items.len());
    assert_eq!(items[0]["text"], "\u{4e2d}");
    assert_eq!(items[1]["text"], "\u{6587}");

    for (i, (item, record)) in items.iter().zip(records).enumerate() {
        let start = record["start"].as_u64().unwrap() as usize;
        let end = record["end"].as_u64().unwrap() as usize;
        assert!(start < end && end <= output_pcm.len(), "chunk {i} span");
        let (clip_rate, clip) = decode_flac(data_uri_to_bytes(item["audio"].as_str().unwrap()));
        assert_eq!(clip_rate, rate, "chunk {i} rate");
        assert_eq!(
            clip,
            output_pcm[start..end],
            "page chunk {i} must be the final output slice"
        );
        assert_eq!(record["draw"], 0, "fake render accepts its first draw in this CLI case");
        assert_eq!(record["chunk"], i);
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn cli_refuses_missing_or_conflicting_output_evidence_before_rendering() {
    let dir = scratch("refuse");
    let page = dir.join("feedback.html");
    let no_out = render(&dir, &["--feedback-page", page.to_str().unwrap()]);
    assert!(!no_out.status.success());
    assert!(String::from_utf8_lossy(&no_out.stderr).contains("--feedback-page requires --out"));
    assert!(!page.exists());

    let shared = dir.join("same.flac");
    let alias = render(
        &dir,
        &[
            "--out",
            shared.to_str().unwrap(),
            "--feedback-page",
            shared.to_str().unwrap(),
        ],
    );
    assert!(!alias.status.success());
    assert!(String::from_utf8_lossy(&alias.stderr).contains("distinct output paths"));
    assert!(!shared.exists());
    let _ = std::fs::remove_dir_all(dir);
}
