//! The pipeline's parity cases (`fixtures/pipeline/*.json`), run through the core and the fake engine.
//!
//! Each case was recorded from the reference implementation. A case passes only when every field matches:
//! integers, strings and hashes exactly, and the values the case file rounded at the same rounding. Cases the core
//! does not implement are listed by name in `out-of-scope.json` and reported as such, never counted as passes.
//!
//! `planted_ceiling_44_turns_red` is the planted change: the same cases at a story ceiling of 44 must fail, and
//! the failures must be named. Run with `--nocapture` to see the summary.

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use ggtts_core::audio::{downmix_stereo, judge, judge_f32, trim_edges, TrimInfo, Verdict};
use ggtts_core::flac::{to_i16, write_flac};
use ggtts_core::numeric::{log10, mean_f64};
use ggtts_core::pipeline::{render_story, RenderOptions, DEFAULT_REDRAW, EDGE_S, PASSAGE_CEILING, STORY_CEILING};
use ggtts_core::seed::{hex, seed_key, seed_of, sha256};
use ggtts_core::text::{chunk, cjk_len, plan, read_text};
use ggtts_fake::{signal, FakeEngine, Shape};
use serde_json::Value;

const FORMAT: &str = "ggtts-pipeline-cases/1";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/pipeline")
}

fn load(name: &str) -> Value {
    let path = fixtures().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(v["format"], FORMAT, "{name}: unexpected fixture format");
    v
}

#[derive(Default)]
struct Report {
    passed: Vec<String>,
    failed: Vec<(String, Vec<String>)>,
}

impl Report {
    fn record(&mut self, name: String, diffs: Vec<String>) {
        if diffs.is_empty() {
            self.passed.push(name);
        } else {
            self.failed.push((name, diffs));
        }
    }
}

/// Collects the differences of one case.
#[derive(Default)]
struct Diffs(Vec<String>);

impl Diffs {
    fn eq<T: PartialEq + Debug>(&mut self, what: &str, got: T, want: T) {
        if got != want {
            self.0.push(format!("{what}: got {got:?}, want {want:?}"));
        }
    }
    /// A value the case file rounded to `places` decimals, compared at that rounding.
    fn rounded(&mut self, what: &str, got: f64, want: &Value, places: usize) {
        let want = want.as_f64().unwrap_or(f64::NAN);
        let (g, w) = (format!("{got:.places$}"), format!("{want:.places$}"));
        if g != w {
            self.0.push(format!("{what}: got {g} ({got:e}), want {w}"));
        }
    }
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| panic!("not a string: {v}"))
}
fn u(v: &Value) -> u64 {
    v.as_u64().unwrap_or_else(|| panic!("not an unsigned integer: {v}"))
}
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}
fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| s(x).to_string()).collect()
}
fn verdict(v: Option<Verdict>) -> Value {
    v.map_or(Value::Null, |v| Value::from(v.as_str()))
}

// ── defaults ─────────────────────────────────────────────────────────────────────────────────────────────────
fn run_defaults(r: &mut Report, story_ceiling: usize) {
    let d = &load("defaults.json")["defaults"];
    let mut x = Diffs::default();
    x.eq("story ceiling", story_ceiling as u64, u(&d["story_ceiling"]));
    x.eq("passage ceiling", PASSAGE_CEILING as u64, u(&d["passage_ceiling"]));
    x.eq("redraw", u64::from(DEFAULT_REDRAW), u(&d["redraw"]));
    x.eq(
        "seed salt",
        RenderOptions::default().seed_salt.as_str(),
        s(&d["seed_salt"]),
    );
    x.eq("edge cap", EDGE_S, f(&d["edge_s"]));
    r.record("defaults".into(), x.0);
}

// ── chunks, read_text, seeds ─────────────────────────────────────────────────────────────────────────────────
fn check_plan(x: &mut Diffs, label: &str, story_id: &str, paras: &[String], ceiling: usize, want: &Value) {
    let got = plan(paras, ceiling);
    let want = want["chunks"].as_array().unwrap();
    x.eq(&format!("{label}: chunk count"), got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let p = format!("{label}[{i}]");
        x.eq(&format!("{p}.para"), g.para as u64, u(&w["para"]));
        x.eq(&format!("{p}.text"), g.text.as_str(), s(&w["text"]));
        x.eq(&format!("{p}.punct"), g.punct.as_str(), s(&w["punct"]));
        x.eq(
            &format!("{p}.last_in_para"),
            Some(g.last_in_para),
            w["last_in_para"].as_bool(),
        );
        let body_len = (g.text.chars().count() - g.punct.chars().count()) as u64;
        x.eq(&format!("{p}.body_len"), body_len, u(&w["body_len"]));
        x.eq(
            &format!("{p}.seed_k0"),
            u64::from(seed_of(story_id, i, &g.text, 0, "")),
            u(&w["seed_k0"]),
        );
        x.eq(
            &format!("{p}.seed_k1"),
            u64::from(seed_of(story_id, i, &g.text, 1, "")),
            u(&w["seed_k1"]),
        );
    }
}

fn run_chunks(r: &mut Report, story_ceiling: usize) {
    let v = load("chunks.json");
    for c in v["cases"].as_array().unwrap() {
        let id = s(&c["id"]);
        let story_id = s(&c["story_id"]);
        let paras = strings(&c["paragraphs"]);

        let mut x = Diffs::default();
        let raw: Vec<Vec<(String, String)>> = paras.iter().map(|p| chunk(p, story_ceiling)).collect();
        let want_raw: Vec<Vec<(String, String)>> = c["raw_chunks_story"]
            .as_array()
            .unwrap()
            .iter()
            .map(|para| {
                para.as_array()
                    .unwrap()
                    .iter()
                    .map(|t| (s(&t[0]).to_string(), s(&t[1]).to_string()))
                    .collect()
            })
            .collect();
        x.eq("chunk() pairs", raw, want_raw);
        check_plan(&mut x, "story", story_id, &paras, story_ceiling, &c["story"]);
        r.record(format!("chunks/{id}/story"), x.0);

        let mut x = Diffs::default();
        check_plan(&mut x, "passage", story_id, &paras, PASSAGE_CEILING, &c["passage"]);
        r.record(format!("chunks/{id}/passage"), x.0);
    }
    for c in v["read_text"].as_array().unwrap() {
        let (title, paras) = read_text(s(&c["file_utf8"]));
        let mut x = Diffs::default();
        x.eq("title", title.as_str(), s(&c["title"]));
        x.eq("paragraphs", paras, strings(&c["paragraphs"]));
        r.record(format!("chunks/{}", s(&c["id"])), x.0);
    }
}

fn run_seeds(r: &mut Report) {
    let v = load("seeds.json");
    for (n, row) in v["rows"].as_array().unwrap().iter().enumerate() {
        let (story, i, text, k, salt) = (
            s(&row["story"]),
            u(&row["i"]) as usize,
            s(&row["text"]),
            u(&row["k"]) as u32,
            s(&row["salt"]),
        );
        let mut x = Diffs::default();
        x.eq("key", seed_key(story, i, text, k, salt).as_str(), s(&row["sha256_key"]));
        x.eq("seed", u64::from(seed_of(story, i, text, k, salt)), u(&row["seed"]));
        r.record(format!("seeds/{n}:{story}|{i}|k={k}|salt={salt}"), x.0);
    }
}

// ── trim and judge ───────────────────────────────────────────────────────────────────────────────────────────
/// A signal spec as float64 samples and its sample rate.
fn build_signal(spec: &Value) -> (Vec<f64>, u32) {
    let wide = |y: Vec<f32>| y.into_iter().map(f64::from).collect::<Vec<f64>>();
    match s(&spec["kind"]) {
        "tone" => {
            let sr = u(&spec["sample_rate"]) as u32;
            let y = signal::tone(f(&spec["speech_s"]), f(&spec["lead_s"]), f(&spec["tail_s"]), sr);
            (wide(y), sr)
        }
        "concat" => {
            let mut out = Vec::new();
            let mut rate = 0;
            for p in spec["parts"].as_array().unwrap() {
                let (y, sr) = build_signal(p);
                out.extend(y);
                rate = sr;
            }
            (out, rate)
        }
        "stereo-cancel" => (wide(downmix_stereo(&signal::stereo_cancel())), 24_000),
        "empty" => (Vec::new(), 24_000),
        "dc" => (
            signal::dc(
                f(&spec["edge"]),
                u(&spec["edge_n"]) as usize,
                f(&spec["middle"]),
                u(&spec["middle_n"]) as usize,
            ),
            u(&spec["sample_rate"]) as u32,
        ),
        other => panic!("unknown signal kind {other}"),
    }
}

fn check_trim(x: &mut Diffs, got: &TrimInfo, want: &Value) {
    x.eq("start_sample", got.start_sample as u64, u(&want["start_sample"]));
    x.eq("end_sample", got.end_sample as u64, u(&want["end_sample"]));
    x.eq("raw_samples", got.raw_samples as u64, u(&want["raw_samples"]));
    x.rounded("raw_s", got.raw_s, &want["raw_s"], 3);
    x.rounded("trimmed_s", got.trimmed_s, &want["trimmed_s"], 3);
    x.rounded("lead_cut_s", got.lead_cut_s, &want["lead_cut_s"], 3);
    x.rounded("tail_cut_s", got.tail_cut_s, &want["tail_cut_s"], 3);
    x.rounded("lead_silence_s", got.lead_silence_s, &want["lead_silence_s"], 3);
    x.rounded("tail_silence_s", got.tail_silence_s, &want["tail_silence_s"], 3);
}

fn run_trim(r: &mut Report) {
    let v = load("trim.json");
    for row in v["rows"].as_array().unwrap() {
        let (y, sr) = build_signal(&row["signal"]);
        let got = trim_edges(&y, sr, f(&row["max_edge_s"]));
        let mut x = Diffs::default();
        check_trim(&mut x, &got, &row["info"]);
        r.record(format!("trim/{}", s(&row["name"])), x.0);
    }
}

fn run_judge(r: &mut Report) {
    let v = load("judge.json");
    for row in v["rows"].as_array().unwrap() {
        let (y, sr) = build_signal(&row["signal"]);
        let text = s(&row["text"]);
        let j = judge(&y, sr, text);
        let mut x = Diffs::default();
        x.eq("verdict", verdict(j.verdict), row["verdict"].clone());
        x.rounded("s_per_char", j.s_per_char, &row["s_per_char"], 3);
        if let Some(n) = row["cjk_len"].as_u64() {
            x.eq("cjk_len", cjk_len(text) as u64, n);
        }
        r.record(format!("judge/{}", s(&row["name"])), x.0);
    }
    for row in v["cjk_len"].as_array().unwrap() {
        let cp = s(&row["codepoint"]);
        let c = char::from_u32(u32::from_str_radix(&cp[2..], 16).unwrap()).unwrap();
        let mut x = Diffs::default();
        x.eq("cjk_len", cjk_len(&c.to_string()) as u64, u(&row["cjk_len"]));
        r.record(format!("judge/cjk-len/{cp}"), x.0);
    }
}

// ── the story path end to end ────────────────────────────────────────────────────────────────────────────────
fn loud_rms_db(x: &[f32]) -> f64 {
    let sq: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(v))
        .filter(|v| v.abs() > 1e-3)
        .map(|v| v * v)
        .collect();
    if sq.is_empty() {
        f64::NEG_INFINITY
    } else {
        20.0 * log10(mean_f64(&sq).sqrt())
    }
}

fn check_level(x: &mut Diffs, read: &[f32], kept: &[f32], spans: &[(usize, usize)], want: &Value) {
    let pre = loud_rms_db(kept);
    let post = loud_rms_db(read);
    if pre.is_finite() && post.is_finite() {
        x.rounded("level.read_rms_dbfs", post, &want["read_rms_dbfs"], 4);
        x.rounded("level.engine_rms_dbfs", pre, &want["engine_rms_dbfs"], 4);
        x.rounded("level.applied_gain_db", post - pre, &want["applied_gain_db"], 4);
    } else {
        x.eq("level.read_rms_dbfs", Value::Null, want["read_rms_dbfs"].clone());
        x.eq("level.engine_rms_dbfs", Value::Null, want["engine_rms_dbfs"].clone());
        x.eq("level.applied_gain_db", Value::Null, want["applied_gain_db"].clone());
    }
    let peak = read.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    x.rounded("level.read_peak", f64::from(peak), &want["read_peak"], 6);
    let q: Vec<i16> = read.iter().map(|&v| to_i16(v)).collect();
    let bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    x.eq(
        "level.int16_sha256",
        hex(&sha256(&bytes)).as_str(),
        s(&want["int16_sha256"]),
    );
    let peaks: Vec<Value> = spans
        .iter()
        .map(|&(a, b)| {
            q.get(a..b).filter(|s| !s.is_empty()).map_or(Value::Null, |s| {
                Value::from(s.iter().map(|v| i32::from(*v).abs()).max().unwrap())
            })
        })
        .collect();
    x.eq(
        "level.int16_peak_per_span",
        Value::from(peaks),
        want["int16_peak_per_span"].clone(),
    );
    let first: Vec<i64> = q.iter().filter(|&&v| v != 0).take(6).map(|&v| i64::from(v)).collect();
    let want_first: Vec<i64> = want["int16_first_nonzero"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    x.eq("level.int16_first_nonzero", first, want_first);
}

fn run_render(r: &mut Report, story_ceiling: usize) {
    let v = load("render.json");
    for c in v["cases"].as_array().unwrap() {
        let id = s(&c["id"]);
        let mut x = Diffs::default();
        let script: HashMap<String, Vec<f64>> = c["script"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_array().unwrap().iter().map(f).collect()))
            .collect();
        let shape: HashMap<String, Shape> = c["shape"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Shape {
                        amplitude_scale: v["amplitude_scale"].as_f64(),
                        spike: v["spike"].as_f64(),
                        sample_rate: v["sample_rate"].as_u64().map(|n| n as u32),
                    },
                )
            })
            .collect();
        let mut engine = FakeEngine::new(script, shape);
        let chunks = plan(&strings(&c["paragraphs"]), story_ceiling);
        let opts = RenderOptions {
            redraw: u(&c["redraw"]) as u32,
            seed_salt: s(&c["seed_salt"]).to_string(),
            edge_s: EDGE_S,
        };
        let read = match render_story(&mut engine, s(&c["story_id"]), &chunks, &opts) {
            Ok(read) => read,
            Err(e) => {
                x.0.push(format!("render failed: {e}"));
                r.record(format!("render/{id}"), x.0);
                continue;
            }
        };

        let want_calls = c["engine_calls"].as_array().unwrap();
        x.eq("engine call count", engine.calls.len(), want_calls.len());
        for (n, (g, w)) in engine.calls.iter().zip(want_calls).enumerate() {
            x.eq(&format!("call[{n}].text"), g.text.as_str(), s(&w["text"]));
            x.eq(&format!("call[{n}].seed"), g.seed, u(&w["seed"]));
            x.eq(&format!("call[{n}].speech_s"), g.speech_s, f(&w["speech_s"]));
            x.eq(
                &format!("call[{n}].amplitude_scale"),
                g.amplitude_scale,
                f(&w["amplitude_scale"]),
            );
        }

        let res = &c["result"];
        let want_chunks = res["chunks"].as_array().unwrap();
        x.eq("chunk count", read.chunks.len(), want_chunks.len());
        for (g, w) in read.chunks.iter().zip(want_chunks) {
            let p = format!("chunk[{}]", g.i);
            x.eq(&format!("{p}.i"), g.i as u64, u(&w["i"]));
            x.eq(&format!("{p}.text"), g.text.as_str(), s(&w["text"]));
            x.eq(&format!("{p}.seed"), u64::from(g.seed), u(&w["seed"]));
            x.eq(&format!("{p}.redraws"), u64::from(g.redraws), u(&w["redraws"]));
            x.eq(
                &format!("{p}.first_draw_defect"),
                verdict(g.first_draw_defect),
                w["first_draw_defect"].clone(),
            );
            x.eq(
                &format!("{p}.final_defect"),
                verdict(g.final_defect),
                w["final_defect"].clone(),
            );
            x.rounded(&format!("{p}.s_per_char"), g.s_per_char, &w["s_per_char"], 3);
            x.rounded(
                &format!("{p}.first_s_per_char"),
                g.first_s_per_char,
                &w["first_s_per_char"],
                3,
            );
            let span = (u(&w["span"][0]) as usize, u(&w["span"][1]) as usize);
            x.eq(&format!("{p}.span"), g.span, span);
        }
        x.rounded("seconds", read.seconds(), &res["seconds"], 2);
        x.eq("sr", u64::from(read.sample_rate), u(&res["sr"]));
        x.eq(
            "first_draw_defects",
            read.first_draw_defects() as u64,
            u(&res["first_draw_defects"]),
        );
        x.eq("final_defects", read.final_defects() as u64, u(&res["final_defects"]));
        x.eq("read_samples", read.samples.len() as u64, u(&res["read_samples"]));

        let kept: Vec<f32> = read
            .chunks
            .iter()
            .flat_map(|ch| engine.kept.get(&ch.text).cloned().unwrap_or_default())
            .collect();
        let spans: Vec<(usize, usize)> = read.chunks.iter().map(|ch| ch.span).collect();
        check_level(&mut x, &read.samples, &kept, &spans, &res["level"]);
        r.record(format!("render/{id}"), x.0);
    }

    let w = &v["write_flac"];
    let mut x = Diffs::default();
    let (y, sr) = build_signal(&w["signal"]);
    let y: Vec<f32> = y.into_iter().map(|v| v as f32).collect();
    let dir = std::env::temp_dir().join(format!("ggtts-flac-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let written = write_flac(&dir.join(s(&w["asked_for"])), &y, sr).unwrap();
    x.eq(
        "written name",
        written.file_name().unwrap().to_string_lossy().to_string(),
        s(&w["written"]).to_string(),
    );
    match claxon::FlacReader::open(&written) {
        Ok(mut reader) => {
            let info = reader.streaminfo();
            x.eq("format", "FLAC", s(&w["format"]));
            x.eq(
                "subtype (bits)",
                info.bits_per_sample,
                if s(&w["subtype"]) == "PCM_16" { 16 } else { 0 },
            );
            x.eq("sample rate", u64::from(info.sample_rate), u(&w["sample_rate"]));
            x.eq("channels", u64::from(info.channels), u(&w["channels"]));
            x.eq("frames (streaminfo)", info.samples, Some(u(&w["frames"])));
            let back: Result<Vec<i32>, _> = reader.samples().collect();
            match back {
                Ok(back) => {
                    x.eq("frames (decoded)", back.len() as u64, u(&w["frames"]));
                    let want: Vec<i32> = y.iter().map(|&v| i32::from(to_i16(v))).collect();
                    x.eq("decoded samples equal the written ones", back == want, true);
                    let peak = back.iter().map(|v| v.abs()).max().unwrap_or(0);
                    x.eq("int16 peak", u64::from(peak.unsigned_abs()), u(&w["int16_peak"]));
                }
                Err(e) => x.0.push(format!("decode failed: {e}")),
            }
        }
        Err(e) => x.0.push(format!("not readable as FLAC: {e}")),
    }
    let _ = std::fs::remove_dir_all(&dir);
    r.record(format!("render/{}", s(&w["id"])), x.0);
}

// ── the suite ────────────────────────────────────────────────────────────────────────────────────────────────
fn run_all(story_ceiling: usize) -> Report {
    let mut r = Report::default();
    run_defaults(&mut r, story_ceiling);
    run_chunks(&mut r, story_ceiling);
    run_seeds(&mut r);
    run_trim(&mut r);
    run_judge(&mut r);
    run_render(&mut r, story_ceiling);
    r
}

fn out_of_scope() -> Vec<String> {
    load("out-of-scope.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| format!("{}/{} ({})", s(&c["file"]), s(&c["id"]), s(&c["reason"])))
        .collect()
}

fn summary(r: &Report, oos: &[String]) -> String {
    let mut out = format!(
        "pipeline cases: {} passed, {} failed, {} out of scope\n",
        r.passed.len(),
        r.failed.len(),
        oos.len()
    );
    for (name, diffs) in &r.failed {
        out += &format!("  FAILED {name}\n");
        for d in diffs.iter().take(8) {
            out += &format!("      {d}\n");
        }
        if diffs.len() > 8 {
            out += &format!("      ... and {} more\n", diffs.len() - 8);
        }
    }
    for name in oos {
        out += &format!("  out of scope: {name}\n");
    }
    out
}

#[test]
fn pipeline_cases() {
    let r = run_all(STORY_CEILING);
    let oos = out_of_scope();
    let text = summary(&r, &oos);
    println!("{text}");
    for name in &r.passed {
        println!("  passed {name}");
    }
    assert!(!r.passed.is_empty(), "no case ran");
    assert!(r.failed.is_empty(), "{text}");
}

/// AC4's planted change: at a story ceiling of 44 the same cases must fail, by name.
#[test]
fn planted_ceiling_44_turns_red() {
    let r = run_all(44);
    println!("{}", summary(&r, &[]));
    assert!(
        !r.failed.is_empty(),
        "a ceiling of 44 passed every case: the suite cannot see the ceiling"
    );
    let names: Vec<&str> = r.failed.iter().map(|(n, _)| n.as_str()).collect();
    for must in ["defaults", "chunks/ceiling-exact/story", "render/arm-join-every-mark"] {
        assert!(
            names.contains(&must),
            "{must} did not fail at ceiling 44; failed: {names:?}"
        );
    }
}

/// A no-op must read as no change: the suite run twice gives the same verdicts.
#[test]
fn suite_is_deterministic() {
    let (a, b) = (run_all(STORY_CEILING), run_all(STORY_CEILING));
    assert_eq!(a.passed, b.passed);
    assert_eq!(
        a.failed.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        b.failed.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
    );
}

/// The f32 judge entry point agrees with the f64 one on float32 input.
#[test]
fn judge_f32_matches_widened() {
    let y = signal::tone(1.0, 0.3, 0.5, 24_000);
    let wide: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
    assert_eq!(judge_f32(&y, 24_000, "今日好熱"), judge(&wide, 24_000, "今日好熱"));
}
