//! #351: the Qwen3-TTS sampler settings, frame cap and language are load options of the shared table. Each is unset by
//! default (the engine's fixed value), read from its flag and from a document, refused by name on an engine without
//! them (kitten-tts-2, fake), refused with its reason when it asks for greedy decoding, and recorded when given.

use ggtts_core::capabilities::Capabilities;
use ggtts_core::engine::EngineInfo;
use ggtts_core::options::{check, descriptor, record, Options, SAMPLER};

fn flags(list: &[&str]) -> Options {
    let args: Vec<String> = list.iter().map(|s| s.to_string()).collect();
    Options::from_args(&args).unwrap_or_else(|e| panic!("{list:?}: {e}"))
}

/// A sample value for a key's flag: the first language, else a number every key takes.
fn sample(key: &str) -> &'static str {
    if key == "load.language" {
        "en"
    } else {
        "2"
    }
}

#[test]
fn every_setting_is_a_load_row_unset_by_default() {
    let d = Options::default();
    assert!(
        d.load.sampling().is_empty(),
        "unset by default: {:?}",
        d.load.sampling()
    );
    for key in SAMPLER {
        let row = descriptor(key).unwrap_or_else(|| panic!("{key} is a row"));
        assert_eq!(row.phase.name(), "load", "{key}");
        assert_eq!(row.default, "null", "{key}: unset is the engine's fixed value");
        assert!(!row.flags.is_empty() && !row.help.is_empty(), "{key}");
    }
}

#[test]
fn each_setting_is_read_from_its_flag_and_from_a_document() {
    for key in SAMPLER {
        let row = descriptor(key).unwrap();
        let o = flags(&[row.flags[0], sample(key)]);
        let given = o.load.sampling();
        assert_eq!(given.len(), 1, "{key}: {given:?}");
        assert!(given.contains_key(row.name), "{key}: {given:?}");
        let value = if key == "load.language" { "\"en\"" } else { "2" };
        let doc = format!(r#"{{"load": {{"{}": {value}}}}}"#, row.name);
        let o = Options::from_json(&doc).unwrap_or_else(|e| panic!("{doc}: {e}"));
        assert!(o.load.sampling().contains_key(row.name), "{doc}");
    }
    // A language the reference CLI does not take is refused by the table.
    let args: Vec<String> = ["--language", "yue"].iter().map(|s| s.to_string()).collect();
    assert!(Options::from_args(&args)
        .unwrap_err()
        .to_string()
        .contains("load.language"));
}

/// AC3: an engine without them refuses each one given, naming its key.
#[test]
fn an_engine_without_them_refuses_each_by_its_key() {
    for engine in ["kitten-tts-2", "fake"] {
        let caps = Capabilities::of(engine).unwrap();
        assert!(!caps.sampler, "{engine}");
        for key in SAMPLER {
            let row = descriptor(key).unwrap();
            let r = check(&flags(&[row.flags[0], sample(key)]), &caps).unwrap_err();
            assert_eq!(r.key, key, "{engine}");
            assert!(r.message.contains(engine), "{engine} {key}: {}", r.message);
        }
    }
    // Qwen3-TTS takes every one of them.
    let qwen = Capabilities::of("qwen3-tts").unwrap();
    assert!(qwen.sampler);
    for key in SAMPLER {
        let row = descriptor(key).unwrap();
        check(&flags(&[row.flags[0], sample(key)]), &qwen).unwrap_or_else(|e| panic!("{key}: {e}"));
    }
}

/// AC4: greedy decoding is refused with the reason, talker and sub-talker alike, and so is a repetition penalty the
/// sampler would divide by zero with.
#[test]
fn greedy_decoding_is_refused_with_the_reason() {
    let qwen = Capabilities::of("qwen3-tts").unwrap();
    for (flag, key, says) in [
        (
            "--temperature",
            "load.temperature",
            "greedy decoding does not terminate",
        ),
        (
            "--subtalker-temperature",
            "load.subtalker_temperature",
            "greedy decoding of the sub-talker does not terminate",
        ),
        ("--repetition-penalty", "load.repetition_penalty", "divides by it"),
    ] {
        for value in ["0", "-0.5"] {
            let r = check(&flags(&[flag, value]), &qwen).unwrap_err();
            assert_eq!(r.key, key, "{flag} {value}");
            assert!(r.message.contains(says), "{flag} {value}: {}", r.message);
        }
    }
}

#[test]
fn the_record_carries_the_settings_given_and_only_those() {
    let info = EngineInfo {
        family: "qwen3-tts".into(),
        backend: "cpu".into(),
        parity: "exact".into(),
    };
    assert!(record(&Options::default(), &info).get("sampling").is_none());
    let o = flags(&["--temperature", "0.7", "--language", "en"]);
    let r = record(&o, &info);
    assert_eq!(r["sampling"], serde_json::json!({"temperature": 0.7, "language": "en"}));
    assert_eq!(
        ggtts_core::options::reported(&r).get("sampling"),
        Some(&r["sampling"]),
        "every report carries it"
    );
}
