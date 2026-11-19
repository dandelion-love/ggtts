//! What a model file declares about its voices: each voice's language (a BCP 47 tag, such as `yue-HK`) and a name to
//! show a person. This module names the keys, and it is the one reader every front end uses, so none of them parses
//! the keys itself. `docs/voice-keys.md` is the same specification in prose.
//!
//! - A Qwen3-TTS talker GGUF declares each speaker: `ggtts.voice.<speaker>.language` and `ggtts.voice.<speaker>.name`.
//!   A speaker name may contain `.`: the reader takes the suffix (`.language` or `.name`) from the end, and what lies
//!   between `ggtts.voice.` and it is the speaker, dots and all.
//! - A Kitten-TTS 2 language-model GGUF declares its one voice: `ggtts.voice.language` and `ggtts.voice.name`.
//! - A prepared voice JSON declares its voice in the fields `language` and `name`.
//!
//! A key that is absent reads as `None`, never a guess. A language key that does not hold a well-formed tag is refused,
//! with the key named, rather than passed on.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// The prefix of every voice key in a GGUF.
pub const PREFIX: &str = "ggtts.voice.";
/// A Kitten-TTS 2 language model's voice language.
pub const FILE_LANGUAGE: &str = "ggtts.voice.language";
/// A Kitten-TTS 2 language model's voice name.
pub const FILE_NAME: &str = "ggtts.voice.name";
/// The suffix of a Qwen3-TTS speaker's language key: `ggtts.voice.<speaker>.language`.
pub const LANGUAGE_SUFFIX: &str = ".language";
/// The suffix of a Qwen3-TTS speaker's name key: `ggtts.voice.<speaker>.name`.
pub const NAME_SUFFIX: &str = ".name";
/// A prepared voice JSON's language field.
pub const JSON_LANGUAGE: &str = "language";
/// A prepared voice JSON's name field.
pub const JSON_NAME: &str = "name";

/// A Qwen3-TTS speaker's language key.
pub fn speaker_language_key(speaker: &str) -> String {
    format!("{PREFIX}{speaker}{LANGUAGE_SUFFIX}")
}

/// A Qwen3-TTS speaker's name key.
pub fn speaker_name_key(speaker: &str) -> String {
    format!("{PREFIX}{speaker}{NAME_SUFFIX}")
}

/// What one voice declares. Either may be absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Declared {
    pub language: Option<String>,
    pub name: Option<String>,
}

impl Declared {
    pub fn is_empty(&self) -> bool {
        self.language.is_none() && self.name.is_none()
    }
}

/// What a file declares: its one voice (a Kitten-TTS 2 language model, a prepared voice) and its speakers (a Qwen3-TTS
/// talker), by speaker name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileVoices {
    pub file: Declared,
    pub speakers: BTreeMap<String, Declared>,
}

/// A voice key the reader refuses, with the key and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub key: String,
    pub why: String,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.why)
    }
}

impl std::error::Error for Refused {}

/// Check BCP 47 syntax; registration of individual subtags is not checked.
pub fn check_language(tag: &str) -> Result<(), String> {
    language_tags::LanguageTag::parse(tag)
        .map(|_| ())
        .map_err(|e| format!("{tag:?} is not a well-formed BCP 47 language tag: {e}"))
}

fn language(key: &str, value: &str) -> Result<String, Refused> {
    check_language(value).map_err(|why| Refused {
        key: key.to_string(),
        why,
    })?;
    Ok(value.to_string())
}

fn name(key: &str, value: &str) -> Result<String, Refused> {
    if value.trim().is_empty() {
        return Err(Refused {
            key: key.to_string(),
            why: "the name is empty".into(),
        });
    }
    Ok(value.to_string())
}

/// The voices declared by a GGUF's string keys (key to value; other keys are ignored).
pub fn from_keys(kv: &BTreeMap<String, String>) -> Result<FileVoices, Refused> {
    let mut out = FileVoices::default();
    for (k, v) in kv.range(PREFIX.to_string()..) {
        let Some(rest) = k.strip_prefix(PREFIX) else { break };
        if k == FILE_LANGUAGE {
            out.file.language = Some(language(k, v)?);
        } else if k == FILE_NAME {
            out.file.name = Some(name(k, v)?);
        } else if let Some(speaker) = rest.strip_suffix(LANGUAGE_SUFFIX).filter(|s| !s.is_empty()) {
            out.speakers.entry(speaker.to_string()).or_default().language = Some(language(k, v)?);
        } else if let Some(speaker) = rest.strip_suffix(NAME_SUFFIX).filter(|s| !s.is_empty()) {
            out.speakers.entry(speaker.to_string()).or_default().name = Some(name(k, v)?);
        }
    }
    Ok(out)
}

/// The error of [`read_gguf`] and [`read_voice_json`]: the file could not be read, or a key was refused.
#[derive(Debug)]
pub enum ReadError {
    Io(std::io::Error),
    Refused(Refused),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "{e}"),
            ReadError::Refused(r) => write!(f, "{r}"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::Io(e)
    }
}

impl From<Refused> for ReadError {
    fn from(r: Refused) -> Self {
        ReadError::Refused(r)
    }
}

/// The voices a GGUF declares, read from its header only (no tensor is read).
pub fn read_gguf(path: &Path) -> Result<FileVoices, ReadError> {
    let header = crate::gguf_meta::header(path)?;
    for entry in &header.kv {
        let Some(rest) = entry.key.strip_prefix(PREFIX) else {
            continue;
        };
        let known = entry.key == FILE_LANGUAGE
            || entry.key == FILE_NAME
            || rest.strip_suffix(LANGUAGE_SUFFIX).is_some_and(|s| !s.is_empty())
            || rest.strip_suffix(NAME_SUFFIX).is_some_and(|s| !s.is_empty());
        if known && entry.as_str().is_none() {
            return Err(Refused {
                key: entry.key.clone(),
                why: "is not a UTF-8 string".into(),
            }
            .into());
        }
    }
    Ok(from_keys(&header.strings(PREFIX))?)
}

/// The voice a prepared voice JSON declares, in [`FileVoices::file`]. A field that is not a string is refused.
pub fn from_voice_json(doc: &serde_json::Value) -> Result<FileVoices, Refused> {
    if !doc.is_object() {
        return Err(Refused {
            key: "voice JSON".into(),
            why: "is not an object".into(),
        });
    }
    let field = |f: &str| -> Result<Option<String>, Refused> {
        match doc.get(f) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(Refused {
                key: f.to_string(),
                why: "is not a string".into(),
            }),
        }
    };
    let mut out = FileVoices::default();
    if let Some(v) = field(JSON_LANGUAGE)? {
        out.file.language = Some(language(JSON_LANGUAGE, &v)?);
    }
    if let Some(v) = field(JSON_NAME)? {
        out.file.name = Some(name(JSON_NAME, &v)?);
    }
    Ok(out)
}

/// [`from_voice_json`] of the file at `path`.
pub fn read_voice_json(path: &Path) -> Result<FileVoices, ReadError> {
    let mut text = String::new();
    std::io::Read::read_to_string(&mut crate::fdpath::open(path)?, &mut text)?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))?;
    Ok(from_voice_json(&doc)?)
}

/// Identify the format by content. Descriptor duplicates share offsets; concurrent use is unsupported.
pub fn read(path: &Path) -> Result<FileVoices, ReadError> {
    use std::io::Read;
    let mut file = crate::fdpath::open(path)?;
    let mut magic = [0u8; 4];
    let mut count = 0;
    while count < magic.len() {
        match file.read(&mut magic[count..])? {
            0 => break,
            n => count += n,
        }
    }
    drop(file);
    // Each reader opens at byte zero, including the duplicate of a descriptor.
    if count == 4 && &magic == b"GGUF" {
        read_gguf(path)
    } else {
        read_voice_json(path)
    }
}

/// The keys that declare `speaker`'s language and name in a talker GGUF, for [`crate::gguf_meta::rewrite`]. Each
/// value is checked as the reader would check it, so a file written with them reads back.
pub fn speaker_keys(speaker: &str, declared: &Declared) -> Result<BTreeMap<String, String>, Refused> {
    if speaker.is_empty() {
        return Err(Refused {
            key: PREFIX.into(),
            why: "the speaker name is empty".into(),
        });
    }
    let mut out = BTreeMap::new();
    if let Some(l) = &declared.language {
        let k = speaker_language_key(speaker);
        out.insert(k.clone(), language(&k, l)?);
    }
    if let Some(n) = &declared.name {
        let k = speaker_name_key(speaker);
        out.insert(k.clone(), name(&k, n)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn a_talker_declares_each_speaker_and_a_dotted_speaker_reads_from_the_end() {
        let v = from_keys(&kv(&[
            ("ggtts.voice.hk_voice_343.language", "yue-HK"),
            ("ggtts.voice.hk_voice_343.name", "Voice 343"),
            ("ggtts.voice.v1.2.language", "en-US"),
            ("general.architecture", "qwen3-tts"),
        ]))
        .unwrap();
        assert!(v.file.is_empty());
        assert_eq!(
            v.speakers["hk_voice_343"],
            Declared {
                language: Some("yue-HK".into()),
                name: Some("Voice 343".into())
            }
        );
        assert_eq!(
            v.speakers["v1.2"],
            Declared {
                language: Some("en-US".into()),
                name: None
            }
        );
        assert_eq!(speaker_language_key("v1.2"), "ggtts.voice.v1.2.language");
    }

    #[test]
    fn a_language_model_declares_its_one_voice() {
        let v = from_keys(&kv(&[(FILE_LANGUAGE, "en-US"), (FILE_NAME, "Kitten")])).unwrap();
        assert_eq!(
            v.file,
            Declared {
                language: Some("en-US".into()),
                name: Some("Kitten".into())
            }
        );
        assert!(v.speakers.is_empty());
    }

    /// AC4: no keys read as none, never a guess.
    #[test]
    fn a_file_without_the_keys_declares_nothing() {
        let v = from_keys(&kv(&[
            ("general.architecture", "qwen3-tts"),
            ("tokenizer.ggml.model", "gpt2"),
        ]))
        .unwrap();
        assert_eq!(v, FileVoices::default());
        assert_eq!(
            from_voice_json(&serde_json::json!({"transcript": "x"})).unwrap(),
            FileVoices::default()
        );
    }

    /// AC4: a malformed tag is refused with its key named (planted).
    #[test]
    fn a_malformed_language_is_refused_by_its_key() {
        for bad in [
            "",
            "Cantonese",
            "yue_HK",
            "yue-",
            "y",
            "yue-HKHKHKHKHK",
            "en-a",
            "en-1",
            "粵-HK",
        ] {
            let e = from_keys(&kv(&[("ggtts.voice.a.language", bad)])).unwrap_err();
            assert_eq!(e.key, "ggtts.voice.a.language", "{bad:?}");
        }
        let e = from_keys(&kv(&[(FILE_LANGUAGE, "en_US")])).unwrap_err();
        assert_eq!(e.key, FILE_LANGUAGE, "{e}");
        let e = from_voice_json(&serde_json::json!({"language": "en US"})).unwrap_err();
        assert_eq!(e.key, "language");
        let e = from_voice_json(&serde_json::json!({"name": 3})).unwrap_err();
        assert_eq!(e.key, "name");
        let e = from_keys(&kv(&[("ggtts.voice.a.name", "  ")])).unwrap_err();
        assert_eq!(e.key, "ggtts.voice.a.name");
        for good in ["yue-HK", "en-US", "zh-Hant-HK", "en", "yue", "cmn-Hans-CN"] {
            check_language(good).unwrap();
        }
    }

    #[test]
    fn a_non_string_voice_key_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("ggtts-voice-type-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = "ggtts.voice.a.language";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&42u32.to_le_bytes());
        std::fs::write(&path, b).unwrap();
        let error = read_gguf(&path).unwrap_err();
        assert!(error.to_string().contains(key));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_prepared_voice_declares_its_voice() {
        let v =
            from_voice_json(&serde_json::json!({"language": "en-GB", "name": "Reader", "transcript": "x"})).unwrap();
        assert_eq!(
            v.file,
            Declared {
                language: Some("en-GB".into()),
                name: Some("Reader".into())
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_formats_and_rejections() {
        use std::io::Read;
        use std::os::fd::AsRawFd;
        use std::process::{Command, Stdio};
        let dir = std::env::temp_dir().join(format!("ggtts-voice-fd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cases: Vec<(&str, Vec<u8>, Option<Declared>)> = vec![
            (
                "voice",
                br#" {"language":"en-GB","name":"Reader"}"#.to_vec(),
                Some(Declared {
                    language: Some("en-GB".into()),
                    name: Some("Reader".into()),
                }),
            ),
            ("empty", b"{}".to_vec(), Some(Declared::default())),
            (
                "model",
                crate::gguf_meta::tests::tiny(&[(FILE_LANGUAGE, "yue-HK"), (FILE_NAME, "Model")], &[7; 64]),
                Some(Declared {
                    language: Some("yue-HK".into()),
                    name: Some("Model".into()),
                }),
            ),
            ("bad-language", br#"{"language":"en US"}"#.to_vec(), None),
            ("array", b"[1,2,3]".to_vec(), None),
            ("junk", b"not a model".to_vec(), None),
        ];
        for (name, bytes, expected) in cases {
            let path = dir.join(name);
            std::fs::write(&path, bytes).unwrap();
            let held = std::fs::File::open(&path).unwrap();
            let fdpath = std::path::PathBuf::from(format!("/proc/self/fd/{}", held.as_raw_fd()));
            let result = read(&fdpath);
            if let Some(expected) = expected {
                assert_eq!(result.unwrap().file, expected, "{name}");
                assert_eq!(read(&path).unwrap().file, expected, "ordinary {name}");
            } else {
                let error = result.unwrap_err().to_string();
                if name == "bad-language" {
                    assert!(error.contains("language"), "{error}");
                }
            }
            assert!(held.metadata().is_ok());
        }
        let mut child = Command::new("printf")
            .arg("sentinel")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut reader = child.stdout.take().unwrap();
        let fdpath = std::path::PathBuf::from(format!("/proc/self/fd/{}", reader.as_raw_fd()));
        assert!(read(&fdpath).is_err());
        let mut bytes = [0; 8];
        reader.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"sentinel");
        assert!(child.wait().unwrap().success());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_rekeyed_gguf_reads_back_through_the_reader() {
        let data = [7u8; 64];
        let dir = std::env::temp_dir().join(format!("ggtts-voices-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("talker.gguf");
        std::fs::write(
            &input,
            crate::gguf_meta::tests::tiny(&[("general.architecture", "qwen3-tts")], &data),
        )
        .unwrap();
        assert_eq!(read_gguf(&input).unwrap(), FileVoices::default());
        let declared = Declared {
            language: Some("yue-HK".into()),
            name: Some("Voice A".into()),
        };
        let set = speaker_keys("a.b", &declared).unwrap();
        let output = dir.join("out.gguf");
        crate::gguf_meta::rewrite(&input, &output, &set).unwrap();
        assert_eq!(read(&output).unwrap().speakers["a.b"], declared);
        assert!(speaker_keys(
            "a",
            &Declared {
                language: Some("nope nope".into()),
                name: None
            }
        )
        .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
