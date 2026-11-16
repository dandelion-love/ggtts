use ggtts_core::{
    gguf_meta,
    voices::{self, Declared},
};
use std::path::Path;
const USAGE: &str = "Usage: ggtts-gguf-voices INPUT OUTPUT --speaker ID [--language BCP47] [--name NAME]";
fn run(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let input = args.next().ok_or(USAGE)?;
    if input == "--help" || input == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let output = args.next().ok_or(USAGE)?;
    let mut speaker = None;
    let mut declared = Declared::default();
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let slot = match flag.as_str() {
            "--speaker" => &mut speaker,
            "--language" => &mut declared.language,
            "--name" => &mut declared.name,
            _ => return Err(format!("unknown option {flag}")),
        };
        if slot.replace(value).is_some() {
            return Err(format!("{flag} given more than once"));
        }
    }
    let speaker = speaker.ok_or("--speaker is required")?;
    if declared.is_empty() {
        return Err("provide --language or --name".into());
    }
    let set = voices::speaker_keys(&speaker, &declared).map_err(|e| e.to_string())?;
    if Path::new(&output).exists() {
        return Err(format!("output already exists: {output}"));
    }
    gguf_meta::rewrite(Path::new(&input), Path::new(&output), &set).map_err(|e| e.to_string())
}
fn main() {
    if let Err(e) = run(std::env::args().skip(1)) {
        eprintln!("ggtts-gguf-voices: {e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn fixture() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        fn string(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        string(&mut b, "ggtts.voice.a.name");
        b.extend_from_slice(&8u32.to_le_bytes());
        string(&mut b, "old");
        for (name, offset) in [("a", 0u64), ("b", 32u64)] {
            string(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&2u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend((0..40u8).map(|n| n.wrapping_mul(37)));
        b
    }
    #[test]
    fn rekeys_and_preserves_both_tensors() {
        let dir = std::env::temp_dir().join(format!("ggtts-rekey-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.gguf");
        let output = dir.join("output.gguf");
        let before = fixture();
        fs::write(&input, &before).unwrap();
        let _ = fs::remove_file(&output);
        run([
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--speaker",
            "a",
            "--language",
            "yue-HK",
            "--name",
            "New voice",
        ]
        .into_iter()
        .map(str::to_string))
        .unwrap();
        let after = fs::read(&output).unwrap();
        let old_header = gguf_meta::read_header(&before[..]).unwrap();
        let new_header = gguf_meta::read_header(&after[..]).unwrap();
        assert_eq!(old_header.tensor_infos, new_header.tensor_infos);
        for (offset, len) in [(0, 8), (32, 8)] {
            let old = old_header.data_start() as usize + offset;
            let new = new_header.data_start() as usize + offset;
            assert_eq!(&before[old..old + len], &after[new..new + len]);
        }
        let voice = &voices::read_gguf(&output).unwrap().speakers["a"];
        assert_eq!(voice.language.as_deref(), Some("yue-HK"));
        assert_eq!(voice.name.as_deref(), Some("New voice"));
        assert!(run([
            input.to_str().unwrap(),
            input.to_str().unwrap(),
            "--speaker",
            "a",
            "--name",
            "x"
        ]
        .into_iter()
        .map(str::to_string))
        .is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
