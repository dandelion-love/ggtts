//! Which Qwen3-TTS checkpoints take a style instruction, decided as the reference decides it, and the error for one
//! that does not.
//!
//! The checkpoint type is qwen3-tts.cpp's `tts_model_type` at `16bb5af`: the `qwen3-tts.model_type` metadata
//! (`transformer_loader_metadata.cpp:235-238`, default `base`, lower-cased), then `custom_voice` for a `base`
//! checkpoint with named speakers (`transformer_loader_metadata.cpp:350-353`, still inside `parse_config`), and only
//! then, for a checkpoint still `base`, a file name containing `customvoice` or `voicedesign`
//! (`transformer_loader.cpp:200-210`, after `parse_config` returns). Whether it takes an instruction is
//! `get_model_capabilities` (`pipeline_models.cpp:471-479`): the `qwen3-tts.supports_instruction` flag when the file
//! has one, else `custom_voice` at a hidden size of 2048 or more (the 1.7B; the upstream Python package drops the
//! instruction for the 0.6B), `voice_design` always, and nothing else.
//!
//! The reference only reports that capability and renders an instruction on any checkpoint; this engine refuses one
//! where the capability is false, so an instruction is never silently dropped or applied to a model never trained on
//! one.

use ggtts_core::engine::EngineError;

/// The metadata keys of the checkpoint type, in the order the reference reads them.
pub const MODEL_TYPE_KEYS: [&str; 2] = ["qwen3-tts.model_type", "qwen3-tts.tts_model_type"];

/// The metadata keys of the explicit instruction flag, in the order the reference reads them.
pub const SUPPORTS_KEYS: [&str; 3] = [
    "qwen3-tts.supports_instruction",
    "qwen3-tts.instruction_supported",
    "qwen3-tts.instruct_supported",
];

/// The checkpoint type: `declared` is the first non-empty `model_type` string in the metadata, `file_name` the
/// talker's file name, `has_speakers` whether its metadata names any speaker.
pub fn model_type(declared: Option<&str>, file_name: &str, has_speakers: bool) -> String {
    let mut t = declared
        .filter(|s| !s.is_empty())
        .unwrap_or("base")
        .to_ascii_lowercase();
    if t == "base" && has_speakers {
        t = "custom_voice".into();
    }
    let name = file_name.to_ascii_lowercase();
    if t == "base" {
        if name.contains("customvoice") {
            t = "custom_voice".into();
        } else if name.contains("voicedesign") {
            t = "voice_design".into();
        }
    }
    t
}

/// Whether a checkpoint of `model_type` and `hidden_size` takes a style instruction; `declared` is the metadata's
/// explicit flag, which wins when present.
pub fn supports_instruction(model_type: &str, hidden_size: i32, declared: Option<bool>) -> bool {
    match declared {
        Some(d) => d,
        None => match model_type {
            "custom_voice" => hidden_size >= 2048,
            "voice_design" => true,
            _ => false,
        },
    }
}

/// `Ok` for no instruction (`None` or empty) or a checkpoint that takes one; otherwise the error a user sees, naming
/// the checkpoint's type and the types that take an instruction.
pub fn check(instruction: Option<&str>, model_type: &str, hidden_size: i32, supports: bool) -> Result<(), EngineError> {
    match instruction {
        Some(i) if !i.is_empty() && !supports => Err(EngineError(format!(
            "this checkpoint is a Qwen3-TTS {model_type} model (hidden size {hidden_size}), which takes no style \
             instruction; instructions are accepted by custom_voice checkpoints of 1.7B (hidden size 2048) and by \
             voice_design checkpoints"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_type_is_the_metadata_then_the_file_name_then_the_speakers() {
        assert_eq!(model_type(Some("custom_voice"), "x.gguf", false), "custom_voice");
        assert_eq!(model_type(Some("Voice_Design"), "x.gguf", false), "voice_design");
        assert_eq!(model_type(None, "x.gguf", false), "base");
        assert_eq!(model_type(Some(""), "x.gguf", false), "base");
        assert_eq!(
            model_type(Some("base"), "Qwen3-TTS-CustomVoice.gguf", false),
            "custom_voice"
        );
        assert_eq!(model_type(None, "talker-voicedesign-q8_0.gguf", false), "voice_design");
        assert_eq!(model_type(Some("base"), "x.gguf", true), "custom_voice");
        // The speakers decide before the file name, as in the reference: a base checkpoint with speakers is
        // custom_voice whatever its name says.
        assert_eq!(
            model_type(Some("base"), "x-voicedesign-q8_0.gguf", true),
            "custom_voice"
        );
        // A declared type other than base is never overridden.
        assert_eq!(
            model_type(Some("voice_design"), "customvoice.gguf", true),
            "voice_design"
        );
    }

    #[test]
    fn only_the_large_custom_voice_and_voice_design_take_instructions() {
        assert!(supports_instruction("custom_voice", 2048, None));
        assert!(!supports_instruction("custom_voice", 1024, None));
        assert!(supports_instruction("voice_design", 1024, None));
        assert!(!supports_instruction("base", 2048, None));
        assert!(supports_instruction("base", 2048, Some(true)));
        assert!(!supports_instruction("custom_voice", 2048, Some(false)));
    }

    /// Declared base, with speakers, 0.6B (hidden size 1024), named like a VoiceDesign file: the reference makes it
    /// custom_voice from its speakers before it reads the name, and a 0.6B custom_voice takes no instruction.
    #[test]
    fn speakers_before_the_file_name_decide_a_small_checkpoint_takes_no_instruction() {
        let t = model_type(Some("base"), "qwen-talker-0.6b-voicedesign-q8_0.gguf", true);
        assert_eq!(t, "custom_voice");
        assert!(!supports_instruction(&t, 1024, None));
        assert!(check(Some("Speak slowly."), &t, 1024, false).is_err());
    }

    #[test]
    fn a_base_checkpoint_refuses_an_instruction_by_name() {
        let e = check(Some("speak slowly"), "base", 2048, false).unwrap_err();
        assert!(e.0.contains("base model"), "{}", e.0);
        assert!(e.0.contains("custom_voice") && e.0.contains("voice_design"), "{}", e.0);
        // No instruction, or an empty one, is never refused.
        assert!(check(None, "base", 2048, false).is_ok());
        assert!(check(Some(""), "base", 2048, false).is_ok());
        assert!(check(Some("speak slowly"), "custom_voice", 2048, true).is_ok());
    }
}
