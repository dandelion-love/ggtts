# Voice declarations

A voice declaration is supplied by the model or its publisher. It is metadata, not a language detector or a claim about output quality. Missing fields stay absent; applications must not infer them from a filename or speaker ID.

All GGUF values below are strings. A Qwen3-TTS talker may declare each speaker:

- `ggtts.voice.<speaker>.language`: a BCP 47 language tag, such as `yue-HK`.
- `ggtts.voice.<speaker>.name`: a human-readable display name.

Speaker IDs can include periods. The reader recognizes the suffix from the end of the key; everything between the prefix and suffix is the speaker ID.

A Kitten-TTS 2 language-model GGUF declares its single voice with `ggtts.voice.language` and `ggtts.voice.name`. A prepared voice JSON uses `language` and `name` fields. The `ggtts_core::voices` module defines these keys and reads them for all front ends. It reports absent values as `None` and refuses malformed language tags, identifying the offending key. It checks tag syntax, not registration or whether a voice actually speaks that language.

Converter options and the GGUF re-key tool write only metadata the operator supplies; they do not infer provenance, language or a name. Keep the original model and write a new GGUF when adding talker keys. The re-key operation copies the tensor directory and available data unchanged. It does not validate the model architecture or certify that tensor data is complete; use it on a verified Qwen talker.

For a talker, create a separate output file:

```sh
ggtts-gguf-voices talker.gguf talker-labelled.gguf --speaker speaker_id --language yue-HK --name "Cantonese voice"
```

For Kitten, provide `--language TAG` and `--name NAME` to `ggtts-kitten-convert` when converting the language model, or to `ggtts-kitten prepare` when producing a voice JSON. Both fields are optional. The tools reject invalid declared values instead of writing a guess.
