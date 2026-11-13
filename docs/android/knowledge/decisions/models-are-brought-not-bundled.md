---
type: Decision
title: No model ships; imports are checked
description: The APK and AAR carry no weights; the user supplies model files, and the demo checks each one on import.
tags: [android, models, ux]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:30:00-07:00 }
---
# Decision

- **No weights in the APK or AAR.** The build refuses an APK that carries a weights file, by name (`.gguf`,
  `.safetensors`, `.bin`, `.pt`, `.onnx`) or by content (any entry starting with GGUF's magic bytes), and checks
  both refusals on planted APKs.
- **The user brings the model.** Files go into the app's own directory, by `adb push` or by "Import a model file".
- **The import checks every file:**
  - a `.gguf` must start with the GGUF magic, and a `.json` must be a voice object;
  - empty files and short copies are refused;
  - an existing file is never replaced;
  - the default Qwen3-TTS files must match their published sha256;
  - the copy goes to `NAME.part` and is renamed only when complete, so a killed copy never appears in the lists.

# Why

The models are 2.3 to 2.5 GB, larger than an app store accepts in an APK, and each carries its own licence. A
2 GB copy can be cut short in many ways, and a wrong or partial file would otherwise surface as an engine error
far from its cause.

# Undo

The checks live in `android/demo/src/main/kotlin/ggtts/demo/ModelImport.kt`, and the APK check in
`scripts/android_build.sh`. Bundling a model would need a store that accepts the size, and a licence review.
