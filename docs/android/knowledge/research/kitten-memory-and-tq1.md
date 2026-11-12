---
type: Research Finding
title: Kitten-TTS 2's memory, and the released TQ1_0 model on a phone
description: On a Galaxy Z Fold5, the released Cantonese TQ1_0 LM peaks 195 MB lower than the base Q4_0 LM but generates 24% and 35% slower in two pairs; Kitten-TTS 2's 3.78 GB peak is a load-time spike, about 0.9 GB above what a render holds, from whole-file reads and per-tensor copies.
tags: [android, kitten-tts-2, memory, tq1_0, measurement]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T12:05:00-07:00 }
sources:
  - { id: record, resource: "/research/on-device-parity-and-budget.md", title: "On-device parity and resource budget" }
  - { id: scripts, resource: "https://github.com/dandelion-love/ggtts/tree/track/android/scripts/android_on_device", title: "The phone scripts: run-mem.sh, tq1-measure.sh" }
---
# What was compared

Kitten-TTS 2 on a Galaxy Z Fold5 (Android 16, Snapdragon 8 Gen 2), CPU only, through the Android `ggtts-render`
built at `be554be` (the track with rc v0.1.0-rc.11 merged in; earlier track builds cannot load TQ1_0). Two language
models, everything else the same:[^scripts]

| | Base LM, Q4_0 | Released Cantonese QAT LM, TQ1_0 (padded to 128) |
|---|---|---|
| File | `kitten-lm.gguf`, sha256 `263b8e69…`, 1.458 GB | `kitten-cantonese-qat-tq1_0.gguf`, sha256 `11555aa3…`, 1.260 GB |
| S3Gen decoder | `kitten-s3gen.gguf`, `54ad2c36…`, 1.082 GB | the same file |
| Voice | the released Cantonese `voice.json`, `d7b0aec8…` | the same file |

The text was a two-sentence Cantonese story (`tq1-measure.sh`), at the default seed and 4 threads, with `--stream-raw`
so first audio is the first chunk. The runs alternated Q4_0, TQ1_0, Q4_0, TQ1_0 with a minute between them, so warming
reaches both. `run-mem.sh` sampled the process's memory every 0.25 s. The raw files (each run's log, timing report, peak and
memory timeline) are in `docs/android/data/kitten-tq1-2026-11-12/`.

# Results

| Run (order) | Audio | Frames | First audio | Render | RTF | LM | Decoder | Peak (VmHWM) | Most during the render |
|---|---|---|---|---|---|---|---|---|---|
| Q4_0 (1) | 9.44 s | 224 | 37.3 s | 87.4 s | 9.3 | 62.8 s, 3.56 frames/s | 24.6 s | 3.78 GB | 2.90 GB |
| TQ1_0 (2) | 9.86 s | 235 | 50.1 s | 114.8 s | 11.7 | 87.1 s, 2.70 frames/s | 27.8 s | 3.59 GB | 2.72 GB |
| Q4_0 (3) | 9.44 s | 224 | 39.1 s | 106.4 s | 11.3 | 78.2 s, 2.87 frames/s | 28.2 s | 3.78 GB | 2.91 GB |
| TQ1_0 (4) | 9.86 s | 235 | 74.0 s | 156.8 s | 15.9 | 125.6 s, 1.87 frames/s | 31.1 s | 3.59 GB | 2.72 GB |

GB here are 10^9 bytes; `/proc` reports kB that are KiB (Q4_0's peak was 3,693,596 KiB). The two models generate
different speech (224 against 235 frames for the same text), so speed is compared per frame.

- **Memory: TQ1_0 saves 195 MB at the peak and about 200 MB during a render.** That is the difference between the two
  LMs' layer matrices (0.793 GB as Q4_0, 0.595 GB as TQ1_0), and nothing else. It does not bring Kitten-TTS 2 within the
  budget's 3.0 GB.[^record]
- **Speed: TQ1_0 generates more slowly.** Its LM ran at 2.70 against 3.56 frames per second in the first pair and 1.87
  against 2.87 in the second: 24% and 35% slower. The phone warmed through the session (each model's second run was
  slower than its first), so the absolute numbers drift. Within each pair TQ1_0 was slower, and the first TQ1_0 run was
  slower than the Q4_0 run that followed it, so the gap is not only warming. The decoder, which is the same in both, took about the same time.
  So TQ1_0's first audio and RTF are worse, not better. Two runs a side; the keep-awake session also drove the phone
  during them.
- **The audio is plausible on both** (0 first-draw defects, 2 chunks each); nobody listened for quality here.

# Where the 3.78 GB goes

The memory timeline puts the peak in the 5.7 s load, not the render. Afterwards the process holds 2.70 GB (Q4_0) or
2.50 GB (TQ1_0), and at most 2.90 or 2.72 GB while rendering. The load has two spikes, and the code explains both
(`crates/ggtts-kitten/src/engine.rs`, `KittenEngine::open`; `model.rs`, `Model::load`):

1. **The LM's load, 3.61 GB (Q4_0) at about 4.2 s.** The LM GGUF is read whole into memory. Each tensor is then copied
   once more (`data.to_vec()`, queued in `pending`) before it is written into the CPU backend's buffer, and the token
   embedding is copied a third time into the host table. The file bytes are dropped when the LM is loaded.
2. **The decoder's load, 3.78 GB (Q4_0) at about 5.6 s, the peak.** With the LM resident, the S3Gen GGUF is read whole
   (1.08 GB), and the flow and HiFT weights are copied out of it. The file bytes stay until the load returns.

At the peak, these are resident at once:

| Part | Q4_0 | TQ1_0 | What it is |
|---|---|---|---|
| LM layer matrices (ternary) | 793 MB | 595 MB | 196 matrices, Q4_0 or TQ1_0, in the backend buffer |
| LM token embedding, backend copy | 648 MB | 648 MB | BF16, 158,241 by 2048; also the output head (tied) |
| LM token embedding, host copy | 648 MB | 648 MB | the same table again (`Model::embed`), for the input lookup |
| LM dense tensors | 5 MB | 5 MB | norms and the speaker projection, F32 |
| S3Gen file bytes, held during the load | 1,082 MB | 1,082 MB | the whole decoder GGUF, read to copy from |
| S3Gen flow weights | 459 MB | 459 MB | F32, copied into the backend |
| S3Gen HiFT (mel2wav) weights | 83 MB | 83 MB | F32, copied into the backend |
| **Sum** | **3,718 MB** | **3,520 MB** | measured peak: 3,782 MB and 3,587 MB |

The sum falls about 65 MB short of each measured peak, which is the binary, its libraries and ggml's own buffers.
Predicted before the measurement: a peak 198 MB lower for TQ1_0; measured, 195 MB lower.

After the load, the S3Gen bytes are freed, which leaves about 2.64 GB of weights (2.44 GB with TQ1_0). The render adds
up to 0.2 GB more: the KV cache, compute buffers and the audio being kept.

Half of the S3Gen file is never used to render: the speech tokenizer's encoder (495 MB) and the speaker encoder (28 MB)
are needed only to prepare a voice, and the render path does not load them.

# What would lower it

In the order of what each saves, each measurable with `run-mem.sh`:

- **Load each tensor straight from the file into the backend**, in both loaders: map the file, or read one tensor at a
  time, instead of reading the file whole and then copying each tensor once more. Both spikes go, and the peak becomes
  the render's own 2.9 GB (Q4_0) or 2.7 GB (TQ1_0), inside the 3.0 GB budget. Fixing only the decoder's load would
  leave the LM's 3.6 GB spike.
- **Drop the host copy of the token embedding**, reading input rows from the backend's copy (on the CPU backend it is in
  host memory): 648 MB off both the peak and the render.
- Both together: a render held in about 2.1 to 2.3 GB.

TQ1_0 then saves another 0.2 GB, at its speed cost on this CPU.

# Units in earlier figures

The phone figures recorded before this one read `VmHWM`'s KiB as if they were 1000 bytes: "3.69 GB" for Kitten-TTS 2 is
3,693,672 KiB, which is 3.78 GB (3.52 GiB). Qwen3-TTS's "2.90 to 2.95 GB" is about 2.97 to 3.02 GB. The record, the
bundle and the demo's cost line (`EngineCosts`) are corrected with this finding.
