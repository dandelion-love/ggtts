//! The Kitten-TTS 2 engine: a ternary (BitNet-style) Qwen3 speech language model on ggml.
//!
//! See `DESIGN.md` for what is ported from where, the ternary storage and why, and the parity levels.
//!
//! * [`convert`]: Kitten's language model, or a QAT export in its key layout, to a ggtts GGUF, with the readback
//!   check that every ternary weight survives exactly.
//! * [`ternary`]: Kitten's ternary rule and its `TQ2_0` storage on zero-padded 128-weight groups.

pub mod convert;
pub mod gguf;
pub mod safetensors;
pub mod ternary;
