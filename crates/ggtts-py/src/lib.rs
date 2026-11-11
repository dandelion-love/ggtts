//! ggtts for Python. The `python` feature builds the native module `ggtts._native`, which the pure-Python package in
//! `python/ggtts` wraps; [`core`] is the engine layer under it, in plain Rust.
//!
//! * **The settings are the C-ABI's** (`ggtts_load_with`): an engine, its model files, a Qwen3-TTS speaker and style
//!   instruction, and a device. A setting an engine does not take is refused, never ignored. With no model given,
//!   Qwen3-TTS loads its pinned default model from the per-user cache, downloading it first unless offline.
//! * **Errors are a code and a message:** `ggtts.GgttsError` with `code` (the C-ABI's codes) and `message`.
//! * **No panic crosses into Python.** Every call into the engine runs inside `catch_unwind`; a panic is a
//!   `GgttsError` with code `ERR_PANIC`, and the engine that panicked is dropped.
//! * **The GIL is released while an engine loads or renders**, so other Python threads run meanwhile.

pub mod core;

#[cfg(feature = "python")]
mod python;
