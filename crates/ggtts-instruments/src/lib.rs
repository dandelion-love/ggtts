//! Instruments: what measures a render, run so that it can never break one.
//!
//! An *event instrument* is a separate program. After the render, the runner starts it once with the run's and the
//! render's identities and artifacts (the audio, and the event stream when the run has one), and reads back the
//! `ggtts-report/1` envelope it writes ([`ggtts_core::report`] owns that format, its validation and the merger).
//! Every instrument has its own log file, a timeout and a crash boundary: a crash, a hang, an invalid envelope or a
//! write to the run's audio is recorded as that instrument's status, and the render's output is left as it was.
//!
//! * [`config`]: which instruments run and which are required, from flags and an `instruments.yaml`, mapping 1:1,
//!   with the flags overriding the file. The file is validated against `docs/instruments/instruments.schema.json`.
//! * [`catalogue`]: the instruments this build ships, and the template config generated from them.
//! * [`runner`]: runs them, isolated, merges the report, logs what ran and what it cost, and decides whether the
//!   measurement is complete.
//! * [`options`]: the instrument run from the shared options value, the one call every front end makes.
//! * [`judge`]: the render's judge from the options value: the duration judge, or with an ASR reading each draw back.

pub mod catalogue;
pub mod config;
pub mod judge;
pub mod options;
pub mod runner;

pub use config::{Config, Entry, Flags, OnError};
pub use runner::{run, Inputs, RunSummary};
