//! The fake engine.
//!
//! It will implement the core's engine interface with no model and no compute backend: a scripted clip length
//! per chunk, a synthetic waveform, codec tokens derived deterministically from the seed, and every instrument
//! event with its documented meaning. The pipeline's parity cases run against it, so the pipeline can be tested
//! before any real engine exists. In this skeleton it is empty.
