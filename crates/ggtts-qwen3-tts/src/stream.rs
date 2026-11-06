//! Frame-level streaming for the codec decoder: samples out every `N` frames while frames are still arriving.
//!
//! The decoder is causal end to end (see [`codec`](crate::codec)): every conv pads on the left only, the
//! pre-transformer's mask is causal with a 72-frame window, every transposed conv trims its overhang from the right,
//! and the norm, snake and pointwise layers act per time step. So decoding the prefix `codes[0..e)` gives, in exact
//! arithmetic, the full decode's samples for every frame before `e`. [`IncrementalDecoder`] re-decodes the growing
//! prefix every `N` frames and emits the samples of the frames it has not emitted, up to `e - hold`.
//!
//! Why the whole prefix rather than the last 72 frames: the window bounds one attention layer, and the 8 layers and the
//! convs stack to a receptive field of about 45 s, longer than any story chunk. A shorter context would change
//! the samples.
//!
//! Why a hold-back: in floating point the prefix decode is not guaranteed identical for its last frames, because
//! ggml's softmax and key-value dot products reduce over `e` keys instead of the full length, in aligned blocks.
//! Frames at least one block (32) behind the decode end reduce identically. Measured on the 8 CPU fixtures
//! (`tests/stream_identity.rs`): at `hold = 32` the stream is bit-identical to the batch decode at every `every`
//! tried; at `hold = 0` the newest frames of a prefix differ by up to 458 LSB at 16 bits, so `hold = 0` is earlier
//! but not exact. The last call, [`finish`](IncrementalDecoder::finish), decodes the whole utterance, so the tail is
//! always exact.
//!
//! The cost: every step re-decodes the prefix. At `every = 12, hold = 32` the decoder did 3 to 7 times the batch
//! decode's work on CPU over the fixtures.

use ggtts_core::engine::EngineError;

use crate::codec::Decoder;

/// Frames from the decode end that are held back by default: one 32-wide reduction block.
pub const DEFAULT_HOLD: usize = 32;
/// Frames between decodes by default: about a second at 12.5 frames per second.
pub const DEFAULT_EVERY: usize = 12;

/// What each step decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Context {
    /// The whole prefix: exact.
    #[default]
    Prefix,
    /// A deliberate defect, for showing that the identity check can fail: only the frames not yet emitted, with no
    /// left context, so every seam loses the overlap the convs and the attention need. Only with the test-only
    /// `plant` feature.
    #[cfg(feature = "plant")]
    NewFramesOnly,
}

/// Decodes a growing utterance every `every` frames and hands back the samples ready so far.
pub struct IncrementalDecoder<'d> {
    dec: &'d mut Decoder,
    every: usize,
    hold: usize,
    context: Context,
    codebooks: usize,
    codes: Vec<i32>,
    since: usize,
    emitted: usize,
    samples_per_frame: Option<usize>,
    /// Prefix decodes run so far.
    pub decodes: usize,
    /// Frames decoded in total, over every prefix decode: the cost against the batch decode's `n`.
    pub frames_decoded: usize,
}

impl<'d> IncrementalDecoder<'d> {
    pub fn new(dec: &'d mut Decoder, every: usize, hold: usize) -> Self {
        Self::with_context(dec, every, hold, Context::Prefix)
    }

    pub fn with_context(dec: &'d mut Decoder, every: usize, hold: usize, context: Context) -> Self {
        let codebooks = dec.config.n_codebooks as usize;
        Self {
            dec,
            every: every.max(1),
            hold,
            context,
            codebooks,
            codes: Vec::new(),
            since: 0,
            emitted: 0,
            samples_per_frame: None,
            decodes: 0,
            frames_decoded: 0,
        }
    }

    /// Frames received so far.
    pub fn frames(&self) -> usize {
        self.codes.len() / self.codebooks
    }

    /// Frames whose samples have been handed back.
    pub fn emitted_frames(&self) -> usize {
        self.emitted
    }

    /// Add one frame (`codebooks` codes). Every `every` frames this decodes and returns the newly ready samples,
    /// which may be empty while fewer than `hold` frames have arrived.
    pub fn push(&mut self, frame: &[i32]) -> Result<Option<Vec<f32>>, EngineError> {
        if frame.len() != self.codebooks {
            return Err(EngineError(format!(
                "a frame has {} codes, not {}",
                frame.len(),
                self.codebooks
            )));
        }
        self.codes.extend_from_slice(frame);
        self.since += 1;
        if self.since < self.every {
            return Ok(None);
        }
        self.since = 0;
        let upto = self.frames().saturating_sub(self.hold);
        self.emit(upto).map(Some)
    }

    /// Decode the whole utterance and return every sample not yet handed back. The concatenation of every
    /// [`push`](Self::push) result and this one is the stream.
    pub fn finish(&mut self) -> Result<Vec<f32>, EngineError> {
        let n = self.frames();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.emit(n)
    }

    /// Decode and return the samples of frames `[emitted, upto)`.
    fn emit(&mut self, upto: usize) -> Result<Vec<f32>, EngineError> {
        if upto <= self.emitted {
            return Ok(Vec::new());
        }
        let n = self.frames();
        let first = match self.context {
            Context::Prefix => 0,
            #[cfg(feature = "plant")]
            Context::NewFramesOnly => self.emitted,
        };
        let pcm = self
            .dec
            .decode(&self.codes[first * self.codebooks..n * self.codebooks])?;
        self.decodes += 1;
        self.frames_decoded += n - first;
        let decoded = n - first;
        if pcm.samples.len() % decoded != 0 {
            return Err(EngineError(format!(
                "{} samples for {decoded} frames is not a whole number per frame",
                pcm.samples.len()
            )));
        }
        let spf = pcm.samples.len() / decoded;
        if *self.samples_per_frame.get_or_insert(spf) != spf {
            return Err(EngineError("the samples per frame changed between decodes".into()));
        }
        let out = pcm.samples[(self.emitted - first) * spf..(upto - first) * spf].to_vec();
        self.emitted = upto;
        Ok(out)
    }
}

/// Stream `codes` (frame-major) through an [`IncrementalDecoder`] and return the concatenated stream, plus the
/// number of samples handed back at each step.
pub fn stream_codes(
    dec: &mut Decoder,
    codes: &[i32],
    every: usize,
    hold: usize,
    context: Context,
) -> Result<(Vec<f32>, Vec<usize>), EngineError> {
    let nc = dec.config.n_codebooks as usize;
    let mut inc = IncrementalDecoder::with_context(dec, every, hold, context);
    let mut out = Vec::new();
    let mut steps = Vec::new();
    for frame in codes.chunks_exact(nc) {
        if let Some(s) = inc.push(frame)? {
            steps.push(s.len());
            out.extend(s);
        }
    }
    let tail = inc.finish()?;
    steps.push(tail.len());
    out.extend(tail);
    Ok((out, steps))
}
