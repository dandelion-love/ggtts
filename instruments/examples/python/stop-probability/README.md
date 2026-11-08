# Per-step stop probability

This instrument writes `stop-probability.jsonl` beside its envelope, one row per Qwen draw step. Each row contains
the chunk, draw and step indices, sampled codebook-0 token, `p_stop`, draw stop reason, and whether the step is the
terminal EOS draw.
The envelope records the trace hash and declares `qwen-sampler-weights/1`; the artifact contains generated per-step
data and should be kept outside version control. It contains no input text.
Each JSON line follows [the artifact schema](../../../../schemas/ggtts-stop-probability-1.schema.json).

The instrument requires a complete Qwen trace with a declared `p_stop_semantics` and the producer's codec EOS token.
It checks contiguous step indices, draw boundaries, probability range and that an EOS-ended draw's last token is
the declared EOS token. A capped draw has no terminal EOS; an aborted draw, missing or inconsistent required step
data, an unmarked trace, or an unsupported producer is skipped or truncated, so a required result is incomplete.
The shared trace reader stops at the first `end`; this instrument is not a full trace-schema validator. Existing
unmarked Qwen traces are not silently
reinterpreted. The fake engine and Kitten do not provide the Qwen EOS-step contract and are not treated as Qwen.
Qwen steps need not equal output codec-frame count: the terminal EOS and any non-audio thinking tokens are steps but
are not delivered as codec frames.

The header metadata is a producer declaration, not a proof against a malicious or corrupted producer. Stop weights
are filtered normalized f32 sampler weights, not quality scores or a continuous-RNG guarantee. See
[the hook contract](../../../../docs/instruments/hooks.md). The instrument does not replace the owner's listening or
baseline gates.

```sh
ggtts-measure --audio input.flac --events trace.jsonl --report /new/report \
  --instrument stop-probability --require stop-probability
```
