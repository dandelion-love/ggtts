# Offline Qwen3-ASR instrument

`asr-qwen` is an optional, ordinary completed-audio instrument. It loads a local
Qwen3-ASR snapshot on CPU, compares its Cantonese recognition against a local
UTF-8 reference, and emits normalized character error rate. It never decides
speech quality or proves that a low score means every word was spoken. The
runner executes selected instruments in order; requiring this instrument makes
an unavailable, timed-out or malformed result incomplete. An optional failure
stays visible without alone making the measurement incomplete.

Install `qwen-asr`, CPU-capable PyTorch and the snapshot separately. Supply
only a local snapshot containing `config.json` and `.safetensors` weights; this
instrument sets the Hugging Face offline flags and asks the model loader for
local files only. Record the model source, revision and licence when obtaining a
snapshot. There is no model download or bundled model in this repository.
The [Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
declares Apache-2.0; record the actual snapshot revision and digest separately.

Example configuration through an instrument flag:

```
--instrument asr-qwen,arg.model_dir=/local/snapshot,arg.reference_file=/local/reference.txt
--require asr-qwen
```

The instrument does not put reference or recognized text in arguments, logs or
its envelope. It hashes the completed audio and reference with fixed public
identifiers; the model identity is SHA-256 over each sorted relative snapshot
filename (length-prefixed) and that file's SHA-256 bytes. It records the model
hash, engine, CPU, thread count, package version and normalization rule. The
caller supplies and identifies the audio trim state; this instrument does not
trim it. The generic runner's failure reports can retain configured file paths,
so keep run reports private and use paths without secrets or transcript text.

Synthetic unit tests use a fake recognizer. A real-model CPU smoke and a
validated omission-sensitive comparison remain required before making claims
about this engine's behavior on a corpus. An empty recognition against a
nonempty reference fails rather than reporting a zero error rate.
