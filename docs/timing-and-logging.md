# Timing and logging

Both `ggtts-render` and `ggtts-measure` accept `--log-level error|warn|info|debug|trace` and `--log-format text|json`. The defaults are `info` and `text`; `GGTTS_LOG_LEVEL` and `GGTTS_LOG_FORMAT` set defaults, and command-line options override them. Errors remain visible at every level. Logs go to stderr, leaving audio and data on stdout intact. JSON logs use one `ggtts-log/1` object per line; the schema is in `schemas/ggtts-log-1.schema.json`. The instrument runner uses the same settings for its `run.log`. An instrument's own output is captured separately in its per-instrument log and is not converted to this format. Native ggml messages go through the CLI logger where the backend uses its callback; other native or child-process output may bypass it.

Errors encountered while parsing the logging configuration itself are currently plain text, including when JSON was requested. Consumers must handle that startup failure separately.

`ggtts-render --timing-report path.json` writes a new `ggtts-timing/1` report even at error level or without a trace. The path must not already exist. Its schema is `schemas/ggtts-timing-1.schema.json`. `--trace` enables the story event trace; `--trace-level 1` selects summary events and `--trace-level 2` adds detailed events. Traces can contain input text and should be handled as user data.

The report uses host wall clocks. `load_ms` is model loading; `render_ms` runs from the start of rendering through stream handoffs. Generation, decoding and judging are accumulated per draw; generated frames per second uses codec frames and generation wall time. Generation includes synchronous observers, including trace emission, and is not an isolated GPU kernel measurement. RTF is render wall seconds divided by audio seconds. Artifact writing and the instrument phase are separate. Stream preparation includes levelling; stream write time measures calls into the configured writer. First audio is measured immediately before the first write, not playback at an audio device. It is absent when no stream write occurs. Trace sink time covers serialization and writes; flush is reported separately. Logger counters cover emitted records, sink time and stderr lock waits in this process, excluding caller preparation and child logs.

These intervals overlap and must not be summed or blindly subtracted. GPU work may be asynchronous or fall back to the CPU; these measurements do not prove exclusive GPU compute time. A missing stage is `null`, not zero. In particular, the legacy single-text trace is not measured by the story trace-sink counter.

For a repeatable comparison, run `scripts/logging_overhead_probe.py --help`. It warms up, rotates the order of logging and trace settings, records individual timings and ranges, and checks output identity. Supply a fresh scratch directory outside the checkout: it will contain generated audio and potentially text-bearing traces. Compare runs on an otherwise idle machine with the same input, models, seed, thread count and backend, and report the distribution and any fallback or synchronization limits. The probe itself requests timing instrumentation in every mode; it does not measure an entirely uninstrumented build.

For example, replace the paths with local assets and choose `cpu` or `cuda`:

```sh
python3 scripts/logging_overhead_probe.py --render /path/to/ggtts-render \
  --engine qwen3-tts --device cpu --threads 4 --story /path/to/story.txt \
  --model /path/to/talker.gguf --codec /path/to/codec.gguf \
  --work /tmp/ggtts-probe-cpu --repeats 3
```

## Example comparison

On an i9-12900K with four threads and an RTX 3080 Ti, the probe rendered one synthetic Qwen story three times per setting after a warmup. Times below are median render milliseconds (minimum–maximum). CPU and CUDA used separately built executables linked against existing ggml libraries. All settings on each backend produced identical FLAC hashes and frame counts; traced modes also had identical emitted codebook-0 tokens and draw seeds.

| Setting | CPU ms | CUDA-requested ms |
| --- | ---: | ---: |
| error, no trace | 10,965 (10,945–11,215) | 4,570 (4,540–4,578) |
| info, no trace | 10,902 (10,776–11,049) | 4,507 (4,299–4,548) |
| debug, no trace | 10,908 (10,750–11,026) | 4,551 (4,520–4,562) |
| info, trace 1 | 10,854 (10,761–11,115) | 4,610 (4,597–4,721) |
| info, trace 2 | 10,738 (10,727–11,169) | 4,525 (4,419–4,557) |

For example, trace 1 on CUDA had a 40 ms (+0.88%) median difference from the error-level baseline. Three runs on one input are insufficient to establish a general overhead; the variations and backend caveats above apply. In the CPU info/no-trace runs, median stream-write and logger-sink times were 0.454 ms and 0.122 ms; on CUDA they were 0.386 ms and 0.170 ms. These costs are included in or overlap other intervals. Error-level is the quiet baseline, not a switch that suppresses errors.

The story SHA-256 was `90e57e18382c982034cf5da44230689c89634576cb2daef2fb1d54b7679e5e3a`; model and codec hashes were `33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67` and `7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824`. CPU and CUDA executable hashes were `ebd7b68fba8ba1ad01c937d71eb82e1f9e9d46b5124dcb9c88759e13411176b3` and `f3cb0a5a36b36e08e49a3f11a70a229660f3badd1a6bf495287a29f3727cd3e9`. The probe's summary retains every individual sample and input hash in its scratch directory; no model, voice, story, audio or trace is included here.

The same CUDA-requested comparison with Kitten produced median (minimum–maximum) render times of 2,893 (2,836–2,914) ms at error/no trace, 2,908 (2,874–2,992) ms at info/no trace, 2,903 (2,862–2,941) ms at debug/no trace, 2,875 (2,861–2,883) ms at info/trace 1, and 2,929 (2,831–2,938) ms at info/trace 2. Audio hashes and generated frame counts matched across modes. This engine did not emit per-step token events, so the comparison does not claim token identity. Its model, codec and voice SHA-256 hashes were `263b8e69682a4b312fed4144aa29d9d3a49bb466f61381f5e73d61ac7707a431`, `54ad2c36c0b322788a55f79dcd0207e3d9114bb6370b6e3f2800ac9b8e362321`, and `42a961894afcb667c1787e338a2097521f315e82934ede7a7fa3ed6c13aaa46d` respectively.
