#!/usr/bin/env bash
# Records golden fixtures: for each line of fixtures/cpu/texts.tsv (name TAB text, written for this purpose),
# the reference CLI's codec tokens, and the reference decoder's samples as a sha256 of their little-endian f32 bytes.
#
#   QWEN3_TTS_CPP=<checkout at 16bb5af> QWEN3_TTS_BUILD=<its CPU build> TALKER=<talker.gguf> CODEC=<codec.gguf> \
#   THREADS=4 scripts/record_fixtures.sh
#
# TEXTS (default fixtures/cpu/texts.tsv) is the list of cases. A third column, when present and not empty, is a
# style instruction, passed to the reference CLI as `--instruction` and recorded as `request.instruction`; the
# instruction fixtures are `TEXTS=fixtures/instruct/texts.tsv OUT=fixtures/instruct`.
#
# OUT (default fixtures/cpu) is where the fixture files go. BACKEND (default cpu) is the label recorded; it must say
# what the build runs on, since a CUDA build of the reference picks the GPU by itself. With REPEAT_DRIFT=report a
# case whose two recordings differ is reported and skipped, rather than stopping the script.
#
# The talker and codec are the published v0.0.1 q8_0 GGUFs; their hashes go into every fixture and the checks refuse
# other files. Every case is recorded twice and must repeat byte for byte, or the script stops.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
build=${QWEN3_TTS_BUILD:?}
talker=${TALKER:?}
codec=${CODEC:?}
threads=${THREADS:-4}
out_dir=${OUT:-$here/fixtures/cpu}
texts=${TEXTS:-$here/fixtures/cpu/texts.tsv}
backend=${BACKEND:-cpu}
mkdir -p "$out_dir"
seed=20260905
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"$here/reference/build.sh" "$work/bin" > /dev/null
mkdir -p "$work/model"
ln -s "$(readlink -f "$talker")" "$work/model/$(basename "$talker")"
talker_sha=$(sha256sum "$talker" | cut -d' ' -f1)
codec_sha=$(sha256sum "$codec" | cut -d' ' -f1)
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)
if [ "$backend" = cuda ]; then
  cpu="$cpu; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
fi

while IFS=$'\t' read -r name text instruction; do
  [ -n "$name" ] || continue
  instruct=()
  [ -z "$instruction" ] || instruct=(--instruction "$instruction")
  for take in 1 2; do
    "$build/qwen3-tts-cli" -m "$work/model" --codec-model "$codec" --speaker hk_cantonese_speaker \
      --max-tokens 1200 -l zh --seed "$seed" -j "$threads" -t "$text" "${instruct[@]}" -o "$work/$name.wav" \
      --dump-generated-codes "$work/$name.$take.codes" > "$work/$name.$take.log" 2>&1
  done
  if ! cmp -s "$work/$name.1.codes" "$work/$name.2.codes"; then
    echo "$name: REPEAT-DIFFERS: the reference's two recordings differ" >&2
    [ "${REPEAT_DRIFT:-stop}" = report ] && continue
    exit 1
  fi
  "$work/bin/decode_ref" "$codec" "$work/$name.1.codes" "$work/$name.f32" "$threads" 2> /dev/null
  python3 - "$name" "$text" "$work/$name.1.codes" "$work/$name.f32" "$out_dir/$name.json" \
    "$talker_sha" "$codec_sha" "$threads" "$cpu" "$seed" "$backend" "$instruction" <<'PY'
import hashlib, json, sys
name, text, codes_path, f32_path, out, talker_sha, codec_sha, threads, cpu, seed, backend, instruction = sys.argv[1:]
dump = json.load(open(codes_path))
samples = open(f32_path, "rb").read()
fixture = {
    "format": "ggtts-qwen3-tts-golden/1",
    "name": name,
    "text": text,
    "text_source": "written for this fixture set by a language model",
    "request": {"speaker": "hk_cantonese_speaker", "language": "zh", "seed": int(seed), "max_tokens": 1200,
                "sampler": "the reference CLI defaults: temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05"},
    "reference": {"engine": "qwen3-tts.cpp", "commit": "16bb5afcd06311031c72a8488f8d59660dc2fb46",
                  "ggml": "af97976c7810cdabb1863172f31c432dab767de7", "backend": backend,
                  "threads": int(threads), "cpu": cpu, "repeated": True},
    "talker_sha256": talker_sha,
    "codec_sha256": codec_sha,
    "frames": dump["frames"],
    "codebooks": dump["codebooks"],
    "codes": "@CODES@",
    "samples": {"count": len(samples) // 4, "sha256_f32le": hashlib.sha256(samples).hexdigest()},
}
if instruction:
    fixture["request"]["instruction"] = instruction
cb = dump["codebooks"]
rows = [",".join(str(c) for c in dump["codes"][i:i + cb]) for i in range(0, len(dump["codes"]), cb)]
body = json.dumps(fixture, ensure_ascii=False, indent=1)
body = body.replace('"@CODES@"', "[\n  " + ",\n  ".join(rows) + "\n ]")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(body + "\n")
print(f"{name}: {dump['frames']} frames, {len(samples) // 4} samples")
PY
done < "$texts"
