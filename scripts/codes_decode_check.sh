#!/usr/bin/env bash
# codes_decode_check.sh: the codes a render writes (--codes) decode again, through the decode example, to the
# render's own samples; and the FLAC is the same with or without --codes. docs/codes.md.
#
#   scripts/codes_decode_check.sh [cpu|cuda] [OUT_DIR]
#
# Renders one --text line with the default Qwen3-TTS model from the per-user cache (--offline: it never downloads),
# on the device given, with --threads 4. The raw s16le stream is the decoded draw (a --text render has no trim and
# no level step); the decode example decodes the codes line on the same device, and the samples are compared after
# the same 16-bit rounding (ggtts_core::flac::to_i16). Builds ggtts-render and the example in release with ggml, and
# with cuda for cuda. Exit 0 when both hold, 1 when either differs.
set -euo pipefail
device=${1:-cpu}
case "$device" in
  cpu) features=ggml ;;
  cuda) features=ggml,cuda ;;
  *) echo "codes_decode_check: the device is cpu or cuda, not $device" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
out=${2:-$(mktemp -d)}
mkdir -p "$out"
cd "$root"
cargo build --release --locked -p ggtts-cli --features "$features" --bin ggtts-render
cargo build --release --locked -p ggtts-qwen3-tts --features "$features" --example decode
target=${CARGO_TARGET_DIR:-$root/target}
render=$target/release/ggtts-render
text="今日天氣好好，我哋去公園行下。"
"$render" --offline --threads 4 --device "$device" --text "$text" --out "$out/a.flac" \
  --stream "$out/a.s16" --stream-format s16le --stream-raw --codes "$out/a.codes.jsonl" --log-level warn
"$render" --offline --threads 4 --device "$device" --text "$text" --out "$out/b.flac" --log-level warn
codec=$(find "${GGTTS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ggtts}" -name 'qwen-tokenizer-12hz-q8_0.gguf' | head -1)
test -n "$codec" || { echo "codes_decode_check: the default codec is not in the cache" >&2; exit 2; }
sed -n 1p "$out/a.codes.jsonl" > "$out/chunk0.json"
"$target/release/examples/decode" "$codec" "$out/chunk0.json" "$out/chunk0.f32" 4 none "$device"
python3 - "$out" "$device" <<'PY'
import array, json, sys

out, device = sys.argv[1], sys.argv[2]
line = json.loads(open(f"{out}/a.codes.jsonl").readline())
print("codes:", {k: line[k] for k in ("format", "engine", "chunk", "draw", "seed", "frames", "codebooks")})
raw = open(f"{out}/a.s16", "rb").read()
stream = array.array("h")
stream.frombytes(raw[raw.index(b"\n") + 1:])
dec = array.array("f")
dec.frombytes(open(f"{out}/chunk0.f32", "rb").read())
if sys.byteorder != "little":
    stream.byteswap()
    dec.byteswap()
again = [max(-32768, min(32767, round(float(x) * 32767.0))) for x in dec]  # round() is half to even
same_flac = open(f"{out}/a.flac", "rb").read() == open(f"{out}/b.flac", "rb").read()
same_codes = list(stream) == again
print(f"{device}: {len(stream)} samples rendered, {len(again)} decoded again from the codes")
print(f"{device}: the FLAC is the same with or without --codes: {same_flac}")
print(f"{device}: the codes decode again to the render's samples: {same_codes}")
sys.exit(0 if same_flac and same_codes else 1)
PY
