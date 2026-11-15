#!/usr/bin/env bash
# The sampler settings, frame cap and language against the reference CLI (#351): for each setting changed one at a
# time, at a fixed seed, on CPU, the codec tokens of `examples/generate` and of the reference CLI must be the same file.
# The reference has no sub-talker settings (its code predictor samples with the talker's), so for those it checks that
# giving the talker's own value changes nothing, and that another value changes the codes.
#
#   QWEN3_TTS_BUILD=<the reference's CPU build at 16bb5af> TALKER=<talker.gguf> CODEC=<codec.gguf> THREADS=4 \
#   crates/ggtts-qwen3-tts/scripts/sampler_parity.sh
#
# Prints one row per setting and exits 1 if any differs. TEXT (default: a fixture line) and SEED change the case.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
build=${QWEN3_TTS_BUILD:?}
talker=${TALKER:?}
codec=${CODEC:?}
threads=${THREADS:-4}
seed=${SEED:-20260905}
text=${TEXT:-$(sed -n 's/^park\t//p' "$here/fixtures/cpu/texts.tsv")}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/model"
ln -s "$(readlink -f "$talker")" "$work/model/$(basename "$talker")"
(cd "$root" && cargo build --release --locked -q -p ggtts-qwen3-tts --features ggml --example generate)
gen="${CARGO_TARGET_DIR:-$root/target}/release/examples/generate"

ours() { "$gen" "$talker" "$text" "$work/$1.ggtts.json" "$seed" "$threads" "${@:2}" 2> "$work/$1.ggtts.log"; }
theirs() {
  "$build/qwen3-tts-cli" -m "$work/model" --codec-model "$codec" --speaker hk_cantonese_speaker --max-tokens 1200 \
    -l zh --seed "$seed" -j "$threads" -t "$text" "${@:2}" -o "$work/$1.wav" \
    --dump-generated-codes "$work/$1.ref.json" > "$work/$1.ref.log" 2>&1
}
frames() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$1"; }

fail=0
printf '%-28s %-34s %8s %8s  %s\n' setting value ggtts reference verdict
row() {
  local name=$1 value=$2; shift 2
  ours "$name" "$@"
  theirs "$name" "$@"
  local verdict=same
  cmp -s "$work/$name.ggtts.json" "$work/$name.ref.json" || { verdict=DIFFERENT; fail=1; }
  printf '%-28s %-34s %8s %8s  %s\n' "$name" "$value" "$(frames "$work/$name.ggtts.json")" \
    "$(frames "$work/$name.ref.json")" "$verdict"
}
row default "(the reference CLI's)"
row temperature "--temperature 0.7" --temperature 0.7
row top_k "--top-k 10" --top-k 10
row top_p "--top-p 0.8" --top-p 0.8
row repetition_penalty "--repetition-penalty 1.3" --repetition-penalty 1.3
row max_frames "--max-tokens 24" --max-tokens 24
row language "-l en" -l en
row language_auto "-l auto" -l auto

# The sub-talker: the talker's own values are no change; another value is a change.
sub() {
  local name=$1 value=$2 want=$3; shift 3
  ours "$name" "$@"
  local got=change
  cmp -s "$work/$name.ggtts.json" "$work/default.ggtts.json" && got=none
  local verdict=as-expected
  [ "$got" = "$want" ] || { verdict=UNEXPECTED; fail=1; }
  printf '%-28s %-34s %8s %8s  %s (%s)\n' "$name" "$value" "$(frames "$work/$name.ggtts.json")" - "$verdict" \
    "$got against default"
}
sub subtalker_same "--subtalker-temperature 0.9 ..." none --subtalker-temperature 0.9 --subtalker-top-k 50 \
  --subtalker-top-p 1.0
sub subtalker_temperature "--subtalker-temperature 0.6" change --subtalker-temperature 0.6
sub subtalker_top_k "--subtalker-top-k 5" change --subtalker-top-k 5
sub subtalker_top_p "--subtalker-top-p 0.5" change --subtalker-top-p 0.5
exit $fail
