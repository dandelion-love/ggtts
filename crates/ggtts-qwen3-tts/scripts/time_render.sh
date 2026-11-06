#!/usr/bin/env bash
# Times the shipped ggtts-render against the reference CLI on one fixture text: same machine, same thread count,
# same request (speaker, language, seed, frame cap), alternating which goes first each round. Prints every round and
# the medians. The committed record is records/render-timing-cpu.log.
#
#   QWEN3_TTS_BUILD=<reference CPU build at 16bb5af> TALKER=<talker.gguf> CODEC=<codec.gguf> \
#   scripts/time_render.sh > records/render-timing-cpu.log
#
# Optional: RENDER (default target/release/ggtts-render, built here with the ggml feature), ROUNDS (9), THREADS (4),
# CASE (bus, a name in fixtures/cpu/texts.tsv), NICE (empty: timings under nice measure the scheduler too).
# Each process is timed from outside (wall clock, model load included) and by its own generate and decode lines.
# Both must report the same frame and sample counts every round, or the script fails.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
build=${QWEN3_TTS_BUILD:?}
talker=${TALKER:?}
codec=${CODEC:?}
rounds=${ROUNDS:-9}
threads=${THREADS:-4}
case_name=${CASE:-bus}
nice=${NICE:-}
seed=20260905
text=$(awk -F'\t' -v n="$case_name" '$1 == n { print $2 }' "$here/fixtures/cpu/texts.tsv")
[ -n "$text" ] || { echo "no case $case_name in fixtures/cpu/texts.tsv" >&2; exit 2; }
render=${RENDER:-}
if [ -z "$render" ]; then
  (cd "$root" && cargo build --release --locked -p ggtts-cli --features ggml) > /dev/null 2>&1 \
    || { echo "build failed" >&2; exit 2; }
  render=$root/target/release/ggtts-render
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/model"
ln -s "$(readlink -f "$talker")" "$work/model/$(basename "$talker")"

echo "# ggtts-render against the reference CLI, wall clock and stage times"
echo "commit: $(git -C "$root" rev-parse HEAD)"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cpu: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1); load before: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "reference: qwen3-tts-cli at $(git -C "${QWEN3_TTS_CPP:-$build/..}" rev-parse --short HEAD 2> /dev/null || echo '?')"
echo "case: $case_name; threads: $threads; seed: $seed; rounds: $rounds; nice: ${nice:-none}"
echo "talker sha256: $(sha256sum "$talker" | cut -d' ' -f1)"
echo "codec sha256: $(sha256sum "$codec" | cut -d' ' -f1)"
echo
printf 'round\torder\ttool\twall_ms\tgenerate_ms\tdecode_ms\tframes\tsamples\n'

ms() { echo $(($(date +%s%N) / 1000000)); }
run_ref() {
  local t0 t1
  t0=$(ms)
  $nice "$build/qwen3-tts-cli" -m "$work/model" --codec-model "$codec" --speaker hk_cantonese_speaker \
    --max-tokens 1200 -l zh --seed "$seed" -j "$threads" -t "$text" -o "$work/ref.wav" > "$work/ref.log" 2>&1
  t1=$(ms)
  local gen dec fs
  gen=$(sed -n 's/^  Generate: *\([0-9]*\) ms.*/\1/p' "$work/ref.log" | head -n 1)
  dec=$(sed -n 's/^  Decode: *\([0-9]*\) ms.*/\1/p' "$work/ref.log" | head -n 1)
  fs=$(sed -n 's|^    frames/samples: *\([0-9]*\) / \([0-9]*\).*|\1\t\2|p' "$work/ref.log" | head -n 1)
  printf '%s\t%s\treference\t%s\t%s\t%s\t%s\n' "$1" "$2" $((t1 - t0)) "$gen" "$dec" "$fs"
}
run_ggtts() {
  local t0 t1
  t0=$(ms)
  $nice "$render" --model "$talker" --codec "$codec" -j "$threads" --seed "$seed" --text "$text" \
    --out "$work/ggtts.flac" > "$work/ggtts.log" 2>&1
  t1=$(ms)
  local gen dec frames samples
  gen=$(sed -n 's/.*generate \([0-9]*\) ms.*/\1/p' "$work/ggtts.log" | head -n 1)
  dec=$(sed -n 's/.*decode \([0-9]*\) ms.*/\1/p' "$work/ggtts.log" | head -n 1)
  frames=$(sed -n 's/^ggtts-render: \([0-9]*\) frames.*/\1/p' "$work/ggtts.log" | head -n 1)
  samples=$(sed -n 's/.*(\([0-9]*\) samples at.*/\1/p' "$work/ggtts.log" | head -n 1)
  printf '%s\t%s\tggtts-render\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" $((t1 - t0)) "$gen" "$dec" "$frames" "$samples"
}

: > "$work/rows.tsv"
for r in $(seq 1 "$rounds"); do
  if [ $((r % 2)) -eq 1 ]; then
    run_ref "$r" ref-first | tee -a "$work/rows.tsv"
    run_ggtts "$r" ref-first | tee -a "$work/rows.tsv"
  else
    run_ggtts "$r" ggtts-first | tee -a "$work/rows.tsv"
    run_ref "$r" ggtts-first | tee -a "$work/rows.tsv"
  fi
done

echo
echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)"
python3 - "$work/rows.tsv" <<'PY'
import statistics, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r[2], []).append(r)
counts = {tuple(r[6:8]) for r in rows}
if len(counts) != 1 or any(not c for c in next(iter(counts))):
    print(f"FAILED: frame and sample counts differ or are missing: {sorted(counts)}")
    sys.exit(1)
frames, samples = next(iter(counts))
print(f"every run: {frames} frames, {samples} samples, on both tools")
print("medians over", len(by["reference"]), "rounds:")
for col, name in [(3, "wall clock, model load included"), (4, "generate"), (5, "decode")]:
    ref = statistics.median(int(r[col]) for r in by["reference"])
    ours = statistics.median(int(r[col]) for r in by["ggtts-render"])
    lo = min(int(r[col]) for r in by["ggtts-render"]); hi = max(int(r[col]) for r in by["ggtts-render"])
    rlo = min(int(r[col]) for r in by["reference"]); rhi = max(int(r[col]) for r in by["reference"])
    print(f"  {name}: ggtts-render {ours:.0f} ms [{lo}-{hi}] against reference {ref:.0f} ms [{rlo}-{rhi}]: "
          f"{100 * (ours - ref) / ref:+.1f}%")
PY
