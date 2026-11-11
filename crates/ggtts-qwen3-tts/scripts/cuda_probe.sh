#!/usr/bin/env bash
# CUDA parity probe: the reference and this engine, both on CUDA, both linking the same ggml archives, on the eight
# fixture texts and the three style-instruction texts. Prints one line per fixture for the decoder and for the
# tokens, then the planted defects.
#
#   QWEN3_TTS_CPP=<checkout at 16bb5af> QWEN3_TTS_BUILD=<its static CUDA build, QWEN3_TTS_CUDA=ON> \
#   TALKER=<talker.gguf> CODEC=<codec.gguf> WORK=<scratch dir> scripts/cuda_probe.sh
#
# Optional: THREADS (4), CUDA_HOME (/usr/local/cuda), CARGO_TARGET_DIR (WORK/target), NICE ("nice -n 19"),
# GPU_WAIT_FILE (wait while this file exists, before any GPU work).
#
# Steps: (1) record CUDA fixtures with the reference CLI, each text twice, and the reference decoder's samples
# (scripts/record_fixtures.sh, REPEAT_DRIFT=report), then the instruction fixtures (fixtures/instruct/texts.tsv);
# (2) build the tests with the cuda and plant features against the same archives; (3) decoder and token identity
# on CUDA; (4) each planted defect's expected-failure test (`planted::*`), which passes only when its plant leaves
# fixtures differing; (5) for information, how many CUDA
# fixtures have the CPU fixtures' tokens. Exit status 0 only when every recorded fixture is identical on both checks
# and every plant is seen.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
: "${QWEN3_TTS_CPP:?}" "${QWEN3_TTS_BUILD:?}" "${TALKER:?}" "${CODEC:?}" "${WORK:?}"
nice=${NICE:-nice -n 19}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$WORK/target}
export GGML_LIB_DIR=$QWEN3_TTS_BUILD/ggml/src
[ -f "$GGML_LIB_DIR/ggml-cuda/libggml-cuda.a" ] || { echo "not a static CUDA build: $QWEN3_TTS_BUILD" >&2; exit 2; }
mkdir -p "$WORK"
fixtures=$WORK/fixtures-cuda
instruct_fixtures=$WORK/fixtures-cuda-instruct
status=0

# Build first, so the GPU wait covers only GPU work.
echo "== build (cuda and plant features, ggml archives from $GGML_LIB_DIR)"
(cd "$root" && $nice cargo test -j 4 --release --locked -p ggtts-qwen3-tts --features cuda,plant --no-run) > "$WORK/build.log" 2>&1 \
  || { echo "build failed: $WORK/build.log" >&2; exit 2; }

if [ -n "${GPU_WAIT_FILE:-}" ]; then
  while [ -e "$GPU_WAIT_FILE" ]; do echo "waiting: $GPU_WAIT_FILE exists"; sleep 60; done
fi
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | sed 's/^/device: /'

echo "== reference on CUDA: recording $fixtures"
rm -rf "$fixtures"
OUT=$fixtures BACKEND=cuda REPEAT_DRIFT=report THREADS=${THREADS:-4} \
  $nice "$here/scripts/record_fixtures.sh" 2>&1 | sed 's/^/reference: /'
n=$(ls "$fixtures"/*.json 2> /dev/null | wc -l)
echo "reference: $n of 8 fixtures recorded (a case is left out if its two recordings differ)"
[ "$n" -gt 0 ] || { echo "no CUDA fixtures recorded" >&2; exit 1; }
[ "$n" -eq 8 ] || status=1
echo "== reference on CUDA, with style instructions: recording $instruct_fixtures"
rm -rf "$instruct_fixtures"
OUT=$instruct_fixtures TEXTS=$here/fixtures/instruct/texts.tsv BACKEND=cuda REPEAT_DRIFT=report THREADS=${THREADS:-4} \
  $nice "$here/scripts/record_fixtures.sh" 2>&1 | sed 's/^/reference: /'
m=$(ls "$instruct_fixtures"/*.json 2> /dev/null | wc -l)
echo "reference: $m of 3 instruction fixtures recorded"
[ "$m" -eq 3 ] || status=1

run() { # test-file test-name log-name; returns the test's exit status
  (cd "$root" && GGTTS_DEVICE=cuda GGTTS_FIXTURES=$fixtures GGTTS_INSTRUCT_FIXTURES=$instruct_fixtures \
    GGTTS_TALKER=$TALKER GGTTS_CODEC=$CODEC \
    $nice cargo test -j 4 --release --locked -p ggtts-qwen3-tts --features cuda,plant --test "$1" \
      -- --ignored --exact "$2" --nocapture) > "$WORK/$1.$3.log" 2>&1
}

for t in decoder_identity:decoder_samples_are_bit_identical_to_the_reference \
         token_identity:codec_tokens_are_identical_to_the_reference \
         token_identity:instructed_codec_tokens_are_identical_to_the_reference; do
  check=${t%%:*} name=${t#*:}
  echo "== $check $name on CUDA"
  log=$WORK/$check.$name.log
  run "$check" "$name" "$name" || status=1
  grep -E '^(IDENTICAL|DIFFERS) |identity \(' "$log" || { echo "no result: $log"; tail -5 "$log"; }
  grep -q ', 0 differ' "$log" || status=1
done

echo "== planted defects on CUDA (each expected-failure test passes only when its plant is seen)"
for p in decoder_identity:rope_normal_is_seen decoder_identity:snake_no_epsilon_is_seen \
         decoder_identity:window_off_by_one_is_seen token_identity:rope_normal_is_seen \
         token_identity:sampler_off_by_one_is_seen token_identity:instruction_after_role_is_seen; do
  check=${p%%:*} name=${p#*:}
  if run "$check" "planted::$name" "$name"; then verdict="red as required"; else verdict="NOT SEEN"; status=1; fi
  line=$(grep -E 'identity \(' "$WORK/$check.$name.log" || echo "no result line (see $WORK/$check.$name.log)")
  echo "$verdict: $check $name: $line"
done

echo "== CUDA tokens against the CPU fixtures (information only)"
python3 - "$fixtures" "$here/fixtures/cpu" <<'PY'
import json, os, sys
cuda, cpu = sys.argv[1:]
same = 0
names = sorted(f for f in os.listdir(cuda) if f.endswith(".json"))
for f in names:
    a, b = json.load(open(os.path.join(cuda, f))), json.load(open(os.path.join(cpu, f)))
    eq = a["codes"] == b["codes"]
    same += eq
    print(f"  {f[:-5]}: {'same as CPU' if eq else 'differs from CPU'} ({a['frames']} vs {b['frames']} frames)")
print(f"  {same} of {len(names)} CUDA fixtures have the CPU tokens")
PY

echo "cuda probe: exit $status (logs in $WORK)"
exit $status
