#!/usr/bin/env bash
# The Kitten-TTS 2 engine on CUDA: every parity check the CPU path passes, rerun with --device cuda against the same
# CPU PyTorch recordings and the same bars (DESIGN.md), each planted fault expected red, then the same render twice.
#
#   GGML_LIB_DIR=<static CUDA build of ggml at the pin: libggml.a, libggml-base.a, libggml-cpu.a, ggml-cuda/>
#   KITTEN_REPO=<Kitten-TTS 2 model directory>   LM_GGUF=<converted LM, Q4_0>   S3_GGUF=<converted S3Gen>
#   REF_LM=<kitten_ref.py logits dir>   REF_S3=<kitten_ref.py s3dump dir>   REF_COND=<kitten_ref.py conddump dir>
#   VOICE=<ggtts-kitten prepare JSON>   WORK=<scratch dir>   scripts/cuda_probe.sh
#
# Optional: THREADS (4), CUDA_HOME (/usr/local/cuda), CARGO_TARGET_DIR (WORK/target), NICE ("nice -n 19"),
# GPU_WAIT_FILE (wait while this file exists, before any GPU work). Exit 0 only when every check passes and every
# plant is red. Nothing it writes is committed: it reads a Kitten model and writes under WORK.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
: "${GGML_LIB_DIR:?}" "${KITTEN_REPO:?}" "${LM_GGUF:?}" "${S3_GGUF:?}" "${REF_LM:?}" "${REF_S3:?}" "${REF_COND:?}"
: "${VOICE:?}" "${WORK:?}"
nice=${NICE:-nice -n 19}
threads=${THREADS:-4}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export GGML_LIB_DIR
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$WORK/target}
[ -f "$GGML_LIB_DIR/ggml-cuda/libggml-cuda.a" ] || { echo "not a static CUDA build: $GGML_LIB_DIR" >&2; exit 2; }
mkdir -p "$WORK"
status=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; status=1; }

echo "== build (cuda feature, ggml archives from $GGML_LIB_DIR)"
(cd "$root" && $nice cargo build -j 4 --release --locked -p ggtts-kitten -p ggtts-cli \
  --features ggtts-kitten/cuda,ggtts-cli/ggml) > "$WORK/build.log" 2>&1 \
  || { echo "build failed: $WORK/build.log" >&2; exit 2; }
K=$CARGO_TARGET_DIR/release/ggtts-kitten
C=$CARGO_TARGET_DIR/release/ggtts-kitten-convert
R=$CARGO_TARGET_DIR/release/ggtts-render

echo "== converter readback (CPU; the GGUF is the same on every device)"
if $nice "$C" --repo "$KITTEN_REPO" --verify "$LM_GGUF" > "$WORK/readback.log" 2>&1; then
  pass "readback: $(grep -o 'READBACK PASS.*groups of 128' "$WORK/readback.log")"
else fail "readback: $(tail -1 "$WORK/readback.log")"; fi
echo "== planted LM (group 256), converted on the CPU before any GPU work"
$nice "$C" --repo "$KITTEN_REPO" --out "$WORK/planted-g256.gguf" --plant group256 > "$WORK/planted-convert.log" 2>&1
grep -q 'READBACK FAIL' "$WORK/planted-convert.log" && pass "planted readback is red" || fail "planted readback was not red"

if [ -n "${GPU_WAIT_FILE:-}" ]; then
  while [ -e "$GPU_WAIT_FILE" ]; do echo "waiting: $GPU_WAIT_FILE exists"; sleep 60; done
fi
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | sed 's/^/device: /'

check() { # name expect(pass|fail) log command...
  local name=$1 expect=$2 log=$WORK/$3; shift 3
  "$@" > "$log" 2>&1; local rc=$?
  local verdict; verdict=$(grep -E 'PARITY (PASS|FAIL)' "$log" | tail -1)
  if [ "$expect" = pass ] && [ $rc -eq 0 ]; then pass "$name: $verdict"
  elif [ "$expect" = fail ] && [ $rc -eq 1 ]; then pass "$name (planted, red as required): $verdict"
  else fail "$name (expected $expect, exit $rc): ${verdict:-$(tail -1 "$log")}"; fi
}

echo "== LM logits on CUDA against CPU float32 PyTorch"
check "LM f32 arm" pass lm-f32.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights f32
check "LM ternary arm" pass lm-ternary.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights ternary
check "LM ternary arm, group 256" fail lm-planted.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$WORK/planted-g256.gguf" --ref "$REF_LM" --weights ternary
rm -f "$WORK/planted-g256.gguf"

echo "== S3Gen decoder on CUDA against PyTorch (recorded noise and source)"
check "vocoder" pass hift.log $nice "$K" hift-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"
check "flow and vocoder" pass decoder.log $nice "$K" decoder-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"

echo "== reference-clip preparation on CUDA against PyTorch"
check "conditioning" pass cond.log $nice "$K" cond-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_COND"
for p in no-rope no-batchnorm no-abs no-preemphasis; do
  check "conditioning, plant $p" fail "cond-$p.log" $nice "$K" cond-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_COND" --plant "$p"
done
grep -hE '^[A-Za-z].* (pass|FAIL)$' "$WORK/cond.log" | sed 's/^/  /'

echo "== the same render twice on CUDA"
text="The little lighthouse kept its lamp burning all through the stormy night."
for i in 1 2; do
  $nice "$R" --device cuda --threads "$threads" --model "$LM_GGUF" --codec "$S3_GGUF" --voice "$VOICE" --seed 1234 \
    --text "$text" --out "$WORK/cuda-$i.flac" > "$WORK/render-$i.log" 2>&1 || fail "render $i: $(tail -1 "$WORK/render-$i.log")"
  grep -h 'frames, stop\|wrote' "$WORK/render-$i.log" | sed 's/^/  /'
done
a=$(sha256sum < "$WORK/cuda-1.flac" | cut -c1-16); b=$(sha256sum < "$WORK/cuda-2.flac" | cut -c1-16)
[ -n "$a" ] && [ "$a" = "$b" ] && pass "CUDA render byte-identical twice (sha256 $a...)" || fail "CUDA renders differ ($a vs $b)"

echo "== result: $([ $status -eq 0 ] && echo ALL PASS || echo SOME FAIL)"
exit $status
