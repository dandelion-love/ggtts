#!/usr/bin/env bash
# The Kitten-TTS 2 engine on CUDA: every parity check the CPU path passes, rerun with --device cuda against the same
# CPU PyTorch recordings and the same bars (DESIGN.md), each planted fault expected red, then the same render twice.
#
#   GGML_LIB_DIR=<static CUDA build of ggml at the pin: libggml.a, libggml-base.a, libggml-cpu.a, ggml-cuda/>
#   KITTEN_REPO=<Kitten-TTS 2 model directory>   S3_GGUF=<converted S3Gen>
#   REF_LM=<kitten_ref.py logits dir>   REF_S3=<kitten_ref.py s3dump dir>   REF_COND=<kitten_ref.py conddump dir>
#   VOICE=<ggtts-kitten prepare JSON>   WORK=<scratch dir>   scripts/cuda_probe.sh
#
# The LM GGUF is converted fresh from KITTEN_REPO into WORK on every run, by the binaries this run builds, and the
# probe refuses it unless the tokenizer is embedded. A GGUF passed in from an earlier converter is not used: one
# converted before the tokenizer was embedded made both renders of an earlier run fail ("this GGUF has no embedded
# tokenizer"), so the determinism check went unmeasured.
#
# Optional: THREADS (4), CUDA_HOME (/usr/local/cuda), CARGO_TARGET_DIR (WORK/target), NICE ("nice -n 19"),
# GPU_WAIT_FILE (wait while this file exists, before any GPU work), KEEP_GGUF (1 keeps WORK/lm.gguf).
# TQ1_GGUF=<a Kitten GGUF in the experimental tq1_0-256-experimental layout> adds the TQ1 checks on CUDA. The
# converter here does not write TQ1, so that model is passed in, and no PyTorch recording exists for it: its
# reference is this engine's CPU f32 arm of the same GGUF on the recorded prompt (written into WORK/tq1-ref by
# `parity --write-ref`), held to the same bars. The plant is a copy with one TQ1 matrix zeroed, which must be red.
# Needs room in WORK for that copy. Without TQ1_GGUF, or with STEPS=render, the TQ1 checks do not run and the
# result line says so (`PASS, TQ1_0 NOT RUN`) instead of `ALL PASS`; scripts/rc_gate.py reads that line.
# DEVICE=cpu with STEPS=render runs only the conversion and the render twice, on the CPU, without CUDA: the way to
# check this script's own render step on a machine whose GPU is busy (GGML_LIB_DIR is then not needed).
# Exit 0 only when every check passes and every plant is red. Nothing it writes is committed: it reads a Kitten
# model and writes under WORK.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
device=${DEVICE:-cuda}
steps=${STEPS:-all}
case "$device/$steps" in
  cuda/all | cuda/render | cpu/render) ;;
  *) echo "DEVICE=$device STEPS=$steps: use DEVICE=cuda (STEPS=all|render) or DEVICE=cpu STEPS=render" >&2; exit 2 ;;
esac
: "${KITTEN_REPO:?}" "${S3_GGUF:?}" "${VOICE:?}" "${WORK:?}"
[ "$steps" = all ] && : "${REF_LM:?}" "${REF_S3:?}" "${REF_COND:?}"
[ -n "${LM_GGUF:-}" ] && echo "note: LM_GGUF is no longer read; the LM is converted fresh into $WORK/lm.gguf"
nice=${NICE:-nice -n 19}
threads=${THREADS:-4}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$WORK/target}
mkdir -p "$WORK"
status=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; status=1; }

if [ "$device" = cuda ]; then
  : "${GGML_LIB_DIR:?}"
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export PATH=$CUDA_HOME/bin:$PATH
  export GGML_LIB_DIR
  [ -f "$GGML_LIB_DIR/ggml-cuda/libggml-cuda.a" ] || { echo "not a static CUDA build: $GGML_LIB_DIR" >&2; exit 2; }
  features=ggtts-kitten/cuda,ggtts-cli/ggml
  echo "== build (cuda feature, ggml archives from $GGML_LIB_DIR)"
else
  unset GGML_LIB_DIR # the build would link those archives, CUDA ones included, instead of building ggml for the CPU
  features=ggtts-kitten/ggml,ggtts-cli/ggml
  echo "== build (CPU, ggml from source)"
fi
(cd "$root" && $nice cargo build -j 4 --release --locked -p ggtts-kitten -p ggtts-cli --features "$features") \
  > "$WORK/build.log" 2>&1 || { echo "build failed: $WORK/build.log" >&2; exit 2; }
K=$CARGO_TARGET_DIR/release/ggtts-kitten
C=$CARGO_TARGET_DIR/release/ggtts-kitten-convert
R=$CARGO_TARGET_DIR/release/ggtts-render

echo "== LM converted fresh, then its readback (CPU; the GGUF is the same on every device)"
LM_GGUF=$WORK/lm.gguf
rm -f "$LM_GGUF"
if $nice "$C" --repo "$KITTEN_REPO" --out "$LM_GGUF" > "$WORK/convert.log" 2>&1; then
  pass "readback: $(grep -o 'READBACK PASS.*groups of 128' "$WORK/convert.log")"
else
  fail "conversion or readback: $(grep -hE 'READBACK|error' "$WORK/convert.log" | tail -1)"
fi
# The renderer reads the tokenizer from the GGUF's metadata, which sits before the tensor data.
if [ -f "$LM_GGUF" ] && grep -aqm1 'kitten.tokenizer.json' <(head -c 67108864 "$LM_GGUF"); then
  pass "the LM GGUF carries its tokenizer"
else
  echo "FAIL  $LM_GGUF has no embedded tokenizer: refusing to run the rest of the probe on it"
  exit 1
fi

if [ "$steps" = all ]; then
  echo "== planted LM (group 256), converted on the CPU before any GPU work"
  $nice "$C" --repo "$KITTEN_REPO" --out "$WORK/planted-g256.gguf" --plant group256 > "$WORK/planted-convert.log" 2>&1
  grep -q 'READBACK FAIL' "$WORK/planted-convert.log" && pass "planted readback is red" || fail "planted readback was not red"
fi

tq1=0
if [ "$steps" = all ] && [ -n "${TQ1_GGUF:-}" ]; then
  echo "== TQ1_0 model: its CPU f32 reference and the planted copy, before any GPU work"
  [ -f "$TQ1_GGUF" ] || { echo "TQ1_GGUF is not a file: $TQ1_GGUF" >&2; exit 2; }
  rm -rf "$WORK/tq1-ref" "$WORK/tq1-planted.gguf"
  if $nice "$K" parity --device cpu --threads "$threads" --gguf "$TQ1_GGUF" --ref "$REF_LM" --weights f32 \
    --write-ref "$WORK/tq1-ref" > "$WORK/tq1-ref.log" 2>&1; then
    tq1=1
    pass "TQ1 reference: $(grep -o 'REFERENCE WRITTEN.*' "$WORK/tq1-ref.log" | sed "s|$WORK/||")"
    # One TQ1 matrix zeroed in a copy, at the byte range the reference run printed. A TQ1 block of zeros is a
    # valid block with a zero scale, so the planted copy loads and the comparison is what refuses it.
    read -r off len < <(sed -n 's/^PLANT SPAN: .* bytes \([0-9]*\) \([0-9]*\)$/\1 \2/p' "$WORK/tq1-ref.log")
    cp "$TQ1_GGUF" "$WORK/tq1-planted.gguf"
    { [ -n "${len:-}" ] && dd if=/dev/zero of="$WORK/tq1-planted.gguf" bs=1M seek="$off" count="$len" \
      oflag=seek_bytes iflag=count_bytes conv=notrunc status=none; } || fail "could not plant $WORK/tq1-planted.gguf"
  else
    fail "TQ1 reference (CPU f32 arm): $(grep -hE 'error|panicked' "$WORK/tq1-ref.log" | tail -1)"
  fi
elif [ "$steps" = all ]; then
  echo "NOTE  TQ1_GGUF is not set: the TQ1_0 checks on CUDA did not run"
fi
tq1_checked=0

if [ "$device" = cuda ]; then
  if [ -n "${GPU_WAIT_FILE:-}" ]; then
    while [ -e "$GPU_WAIT_FILE" ]; do echo "waiting: $GPU_WAIT_FILE exists"; sleep 60; done
  fi
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | sed 's/^/device: /'
fi

check() { # name expect(pass|fail) log command...
  local name=$1 expect=$2 log=$WORK/$3; shift 3
  "$@" > "$log" 2>&1; local rc=$?
  local verdict; verdict=$(grep -E 'PARITY (PASS|FAIL)' "$log" | tail -1)
  if [ "$expect" = pass ] && [ $rc -eq 0 ]; then pass "$name: $verdict"
  elif [ "$expect" = fail ] && [ $rc -eq 1 ]; then pass "$name (planted, red as required): $verdict"
  else fail "$name (expected $expect, exit $rc): ${verdict:-$(tail -1 "$log")}"; fi
}
# A diagnostic run: its numbers are printed and never decide the exit status.
diag() { # name log command...
  local name=$1 log=$WORK/$2; shift 2
  "$@" > "$log" 2>&1
  echo "INFO  $name: $(grep -hE 'SNR|min cosine|error' "$log" | tail -1)"
}

if [ "$steps" = all ]; then
  echo "== LM logits on CUDA against CPU float32 PyTorch"
  check "LM f32 arm" pass lm-f32.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights f32
  check "LM ternary arm" pass lm-ternary.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights ternary
  check "LM ternary arm, group 256" fail lm-planted.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$WORK/planted-g256.gguf" --ref "$REF_LM" --weights ternary
  rm -f "$WORK/planted-g256.gguf"
  grep -h '^ternary matrices\|NVIDIA_TF32' "$WORK/lm-ternary.log" | sed 's/^/  /'

  echo "== single-token forward against the batched forward on CUDA (the path generation takes)"
  check "single-token, ternary" pass step-ternary.log $nice "$K" step-parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights ternary
  check "single-token, f32" pass step-f32.log $nice "$K" step-parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights f32
  # The control: ggml's fusions back on, which the engine turns off. Expected to fail, or to error naming a layer.
  diag "single-token, ternary, fusions on (GGML_CUDA_DISABLE_FUSION=0, control)" step-fused.log \
    env GGML_CUDA_DISABLE_FUSION=0 $nice "$K" step-parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights ternary

  if [ $tq1 = 1 ]; then
    echo "== TQ1_0 model on CUDA against its CPU f32 arm, both weights modes, then the single-token path"
    check "TQ1 f32 arm" pass tq1-f32.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$TQ1_GGUF" --ref "$WORK/tq1-ref" --weights f32
    check "TQ1 ternary arm" pass tq1-ternary.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$TQ1_GGUF" --ref "$WORK/tq1-ref" --weights ternary
    check "TQ1 ternary arm, one matrix zeroed" fail tq1-planted.log $nice "$K" parity --device cuda --threads "$threads" --gguf "$WORK/tq1-planted.gguf" --ref "$WORK/tq1-ref" --weights ternary
    rm -f "$WORK/tq1-planted.gguf"
    grep -h '^ternary matrices' "$WORK/tq1-ternary.log" | sed 's/^/  /'
    check "TQ1 single-token, ternary" pass tq1-step-ternary.log $nice "$K" step-parity --device cuda --threads "$threads" --gguf "$TQ1_GGUF" --ref "$REF_LM" --weights ternary
    tq1_checked=1
    check "TQ1 single-token, f32" pass tq1-step-f32.log $nice "$K" step-parity --device cuda --threads "$threads" --gguf "$TQ1_GGUF" --ref "$REF_LM" --weights f32
  fi

  echo "== S3Gen decoder on CUDA against PyTorch (recorded noise and source)"
  check "vocoder" pass hift.log $nice "$K" hift-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"
  check "flow and vocoder" pass decoder.log $nice "$K" decoder-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"
  # The engine sets NVIDIA_TF32_OVERRIDE=0 inside the process. These two runs set it from the shell instead: =1
  # gives back ggml's TF32 path (the control: it should read near the 30.7 dB measured before the fix), and =0
  # exported tells an override cuBLAS ignored when set in-process (in-process fails, exported passes) from a cause
  # that is not TF32 (both fail alike).
  diag "flow and vocoder, TF32 allowed (NVIDIA_TF32_OVERRIDE=1, control)" decoder-tf32.log \
    env NVIDIA_TF32_OVERRIDE=1 $nice "$K" decoder-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"
  diag "flow and vocoder, NVIDIA_TF32_OVERRIDE=0 exported" decoder-notf32-env.log \
    env NVIDIA_TF32_OVERRIDE=0 $nice "$K" decoder-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_S3"
  diag "LM ternary arm, TF32 allowed (NVIDIA_TF32_OVERRIDE=1)" lm-ternary-tf32.log \
    env NVIDIA_TF32_OVERRIDE=1 $nice "$K" parity --device cuda --threads "$threads" --gguf "$LM_GGUF" --ref "$REF_LM" --weights ternary

  echo "== reference-clip preparation on CUDA against PyTorch"
  check "conditioning" pass cond.log $nice "$K" cond-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_COND"
  for p in no-rope no-batchnorm no-abs no-preemphasis; do
    check "conditioning, plant $p" fail "cond-$p.log" $nice "$K" cond-parity --device cuda --threads "$threads" --s3gguf "$S3_GGUF" --ref "$REF_COND" --plant "$p"
  done
  grep -hE '^[A-Za-z].* (pass|FAIL)$' "$WORK/cond.log" | sed 's/^/  /'
fi

text="The little lighthouse kept its lamp burning all through the stormy night."
twice() { # tag gguf: the same render twice, byte-identical
  local tag=$1 gguf=$2 i a= b=
  rm -f "$WORK/$tag"-[12].flac
  for i in 1 2; do
    $nice "$R" --device "$device" --threads "$threads" --model "$gguf" --codec "$S3_GGUF" --voice "$VOICE" --seed 1234 \
      --text "$text" --out "$WORK/$tag-$i.flac" > "$WORK/render-$tag-$i.log" 2>&1 \
      || fail "$tag render $i: $(grep -hE 'error|panicked' "$WORK/render-$tag-$i.log" | head -1)"
    grep -h 'frames, stop\|wrote' "$WORK/render-$tag-$i.log" | sed 's/^/  /'
  done
  [ -f "$WORK/$tag-1.flac" ] && a=$(sha256sum < "$WORK/$tag-1.flac" | cut -c1-16)
  [ -f "$WORK/$tag-2.flac" ] && b=$(sha256sum < "$WORK/$tag-2.flac" | cut -c1-16)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then pass "$tag render byte-identical twice (sha256 $a...)"
  else fail "$tag renders differ or are missing (${a:-none} vs ${b:-none})"; fi
}
echo "== the same render twice on $device"
twice "$device" "$LM_GGUF"
if [ $tq1 = 1 ]; then
  echo "== the TQ1_0 model, the same render twice on $device"
  twice "$device-tq1" "$TQ1_GGUF"
fi

[ "${KEEP_GGUF:-0}" = 1 ] || rm -f "$LM_GGUF"
if [ $status -ne 0 ]; then result="SOME FAIL"
elif [ "$device" = cuda ] && [ $tq1_checked != 1 ]; then result="PASS, TQ1_0 NOT RUN"
else result="ALL PASS"; fi
echo "== result: $result"
exit $status
