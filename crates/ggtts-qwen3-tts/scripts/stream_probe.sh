#!/usr/bin/env bash
# Streaming timings on one device: time to first audio and real-time factor for the chunk-level story stream, and the
# frame-level decoder's cost against the batch decode. CPU by default; DEVICE=cuda for the GPU.
#
#   TALKER=<talker.gguf> CODEC=<codec.gguf> WORK=<scratch dir> [DEVICE=cuda] scripts/stream_probe.sh
#
# Optional: THREADS (4), CUDA_HOME (/usr/local/cuda), GGML_CUDA_ARCH (86), CARGO_TARGET_DIR (WORK/target),
# NICE ("nice -n 19"), GPU_WAIT_FILE (wait while this file exists, before any GPU work), STORY (the committed
# fixtures/story/outing.txt).
#
# Steps: (1) build ggtts-render and the frame-level test for the device; (2) render the story twice, batch and with
# --stream, and print the render time, the audio length, the real-time factor and the time to first audio, and check
# that the two FLACs are byte-identical; (3) run tests/stream_identity.rs on the device and print its table (on CUDA
# it reports and does not gate). Exit status 0 only when the FLACs match and every step ran.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
: "${TALKER:?}" "${CODEC:?}" "${WORK:?}"
device=${DEVICE:-cpu}
threads=${THREADS:-4}
nice=${NICE:-nice -n 19}
story=${STORY:-$here/fixtures/story/outing.txt}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$WORK/target}
case $device in
  cpu) features=ggml ;;
  cuda)
    features=cuda
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export PATH=$CUDA_HOME/bin:$PATH
    ;;
  *) echo "DEVICE must be cpu or cuda, not $device" >&2; exit 2 ;;
esac
mkdir -p "$WORK"
status=0

# Build first, so the GPU wait covers only GPU work.
echo "== build ($device)"
(cd "$root" && $nice cargo build -j 4 --release --locked -p ggtts-cli --features "ggtts-cli/$features" \
  && $nice cargo test -j 4 --release --locked -p ggtts-qwen3-tts --features "$features plant" --test stream_identity --no-run) \
  > "$WORK/build.log" 2>&1 || { echo "build failed: $WORK/build.log" >&2; exit 2; }
bin=$CARGO_TARGET_DIR/release/ggtts-render

if [ -n "${GPU_WAIT_FILE:-}" ]; then
  while [ -e "$GPU_WAIT_FILE" ]; do echo "waiting: $GPU_WAIT_FILE exists"; sleep 60; done
fi
if [ "$device" = cuda ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | sed 's/^/device: /'
fi
sha256sum "$TALKER" "$CODEC" | awk '{print "model: " substr($1, 1, 16) "  " $2}'

render() { # name extra-args...
  local name=$1
  shift
  /usr/bin/time -f "%e" -o "$WORK/$name.wall" $nice "$bin" --model "$TALKER" --codec "$CODEC" -j "$threads" \
    --device "$device" --story "$story" --out "$WORK/$name.flac" "$@" 2> "$WORK/$name.log"
}

echo "== story, chunk-level stream ($story)"
render batch || { echo "batch render failed: $WORK/batch.log"; exit 1; }
render stream --stream "$WORK/stream.wav" || { echo "stream render failed: $WORK/stream.log"; exit 1; }
python3 - "$WORK" <<'PY' || status=1
import re, sys
work = sys.argv[1]
def grab(name, pat):
    m = re.search(pat, open(f"{work}/{name}.log").read())
    return float(m.group(1)) if m else float("nan")
for name in ("batch", "stream"):
    render_ms = grab(name, r"render (\d+) ms")
    samples = grab(name, r"\((\d+) samples at")
    rate = grab(name, r"samples at (\d+) Hz")
    audio = samples / rate
    line = f"  {name}: render {render_ms / 1e3:.2f} s for {audio:.2f} s of audio, RTF {render_ms / 1e3 / audio:.3f}"
    if name == "stream":
        first = grab(name, r"first audio at (\d+) ms") / 1e3
        line += f", first audio at {first:.2f} s"
    print(line + f", wall {open(f'{work}/{name}.wall').read().strip()} s")
PY
if cmp -s "$WORK/batch.flac" "$WORK/stream.flac"; then
  echo "  FLAC identical with and without --stream"
else
  echo "  FLAC DIFFERS with --stream"
  status=1
fi
grep -h "level gain" "$WORK/stream.log" | sed 's/^/  /'

echo "== frame-level decoder on $device (tests/stream_identity.rs)"
(cd "$root" && GGTTS_DEVICE=$device GGTTS_CODEC=$CODEC $nice cargo test -j 4 --release --locked -p ggtts-qwen3-tts \
  --features "$features plant" --test stream_identity -- --ignored --nocapture) > "$WORK/frame.log" 2>&1 || status=1
grep -E "^(\S+) +n=|test result" "$WORK/frame.log" || { echo "no result: $WORK/frame.log"; status=1; }

echo "stream probe ($device): exit $status (logs in $WORK)"
exit $status
