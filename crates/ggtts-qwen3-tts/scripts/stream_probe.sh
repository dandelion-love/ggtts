#!/usr/bin/env bash
# Streaming timings on one device: time to first audio and real-time factor for the chunk-level story stream, and,
# for Qwen3-TTS, the frame-level decoder's cost against the batch decode. CPU by default; DEVICE=cuda for the GPU.
#
#   TALKER=<model.gguf> CODEC=<codec.gguf> WORK=<scratch dir> [DEVICE=cuda] scripts/stream_probe.sh
#   ENGINE=kitten-tts-2 TALKER=<lm.gguf> CODEC=<s3gen.gguf> VOICE=<voice.json> WORK=<dir> scripts/stream_probe.sh
#
# FRAME=0 skips the frame-level table (Qwen only; about 20 minutes on CPU).
#
# Four story renders: batch; --stream --stream-raw (the parity stream: the FLAC must be identical); --stream at
# --stream-level-seconds 0 (the first-chunk gain, for the latency the default hold costs); and --stream
# (the listener's levelled stream: its loudness against the FLAC's, declared within 3 dB, and its clamped samples
# against the FLAC's, declared at most max(16, 0.01%) more).
#
# ENGINE is qwen3-tts (the default) or kitten-tts-2. For Kitten, VOICE (a prepared reference voice) is required, the
# story defaults to ../ggtts-kitten/fixtures/story/lighthouse.txt, and the frame-level step is skipped: Kitten
# streams at chunk level only.
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
engine=${ENGINE:-qwen3-tts}
case $engine in
  qwen3-tts)
    story=${STORY:-$here/fixtures/story/outing.txt}
    voice_args=()
    ;;
  kitten-tts-2)
    : "${VOICE:?kitten-tts-2 needs VOICE, a prepared reference voice}"
    story=${STORY:-$root/crates/ggtts-kitten/fixtures/story/lighthouse.txt}
    voice_args=(--voice "$VOICE")
    ;;
  *) echo "ENGINE must be qwen3-tts or kitten-tts-2, not $engine" >&2; exit 2 ;;
esac
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
  /usr/bin/time -f "%e" -o "$WORK/$name.wall" $nice "$bin" --engine "$engine" --model "$TALKER" --codec "$CODEC" \
    -j "$threads" --device "$device" "${voice_args[@]}" --story "$story" --out "$WORK/$name.flac" "$@" \
    2> "$WORK/$name.log"
}

echo "== $engine on $device: story, chunk-level stream ($story)"
render batch || { echo "batch render failed: $WORK/batch.log"; exit 1; }
# The parity run asks for the raw stream by name; the listener's levelled stream is a third run.
render stream --stream "$WORK/stream.wav" --stream-raw || { echo "stream render failed: $WORK/stream.log"; exit 1; }
render level --stream "$WORK/level.wav" || { echo "levelled stream render failed: $WORK/level.log"; exit 1; }
# The first-chunk mode, for the latency the default hold costs (#44).
render level0 --stream "$WORK/level0.wav" --stream-level-seconds 0 \
  || { echo "first-chunk stream render failed: $WORK/level0.log"; exit 1; }
python3 - "$WORK" <<'PY' || status=1
import re, sys
work = sys.argv[1]
def grab(name, pat):
    m = re.search(pat, open(f"{work}/{name}.log").read())
    return float(m.group(1)) if m else float("nan")
for name in ("batch", "stream", "level", "level0"):
    render_ms = grab(name, r"render (\d+) ms")
    samples = grab(name, r"\((\d+) samples at")
    rate = grab(name, r"samples at (\d+) Hz")
    audio = samples / rate
    line = f"  {name}: render {render_ms / 1e3:.2f} s for {audio:.2f} s of audio, RTF {render_ms / 1e3 / audio:.3f}"
    if name != "batch":
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
grep -h "level gain" "$WORK/stream.log" "$WORK/level.log" "$WORK/level0.log" | sed 's/^/  /'
# The listener's stream against the FLAC: loudness (RMS over samples above the level floor, 1e-3) and peak. The
# bound declared for it is 3 dB; ffmpeg decodes the FLAC, the WAV is read directly.
python3 - "$WORK" <<'PY' || status=1
import math, struct, subprocess, sys
work = sys.argv[1]
def pcm16(b):
    return struct.unpack(f"<{len(b) // 2}h", b[: len(b) // 2 * 2])
wav = open(f"{work}/level.wav", "rb").read()[44:]
flac = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{work}/batch.flac", "-f", "s16le", "-"],
                      capture_output=True, check=True).stdout
def loud_db(x):
    v = [s / 32767 for s in x if abs(s / 32767) > 1e-3]
    return 10 * math.log10(sum(a * a for a in v) / len(v))
s, f = pcm16(wav), pcm16(flac)
d = loud_db(s) - loud_db(f)
# Clamped samples (declared on the issue before measuring): |s| >= round(0.999 * 32767), the stream's count at most
# the FLAC's plus max(16, 0.01% of the samples).
top = round(0.999 * 32767)
cs, cf = sum(1 for v in s if abs(v) >= top), sum(1 for v in f if abs(v) >= top)
allow = max(16, len(s) // 10000)
print(f"  levelled stream {loud_db(s):.2f} dB vs FLAC {loud_db(f):.2f} dB: {d:+.2f} dB "
      f"({'within' if abs(d) <= 3 else 'OVER'} the declared 3 dB), peak {max(abs(v) for v in s) / 32767:.4f}, "
      f"clamped {cs} vs FLAC {cf} ({'within' if cs <= cf + allow else 'OVER'} the allowance {allow})")
# The clamp count shown to see a defect: the same stream at +6 dB (as if the first chunk's gain were 6 dB too high)
# must be over the allowance wherever the stream already reaches near the clip.
cp = sum(1 for v in s if abs(min(32767, max(-32768, 2 * v))) >= top)
near = max(abs(v) for v in s) >= top // 2
print(f"  +6 dB plant on the stream: clamped {cp}: "
      + (("red, as it must be" if cp > cf + allow else "NOT RED") if near
         else "cannot clamp here (the stream peaks under half the clip)"))
ok = abs(d) <= 3 and cs <= cf + allow and (cp > cf + allow or not near)
sys.exit(0 if ok else 1)
PY

if [ "$engine" != qwen3-tts ] || [ "${FRAME:-1}" = 0 ]; then
  echo "stream probe ($engine, $device): exit $status (logs in $WORK)"
  exit $status
fi
echo "== frame-level decoder on $device (tests/stream_identity.rs)"
(cd "$root" && GGTTS_DEVICE=$device GGTTS_CODEC=$CODEC $nice cargo test -j 4 --release --locked -p ggtts-qwen3-tts \
  --features "$features plant" --test stream_identity -- --ignored --nocapture) > "$WORK/frame.log" 2>&1 || status=1
grep -E "^(\S+) +n=|test result" "$WORK/frame.log" || { echo "no result: $WORK/frame.log"; status=1; }

echo "stream probe ($device): exit $status (logs in $WORK)"
exit $status
