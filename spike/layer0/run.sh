#!/usr/bin/env bash
# Builds both sides on CPU, runs layer 0 of the talker through each, and checks:
#   1. the Rust output is byte-identical to the C++ reference output;
#   2. every planted defect on the Rust side makes it differ (so the check can fail);
#   3. optionally (TIMING=1), 100 timed calls per side after warm-up, same thread count.
#
# Environment:
#   WORK           scratch directory for the model, the reference checkout and builds (default: ./work)
#   MODEL          a local copy of the talker GGUF; it is checked against MODEL_SHA256 either way
#   QWEN3_TTS_CPP  an existing qwen3-tts.cpp checkout at REF_COMMIT with its ggml submodule (default: cloned)
#   THREADS        CPU threads for both sides (default 4)
#   TIMING=1       also time both sides
#   NICE           a prefix for the heavy commands, e.g. "nice -n 19"
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$here/work}
THREADS=${THREADS:-4}
NICE=${NICE:-}
mkdir -p "$WORK"

# The published talker: Hugging Face jmtl/hk-cantonese-cc0-qwen3-tts-GGUF, tag v0.0.1 (commit below), CC-0 data.
MODEL_URL=https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/resolve/5a7660276b52d9efeed92d1c87f218d7a296110a/qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf
MODEL_SHA256=33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67
REF_REPO=https://github.com/Danmoreng/qwen3-tts.cpp.git
REF_COMMIT=16bb5afcd06311031c72a8488f8d59660dc2fb46
GGML_COMMIT=af97976c7810cdabb1863172f31c432dab767de7

if [ -z "${MODEL:-}" ]; then
    MODEL=$WORK/qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf
    [ -f "$MODEL" ] || curl -fL --retry 3 -o "$MODEL" "$MODEL_URL"
fi
echo "$MODEL_SHA256  $MODEL" | sha256sum -c -

if [ -z "${QWEN3_TTS_CPP:-}" ]; then
    QWEN3_TTS_CPP=$WORK/qwen3-tts.cpp
    if [ ! -d "$QWEN3_TTS_CPP/.git" ]; then
        git clone --quiet "$REF_REPO" "$QWEN3_TTS_CPP"
    fi
    git -C "$QWEN3_TTS_CPP" checkout --quiet "$REF_COMMIT"
    git -C "$QWEN3_TTS_CPP" submodule update --init --quiet ggml
fi
test "$(git -C "$QWEN3_TTS_CPP" rev-parse HEAD)" = "$REF_COMMIT"
ref_ggml=$(git -C "$QWEN3_TTS_CPP/ggml" rev-parse HEAD)
our_ggml=$(git -C "$here/ggml-sys/ggml" rev-parse HEAD)
echo "ggml: reference vendors $ref_ggml, ggml-sys pins $our_ggml"
if [ "$ref_ggml" != "$GGML_COMMIT" ] || [ "$our_ggml" != "$GGML_COMMIT" ]; then
    echo "ggml commits differ from $GGML_COMMIT; a bit-for-bit comparison would compare two ggmls" >&2
    exit 1
fi

# Both sides: Release, static, position-independent, CPU only, ggml's defaults for everything else.
$NICE cmake -S "$here/reference" -B "$WORK/ref-build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_CUDA=OFF -DQWEN3_TTS_CPP="$QWEN3_TTS_CPP" > "$WORK/ref-config.log"
$NICE cmake --build "$WORK/ref-build" -j "$THREADS" > "$WORK/ref-build.log"
# The Rust side links the reference build's own ggml archives and binds the reference tree's headers, so both
# executables carry the same compiled ggml and only the graph-building code differs.
export GGML_LIB_DIR=$WORK/ref-build/ggml/src GGML_SRC=$QWEN3_TTS_CPP/ggml
echo "one ggml build for both sides ($GGML_LIB_DIR):"
(cd "$GGML_LIB_DIR" && sha256sum libggml.a libggml-cpu.a libggml-base.a)
grep -E '^CXX_FLAGS' "$WORK/ref-build/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make"
(cd "$here" && $NICE cargo build --release --locked -j "$THREADS")
echo "threads: $THREADS on both sides"

ref=$WORK/ref-build/layer0_ref
rust=$here/target/release/layer0
hidden=2048

$NICE "$ref" "$MODEL" "$WORK/out-ref.bin" --threads "$THREADS"
$NICE "$rust" "$MODEL" "$WORK/out-rust.bin" --threads "$THREADS"
echo "rust vs reference:"
python3 "$here/diff.py" "$WORK/out-ref.bin" "$WORK/out-rust.bin" "$hidden"

# Not a plant: with text-only positions (the same p in all three M-RoPE sections) ggml's M-RoPE rotates exactly as
# NEOX does, so this one is expected to stay identical. It is checked so a change in that equivalence is seen.
$NICE "$rust" "$MODEL" "$WORK/out-rope-neox.bin" --threads "$THREADS" --plant rope-neox
echo "rope-neox (expected identical on text-only positions) vs reference:"
python3 "$here/diff.py" "$WORK/out-ref.bin" "$WORK/out-rope-neox.bin" "$hidden"

for plant in rope-normal swap-gate-up norm-order; do
    $NICE "$rust" "$MODEL" "$WORK/out-$plant.bin" --threads "$THREADS" --plant "$plant"
    echo "planted $plant vs reference:"
    if python3 "$here/diff.py" "$WORK/out-ref.bin" "$WORK/out-$plant.bin" "$hidden"; then
        echo "the planted $plant was not seen" >&2
        exit 1
    fi
done
echo "red as required: every planted defect is seen"

if [ "${TIMING:-0}" = 1 ]; then
    # Alternate the two sides so drift in machine load hits both.
    for round in 1 2 3; do
        $NICE "$ref" "$MODEL" "$WORK/t.bin" --threads "$THREADS" --iters 100 --warmup 10 2>/dev/null
        $NICE "$rust" "$MODEL" "$WORK/t.bin" --threads "$THREADS" --iters 100 --warmup 10 2>/dev/null
    done
fi
