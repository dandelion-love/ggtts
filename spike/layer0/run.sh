#!/usr/bin/env bash
# Builds both sides, runs layer 0 of the talker through each, and checks:
#   1. the Rust output is byte-identical to the C++ reference output;
#   2. every planted defect on the Rust side makes it differ (so the check can fail);
#   3. optionally (TIMING=1), 100 timed calls per side after warm-up, same thread count.
#
# Environment:
#   WORK           scratch directory for the model, the reference checkout and builds (default: ./work)
#   MODEL          a local copy of the talker GGUF; it is checked against MODEL_SHA256 either way
#   QWEN3_TTS_CPP  an existing qwen3-tts.cpp checkout at REF_COMMIT with its ggml submodule (default: cloned)
#   THREADS        CPU threads for both sides (default 4)
#   DEVICE         cpu (default) or cuda. cuda builds ggml with GGML_CUDA=ON in its own build directories and runs
#                  both sides on the GPU, with a CPU backend behind it as the reference does
#   CUDA_ARCH      CUDA architectures for DEVICE=cuda, cmake syntax (default 86)
#   CUDACXX        nvcc for DEVICE=cuda (default: on PATH, else the newest /usr/local/cuda-*/bin/nvcc)
#   GPU_WAIT_FILE  with DEVICE=cuda, wait while this file exists before using the GPU (e.g. a machine's busy marker)
#   TIMING=1       also time both sides
#   NICE           a prefix for the heavy commands, e.g. "nice -n 19"
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$here/work}
THREADS=${THREADS:-4}
NICE=${NICE:-}
DEVICE=${DEVICE:-cpu}
mkdir -p "$WORK"
case "$DEVICE" in cpu|cuda) ;; *) echo "DEVICE must be cpu or cuda" >&2; exit 2 ;; esac

# bindgen needs clang's resource headers (stdbool.h). A pip-wheel libclang ships without them, so point clang at the
# C compiler's own include directory unless the caller already set something.
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ] && command -v gcc >/dev/null 2>&1; then
    export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
fi

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

# Both sides: Release, static, position-independent, ggml's defaults for everything else. DEVICE=cuda adds
# GGML_CUDA=ON and builds into its own directories, so the CPU build is never overwritten.
build=$WORK/ref-build
cuda_args=(-DGGML_CUDA=OFF)
export CARGO_TARGET_DIR=$here/target
if [ "$DEVICE" = cuda ]; then
    nvcc=${CUDACXX:-$(command -v nvcc || ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)}
    [ -x "$nvcc" ] || { echo "DEVICE=cuda: no nvcc found; set CUDACXX" >&2; exit 1; }
    export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$nvcc")")}
    echo "nvcc: $nvcc ($("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')), architectures ${CUDA_ARCH:-86}"
    build=$WORK/ref-build-cuda
    cuda_args=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-86}")
    export CARGO_TARGET_DIR=$here/target-cuda GGML_CUDA=1
fi
$NICE cmake -S "$here/reference" -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON "${cuda_args[@]}" -DQWEN3_TTS_CPP="$QWEN3_TTS_CPP" > "$WORK/ref-config-$DEVICE.log"
$NICE cmake --build "$build" -j "$THREADS" > "$WORK/ref-build-$DEVICE.log"
# The Rust side links the reference build's own ggml archives and binds the reference tree's headers, so both
# executables carry the same compiled ggml and only the graph-building code differs.
export GGML_LIB_DIR=$build/ggml/src GGML_SRC=$QWEN3_TTS_CPP/ggml
echo "one ggml build for both sides ($GGML_LIB_DIR):"
(cd "$GGML_LIB_DIR" && sha256sum libggml.a libggml-cpu.a libggml-base.a)
[ "$DEVICE" = cuda ] && (cd "$GGML_LIB_DIR" && sha256sum ggml-cuda/libggml-cuda.a)
grep -E '^CXX_FLAGS' "$build/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make"
(cd "$here" && $NICE cargo build --release --locked -j "$THREADS")
echo "device: $DEVICE; threads: $THREADS on both sides"

ref=$build/layer0_ref
rust=$CARGO_TARGET_DIR/release/layer0
hidden=2048
dev=(--device "$DEVICE")
out=$WORK/$DEVICE
mkdir -p "$out"

if [ "$DEVICE" = cuda ] && [ -n "${GPU_WAIT_FILE:-}" ]; then
    while [ -e "$GPU_WAIT_FILE" ]; do echo "waiting: $GPU_WAIT_FILE exists"; sleep 60; done
fi

$NICE "$ref" "$MODEL" "$out/out-ref.bin" --threads "$THREADS" "${dev[@]}"
$NICE "$rust" "$MODEL" "$out/out-rust.bin" --threads "$THREADS" "${dev[@]}"
echo "rust vs reference ($DEVICE):"
python3 "$here/diff.py" "$out/out-ref.bin" "$out/out-rust.bin" "$hidden"

# Not a plant: with text-only positions (the same p in all three M-RoPE sections) ggml's M-RoPE rotates exactly as
# NEOX does, so this one is expected to stay identical. It is checked so a change in that equivalence is seen.
$NICE "$rust" "$MODEL" "$out/out-rope-neox.bin" --threads "$THREADS" "${dev[@]}" --plant rope-neox
echo "rope-neox (expected identical on text-only positions) vs reference:"
python3 "$here/diff.py" "$out/out-ref.bin" "$out/out-rope-neox.bin" "$hidden"

for plant in rope-normal swap-gate-up norm-order; do
    $NICE "$rust" "$MODEL" "$out/out-$plant.bin" --threads "$THREADS" "${dev[@]}" --plant "$plant"
    echo "planted $plant vs reference:"
    if python3 "$here/diff.py" "$out/out-ref.bin" "$out/out-$plant.bin" "$hidden"; then
        echo "the planted $plant was not seen" >&2
        exit 1
    fi
done
echo "red as required: every planted defect is seen"

# Informational: how far the GPU result is from the CPU one. Not a pass or fail; different kernels are expected to
# round differently. Shown only when a CPU run of the same WORK exists.
if [ "$DEVICE" = cuda ] && [ -f "$WORK/cpu/out-ref.bin" ]; then
    echo "informational: reference on cuda vs reference on cpu:"
    python3 "$here/diff.py" "$WORK/cpu/out-ref.bin" "$out/out-ref.bin" "$hidden" || true
fi

if [ "${TIMING:-0}" = 1 ]; then
    # Alternate the two sides so drift in machine load hits both.
    for round in 1 2 3; do
        $NICE "$ref" "$MODEL" "$WORK/t.bin" --threads "$THREADS" "${dev[@]}" --iters 100 --warmup 10 2>/dev/null
        $NICE "$rust" "$MODEL" "$WORK/t.bin" --threads "$THREADS" "${dev[@]}" --iters 100 --warmup 10 2>/dev/null
    done
fi
