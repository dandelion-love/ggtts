#!/usr/bin/env bash
# Builds the reference harnesses against a qwen3-tts.cpp checkout at 16bb5af and its static build directory.
#
#   QWEN3_TTS_CPP=<checkout> QWEN3_TTS_BUILD=<its cmake build dir> reference/build.sh <out dir>
#
# The harnesses link the reference's own static libraries and the ggml archives inside its build, so the Rust side
# can link the very same ggml archives (GGML_LIB_DIR=$QWEN3_TTS_BUILD/ggml/src) and a difference can only come from
# the graph each side builds. A CUDA build (QWEN3_TTS_CUDA=ON) is detected by its libggml-cuda.a and linked with the
# CUDA runtime and cuBLAS from CUDA_HOME (default /usr/local/cuda).
set -euo pipefail
src=${QWEN3_TTS_CPP:?set QWEN3_TTS_CPP to a qwen3-tts.cpp checkout}
build=${QWEN3_TTS_BUILD:?set QWEN3_TTS_BUILD to its cmake build directory}
out=${1:?output directory}
here=$(cd "$(dirname "$0")" && pwd)
pin=16bb5afcd06311031c72a8488f8d59660dc2fb46
head=$(git -C "$src" rev-parse HEAD)
[ "$head" = "$pin" ] || { echo "qwen3-tts.cpp is at $head, not $pin" >&2; exit 2; }
ggml_head=$(git -C "$src/ggml" rev-parse HEAD)
[ "$ggml_head" = af97976c7810cdabb1863172f31c432dab767de7 ] || { echo "its ggml is at $ggml_head" >&2; exit 2; }
mkdir -p "$out"

ggml=("$build/ggml/src/libggml.a")
extra=()
if [ -f "$build/ggml/src/ggml-cuda/libggml-cuda.a" ]; then
  cuda=${CUDA_HOME:-/usr/local/cuda}
  ggml+=("$build/ggml/src/ggml-cuda/libggml-cuda.a")
  extra=(-L"$cuda/lib64" -L"$cuda/lib64/stubs" -L/usr/lib/wsl/lib -lcudart -lcublas -lcublasLt -lcuda)
fi
ggml+=("$build/ggml/src/libggml-cpu.a" "$build/ggml/src/libggml-base.a")

${CXX:-g++} -O2 -std=c++17 -I"$src/src" -I"$src/ggml/include" "$here/decode_ref.cpp" \
  "$build/libaudio_tokenizer_decoder.a" "${ggml[@]}" "${extra[@]}" -fopenmp -lpthread -o "$out/decode_ref"
${CXX:-g++} -O2 -std=c++17 -I"$src/src" -I"$src/ggml/include" "$here/tokenize_ref.cpp" \
  "$build/libtext_tokenizer.a" "$build/ggml/src/libggml-base.a" -fopenmp -lpthread -o "$out/tokenize_ref"
echo "built $out/decode_ref $out/tokenize_ref"
