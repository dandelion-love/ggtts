#!/usr/bin/env bash
# Builds the reference harnesses against a qwen3-tts.cpp checkout at 16bb5af and its CPU build directory.
#
#   QWEN3_TTS_CPP=<checkout> QWEN3_TTS_BUILD=<its cmake build dir, GGML_CUDA=OFF> reference/build.sh <out dir>
#
# The harnesses link the reference's own static libraries and the ggml archives inside its build, so the Rust side
# can link the very same ggml archives (GGML_LIB_DIR=$QWEN3_TTS_BUILD/ggml/src) and a difference can only come from
# the graph each side builds.
set -euo pipefail
src=${QWEN3_TTS_CPP:?set QWEN3_TTS_CPP to a qwen3-tts.cpp checkout}
build=${QWEN3_TTS_BUILD:?set QWEN3_TTS_BUILD to its CPU cmake build directory}
out=${1:?output directory}
here=$(cd "$(dirname "$0")" && pwd)
pin=16bb5afcd06311031c72a8488f8d59660dc2fb46
head=$(git -C "$src" rev-parse HEAD)
[ "$head" = "$pin" ] || { echo "qwen3-tts.cpp is at $head, not $pin" >&2; exit 2; }
ggml_head=$(git -C "$src/ggml" rev-parse HEAD)
[ "$ggml_head" = af97976c7810cdabb1863172f31c432dab767de7 ] || { echo "its ggml is at $ggml_head" >&2; exit 2; }
mkdir -p "$out"
libs=("$build/libaudio_tokenizer_decoder.a" "$build/ggml/src/libggml.a" "$build/ggml/src/libggml-cpu.a" "$build/ggml/src/libggml-base.a")
${CXX:-g++} -O2 -std=c++17 -I"$src/src" -I"$src/ggml/include" "$here/decode_ref.cpp" "${libs[@]}" \
  -fopenmp -lpthread -o "$out/decode_ref"
${CXX:-g++} -O2 -std=c++17 -I"$src/src" -I"$src/ggml/include" "$here/tokenize_ref.cpp" \n  "$build/libtext_tokenizer.a" "$build/ggml/src/libggml-base.a" -fopenmp -lpthread -o "$out/tokenize_ref"
echo "built $out/decode_ref $out/tokenize_ref"
