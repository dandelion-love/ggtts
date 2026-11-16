#!/usr/bin/env bash
# Actual CPU Qwen and Dart isolates; fake output only, no audio device.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$app/../.." && pwd)
cd "$app"
: "${GGTTS_LIB:?set to CPU ggml libggtts_ffi.so}"
: "${GGTTS_TINY_TALKER:?set to pinned tiny talker}"
: "${GGTTS_CODEC:?set to pinned codec}"
source_file="$repo/crates/ggtts-qwen3-tts/fixtures/tiny/SOURCE"
value() { sed -n "s/^$1=//p" "$source_file"; }
printf '%s  %s\n' "$(value CUT_SHA256)" "$GGTTS_TINY_TALKER" | sha256sum -c -
printf '%s  %s\n' "$(value CODEC_SHA256)" "$GGTTS_CODEC" | sha256sum -c -
fixture_dir=$(mktemp -d)
trap 'rm -rf -- "$fixture_dir"' EXIT
cc -std=gnu11 -shared -fPIC -pthread test/fixtures/read_native_fixture.c -o "$fixture_dir/libmp_audio_stream.so"
GGTTS_FAKE_AUDIO="$fixture_dir/libmp_audio_stream.so" LD_LIBRARY_PATH="$fixture_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" GGTTS_TINY_READ_PROBE=1   timeout 110s "${FLUTTER:-flutter}" test --no-pub test/read_tiny_qwen_integration_test.dart
