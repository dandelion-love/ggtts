#!/usr/bin/env bash
# CPU-only fake device seam; only miniaudio null backend is compiled.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
src=vendor/mp_audio_stream
${CXX:-c++} -std=c++17 -O1 -pthread -DMA_ENABLE_ONLY_SPECIFIC_BACKENDS -DMA_ENABLE_NULL \
  -DGGTTS_TEST_ANDROID_FALLBACK "$src/src/mp_audio_stream.cc" \
  "$src/ggtts_tests/android_fallback_fake_test.cc" -o "$tmp/test" -ldl -lm
ulimit -c 0
if timeout --kill-after=5s 20s "$tmp/test" --plant >"$tmp/plant.log" 2>&1; then
  echo "Fallback plant unexpectedly passed" >&2
  exit 1
else
  status=$?
  if [[ $status -ne 134 ]]; then
    cat "$tmp/plant.log" >&2
    echo "Fallback plant failed for an unexpected reason: $status" >&2
    exit 1
  fi
fi
timeout --kill-after=5s 20s "$tmp/test"
