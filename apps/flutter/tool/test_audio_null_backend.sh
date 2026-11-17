#!/usr/bin/env bash
# CPU-only integration: compile ONLY the miniaudio null backend; never opens physical audio.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
src=vendor/mp_audio_stream
# Exercise the ordinary production guard with null as the only compiled backend.
# This binary does not define the callback-test exception.
${CXX:-c++} -std=c++17 -O1 -pthread -DMA_ENABLE_ONLY_SPECIFIC_BACKENDS -DMA_ENABLE_NULL \
  "$src/src/mp_audio_stream.cc" "$src/ggtts_tests/null_backend_refusal_test.cc" \
  -o "$tmp/refusal" -ldl -lm
timeout --kill-after=5s 20s "$tmp/refusal"
${CXX:-c++} -std=c++17 -O1 -pthread -DMA_ENABLE_ONLY_SPECIFIC_BACKENDS -DMA_ENABLE_NULL \
  -DGGTTS_TEST_OBSERVE_CALLBACK "$src/src/mp_audio_stream.cc" \
  "$src/ggtts_tests/null_backend_test.cc" -o "$tmp/test" -ldl -lm
ulimit -c 0
# Verify the callback sample oracle detects a deliberately inserted silence.
if timeout --kill-after=5s 20s "$tmp/test" --plant-silence >"$tmp/plant.log" 2>&1; then
  echo "Silence plant unexpectedly passed" >&2
  exit 1
else
  status=$?
  if [[ $status -ne 134 ]]; then
    cat "$tmp/plant.log" >&2
    echo "Silence plant failed for an unexpected reason: $status" >&2
    exit 1
  fi
fi
timeout --kill-after=5s 20s "$tmp/test"
