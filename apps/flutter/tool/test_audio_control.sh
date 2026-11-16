#!/usr/bin/env bash
# Deterministic backend control tests; no audio device is opened.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
test_bin=$(mktemp "${TMPDIR:-/tmp}/ggtts-audio-control.XXXXXX")
trap 'rm -f "$test_bin"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -pthread \
  "$app/vendor/mp_audio_stream/ggtts_tests/control_test.cc" -o "$test_bin"
"$test_bin"
echo "audio control tests passed"
