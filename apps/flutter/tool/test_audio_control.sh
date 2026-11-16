#!/usr/bin/env bash
# Deterministic backend control tests; no audio device is opened.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
test_bin=$(mktemp "${TMPDIR:-/tmp}/ggtts-audio-control.XXXXXX")
shared_lib="${test_bin}.so"
trap 'rm -f "$test_bin" "$shared_lib"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -pthread \
  "$app/vendor/mp_audio_stream/ggtts_tests/control_test.cc" -o "$test_bin"
"$test_bin"
echo "audio control tests passed"

"${CXX:-c++}" -std=c++17 -fPIC -shared -pthread \
  "$app/vendor/mp_audio_stream/src/mp_audio_stream.cc" -o "$shared_lib" -ldl -lm
python3 "$app/vendor/mp_audio_stream/ggtts_tests/session_wrapper_test.py" "$shared_lib"
