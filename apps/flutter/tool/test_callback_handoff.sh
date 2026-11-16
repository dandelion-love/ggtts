#!/usr/bin/env bash
# Fake native audio only. This proves software handoff, not audible playback.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then echo "SKIP: Linux required"; exit 0; fi
command -v cc >/dev/null || { echo "SKIP: C compiler required"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
run_test() {
  GGTTS_LIB="$tmp/libfixture.so" LD_LIBRARY_PATH="$tmp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" GGTTS_HANDOFF_FIXTURE=1     timeout 35s "${FLUTTER:-flutter}" test --no-pub test/callback_handoff_integration_test.dart
}
cc -std=gnu11 -shared -fPIC -DPLANT_DROP_SECOND test/fixtures/callback_handoff_fixture.c -o "$tmp/libfixture.so"
ln -s libfixture.so "$tmp/libmp_audio_stream.so"
if run_test >"$tmp/plant.log" 2>&1; then
  echo "Dropped-piece plant unexpectedly passed" >&2
  exit 1
else
  status=$?
  if [[ $status -eq 124 || $status -eq 137 ]]; then
    cat "$tmp/plant.log" >&2
    echo "Dropped-piece plant timed out" >&2
    exit 1
  fi
fi
if ! grep -q 'callback samples dropped or reordered' "$tmp/plant.log"; then cat "$tmp/plant.log" >&2; exit 1; fi
echo "Dropped-piece plant detected"
cc -std=gnu11 -shared -fPIC test/fixtures/callback_handoff_fixture.c -o "$tmp/libfixture.so"
run_test
