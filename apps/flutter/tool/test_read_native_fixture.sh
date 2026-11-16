#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then echo "SKIP: native fixture requires Linux"; exit 0; fi
if ! command -v cc >/dev/null; then echo "SKIP: native fixture requires cc"; exit 0; fi
fixture_dir=$(mktemp -d)
trap 'rm -rf -- "$fixture_dir"' EXIT
cc -std=gnu11 -shared -fPIC -pthread test/fixtures/read_native_fixture.c -o "$fixture_dir/libggtts_fixture.so"
ln -s libggtts_fixture.so "$fixture_dir/libmp_audio_stream.so"
GGTTS_LIB="$fixture_dir/libggtts_fixture.so" LD_LIBRARY_PATH="$fixture_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" GGTTS_READ_NATIVE_FIXTURE=1   timeout 30s "${FLUTTER:-flutter}" test --no-pub test/read_native_integration_test.dart

cc -std=gnu11 -DFAKE_ABI=5 -shared -fPIC -pthread test/fixtures/read_native_fixture.c -o "$fixture_dir/libggtts_fixture.so"
GGTTS_LIB="$fixture_dir/libggtts_fixture.so" LD_LIBRARY_PATH="$fixture_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" GGTTS_READ_NATIVE_FIXTURE=1 GGTTS_FIXTURE_ABI=5   timeout 30s "${FLUTTER:-flutter}" test --no-pub test/read_native_integration_test.dart
