#!/usr/bin/env bash
# Build the C-ABI with its JNI layer for the host, compile the Kotlin binding and its test with kotlinc, and run the
# test on the host JVM. Needs cargo, a JDK and kotlinc on PATH.
#
#   scripts/kotlin_binding_test.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
cargo build --release --locked -p ggtts-ffi --features jni
target=${CARGO_TARGET_DIR:-$root/target}
case "$(uname -s)" in
    Linux*) lib="$target/release/libggtts_ffi.so" ;;
    Darwin*) lib="$target/release/libggtts_ffi.dylib" ;;
    *) lib="$target/release/ggtts_ffi.dll" ;;
esac
[ -f "$lib" ] || { echo "kotlin_binding_test: no library at $lib" >&2; exit 1; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
kotlinc -version 2>&1 | head -1
kotlinc android/ggtts/src/main/kotlin/ggtts/Ggtts.kt android/ggtts/src/test/kotlin/ggtts/BindingTest.kt \
    -d "$out/classes" 2>&1 | grep -v "^warning: unable to find kotlin-stdlib" || true
[ -f "$out/classes/ggtts/BindingTestKt.class" ] || { echo "kotlin_binding_test: compile failed" >&2; exit 1; }
stdlib=$(dirname "$(readlink -f "$(command -v kotlinc)")")/../lib/kotlin-stdlib.jar
java -Dggtts.lib="$lib" -cp "$out/classes:$stdlib" ggtts.BindingTestKt
