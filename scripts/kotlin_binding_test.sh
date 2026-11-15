#!/usr/bin/env bash
# Build the C-ABI with its JNI layer for the host, compile the Kotlin binding and its test with kotlinc, and run the
# test on the host JVM; then compile the documentation's Kotlin examples (docs/bindings/examples/kotlin) against the
# same class and run each on the fake engine, so a doc example that no longer compiles or runs is red. Needs cargo, a
# JDK and kotlinc on PATH.
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

# The demo app's binding seam (its request as an options document, android/demo/src/hostTest), compiled against the
# same binding and run on the fake engine: the app's documents meet the library here, not only a fake binding.
kotlinc -cp "$out/classes" android/demo/src/main/kotlin/ggtts/demo/Options.kt \
    android/demo/src/main/kotlin/ggtts/demo/TtsBinding.kt android/demo/src/hostTest/kotlin/ggtts/demo/AppBindingTest.kt \
    -d "$out/app" 2>&1 | grep -v "^warning: unable to find kotlin-stdlib" || true
[ -f "$out/app/ggtts/demo/AppBindingTestKt.class" ] || { echo "kotlin_binding_test: the app test did not compile" >&2; exit 1; }
java -Dggtts.lib="$lib" -cp "$out/app:$out/classes:$stdlib" ggtts.demo.AppBindingTestKt

# The documentation's examples, compiled against the binding's classes, then each run on the fake engine from an
# empty directory (FirstRender writes its WAV there).
kotlinc -cp "$out/classes" docs/bindings/examples/kotlin/*.kt -d "$out/examples" 2>&1 \
    | grep -v "^warning: unable to find kotlin-stdlib" || true
for main in firstrender.FirstRenderKt streaming.StreamKt options.OptionsKt; do
    class="$out/examples/$(echo "$main" | tr . /).class"
    [ -f "$class" ] || { echo "kotlin_binding_test: doc example $main did not compile" >&2; exit 1; }
    mkdir -p "$out/run-$main"
    (cd "$out/run-$main" && java -Dggtts.lib="$lib" -cp "$out/examples:$out/classes:$stdlib" "$main" fake) \
        || { echo "kotlin_binding_test: doc example $main failed on the fake engine" >&2; exit 1; }
done
[ -s "$out/run-firstrender.FirstRenderKt/first_render.wav" ] \
    || { echo "kotlin_binding_test: FirstRender wrote no first_render.wav" >&2; exit 1; }
echo "kotlin_binding_test: the doc examples compiled and ran on the fake engine"
