#!/usr/bin/env bash
# Build and bundle the engine and Flutter app for the selected platform.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
case "${1:-}" in
  linux)
    "$app/tool/build_native.sh" linux
    cd "$app"
    flutter build linux --release
    ;;
  android)
    python3 "$app/tool/test_check_android_apk.py"
    "$app/tool/build_native.sh" android
    cd "$app"
    flutter build apk --release --target-platform android-arm64,android-x64
    : "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the NDK used to build the native library}"
    llvm="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
    python3 "$app/tool/check_android_apk.py" "$app/build/app/outputs/flutter-apk/app-release.apk" \
      "$app/android/app/src/main/jniLibs" "$llvm/llvm-nm" "$llvm/llvm-strip"
    ;;
  *) echo "usage: $0 linux | android" >&2; exit 2 ;;
esac
