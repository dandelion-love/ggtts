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
    "$app/tool/build_native.sh" android
    cd "$app"
    flutter build apk --release --target-platform android-arm64,android-x64
    ;;
  *) echo "usage: $0 linux | android" >&2; exit 2 ;;
esac
