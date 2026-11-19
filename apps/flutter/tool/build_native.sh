#!/usr/bin/env bash
# Build the ggtts C library (crates/ggtts-ffi, over ggml) for the Flutter spike and put it where the app loads it.
#
#   apps/flutter/tool/build_native.sh linux     # -> apps/flutter/native/linux-x64/libggtts_ffi.so
#   apps/flutter/tool/build_native.sh android   # -> apps/flutter/android/app/src/main/jniLibs/{arm64-v8a,x86_64}/
#
# android needs the NDK (ANDROID_NDK_HOME, 27.2.12479018 as scripts/android_build.sh) and cargo-ndk. The JNI entry
# points are not built: the Flutter app calls the C ABI through dart:ffi, never through Kotlin.
set -euo pipefail

app=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$app/../.." && pwd)
features=ggtts-ffi/ggml
say() { echo "build_native: $*" >&2; }

linux() {
    local tdir="$root/target/flutter-linux"
    (cd "$root" && cargo build --release --locked -p ggtts-ffi --features "$features" --target-dir "$tdir")
    mkdir -p "$app/native/linux-x64"
    cp "$tdir/release/libggtts_ffi.so" "$app/native/linux-x64/"
    say "linux: $app/native/linux-x64/libggtts_ffi.so"
}

android() {
    local api=${ANDROID_API:-24}
    : "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the NDK}"
    # CMake (ggml's build) finds the NDK through ANDROID_NDK_ROOT, not cargo-ndk's ANDROID_NDK_HOME.
    export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-$ANDROID_NDK_HOME}
    # ggml has committed bindings for x86_64 Linux only, so Android regenerates them with bindgen, which needs a host
    # libclang: set LIBCLANG_PATH to the directory holding libclang.so (any recent LLVM).
    [[ -n "${LIBCLANG_PATH:-}" ]] || say "LIBCLANG_PATH unset: bindgen looks for libclang in the system paths"
    local sysroot="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--sysroot=$sysroot --target=aarch64-linux-android$api"
    export BINDGEN_EXTRA_CLANG_ARGS_x86_64_linux_android="--sysroot=$sysroot --target=x86_64-linux-android$api"
    # 16 KB pages, as scripts/android_build.sh.
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="${CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS:-} -C link-arg=-Wl,-z,max-page-size=16384"
    export CARGO_TARGET_X86_64_LINUX_ANDROID_RUSTFLAGS="${CARGO_TARGET_X86_64_LINUX_ANDROID_RUSTFLAGS:-} -C link-arg=-Wl,-z,max-page-size=16384"
    local tdir="$root/target/flutter-android"
    (cd "$root" && cargo ndk -t arm64-v8a -t x86_64 --platform "$api" build --release --locked -p ggtts-ffi -p ggtts-ggml-sys \
        --features "$features,ggtts-ggml-sys/bindgen" --target-dir "$tdir")
    local abi target out
    for abi in arm64-v8a x86_64; do
        case "$abi" in
            arm64-v8a) target=aarch64-linux-android ;;
            x86_64) target=x86_64-linux-android ;;
        esac
        out="$app/android/app/src/main/jniLibs/$abi"
        mkdir -p "$out"
        cp "$tdir/$target/release/libggtts_ffi.so" "$out/"
        cp "$sysroot/usr/lib/$target/libc++_shared.so" "$out/"
        say "android: $out"
    done
}

case "${1:-}" in
    linux) linux ;;
    android) android ;;
    *) echo "usage: $0 linux | android" >&2; exit 2 ;;
esac
