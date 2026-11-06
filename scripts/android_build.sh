#!/usr/bin/env bash
# Build ggtts for Android arm64 with cargo-ndk, and check what was built: ggml at the pin, NEON in the kernels.
#
#   scripts/android_build.sh build [OUT_DIR]   # build ggtts-render for aarch64-linux-android, check it, write
#                                              # OUT_DIR (default dist/android-arm64): the binary and BUILDINFO.json
#   scripts/android_build.sh check-pin PIN     # refuse unless the ggml submodule is at PIN (the planted case)
#   scripts/android_build.sh check-neon BIN    # refuse unless BIN's ggml dot product uses NEON vector instructions
#
# Needs the NDK (ANDROID_NDK_HOME) and cargo-ndk. The NEON check reads the disassembly of ggml's Q8_0 dot product,
# the kernel the engines spend their time in, with the NDK's llvm-objdump: NEON's vector registers (`v0.16b` and the
# like) appear only when the build used NEON. `GGML_CPU_ARM_ARCH=armv8-a+nosimd` builds it without, which the check
# must refuse (CI plants that).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
say() { echo "android_build: $*" >&2; }
target=aarch64-linux-android
api=${ANDROID_API:-24}

pin() { sed -n 's/^pub const GGML_COMMIT: &str = "\([0-9a-f]\{40\}\)";$/\1/p' "$root/crates/ggtts-ggml-sys/src/lib.rs"; }

check_pin() { # EXPECTED
    local head
    head=$(git -C "$root/crates/ggtts-ggml-sys/ggml" rev-parse HEAD)
    if [ "$head" != "$1" ]; then
        say "ggml is at $head, not the pinned $1"
        return 1
    fi
    say "ggml at the pin $1"
}

objdump_bin() {
    local d
    d=$(ls -d "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"/toolchains/llvm/prebuilt/*/bin | head -1)
    echo "$d/llvm-objdump"
}

check_neon() { # BIN
    local n
    n=$("$(objdump_bin)" -d --no-show-raw-insn "$1" \
        | awk '/<ggml_vec_dot_q8_0_q8_0>:/{on=1; next} on && /^$/{exit} on' \
        | grep -cE 'v[0-9]+\.(16b|8h|4s|2d)' || true)
    if [ "${n:-0}" -eq 0 ]; then
        say "no NEON vector instructions in ggml_vec_dot_q8_0_q8_0 in $1"
        return 1
    fi
    say "NEON in ggml_vec_dot_q8_0_q8_0: $n vector instructions"
}

build() { # OUT_DIR
    local out=${1:-dist/android-arm64}
    check_pin "$(pin)"
    (cd "$root" && cargo ndk -t arm64-v8a --platform "$api" build --release --locked -p ggtts-cli -p ggtts-ggml-sys \
        --features ggtts-cli/ggml,ggtts-ggml-sys/bindgen)
    local bin="$root/target/$target/release/ggtts-render"
    check_neon "$bin"
    mkdir -p "$out"
    cp "$bin" "$out/"
    local commit describe
    commit=$(pin)
    describe=$(sed -n 's/^ *pub const DESCRIBE: &str = "\(.*\)";$/\1/p' "$root/crates/ggtts-core/src/lib.rs")
    cat > "$out/BUILDINFO.json" <<EOF
{
  "ggtts_commit": "$(git -C "$root" rev-parse HEAD)",
  "ggml_commit": "$commit",
  "ggml_describe": "$describe",
  "rustc": "$(rustc -V)",
  "target": "$target",
  "android_api": $api,
  "ndk": "$(basename "$ANDROID_NDK_HOME")",
  "features": ["cpu", "ggml", "neon", "qwen3-tts", "kitten-tts-2"]
}
EOF
    say "built $out"
}

case "${1:-}" in
    build) build "${2:-}" ;;
    check-pin) check_pin "$2" ;;
    check-neon) check_neon "$2" ;;
    *) echo "usage: $0 build [OUT_DIR] | check-pin PIN | check-neon BIN" >&2; exit 2 ;;
esac
