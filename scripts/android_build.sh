#!/usr/bin/env bash
# Build ggtts for Android arm64 with cargo-ndk, and check what was built: ggml at the pin, NEON in the kernels, and
# every shared library the binary needs either on the device or shipped beside it.
#
#   scripts/android_build.sh build [OUT_DIR]   # build ggtts-render for aarch64-linux-android, check it, write
#                                              # OUT_DIR (default dist/android-arm64): the binary, the libraries it
#                                              # needs that a device lacks (libc++_shared.so), and BUILDINFO.json
#   scripts/android_build.sh cargo-build TARGET_DIR   # only the cargo build (the planted builds use it)
#   scripts/android_build.sh check-pin PIN     # refuse unless the ggml submodule is at PIN (the planted case)
#   scripts/android_build.sh check-neon BIN    # refuse unless BIN's ggml dot product uses NEON vector instructions
#
# Needs the NDK (ANDROID_NDK_HOME) and cargo-ndk. The NEON check reads the disassembly of ggml's Q8_0 dot product,
# the kernel the engines spend their time in, with the NDK's llvm-objdump: NEON's vector registers (`v0.16b` and the
# like) appear only when the build used NEON. `GGML_CPU_ARM_ARCH=armv8-a+nosimd` builds it without, which the check
# must refuse (CI plants that).
#
# On a device, run the binary with its directory on the library path: `LD_LIBRARY_PATH=. ./ggtts-render`.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
say() { echo "android_build: $*" >&2; }
target=aarch64-linux-android
api=${ANDROID_API:-24}
# The one feature string: the cargo build uses it, and BUILDINFO's features are read from it (see `features_json`).
features=ggtts-cli/ggml,ggtts-ggml-sys/bindgen
# What every Android device has (the NDK's stable system libraries the build can link); anything else is shipped.
system_libs=" libc.so libm.so libdl.so liblog.so libandroid.so "

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

llvm_bin() { # TOOL
    local d
    d=$(ls -d "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"/toolchains/llvm/prebuilt/*/bin | head -1)
    echo "$d/$1"
}

sysroot() { ls -d "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"/toolchains/llvm/prebuilt/*/sysroot | head -1; }

# The disassembly of ggml's Q8_0 dot product in BIN.
dot_kernel() { # BIN
    "$(llvm_bin llvm-objdump)" -d --no-show-raw-insn "$1" \
        | awk '/<ggml_vec_dot_q8_0_q8_0>:/{on=1; next} on && /^$/{exit} on'
}

check_neon() { # BIN
    local n
    n=$(dot_kernel "$1" | grep -cE 'v[0-9]+\.(16b|8h|4s|2d)' || true)
    if [ "${n:-0}" -eq 0 ]; then
        say "no NEON vector instructions in ggml_vec_dot_q8_0_q8_0 in $1"
        return 1
    fi
    say "NEON in ggml_vec_dot_q8_0_q8_0: $n vector instructions"
}

cargo_build() { # TARGET_DIR
    # bindgen runs the host's libclang: point it at the NDK's headers for the Android target, not /usr/include.
    export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--sysroot=$(sysroot) --target=$target$api"
    (cd "$root" && cargo ndk -t arm64-v8a --platform "$api" build --release --locked -p ggtts-cli -p ggtts-ggml-sys \
        --features "$features" --target-dir "$1")
}

# Copy into OUT every library BIN needs that a device lacks, and print their names; refuse one the NDK cannot supply.
ship_libs() { # BIN OUT
    local lib src shipped=""
    for lib in $("$(llvm_bin llvm-readelf)" -d "$1" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'); do
        case "$system_libs" in *" $lib "*) continue ;; esac
        src="$(sysroot)/usr/lib/$target/$lib"
        if [ ! -f "$src" ]; then
            say "$1 needs $lib, which is neither on a device nor in the NDK at $src"
            return 1
        fi
        cp "$src" "$2/"
        shipped="$shipped $lib"
    done
    say "shipped beside the binary:${shipped:- nothing}"
    echo "$shipped"
}

# BUILDINFO's features, from what was built rather than written down: the backend and engines that `$features` turns
# on (ggtts-render's `engine_lines` lists exactly these for the ggml feature, and a cuda feature would add a backend),
# then the CPU extensions the dot-product kernel actually uses.
features_json() { # BIN
    local f=() kernel
    case ",$features," in *,ggtts-cli/ggml,*) f+=(cpu ggml qwen3-tts kitten-tts-2) ;; esac
    case ",$features," in *,ggtts-cli/cuda,*) f+=(cuda) ;; esac
    kernel=$(dot_kernel "$1")
    grep -qE 'v[0-9]+\.(16b|8h|4s|2d)' <<<"$kernel" && f+=(neon)
    grep -qE '^\s*[0-9a-f]+:\s+sdot\b' <<<"$kernel" && f+=(dotprod)
    printf '"%s", ' "${f[@]}" | sed 's/, $//; s/^/[/; s/$/]/'
}

build() { # OUT_DIR
    local out=${1:-dist/android-arm64}
    check_pin "$(pin)"
    cargo_build "$root/target"
    local bin="$root/target/$target/release/ggtts-render"
    check_neon "$bin"
    mkdir -p "$out"
    cp "$bin" "$out/"
    local shipped
    shipped=$(ship_libs "$bin" "$out")
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
  "cargo_features": "$features",
  "features": $(features_json "$bin"),
  "ships": [$(for l in $shipped; do printf '"%s",' "$l"; done | sed 's/,$//')]
}
EOF
    cat "$out/BUILDINFO.json" >&2
    say "built $out"
}

case "${1:-}" in
    build) build "${2:-}" ;;
    cargo-build) cargo_build "$2" ;;
    check-pin) check_pin "$2" ;;
    check-neon) check_neon "$2" ;;
    *) echo "usage: $0 build [OUT_DIR] | cargo-build TARGET_DIR | check-pin PIN | check-neon BIN" >&2; exit 2 ;;
esac
