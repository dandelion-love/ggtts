#!/usr/bin/env bash
# Build ggtts for Android arm64 with cargo-ndk, and check what was built: ggml at the pin, NEON in the kernels, and
# every shared library the binary needs either on the device or shipped beside it.
#
#   scripts/android_build.sh build [OUT_DIR]   # build ggtts-render for aarch64-linux-android, check it, write
#                                              # OUT_DIR (default dist/android-arm64): the binary, the libraries it
#                                              # needs that a device lacks (libc++_shared.so), and BUILDINFO.json;
#                                              # then `app OUT_DIR`
#   scripts/android_build.sh app OUT_DIR       # build the Android library (AAR) and the demo app (APK) from android/,
#                                              # run the demo's JVM tests, check both, and copy them to OUT_DIR
#   scripts/android_build.sh cargo-build TARGET_DIR   # only the cargo build (the planted builds use it)
#   scripts/android_build.sh check-pin PIN     # refuse unless the ggml submodule is at PIN (the planted case)
#   scripts/android_build.sh check-neon BIN    # refuse unless BIN's ggml dot product uses NEON vector instructions
#   scripts/android_build.sh check-align FILE...  # refuse unless every LOAD segment is aligned to 16 KB or more
#   scripts/android_build.sh check-aar AAR     # refuse unless AAR holds the arm64-v8a libraries and the binding classes
#   scripts/android_build.sh check-apk APK     # refuse unless APK holds the arm64-v8a library and no model file
#
# Needs the NDK (ANDROID_NDK_HOME) and cargo-ndk; `app` also needs a JDK 17 or later (JAVA_HOME), the Android SDK
# (ANDROID_HOME or ANDROID_SDK_ROOT), python3, and the network once, for Gradle (pinned below, checked against its
# sha256) and the Android Gradle plugin. The NEON check reads the disassembly of ggml's Q8_0 dot product,
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
# The Android library: the C-ABI with its JNI entry points, over ggml.
ffi_features=ggtts-ffi/ggml,ggtts-ffi/jni,ggtts-ggml-sys/bindgen
# Gradle for android/, pinned by version and by the sha256 Gradle publishes for that distribution.
gradle_version=8.11.1
gradle_sha256=f397b287023acdba1e9f6fc5ea72d22dd63669d59ed4a289a29b1a76eee151c6
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

cargo_ndk() { # TARGET_DIR PACKAGE FEATURES
    # bindgen runs the host's libclang: point it at the NDK's headers for the Android target, not /usr/include.
    export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--sysroot=$(sysroot) --target=$target$api"
    # 16 KB pages: Android 15 and later devices may use them, and a library whose LOAD segments are aligned to 4 KB
    # is flagged there (or does not load). See `check_align`.
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="${CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS:-} -C link-arg=-Wl,-z,max-page-size=16384"
    (cd "$root" && cargo ndk -t arm64-v8a --platform "$api" build --release --locked -p "$2" -p ggtts-ggml-sys \
        --features "$3" --target-dir "$1")
}

cargo_build() { # TARGET_DIR
    cargo_ndk "$1" ggtts-cli "$features"
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

# Refuse unless every LOAD segment of each FILE is aligned to 16 KB or more, so it loads on a 16 KB-page device.
check_align() { # FILE...
    local f a aligns bad
    for f in "$@"; do
        aligns=$("$(llvm_bin llvm-readelf)" -lW "$f" | awk '$1 == "LOAD" { print $NF }')
        bad=""
        for a in $aligns; do
            [ $((a)) -ge 16384 ] || bad="$bad $a"
        done
        if [ -z "$aligns" ] || [ -n "$bad" ]; then
            say "$f has LOAD segments aligned below 16 KB (${bad:- none found}): a 16 KB-page device flags or refuses it"
            return 1
        fi
    done
    say "16 KB-aligned: $*"
}

# The entries of a zip file (an AAR or an APK), one per line.
zip_entries() { # ZIP
    python3 -c 'import sys, zipfile; print("\n".join(zipfile.ZipFile(sys.argv[1]).namelist()))' "$1"
}

# Refuse unless AAR packs the Android library and every library it ships, and the binding's classes.
check_aar() { # AAR
    local entries classes lib missing=""
    entries=$(zip_entries "$1")
    for lib in libggtts_ffi.so $(ls "$root/target/android-jniLibs/arm64-v8a" 2>/dev/null | grep -v '^libggtts_ffi.so$'); do
        grep -qx "jni/arm64-v8a/$lib" <<<"$entries" || missing="$missing jni/arm64-v8a/$lib"
    done
    grep -qx classes.jar <<<"$entries" || missing="$missing classes.jar"
    if [ -z "$missing" ]; then
        classes=$(python3 -c 'import io, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
print("\n".join(zipfile.ZipFile(io.BytesIO(z.read("classes.jar"))).namelist()))' "$1")
        for c in ggtts/Ggtts.class ggtts/Native.class ggtts/GgttsException.class ggtts/Pcm.class; do
            grep -qx "$c" <<<"$classes" || missing="$missing classes.jar:$c"
        done
    fi
    if [ -n "$missing" ]; then
        say "$1 lacks:$missing"
        return 1
    fi
    say "$1 holds $(grep -c '^jni/arm64-v8a/.*\.so$' <<<"$entries") libraries in jni/arm64-v8a/ and the binding's classes"
}

# Refuse unless APK holds the Android library for arm64-v8a, and refuse any model file in it: no weights ship.
check_apk() { # APK
    local entries models
    entries=$(zip_entries "$1")
    models=$(grep -iE '\.(gguf|safetensors|bin|pt|onnx)$' <<<"$entries" || true)
    if [ -n "$models" ]; then
        say "$1 carries model files: $(tr '\n' ' ' <<<"$models")"
        return 1
    fi
    if ! grep -qx lib/arm64-v8a/libggtts_ffi.so <<<"$entries"; then
        say "$1 has no lib/arm64-v8a/libggtts_ffi.so"
        return 1
    fi
    say "$1 holds lib/arm64-v8a/libggtts_ffi.so and no model file"
}

# A copy of zip IN at OUT with ENTRY removed (MODE drop) or added (MODE add): the planted cases below.
plant_zip() { # MODE IN OUT ENTRY
    python3 - "$@" <<'EOF'
import sys, zipfile
mode, src, dst, entry = sys.argv[1:]
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
    for item in zin.infolist():
        if not (mode == "drop" and item.filename == entry):
            zout.writestr(item, zin.read(item.filename))
    if mode == "add":
        zout.writestr(entry, b"GGUF planted")
EOF
}

# A pinned Gradle, downloaded once into a cache and checked against its sha256 before it is used.
gradle_bin() {
    local home=${GGTTS_GRADLE_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/ggtts-gradle}
    local dir="$home/gradle-$gradle_version" zip
    if [ ! -x "$dir/bin/gradle" ]; then
        mkdir -p "$home"
        zip="$home/gradle-$gradle_version-bin.zip"
        curl -fsSL -o "$zip.part" "https://services.gradle.org/distributions/gradle-$gradle_version-bin.zip"
        if ! echo "$gradle_sha256  $zip.part" | sha256sum -c --quiet - >&2; then
            say "the Gradle $gradle_version download does not match its pinned sha256"
            return 1
        fi
        mv "$zip.part" "$zip"
        (cd "$home" && unzip -q -o "$zip")
    fi
    echo "$dir/bin/gradle"
}

app() { # OUT_DIR
    local out=${1:?app OUT_DIR} tdir="$root/target"
    mkdir -p "$out"
    cargo_ndk "$tdir" ggtts-ffi "$ffi_features"
    local lib="$tdir/$target/release/libggtts_ffi.so"
    check_neon "$lib"
    if ! "$(llvm_bin llvm-nm)" -D --defined-only "$lib" | grep -q ' Java_ggtts_Native_loadWithModels$'; then
        say "$lib does not export the JNI entry points"
        return 1
    fi
    local jni="$tdir/android-jniLibs"
    rm -rf "$jni"
    mkdir -p "$jni/arm64-v8a"
    cp "$lib" "$jni/arm64-v8a/"
    ship_libs "$lib" "$jni/arm64-v8a" > /dev/null
    check_align "$jni"/arm64-v8a/*.so

    # Gradle 8.11 runs on JDK 17 to 23. A runner may default to a newer one (GitHub's ubuntu-latest has 25, which
    # fails with nothing but its version number), so take its JDK 17 when it names one.
    export JAVA_HOME=${JAVA_HOME_17_X64:-${JAVA_HOME:?set JAVA_HOME to a JDK 17 to 23}}
    local major
    major=$("$JAVA_HOME/bin/java" -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java.specification.version = //p')
    if [ "${major:-0}" -lt 17 ] || [ "${major:-0}" -gt 23 ]; then
        say "JAVA_HOME ($JAVA_HOME) is Java ${major:-unknown}; Gradle $gradle_version needs 17 to 23"
        return 1
    fi
    local gradle
    gradle=$(gradle_bin)
    export ANDROID_HOME=${ANDROID_HOME:-${ANDROID_SDK_ROOT:?set ANDROID_HOME or ANDROID_SDK_ROOT}}
    # The library's release AAR, the demo's debug APK (debug-signed, so it installs with adb), and the demo's JVM tests
    # (its error path, against a fake binding).
    "$gradle" --no-daemon --console=plain -p "$root/android" -Pggtts.jniLibs="$jni" \
        :ggtts:assembleRelease :demo:assembleDebug :demo:testDebugUnitTest
    cp "$root/android/ggtts/build/outputs/aar/ggtts-release.aar" "$out/ggtts.aar"
    cp "$root/android/demo/build/outputs/apk/debug/demo-debug.apk" "$out/ggtts-demo-debug.apk"
    check_aar "$out/ggtts.aar"
    check_apk "$out/ggtts-demo-debug.apk"

    # Planted: each check must refuse what it exists to refuse.
    local p="$tdir/android-planted"
    mkdir -p "$p"
    plant_zip drop "$out/ggtts.aar" "$p/no-lib.aar" jni/arm64-v8a/libggtts_ffi.so
    if check_aar "$p/no-lib.aar" 2>/dev/null; then say "planted: an AAR without libggtts_ffi.so passed"; return 1; fi
    plant_zip add "$out/ggtts-demo-debug.apk" "$p/with-model.apk" assets/model.gguf
    if check_apk "$p/with-model.apk" 2>/dev/null; then say "planted: an APK carrying a .gguf passed"; return 1; fi
    echo 'int ggtts_planted(void) { return 0; }' > "$p/planted.c"
    "$(llvm_bin clang)" --target="$target$api" -shared -Wl,-z,max-page-size=4096 -o "$p/lib4k.so" "$p/planted.c"
    if check_align "$p/lib4k.so" 2>/dev/null; then say "planted: a 4 KB-aligned library passed"; return 1; fi
    say "planted: an AAR without the library, an APK with a .gguf and a 4 KB-aligned library are all refused"
    say "built $out/ggtts.aar and $out/ggtts-demo-debug.apk"
}

build() { # OUT_DIR
    local out=${1:-dist/android-arm64}
    check_pin "$(pin)"
    cargo_build "$root/target"
    local bin="$root/target/$target/release/ggtts-render"
    check_neon "$bin"
    check_align "$bin"
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
    app "$out"
}

case "${1:-}" in
    build) build "${2:-}" ;;
    app) app "${2:-}" ;;
    cargo-build) cargo_build "$2" ;;
    check-pin) check_pin "$2" ;;
    check-neon) check_neon "$2" ;;
    check-align) shift; check_align "$@" ;;
    check-aar) check_aar "$2" ;;
    check-apk) check_apk "$2" ;;
    *) echo "usage: $0 build [OUT_DIR] | app OUT_DIR | cargo-build TARGET_DIR | check-pin PIN | check-neon BIN" \
            "| check-align FILE... | check-aar AAR | check-apk APK" >&2; exit 2 ;;
esac
