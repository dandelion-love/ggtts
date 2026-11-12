#!/usr/bin/env bash
# Compile the documentation's C examples (docs/bindings/examples/c/*.c) against a packed release directory's
# include/ggtts.h and lib/, as a user of the archive would, and run each on the fake engine. A doc example that no
# longer compiles against the shipped header, or fails on the shipped library, is red.
#
#   scripts/c_examples.sh DIR     # DIR: a packed directory (scripts/package_and_smoke.sh pack), e.g. dist/ggtts-ci
#
# The compiler is cc, gcc or clang (GGTTS_CC overrides); on Windows LLVM's clang, which links the MSVC import library
# lib/ggtts_ffi.dll.lib. On Linux the examples build with -std=c99 -Wall -Werror and run with LD_LIBRARY_PATH unset, on
# the rpath to DIR/lib; on Windows with DIR/lib first on PATH. No compiler is a failure: a skipped check is not a pass.
set -euo pipefail

say() { echo "c_examples: $*" >&2; }
root=$(cd "$(dirname "$0")/.." && pwd)
[ $# -eq 1 ] || { say "usage: $0 DIR"; exit 2; }
dir=$(cd "$1" && pwd)
[ -s "$dir/include/ggtts.h" ] || { say "no include/ggtts.h in $1"; exit 1; }

windows=false
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) windows=true ;; esac
cc=""
if [ -n "${GGTTS_CC:-}" ]; then cc=$GGTTS_CC
elif $windows; then
    if command -v clang >/dev/null 2>&1; then cc=clang
    elif [ -x "/c/Program Files/LLVM/bin/clang.exe" ]; then cc="/c/Program Files/LLVM/bin/clang.exe"
    fi
else
    for f in cc gcc clang; do command -v "$f" >/dev/null 2>&1 && { cc=$f; break; }; done
fi
[ -n "$cc" ] || { say "FAILED: no C compiler (set GGTTS_CC)"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
n=0
for src in "$root"/docs/bindings/examples/c/*.c; do
    name=$(basename "$src" .c)
    run="$work/$name"
    mkdir -p "$run"
    if $windows; then
        "$cc" -Wall -D_CRT_SECURE_NO_WARNINGS -I "$dir/include" "$src" "$dir/lib/ggtts_ffi.dll.lib" -o "$run/$name.exe" \
            || { say "FAILED: $name.c did not compile against $1 with $cc"; exit 1; }
        (cd "$run" && PATH="$dir/lib:$PATH" "./$name.exe" fake) || { say "FAILED: $name exited $? on the fake engine"; exit 1; }
    else
        "$cc" -std=c99 -Wall -Werror -I "$dir/include" "$src" -L "$dir/lib" -lggtts_ffi -Wl,-rpath,"$dir/lib" \
            -o "$run/$name" || { say "FAILED: $name.c did not compile against $1 with $cc"; exit 1; }
        (cd "$run" && env -u LD_LIBRARY_PATH "./$name" fake) || { say "FAILED: $name exited $? on the fake engine"; exit 1; }
    fi
    n=$((n + 1))
done
[ "$n" -gt 0 ] || { say "FAILED: no examples in docs/bindings/examples/c"; exit 1; }
[ -s "$work/first_render/first_render.wav" ] || { say "FAILED: first_render wrote no first_render.wav"; exit 1; }
say "ok: $n C examples compiled against $1 and ran on the fake engine"
