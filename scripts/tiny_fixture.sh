#!/usr/bin/env bash
# The tiny real-model fixture: the published v0.0.1 talker cut to one talker layer and one code-predictor layer, and
# the published codec, for the no-change check on a real engine (crates/ggtts-qwen3-tts/tests/tiny_model.rs).
#
#   eval "$(scripts/tiny_fixture.sh)"     # prints: export GGTTS_TINY_TALKER=... GGTTS_CODEC=...
#
# Everything is pinned in crates/ggtts-qwen3-tts/fixtures/tiny/SOURCE: the Hugging Face repository and revision, both
# source files' sha256, the layer counts, and the cut's sha256. Every file is checked against its hash, and a
# mismatch stops the script (exit 1): it never hands a check the wrong file. The fixture is never edited by hand; it
# is regenerated here by scripts/cut_tiny_fixture.py, whose output is deterministic.
#
# GGTTS_TINY_CACHE (default ~/.cache/ggtts-tiny) holds the codec and the cut. GGTTS_TINY_SOURCE_TALKER and
# GGTTS_TINY_SOURCE_CODEC use local copies of the published files instead of downloading them (still hash-checked).
# The 2 GB talker is downloaded only to cut it, and deleted afterwards unless it was a local copy.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src="$root/crates/ggtts-qwen3-tts/fixtures/tiny/SOURCE"
val() { sed -n "s/^$1=//p" "$src"; }
repo=$(val REPO) rev=$(val REVISION)
cache=${GGTTS_TINY_CACHE:-$HOME/.cache/ggtts-tiny}
mkdir -p "$cache"
say() { echo "tiny_fixture: $*" >&2; }

check() { # FILE SHA256
    local got
    got=$(sha256sum "$1" | cut -d' ' -f1)
    if [ "$got" != "$2" ]; then
        say "$1 has sha256 $got, not the pinned $2"
        return 1
    fi
}

fetch() { # FILE SHA256 LOCAL OUT: the published file at the pinned revision, or a local copy of it
    local file=$1 sha=$2 local_copy=$3 out=$4
    if [ -f "$out" ] && check "$out" "$sha" 2>/dev/null; then
        return 0
    fi
    if [ -n "$local_copy" ]; then
        say "using $local_copy"
        check "$local_copy" "$sha"
        ln -sf "$(realpath "$local_copy")" "$out"
        return 0
    fi
    say "downloading $repo@$rev/$file"
    curl -fL --retry 3 --silent --show-error -o "$out.part" "https://huggingface.co/$repo/resolve/$rev/$file"
    check "$out.part" "$sha"
    mv "$out.part" "$out"
}

cut="$cache/qwen-talker-tiny-t$(val TALKER_LAYERS)-c$(val CODE_PRED_LAYERS)-q8_0.gguf"
codec="$cache/$(val CODEC_FILE)"
fetch "$(val CODEC_FILE)" "$(val CODEC_SHA256)" "${GGTTS_TINY_SOURCE_CODEC:-}" "$codec"

if ! { [ -f "$cut" ] && check "$cut" "$(val CUT_SHA256)" 2>/dev/null; }; then
    talker="$cache/$(val TALKER_FILE)"
    fetch "$(val TALKER_FILE)" "$(val TALKER_SHA256)" "${GGTTS_TINY_SOURCE_TALKER:-}" "$talker"
    python3 "$root/scripts/cut_tiny_fixture.py" "$talker" "$cut.part" \
        --talker-layers "$(val TALKER_LAYERS)" --code-pred-layers "$(val CODE_PRED_LAYERS)" >&2
    check "$cut.part" "$(val CUT_SHA256)"
    mv "$cut.part" "$cut"
    [ -L "$talker" ] || rm -f "$talker"
fi

say "fixture $cut ($(du -h "$cut" | cut -f1)), codec $codec"
echo "export GGTTS_TINY_TALKER=$cut GGTTS_CODEC=$codec"
