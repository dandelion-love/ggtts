#!/usr/bin/env bash
# Pack a ggtts release directory, and smoke-test that its script instruments run from it. One script, called by
# ci.yml on every pull request (Linux and Windows) and by release.yml on every real archive before upload.
#
#   scripts/package_and_smoke.sh pack BINARY DIR    # DIR gets the binary, LICENSE, README.md and instruments/
#   scripts/package_and_smoke.sh smoke PATH         # PATH: a packed DIR, or a .tar.gz or .zip of one
#
# `pack` puts the script instruments where the runner looks first: instruments/ beside ggtts-render
# (instruments/python/ggtts_instrument, instruments/templates, instruments/examples, the README and the template
# instruments.yaml).
#
# `smoke` renders a short story on the fake engine through the packed binary with
# `--instrument audio-levels --require audio-levels`, with GGTTS_INSTRUMENTS_DIR unset and the default interpreter
# (GGTTS_PYTHON, else python3, python on Windows). It passes only on exit 0 and an `ok` envelope. Otherwise it exits
# 1 and prints the runner's own reason: given a directory without instruments/, that is "instruments/ directory was
# not found".
set -euo pipefail

say() { echo "package_and_smoke: $*" >&2; }
root=$(cd "$(dirname "$0")/.." && pwd)

pack() {
    local bin=$1 dir=$2
    mkdir -p "$dir/instruments/python"
    cp "$bin" "$dir/"
    cp "$root/LICENSE" "$root/README.md" "$dir/"
    cp -r "$root/instruments/python/ggtts_instrument" "$dir/instruments/python/"
    cp -r "$root/instruments/templates" "$root/instruments/examples" "$root/instruments/README.md" \
        "$root/instruments/instruments.yaml" "$dir/instruments/"
    find "$dir/instruments" -name __pycache__ -prune -exec rm -rf {} +
    say "packed $dir"
}

smoke() {
    local path=$1 work dir exe code
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN
    case "$path" in
        *.tar.gz) tar -xzf "$path" -C "$work"; dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1) ;;
        *.zip) (cd "$work" && { 7z x "$(realpath "$path")" >/dev/null 2>&1 || unzip -q "$(realpath "$path")"; })
               dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1) ;;
        # A directory is copied out first: inside a checkout, the runner's search upward from the binary would find
        # the repository's own instruments/ and hide a directory packed without one.
        *) cp -r "$path" "$work/packed"; dir="$work/packed" ;;
    esac
    exe="$dir/ggtts-render"
    [ -f "$exe" ] || exe="$exe.exe"
    [ -f "$exe" ] || { say "no ggtts-render in $path"; return 1; }
    printf 'Smoke\n今日天氣好好，我哋去公園行下。\n行完之後，去食飯。\n' > "$work/story.txt"
    set +e
    (unset GGTTS_INSTRUMENTS_DIR; cd "$work" && "$exe" --engine fake --story story.txt --out out.flac \
        --report report --instrument audio-levels --require audio-levels) 2> "$work/err.txt"
    code=$?
    set -e
    local status
    status=$(sed -n 's/^ *"status": "\([a-z]*\)".*/\1/p' "$work/report/instruments/audio-levels/envelope.json" 2>/dev/null | head -1)
    if [ "$code" -ne 0 ] || [ "$status" != ok ]; then
        say "FAILED on $path: exit $code, audio-levels ${status:-without an envelope}"
        grep -E "instrument audio-levels|INCOMPLETE|error" "$work/err.txt" >&2 || cat "$work/err.txt" >&2
        return 1
    fi
    say "ok on $path: audio-levels ran from the packed instruments/ (exit 0, envelope ok)"
}

case "${1:-}" in
    pack) pack "$2" "$3" ;;
    smoke) smoke "$2" ;;
    *) echo "usage: $0 pack BINARY DIR | smoke PATH" >&2; exit 2 ;;
esac
