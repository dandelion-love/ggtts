#!/usr/bin/env bash
# Smoke-test a release archive's script instruments: unpack it, run one Python instrument through the shipped
# ggtts-render on the fake engine, and check that a planted missing instruments/ directory is reported.
#
#   scripts/smoke_release_archive.sh dist/ggtts-<tag>-<target>-....tar.gz   (or .zip)
#
# Passes only when:
#   1. the shipped binary runs `audio-levels` (a shipped Python instrument) from the archive's own instruments/
#      directory, required, and the run reports it ok with a complete measurement (exit 0);
#   2. with instruments/ removed from the unpacked copy, the same run reports the instrument failed with the
#      runner's "instruments/ directory was not found" reason. The exit code is not asserted: with --require the
#      measurement is INCOMPLETE (exit 3), and what matters is that the cause is named.
# GGTTS_INSTRUMENTS_DIR is unset for both runs, so the archive's own layout is what is tested.
set -euo pipefail

archive=$(realpath "$1")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
case "$archive" in
    *.zip) 7z x "$archive" >/dev/null 2>&1 || unzip -q "$archive" ;;
    *.tar.gz) tar -xzf "$archive" ;;
    *) echo "smoke: unknown archive type: $archive" >&2; exit 2 ;;
esac
root=$(find . -mindepth 1 -maxdepth 1 -type d | head -1)
exe="$root/ggtts-render"
[ -x "$exe" ] || exe="$exe.exe"
[ -x "$exe" ] || { echo "smoke: no ggtts-render in $archive" >&2; exit 1; }
unset GGTTS_INSTRUMENTS_DIR

for d in instruments/python/ggtts_instrument instruments/templates/python instruments/examples/python; do
    [ -d "$root/$d" ] || { echo "smoke: the archive has no $d" >&2; exit 1; }
done

echo "smoke: 1. a shipped Python instrument runs from the archive"
"$exe" --engine fake --text "smoke" --out ok.flac --report ok.report \
    --instrument audio-levels --require audio-levels 2> ok.err || {
    cat ok.err >&2
    echo "smoke: the shipped instrument did not run cleanly" >&2
    exit 1
}
grep -q "instrument audio-levels .*: ok" ok.err || { cat ok.err >&2; echo "smoke: audio-levels was not ok" >&2; exit 1; }
grep -q "measurement: complete" ok.report/run.log || { cat ok.report/run.log >&2; exit 1; }

echo "smoke: 2. planted: without instruments/, the runner names the missing directory"
mv "$root/instruments" "$root/instruments.moved"
set +e
"$exe" --engine fake --text "smoke" --out planted.flac --report planted.report \
    --instrument audio-levels --require audio-levels 2> planted.err
code=$?
set -e
mv "$root/instruments.moved" "$root/instruments"
if ! grep -q "instruments/ directory was not found" planted.err; then
    cat planted.err >&2
    echo "smoke: the planted missing directory was not reported (exit $code)" >&2
    exit 1
fi
echo "smoke: ok ($archive): the instrument ran, and the planted missing directory was reported (exit $code)"
