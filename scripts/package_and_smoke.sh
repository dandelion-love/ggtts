#!/usr/bin/env bash
# Pack a ggtts release directory, and smoke-test that its script instruments run from it. One script, called by
# ci.yml on every pull request (Linux and Windows) and by release.yml on every real archive before upload.
#
#   scripts/package_and_smoke.sh pack BINARY DIR    # DIR gets the binaries (and the Kitten tools when built),
#                                                   # LICENSE, THIRD_PARTY.md, licenses/,
#                                                   # README.md and instruments/
#   scripts/package_and_smoke.sh archive DIR zip|tar.gz   # DIR.zip or DIR.tar.gz beside DIR, as released
#   scripts/package_and_smoke.sh smoke PATH [TAG]   # unpack; if TAG is set, check both binaries and BUILDINFO
#
# `archive` is the one way an archive is made, by release.yml and by ci.yml's package-smoke alike, so a PR smokes
# the same kind of file a release ships: 7z for a .zip where it exists (windows-latest), else zip, else Python's
# zipfile; tar for a .tar.gz. Run it from DIR's parent.
#
# `pack` takes BINARY (ggtts-render, with .exe on Windows) and the ggtts-measure built beside it, and refuses when
# either is missing. It also copies the Kitten tools built beside them, ggtts-kitten-convert (the language model and
# decoder GGUFs) and ggtts-kitten (`prepare`, the reference voice): `cargo build -p ggtts-kitten` with the ggml
# feature builds both. `smoke` refuses an archive whose ggtts-render carries kitten-tts-2 without them. It puts the script instruments where the runner looks first: instruments/ beside ggtts-render
# (instruments/python/ggtts_instrument, instruments/templates, instruments/examples, the README and the template
# instruments.yaml). It prints the binaries it packed as a JSON list on stdout, for BUILDINFO's "binaries".
#
# `smoke` renders a short story on the fake engine through the packed binary with
# `--instrument audio-levels --require audio-levels`, with GGTTS_INSTRUMENTS_DIR unset and the default interpreter
# (GGTTS_PYTHON, else python3, python on Windows). It passes only on exit 0 and an `ok` envelope. Otherwise it exits
# 1 and prints the runner's own reason: given a directory without instruments/, that is "instruments/ directory was
# not found". It then measures that render again with the packed ggtts-measure, with the same instrument required,
# and passes only on exit 0 and an `ok` envelope there too. When the directory has a BUILDINFO.json, its "binaries"
# must name both binaries. When ggtts-render --version lists the kitten-tts-2 engine, each Kitten tool must be in the
# directory, run `--help` with exit 0 and be named in BUILDINFO's "binaries"; a missing one fails the smoke with
# "no <tool> in", naming it. With TAG, scripts/check_release_version.py also holds every listed binary's --version to
# the tag and to BUILDINFO's ggtts_commit. After the measure it checks the licence notices: LICENSE, THIRD_PARTY.md,
# licenses/ (Apache-2.0.txt, ggml-LICENSE.txt for the statically linked ggml, rust-crates.txt for every linked
# crate), and the licence file beside each vendored or derived source under instruments/. A missing one fails the
# smoke with "missing licence notice", naming it. `pack` first runs scripts/third_party_notices.py --check, which
# refuses when a crate linked into the binaries has no notice in licenses/rust-crates.txt.
set -euo pipefail

say() { echo "package_and_smoke: $*" >&2; }
# The tools a user needs, beside ggtts-render, to make kitten-tts-2's inputs from the published model.
KITTEN_TOOLS="ggtts-kitten-convert ggtts-kitten"
root=$(cd "$(dirname "$0")/.." && pwd)

pack() {
    local bin=$1 dir=$2 ext="" b list=""
    case "$bin" in *.exe) ext=.exe ;; esac
    mkdir -p "$dir/instruments/python"
    for b in ggtts-render ggtts-measure; do
        [ -f "$(dirname "$bin")/$b$ext" ] || { say "no $b$ext beside $bin"; return 1; }
        cp "$(dirname "$bin")/$b$ext" "$dir/"
        list="$list${list:+,}\"$b$ext\""
    done
    # The Kitten tools make that engine's inputs; built only with the ggml feature, and refused missing by `smoke`.
    for b in $KITTEN_TOOLS; do
        [ -f "$(dirname "$bin")/$b$ext" ] || continue
        cp "$(dirname "$bin")/$b$ext" "$dir/"
        list="$list${list:+,}\"$b$ext\""
    done
    # Refuse to pack when a crate linked into the binaries, or ggml, has no licence notice.
    python3 "$root/scripts/third_party_notices.py" --check >&2 || { say "the licence notices are incomplete"; return 1; }
    cp "$root/LICENSE" "$root/README.md" "$root/THIRD_PARTY.md" "$dir/"
    cp -r "$root/licenses" "$dir/"
    cp -r "$root/instruments/python/ggtts_instrument" "$dir/instruments/python/"
    cp -r "$root/instruments/templates" "$root/instruments/examples" "$root/instruments/vendor" "$root/instruments/README.md" \
        "$root/instruments/instruments.yaml" "$dir/instruments/"
    find "$dir/instruments" -name __pycache__ -prune -exec rm -rf {} +
    say "packed $dir"
    echo "[$list]"
}

archive() {
    local dir=$1 kind=$2
    [ -d "$dir" ] || { say "no directory $dir to archive"; return 1; }
    case "$kind" in
        zip)
            rm -f "$dir.zip"
            if command -v 7z >/dev/null; then 7z a -tzip "$dir.zip" "$dir" >/dev/null
            elif command -v zip >/dev/null; then zip -qr "$dir.zip" "$dir"
            else python3 -c 'import shutil, sys; shutil.make_archive(sys.argv[1], "zip", ".", sys.argv[1])' "$dir"
            fi ;;
        tar.gz) tar -czf "$dir.tar.gz" "$dir" ;;
        *) say "archive kind must be zip or tar.gz, not $kind"; return 1 ;;
    esac
    say "archived $dir.$kind"
}

smoke() {
    local path=$1 expected_tag=${2:-} work dir exe code abs
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN
    case "$path" in
        *.tar.gz|*.zip)
            [ -f "$path" ] || { say "no such archive: $path"; return 1; }
            # Absolute before any cd: the extraction runs inside $work, where a relative path names nothing. That
            # was v0.1.0-rc.3's Windows failure (realpath ran after the cd).
            abs="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")" ;;
    esac
    case "$path" in
        *.tar.gz) tar -xzf "$abs" -C "$work"; dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1) ;;
        *.zip) (cd "$work" && { 7z x "$abs" >/dev/null 2>&1 || unzip -q "$abs"; }) \
                   || { say "could not extract $path"; return 1; }
               dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1) ;;
        # A directory is copied out first: inside a checkout, the runner's search upward from the binary would find
        # the repository's own instruments/ and hide a directory packed without one.
        *) cp -r "$path" "$work/packed"; dir="$work/packed" ;;
    esac
    [ -n "${dir:-}" ] || { say "$path holds no directory"; return 1; }
    if [ -n "$expected_tag" ]; then
        python3 "$root/scripts/check_release_version.py" "$dir" "$expected_tag"
    fi
    # Git Bash's -f accepts an extensionless name for an existing .exe. Prefer
    # the explicit Windows filename so BUILDINFO is checked against its real name.
    exe="$dir/ggtts-render.exe"
    [ -f "$exe" ] || exe="$dir/ggtts-render"
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

    # The same render, measured again by the packed ggtts-measure: the path that needs no render.
    local measure="$dir/ggtts-measure.exe"
    [ -f "$measure" ] || measure="$dir/ggtts-measure"
    [ -f "$measure" ] || { say "no ggtts-measure in $path"; return 1; }
    set +e
    (unset GGTTS_INSTRUMENTS_DIR; cd "$work" && "$measure" --audio out.flac --report measured \
        --instrument audio-levels --require audio-levels) 2> "$work/measure-err.txt"
    code=$?
    set -e
    status=$(sed -n 's/^ *"status": "\([a-z]*\)".*/\1/p' "$work/measured/instruments/audio-levels/envelope.json" 2>/dev/null | head -1)
    if [ "$code" -ne 0 ] || [ "$status" != ok ]; then
        say "FAILED on $path: ggtts-measure exit $code, audio-levels ${status:-without an envelope}"
        cat "$work/measure-err.txt" >&2
        return 1
    fi
    say "ok on $path: ggtts-measure measured the render from the packed instruments/ (exit 0, envelope ok)"
    [ -f "$dir/instruments/vendor/pitchy-4.1.0/index.js" ] && \
        [ -f "$dir/instruments/vendor/fft.js-4.0.4/fft.js" ] || \
        { say "missing bundled pitch detector or FFT in $path"; return 1; }
    local notice
    for notice in LICENSE THIRD_PARTY.md licenses/Apache-2.0.txt licenses/ggml-LICENSE.txt licenses/rust-crates.txt \
        instruments/vendor/fft.js-4.0.4/LICENSE \
        instruments/vendor/pitchy-4.1.0/LICENSE instruments/examples/python/loudness-ends/LICENSE-APACHE-2.0; do
        [ -s "$dir/$notice" ] || { say "missing licence notice in $path: $notice"; return 1; }
    done
    say "ok on $path: the licence notices are present"
    if command -v "${GGTTS_NODE:-node}" >/dev/null 2>&1; then
        set +e
        (unset GGTTS_INSTRUMENTS_DIR; cd "$work" && "$measure" --audio out.flac --report pitch-report \
            --instrument pitch-range --require pitch-range) 2> "$work/pitch-err.txt"
        code=$?
        set -e
        status=$(sed -n 's/^ *"status": "\([^"]*\)".*/\1/p' "$work/pitch-report/instruments/pitch-range/envelope.json" 2>/dev/null | head -1)
        if [ "$code" -ne 0 ] || [ "$status" != ok ]; then
            say "FAILED on $path: pitch-range exit $code, status ${status:-without an envelope}"
            cat "$work/pitch-err.txt" >&2
            return 1
        fi
        say "ok on $path: pitch-range ran from the packed instruments/"
    else
        say "pitch-range not run: Node.js is unavailable"
    fi
    if [ -f "$dir/BUILDINFO.json" ]; then
        local binaries
        binaries=$(sed -n 's/.*"binaries":[[:space:]]*\(\[[^]]*\]\).*/\1/p' "$dir/BUILDINFO.json")
        for b in "$(basename "$exe")" "$(basename "$measure")"; do
            printf '%s\n' "$binaries" | grep -Fq "\"$b\"" \
                || { say "BUILDINFO.json's binaries does not name $b"; return 1; }
        done
        say "ok on $path: BUILDINFO.json lists both binaries"
    fi
    # An archive that renders kitten-tts-2 must carry what makes its inputs, each runnable from the archive.
    local versions
    versions=$("$exe" --version) || { say "FAILED on $path: ggtts-render --version exited non-zero"; return 1; }
    if printf '%s
' "$versions" | grep -q '^engine kitten-tts-2 '; then
        local ext="" tool
        case "$exe" in *.exe) ext=.exe ;; esac
        for b in $KITTEN_TOOLS; do
            tool="$dir/$b$ext"
            [ -f "$tool" ] || { say "no $b$ext in $path: ggtts-render carries kitten-tts-2, whose inputs $b makes"; return 1; }
            "$tool" --help > "$work/$b-help.txt" 2>&1                 || { say "FAILED on $path: $b$ext --help exited non-zero"; cat "$work/$b-help.txt" >&2; return 1; }
            if [ -f "$dir/BUILDINFO.json" ]; then
                printf '%s
' "$binaries" | grep -Fq "\"$b$ext\""                     || { say "BUILDINFO.json's binaries does not name $b$ext"; return 1; }
            fi
        done
        say "ok on $path: the Kitten tools ($KITTEN_TOOLS) run --help from it"
    fi
}

case "${1:-}" in
    pack) pack "$2" "$3" ;;
    archive) archive "$2" "$3" ;;
    smoke) smoke "$2" "${3:-}" ;;
    *) echo "usage: $0 pack BINARY DIR | archive DIR zip|tar.gz | smoke PATH [TAG]" >&2; exit 2 ;;
esac
