#!/usr/bin/env bash
# Build, check and smoke the ggtts Python wheel (crates/ggtts-py) the way release.yml ships it. ci.yml's python job
# runs every subcommand on every pull request that selects it, with a fake tag, so a change here is proven before a
# tag needs it.
#
#   scripts/python_wheel.sh version TAG                   # the wheel's PEP 440 version for a tag: v0.1.0-rc.9 -> 0.1.0rc9
#   scripts/python_wheel.sh build TAG ENGINES OUT_DIR     # the release wheel into OUT_DIR (ENGINES: ggml or bindgen)
#   scripts/python_wheel.sh check WHEEL TAG               # what is in the wheel, before it is installed
#   scripts/python_wheel.sh smoke WHEEL TAG [--render]    # install into a clean venv and render
#
# `build` builds with maturin (from crates/ggtts-py/requirements-build.txt), the features `python`,
# `pyo3/extension-module` and ENGINES only: never the test-only `plant`. The version is the tag's, written into
# pyproject.toml for the build and restored after it, and GGTTS_RELEASE_VERSION=TAG makes ggtts.VERSION the tag. ggml
# is built portable (GGML_PORTABLE=1), as the archives are. On Linux the wheel is tagged manylinux_2_35, the archives'
# floor (Ubuntu 22.04), and maturin bundles the libgomp it links. The licence notices are copied into the package
# (ggtts/licenses/): LICENSE, THIRD_PARTY.md, licenses/python-crates.txt, licenses/ggml-LICENSE.txt, and on Linux the
# bundled libgomp's copyright file and the GPL-3 text it refers to, from the build host's libgomp1 package.
#
# `check` refuses a wheel whose name is not ggtts-<version>-cp39-abi3-<platform>.whl for TAG's version (the platform
# manylinux_2_35_x86_64 or win_amd64), that lacks the package, the native module or a notice, or (Linux) that has an
# ELF file needing more than Ubuntu 22.04's runtime (scripts/check_glibc_floor.py on every ELF in the wheel).
#
# `smoke` creates a venv with PYTHON (default python3, or python), installs WHEEL alone (no index, no numpy), and
# passes when ggtts.VERSION is TAG, ggtts.ENGINES is exactly the shipped engines (fake, qwen3-tts, kitten-tts-2; a
# wheel built with `plant` carries test engines and is refused), a fake-engine line renders and writes a FLAC, and an
# unknown engine is a GgttsError with ERR_ENGINE. --render also loads Qwen3-TTS with the default model (from
# GGTTS_CACHE_DIR, downloaded first if missing), renders one line with the default speaker hk_voice_343 to a FLAC of
# 0.5 s to 30 s, and requires an unknown speaker to be refused, naming hk_voice_343.
set -euo pipefail

say() { echo "python_wheel: $*" >&2; }
die() { say "FAILED: $*"; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
crate="$root/crates/ggtts-py"
SHIPPED_ENGINES='["fake", "qwen3-tts", "kitten-tts-2"]'

python_cmd() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"
    elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys' >/dev/null 2>&1; then echo python3
    else echo python
    fi
}

# v1.2.3 -> 1.2.3, v1.2.3-rc.4 -> 1.2.3rc4, -beta.N -> bN, -alpha.N -> aN. Anything else is refused.
pep440() {
    local tag=$1
    if [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}"
    elif [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-rc\.([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}rc${BASH_REMATCH[2]}"
    elif [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-beta\.([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}b${BASH_REMATCH[2]}"
    elif [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-alpha\.([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}a${BASH_REMATCH[2]}"
    else die "tag '$tag' is not vX.Y.Z, vX.Y.Z-rc.N, -beta.N or -alpha.N: no wheel version for it"
    fi
}

is_linux() { [ "$(uname -s)" = Linux ]; }

stage_notices() {
    local dest="$crate/python/ggtts/licenses"
    rm -rf "$dest"; mkdir -p "$dest"
    cp "$root/LICENSE" "$root/THIRD_PARTY.md" "$root/licenses/python-crates.txt" "$root/licenses/ggml-LICENSE.txt" "$dest/"
    if is_linux; then
        local copyright=/usr/share/doc/libgomp1/copyright gpl=/usr/share/common-licenses/GPL-3 version
        [ -f "$copyright" ] && [ -f "$gpl" ] || die "no $copyright or $gpl: the bundled libgomp's notices come from the build host's libgomp1"
        version=$(dpkg-query -W -f='${Version}' libgomp1 2>/dev/null) || die "cannot read the libgomp1 package version"
        cp "$copyright" "$dest/libgomp-copyright.txt"
        cp "$gpl" "$dest/GPL-3.txt"
        {
            echo "The Linux wheel bundles libgomp (ggtts.libs/libgomp-*.so.1), the GNU OpenMP runtime the engines link,"
            echo "copied unmodified from the build host's Ubuntu package libgomp1 ${version} (source package gcc-12)."
            echo "It is under the GPL-3.0 with the GCC Runtime Library Exception 3.1: libgomp-copyright.txt (the package's"
            echo "copyright file, which includes the exception) and GPL-3.txt. Its source is the Ubuntu source package:"
            echo "https://launchpad.net/ubuntu/+source/gcc-12/${version}"
        } > "$dest/libgomp-NOTICE.txt"
    fi
    say "notices staged in $dest: $(cd "$dest" && ls | tr '\n' ' ')"
}

build() {
    local tag=$1 engines=$2 out=$3 version py compat=()
    case "$engines" in ggml|bindgen) ;; *) die "ENGINES must be ggml or bindgen, not '$engines'" ;; esac
    version=$(pep440 "$tag")
    py=$(python_cmd)
    mkdir -p "$out"; out=$(cd "$out" && pwd)
    stage_notices
    cp "$crate/pyproject.toml" "$crate/pyproject.toml.orig"
    trap 'mv -f "$crate/pyproject.toml.orig" "$crate/pyproject.toml"' EXIT
    # The version is the tag's: the static version replaces the crate's (dynamic) one for this build only.
    sed -i.sedbak "s/^dynamic = \[\"version\"\]$/version = \"$version\"/" "$crate/pyproject.toml"
    rm -f "$crate/pyproject.toml.sedbak"
    grep -qx "version = \"$version\"" "$crate/pyproject.toml" || die "could not set the wheel version in pyproject.toml"
    is_linux && compat=(--compatibility manylinux_2_35)
    say "building the ggtts $version wheel with python,pyo3/extension-module,$engines"
    (cd "$crate" && GGTTS_RELEASE_VERSION=$tag GGML_PORTABLE=1 "$py" -m maturin build --release --locked \
        --features "python,pyo3/extension-module,$engines" "${compat[@]}" --out "$out")
    ls -l "$out" >&2
}

check() {
    local wheel=$1 tag=$2 version py work
    version=$(pep440 "$tag")
    py=$(python_cmd)
    [ -f "$wheel" ] || die "no such wheel: $wheel"
    "$py" - "$wheel" "$version" <<'PY' || die "$(basename "$wheel") is not the release wheel"
import re, sys, zipfile
from pathlib import PurePosixPath
wheel, version = sys.argv[1], sys.argv[2]
name = PurePosixPath(wheel.replace("\\", "/")).name
m = re.fullmatch(r"ggtts-([^-]+)-cp39-abi3-(manylinux_2_35_x86_64|win_amd64)\.whl", name)
problems = []
if not m:
    problems.append(f"{name} is not ggtts-<version>-cp39-abi3-(manylinux_2_35_x86_64|win_amd64).whl")
elif m.group(1) != version:
    problems.append(f"{name} has version {m.group(1)}; the tag's is {version}")
files = set(zipfile.ZipFile(wheel).namelist())
linux = bool(m) and m.group(2).startswith("manylinux")
need = ["ggtts/__init__.py", "ggtts/licenses/LICENSE", "ggtts/licenses/THIRD_PARTY.md",
        "ggtts/licenses/python-crates.txt", "ggtts/licenses/ggml-LICENSE.txt"]
if linux:
    need += ["ggtts/licenses/libgomp-copyright.txt", "ggtts/licenses/GPL-3.txt", "ggtts/licenses/libgomp-NOTICE.txt"]
    if not any(re.fullmatch(r"ggtts\.libs/libgomp-[0-9a-f]+\.so\.1", f) for f in files):
        problems.append("no bundled ggtts.libs/libgomp-*.so.1 (its notices are packed for it)")
for f in need:
    if f not in files:
        problems.append(f"no {f}")
if not any(re.fullmatch(r"ggtts/_native\.(abi3\.so|pyd)", f) for f in files):
    problems.append("no native module ggtts/_native.abi3.so or ggtts/_native.pyd")
if any("/tests/" in f or f.startswith("tests/") for f in files):
    problems.append("the wheel carries tests")
for p in problems:
    print("python_wheel: " + p, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
    if [[ "$wheel" == *manylinux* ]]; then
        work=$(mktemp -d)
        "$py" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$wheel" "$work"
        local elves=()
        while IFS= read -r f; do
            [ "$(head -c 4 "$f" | od -An -c | tr -d ' ')" = '177ELF' ] && elves+=("$f")
        done < <(find "$work" -type f)
        [ "${#elves[@]}" -ge 2 ] || die "expected the native module and the bundled libgomp as ELF files, found ${#elves[@]}"
        "$py" "$root/scripts/check_glibc_floor.py" "${elves[@]}" || die "an ELF file in the wheel needs more than Ubuntu 22.04's runtime"
        rm -rf "$work"
    fi
    say "ok: $(basename "$wheel") is the ggtts $version release wheel, with its notices"
}

smoke() {
    local wheel=$1 tag=$2 render="" py work venv vpy
    shift 2
    while [ $# -gt 0 ]; do
        case "$1" in --render) render=1 ;; *) die "unknown option $1" ;; esac
        shift
    done
    [ -f "$wheel" ] || die "no such wheel: $wheel"
    wheel=$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")
    py=$(python_cmd)
    work=$(mktemp -d)
    venv="$work/venv"
    "$py" -m venv "$venv" || die "could not create a venv with $py"
    if [ -x "$venv/bin/python" ]; then vpy="$venv/bin/python"; else vpy="$venv/Scripts/python.exe"; fi
    "$vpy" -m pip install --quiet --no-index --no-deps "$wheel" || die "pip could not install $(basename "$wheel")"
    # Run away from the source tree, so the installed wheel is what is imported.
    (cd "$work" && "$vpy" - "$tag" "$SHIPPED_ENGINES" "$render" <<'PY') || die "$(basename "$wheel") failed its smoke"
import json, os, sys
tag, shipped, render = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] == "1"
import ggtts
def ok(msg): print("python_wheel: ok: " + msg, file=sys.stderr)
def fail(msg): print("python_wheel: " + msg, file=sys.stderr); sys.exit(1)
if ggtts.VERSION != tag:
    fail(f"ggtts.VERSION is {ggtts.VERSION!r}, not the tag {tag!r}")
if sorted(ggtts.ENGINES) != sorted(shipped):
    fail(f"ggtts.ENGINES is {ggtts.ENGINES}, not exactly the shipped engines {shipped} (a test build?)")
ok(f"ggtts {ggtts.VERSION} ({ggtts.COMMIT}) from {os.path.dirname(ggtts.__file__)}, engines {ggtts.ENGINES}")
licenses = os.path.join(os.path.dirname(ggtts.__file__), "licenses")
if not os.path.isfile(os.path.join(licenses, "python-crates.txt")):
    fail("the installed package has no licenses/python-crates.txt")
with ggtts.load("fake") as tts:
    audio = tts.render("今日天氣好好，我哋去公園行下。")
    path = audio.write_flac("fake.flac")
if len(audio) == 0 or open(path, "rb").read(4) != b"fLaC":
    fail("the fake engine rendered nothing, or wrote no FLAC")
ok(f"fake engine: {len(audio)} samples at {audio.sample_rate} Hz, written to FLAC")
try:
    ggtts.load("nonesuch")
    fail("an unknown engine loaded")
except ggtts.GgttsError as e:
    if e.code != ggtts.ERR_ENGINE:
        fail(f"an unknown engine raised code {e.code}, not ERR_ENGINE")
ok("an unknown engine is a GgttsError with ERR_ENGINE")
if render:
    line = "今日天氣好好，我哋去公園行下。"
    with ggtts.load() as tts:
        if tts.speaker != "hk_voice_343" or [f["source"] for f in tts.model_files] != ["cache", "cache"]:
            fail(f"the default model loaded with speaker {tts.speaker} and files {tts.model_files}")
        audio = tts.render(line)
        audio.write_flac("qwen.flac")
    if not 0.5 <= audio.duration <= 30:
        fail(f"a {len(line)}-character Qwen3-TTS line rendered {audio.duration:.2f} s")
    ok(f"Qwen3-TTS, default model, speaker hk_voice_343: {audio.duration:.2f} s of audio, written to FLAC")
    try:
        ggtts.load(speaker="no_such_speaker", offline=True)
        fail("an unknown speaker loaded")
    except ggtts.GgttsError as e:
        if "no_such_speaker" not in e.message or "hk_voice_343" not in e.message:
            fail(f"the unknown-speaker refusal does not name it and the model's speakers: {e.message}")
    ok("an unknown speaker is refused, naming the model's speakers")
PY
    rm -rf "$work"
    say "ok: $(basename "$wheel") installs into a clean venv and renders"
}

case "${1:-}" in
    version) pep440 "$2" ;;
    build) build "$2" "$3" "$4" ;;
    check) check "$2" "$3" ;;
    smoke) shift; smoke "$@" ;;
    *) echo "usage: $0 version TAG | build TAG ggml|bindgen OUT_DIR | check WHEEL TAG | smoke WHEEL TAG [--render]" >&2
       exit 2 ;;
esac
