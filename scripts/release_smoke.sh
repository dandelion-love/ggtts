#!/usr/bin/env bash
# Smoke a release archive the way a user meets it, before it is published, and check what was published after.
# release.yml runs every subcommand on the real archives; ci.yml runs the same subcommands on every pull request that
# touches them (package-smoke on the fake-engine build, release-build on the ggml build, cuda-compile on the CUDA
# archive), so a change here is proven before a tag needs it.
#
#   scripts/release_smoke.sh sum FILE                    # "<sha256>  <name>", the sha256sum format, on stdout
#   scripts/release_smoke.sh check-sum FILE SUMS_DIR     # FILE's sha256 equals SUMS_DIR/<name>.sha256
#   scripts/release_smoke.sh version DIR TAG COMMIT      # ggtts-render --version names TAG and the ggtts COMMIT
#   scripts/release_smoke.sh render DIR                  # one real Qwen3-TTS line on the CPU, default model
#   scripts/release_smoke.sh container DIR IMAGE         # version and render in a bare IMAGE with only libgomp1
#   scripts/release_smoke.sh model-key                   # the default model's cache key, from its pinned sha256s
#   scripts/release_smoke.sh flac-ms FILE                # a FLAC's length in ms, from its STREAMINFO (render's check)
#   scripts/release_smoke.sh archive FILE SUMS_DIR TAG COMMIT [--render] [--container IMAGE] [--cuda-stub]
#   scripts/release_smoke.sh published TAG REPO [SUMS_DIR]
#
# `archive` is the release smoke of one archive: its sha256 is the one its build job recorded (check-sum), the whole
# package smoke passes on it (scripts/package_and_smoke.sh smoke FILE TAG: the fake engine through the packed
# instruments, ggtts-measure, the licence notices, the C program against lib/ and include/ggtts.h, the Kitten tools,
# and --version against the tag and BUILDINFO), BUILDINFO's ggtts_commit is COMMIT, and `version` passes. --render
# adds `render`; --container IMAGE adds `container` (Linux only); --cuda-stub puts the CUDA toolkit's stub driver on
# the loader path first, as scripts/package_cuda_release.sh does, so a CUDA build starts on a runner without a GPU.
# The stub is never packed and must never be on a user's loader path.
#
# `render` renders one Cantonese line with the default Qwen3-TTS model (downloaded into GGTTS_CACHE_DIR, or the
# per-user cache, and verified against its pinned sha256 by ggtts-render itself) and `--speaker hk_voice_343`, on the
# CPU. It passes when the run exits 0, says it used the pinned talker and codec, and writes a FLAC whose STREAMINFO
# gives a duration between 0.5 s and 30 s (read here, with od, so a bare system can check it). Where python3 is
# present it also measures the FLAC with the packed ggtts-measure (audio-levels), which decodes it. It then renders
# with `--speaker no_such_speaker` and passes only when that is refused, naming the speakers the model has.
# RENDER_OFFLINE=1 adds --offline, so a run that would download fails instead.
#
# `container` runs `version` and `render` (offline, on the cache the host already holds) inside IMAGE with nothing
# added but libgomp1, which the README says a minimal system must install. It needs docker, and GGTTS_CACHE_DIR set
# to a directory holding the default model.
#
# `published` downloads every asset of the published release TAG of REPO (owner/name) with gh, and passes only when
# each asset's sha256 equals the digest GitHub reports for it and, with SUMS_DIR, the one its build job recorded
# (and SUMS_DIR names no asset that was not published), and `gh attestation verify` accepts each asset as built by
# REPO's release.yml at refs/tags/TAG. It needs GH_TOKEN; read access to the repository is enough for a public one.
set -euo pipefail

say() { echo "release_smoke: $*" >&2; }
die() { say "FAILED: $*"; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
SPEAKER=hk_voice_343
LINE='今日天氣好好，我哋去公園行下。'
# The default model's pinned files, read from the table the binary is built from.
MODELS_RS="$root/crates/ggtts-models/src/lib.rs"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1 | tr -d '\\'
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else "${GGTTS_PYTHON:-python3}" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
    fi
}

sum() {
    local f=$1
    [ -f "$f" ] || die "no such file: $f"
    printf '%s  %s\n' "$(sha256_of "$f")" "$(basename "$f")"
}

check_sum() {
    local f=$1 sums=$2 name want got
    name=$(basename "$f")
    [ -f "$f" ] || die "no such file: $f"
    [ -s "$sums/$name.sha256" ] || die "no recorded sha256 for $name in $sums ($name.sha256)"
    want=$(cut -d' ' -f1 "$sums/$name.sha256")
    got=$(sha256_of "$f")
    [ "$want" = "$got" ] || die "$name has sha256 $got; its build recorded $want"
    say "ok: $name has the sha256 its build recorded ($got)"
}

render_exe() {
    local dir=$1
    if [ -f "$dir/ggtts-render.exe" ]; then echo "$dir/ggtts-render.exe"
    elif [ -f "$dir/ggtts-render" ]; then echo "$dir/ggtts-render"
    else die "no ggtts-render in $dir"
    fi
}

version() {
    local dir=$1 tag=$2 commit=$3 exe out
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "the expected commit must be a full sha, not '$commit'"
    exe=$(render_exe "$dir")
    out=$("$exe" --version | tr -d '\r') || die "ggtts-render --version exited non-zero (it may not start on this system)"
    printf '%s\n' "$out" >&2
    [ "$(printf '%s\n' "$out" | head -1)" = "ggtts-render $tag" ] \
        || die "ggtts-render --version's first line is '$(printf '%s\n' "$out" | head -1)', expected 'ggtts-render $tag'"
    printf '%s\n' "$out" | grep -q "^ggtts $commit " \
        || die "ggtts-render --version does not name the ggtts commit $commit"
    say "ok: ggtts-render --version names $tag and ggtts $commit"
}

# The duration, in milliseconds, from a FLAC's STREAMINFO; non-zero exit when the file is not a FLAC with one.
flac_ms() {
    local f=$1 b rate samples
    [ "$(head -c 4 "$f")" = fLaC ] || return 1
    # The first metadata block (byte 4, type in the low 7 bits) must be STREAMINFO (type 0).
    b=$(od -An -tu1 -j4 -N1 "$f" | tr -d ' ')
    [ $((b & 127)) -eq 0 ] || return 1
    read -r -a b <<< "$(od -An -tu1 -j18 -N8 "$f")"
    [ "${#b[@]}" -eq 8 ] || return 1
    rate=$(( (b[0] << 12) | (b[1] << 4) | (b[2] >> 4) ))
    samples=$(( ((b[3] & 15) << 32) | (b[4] << 24) | (b[5] << 16) | (b[6] << 8) | b[7] ))
    [ "$rate" -gt 0 ] && [ "$samples" -gt 0 ] || return 1
    echo $(( samples * 1000 / rate ))
}

pinned_sha() {  # the pinned sha256 of the default Qwen3-TTS file with ROLE
    sed -n "/role: \"$1\"/,/sha256:/s/^ *sha256: \"\([0-9a-f]\{64\}\)\",$/\1/p" "$MODELS_RS" | head -1
}

model_key() {
    local talker codec
    talker=$(pinned_sha talker); codec=$(pinned_sha codec)
    [ ${#talker} -eq 64 ] && [ ${#codec} -eq 64 ] || die "cannot read the pinned talker and codec sha256 from $MODELS_RS"
    echo "ggtts-qwen3-tts-default-${talker:0:16}-${codec:0:16}"
}

render() {
    local dir=$1 exe work code ms offline=() talker codec
    exe=$(render_exe "$dir")
    # The pins are passed in by `container`, where the source tree is not mounted.
    talker=${PINNED_TALKER:-$(pinned_sha talker)}; codec=${PINNED_CODEC:-$(pinned_sha codec)}
    [ ${#talker} -eq 64 ] && [ ${#codec} -eq 64 ] || die "no pinned talker and codec sha256"
    [ "${RENDER_OFFLINE:-}" = 1 ] && offline=(--offline)
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN
    local start=$SECONDS
    set +e
    (cd "$work" && "$exe" --text "$LINE" --speaker "$SPEAKER" --device cpu --seed 20260905 "${offline[@]}" \
        --out line.flac) > "$work/out.txt" 2> "$work/err.txt"
    code=$?
    set -e
    tr -d '\r' < "$work/err.txt" | tail -40 >&2
    [ "$code" -eq 0 ] || die "the Qwen3-TTS render with --speaker $SPEAKER exited $code"
    grep -q "$talker" "$work/err.txt" || die "the render did not report the pinned talker sha256 $talker"
    grep -q "$codec" "$work/err.txt" || die "the render did not report the pinned codec sha256 $codec"
    [ -s "$work/line.flac" ] || die "the render wrote no line.flac"
    ms=$(flac_ms "$work/line.flac") || die "line.flac is not a FLAC with a STREAMINFO block and a sample count"
    [ "$ms" -ge 500 ] && [ "$ms" -le 30000 ] \
        || die "line.flac is ${ms} ms long; a ${#LINE}-character line should take between 0.5 s and 30 s"
    say "ok: rendered one Qwen3-TTS line with --speaker $SPEAKER on the CPU in $((SECONDS - start)) s: ${ms} ms of FLAC, pinned talker and codec"
    local measure="$dir/ggtts-measure"
    [ -f "$measure.exe" ] && measure="$measure.exe"
    if command -v "${GGTTS_PYTHON:-python3}" >/dev/null 2>&1 && [ -f "$measure" ]; then
        set +e
        (unset GGTTS_INSTRUMENTS_DIR; cd "$work" && "$measure" --audio line.flac --report measured \
            --instrument audio-levels --require audio-levels) 2> "$work/measure-err.txt"
        code=$?
        set -e
        local status
        status=$(sed -n 's/^ *"status": "\([a-z]*\)".*/\1/p' "$work/measured/instruments/audio-levels/envelope.json" 2>/dev/null | head -1)
        if [ "$code" -ne 0 ] || [ "$status" != ok ]; then
            cat "$work/measure-err.txt" >&2
            die "ggtts-measure could not measure line.flac (exit $code, audio-levels ${status:-without an envelope})"
        fi
        say "ok: ggtts-measure decoded and measured line.flac (audio-levels ok)"
    else
        say "line.flac not decoded by ggtts-measure here (no python3 or no ggtts-measure); its STREAMINFO was read"
    fi
    set +e
    (cd "$work" && "$exe" --text "$LINE" --speaker no_such_speaker --device cpu "${offline[@]}" --out refused.flac) \
        > "$work/out.txt" 2> "$work/refused.txt"
    code=$?
    set -e
    if [ "$code" -eq 0 ]; then
        tr -d '\r' < "$work/refused.txt" | tail -20 >&2
        die "ggtts-render accepted --speaker no_such_speaker"
    fi
    grep -q "no_such_speaker" "$work/refused.txt" && grep -q "$SPEAKER" "$work/refused.txt" || {
        tr -d '\r' < "$work/refused.txt" | tail -20 >&2
        die "the refusal of --speaker no_such_speaker does not name it and the model's speakers"
    }
    [ ! -e "$work/refused.flac" ] || die "a refused render left refused.flac behind"
    say "ok: --speaker no_such_speaker is refused (exit $code), naming the model's speakers: $(grep -m1 no_such_speaker "$work/refused.txt" | tr -d '\r')"
}

container() {
    local dir=$1 image=$2 abs tag commit
    command -v docker >/dev/null 2>&1 || die "docker is needed to smoke in $image"
    [ -n "${GGTTS_CACHE_DIR:-}" ] && [ -d "$GGTTS_CACHE_DIR" ] || die "set GGTTS_CACHE_DIR to the cache holding the default model"
    abs=$(cd "$dir" && pwd)
    tag=${SMOKE_TAG:?set SMOKE_TAG}; commit=${SMOKE_COMMIT:?set SMOKE_COMMIT}
    # Only libgomp1 is installed (the README's instruction for a minimal system); the render is offline, on the cache
    # the host run filled, so the container needs the network for apt-get alone. Three apt attempts: the archive
    # fails now and then.
    docker run --rm -v "$abs:/ggtts:ro" -v "$root/scripts/release_smoke.sh:/smoke/release_smoke.sh:ro" \
        -v "$GGTTS_CACHE_DIR:/cache" -e GGTTS_CACHE_DIR=/cache -e RENDER_OFFLINE=1 \
        -e PINNED_TALKER="$(pinned_sha talker)" -e PINNED_CODEC="$(pinned_sha codec)" \
        "$image" bash -c '
            set -e
            for attempt in 1 2 3; do
                apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libgomp1 >/dev/null && break
                [ "$attempt" -lt 3 ] || { echo "apt-get failed 3 times" >&2; exit 1; }
                sleep 20
            done
            . /etc/os-release; echo "release_smoke: in $PRETTY_NAME, $(ldd --version | head -1), with libgomp1 added" >&2
            bash /smoke/release_smoke.sh version /ggtts "$0" "$1"
            bash /smoke/release_smoke.sh render /ggtts' "$tag" "$commit" \
        || die "the archive did not pass in $image with only libgomp1 added"
    say "ok: in $image with only libgomp1 added, --version and an offline Qwen3-TTS render pass"
}

unpack() {  # unpack ARCHIVE into WORK; print the directory it holds
    local f=$1 work=$2 abs
    abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
    case "$f" in
        *.tar.gz) tar -xzf "$abs" -C "$work" ;;
        *.zip) (cd "$work" && { 7z x "$abs" >/dev/null 2>&1 || unzip -q "$abs"; }) || die "could not extract $f" ;;
        *) die "not a release archive: $f" ;;
    esac
    find "$work" -mindepth 1 -maxdepth 1 -type d | head -1
}

stub_driver() {  # the CUDA toolkit's stub libcuda first on the loader path, as package_cuda_release.sh does
    local cuda=${CUDA_HOME:-/usr/local/cuda} stub
    [ -f "$cuda/lib64/stubs/libcuda.so" ] || die "no CUDA stub driver at $cuda/lib64/stubs/libcuda.so"
    stub=$(mktemp -d)
    ln -s "$cuda/lib64/stubs/libcuda.so" "$stub/libcuda.so.1"
    export LD_LIBRARY_PATH="$stub:$cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    say "the CUDA stub driver is on the loader path for this smoke only ($stub)"
}

archive_smoke() {
    local f=$1 sums=$2 tag=$3 commit=$4 do_render="" image="" work dir
    shift 4
    while [ $# -gt 0 ]; do
        case "$1" in
            --render) do_render=1 ;;
            --container) image=$2; shift ;;
            --cuda-stub) stub_driver ;;
            *) die "unknown option $1" ;;
        esac
        shift
    done
    check_sum "$f" "$sums"
    bash "$root/scripts/package_and_smoke.sh" smoke "$f" "$tag" || die "the package smoke failed on $f"
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN
    dir=$(unpack "$f" "$work")
    [ -n "$dir" ] || die "$f holds no directory"
    grep -q "\"ggtts_commit\": \"$commit\"" "$dir/BUILDINFO.json" || die "BUILDINFO.json's ggtts_commit is not $commit"
    version "$dir" "$tag" "$commit"
    [ -z "$do_render" ] || render "$dir"
    [ -z "$image" ] || SMOKE_TAG=$tag SMOKE_COMMIT=$commit container "$dir" "$image"
    say "ok: $(basename "$f") passed the release smoke"
}

published() {
    local tag=$1 repo=$2 sums=${3:-} work name f got digest n=0
    command -v gh >/dev/null 2>&1 || die "gh is needed to check a published release"
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN
    gh release download "$tag" -R "$repo" -D "$work/assets" || die "could not download the assets of $repo $tag"
    gh api "repos/$repo/releases/tags/$tag" --jq '.assets[] | "\(.name) \(.digest)"' > "$work/digests.txt" \
        || die "could not read the asset digests of $repo $tag"
    for f in "$work/assets"/*; do
        [ -f "$f" ] || continue
        name=$(basename "$f"); n=$((n + 1))
        got=$(sha256_of "$f")
        digest=$(awk -v n="$name" '$1 == n {print $2}' "$work/digests.txt")
        [ "$digest" = "sha256:$got" ] || die "$name downloads with sha256 $got; GitHub reports '${digest:-no digest}'"
        if [ -n "$sums" ]; then check_sum "$f" "$sums"; fi
        gh attestation verify "$f" --repo "$repo" --signer-workflow "$repo/.github/workflows/release.yml" \
            --source-ref "refs/tags/$tag" > "$work/attest.txt" 2>&1 \
            || { cat "$work/attest.txt" >&2; die "gh attestation verify refused $name"; }
        say "ok: $name: sha256 $got as GitHub reports it, attested by $repo's release.yml at $tag"
    done
    [ "$n" -gt 0 ] || die "$repo $tag has no assets"
    if [ -n "$sums" ]; then
        for f in "$sums"/*.sha256; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .sha256)
            [ -f "$work/assets/$name" ] || die "$name was built and smoked but is not published"
        done
    fi
    say "ok: $n assets of $repo $tag verified"
}

case "${1:-}" in
    sum) sum "$2" ;;
    check-sum) check_sum "$2" "$3" ;;
    version) version "$2" "$3" "$4" ;;
    render) render "$2" ;;
    container) container "$2" "$3" ;;
    model-key) model_key ;;
    # The FLAC reader `render` uses, on its own, so CI can show it reads a render and refuses a truncated file.
    flac-ms) flac_ms "$2" || die "$2 is not a FLAC with a STREAMINFO block and a sample count" ;;
    archive) shift; archive_smoke "$@" ;;
    published) published "$2" "$3" "${4:-}" ;;
    *) echo "usage: $0 sum FILE | check-sum FILE SUMS_DIR | version DIR TAG COMMIT | render DIR | container DIR IMAGE" \
            "| model-key | archive FILE SUMS_DIR TAG COMMIT [--render] [--container IMAGE] [--cuda-stub]" \
            "| published TAG REPO [SUMS_DIR]" >&2; exit 2 ;;
esac
