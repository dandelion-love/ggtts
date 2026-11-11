#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
cat > "$tmp/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$MOCK_LOG"
case "$1 ${2:-}" in
    "api "*) printf '%s\n' "${MOCK_SHA:-abc}" ;;
    "release view")
        case "$MOCK_MODE" in
            new|create-fails) exit 1 ;;
            wrong-type) printf 'false\tfalse\n' ;;
            draft) printf 'true\ttrue\n' ;;
            *) printf 'true\tfalse\n' ;;
        esac ;;
    "release create")
        # Keep the notes the script passed, so the test can read what the release page would say.
        prev=
        for a in "$@"; do
            [ "$prev" = --notes-file ] && cp "$a" "$MOCK_NOTES"
            prev=$a
        done
        [ "$MOCK_MODE" != create-fails ] ;;
    "release upload") [ "$MOCK_MODE" != duplicate ] ;;
    *) echo "unexpected gh call: $*" >&2; exit 2 ;;
esac
EOF
chmod +x "$tmp/bin/gh"
touch "$tmp/linux.tar.gz" "$tmp/windows.zip"
cat > "$tmp/CHANGELOG.md" <<'EOF'
# Changelog

## Unreleased

- Not released yet.

## v0.1.0-rc.60 (2026-01-02)

- A later release whose tag starts with the tested one.

## v0.1.0-rc.6 (2026-01-01)

### Added

- **A new option** for the tested release.

## v0.1.0-rc.5

### Empty section follows

## v0.1.0-rc.4

## v0.1.0-rc.3

- Older.
EOF
export PATH="$tmp/bin:$PATH" TAG=v0.1.0-rc.6 GITHUB_REPOSITORY=example/repo GITHUB_SHA=abc MOCK_LOG="$tmp/log" \
    MOCK_NOTES="$tmp/notes" CHANGELOG="$tmp/CHANGELOG.md"
run() { bash "$root/scripts/publish_release.sh" "$tmp/linux.tar.gz" "$tmp/windows.zip"; }
fail() { echo "$*" >&2; exit 1; }

export MOCK_MODE=new
: > "$MOCK_LOG"
rm -f "$MOCK_NOTES"
run
grep -Fq 'release create v0.1.0-rc.6' "$MOCK_LOG"
grep -Fq -- '--verify-tag' "$MOCK_LOG"
grep -Fq -- '--prerelease --latest=false' "$MOCK_LOG"
# The notes are the tag's section, then the provenance; nothing from the neighbouring sections or the heading.
[ "$(head -1 "$MOCK_NOTES")" = "### Added" ] || fail "notes do not start with the section body: $(head -1 "$MOCK_NOTES")"
grep -Fq -- '- **A new option** for the tested release.' "$MOCK_NOTES" || fail "notes lack the section"
grep -Fq 'Built from abc.' "$MOCK_NOTES" || fail "notes lack the provenance"
grep -Fq 'gh attestation verify <file> -R example/repo' "$MOCK_NOTES" || fail "notes lack the verify command"
for other in 'rc.60' 'A later release' 'Not released yet' 'Older.' '## v0.1.0-rc.6'; do
    if grep -Fq -- "$other" "$MOCK_NOTES"; then fail "notes carry '$other' from outside the section"; fi
done

export MOCK_MODE=existing
: > "$MOCK_LOG"
run
grep -Fq 'release upload v0.1.0-rc.6' "$MOCK_LOG"
if grep -Fq -- '--clobber' "$MOCK_LOG"; then echo "unexpected overwrite" >&2; exit 1; fi
for mode in wrong-type draft duplicate create-fails; do
    export MOCK_MODE="$mode"
    : > "$MOCK_LOG"
    if run; then echo "$mode unexpectedly passed" >&2; exit 1; fi
done
export MOCK_MODE=existing MOCK_SHA=other
: > "$MOCK_LOG"
if run; then echo "mismatched tag unexpectedly passed" >&2; exit 1; fi
if grep -Fq 'release upload' "$MOCK_LOG"; then echo "uploaded on mismatched tag" >&2; exit 1; fi
export MOCK_SHA=abc

# The refusal: a tag with no section (rc.7; rc.1 and rc, whose headings would be prefixes of rc.6's and rc.60's),
# or with an empty one (rc.4), is refused before any gh call, on both the new and the existing path.
for tag in v0.1.0-rc.7 v0.1.0-rc.1 v0.1.0-rc v0.1.0-rc.4; do
    for mode in new existing; do
        export TAG="$tag" MOCK_MODE="$mode"
        : > "$MOCK_LOG"
        if run 2> "$tmp/err"; then fail "$tag ($mode) without a section unexpectedly published"; fi
        grep -Fq "CHANGELOG.md has no section for $tag" "$tmp/err" || fail "$tag ($mode): wrong refusal: $(cat "$tmp/err")"
        [ ! -s "$MOCK_LOG" ] || fail "$tag ($mode): gh was called before the refusal: $(cat "$MOCK_LOG")"
    done
done
# A section holding only a sub-heading is not empty: it is published as it is.
export TAG=v0.1.0-rc.5 MOCK_MODE=new
: > "$MOCK_LOG"
run
grep -Fq '### Empty section follows' "$MOCK_NOTES"
export TAG=v0.1.0-rc.6
mv "$CHANGELOG" "$tmp/moved.md"
if run 2> "$tmp/err"; then fail "a missing changelog unexpectedly published"; fi
grep -Fq 'no changelog at' "$tmp/err"
mv "$tmp/moved.md" "$CHANGELOG"

# This repository's own CHANGELOG.md has a section for every published release candidate since it was started.
unset CHANGELOG
for tag in v0.1.0-rc.8 v0.1.0-rc.9 v0.1.0-rc.10; do
    export TAG="$tag" MOCK_MODE=new
    : > "$MOCK_LOG"
    rm -f "$MOCK_NOTES"
    run
    [ "$(grep -c '^## ' "$MOCK_NOTES")" = 0 ] || fail "$tag: notes run into another release's section"
    [ "$(grep -c '^- ' "$MOCK_NOTES")" -gt 0 ] || fail "$tag: notes have no entries"
done
echo "release publication selection tests passed"
