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
- **A wrapped entry** whose text
  continues on a second line
  and a third.
- **Built on Ubuntu**
  24.04 before, and
  -O2 flags.
1. A numbered entry
   that wraps.

A paragraph that
wraps once.

```sh
ggtts-render --one
  --two
```

## v0.1.0-rc.5

### Empty section follows

## v0.1.0-rc.4

## v0.1.0-rc.3

- Older.
EOF
export PATH="$tmp/bin:$PATH" TAG=v0.1.0-rc.6 GITHUB_REPOSITORY=example/repo GITHUB_SHA=abc MOCK_LOG="$tmp/log" \
    MOCK_NOTES="$tmp/notes" CHANGELOG="$tmp/CHANGELOG.md" CHANGELOG_FROZEN="$tmp/frozen"
bash "$root/scripts/check_changelog.sh" --print > "$CHANGELOG_FROZEN"
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
# A release page breaks the line at every newline, so wrapped lines are joined; a code block is kept as written.
grep -Fxq -- '- **A wrapped entry** whose text continues on a second line and a third.' "$MOCK_NOTES" \
    || fail "a wrapped list entry was not joined: $(cat "$MOCK_NOTES")"
grep -Fxq -- '- **Built on Ubuntu** 24.04 before, and -O2 flags.' "$MOCK_NOTES" \
    || fail "a continuation starting with a number or a dash was not joined"
grep -Fxq '1. A numbered entry that wraps.' "$MOCK_NOTES" || fail "a numbered entry was not kept and joined"
grep -Fxq 'A paragraph that wraps once.' "$MOCK_NOTES" || fail "a wrapped paragraph was not joined"
grep -Fxq '  --two' "$MOCK_NOTES" || fail "a code block line was joined or changed"
grep -Fxq '```' "$MOCK_NOTES" || fail "the code fence was lost"
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
# A released section that changed after it was recorded is not published, even when the tag's own section is intact.
cp "$CHANGELOG" "$tmp/kept.md"
sed -i 's/^- Older\.$/- Older, and an entry that arrived later./' "$CHANGELOG"
: > "$MOCK_LOG"
if run 2> "$tmp/err"; then fail "a changed released section unexpectedly published"; fi
grep -Fq 'v0.1.0-rc.3: the released section changed' "$tmp/err" || fail "wrong refusal: $(cat "$tmp/err")"
[ ! -s "$MOCK_LOG" ] || fail "gh was called before the frozen-section refusal"
mv "$tmp/kept.md" "$CHANGELOG"

# scripts/check_changelog.sh: released sections are frozen. First this repository's own file, as committed.
check() { bash "$root/scripts/check_changelog.sh"; }
(unset CHANGELOG CHANGELOG_FROZEN; check > /dev/null) || fail "this repository's CHANGELOG.md fails its own check"
own="$tmp/own.md"
cp "$root/CHANGELOG.md" "$own"
cp "$root/scripts/changelog-released.sha256" "$tmp/own.sha256"
export CHANGELOG="$own" CHANGELOG_FROZEN="$tmp/own.sha256"
check > /dev/null
# What a merge did once: a branch cut before v0.1.0-rc.10 added its entry under the first "### Added" it saw, and
# after the merge that heading belonged to the released v0.1.0-rc.9 section.
awk '{ print } /^## v0\.1\.0-rc\.9/ { s = 1 } s && /^### Added$/ { getline; print; print "- **An entry merged after the release.**"; s = 0 }' \
    "$root/CHANGELOG.md" > "$own"
grep -Fq -- '- **An entry merged after the release.**' "$own" || fail "the planted entry was not placed"
if check 2> "$tmp/err"; then fail "an entry in a released section was not caught"; fi
grep -Fq 'v0.1.0-rc.9: the released section changed' "$tmp/err" || fail "wrong message: $(cat "$tmp/err")"
# The same entry under Unreleased is fine.
awk '{ print } /^## Unreleased$/ { getline; print; print "- **An entry merged after the release.**" }' \
    "$root/CHANGELOG.md" > "$own"
check > /dev/null || fail "an entry under Unreleased was refused"
# A released section with no recorded hash, a recorded tag with no section, and Unreleased not on top: each red.
sed 's/^## Unreleased$/## Unreleased\n\n- New.\n\n## v9.9.9 (2099-01-01)/' "$root/CHANGELOG.md" > "$own"
grep -q '^## v9.9.9' "$own" || fail "the unrecorded section was not placed"
if check 2> "$tmp/err"; then fail "an unrecorded released section passed"; fi
grep -Fq 'v9.9.9: no recorded hash' "$tmp/err" || fail "wrong message: $(cat "$tmp/err")"
bash "$root/scripts/check_changelog.sh" --print v9.9.9 >> "$CHANGELOG_FROZEN"
check > /dev/null || fail "a section recorded with --print still fails"
cp "$root/CHANGELOG.md" "$own"
if check 2> "$tmp/err"; then fail "a recorded tag with no section passed"; fi
grep -Fq 'v9.9.9: recorded as released, but' "$tmp/err" || fail "wrong message: $(cat "$tmp/err")"
cp "$root/scripts/changelog-released.sha256" "$CHANGELOG_FROZEN"
sed '0,/^## Unreleased$/s//## Next/' "$root/CHANGELOG.md" > "$own"
if check 2> "$tmp/err"; then fail "a changelog without Unreleased on top passed"; fi
grep -Fq 'must be "## Unreleased"' "$tmp/err" || fail "wrong message: $(cat "$tmp/err")"

# This repository's own CHANGELOG.md has a section for every published release candidate since it was started.
unset CHANGELOG CHANGELOG_FROZEN
for tag in v0.1.0-rc.8 v0.1.0-rc.9 v0.1.0-rc.10; do
    export TAG="$tag" MOCK_MODE=new
    : > "$MOCK_LOG"
    rm -f "$MOCK_NOTES"
    run
    [ "$(grep -c '^## ' "$MOCK_NOTES")" = 0 ] || fail "$tag: notes run into another release's section"
    [ "$(grep -c '^- ' "$MOCK_NOTES")" -gt 0 ] || fail "$tag: notes have no entries"
    if grep -n '^  ' "$MOCK_NOTES"; then fail "$tag: a wrapped line was left unjoined"; fi
done
echo "release publication selection tests passed"
