#!/usr/bin/env bash
# Publish the workflow's built and attested archives; never overwrite an asset.
# The release notes are the tag's section of CHANGELOG.md (a heading "## <tag>" or "## <tag> (date)", up to the next
# "## " heading), followed by the build provenance. A tag with no section, or an empty one, is refused before anything
# is published. Wrapped lines are joined, since a release page breaks at every newline. CHANGELOG overrides the file
# read (the tests use it).
set -euo pipefail
: "${TAG:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"
[ "$#" -gt 0 ] || { echo "no release assets" >&2; exit 1; }
for asset in "$@"; do
    [ -f "$asset" ] || { echo "missing release asset: $asset" >&2; exit 1; }
done

changelog=${CHANGELOG:-$(cd "$(dirname "$0")/.." && pwd)/CHANGELOG.md}
[ -f "$changelog" ] || { echo "no changelog at $changelog" >&2; exit 1; }
# The heading must be the tag exactly, alone or followed by a space: v0.1.0-rc.1 never takes v0.1.0-rc.10's section.
# Leading and trailing blank lines are dropped.
section=$(awk -v h="## $TAG" '
    /^## / { if (on) exit; if ($0 == h || index($0, h " ") == 1) { on = 1; next } }
    on { lines[++n] = $0 }
    END {
        first = 1; while (first <= n && lines[first] ~ /^[[:space:]]*$/) first++
        last = n; while (last >= first && lines[last] ~ /^[[:space:]]*$/) last--
        # A release page renders every newline as a line break, so a wrapped line is joined to the one it continues.
        # A line starts a new block when it is blank, a heading, a list item, a table row, a quote or a code fence,
        # or follows a blank line, a heading or a fence; lines inside a fence are kept as they are.
        out = ""; fence = 0; prev = ""
        for (i = first; i <= last; i++) {
            l = lines[i]
            if (l ~ /^[[:space:]]*```/) { fence = !fence; out = out (out == "" ? "" : "\n") l; prev = l; continue }
            starts = fence || l ~ /^[[:space:]]*$/ || l ~ /^[[:space:]]*([-*+][[:space:]]|[0-9]+[.)][[:space:]]|[#|>])/ \
                || prev ~ /^[[:space:]]*$/ || prev ~ /^#/ || prev ~ /^[[:space:]]*```/ || i == first
            if (starts) out = out (i == first ? "" : "\n") l
            else { sub(/^[[:space:]]+/, "", l); out = out " " l }
            prev = lines[i]
        }
        print out
    }' "$changelog")
[ -n "$(printf '%s' "$section" | tr -d '[:space:]')" ] || {
    echo "CHANGELOG.md has no section for $TAG: add a \"## $TAG (date)\" section before tagging" >&2
    exit 1
}

# Verify even the existing-release path resolves the tag to the commit that built these artifacts.
actual=$(gh api "repos/$GITHUB_REPOSITORY/commits/$TAG" --jq .sha)
[ "$actual" = "$GITHUB_SHA" ] || { echo "tag does not resolve to the build commit" >&2; exit 1; }
pre=()
expected=false
case "$TAG" in *-rc.*) pre=(--prerelease --latest=false); expected=true ;; esac

if state=$(gh release view "$TAG" --repo "$GITHUB_REPOSITORY" --json isPrerelease,isDraft \
        --jq '[.isPrerelease,.isDraft] | @tsv' 2>/dev/null); then
    [ "$state" = "$(printf '%s\tfalse' "$expected")" ] || {
        echo "existing release has the wrong type or is a draft" >&2
        exit 1
    }
    # No --clobber: an existing asset is never replaced. A partial upload fails visibly.
    gh release upload "$TAG" "$@" --repo "$GITHUB_REPOSITORY"
else
    notes=$(mktemp)
    trap 'rm -f "$notes"' EXIT
    {
        printf '%s\n\n' "$section"
        printf '%s\n' "---" ""
        printf 'Built from %s. Each archive is named with the ggml commit it is pinned to and carries BUILDINFO.json. Verify one with: gh attestation verify <file> -R %s\n' \
            "$GITHUB_SHA" "$GITHUB_REPOSITORY"
    } > "$notes"
    gh release create "$TAG" "$@" "${pre[@]}" --repo "$GITHUB_REPOSITORY" --verify-tag --title "ggtts $TAG" \
        --notes-file "$notes"
fi
