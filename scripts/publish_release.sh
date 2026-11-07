#!/usr/bin/env bash
# Publish the workflow's built and attested archives; never overwrite an asset.
set -euo pipefail
: "${TAG:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}"
[ "$#" -gt 0 ] || { echo "no release assets" >&2; exit 1; }
for asset in "$@"; do
    [ -f "$asset" ] || { echo "missing release asset: $asset" >&2; exit 1; }
done

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
    gh release create "$TAG" "$@" "${pre[@]}" --repo "$GITHUB_REPOSITORY" --verify-tag --title "ggtts $TAG" \
        --notes "Built from ${GITHUB_SHA}. Each archive is named with the ggml commit it is pinned to and carries BUILDINFO.json. Verify one with: gh attestation verify <file> -R ${GITHUB_REPOSITORY}"
fi
