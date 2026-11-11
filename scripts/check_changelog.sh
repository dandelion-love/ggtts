#!/usr/bin/env bash
# A released section of CHANGELOG.md is frozen. Every "## v..." section (its heading and body, trailing blank lines
# dropped) must hash to the sha256 recorded for its tag in scripts/changelog-released.sha256, and every recorded tag
# must still have its section. The first "## " heading must be "## Unreleased".
#
# Why: git merges a branch's new changelog entry by position, not by meaning. A branch cut before a release adds its
# entry under the "### Added" it saw, and after the release that heading belongs to the released section, so the
# entry lands, conflict-free, in a release that never shipped it. This check turns that red.
#
# Releasing: rename the Unreleased heading to "## <tag> (<date, UTC>)", start a new Unreleased above it, and record
# the section with `scripts/check_changelog.sh --print <tag> >> scripts/changelog-released.sha256`. A deliberate
# correction to a released section replaces its line the same way, in the same commit, so the change is visible.
# Offline and deterministic: it reads only the two files. CHANGELOG and CHANGELOG_FROZEN override their paths.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
changelog=${CHANGELOG:-$root/CHANGELOG.md}
frozen=${CHANGELOG_FROZEN:-$root/scripts/changelog-released.sha256}
[ -f "$changelog" ] || { echo "no changelog at $changelog" >&2; exit 1; }

# The tag of each released section, in file order.
tags() { sed -n 's/^## \(v[^ ]*\).*$/\1/p' "$changelog"; }
# The hash of one released section: its heading line through the line before the next "## ", trailing blanks dropped.
section_hash() {
    awk -v t="$1" '
        /^## / { if (on) exit; if ($0 == "## " t || index($0, "## " t " ") == 1) on = 1 }
        on { lines[++n] = $0 }
        END { while (n > 0 && lines[n] ~ /^[[:space:]]*$/) n--; for (i = 1; i <= n; i++) print lines[i] }
    ' "$changelog" | sha256sum | cut -d' ' -f1
}

if [ "${1:-}" = --print ]; then
    shift
    if [ "$#" -eq 0 ]; then set -- $(tags); fi
    for t in "$@"; do printf '%s  %s\n' "$(section_hash "$t")" "$t"; done
    exit 0
fi

[ -f "$frozen" ] || { echo "no recorded released sections at $frozen" >&2; exit 1; }
bad=0
first=$(grep -m1 '^## ' "$changelog" || true)
if [ "$first" != "## Unreleased" ]; then
    echo "the first section of $changelog must be \"## Unreleased\", not \"$first\"" >&2
    bad=1
fi
dupes=$(tags | sort | uniq -d)
if [ -n "$dupes" ]; then echo "released sections appear twice: $dupes" >&2; bad=1; fi
for t in $(tags); do
    want=$(awk -v t="$t" '$2 == t { print $1 }' "$frozen")
    got=$(section_hash "$t")
    if [ -z "$want" ]; then
        echo "$t: no recorded hash. If $t is released, record it: scripts/check_changelog.sh --print $t >> scripts/changelog-released.sha256" >&2
        bad=1
    elif [ "$want" != "$got" ]; then
        echo "$t: the released section changed (recorded $want, now $got). A new entry belongs under \"## Unreleased\";" \
            "a deliberate correction re-records the line with --print $t." >&2
        bad=1
    fi
done
while read -r _ t; do
    [ -z "$t" ] && continue
    tags | grep -Fxq -- "$t" || { echo "$t: recorded as released, but $changelog has no section for it" >&2; bad=1; }
done < "$frozen"
[ "$bad" = 0 ] && echo "changelog: $(tags | wc -l) released sections unchanged, Unreleased on top"
exit "$bad"
