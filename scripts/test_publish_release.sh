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
    "release create") [ "$MOCK_MODE" != create-fails ] ;;
    "release upload") [ "$MOCK_MODE" != duplicate ] ;;
    *) echo "unexpected gh call: $*" >&2; exit 2 ;;
esac
EOF
chmod +x "$tmp/bin/gh"
touch "$tmp/linux.tar.gz" "$tmp/windows.zip"
export PATH="$tmp/bin:$PATH" TAG=v0.1.0-rc.6 GITHUB_REPOSITORY=example/repo GITHUB_SHA=abc MOCK_LOG="$tmp/log"
run() { bash "$root/scripts/publish_release.sh" "$tmp/linux.tar.gz" "$tmp/windows.zip"; }

export MOCK_MODE=new
: > "$MOCK_LOG"
run
grep -Fq 'release create v0.1.0-rc.6' "$MOCK_LOG"
grep -Fq -- '--verify-tag' "$MOCK_LOG"
grep -Fq -- '--prerelease --latest=false' "$MOCK_LOG"
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
echo "release publication selection tests passed"
