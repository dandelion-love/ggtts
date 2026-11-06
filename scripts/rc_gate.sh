#!/usr/bin/env bash
# Local release-candidate gate: run CI's CPU steps on this machine at a given ref, then the GPU steps that hosted
# runners cannot, and write one JSON result for the promotion pull request (rc -> main).
#
#     scripts/rc_gate.sh                      # the checked-out HEAD
#     scripts/rc_gate.sh --ref origin/rc      # a ref, in a temporary worktree
#     scripts/rc_gate.sh --wait-file PATH     # wait while PATH exists (e.g. a machine's "GPU busy" marker)
#     scripts/rc_gate.sh --jobs 4 --out rc-gate.json
#
# GPU steps: none exist yet. Until an engine crate links ggml, the result records gpu: "not-run (no engine)" rather
# than a pass, so a green CPU gate is never read as a GPU parity result.
set -euo pipefail

REF=""; JOBS=4; OUT="rc-gate.json"; WAIT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) shift; REF="$1" ;;
    --jobs) shift; JOBS="$1" ;;
    --out) shift; OUT="$1" ;;
    --wait-file) shift; WAIT_FILE="$1" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(git rev-parse --show-toplevel)"
OUT="$(realpath -m "$OUT")"
WORK="$ROOT"
if [ -n "$REF" ]; then
  WORK="$(mktemp -d)/ggtts-rc"
  git -C "$ROOT" worktree add --detach -q "$WORK" "$REF"
  trap 'git -C "$ROOT" worktree remove --force "$WORK" >/dev/null 2>&1 || true' EXIT
fi
cd "$WORK"

if [ -n "$WAIT_FILE" ]; then
  while [ -e "$WAIT_FILE" ]; do echo "waiting: $WAIT_FILE exists"; sleep 60; done
fi

want_rustc="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)"
have_rustc="$(rustc --version | awk '{print $2}')"
if [ "$want_rustc" != "$have_rustc" ]; then
  echo "refusing: rust-toolchain.toml pins $want_rustc, rustc here is $have_rustc" >&2
  exit 1
fi

commit="$(git rev-parse HEAD)"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
steps=()
run() {
  local name="$1"; shift
  echo "== $name"
  if nice -n 19 "$@"; then steps+=("\"$name\": \"pass\""); else steps+=("\"$name\": \"FAIL\""); FAILED=1; fi
}
FAILED=0
run fmt cargo fmt --all --check
run clippy cargo clippy --workspace --all-targets --locked -j "$JOBS" -- -D warnings
run build cargo build --workspace --locked -j "$JOBS"
run test cargo test --workspace --locked -j "$JOBS"

{
  printf '{\n  "commit": "%s",\n  "ref": "%s",\n  "rustc": "%s",\n  "started": "%s",\n  "finished": "%s",\n' \
    "$commit" "${REF:-HEAD}" "$have_rustc" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "cpu": {%s},\n' "$(IFS=,; echo "${steps[*]}")"
  printf '  "gpu": "not-run (no engine)",\n'
  printf '  "result": "%s"\n}\n' "$([ "$FAILED" = 0 ] && echo pass || echo FAIL)"
} > "$OUT"
cat "$OUT"
exit "$FAILED"
