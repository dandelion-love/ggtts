#!/usr/bin/env bash
# Decide whether a PR needs the hosted ggml CUDA compile. A push always needs it.
set -euo pipefail

relevant() {
  case "$1" in
    .github/workflows/ci.yml|scripts/cuda_ci_decision.sh|.gitmodules|Cargo.toml|Cargo.lock|rust-toolchain.toml|crates/ggtts-core/*|crates/ggtts-ggml-sys/*|crates/ggtts-qwen3-tts/*|crates/ggtts-kitten/*) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "${1:-}" = "--self-test" ]; then
  for path in .github/workflows/ci.yml scripts/cuda_ci_decision.sh .gitmodules crates/ggtts-core/src/lib.rs crates/ggtts-ggml-sys/ggml crates/ggtts-qwen3-tts/src/lib.rs crates/ggtts-kitten/src/lib.rs; do
    relevant "$path" || { echo "missed relevant path: $path" >&2; exit 1; }
  done
  for path in README.md docs/example.md instruments/example.py; do
    if relevant "$path"; then echo "unrelated path selected: $path" >&2; exit 1; fi
  done
  echo "CUDA path selection self-test passed"
  exit 0
fi

event="${1:?event required}"
if [ "$event" != pull_request ]; then
  echo "run=true"
  echo "CUDA compile selected for integration push" >&2
  exit 0
fi

base="${2:?PR base SHA required}"
head="${3:?PR head SHA required}"
files="$(mktemp)"
trap 'rm -f "$files"' EXIT
# Diff the merge base to the PR head, including both names of a rename.
# A failed diff fails this job; it cannot silently skip the compile.
git diff --no-renames --name-only -z "$base...$head" -- > "$files"
while IFS= read -r -d '' path; do
  if relevant "$path"; then
    echo "run=true"
    echo "CUDA compile selected: $path" >&2
    exit 0
  fi
done < "$files"
echo "run=false"
echo "CUDA compile skipped: no relevant paths changed; this is not GPU parity evidence" >&2
