#!/usr/bin/env bash
# Opt-in CPU test; ignored by default and not run automatically in CI.
# Requires local fixtures, a prepared Cargo cache and local ggml source.
# No downloads: Cargo runs offline. The deadline covers build and test.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GGTTS_TINY_TALKER:?Set GGTTS_TINY_TALKER to the local tiny talker GGUF}"
: "${GGTTS_CODEC:?Set GGTTS_CODEC to the local codec GGUF}"
: "${GGML_SRC:?Set GGML_SRC to the local ggml source directory}"
[[ -d "$GGML_SRC" ]] || { echo "Missing local GGML_SRC directory" >&2; exit 1; }
verify() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "Missing local fixture: $path" >&2; exit 1; }
  actual=$(sha256sum -- "$path")
  [[ "${actual%% *}" == "$expected" ]] || { echo "Fixture SHA-256 mismatch: $path" >&2; exit 1; }
}
verify "$GGTTS_TINY_TALKER" 75b126568ce4c90c99b4fc4e44db6c5953803bff486db00be51bc85284932c33
verify "$GGTTS_CODEC" 7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824
# GNU timeout signals its process group and escalates to KILL if cleanup stalls.
exec timeout --kill-after=5s 150s cargo test --offline -p ggtts-qwen3-tts \
  --features ggml,plant --test cancellation_tiny --locked -- --ignored --nocapture
