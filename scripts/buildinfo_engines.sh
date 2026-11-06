#!/usr/bin/env bash
# The release's engine list, read from what the binary says was compiled in, never written by hand.
#
#   scripts/buildinfo_engines.sh BIN [REQUIRED_ENGINE...]
#
# Prints a JSON array of the engines `BIN --version` lists (its "engine <name> backends <...>" lines), for
# BUILDINFO.json. Exits 1, naming what is wrong, when the list is empty or misses a REQUIRED_ENGINE, so a build that
# silently dropped an engine cannot ship claiming it. Exit 2 on a usage or run error.
set -euo pipefail
bin=${1:?usage: buildinfo_engines.sh BIN [REQUIRED_ENGINE...]}
shift
out=$("$bin" --version) || { echo "buildinfo_engines: $bin --version failed" >&2; exit 2; }
engines=$(printf '%s\n' "$out" | sed -n 's/^engine \([A-Za-z0-9_-]*\) backends .*/\1/p')
if [ -z "$engines" ]; then
  echo "buildinfo_engines: $bin lists no engines:" >&2
  printf '%s\n' "$out" | sed 's/^/  /' >&2
  exit 1
fi
for want in "$@"; do
  if ! printf '%s\n' "$engines" | grep -qx "$want"; then
    echo "buildinfo_engines: $bin was asked to carry $want and does not; it lists: $(echo $engines)" >&2
    exit 1
  fi
done
printf '%s\n' "$engines" | sed 's/.*/"&"/' | paste -sd, - | sed 's/^/[/; s/$/]/'
