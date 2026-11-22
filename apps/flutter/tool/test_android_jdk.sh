#!/usr/bin/env bash
# Exercise the real settings guard with JDK17_HOME, JDK21_HOME, and GRADLE_BIN.
set -euo pipefail
: "${JDK17_HOME:?Set JDK17_HOME to a JDK 17 installation}"
: "${JDK21_HOME:?Set JDK21_HOME to a JDK 21 installation}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../android" && pwd)"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
if JAVA_HOME="$JDK17_HOME" "$GRADLE_BIN" -p "$here" help --offline --no-daemon >"$log" 2>&1; then
  echo "JDK 17 unexpectedly passed" >&2
  exit 1
fi
grep -F 'Flutter Android builds require JDK 21 or newer for Android lint' "$log"
grep -E 'Gradle is running on Java 17([. +_-]|$)' "$log"
JAVA_HOME="$JDK21_HOME" "$GRADLE_BIN" -p "$here" help --offline --no-daemon
