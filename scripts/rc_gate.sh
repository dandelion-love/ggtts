#!/usr/bin/env bash
# Promotion gate. See scripts/rc_gate.py --help and CONTRIBUTING.md.
set -euo pipefail
exec python3 "$(dirname "$0")/rc_gate.py" "$@"
