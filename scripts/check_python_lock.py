#!/usr/bin/env python3
"""Refuse a Python binding lockfile that has drifted from the main one.

    python3 scripts/check_python_lock.py [ROOT_LOCK PY_LOCK]

crates/ggtts-py is its own Cargo workspace (so the main builds never need Python), with its own Cargo.lock. The
binding links the same engine crates as the binaries, so a crate both lockfiles hold must resolve to the same
versions in both: otherwise the wheel ships different code than the archives, and its notices list other versions.
For each crate named in both, every version the binding's lockfile resolves must be one the main lockfile resolves
(the main lockfile may hold more, for crates only its other packages need). Crates only the binding needs (PyO3 and
its dependencies) are not compared. The defaults are Cargo.lock and crates/ggtts-py/Cargo.lock.

It reads the lockfiles without a TOML parser, so it runs on any Python 3, and fails when a lockfile has no packages.
To fix a drift, copy the main lockfile over the binding's and let cargo prune it:
`cp Cargo.lock crates/ggtts-py/Cargo.lock && cargo metadata --manifest-path crates/ggtts-py/Cargo.toml > /dev/null`.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELD = re.compile(r'^(name|version|source) = "([^"]*)"$')


def packages(text):
    """{name: {version, ...}} from a Cargo.lock's [[package]] tables."""
    found, current = {}, None
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line == "[[package]]":
            current = {}
            continue
        if line.startswith("["):
            current = None
            continue
        m = FIELD.match(line) if current is not None else None
        if m:
            current[m.group(1)] = m.group(2)
            if "name" in current and "version" in current and not current.get("_seen"):
                found.setdefault(current["name"], set()).add(current["version"])
                current["_seen"] = True
    return found


def drift(root_text, py_text):
    """Every difference, as lines; empty when the binding's lockfile agrees with the main one."""
    root, py = packages(root_text), packages(py_text)
    if not root or not py:
        return ["a lockfile holds no [[package]] entries: " + ("main" if not root else "binding")]
    problems = []
    for name in sorted(root.keys() & py.keys()):
        extra = py[name] - root[name]
        if extra:
            problems.append(f"{name}: the binding resolves {', '.join(sorted(extra))}; the main lockfile resolves "
                            f"{', '.join(sorted(root[name]))}")
    return problems


def main(argv):
    if len(argv) not in (0, 2):
        print(__doc__, file=sys.stderr)
        return 2
    root_lock, py_lock = (Path(a) for a in argv) if argv else (ROOT / "Cargo.lock", ROOT / "crates/ggtts-py/Cargo.lock")
    problems = drift(root_lock.read_text(encoding="utf-8"), py_lock.read_text(encoding="utf-8"))
    for p in problems:
        print(f"python lock drift: {p}", file=sys.stderr)
    if problems:
        print(f"python lock drift: {py_lock} differs from {root_lock}; see this script's docstring for the fix",
              file=sys.stderr)
        return 1
    shared = len(packages(root_lock.read_text(encoding="utf-8")).keys()
                 & packages(py_lock.read_text(encoding="utf-8")).keys())
    print(f"python lock: the {shared} crates both lockfiles hold resolve to the main lockfile's versions",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
