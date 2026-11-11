#!/usr/bin/env python3
"""Refuse a release if a packaged binary or BUILDINFO disagrees with its tag or its ggtts commit.

Every binary BUILDINFO.json's "binaries" names (ggtts-render and ggtts-measure always) must print `<name> <TAG>` as
the first line of --version, and `ggtts <ggtts_commit> (...)` as a later one, where ggtts_commit is BUILDINFO's.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

REQUIRED = ("ggtts-render", "ggtts-measure")


def main(directory, tag):
    directory = Path(directory)
    if not tag:
        raise ValueError("release tag is empty")
    with (directory / "BUILDINFO.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("ggtts") != tag:
        raise ValueError(f"BUILDINFO ggtts is {metadata.get('ggtts')!r}, expected {tag!r}")
    commit = metadata.get("ggtts_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"BUILDINFO ggtts_commit is {commit!r}, expected a full commit SHA")
    names = list(REQUIRED)
    for listed in metadata.get("binaries", []):
        name = listed[:-4] if listed.endswith(".exe") else listed
        if name not in names:
            names.append(name)
    for name in names:
        exe = directory / (name + ".exe")
        if not exe.is_file():
            exe = directory / name
        if not exe.is_file():
            raise ValueError(f"{name} is listed in BUILDINFO binaries or required, and is not in {directory}")
        output = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True)
        lines = output.stdout.splitlines()
        first = lines[0] if lines else ""
        if first != f"{name} {tag}":
            raise ValueError(f"{exe.name} reports {first!r}, expected {name} {tag}")
        built = [line.split()[1] for line in lines[1:] if line.startswith("ggtts ") and len(line.split()) > 1]
        if built != [commit]:
            raise ValueError(f"{exe.name} reports ggtts commit {built[0] if built else None!r}, "
                             f"BUILDINFO ggtts_commit is {commit!r}")
    print(f"release version verified: {tag} at {commit} ({', '.join(names)})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_release_version.py PACKED_DIR TAG")
    try:
        main(*sys.argv[1:])
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"release version check failed: {exc}")
