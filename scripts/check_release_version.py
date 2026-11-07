#!/usr/bin/env python3
"""Refuse a release if either packaged binary or BUILDINFO disagrees with its tag."""
import json
from pathlib import Path
import subprocess
import sys


def main(directory, tag):
    directory = Path(directory)
    if not tag:
        raise ValueError("release tag is empty")
    with (directory / "BUILDINFO.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("ggtts") != tag:
        raise ValueError(f"BUILDINFO ggtts is {metadata.get('ggtts')!r}, expected {tag!r}")
    for name in ("ggtts-render", "ggtts-measure"):
        exe = directory / (name + ".exe")
        if not exe.is_file():
            exe = directory / name
        output = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True)
        first = output.stdout.splitlines()[0] if output.stdout.splitlines() else ""
        if first != f"{name} {tag}":
            raise ValueError(f"{exe.name} reports {first!r}, expected {name} {tag}")
    print(f"release version verified: {tag}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_release_version.py PACKED_DIR TAG")
    try:
        main(*sys.argv[1:])
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"release version check failed: {exc}")
