#!/usr/bin/env python3
"""Select specialized CI jobs. A skipped job is not a pass."""
import os
import subprocess
import sys

JOBS = ("package", "binding", "android", "release")


def decide(event, base_ref, head_ref, same_repo, paths):
    if event != "pull_request":
        return set()
    if base_ref == "main" and head_ref == "rc" and same_repo:
        return set(JOBS)
    selected = set()
    for path in paths:
        if path in (".github/workflows/ci.yml", "scripts/select_specialized_ci.py",
                    "scripts/test_select_specialized_ci.py"):
            return set(JOBS)
        if path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".gitmodules"):
            return set(JOBS)
        if path.startswith("crates/"):
            # Desktop and Android builds include the engines. The FFI also links
            # the engine and core crates; favor coverage over a fragile dependency map.
            selected.update(("binding", "android", "release"))
            if path.startswith(("crates/ggtts-cli/", "crates/ggtts-instruments/")):
                selected.add("package")
        if path.startswith(("android/", "scripts/android_build.sh")):
            selected.update(("android", "binding"))
        if path.startswith(("scripts/kotlin_binding_test.sh",)):
            selected.add("binding")
        if path.startswith(("scripts/package_and_smoke.sh", "instruments/", "fixtures/")):
            selected.add("package")
        if path == ".github/workflows/release.yml":
            selected.update(("package", "release"))
    return selected


def main():
    event = os.environ["EVENT"]
    base_ref = os.environ.get("BASE_REF", "")
    head_ref = os.environ.get("HEAD_REF", "")
    same_repo = bool(os.environ.get("REPOSITORY")) and os.environ.get("REPOSITORY") == os.environ.get("HEAD_REPO")
    paths = []
    if event == "pull_request" and not (base_ref == "main" and head_ref == "rc" and same_repo):
        base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
        # --no-renames includes both the old and new paths. Failure is fatal.
        paths = os.fsdecode(subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"]
        )).split("\0")
    selected = decide(event, base_ref, head_ref, same_repo, paths)
    for job in JOBS:
        print(f"{job}={'true' if job in selected else 'false'}")
    print("Selected: " + (", ".join(sorted(selected)) or "none") + "; skipped is not passed", file=sys.stderr)


if __name__ == "__main__":
    main()
