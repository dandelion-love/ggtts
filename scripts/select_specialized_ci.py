#!/usr/bin/env python3
"""Select specialized CI jobs. A skipped job is not a pass."""
import os
import re
import subprocess
import sys

JOBS = ("package", "binding", "android", "release", "ggml", "cuda")
WORKFLOW = ".github/workflows/ci.yml"
JOB_MAP = {
    "package-smoke": {"package"}, "binding": {"binding"}, "android-arm64": {"android"},
    "release-build": {"release"}, "ggml": {"ggml"}, "ggml-windows": {"ggml"},
    "cuda-compile": {"cuda"}, "specialized-selection": set(JOBS),
    "check": set(), "build-test": set(), "planted-ceiling": set(),
}

# Generic measurement/reporting paths are compiled and tested on both CPU hosts for every PR. Full engine,
# Android and release-archive builds still run on the rc-to-main promotion. This is deliberately an exact list:
# a new or moved crate path falls back to all specialized jobs until classified.
MEASUREMENT_PATHS = {
    "crates/ggtts-cli/src/bin/ggtts-measure.rs",
    "crates/ggtts-cli/src/bin/ggtts-render.rs",
    "crates/ggtts-cli/tests/measure_cli.rs",
    "crates/ggtts-core/src/report.rs",
    "crates/ggtts-core/tests/report.rs",
    "crates/ggtts-instruments/src/runner.rs",
    "crates/ggtts-instruments/tests/runner.rs",
    "crates/ggtts-instruments/tests/no_change.rs",
    "crates/ggtts-qwen3-tts/tests/frame_hook.rs",
}


def workflow_sections(text):
    """Split this repository's workflow into its header and top-level job blocks."""
    header, separator, jobs = text.partition("\njobs:\n")
    if not separator:
        raise ValueError("cannot find workflow jobs; refusing to skip checks")
    sections = {}
    current = None
    for line in jobs.splitlines(keepends=True):
        match = re.match(r"^  ([a-z][a-z0-9_-]+):\s*$", line)
        if match:
            current = match.group(1)
            if current in sections:
                raise ValueError(f"duplicate workflow job {current}")
            sections[current] = line
        elif line.strip() and not line.startswith("    ") and not line.lstrip().startswith("#"):
            raise ValueError("unrecognized workflow job layout; refusing to skip checks")
        elif current is None:
            if line.strip() and not line.lstrip().startswith("#"):
                raise ValueError("unexpected workflow text before first job")
        else:
            sections[current] += line
    if not sections:
        raise ValueError("workflow has no jobs")
    return header, sections


def workflow_changes(before, after):
    old_header, old = workflow_sections(before)
    new_header, new = workflow_sections(after)
    if old_header != new_header:
        return set(JOBS)
    selected = set()
    for name in old.keys() | new.keys():
        if old.get(name) != new.get(name):
            selected.update(JOB_MAP.get(name, set(JOBS)))
    return selected


def decide(event, base_ref, head_ref, same_repo, paths, workflow_selected=None):
    if event != "pull_request":
        return {"ggml", "cuda"}
    if base_ref == "main" and head_ref == "rc" and same_repo:
        return set(JOBS)
    selected = set()
    for path in paths:
        if path == WORKFLOW:
            selected.update(set(JOBS) if workflow_selected is None else workflow_selected)
        if path in ("scripts/select_specialized_ci.py", "scripts/test_select_specialized_ci.py"):
            # CI always runs the selector tests. The full matrix is reserved for promotion, unless another
            # changed path independently needs one of these jobs.
            continue
        if path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".gitmodules"):
            return set(JOBS)
        if path.startswith("crates/"):
            if path in MEASUREMENT_PATHS:
                # The package job exercises the actual CLI/instrument packaging without CUDA or Android.
                if path.startswith(("crates/ggtts-cli/", "crates/ggtts-instruments/")):
                    selected.add("package")
                continue
            if not path.startswith(("crates/ggtts-core/", "crates/ggtts-cli/", "crates/ggtts-instruments/",
                                    "crates/ggtts-qwen3-tts/", "crates/ggtts-kitten/", "crates/ggtts-ggml-sys/",
                                    "crates/ggtts-fake/", "crates/ggtts-ffi/")):
                return set(JOBS)
            # Desktop and Android builds include the engines. The FFI also links
            # the engine and core crates; favor coverage over a fragile dependency map.
            selected.update(("binding", "android", "release", "ggml", "cuda"))
            if path.startswith(("crates/ggtts-cli/", "crates/ggtts-instruments/")):
                selected.add("package")
        if path.startswith(("android/", "scripts/android_build.sh")):
            selected.update(("android", "binding"))
        if path.startswith(("scripts/kotlin_binding_test.sh",)):
            selected.add("binding")
        if path.startswith(("scripts/package_and_smoke.sh", "scripts/check_release_version.py", "instruments/", "fixtures/")):
            selected.add("package")
        if path == "scripts/buildinfo_engines.sh":
            selected.update(("ggml", "release"))
        if path in (".github/workflows/release.yml", "scripts/package_cuda_release.sh"):
            selected.update(("package", "release", "cuda"))
    return selected


def main():
    event = os.environ["EVENT"]
    base_ref = os.environ.get("BASE_REF", "")
    head_ref = os.environ.get("HEAD_REF", "")
    same_repo = bool(os.environ.get("REPOSITORY")) and os.environ.get("REPOSITORY") == os.environ.get("HEAD_REPO")
    paths = []
    workflow_selected = None
    if event == "pull_request" and not (base_ref == "main" and head_ref == "rc" and same_repo):
        base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
        # --no-renames includes both the old and new paths. Failure is fatal.
        paths = os.fsdecode(subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"]
        )).split("\0")
        if WORKFLOW in paths:
            merge_base = subprocess.check_output(["git", "merge-base", base, head], text=True).strip()
            old = subprocess.check_output(["git", "show", f"{merge_base}:{WORKFLOW}"], text=True)
            new = subprocess.check_output(["git", "show", f"{head}:{WORKFLOW}"], text=True)
            workflow_selected = workflow_changes(old, new)
    selected = decide(event, base_ref, head_ref, same_repo, paths, workflow_selected)
    for job in JOBS:
        print(f"{job}={'true' if job in selected else 'false'}")
    print("Selected: " + (", ".join(sorted(selected)) or "none") + "; skipped is not passed", file=sys.stderr)


if __name__ == "__main__":
    main()
