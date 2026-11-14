#!/usr/bin/env python3
"""Select specialized CI jobs. A skipped job is not a pass."""
import difflib
import os
from pathlib import Path
import re
import hashlib
import json
import subprocess
import sys

try:
    import tomllib
except ImportError:
    tomllib = None  # Without a TOML parser the full matrix is mandatory.

JOBS = ("package", "binding", "android", "release", "ggml", "python")
WORKFLOW = ".github/workflows/ci.yml"
CORE_MANIFEST = "crates/ggtts-core/Cargo.toml"
CLI_MANIFEST = "crates/ggtts-cli/Cargo.toml"
EVENT_MANIFEST = "instruments/templates/rust/event/Cargo.toml"
# SHA-256 of canonical JSON for each complete Cargo.lock package record, including
# version, source, checksum and dependency edges. This exception is for the
# reviewed tracing migration only; updating any crate requires the full matrix.
LOGGING_PACKAGES = {
    "lazy_static": "15bd801d09e1094b47b5e90f119e99a6e47f5b2e924564a876badd37db69496a",
    "nu-ansi-term": "8ef26ee10c110c45ccc6f886e7d75ba8f44e06a2637917bcef7d002933fa98b1",
    "pin-project-lite": "23b080b81571e7fe356b0e5a6e7c17cb6b85571ac44164eed70c272e534ced3c",
    "sharded-slab": "5994beb14930440eeae3f09785455af17fcf09422b8b32bcb136e0b87c8de25a",
    "thread_local": "7144ce2583c0f92037b14f22453584c933704581994b7780cbb2e5ec678ec254",
    "tracing": "9bc555a9daac0a1322793e97ae1455c0d4b5878150e58c1ca49b4ce3bb649975",
    "tracing-attributes": "8e948dcea3da3585c71ac877273ad24a15855d35c98becc5f69ae96ac58e704d",
    "tracing-core": "fbbbca89e3fbb7b05a1454e06288579b2a12d50624706019ad5e418028300be6",
    "tracing-log": "152a6d7406ccd45fa87f3a2d3d61cdfcbf471fac0405c8378394900ae1594c96",
    "tracing-subscriber": "61f72241c899f2b6e87702784a8356f178412b3d7839ae002c2aaa0531c6dbf6",
    "valuable": "04d5fa6441d8ede51ed68f48ff8d279cc05093c62ae449c9d0a6df383b3ec21e",
}
JOB_MAP = {
    "package-smoke": {"package"}, "binding": {"binding"}, "android-arm64": {"android"},
    "release-build": {"release"}, "ggml": {"ggml"}, "ggml-windows": {"ggml"},
    # cuda-compile never runs on an ordinary PR (cuda_decision below), so an edit to it selects nothing here; the
    # push to rc that lands a workflow edit runs it. A PR that changes a CUDA archive script runs it (CUDA_SCRIPTS).
    "cuda-compile": set(), "specialized-selection": set(JOBS),
    "check": set(), "build-test": set(), "planted-ceiling": set(), "python": {"python"},
    # tiny-model is selected by tiny_decision below, which reads its own block of ci.yml (workflow_touches).
    "tiny-model": set(),
}

# Where `feature = "ggml"` gates code: the default build compiles none of it, so only the ggml job runs its tests.
# The engine crates are gated throughout; elsewhere it is these files. A test greps the tree and fails when a gated
# file is in neither list. (crates/ggtts-ggml-sys/ and crates/ggtts-ffi/ select ggml by their own rule below.)
GGML_GATED_CRATES = ("crates/ggtts-kitten/", "crates/ggtts-qwen3-tts/", "crates/ggtts-load/")
GGML_GATED_FILES = {
    "crates/ggtts-cli/src/bin/ggtts-render.rs",
    "crates/ggtts-cli/tests/default_models_cli.rs",
    "crates/ggtts-cli/tests/kitten_refusals.rs",
    "crates/ggtts-cli/tests/no_change_real.rs",
    "crates/ggtts-cli/tests/steer_cli.rs",
    "crates/ggtts-cli/tests/version_engines.rs",
    # The book's real-model steering test. It is `#[ignore]` and run by hand (its header says how): the ggml job
    # builds the workspace with `ggtts-cli/ggml`, which does not turn on `ggtts-book/ggml`, so there it compiles to
    # nothing. Listed so the gated-file check holds and a change to it still runs the ggml job.
    "crates/ggtts-book/tests/steer_tiny.rs",
}

# The Python binding (crates/ggtts-py) is its own workspace, built and tested by the python job only, which compiles
# its ggml-gated code on Linux. It links these crates (ggtts-ffi only for a test that holds the error codes equal), so a
# change to their source or manifest also runs it. Their own tests do not reach the binding. ggtts-load holds the engine
# glue the binding shares with the C library (`ggtts_load::binding`).
PYTHON_CRATE = "crates/ggtts-py/"
# The book layer (crates/ggtts-book) links no ggml and ships in no archive, wheel or binding: the general CPU
# checks (fmt, clippy, and build-test on Linux and Windows) build and test it, and no specialized job reads it.
BOOK_CRATE = "crates/ggtts-book/"
PYTHON_ENGINE_CRATES = ("crates/ggtts-core/", "crates/ggtts-fake/", "crates/ggtts-qwen3-tts/", "crates/ggtts-kitten/",
                        "crates/ggtts-ggml-sys/", "crates/ggtts-models/", "crates/ggtts-ffi/", "crates/ggtts-load/")


# The scripts the python job runs to build, check and smoke the release wheel as release.yml does, and release.yml
# itself (its build-wheel and smoke-wheel jobs run only on a tag, so the python job is their proof on a pull request).
PYTHON_PATHS = {"scripts/python_wheel.sh", "scripts/check_python_lock.py", "scripts/test_check_python_lock.py",
                "scripts/third_party_notices.py", "licenses/python-crates.txt", "scripts/check_glibc_floor.py",
                ".github/workflows/release.yml"}


# The binding docs (docs/bindings) and the example files their code blocks copy. The check job holds every block to
# its file on every run (scripts/check_doc_examples.py); these jobs compile and run the files on the fake engine: the C
# examples in package-smoke (scripts/c_examples.sh, against the packed lib/ and include/), the Kotlin ones in binding
# (scripts/kotlin_binding_test.sh), the Python ones in python (crates/ggtts-py/tests/test_doc_examples.py). An example
# selects its own language's job; a page, or anything else there, selects all three.
DOC_BINDINGS = "docs/bindings/"
DOC_EXAMPLE_JOBS = {"docs/bindings/examples/c/": "package", "docs/bindings/examples/kotlin/": "binding",
                    "docs/bindings/examples/python/": "python"}


def doc_bindings_jobs(path):
    """The jobs that run the binding docs' examples a change to `path` can break (empty outside docs/bindings)."""
    if not path.startswith(DOC_BINDINGS):
        return set()
    for prefix, job in DOC_EXAMPLE_JOBS.items():
        if path.startswith(prefix):
            return {job}
    return {"package", "binding", "python"}


def python_path(path):
    """A path whose change can break the Python binding's build or behaviour."""
    if path.startswith(PYTHON_CRATE) or path in PYTHON_PATHS:
        return True
    return path.startswith(PYTHON_ENGINE_CRATES) and "/tests/" not in path and "/benches/" not in path


# Generic measurement/reporting paths are compiled and tested on both CPU hosts for every PR.
# CPU release, Android and binding builds run on the rc-to-main promotion.
MEASUREMENT_PATHS = {
    "crates/ggtts-cli/src/bin/ggtts-measure.rs",
    "crates/ggtts-cli/src/bin/ggtts-render.rs",
    "crates/ggtts-cli/tests/measure_cli.rs",
    "crates/ggtts-cli/tests/no_change_real.rs",
    "crates/ggtts-core/src/report.rs",
    "crates/ggtts-core/tests/report.rs",
    "crates/ggtts-core/tests/python_instruments.rs",
    "crates/ggtts-instruments/src/catalogue.rs",
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


def logging_lockfile_change(old_lock, new_lock, old_core, new_core, old_cli=None, new_cli=None,
                            old_event=None, new_event=None):
    """Only the exact, reviewed tracing graph and manifest changes qualify."""
    if tomllib is None:
        return False
    try:
        before, after = tomllib.loads(old_lock), tomllib.loads(new_lock)
        if before.get("version") not in (3, 4) or after.get("version") not in (3, 4):
            return False
        core_before, core_after = tomllib.loads(old_core), tomllib.loads(new_core)
        if (old_event is None) != (new_event is None):
            return False
        if old_event is not None:
            event_before, event_after = tomllib.loads(old_event), tomllib.loads(new_event)
            package = event_before["package"]
            if (package.get("name") != "ggtts-instrument-event-template" or
                    package.get("description") != "Template: an out-of-process ggtts event instrument in Rust, reading ggtts-trace/1 and writing ggtts-report/1"):
                return False
            package["description"] = "Template: an out-of-process ggtts event instrument in Rust, reading ggtts traces and writing ggtts-report/1"
            if event_before != event_after:
                return False
        if (old_cli is None) != (new_cli is None):
            return False
        if old_cli is not None:
            cli_before, cli_after = tomllib.loads(old_cli), tomllib.loads(new_cli)
            if cli_before["dev-dependencies"].pop("serde_json") != {"workspace": True}:
                return False
            cli_before["dependencies"]["serde_json"] = {"workspace": True}
            if cli_before != cli_after:
                return False
        if "tracing" in core_before["dependencies"] or "tracing-subscriber" in core_before["dependencies"]:
            return False
        core_before["dependencies"].update({"tracing": "0.1.44", "tracing-subscriber": "0.3.23"})
        if core_before != core_after:
            return False

        def packages(lock):
            result = {}
            for package in lock.pop("package"):
                key = (package["name"], package["version"])
                if not all(isinstance(part, str) and part for part in key):
                    raise ValueError("invalid package identity")
                if package.get("source", "").startswith("registry+") and not re.fullmatch(
                    r"[0-9a-f]{64}", package.get("checksum", "")
                ):
                    raise ValueError("invalid registry checksum")
                if key in result:
                    raise ValueError("duplicate lockfile package")
                result[key] = package
            return result

        old_packages, new_packages = packages(before), packages(after)
        if before != after or set(old_packages) - set(new_packages):
            return False
        added = set(new_packages) - set(old_packages)
        if len(added) != len(LOGGING_PACKAGES) or {name for name, _ in added} != set(LOGGING_PACKAGES):
            return False
        if {name for name, _ in old_packages} & set(LOGGING_PACKAGES):
            return False
        for key in added:
            digest = hashlib.sha256(json.dumps(new_packages[key], sort_keys=True).encode()).hexdigest()
            if digest != LOGGING_PACKAGES[key[0]]:
                return False
        for key, old in old_packages.items():
            new = new_packages[key]
            if key[0] == "ggtts-core":
                old = dict(old)
                deps = old.get("dependencies")
                if not isinstance(deps, list) or len(deps) != len(set(deps)):
                    return False
                old["dependencies"] = sorted(deps + ["tracing", "tracing-subscriber"])
                if old["dependencies"] != new.get("dependencies"):
                    return False
            if old != new:
                return False
        return sum(key[0] == "ggtts-core" for key in old_packages) == 1
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def decide(event, base_ref, head_ref, same_repo, paths, workflow_selected=None, logging_lock_safe=False):
    if event != "pull_request":
        # A merge to rc already had its PR checks; promotion runs the other release checks.
        return set()
    if base_ref == "main" and head_ref == "rc" and same_repo:
        # The required on-device gate exercises CUDA at this head; the tag builds its archive.
        return set(JOBS)
    selected = set()
    for path in paths:
        if path == WORKFLOW:
            selected.update(set(JOBS) if workflow_selected is None else workflow_selected)
        if path in ("scripts/select_specialized_ci.py", "scripts/test_select_specialized_ci.py"):
            # CI always runs the selector tests. Promotion runs the release checks, unless another
            # changed path independently needs one of these jobs.
            continue
        if path == "Cargo.lock" and logging_lock_safe:
            selected.add("package")
            continue
        if path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".gitmodules"):
            return set(JOBS)
        selected.update(doc_bindings_jobs(path))
        if path == "scripts/c_examples.sh":
            selected.add("package")
        if python_path(path):
            selected.add("python")
            if path.startswith(PYTHON_CRATE):
                continue
        if path.startswith(GGML_GATED_CRATES) or path in GGML_GATED_FILES:
            # Code behind `feature = "ggml"` is compiled and tested only by the ggml job.
            selected.add("ggml")
        if path.startswith(BOOK_CRATE):
            continue
        if path.startswith("crates/"):
            if path == "crates/ggtts-cli/src/bin/ggtts-render.rs":
                # The renderer contains ggml-gated engine loading and routing. Default CPU
                # tests do not compile those branches; exercise them without selecting CUDA.
                selected.update(("ggml", "package"))
                continue
            if path in MEASUREMENT_PATHS:
                # The package job exercises the actual CLI/instrument packaging without CUDA or Android.
                if path.startswith(("crates/ggtts-cli/", "crates/ggtts-instruments/")):
                    selected.add("package")
                continue
            if not path.startswith(("crates/ggtts-core/", "crates/ggtts-cli/", "crates/ggtts-instruments/",
                                    "crates/ggtts-qwen3-tts/", "crates/ggtts-kitten/", "crates/ggtts-ggml-sys/",
                                    "crates/ggtts-fake/", "crates/ggtts-ffi/", "crates/ggtts-load/")):
                return set(JOBS)
            if path.startswith(("crates/ggtts-ggml-sys/", "crates/ggtts-ffi/")):
                # Direct changes to native build or bindings need their targeted checks.
                selected.update(("binding", "android", "ggml"))
                if path.startswith("crates/ggtts-ffi/"):
                    # The C library and ggtts.h ship in the archives: the package smoke compiles a C program
                    # against them and renders through them.
                    selected.add("package")
            else:
                # Ordinary source PRs get the CPU checks and package smoke; promotion
                # above still runs the CPU native, binding and release jobs.
                selected.add("package")
            if path.startswith("crates/ggtts-load/") and "/tests/" not in path:
                # The C library (and Kotlin and Android over it) loads, refuses and renders through ggtts-load's
                # engine glue (`binding`), as the Python binding does: a change to its source runs their jobs.
                selected.update(("binding", "android"))
        if path.startswith(("android/", "scripts/android_build.sh")):
            selected.update(("android", "binding"))
        if path.startswith(("scripts/kotlin_binding_test.sh",)):
            selected.add("binding")
        if path.startswith(("scripts/package_and_smoke.sh", "scripts/check_release_version.py", "instruments/", "fixtures/",
                            "scripts/ffi_smoke.c", "scripts/check_pe_imports.py", "scripts/test_check_pe_imports.py")):
            selected.add("package")
        if path == "scripts/buildinfo_engines.sh":
            selected.update(("ggml", "release"))
        if path == "scripts/check_glibc_floor.py":
            # The release build runs it on the real binaries; its tests run in the selection job on every run.
            selected.add("release")
        if path in (".github/workflows/release.yml", "scripts/package_cuda_release.sh", "scripts/release_smoke.sh"):
            # release_smoke.sh: package-smoke runs its checks on the fake-engine archive, release-build its real render.
            selected.update(("package", "release"))
    return selected


RELEASE_WORKFLOW = ".github/workflows/release.yml"
GGML_SUBMODULE = "crates/ggtts-ggml-sys/ggml"
# The scripts the CUDA archive is built and smoked with. A pull request into rc that changes one runs the CUDA compile,
# so the change is proven before it merges rather than first on the push to rc. No other pull request into rc runs it.
CUDA_SCRIPTS = ("scripts/package_cuda_release.sh", "scripts/release_smoke.sh")


def cuda_push_path(path):
    """A path whose change can break the CUDA archive build.

    A crate's build script, anything in the ggml bindings crate (its submodule pointer included), any manifest, the
    lockfile, the script the archive is built with, and the workflows.
    """
    return (path in ("Cargo.lock", ".gitmodules", "Cargo.toml") or path in CUDA_SCRIPTS
            or path.endswith("/Cargo.toml")
            or path == GGML_SUBMODULE or path.startswith("crates/ggtts-ggml-sys/")
            or path.startswith(".github/workflows/")
            or (path.startswith("crates/") and path.endswith("/build.rs")))


def cuda_decision(event, base_ref, ref, paths):
    """Whether the hosted CUDA archive compile runs. It is a separate output, never part of JOBS.

    It runs on every pull request into main (the promotion), on a pull request into rc that changes one of
    CUDA_SCRIPTS, and on a push to rc that touches a cuda_push_path. It never runs on any other pull request, whatever
    that pull request changes. `paths` is the pull request's changed paths, or the pushed range's; for a push it is
    None when the range could not be read, which runs the compile. Each push is judged by its own range, so ci.yml must
    never cancel or supersede a push run (its concurrency group is per run for pushes; a test holds it to that).
    """
    if event == "pull_request":
        return base_ref == "main" or (base_ref == "rc" and any(path in CUDA_SCRIPTS for path in paths or ()))
    if event == "push" and ref == "refs/heads/rc":
        return paths is None or any(cuda_push_path(path) for path in paths)
    return False


# The tiny-model job: the no-change check on a real Qwen3-TTS engine cut to one layer (crates/ggtts-qwen3-tts/fixtures/
# tiny), on CPU. What it reads: the engine, the core it implements (the engine trait, hooks, trace, pipeline), ggml and
# its bindings (the submodule pointer included), the instruments and the CLI the event check runs them from, the
# fixture's pins and scripts, and the workspace's manifests and toolchain. Documentation never selects it.
TINY_PREFIXES = ("crates/ggtts-qwen3-tts/", "crates/ggtts-core/", "crates/ggtts-ggml-sys/", "crates/ggtts-instruments/",
                 "crates/ggtts-cli/", "instruments/")
TINY_FILES = {"scripts/tiny_fixture.sh", "scripts/cut_tiny_fixture.py", "Cargo.toml", "Cargo.lock",
              "rust-toolchain.toml", ".gitmodules"}
TINY_JOB = "tiny-model"


def tiny_path(path):
    """A path whose change can change what the tiny-model job checks or how it runs."""
    if path in TINY_FILES:
        return True
    return path.startswith(TINY_PREFIXES) and not path.endswith(".md")


def workflow_touches(before, after, job):
    """Whether a ci.yml edit changes `job`'s block, the selection job's, or the header; unreadable is a change."""
    try:
        old_header, old = workflow_sections(before)
        new_header, new = workflow_sections(after)
    except ValueError:
        return True
    return (old_header != new_header or old.get(job) != new.get(job)
            or old.get("specialized-selection") != new.get("specialized-selection"))


def tiny_decision(event, base_ref, head_ref, same_repo, ref, paths, workflow_touched=True):
    """Whether the tiny-model job runs. A separate output, like `cuda`, never part of JOBS.

    It runs on every promotion pull request (rc into main, so the commit being promoted carries a tiny-model result),
    on any other pull request that changes a tiny_path or touches the job in ci.yml (`workflow_touched`), and on a
    push to rc whose range changes a tiny_path or cannot be read (`paths` None).
    """
    if event == "pull_request":
        if base_ref == "main" and head_ref == "rc" and same_repo:
            return True
        return any(tiny_path(p) or (p == WORKFLOW and workflow_touched) for p in paths or ())
    if event == "push" and ref == "refs/heads/rc":
        return paths is None or any(tiny_path(p) for p in paths)
    return False


def pushed_paths(before, after):
    """The paths a push changed, or None when the range cannot be read (a new branch, a rewritten history)."""
    if not before or not after or set(before) == {"0"}:
        return None
    try:
        return os.fsdecode(subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", before, after, "--"], stderr=subprocess.DEVNULL
        )).split("\0")
    except (OSError, subprocess.CalledProcessError):
        return None


def cuda_recipe(block):
    """A CUDA archive job from its runner to its build command, as lines, without what may differ.

    What may differ between ci.yml and release.yml: the job's key, name, needs and if; comments; the value of TAG;
    and anything after the build command (the release uploads the archive). Everything else is the recipe: the
    runner, the timeout, the container image, every step with its name, shell, environment and script.
    """
    lines = []
    for line in block.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or re.match(r"^    (name|needs|if):", line):
            continue
        lines.append(re.sub(r"^(\s*TAG:).*$", r"\1 <tag>", line.rstrip()))
        if stripped.startswith("run:") and "scripts/package_cuda_release.sh" in stripped:
            return lines
    raise ValueError("a CUDA archive job must end its recipe with a run of scripts/package_cuda_release.sh")


def cuda_drift(ci_text, release_text):
    """Differences between ci.yml's cuda-compile and release.yml's cuda-build; empty when they build alike."""
    ci_jobs, release_jobs = workflow_sections(ci_text)[1], workflow_sections(release_text)[1]
    if "cuda-compile" not in ci_jobs:
        return ["ci.yml has no cuda-compile job"]
    if "cuda-build" not in release_jobs:
        return ["release.yml has no cuda-build job"]
    ci, release = cuda_recipe(ci_jobs["cuda-compile"]), cuda_recipe(release_jobs["cuda-build"])
    return [line for line in difflib.unified_diff(release, ci, "release.yml cuda-build", "ci.yml cuda-compile",
                                                  lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]


def check_cuda_drift():
    root = Path(__file__).resolve().parent.parent
    drift = cuda_drift((root / WORKFLOW).read_text(), (root / RELEASE_WORKFLOW).read_text())
    for line in drift:
        print("CUDA build drift: " + line, file=sys.stderr)
    if not drift:
        print("cuda-compile is release.yml's cuda-build, line for line, from the runner to the build command",
              file=sys.stderr)
    return 1 if drift else 0


def main():
    if sys.argv[1:] == ["--check-cuda-drift"]:
        sys.exit(check_cuda_drift())
    event = os.environ["EVENT"]
    base_ref = os.environ.get("BASE_REF", "")
    head_ref = os.environ.get("HEAD_REF", "")
    same_repo = bool(os.environ.get("REPOSITORY")) and os.environ.get("REPOSITORY") == os.environ.get("HEAD_REPO")
    paths = []
    workflow_selected = None
    workflow_touched_tiny = True
    logging_lock_safe = False
    if event == "pull_request" and not (base_ref == "main" and head_ref == "rc" and same_repo):
        base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
        # --no-renames includes both the old and new paths. Failure is fatal.
        paths = os.fsdecode(subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"]
        )).split("\0")
        if WORKFLOW in paths or "Cargo.lock" in paths:
            merge_base = subprocess.check_output(["git", "merge-base", base, head], text=True).strip()
        if WORKFLOW in paths:
            old = subprocess.check_output(["git", "show", f"{merge_base}:{WORKFLOW}"], text=True)
            new = subprocess.check_output(["git", "show", f"{head}:{WORKFLOW}"], text=True)
            workflow_selected = workflow_changes(old, new)
            workflow_touched_tiny = workflow_touches(old, new, TINY_JOB)
        if "Cargo.lock" in paths and CORE_MANIFEST in paths:
            manifests = [path for path in paths if path.endswith("Cargo.toml")]
            if set(manifests) <= {CORE_MANIFEST, CLI_MANIFEST, EVENT_MANIFEST}:
                try:
                    def blob(revision, path):
                        return subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
                    cli = (blob(merge_base, CLI_MANIFEST), blob(head, CLI_MANIFEST)) if CLI_MANIFEST in paths else (None, None)
                    event_blobs = (blob(merge_base, EVENT_MANIFEST), blob(head, EVENT_MANIFEST)) if EVENT_MANIFEST in paths else (None, None)
                    logging_lock_safe = logging_lockfile_change(
                        blob(merge_base, "Cargo.lock"), blob(head, "Cargo.lock"),
                        blob(merge_base, CORE_MANIFEST), blob(head, CORE_MANIFEST), *cli, *event_blobs)
                except (OSError, subprocess.CalledProcessError, UnicodeError):
                    logging_lock_safe = False
    selected = decide(event, base_ref, head_ref, same_repo, paths, workflow_selected, logging_lock_safe)
    for job in JOBS:
        print(f"{job}={'true' if job in selected else 'false'}")
    ref = os.environ.get("REF", "")
    pushed = pushed_paths(os.environ.get("PUSH_BEFORE", ""), os.environ.get("PUSH_AFTER", "")) \
        if event == "push" and ref == "refs/heads/rc" else paths
    cuda = cuda_decision(event, base_ref, ref, pushed)
    print(f"cuda={'true' if cuda else 'false'}")
    if cuda:
        print("CUDA archive compile selected", file=sys.stderr)
    tiny = tiny_decision(event, base_ref, head_ref, same_repo, ref, pushed, workflow_touched_tiny)
    print(f"tiny={'true' if tiny else 'false'}")
    if tiny:
        print("tiny-model no-change check selected", file=sys.stderr)
    print("Selected: " + (", ".join(sorted(selected)) or "none") + "; skipped is not passed", file=sys.stderr)


if __name__ == "__main__":
    main()
