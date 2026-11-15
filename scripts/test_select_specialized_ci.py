import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO

from select_specialized_ci import GGML_GATED_CRATES, GGML_GATED_FILES
from select_specialized_ci import cuda_decision, cuda_drift, cuda_push_path, pushed_paths
from select_specialized_ci import TINY_JOB, tiny_decision, tiny_path, workflow_touches
from select_specialized_ci import JOBS, LOGGING_PACKAGES, decide, logging_lockfile_change, main, tomllib, workflow_changes, workflow_sections


def lock_text(packages, version=4):
    """Build a minimal lockfile using JSON strings, which are valid TOML strings."""
    lines = [f"version = {version}"]
    for package in packages:
        lines.append("[[package]]")
        for key, value in package.items():
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines)


def logging_fixtures():
    root = Path(__file__).resolve().parent.parent
    packages = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    added = [p for p in packages if p["name"] in LOGGING_PACKAGES]
    base = [{"name": "ggtts-core", "version": "0.0.1", "dependencies": ["serde_json"]},
            {"name": "serde_json", "version": "1.0.154", "source": "registry", "checksum": "fixture"}]
    head_core = dict(base[0], dependencies=["serde_json", "tracing", "tracing-subscriber"])
    return (lock_text(base), lock_text([head_core, base[1], *added]),
            '[dependencies]\nserde_json = "1"\n',
            '[dependencies]\nserde_json = "1"\ntracing = "0.1.44"\ntracing-subscriber = "0.3.23"\n')


class SelectionTest(unittest.TestCase):
    def test_cuda_compile_runs_on_promotion_and_build_affecting_rc_pushes_only(self):
        # Every pull request into main, whatever it changes; no pull request into any other branch.
        self.assertTrue(cuda_decision("pull_request", "main", "refs/pull/1/merge", []))
        for base in ("rc", "feature", ""):
            self.assertFalse(cuda_decision("pull_request", base, "refs/pull/1/merge",
                                           ["Cargo.lock", ".github/workflows/ci.yml", "crates/ggtts-ggml-sys/build.rs"]))
            self.assertFalse(cuda_decision("pull_request", base, "refs/pull/1/merge", []))
        # A pull request into rc that changes a script the CUDA archive is built or smoked with runs the compile, so the
        # change is proven before it merges; into any other branch it does not.
        for path in ("scripts/package_cuda_release.sh", "scripts/release_smoke.sh"):
            self.assertTrue(cuda_decision("pull_request", "rc", "refs/pull/1/merge", ["README.md", path]), path)
            self.assertFalse(cuda_decision("pull_request", "feature", "refs/pull/1/merge", [path]), path)
        for path in ("scripts/package_and_smoke.sh", "scripts/package_cuda_release.sh.bak", "scripts/release_smoke.py"):
            self.assertFalse(cuda_decision("pull_request", "rc", "refs/pull/1/merge", [path]), path)
        # A push to rc, only for a path that can break the CUDA build.
        for path in ("Cargo.lock", "crates/ggtts-ggml-sys/ggml", ".gitmodules", ".github/workflows/ci.yml",
                     ".github/workflows/release.yml", "crates/ggtts-ggml-sys/build.rs", "crates/ggtts-core/build.rs",
                     "scripts/package_cuda_release.sh", "scripts/release_smoke.sh", "Cargo.toml",
                     "crates/ggtts-cli/Cargo.toml",
                     "instruments/templates/rust/event/Cargo.toml", "crates/ggtts-ggml-sys/src/lib.rs",
                     "crates/ggtts-ggml-sys/src/bindings/x86_64-unknown-linux-gnu.rs"):
            self.assertTrue(cuda_push_path(path), path)
            self.assertTrue(cuda_decision("push", "", "refs/heads/rc", ["README.md", path]), path)
            self.assertFalse(cuda_decision("push", "", "refs/heads/main", [path]), path)
            self.assertFalse(cuda_decision("push", "", "refs/heads/feature", [path]), path)
        for path in ("README.md", "crates/ggtts-core/src/lib.rs", "scripts/build.rs", "docs/Cargo.toml.md",
                     "crates/ggtts-ggml-sys-notes.md", "docs/build.rs.md", "scripts/package_and_smoke.sh",
                     "scripts/package_cuda_release.sh.bak"):
            self.assertFalse(cuda_push_path(path), path)
            self.assertFalse(cuda_decision("push", "", "refs/heads/rc", [path]), path)
        self.assertFalse(cuda_decision("push", "", "refs/heads/rc", []))
        # An unreadable pushed range runs the compile; it is never read as "nothing changed".
        self.assertTrue(cuda_decision("push", "", "refs/heads/rc", None))
        self.assertIsNone(pushed_paths("0" * 40, "a" * 40))
        self.assertIsNone(pushed_paths("", "a" * 40))
        self.assertIsNone(pushed_paths("not-a-commit", "HEAD"))
        self.assertFalse(cuda_decision("workflow_dispatch", "main", "refs/heads/rc", ["Cargo.lock"]))

    def test_tiny_model_runs_when_the_engine_ggml_or_the_fixture_changes(self):
        engine = ("crates/ggtts-qwen3-tts/src/talker.rs", "crates/ggtts-qwen3-tts/fixtures/tiny/SOURCE",
                  "crates/ggtts-qwen3-tts/tests/tiny_model.rs", "crates/ggtts-core/src/hooks.rs",
                  "crates/ggtts-ggml-sys/ggml", "crates/ggtts-ggml-sys/build.rs",
                  "crates/ggtts-instruments/src/runner.rs", "crates/ggtts-cli/tests/no_change_real.rs",
                  "instruments/examples/python/loudness-ends/instrument.py", "scripts/tiny_fixture.sh",
                  "scripts/cut_tiny_fixture.py", "Cargo.lock", "Cargo.toml", "rust-toolchain.toml", ".gitmodules")
        for path in engine:
            self.assertTrue(tiny_path(path), path)
            self.assertTrue(tiny_decision("pull_request", "rc", "feature", True, "refs/pull/1/merge",
                                          ["README.md", path]), path)
            self.assertTrue(tiny_decision("push", "", "", False, "refs/heads/rc", ["README.md", path]), path)
            self.assertFalse(tiny_decision("push", "", "", False, "refs/heads/main", [path]), path)
            self.assertFalse(tiny_decision("workflow_dispatch", "rc", "", False, "refs/heads/rc", [path]), path)
        unrelated = ("README.md", "crates/ggtts-qwen3-tts/README.md", "fixtures/tiny/SOURCE",
                     "crates/ggtts-kitten/src/lib.rs", "crates/ggtts-py/src/core.rs", "android/app/build.gradle",
                     "scripts/android_build.sh", "scripts/rc_gate.py", "docs/tiny.md")
        for path in unrelated:
            self.assertFalse(tiny_path(path), path)
            self.assertFalse(tiny_decision("pull_request", "rc", "feature", True, "refs/pull/1/merge", [path]), path)
            self.assertFalse(tiny_decision("push", "", "", False, "refs/heads/rc", [path]), path)
        # Every promotion runs it, whatever it changes, so the promoted commit carries a result.
        self.assertTrue(tiny_decision("pull_request", "main", "rc", True, "refs/pull/1/merge", []))
        self.assertFalse(tiny_decision("pull_request", "main", "rc", False, "refs/pull/1/merge", []))
        # An unreadable pushed range runs it; it is never read as "nothing changed".
        self.assertTrue(tiny_decision("push", "", "", False, "refs/heads/rc", None))
        self.assertFalse(tiny_decision("push", "", "", False, "refs/heads/rc", []))
        # A ci.yml edit runs it only when it touches the job, the selection job or the shared header.
        wf = ".github/workflows/ci.yml"
        self.assertTrue(tiny_decision("pull_request", "rc", "f", True, "r", [wf], workflow_touched=True))
        self.assertFalse(tiny_decision("pull_request", "rc", "f", True, "r", [wf], workflow_touched=False))
        base = ("name: CI\njobs:\n  tiny-model:\n    run: old\n  ggml:\n    run: same\n"
                "  specialized-selection:\n    run: s\n")
        self.assertTrue(workflow_touches(base, base.replace("run: old", "run: new"), TINY_JOB))
        self.assertTrue(workflow_touches(base, base.replace("run: s\n", "run: t\n"), TINY_JOB))
        self.assertTrue(workflow_touches(base, base.replace("name: CI", "name: Other"), TINY_JOB))
        self.assertFalse(workflow_touches(base, base.replace("run: same", "run: new"), TINY_JOB))
        self.assertTrue(workflow_touches(base, "not a workflow", TINY_JOB))

    def test_the_tiny_model_job_exists_and_is_gated_on_its_output(self):
        root = Path(__file__).resolve().parent.parent
        jobs = workflow_sections((root / ".github/workflows/ci.yml").read_text())[1]
        self.assertIn(TINY_JOB, jobs)
        self.assertIn("if: needs.specialized-selection.outputs.tiny == 'true'", jobs[TINY_JOB])
        self.assertIn("tiny: ${{ steps.select.outputs.tiny }}", jobs["specialized-selection"])

    def test_cuda_compile_cannot_drift_from_the_release_cuda_build(self):
        root = Path(__file__).resolve().parent.parent
        ci = (root / '.github/workflows/ci.yml').read_text()
        release = (root / '.github/workflows/release.yml').read_text()
        self.assertEqual(cuda_drift(ci, release), [])
        jobs = workflow_sections(ci)[1]
        self.assertIn("needs.specialized-selection.outputs.cuda == 'true'", jobs['cuda-compile'])
        # The check runs in the job every CI run has, not only where the compile is selected.
        self.assertIn("--check-cuda-drift", jobs['specialized-selection'])
        # Planted in either workflow, each of these must be reported.
        plants = {
            "image": ("nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:", "nvidia/cuda:12.9.9-devel-ubuntu22.04@sha256:"),
            "command": ("run: bash scripts/package_cuda_release.sh", "run: bash scripts/package_cuda_release.sh --fast"),
            "tools": ("cmake g++ make python3", "cmake g++ make ninja-build python3"),
            "no submodule init": ("          git submodule update --init crates/ggtts-ggml-sys/ggml\n", ""),
            "toolchain": ("--default-toolchain none", "--default-toolchain stable"),
            "added env": ("        env:\n          TAG:", "        env:\n          GGML_NATIVE: \"1\"\n          TAG:"),
            "no apt retry": ("            sleep 20\n", ""),
            "timeout": ("    timeout-minutes: 90\n", "    timeout-minutes: 30\n"),
            "no pin check": ('          test "$(git -C crates/ggtts-ggml-sys/ggml rev-parse HEAD)" = "$pin"\n', ""),
            "step shell": ("      - name: Fetch the pinned ggml\n", "      - name: Fetch the pinned ggml\n        shell: bash\n"),
        }
        for name, (old, new) in plants.items():
            for which in ("release", "ci"):
                text = release if which == "release" else ci
                blocks = workflow_sections(text)[1]
                job = "cuda-build" if which == "release" else "cuda-compile"
                self.assertIn(old, blocks[job], (name, which))
                planted = text.replace(blocks[job], blocks[job].replace(old, new))
                self.assertNotEqual(planted, text, (name, which))
                drift = cuda_drift(ci, planted) if which == "release" else cuda_drift(planted, release)
                self.assertTrue(drift, (name, which))
        # What may differ does not count: the tag, comments, and steps after the build.
        self.assertEqual(cuda_drift(ci.replace("          TAG: ci\n", "          TAG: other\n"), release), [])
        self.assertEqual(cuda_drift(ci.replace("  cuda-compile:\n", "  cuda-other:\n"), release),
                         ["ci.yml has no cuda-compile job"])

    def test_a_push_run_is_never_cancelled_or_superseded(self):
        root = Path(__file__).resolve().parent.parent
        header = workflow_sections((root / '.github/workflows/ci.yml').read_text())[0]
        # Per-run group for pushes: a shared group would let a later push cancel a pending earlier one, and with it
        # the CUDA selection for that push's range.
        self.assertIn("group: ci-${{ github.event_name == 'pull_request' && github.ref || github.run_id }}", header)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", header)
        self.assertNotIn("cancel-in-progress: true", header)

    def test_cuda_is_not_a_selected_job_and_release_still_builds_cuda(self):
        promotion = decide("pull_request", "main", "rc", True, ["Cargo.lock"])
        self.assertEqual(promotion, set(JOBS))
        self.assertNotIn("cuda", promotion)
        root = Path(__file__).resolve().parent.parent
        ci_jobs = workflow_sections((root / '.github/workflows/ci.yml').read_text())[1]
        release_jobs = workflow_sections((root / '.github/workflows/release.yml').read_text())[1]
        self.assertIn('cuda-compile', ci_jobs)
        self.assertIn('cuda-build', release_jobs)
        self.assertIn('scripts/package_cuda_release.sh', release_jobs['cuda-build'])
        gate = (root / 'scripts/rc_gate.py').read_text()
        self.assertIn('crates/ggtts-qwen3-tts/scripts/cuda_probe.sh', gate)
        self.assertIn('crates/ggtts-kitten/scripts/cuda_probe.sh', gate)
        self.assertIn('instrument_proof(work, scratch', gate)

    def test_release_build_is_the_release_matrix_and_holds_the_glibc_floor(self):
        root = Path(__file__).resolve().parent.parent
        ci = workflow_sections((root / '.github/workflows/ci.yml').read_text())[1]['release-build']
        release = workflow_sections((root / '.github/workflows/release.yml').read_text())[1]['build']

        def matrix(block):
            entries, entry = [], None
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith("- os:"):
                    entry = {"os": stripped.split(":", 1)[1].strip()}
                    entries.append(entry)
                elif entry is not None and stripped.startswith(("target:", "features:")):
                    key, value = stripped.split(":", 1)
                    entry[key] = value.strip()
                elif stripped.startswith("runs-on:"):
                    entry = None
            return entries

        self.assertEqual(matrix(ci), matrix(release))
        linux = [entry for entry in matrix(release) if entry["target"] == "x86_64-linux"]
        # The Linux CPU archive's glibc floor is its runner's; ubuntu-latest (24.04) needed glibc 2.39.
        self.assertEqual(linux, [{"os": "ubuntu-22.04", "target": "x86_64-linux",
                                  "features": "ggtts-cli/ggml,ggtts-ffi/ggml"}])
        # The check reads the packed directory's BUILDINFO "binaries" (pack's own list), never a list kept by hand,
        # so a binary that is packed is always checked; it runs after pack and after BUILDINFO is written.
        for name, block, check in (("ci.yml release-build", ci, "check_glibc_floor.py --packed dist/release-ci"),
                                   ("release.yml build", release, 'check_glibc_floor.py --packed "dist/$name"')):
            self.assertIn(check, block, name)
            self.assertNotIn("check_glibc_floor.py target/release/ggtts-render target/release/ggtts-measure",
                             block, name)
            self.assertLess(block.index("scripts/package_and_smoke.sh pack"), block.index(check), name)
            self.assertLess(block.index("BUILDINFO.json"), block.index(check), name)
        self.assertIn("ggtts-unlisted is an ELF file that BUILDINFO.json's binaries and libraries leave out", ci)
        self.assertIn("--floor GLIBC=2.17", ci)
        # The in-job drift step compares the runner too, not only the features.
        self.assertIn('test "$os" = "$OS"', ci)

    def test_renderer_routes_get_ggml_without_cuda(self):
        renderer = ["crates/ggtts-cli/src/bin/ggtts-render.rs"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, renderer), {"ggml", "package"})
        self.assertNotIn("cuda", decide("pull_request", "rc", "feature", True, renderer))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-cli/src/bin/ggtts-measure.rs"]), {"package"})
        self.assertEqual(decide("pull_request", "main", "rc", True, renderer), set(JOBS))

    def test_release_cuda_checkout_has_git_and_initializes_submodule(self):
        root = Path(__file__).resolve().parent.parent
        for file, job in (("ci.yml", "cuda-compile"), ("release.yml", "cuda-build")):
            text = (root / ".github" / "workflows" / file).read_text()
            block = workflow_sections(text)[1][job]
            install = block.index("apt-get install")
            checkout = block.index("uses: actions/checkout@")
            submodule = block.index("git submodule update --init")
            self.assertLess(install, checkout, file)
            self.assertIn(" git ", block[install:checkout], file)
            self.assertLess(checkout, submodule, file)
            self.assertIn('safe.directory "$GITHUB_WORKSPACE"', block[checkout:submodule], file)
            self.assertIn('safe.directory "$GITHUB_WORKSPACE/crates/ggtts-ggml-sys/ggml"',
                          block[checkout:submodule], file)

    def test_event_and_path_rules(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        self.assertEqual(select(["docs/readme.md"]), set())
        self.assertEqual(select(["android/demo.kt"]), {"android", "binding"})
        self.assertEqual(select(["scripts/package_and_smoke.sh"]), {"package"})
        self.assertEqual(select(["scripts/check_release_version.py"]), {"package"})
        self.assertEqual(select([".github/workflows/release.yml"]), {"package", "release", "python"})
        self.assertEqual(select(["scripts/package_cuda_release.sh"]), {"package", "release"})
        self.assertEqual(select(["scripts/release_smoke.sh"]), {"package", "release"})
        self.assertEqual(select(["scripts/buildinfo_engines.sh"]), {"ggml", "release"})
        self.assertEqual(select(["scripts/check_glibc_floor.py"]), {"release", "python"})
        self.assertEqual(select(["scripts/test_check_glibc_floor.py"]), set())
        self.assertEqual(select(["crates/ggtts-qwen3-tts/src/lib.rs"]), {"ggml", "package", "python"})
        self.assertEqual(select(["crates/ggtts-core/src/pipeline.rs", "crates/ggtts-cli/Cargo.toml"]), {"package", "python"})
        self.assertEqual(select(["crates/ggtts-cli/Cargo.toml"]), {"package"})
        self.assertEqual(select(["crates/ggtts-ffi/src/lib.rs"]), {"binding", "android", "ggml", "package", "python"})
        self.assertEqual(select(["crates/ggtts-ffi/include/ggtts.h"]), {"binding", "android", "ggml", "package", "python"})
        for path in ("scripts/ffi_smoke.c", "scripts/check_pe_imports.py", "scripts/test_check_pe_imports.py"):
            self.assertEqual(select([path]), {"package"}, path)
        self.assertEqual(select(["crates/ggtts-ggml-sys/build.rs"]), {"binding", "android", "ggml", "python"})
        for path in (".github/workflows/ci.yml", "Cargo.lock"):
            self.assertEqual(select([path]), set(JOBS))
        self.assertEqual(select(["scripts/select_specialized_ci.py", "scripts/test_select_specialized_ci.py"]), set())
        self.assertEqual(select(["scripts/select_specialized_ci.py", "scripts/package_cuda_release.sh"]),
                         {"package", "release"})
        self.assertEqual(decide("pull_request", "main", "rc", True, []), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", False, ["docs/x"]), set())
        self.assertEqual(decide("push", "", "", True, ["Cargo.lock"]), set())

    def test_report_provenance_does_not_select_unrelated_release_builds(self):
        paths = ["crates/ggtts-cli/src/bin/ggtts-measure.rs", "crates/ggtts-cli/src/bin/ggtts-render.rs",
                 "crates/ggtts-cli/tests/measure_cli.rs", "crates/ggtts-core/src/report.rs",
                 "crates/ggtts-core/tests/report.rs", "crates/ggtts-instruments/src/runner.rs",
                 "crates/ggtts-instruments/tests/runner.rs", "instruments/python/ggtts_instrument/envelope.py",
                 "schemas/ggtts-report-2.schema.json"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"ggml", "package", "python"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/unknown/src/lib.rs"]), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-instruments/tests/no_change.rs"]), {"package"})
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-qwen3-tts/tests/frame_hook.rs"]), {"ggml"})
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-cli/tests/no_change_real.rs"]), {"ggml", "package"})

    def test_the_book_crate_selects_no_specialized_job(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        for path in ("crates/ggtts-book/src/lib.rs", "crates/ggtts-book/Cargo.toml",
                     "crates/ggtts-book/tests/reader.rs", "crates/ggtts-book/src/bin/ggtts-book.rs"):
            self.assertEqual(select([path]), set(), path)
        # Its new dependencies still change the workspace lockfile, which selects everything.
        self.assertEqual(select(["crates/ggtts-book/src/lib.rs", "Cargo.lock"]), set(JOBS))
        # A crate the selector does not know still selects every job.
        self.assertEqual(select(["crates/ggtts-bookish/src/lib.rs"]), set(JOBS))
        # The promotion still runs everything.
        self.assertEqual(decide("pull_request", "main", "rc", True, ["crates/ggtts-book/src/lib.rs"]), set(JOBS))
        # Except its one ggml-gated file, which selects the ggml job as every gated file does.
        self.assertEqual(select(["crates/ggtts-book/tests/steer_tiny.rs"]), {"ggml"})

    def test_the_loader_crate_selects_the_ggml_job(self):
        # crates/ggtts-load gates its engine code behind `ggml`: only the ggml job compiles and tests it.
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        for path in ("crates/ggtts-load/src/lib.rs", "crates/ggtts-load/tests/load.rs"):
            self.assertIn("ggml", select([path]), path)
            self.assertNotEqual(select([path]), set(JOBS), path)
        # Its engine glue is what the C library, Kotlin, Android and Python load and render through: a change to its
        # source runs each of their jobs; its own tests reach none of them.
        for path in ("crates/ggtts-load/src/binding.rs", "crates/ggtts-load/src/lib.rs", "crates/ggtts-load/Cargo.toml"):
            for job in ("binding", "android", "python", "package"):
                self.assertIn(job, select([path]), (path, job))
        for job in ("binding", "android", "python"):
            self.assertNotIn(job, select(["crates/ggtts-load/tests/load.rs"]), job)

    def test_ggml_gated_code_selects_the_ggml_job(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        for path in ("crates/ggtts-kitten/src/model.rs", "crates/ggtts-kitten/src/refdsp.rs",
                     "crates/ggtts-kitten/tests/parity.rs", "crates/ggtts-qwen3-tts/src/talker.rs",
                     "crates/ggtts-qwen3-tts/src/engine.rs", "crates/ggtts-cli/tests/version_engines.rs",
                     "crates/ggtts-cli/src/bin/ggtts-render.rs", "crates/ggtts-ffi/src/lib.rs",
                     "crates/ggtts-ggml-sys/src/lib.rs"):
            self.assertIn("ggml", select([path]), path)
        # Ungated code does not pay for the ggml build.
        for path in ("crates/ggtts-core/src/pipeline.rs", "crates/ggtts-fake/src/lib.rs",
                     "crates/ggtts-cli/src/bin/ggtts-measure.rs", "crates/ggtts-instruments/src/runner.rs", "README.md"):
            self.assertNotIn("ggml", select([path]), path)
        # Every file in the tree that gates on the feature is covered, so a new gated file cannot go untested.
        root = Path(__file__).resolve().parent.parent
        gated = [p.relative_to(root).as_posix() for p in (root / "crates").rglob("*.rs")
                 if "ggml-sys/ggml/" not in p.as_posix() and "/target/" not in p.as_posix()
                 and 'feature = "ggml"' in p.read_text(encoding="utf-8", errors="replace")]
        self.assertTrue(any(p.startswith(GGML_GATED_CRATES) for p in gated))
        self.assertTrue(GGML_GATED_FILES <= set(gated), sorted(GGML_GATED_FILES - set(gated)))
        for path in gated:
            # The Python binding is its own workspace; the python job compiles its gated code with ggml on Linux.
            job = "python" if path.startswith("crates/ggtts-py/") else "ggml"
            self.assertIn(job, select([path]), path + f" gates on the ggml feature but does not select the {job} job")

    def test_the_binding_docs_select_the_jobs_that_run_their_examples(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        self.assertEqual(select(["docs/bindings/examples/c/first_render.c"]), {"package"})
        self.assertEqual(select(["docs/bindings/examples/kotlin/Stream.kt"]), {"binding"})
        self.assertEqual(select(["docs/bindings/examples/python/stream.py"]), {"python"})
        for path in ("docs/bindings/README.md", "docs/bindings/c.md", "docs/bindings/examples/README.md"):
            self.assertEqual(select([path]), {"package", "binding", "python"}, path)
        self.assertEqual(select(["scripts/c_examples.sh"]), {"package"})
        self.assertEqual(select(["scripts/kotlin_binding_test.sh"]), {"binding"})
        self.assertEqual(select(["crates/ggtts-py/tests/test_doc_examples.py"]), {"python"})
        # Other docs select nothing; the check job holds every binding page's blocks to their files on every run.
        self.assertEqual(select(["docs/android/binding-api.md", "docs/bindingsX.md"]), set())
        self.assertEqual(select(["scripts/check_doc_examples.py"]), set())

    def test_the_python_binding_and_what_it_links_select_the_python_job(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        for path in ("crates/ggtts-py/src/core.rs", "crates/ggtts-py/python/ggtts/__init__.py",
                     "crates/ggtts-py/tests/test_ggtts.py", "crates/ggtts-py/Cargo.toml", "crates/ggtts-py/Cargo.lock",
                     "crates/ggtts-py/pyproject.toml"):
            self.assertEqual(select([path]), {"python"}, path)
        for path in ("crates/ggtts-core/src/pipeline.rs", "crates/ggtts-fake/src/lib.rs", "crates/ggtts-kitten/src/flow.rs",
                     "crates/ggtts-qwen3-tts/Cargo.toml", "crates/ggtts-ggml-sys/build.rs", "crates/ggtts-ffi/src/lib.rs",
                     "crates/ggtts-models/src/lib.rs"):
            self.assertIn("python", select([path]), path)
        # The release wheel's scripts and release.yml, whose wheel jobs run only on a tag, are proven by the python job.
        for path in ("scripts/python_wheel.sh", "scripts/check_python_lock.py", "scripts/test_check_python_lock.py",
                     "scripts/third_party_notices.py", "licenses/python-crates.txt", "scripts/check_glibc_floor.py",
                     ".github/workflows/release.yml"):
            self.assertIn("python", select([path]), path)
        self.assertEqual(select(["scripts/python_wheel.sh"]), {"python"})
        # The engines' own tests, the CLI and the docs do not reach the binding.
        for path in ("crates/ggtts-qwen3-tts/tests/frame_hook.rs", "crates/ggtts-core/tests/report.rs",
                     "crates/ggtts-cli/src/bin/ggtts-measure.rs", "README.md", "android/demo.kt"):
            self.assertNotIn("python", select([path]), path)
        self.assertIn("python", decide("pull_request", "main", "rc", True, []))

    def test_logging_change_gets_lightweight_pr_and_cpu_promotion(self):
        paths = ["crates/ggtts-cli/Cargo.toml", "crates/ggtts-cli/src/bin/ggtts-render.rs",
                 "crates/ggtts-cli/tests/logging_cli.rs", "crates/ggtts-core/src/lib.rs",
                 "crates/ggtts-core/src/logging.rs", "crates/ggtts-core/src/pipeline.rs",
                 "crates/ggtts-core/src/trace.rs", "crates/ggtts-core/tests/trace.rs",
                 "crates/ggtts-qwen3-tts/src/lib.rs", "crates/ggtts-qwen3-tts/src/native_logging.rs",
                 "crates/ggtts-instruments/src/runner.rs", "scripts/select_specialized_ci.py"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"ggml", "package", "python"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))
        self.assertEqual(decide("push", "", "", True, paths), set())

    @unittest.skipIf(tomllib is None, "TOML parser unavailable; selector takes the full matrix")
    def test_logging_lock_exception_checks_complete_semantics(self):
        before, after, core_before, core_after = logging_fixtures()
        safe = lambda old=before, new=after, oc=core_before, nc=core_after: logging_lockfile_change(old, new, oc, nc)
        self.assertTrue(safe())
        self.assertFalse(safe(new=after.replace('name = "tracing"', 'name = "unreviewed"')))
        self.assertFalse(safe(new=after.replace('version = "0.1.44"', 'version = "0.1.45"')))
        self.assertFalse(safe(new=after.replace('name = "tracing-core"', 'name = "tracing-core2"')))
        self.assertFalse(safe(new=after.replace('checksum = "fixture"', 'checksum = "changed"')))
        self.assertFalse(safe(new=after + '\n[[package]]\nname = "cc"\nversion = "1.0"\n'))
        self.assertFalse(safe(new=after.replace('"tracing-subscriber"]', '"native-sys", "tracing-subscriber"]', 1)))
        self.assertFalse(safe(new=after.replace('name = "valuable"', 'name = "valuable2"')))
        self.assertFalse(safe(new=after + '\n[[package]]\nname = "tracing"\nversion = "0.1.44"\n'))
        self.assertFalse(safe(new=after.replace('version = 4', 'version = 3', 1)))
        self.assertFalse(safe(new=after.replace('version = 4', 'version = 99', 1)))
        self.assertFalse(safe(new='not valid = = TOML'))
        self.assertFalse(safe(nc=core_after + 'cc = "1"\n'))
        cli_before = '[dependencies]\n[dev-dependencies]\nserde_json = {workspace = true}\n'
        cli_after = '[dependencies]\nserde_json = {workspace = true}\n[dev-dependencies]\n'
        self.assertTrue(logging_lockfile_change(before, after, core_before, core_after, cli_before, cli_after))
        self.assertFalse(logging_lockfile_change(before, after, core_before, core_after, cli_before,
                                                 cli_after + 'cc = "1"\n'))
        self.assertFalse(logging_lockfile_change(before, after, core_before, core_after, cli_before, None))
        event_before = ('[package]\nname = "ggtts-instrument-event-template"\n'
                        'description = "Template: an out-of-process ggtts event instrument in Rust, reading ggtts-trace/1 and writing ggtts-report/1"\n'
                        '[dependencies]\nserde_json = "1"\n')
        event_after = event_before.replace("reading ggtts-trace/1", "reading ggtts traces")
        compare_event = lambda new: logging_lockfile_change(before, after, core_before, core_after,
                                                            cli_before, cli_after, event_before, new)
        self.assertTrue(compare_event(event_after))
        self.assertFalse(compare_event(event_after.replace('serde_json = "1"', 'serde_json = "2"')))
        self.assertFalse(compare_event(event_after + '[features]\nextra = []\n'))
        self.assertFalse(compare_event(event_after.replace('[dependencies]', 'build = "build.rs"\n[dependencies]')))
        self.assertFalse(compare_event(event_after.replace('ggtts traces', 'different traces')))
        self.assertFalse(compare_event('not valid TOML = ='))
        paths = ["Cargo.lock", "crates/ggtts-core/Cargo.toml"]
        # The core manifest is linked into the Python binding too.
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths, logging_lock_safe=True), {"package", "python"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths, logging_lock_safe=True), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths + ["Cargo.toml"],
                                logging_lock_safe=True), set(JOBS))

    @unittest.skipIf(tomllib is None, "TOML parser unavailable; selector takes the full matrix")
    def test_logging_lock_git_blobs_and_unreadable_fall_back(self):
        before, after, core_before, core_after = logging_fixtures()
        paths = ["Cargo.lock", "crates/ggtts-core/Cargo.toml"]
        blob = {"base:Cargo.lock": before, "head:Cargo.lock": after,
                "base:crates/ggtts-core/Cargo.toml": core_before,
                "head:crates/ggtts-core/Cargo.toml": core_after}
        def git(args, **kwargs):
            if args[1:3] == ["diff", "--no-renames"]:
                return ("\0".join(paths) + "\0").encode()
            if args[1] == "merge-base":
                return "base\n"
            if args[1] == "show":
                if args[2] not in blob:
                    raise subprocess.CalledProcessError(1, args)
                return blob[args[2]]
            raise AssertionError(args)
        env = dict(EVENT="pull_request", BASE_REF="rc", HEAD_REF="branch", BASE_SHA="base", HEAD_SHA="head",
                   REPOSITORY="x/y", HEAD_REPO="x/y")
        with patch.dict(os.environ, env, clear=True), patch("select_specialized_ci.subprocess.check_output", side_effect=git):
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertIn("package=true", stdout.getvalue())
                self.assertIn("python=true", stdout.getvalue())
                # The lockfile is a tiny-model input too (tiny_decision).
                self.assertIn("tiny=true", stdout.getvalue())
                self.assertEqual(stdout.getvalue().count("=true"), 3)
            paths.append(".github/workflows/ci.yml")
            blob["base:.github/workflows/ci.yml"] = "name: CI\njobs:\n  ggml:\n    run: old\n"
            blob["head:.github/workflows/ci.yml"] = "name: CI\njobs:\n  ggml:\n    run: new\n"
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertIn("ggml=true", stdout.getvalue())
                self.assertIn("package=true", stdout.getvalue())
            paths.append("crates/ggtts-ggml-sys/Cargo.toml")
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertEqual(stdout.getvalue().count("=true"), len(JOBS) + 1)
            paths.remove("crates/ggtts-ggml-sys/Cargo.toml")
            paths.remove(".github/workflows/ci.yml")
            event_path = "instruments/templates/rust/event/Cargo.toml"
            paths.append(event_path)
            event_before = ('[package]\nname = "ggtts-instrument-event-template"\n'
                            'description = "Template: an out-of-process ggtts event instrument in Rust, reading ggtts-trace/1 and writing ggtts-report/1"\n'
                            '[dependencies]\nserde_json = "1"\n')
            event_after = event_before.replace("reading ggtts-trace/1", "reading ggtts traces")
            blob["base:" + event_path] = event_before
            blob["head:" + event_path] = event_after
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertEqual(stdout.getvalue().count("=true"), 3)
                self.assertIn("package=true", stdout.getvalue())
                self.assertIn("python=true", stdout.getvalue())
                self.assertIn("tiny=true", stdout.getvalue())
            for altered in (event_after.replace('serde_json = "1"', 'serde_json = "2"'),
                            event_after + '[features]\nextra = []\n',
                            event_after.replace('[dependencies]', 'build = "build.rs"\n[dependencies]')):
                blob["head:" + event_path] = altered
                with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                    main()
                    self.assertEqual(stdout.getvalue().count("=true"), len(JOBS) + 1)
            paths.remove(event_path)
            del blob["base:Cargo.lock"]
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertEqual(stdout.getvalue().count("=true"), len(JOBS) + 1)
    def test_pitch_catalogue_and_package_changes_do_not_select_engine_archives(self):
        paths = ["crates/ggtts-instruments/src/catalogue.rs", "crates/ggtts-core/tests/python_instruments.rs",
                 "instruments/examples/python/pitch-range/instrument.py", "instruments/vendor/pitchy-4.1.0/index.js",
                 "scripts/package_and_smoke.sh", "scripts/select_specialized_ci.py"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"package"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))

    def test_loudness_catalogue_changes_do_not_select_engine_archives(self):
        paths = ["crates/ggtts-instruments/src/catalogue.rs", "crates/ggtts-core/tests/python_instruments.rs",
                 "crates/ggtts-instruments/tests/no_change.rs", "crates/ggtts-instruments/tests/runner.rs",
                 "instruments/examples/python/loudness-ends/instrument.py", "instruments/instruments.yaml",
                 "scripts/test_qwen_event_instrument_probe.py"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"package"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))

    def test_workflow_job_changes(self):
        base = "name: CI\njobs:\n  package-smoke:\n    run: old\n  ggml:\n    run: same\n"
        self.assertEqual(workflow_changes(base, base.replace("run: old", "run: new")), {"package"})
        self.assertEqual(workflow_changes(base, base.replace("  ggml:\n    run: same",
                                                             "  ggml:\n    run: new")), {"ggml"})
        self.assertEqual(workflow_changes(base, base.replace("name: CI", "name: Other")), set(JOBS))
        self.assertEqual(workflow_changes(base, base + "  unfamiliar:\n    run: x\n"), set(JOBS))
        with self.assertRaises(ValueError):
            workflow_changes(base, "not a workflow")

    def test_real_git_diff_and_missing_base(self):
        script = Path(__file__).with_name("select_specialized_ci.py").resolve()
        with tempfile.TemporaryDirectory() as d:
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=d, text=True).strip()
            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci@example.invalid")
            (Path(d) / "README.md").write_text("base")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            (Path(d) / "README.md").write_text("docs")
            git("commit", "-qam", "docs")
            head = git("rev-parse", "HEAD")
            env = dict(os.environ, EVENT="pull_request", BASE_REF="rc", HEAD_REF="branch",
                       BASE_SHA=base, HEAD_SHA=head, REPOSITORY="x/y", HEAD_REPO="x/y")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=false"), len(JOBS) + 2)
            self.assertIn("cuda=false", result.stdout)
            self.assertIn("tiny=false", result.stdout)
            (Path(d) / "scripts").mkdir()
            (Path(d) / "scripts" / "android_build.sh").write_text("build")
            git("add", ".")
            git("commit", "-qm", "android")
            env["HEAD_SHA"] = git("rev-parse", "HEAD")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("android=true", result.stdout)
            self.assertIn("binding=true", result.stdout)
            self.assertIn("release=false", result.stdout)
            workflow_dir = Path(d) / ".github" / "workflows"
            workflow_dir.mkdir(parents=True)
            workflow = workflow_dir / "ci.yml"
            workflow.write_text("name: CI\njobs:\n  package-smoke:\n    run: old\n  ggml:\n    run: same\n")
            git("add", ".")
            git("commit", "-qm", "workflow")
            env["BASE_SHA"] = git("rev-parse", "HEAD")
            workflow.write_text(workflow.read_text().replace("run: old", "run: new"))
            git("commit", "-qam", "package workflow")
            env["HEAD_SHA"] = git("rev-parse", "HEAD")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("package=true", result.stdout)
            self.assertEqual(result.stdout.count("=true"), 1)
            env["BASE_SHA"] = "invalid"
            self.assertNotEqual(subprocess.run([sys.executable, str(script)], cwd=d, env=env,
                                               capture_output=True).returncode, 0)
            env.update(BASE_REF="main", HEAD_REF="rc")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=true"), len(JOBS) + 2)
            self.assertIn("cuda=true", result.stdout)
            self.assertIn("tiny=true", result.stdout)
            # A push to rc: the compile is selected by the pushed range, and nothing else is.
            push = dict(os.environ, EVENT="push", REF="refs/heads/rc", PUSH_BEFORE=base, PUSH_AFTER=head)
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=true"), 0)
            self.assertIn("cuda=false", result.stdout)
            push["PUSH_AFTER"] = git("rev-parse", "HEAD")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertIn("cuda=true", result.stdout.splitlines())
            # The pushed range touches neither the engine nor the fixture, so the tiny-model job is not selected.
            self.assertIn("tiny=false", result.stdout.splitlines())
            self.assertEqual(result.stdout.count("=true"), 1)
            push["REF"] = "refs/heads/main"
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertIn("cuda=false", result.stdout)
            push.update(REF="refs/heads/rc", PUSH_BEFORE="0" * 40)
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertIn("cuda=true", result.stdout)
            self.assertIn("tiny=true", result.stdout)


# CI invokes this suite explicitly; include the logging probe launch test.
from test_logging_overhead_probe import SpeakerPassthroughTest


if __name__ == "__main__":
    unittest.main()
