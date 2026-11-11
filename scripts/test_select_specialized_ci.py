import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO

from select_specialized_ci import cuda_decision, cuda_drift, cuda_push_path, pushed_paths
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
        # A push to rc, only for a path that can break the CUDA build.
        for path in ("Cargo.lock", "crates/ggtts-ggml-sys/ggml", ".gitmodules", ".github/workflows/ci.yml",
                     ".github/workflows/release.yml", "crates/ggtts-ggml-sys/build.rs", "crates/ggtts-core/build.rs",
                     "scripts/package_cuda_release.sh", "Cargo.toml", "crates/ggtts-cli/Cargo.toml",
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
        self.assertEqual(select([".github/workflows/release.yml"]), {"package", "release"})
        self.assertEqual(select(["scripts/package_cuda_release.sh"]), {"package", "release"})
        self.assertEqual(select(["scripts/buildinfo_engines.sh"]), {"ggml", "release"})
        self.assertEqual(select(["crates/ggtts-qwen3-tts/src/lib.rs"]), {"package"})
        self.assertEqual(select(["crates/ggtts-core/src/pipeline.rs", "crates/ggtts-cli/Cargo.toml"]), {"package"})
        self.assertEqual(select(["crates/ggtts-ffi/src/lib.rs"]), {"binding", "android", "ggml"})
        self.assertEqual(select(["crates/ggtts-ggml-sys/build.rs"]), {"binding", "android", "ggml"})
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
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"ggml", "package"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/unknown/src/lib.rs"]), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-instruments/tests/no_change.rs"]), {"package"})
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-qwen3-tts/tests/frame_hook.rs"]), set())
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-cli/tests/no_change_real.rs"]), {"package"})

    def test_logging_change_gets_lightweight_pr_and_cpu_promotion(self):
        paths = ["crates/ggtts-cli/Cargo.toml", "crates/ggtts-cli/src/bin/ggtts-render.rs",
                 "crates/ggtts-cli/tests/logging_cli.rs", "crates/ggtts-core/src/lib.rs",
                 "crates/ggtts-core/src/logging.rs", "crates/ggtts-core/src/pipeline.rs",
                 "crates/ggtts-core/src/trace.rs", "crates/ggtts-core/tests/trace.rs",
                 "crates/ggtts-qwen3-tts/src/lib.rs", "crates/ggtts-qwen3-tts/src/native_logging.rs",
                 "crates/ggtts-instruments/src/runner.rs", "scripts/select_specialized_ci.py"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"ggml", "package"})
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
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths, logging_lock_safe=True), {"package"})
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
                self.assertEqual(stdout.getvalue().count("=true"), 1)
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
                self.assertEqual(stdout.getvalue().count("=true"), len(JOBS))
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
                self.assertEqual(stdout.getvalue().count("=true"), 1)
                self.assertIn("package=true", stdout.getvalue())
            for altered in (event_after.replace('serde_json = "1"', 'serde_json = "2"'),
                            event_after + '[features]\nextra = []\n',
                            event_after.replace('[dependencies]', 'build = "build.rs"\n[dependencies]')):
                blob["head:" + event_path] = altered
                with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                    main()
                    self.assertEqual(stdout.getvalue().count("=true"), len(JOBS))
            paths.remove(event_path)
            del blob["base:Cargo.lock"]
            with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
                main()
                self.assertEqual(stdout.getvalue().count("=true"), len(JOBS))
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
            self.assertEqual(result.stdout.count("=false"), len(JOBS) + 1)
            self.assertIn("cuda=false", result.stdout)
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
            self.assertEqual(result.stdout.count("=true"), len(JOBS) + 1)
            self.assertIn("cuda=true", result.stdout)
            # A push to rc: the compile is selected by the pushed range, and nothing else is.
            push = dict(os.environ, EVENT="push", REF="refs/heads/rc", PUSH_BEFORE=base, PUSH_AFTER=head)
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=true"), 0)
            self.assertIn("cuda=false", result.stdout)
            push["PUSH_AFTER"] = git("rev-parse", "HEAD")
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip().splitlines()[-1], "cuda=true")
            self.assertEqual(result.stdout.count("=true"), 1)
            push["REF"] = "refs/heads/main"
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertIn("cuda=false", result.stdout)
            push.update(REF="refs/heads/rc", PUSH_BEFORE="0" * 40)
            result = subprocess.run([sys.executable, str(script)], cwd=d, env=push, capture_output=True, text=True)
            self.assertIn("cuda=true", result.stdout)


# CI invokes this suite explicitly; include the logging probe launch test.
from test_logging_overhead_probe import SpeakerPassthroughTest


if __name__ == "__main__":
    unittest.main()
