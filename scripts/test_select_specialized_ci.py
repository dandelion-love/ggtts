import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from select_specialized_ci import JOBS, decide, workflow_changes, workflow_sections


class SelectionTest(unittest.TestCase):
    def test_cuda_checkout_has_git_and_initializes_submodule(self):
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
        self.assertEqual(select([".github/workflows/release.yml"]), {"package", "release", "cuda"})
        self.assertEqual(select(["scripts/package_cuda_release.sh"]), {"package", "release", "cuda"})
        self.assertEqual(select(["scripts/buildinfo_engines.sh"]), {"ggml", "release"})
        self.assertEqual(select(["crates/ggtts-qwen3-tts/src/lib.rs"]), {"binding", "android", "release", "ggml", "cuda"})
        for path in (".github/workflows/ci.yml", "Cargo.lock"):
            self.assertEqual(select([path]), set(JOBS))
        self.assertEqual(select(["scripts/select_specialized_ci.py", "scripts/test_select_specialized_ci.py"]), set())
        self.assertEqual(select(["scripts/select_specialized_ci.py", "scripts/package_cuda_release.sh"]),
                         {"package", "release", "cuda"})
        self.assertEqual(decide("pull_request", "main", "rc", True, []), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", False, ["docs/x"]), set())
        self.assertEqual(decide("push", "", "", True, ["Cargo.lock"]), {"ggml", "cuda"})

    def test_report_provenance_does_not_select_unrelated_release_builds(self):
        paths = ["crates/ggtts-cli/src/bin/ggtts-measure.rs", "crates/ggtts-cli/src/bin/ggtts-render.rs",
                 "crates/ggtts-cli/tests/measure_cli.rs", "crates/ggtts-core/src/report.rs",
                 "crates/ggtts-core/tests/report.rs", "crates/ggtts-instruments/src/runner.rs",
                 "crates/ggtts-instruments/tests/runner.rs", "instruments/python/ggtts_instrument/envelope.py",
                 "schemas/ggtts-report-2.schema.json"]
        self.assertEqual(decide("pull_request", "rc", "feature", True, paths), {"package"})
        self.assertEqual(decide("pull_request", "main", "rc", True, paths), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/unknown/src/lib.rs"]), set(JOBS))
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-instruments/tests/no_change.rs"]), {"package"})
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-qwen3-tts/tests/frame_hook.rs"]), set())
        self.assertEqual(decide("pull_request", "rc", "feature", True,
                                ["crates/ggtts-cli/tests/no_change_real.rs"]), {"package"})

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
        base = "name: CI\njobs:\n  package-smoke:\n    run: old\n  cuda-compile:\n    run: same\n  ggml:\n    run: same\n"
        self.assertEqual(workflow_changes(base, base.replace("run: old", "run: new")), {"package"})
        self.assertEqual(workflow_changes(base, base.replace("  cuda-compile:\n    run: same",
                                                             "  cuda-compile:\n    run: new")), {"cuda"})
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
            result = subprocess.run(["python3", str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=false"), 6)
            (Path(d) / "scripts").mkdir()
            (Path(d) / "scripts" / "android_build.sh").write_text("build")
            git("add", ".")
            git("commit", "-qm", "android")
            env["HEAD_SHA"] = git("rev-parse", "HEAD")
            result = subprocess.run(["python3", str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("android=true", result.stdout)
            self.assertIn("binding=true", result.stdout)
            self.assertIn("release=false", result.stdout)
            workflow_dir = Path(d) / ".github" / "workflows"
            workflow_dir.mkdir(parents=True)
            workflow = workflow_dir / "ci.yml"
            workflow.write_text("name: CI\njobs:\n  package-smoke:\n    run: old\n  cuda-compile:\n    run: same\n")
            git("add", ".")
            git("commit", "-qm", "workflow")
            env["BASE_SHA"] = git("rev-parse", "HEAD")
            workflow.write_text(workflow.read_text().replace("run: old", "run: new"))
            git("commit", "-qam", "package workflow")
            env["HEAD_SHA"] = git("rev-parse", "HEAD")
            result = subprocess.run(["python3", str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("package=true", result.stdout)
            self.assertEqual(result.stdout.count("=true"), 1)
            env["BASE_SHA"] = "invalid"
            self.assertNotEqual(subprocess.run(["python3", str(script)], cwd=d, env=env,
                                               capture_output=True).returncode, 0)
            env.update(BASE_REF="main", HEAD_REF="rc")
            result = subprocess.run(["python3", str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=true"), 6)


if __name__ == "__main__":
    unittest.main()
