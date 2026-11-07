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

    def test_event_and_path_rules(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        self.assertEqual(select(["docs/readme.md"]), set())
        self.assertEqual(select(["android/demo.kt"]), {"android", "binding"})
        self.assertEqual(select(["scripts/package_and_smoke.sh"]), {"package"})
        self.assertEqual(select([".github/workflows/release.yml"]), {"package", "release", "cuda"})
        self.assertEqual(select(["scripts/package_cuda_release.sh"]), {"package", "release", "cuda"})
        self.assertEqual(select(["scripts/buildinfo_engines.sh"]), {"ggml", "release"})
        self.assertEqual(select(["crates/ggtts-qwen3-tts/src/lib.rs"]), {"binding", "android", "release", "ggml", "cuda"})
        for path in (".github/workflows/ci.yml", "scripts/select_specialized_ci.py",
                     "scripts/test_select_specialized_ci.py", "Cargo.lock"):
            self.assertEqual(select([path]), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", True, []), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", False, ["docs/x"]), set())
        self.assertEqual(decide("push", "", "", True, ["Cargo.lock"]), {"ggml", "cuda"})

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
