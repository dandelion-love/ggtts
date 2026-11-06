import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from select_specialized_ci import JOBS, decide


class SelectionTest(unittest.TestCase):
    def test_event_and_path_rules(self):
        select = lambda paths: decide("pull_request", "rc", "feature", True, paths)
        self.assertEqual(select(["docs/readme.md"]), set())
        self.assertEqual(select(["android/demo.kt"]), {"android", "binding"})
        self.assertEqual(select(["scripts/package_and_smoke.sh"]), {"package"})
        self.assertEqual(select([".github/workflows/release.yml"]), {"package", "release"})
        self.assertEqual(select(["crates/ggtts-qwen3-tts/src/lib.rs"]), {"binding", "android", "release"})
        for path in (".github/workflows/ci.yml", "scripts/select_specialized_ci.py",
                     "scripts/test_select_specialized_ci.py", "Cargo.lock"):
            self.assertEqual(select([path]), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", True, []), set(JOBS))
        self.assertEqual(decide("pull_request", "main", "rc", False, ["docs/x"]), set())
        self.assertEqual(decide("push", "", "", True, ["Cargo.lock"]), set())

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
            self.assertEqual(result.stdout.count("=false"), 4)
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
            env["BASE_SHA"] = "invalid"
            self.assertNotEqual(subprocess.run(["python3", str(script)], cwd=d, env=env,
                                               capture_output=True).returncode, 0)
            env.update(BASE_REF="main", HEAD_REF="rc")
            result = subprocess.run(["python3", str(script)], cwd=d, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("=true"), 4)


if __name__ == "__main__":
    unittest.main()
