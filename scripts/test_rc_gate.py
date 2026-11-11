import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("rc_gate", Path(__file__).with_name("rc_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


PYTHON_OK = {"interpreter": "python3", "packages": {}, "problems": []}


class GateTest(unittest.TestCase):
    def test_result_never_promotes_missing_or_failed_gpu(self):
        cpu = dict.fromkeys(("fmt", "clippy", "build", "test"), "pass")
        proof = {"cpu": {"status": "pass"}, "cuda": {"status": "pass"}}
        kitten = {"status": "pass", "tq1": {"status": "pass"}}
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": kitten}, proof), "pass")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"},
                                             "kitten": {"status": "incomplete", "tq1": {"status": "pass"}}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "fail"}, "kitten": kitten}, {}), "fail")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": kitten}, {}), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": kitten},
                                       {"cpu": {"status": "pass"}, "cuda": {"status": "fail"}}), "fail")
        # The Kitten probe passing without its TQ1_0 checks is not a pass: unrun or unreported is incomplete.
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "pass"}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"},
                                             "kitten": {"status": "pass", "tq1": {"status": "incomplete"}}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"},
                                             "kitten": {"status": "pass", "tq1": {"status": "fail"}}}, proof), "fail")
        cpu["fmt"] = "fail"
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": kitten}, proof), "fail")

    def test_tq1_entry_passes_only_when_the_checks_ran(self):
        self.assertEqual(gate.tq1_entry(None)["status"], "incomplete")
        self.assertIn("tq1: not run", gate.tq1_entry(None)["reason"])
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "tq1.gguf"
            model.write_text("model")
            sha = gate.file_sha256(model)
            ran = gate.tq1_entry(model, 0, "PASS  TQ1 ternary arm\n== result: ALL PASS\n", sha)
            self.assertEqual((ran["status"], ran["input"]["sha256_tree"]), ("pass", gate.digest(model)))
            self.assertEqual((ran["sha256"], ran["expected_sha256"]), (sha, sha))
            # The checks passed, but nothing says which model they were meant to pass on: incomplete, not a pass.
            unpinned = gate.tq1_entry(model, 0, "== result: ALL PASS\n")
            self.assertEqual((unpinned["status"], unpinned["sha256"], unpinned["expected_sha256"]),
                             ("incomplete", sha, None))
            self.assertIn("no expected sha256", unpinned["reason"])
            # Planted: a model that is not the expected one fails, whatever the probe said.
            wrong = gate.tq1_entry(model, 0, "== result: ALL PASS\n", "0" * 64)
            self.assertEqual(wrong["status"], "fail")
            self.assertIn(sha, wrong["reason"])
            self.assertEqual(gate.tq1_entry(model, None, "", "0" * 64)["status"], "fail")
            skipped = gate.tq1_entry(model, 0, "== result: PASS, TQ1_0 NOT RUN\n")
            self.assertEqual(skipped["status"], "incomplete")
            self.assertIn("tq1: not run", skipped["reason"])
            self.assertEqual(gate.tq1_entry(model, 1, "== result: SOME FAIL\n")["status"], "fail")
            self.assertEqual(gate.tq1_entry(model)["status"], "incomplete")

    def test_hash_changes_on_file_or_name_change(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a"
            f.write_text("one")
            before = gate.digest(d)
            f.write_text("two")
            self.assertNotEqual(before, gate.digest(d))
            before = gate.digest(d)
            f.rename(Path(d) / "b")
            self.assertNotEqual(before, gate.digest(d))

    def test_gate_missing_and_failed_gpu_and_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "rust-toolchain.toml").write_text('channel = "1.2.3"\n')
            paths = ["QWEN3_TTS_CPP", "QWEN3_TTS_BUILD", "TALKER", "CODEC", "GGML_LIB_DIR",
                     "KITTEN_REPO", "S3_GGUF", "REF_LM", "REF_S3", "REF_COND", "VOICE"]
            env = {}
            for name in paths:
                p = root / name
                p.mkdir()
                (p / "fixture").write_text(name)
                env[name] = str(p)
            lib = root / "QWEN3_TTS_BUILD" / "ggml" / "src"
            lib.mkdir(parents=True)
            (lib / "lib.a").write_text("library")
            def output(cmd, **kwargs):
                if cmd[0] == "rustc":
                    return "rustc 1.2.3"
                if "status" in cmd:
                    return ""
                if "rev-parse" in cmd and "--show-toplevel" in cmd:
                    return str(root)
                return "abc123"
            model = root / "tq1.gguf"
            model.write_text("tq1 model")
            seen = {}
            def run_case(missing=False, failing=False, tq1=True, probe_result="ALL PASS"):
                run_env = dict(env)
                if missing:
                    del run_env["VOICE"]
                if tq1:
                    run_env["TQ1_GGUF"] = str(model)
                    run_env["TQ1_GGUF_SHA256"] = gate.file_sha256(model)
                def command(cmd, cwd, log, env=None):
                    if cmd[0] == "bash" and "kitten" in cmd[1]:
                        seen["TQ1_GGUF"] = env.get("TQ1_GGUF")
                        log.parent.mkdir(parents=True, exist_ok=True)
                        log.write_text(f"== result: {probe_result}\n")
                    return int(failing and cmd[0] == "bash" and "kitten" in cmd[1])
                args = ["rc_gate.py", "--work", str(root / "scratch"), "--out", str(root / "report.json"),
                        "--no-reuse", "--cpu-stage"]
                with patch.dict(os.environ, run_env, clear=True), patch.object(gate.sys, "argv", args), \
                     patch.object(gate, "python_requirements", return_value=PYTHON_OK), \
                     patch.object(gate.subprocess, "check_output", side_effect=output), \
                     patch.object(gate, "command", side_effect=command), \
                     patch.object(gate, "instrument_proof", return_value={"cpu": {"status": "pass"},
                                                                           "cuda": {"status": "pass"}}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    code = gate.main()
                return code, json.loads((root / "report.json").read_text())
            code, report = run_case(missing=True)
            self.assertEqual((code, report["result"], report["gpu"]["kitten"]["status"]), (1, "incomplete", "incomplete"))
            code, report = run_case(failing=True)
            self.assertEqual((code, report["result"], report["gpu"]["kitten"]["status"]), (1, "fail", "fail"))
            code, report = run_case()
            self.assertEqual((code, report["result"], report["gpu"]["qwen"]["status"]), (0, "pass", "pass"))
            self.assertEqual(report["commit"], "abc123")
            tq1 = report["gpu"]["kitten"]["tq1"]
            self.assertEqual((tq1["status"], tq1["input"]["sha256_tree"]), ("pass", gate.digest(model)))
            self.assertEqual(seen["TQ1_GGUF"], str(model.resolve()))
            # No TQ1 model: every other check passes, the report says tq1 was not run, and the gate is incomplete.
            code, report = run_case(tq1=False, probe_result="PASS, TQ1_0 NOT RUN")
            kitten = report["gpu"]["kitten"]
            self.assertEqual((code, report["result"], kitten["status"], kitten["tq1"]["status"]),
                             (1, "incomplete", "pass", "incomplete"))
            self.assertIn("tq1: not run", kitten["tq1"]["reason"])
            self.assertIsNone(seen["TQ1_GGUF"])
            # A model was given but the probe's own result line says its TQ1 checks did not run.
            code, report = run_case(probe_result="PASS, TQ1_0 NOT RUN")
            self.assertEqual((code, report["result"], report["gpu"]["kitten"]["tq1"]["status"]), (1, "incomplete", "incomplete"))
            code, report = run_case(failing=True)
            self.assertEqual(report["gpu"]["kitten"]["tq1"]["status"], "fail")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_repo(path):
    """A small repository shaped like this one: an engine crate, the core crate, a lockfile and some docs."""
    git(path.parent, "init", "-q", str(path))
    git(path, "config", "user.email", "gate@example.invalid")
    git(path, "config", "user.name", "gate test")
    git(path, "config", "commit.gpgsign", "false")
    git(path, "config", "core.autocrlf", "false")
    files = {"rust-toolchain.toml": 'channel = "1.2.3"\n', "Cargo.lock": "lock\n", "Cargo.toml": "[workspace]\n",
             "crates/ggtts-core/src/lib.rs": "core\n", "crates/ggtts-qwen3-tts/src/lib.rs": "qwen\n",
             "crates/ggtts-qwen3-tts/scripts/cuda_probe.sh": "probe\n", "crates/ggtts-kitten/src/lib.rs": "kitten\n",
             "crates/ggtts-kitten/scripts/cuda_probe.sh": "probe\n", "docs/notes.md": "notes\n"}
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    return commit(path, "start")


def commit(repo, message, **changes):
    for rel, text in changes.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


QWEN_PATHS = ["Cargo.lock", "Cargo.toml", "crates/ggtts-core", "crates/ggtts-ggml-sys", "crates/ggtts-qwen3-tts",
              "rust-toolchain.toml"]
MACHINE = {"gpu": "NVIDIA GeForce RTX 3080 Ti", "driver": "566.03", "cuda_home": "/usr/local/cuda-12.6",
           "cuda_toolkit": "Cuda compilation tools, release 12.6, V12.6.85", "rustflags": "",
           "cargo_encoded_rustflags": ""}
INPUTS = {"TALKER": {"path": "/models/talker.gguf", "sha256_tree": "aa"}, "QWEN3_TTS_CPP": {"path": "/ref"},
          "reference_commit": "r1", **MACHINE}


def passing(commit_sha, **gpu):
    return {"schema": gate.SCHEMA, "result": "pass", "commit": commit_sha, "gpu": gpu}


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # git objects are read-only on Windows
        self.repo = Path(self.tmp.name) / "repo"
        self.base = make_repo(self.repo)
        self.prev = passing(self.base, qwen={"status": "pass", "inputs": INPUTS, "log": "/old/qwen.log"})

    def tearDown(self):
        self.tmp.cleanup()

    def reuse(self, commit_sha, inputs=INPUTS, previous=None, name="qwen", tq1=None, paths=QWEN_PATHS):
        return gate.reuse_probe(self.repo, previous or self.prev, "/reports/rc-gate.json", name, commit_sha, paths,
                                inputs, tq1)

    def test_an_unchanged_input_set_reuses(self):
        later = commit(self.repo, "docs only", **{"docs/notes.md": "more notes\n"})
        entry, why = self.reuse(later, inputs=dict(INPUTS, TALKER={"path": "/elsewhere", "sha256_tree": "aa"}))
        self.assertIsNone(why)
        self.assertEqual(entry["status"], "pass")
        self.assertEqual(entry["note"], f"reused from {self.base} (/reports/rc-gate.json)")
        self.assertEqual(entry["reused"]["commit"], self.base)
        self.assertEqual(entry["inputs"]["TALKER"]["path"], "/elsewhere")
        self.assertEqual(entry["log"], "/old/qwen.log")

    def test_a_touched_input_reruns(self):
        for rel in ("crates/ggtts-core/src/lib.rs", "crates/ggtts-qwen3-tts/scripts/cuda_probe.sh", "Cargo.lock",
                    "crates/ggtts-ggml-sys/ggml-pin"):
            later = commit(self.repo, "touch " + rel, **{rel: "changed " + rel + "\n"})
            entry, why = self.reuse(later)
            self.assertIsNone(entry, rel)
            self.assertIn(rel, why)

    def test_another_engines_crate_does_not_rerun_this_probe(self):
        later = commit(self.repo, "kitten", **{"crates/ggtts-kitten/src/lib.rs": "kitten 2\n"})
        entry, _ = self.reuse(later)
        self.assertIsNotNone(entry)

    def test_a_changed_model_hash_reruns(self):
        later = commit(self.repo, "docs", **{"docs/notes.md": "x\n"})
        for inputs in (dict(INPUTS, TALKER={"path": "/models/talker.gguf", "sha256_tree": "bb"}),
                       dict(INPUTS, reference_commit="r2"),
                       {k: v for k, v in INPUTS.items() if k != "TALKER"}):
            entry, why = self.reuse(later, inputs=inputs)
            self.assertIsNone(entry)
            self.assertIn("sha256", why)

    def test_a_different_machine_reruns(self):
        # The same commit: only the machine differs, and the reason names what.
        for key, value in (("driver", "570.10"), ("cuda_toolkit", "Cuda compilation tools, release 12.8, V12.8.61"),
                           ("gpu", "NVIDIA GeForce RTX 4090"), ("cuda_home", "/usr/local/cuda-12.8"),
                           ("rustflags", "-C target-cpu=native"), ("cargo_encoded_rustflags", "-Ctarget-cpu=native")):
            entry, why = self.reuse(self.base, inputs=dict(INPUTS, **{key: value}))
            self.assertIsNone(entry, key)
            self.assertIn("the machine differs", why)
            self.assertIn(key, why)
            self.assertIn(value, why)
        # A report from before the machine was recorded, or a machine field left out now, is a difference too.
        entry, why = self.reuse(self.base, inputs={k: v for k, v in INPUTS.items() if k != "driver"})
        self.assertIsNone(entry)
        self.assertIn("driver", why)
        self.assertIsNotNone(self.reuse(self.base)[0])

    def test_a_failed_earlier_report_never_reuses(self):
        later = commit(self.repo, "docs", **{"docs/notes.md": "x\n"})
        report = Path(self.tmp.name) / "rc-gate.json"
        # Planted: the same earlier report, its result failed (and, separately, its probe failed under a passing result).
        report.write_text(json.dumps(dict(self.prev, result="fail")))
        loaded, why = gate.load_previous(report)
        self.assertIsNone(loaded)
        self.assertIn("did not pass", why)
        failed_probe = passing(self.base, qwen={"status": "fail", "inputs": INPUTS})
        entry, why = self.reuse(later, previous=failed_probe)
        self.assertIsNone(entry)
        self.assertIn("did not pass", why)
        entry, why = gate.reuse_probe(self.repo, None, report, "qwen", later, QWEN_PATHS, INPUTS)
        self.assertIsNone(entry)

    def test_a_missing_or_other_schema_report_never_reuses(self):
        report = Path(self.tmp.name) / "rc-gate.json"
        self.assertIn("no earlier report", gate.load_previous(report)[1])
        old = dict(self.prev)
        del old["schema"]  # every report written before reuse existed
        report.write_text(json.dumps(old))
        self.assertIn("schema", gate.load_previous(report)[1])
        report.write_text(json.dumps(dict(self.prev, schema="ggtts-rc-gate/1")))
        self.assertIsNone(gate.load_previous(report)[0])
        report.write_text("{not json")
        self.assertIsNone(gate.load_previous(report)[0])
        report.write_text(json.dumps(self.prev))
        self.assertEqual(gate.load_previous(report)[0]["commit"], self.base)

    def test_a_reused_entry_is_compared_from_where_the_probe_ran(self):
        # The earlier report itself reused the probe from `base`. Planted: core changed between base and that report.
        middle = commit(self.repo, "core", **{"crates/ggtts-core/src/lib.rs": "core 2\n"})
        chained = passing(middle, qwen={"status": "pass", "inputs": INPUTS,
                                        "reused": {"commit": self.base, "report": "/reports/first.json"}})
        entry, why = self.reuse(middle, previous=chained)
        self.assertIsNone(entry)
        self.assertIn("crates/ggtts-core", why)
        # Without that change the origin is carried forward, not the intermediate report.
        repo2 = Path(self.tmp.name) / "repo2"
        first = make_repo(repo2)
        second = commit(repo2, "docs", **{"docs/notes.md": "y\n"})
        chained = passing(second, qwen={"status": "pass", "inputs": INPUTS,
                                        "reused": {"commit": first, "report": "/reports/first.json"}})
        entry, why = gate.reuse_probe(repo2, chained, "/reports/second.json", "qwen", second, QWEN_PATHS, INPUTS)
        self.assertEqual(entry["note"], f"reused from {first} (/reports/first.json)")

    def test_an_unknown_earlier_commit_never_reuses(self):
        entry, why = self.reuse(self.base, previous=passing("0" * 40, qwen={"status": "pass", "inputs": INPUTS}))
        self.assertIsNone(entry)
        self.assertIn("cannot compare", why)

    def test_kitten_reuse_needs_the_same_tq1_model_and_its_expected_hash(self):
        later = commit(self.repo, "docs", **{"docs/notes.md": "x\n"})
        tq1 = {"status": "pass", "sha256": "t1", "expected_sha256": "t1", "input": {"path": "/m", "sha256_tree": "x"}}
        prev = passing(self.base, kitten={"status": "pass", "inputs": INPUTS, "tq1": tq1})
        now = {"status": "incomplete", "sha256": "t1", "expected_sha256": "t1", "input": {"path": "/n", "sha256_tree": "x"}}
        entry, why = self.reuse(later, previous=prev, name="kitten", tq1=now)
        self.assertIsNone(why)
        self.assertEqual((entry["tq1"]["status"], entry["tq1"]["input"]["path"]), ("pass", "/n"))
        for planted in (dict(now, sha256="t2"), dict(now, expected_sha256=None), None):
            entry, why = self.reuse(later, previous=prev, name="kitten", tq1=planted)
            self.assertIsNone(entry, planted)
            self.assertIn("TQ1_0", why)


def check_run(name, conclusion="success", status="completed", app="github-actions"):
    return {"name": name, "status": status, "conclusion": conclusion if status == "completed" else None,
            "app": {"slug": app}}


class CiChecksTest(unittest.TestCase):
    def api(self, overrides):
        def call(path):
            name = next(n for n in gate.REQUIRED_CI_CHECKS if gate._quote(n) in path)
            return {"check_runs": overrides.get(name, [check_run(name)])}
        return call

    def test_green_ci_skips_the_cpu_stage(self):
        ok, detail = gate.ci_cpu_checks("o/r", "abc", self.api({}))
        self.assertTrue(ok)
        self.assertEqual(set(detail["checks"].values()), {"success"})

    def test_anything_but_success_on_a_required_check_is_not_a_pass(self):
        name = gate.REQUIRED_CI_CHECKS[1]
        for planted in ([check_run(name, "failure")], [check_run(name, "skipped")], [check_run(name, "cancelled")],
                        [check_run(name, status="in_progress")], [], [check_run(name, app="someone-else")],
                        [check_run(name), check_run(name, "failure")]):
            ok, detail = gate.ci_cpu_checks("o/r", "abc", self.api({name: planted}))
            self.assertFalse(ok, planted)
            self.assertNotEqual(detail["checks"][name], "success")

    def test_an_unreadable_api_is_not_a_pass(self):
        def broken(path):
            raise subprocess.CalledProcessError(1, ["gh"])
        self.assertFalse(gate.ci_cpu_checks("o/r", "abc", broken)[0])


class PythonRequirementsTest(unittest.TestCase):
    def test_a_missing_or_wrong_package_is_named_with_its_requirements_file(self):
        with tempfile.TemporaryDirectory() as d:
            req = Path(d) / "instruments/req.txt"
            req.parent.mkdir()
            req.write_text("# comment\nggtts-surely-absent-package==1.0\n\n")
            with patch.object(gate, "PYTHON_REQUIREMENTS", ("instruments/req.txt",)):
                result = gate.python_requirements(d, sys.executable)
                self.assertEqual(len(result["problems"]), 1)
                self.assertIn("ggtts-surely-absent-package is not installed", result["problems"][0])
                self.assertIn("instruments/req.txt", result["problems"][0])
                req.write_text("librosa==0.11.0\nnumpy\n")
                fake = subprocess.CompletedProcess([], 0, json.dumps({"librosa": "0.10.2", "numpy": "1.0"}), "")
                with patch.object(gate.subprocess, "run", return_value=fake):
                    result = gate.python_requirements(d, "py")
                self.assertEqual(result["problems"], ["librosa 0.10.2 is installed for py; instruments/req.txt requires 0.11.0"])
                fake = subprocess.CompletedProcess([], 0, json.dumps({"librosa": "0.11.0", "numpy": "2.0"}), "")
                with patch.object(gate.subprocess, "run", return_value=fake):
                    self.assertEqual(gate.python_requirements(d, "py")["problems"], [])

    def test_the_shipped_requirements_files_parse(self):
        root = Path(__file__).resolve().parent.parent
        for rel in gate.PYTHON_REQUIREMENTS:
            self.assertIn("librosa", gate.read_requirements(root / rel))


class MachineTest(unittest.TestCase):
    def test_device_identity_reads_the_query_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "device.log"
            log.write_text("$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader\n"
                           "NVIDIA GeForce RTX 3080 Ti, 566.03\nNVIDIA RTX A2000, 566.03\n")
            self.assertEqual(gate.device_identity(log),
                             ("NVIDIA GeForce RTX 3080 Ti | NVIDIA RTX A2000", "566.03 | 566.03"))
            log.write_text("$ nvidia-smi\nNVIDIA-SMI has failed\n")
            self.assertEqual(gate.device_identity(log), (None, None))
            self.assertEqual(gate.device_identity(Path(d) / "absent.log"), (None, None))

    @unittest.skipUnless(os.name == "posix", "a shell-script nvcc")
    def test_cuda_toolkit_and_machine_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "cuda-12.6"
            (home / "bin").mkdir(parents=True)
            nvcc = home / "bin" / "nvcc"
            nvcc.write_text("#!/bin/sh\necho 'nvcc: NVIDIA (R) Cuda compiler driver'\n"
                            "echo 'Cuda compilation tools, release 12.6, V12.6.85'\necho 'Build cuda_12.6.r12.6/compiler.1'\n")
            nvcc.chmod(0o755)
            self.assertEqual(gate.cuda_toolkit(home),
                             "Cuda compilation tools, release 12.6, V12.6.85 / Build cuda_12.6.r12.6/compiler.1")
            self.assertTrue(gate.cuda_toolkit(Path(d) / "none").startswith("unavailable"))
            link = Path(d) / "cuda"
            link.symlink_to(home)
            log = Path(d) / "device.log"
            log.write_text("NVIDIA GeForce RTX 3080 Ti, 566.03\n")
            with patch.dict(os.environ, {"CUDA_HOME": str(link), "RUSTFLAGS": "-C opt-level=2",
                                         "CARGO_ENCODED_RUSTFLAGS": "-Cdebuginfo=0"}):
                machine = gate.machine_inputs(True, log)
                self.assertEqual(machine, {"gpu": "NVIDIA GeForce RTX 3080 Ti", "driver": "566.03",
                                           "cuda_home": str(home.resolve()), "cuda_toolkit": gate.cuda_toolkit(home),
                                           "rustflags": "-C opt-level=2", "cargo_encoded_rustflags": "-Cdebuginfo=0"})
                self.assertEqual(gate.machine_inputs(False, log)["gpu"], "unavailable")
            self.assertEqual(set(machine), set(gate.MACHINE_KEYS))


class MainTest(unittest.TestCase):
    """The gate end to end on a real repository, with the cargo, probe and GPU commands replaced."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.top = Path(self.tmp.name)
        self.repo = self.top / "repo"
        self.base = make_repo(self.repo)
        names = ["QWEN3_TTS_CPP", "QWEN3_TTS_BUILD", "TALKER", "CODEC", "GGML_LIB_DIR", "KITTEN_REPO", "S3_GGUF",
                 "REF_LM", "REF_S3", "REF_COND", "VOICE"]
        # The real environment (git on Windows needs it), less anything the gate reads.
        self.env = {k: v for k, v in os.environ.items()
                    if k not in names and not k.startswith(("TQ1_", "GGTTS_", "GPU_WAIT"))}
        for name in ["QWEN3_TTS_CPP", "QWEN3_TTS_BUILD", "TALKER", "CODEC", "GGML_LIB_DIR", "KITTEN_REPO", "S3_GGUF",
                     "REF_LM", "REF_S3", "REF_COND", "VOICE"]:
            p = self.top / "inputs" / name
            p.mkdir(parents=True)
            (p / "fixture").write_text(name)
            self.env[name] = str(p)
        lib = self.top / "inputs" / "QWEN3_TTS_BUILD" / "ggml" / "src"
        lib.mkdir(parents=True)
        (lib / "lib.a").write_text("library")
        model = self.top / "inputs" / "tq1.gguf"
        model.write_text("tq1 model")
        self.env.update(TQ1_GGUF=str(model), TQ1_GGUF_SHA256=gate.file_sha256(model))
        self.gpu = "NVIDIA GeForce RTX 3080 Ti, 566.03"
        self.toolkit = "Cuda compilation tools, release 12.6, V12.6.85"

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, *extra, python=PYTHON_OK, ci=(False, {})):
        calls = []
        real = subprocess.check_output

        def output(cmd, **kwargs):
            if cmd[0] == "rustc":
                return "rustc 1.2.3"
            if cmd[:2] == ["git", "rev-parse"] and "--show-toplevel" in cmd:
                return str(self.repo)
            if cmd[:3] == ["git", "-C", self.env["QWEN3_TTS_CPP"]]:
                return "reference-commit"
            return real(cmd, **kwargs)

        def command(cmd, cwd, log, env=None):
            calls.append(cmd)
            if cmd[0] == "nvidia-smi":
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("$ nvidia-smi\n" + self.gpu + "\n")
            if cmd[0] == "bash" and "kitten" in cmd[1]:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("== result: ALL PASS\n")
            return 0

        args = ["rc_gate.py", "--work", str(self.top / "scratch"), "--out", str(self.top / "rc-gate.json"), *extra]
        with patch.dict(os.environ, self.env, clear=True), patch.object(gate.sys, "argv", args), \
             patch.object(gate.subprocess, "check_output", side_effect=output), \
             patch.object(gate, "command", side_effect=command), \
             patch.object(gate, "python_requirements", return_value=python), \
             patch.object(gate, "github_repo", return_value="o/r"), \
             patch.object(gate, "ci_cpu_checks", return_value=ci), \
             patch.object(gate, "crate_paths", side_effect=lambda work, name: {
                 gate.PROBES[name]["script"].split("/scripts")[0]}), \
             patch.object(gate, "cuda_toolkit", side_effect=lambda home: self.toolkit), \
             patch.object(gate, "instrument_proof", return_value={"cpu": {"status": "pass"}, "cuda": {"status": "pass"}}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = gate.main()
        report = json.loads((self.top / "rc-gate.json").read_text())
        probes = [c[1] for c in calls if c[0] == "bash"]
        cargo = [c[1] for c in calls if c[0] == "cargo"]
        return code, report, probes, cargo

    def test_reuse_skips_unchanged_probes_and_reruns_touched_ones(self):
        code, report, probes, _ = self.run_gate()
        self.assertEqual((code, report["result"], report["schema"]), (0, "pass", gate.SCHEMA))
        self.assertEqual(len(probes), 2)
        self.assertIn("no earlier report", report["reuse"]["refused"])
        commit(self.repo, "docs", **{"docs/notes.md": "x\n"})
        code, report, probes, _ = self.run_gate()
        self.assertEqual((code, report["result"], probes), (0, "pass", []))
        for name in ("qwen", "kitten"):
            self.assertTrue(report["gpu"][name]["note"].startswith(f"reused from {self.base} ("), report["gpu"][name])
            self.assertTrue(Path(report["gpu"][name]["reused"]["report"]).is_file())
        commit(self.repo, "kitten", **{"crates/ggtts-kitten/src/lib.rs": "kitten 2\n"})
        code, report, probes, _ = self.run_gate()
        self.assertEqual((code, report["result"]), (0, "pass"))
        self.assertEqual([("kitten" in p) for p in probes], [True])
        self.assertIn("crates/ggtts-kitten", report["gpu"]["kitten"]["reuse_refused"])
        self.assertIn("reused", report["gpu"]["qwen"])
        code, report, probes, _ = self.run_gate("--no-reuse")
        self.assertEqual((code, len(probes), report["reuse"]["refused"]), (0, 2, "--no-reuse"))
        # A changed model hash reruns the probe that reads it.
        (Path(self.env["TALKER"]) / "fixture").write_text("another talker")
        code, report, probes, _ = self.run_gate()
        self.assertEqual([("qwen" in p) for p in probes], [True])
        self.assertIn("sha256", report["gpu"]["qwen"]["reuse_refused"])

    def assert_reruns(self, probes, report, names, reason):
        self.assertEqual(sorted(p.split("/")[1] for p in probes), sorted(f"ggtts-{n}" for n in names))
        for name, crate in (("qwen", "qwen3-tts"), ("kitten", "kitten")):
            if crate in names:
                self.assertIn(reason, report["gpu"][name]["reuse_refused"])
            else:
                self.assertIn("reused", report["gpu"][name])

    def test_the_machine_is_an_input(self):
        names = ("qwen3-tts", "kitten")
        self.run_gate()
        code, report, probes, _ = self.run_gate()
        self.assertEqual((code, probes), (0, []))
        self.assertEqual(report["gpu"]["qwen"]["inputs"]["driver"], "566.03")
        self.assertEqual(report["machine"]["cuda_toolkit"], self.toolkit)
        # The same commit throughout: a driver upgrade, a toolkit upgrade, another card, then RUSTFLAGS.
        self.gpu = "NVIDIA GeForce RTX 3080 Ti, 570.10"
        code, report, probes, _ = self.run_gate()
        self.assert_reruns(probes, report, names, "driver '566.03' -> '570.10'")
        self.toolkit = "Cuda compilation tools, release 12.8, V12.8.61"
        code, report, probes, _ = self.run_gate()
        self.assert_reruns(probes, report, names, "cuda_toolkit")
        self.gpu = "NVIDIA GeForce RTX 4090, 570.10"
        code, report, probes, _ = self.run_gate()
        self.assert_reruns(probes, report, names, "gpu 'NVIDIA GeForce RTX 3080 Ti' -> 'NVIDIA GeForce RTX 4090'")
        self.env["RUSTFLAGS"] = "-C target-cpu=native"
        code, report, probes, _ = self.run_gate()
        self.assert_reruns(probes, report, names, "rustflags")
        self.assertEqual(report["result"], "pass")
        code, report, probes, _ = self.run_gate()
        self.assertEqual(probes, [])

    def test_cargo_config_and_instrument_files_are_inputs(self):
        self.run_gate()
        commit(self.repo, "cargo config", **{".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=native"]\n'})
        code, report, probes, _ = self.run_gate()
        self.assert_reruns(probes, report, ("qwen3-tts", "kitten"), ".cargo/config.toml")
        for rel in ("instruments/instruments.yaml", "docs/instruments/instruments.schema.json"):
            commit(self.repo, rel, **{rel: "changed\n"})
            code, report, probes, _ = self.run_gate()
            self.assert_reruns(probes, report, ("kitten",), rel)
        # Files only the crates' own tests read are not inputs.
        commit(self.repo, "scripts", **{"scripts/elsewhere.py": "x\n"})
        code, report, probes, _ = self.run_gate()
        self.assertEqual(probes, [])

    def test_cpu_stage_comes_from_green_ci_or_runs_here(self):
        code, report, _, cargo = self.run_gate("--no-reuse", ci=(True, {"commit": self.base, "checks": {}}))
        self.assertEqual((code, report["cpu_source"]["mode"], cargo), (0, "ci", []))
        self.assertEqual(report["cpu"], dict.fromkeys(("fmt", "clippy", "build", "test"), "pass"))
        # Planted: CI has a failed required check, so the gate runs the stage itself.
        code, report, _, cargo = self.run_gate("--no-reuse", ci=(False, {"checks": {"x": "failure"}}))
        self.assertEqual((report["cpu_source"]["mode"], cargo), ("local", ["fmt", "clippy", "build", "test"]))
        code, report, _, cargo = self.run_gate("--no-reuse", "--cpu-stage", ci=(True, {}))
        self.assertEqual((report["cpu_source"]["reason"], len(cargo)), ("--cpu-stage", 4))

    def test_missing_python_packages_stop_the_gate_before_any_build(self):
        missing = {"interpreter": "py", "packages": {}, "problems": [
            "librosa is not installed for py (required by instruments/examples/python/loudness-ends/requirements.txt)"]}
        code, report, probes, cargo = self.run_gate(python=missing)
        self.assertEqual((code, report["result"], probes, cargo), (1, "incomplete", [], []))
        self.assertIn("librosa", report["error"])
        self.assertIn("loudness-ends/requirements.txt", report["error"])


if __name__ == "__main__":
    unittest.main()
