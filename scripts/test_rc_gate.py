import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("rc_gate", Path(__file__).with_name("rc_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTest(unittest.TestCase):
    def test_result_never_promotes_missing_or_failed_gpu(self):
        cpu = dict.fromkeys(("fmt", "clippy", "build", "test"), "pass")
        proof = {"cpu": {"status": "pass"}, "cuda": {"status": "pass"}}
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "pass"}}, proof), "pass")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "incomplete"}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "fail"}, "kitten": {"status": "pass"}}, {}), "fail")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}}, proof), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "pass"}}, {}), "incomplete")
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "pass"}},
                                       {"cpu": {"status": "pass"}, "cuda": {"status": "fail"}}), "fail")
        cpu["fmt"] = "fail"
        self.assertEqual(gate.evaluate(cpu, {"qwen": {"status": "pass"}, "kitten": {"status": "pass"}}, proof), "fail")

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
            def run_case(missing=False, failing=False):
                run_env = dict(env)
                if missing:
                    del run_env["VOICE"]
                def command(cmd, cwd, log, env=None):
                    return int(failing and cmd[0] == "bash" and "kitten" in cmd[1])
                args = ["rc_gate.py", "--work", str(root / "scratch"), "--out", str(root / "report.json")]
                with patch.dict(os.environ, run_env, clear=True), patch.object(gate.sys, "argv", args), \
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


if __name__ == "__main__":
    unittest.main()
