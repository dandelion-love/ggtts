import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qwen_event_instrument_probe import catalogue, require_identical


class ProbeTest(unittest.TestCase):
    def test_noop_and_independent_plants(self):
        token = [(0, 0, 17), (0, 1, 42)]
        require_identical("same", "same", token, token, [123], [123])
        for args in (("same", "changed", token, token, [123], [123]),
                     ("same", "same", token, [(0, 0, 18), (0, 1, 42)], [123], [123]),
                     ("same", "same", token, token, [123], [124])):
            with self.assertRaises(AssertionError):
                require_identical(*args)

    def test_catalogue_and_wrong_model_refusal(self):
        self.assertIn("render-timing", catalogue())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.write_bytes(b"not the pinned model")
            work = root / "work"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("qwen_event_instrument_probe.py")),
                                     "--render", str(model), "--talker", str(model), "--codec", str(model),
                                     "--device", "cpu", "--work", str(work)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)
            self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
