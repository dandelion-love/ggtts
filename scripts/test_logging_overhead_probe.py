"""Check optional renderer arguments without running a model."""
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class SpeakerPassthroughTest(unittest.TestCase):
    def test_optional_speaker_reaches_renderer(self):
        script = Path(__file__).resolve().parent / "logging_overhead_probe.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("render", "story", "model", "codec"):
                (root / name).write_text("placeholder")
            for speaker in (None, "hk_voice_343"):
                with self.subTest(speaker=speaker):
                    args = [str(script), "--render", str(root / "render"),
                            "--story", str(root / "story"), "--model", str(root / "model"),
                            "--codec", str(root / "codec"), "--engine", "qwen3-tts",
                            "--device", "cpu", "--work", str(root / ("with" if speaker else "without"))]
                    if speaker is not None:
                        args += ["--speaker", speaker]
                    calls = []
                    def fake_run(command, **kwargs):
                        calls.append(command)
                        return SimpleNamespace(returncode=7)
                    with patch.object(sys, "argv", args), patch.object(subprocess, "run", fake_run):
                        with self.assertRaisesRegex(RuntimeError, "warmup-0: render failed, exit 7"):
                            runpy.run_path(str(script), run_name="__main__")
                    self.assertEqual(len(calls), 1)
                    if speaker is None:
                        self.assertNotIn("--speaker", calls[0])
                    else:
                        i = calls[0].index("--speaker")
                        self.assertEqual(calls[0][i + 1], speaker)


if __name__ == "__main__":
    unittest.main()

