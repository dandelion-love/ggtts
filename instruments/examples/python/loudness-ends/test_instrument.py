import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from ggtts_instrument.testing import Workdir, write_wav, run_instrument, metrics  # noqa: E402

SCRIPT = os.path.join(HERE, "instrument.py")
RATE, HOP = 24000, 240


def pyin_available():
    try:
        import librosa
        return callable(librosa.pyin)
    except Exception:
        return False


def values(result):
    return {key[0]: value for key, value in metrics(result).items()}


class EndsTest(unittest.TestCase):
    def test_speech_core_discards_a_short_separated_edge(self):
        spec = importlib.util.spec_from_file_location("core_selection", os.path.join(HERE, "core_selection.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        levels = [-100.0] * 20 + [-10.0] * 8 + [-100.0] * 72 + [-12.0] * 100 + [-100.0] * 20
        self.assertEqual(module.speech_core(levels), (1.0, 2.0))

    @unittest.skipUnless(pyin_available(), "working librosa pYIN stack unavailable")
    def test_derived_voicing_noop_and_dynamic_compression(self):
        with Workdir() as d:
            samples = []
            for frame in range(120):
                amplitude = (0.05, 0.1, 0.2, 0.4)[frame // 30]
                samples.extend(amplitude * math.sin(2 * math.pi * 200 * (frame * HOP + i) / RATE)
                               for i in range(HOP))
            original = os.path.join(d, "original.wav")
            copy = os.path.join(d, "copy.wav")
            compressed = os.path.join(d, "compressed.wav")
            write_wav(original, samples, RATE)
            write_wav(copy, samples, RATE)
            write_wav(compressed, [math.copysign(0.5 * math.sqrt(abs(x)), x) for x in samples], RATE)
            def measure(path):
                return run_instrument(SCRIPT, d, audio=path, extra=["--render", "r"])
            base, same, plant = measure(original), measure(copy), measure(compressed)
            self.assertEqual(base["status"], "ok")
            self.assertEqual(metrics(base), metrics(same))
            self.assertEqual(plant["status"], "ok")
            self.assertLess(values(plant)["loud_p90_db"], values(base)["loud_p90_db"])
            self.assertGreater(values(plant)["soft_p10_db"], values(base)["soft_p10_db"])

    def test_refuses_missing_audio(self):
        with Workdir() as d:
            self.assertEqual(run_instrument(SCRIPT, d)["status"], "skipped")

    def test_missing_dependencies_are_incomplete(self):
        with Workdir() as d:
            audio, envelope = os.path.join(d, "audio.wav"), os.path.join(d, "envelope.json")
            write_wav(audio, [0.1] * RATE, RATE)
            subprocess.run([sys.executable, "-S", SCRIPT, "--run", "r", "--audio", audio,
                            "--envelope", envelope], capture_output=True, check=False)
            with open(envelope, encoding="utf8") as source:
                self.assertEqual(json.load(source)["status"], "skipped")

    @unittest.skipUnless(pyin_available(), "working librosa pYIN stack unavailable")
    def test_silence_has_no_measurement(self):
        with Workdir() as d:
            audio = os.path.join(d, "silence.wav")
            write_wav(audio, [0.0] * RATE, RATE)
            self.assertEqual(run_instrument(SCRIPT, d, audio=audio)["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
