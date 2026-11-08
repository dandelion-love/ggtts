import math
import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from ggtts_instrument.testing import Workdir, write_wav, run_instrument, metrics  # noqa: E402

SCRIPT = os.path.join(HERE, "instrument.py")
RATE = 24000


def tone(hz, seconds=1):
    return [0.2 * math.sin(2 * math.pi * hz * i / RATE) for i in range(int(RATE * seconds))]


def values(result):
    return {key[0]: value for key, value in metrics(result).items()}


class PitchTest(unittest.TestCase):
    def test_noop_and_planted_octave(self):
        with Workdir() as d:
            quiet = [0.0] * (RATE // 2)
            flat = quiet + tone(200, 2) + quiet
            changed = quiet + tone(200) + tone(400) + quiet
            for name, samples in (("base.wav", flat), ("copy.wav", flat), ("changed.wav", changed)):
                write_wav(os.path.join(d, name), samples, RATE)
            base = run_instrument(SCRIPT, d, audio=os.path.join(d, "base.wav"), extra=["--render", "r"])
            copy = run_instrument(SCRIPT, d, audio=os.path.join(d, "copy.wav"), extra=["--render", "r"])
            planted = run_instrument(SCRIPT, d, audio=os.path.join(d, "changed.wav"), extra=["--render", "r"])
            self.assertEqual(base["status"], "ok")
            self.assertEqual(metrics(base), metrics(copy))
            self.assertAlmostEqual(values(base)["pitch_median_hz"], 200, delta=1)
            self.assertLess(values(base)["pitch_range_st"], 0.1)
            self.assertGreater(values(planted)["pitch_range_st"], 11.0)

    def test_silence_has_no_pitch_and_missing_node_is_skipped(self):
        with Workdir() as d:
            path = os.path.join(d, "silence.wav")
            write_wav(path, [0.0] * RATE, RATE)
            result = run_instrument(SCRIPT, d, audio=path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(values(result)["pitch_voiced_frames"], 0)
            self.assertIsNone(values(result)["pitch_range_st"])
            short = os.path.join(d, "short.wav")
            quiet = [0.0] * (RATE // 2)
            write_wav(short, quiet + tone(200, 0.08) + quiet, RATE)
            few = values(run_instrument(SCRIPT, d, audio=short))
            self.assertGreater(few["pitch_voiced_frames"], 0)
            self.assertLess(few["pitch_voiced_frames"], 10)
            self.assertIsNone(few["pitch_range_st"])
            with patch.dict(os.environ, {"PATH": "", "GGTTS_NODE": ""}):
                missing = run_instrument(SCRIPT, d, audio=path)
            self.assertEqual(missing["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
