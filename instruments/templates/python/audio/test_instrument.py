import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))

from ggtts_instrument.audio import read_audio  # noqa: E402
from ggtts_instrument.testing import (  # noqa: E402
    Workdir, fixture, metrics, read_lines, run_instrument, shift_clock, write_lines, write_wav)

SCRIPT = os.path.join(HERE, "instrument.py")


def total(m, name):
    return next(v for k, v in m.items() if k[0] == name)


class Selftest(unittest.TestCase):
    def test_noop_reads_as_no_change(self):
        with Workdir() as d:
            r = fixture(d)
            a = run_instrument(SCRIPT, d, audio=r.audio, extra=["--render", "r"])
            x = read_audio(r.audio)
            copy = os.path.join(d, "same.wav")
            write_wav(copy, x.samples, x.rate)
            b = run_instrument(SCRIPT, d, audio=copy, extra=["--render", "r"])
            self.assertEqual(a["status"], "ok")
            for k, v in metrics(a).items():
                self.assertAlmostEqual(v, metrics(b)[k], places=6, msg=k)

    def test_planted_clipping_is_seen(self):
        with Workdir() as d:
            r = fixture(d)
            x = read_audio(r.audio)
            a = metrics(run_instrument(SCRIPT, d, audio=r.audio, extra=["--render", "r"]))
            s = list(x.samples)
            mid = len(s) // 2
            newly = sum(1 for v in s[mid:mid + 100] if abs(v) < 0.999)
            self.assertGreater(newly, 0, "the planted region was already clipped")
            s[mid:mid + 100] = [1.0] * 100
            bad = os.path.join(d, "clipped.wav")
            write_wav(bad, s, x.rate)
            b = metrics(run_instrument(SCRIPT, d, audio=bad, extra=["--render", "r"]))
            self.assertEqual(total(b, "clipped_samples"), total(a, "clipped_samples") + newly)
            self.assertAlmostEqual(total(b, "peak_dbfs"), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
