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


def first_sound(samples):
    return next(i for i, x in enumerate(samples) if abs(x) > 0.01)


class Selftest(unittest.TestCase):
    def test_noop_reads_as_no_change(self):
        with Workdir() as d:
            r = fixture(d)
            a = run_instrument(SCRIPT, d, audio=r.audio, extra=["--render", "r"])
            x = read_audio(r.audio)
            copy = os.path.join(d, "same.wav")
            write_wav(copy, x.samples, x.rate)
            self.assertEqual(a["status"], "ok")
            self.assertEqual(metrics(a), metrics(run_instrument(SCRIPT, d, audio=copy, extra=["--render", "r"])))

    def test_a_planted_internal_pause_is_counted(self):
        with Workdir() as d:
            r = fixture(d)
            x = read_audio(r.audio)
            a = metrics(run_instrument(SCRIPT, d, audio=r.audio, extra=["--render", "r"]))
            s = list(x.samples)
            # Into the middle of the first tone: speech on both sides of the planted gap.
            at = first_sound(s) + int(0.05 * x.rate)
            s[at:at] = [0.0] * int(2.0 * x.rate)
            bad = os.path.join(d, "gap.wav")
            write_wav(bad, s, x.rate)
            b = metrics(run_instrument(SCRIPT, d, audio=bad, extra=["--render", "r"]))
            self.assertEqual(total(b, "internal_pauses"), total(a, "internal_pauses") + 1)
            self.assertGreaterEqual(total(b, "longest_pause_s"), 2.0)

    def test_edge_silence_is_not_an_internal_pause(self):
        with Workdir() as d:
            r = fixture(d)
            x = read_audio(r.audio)
            a = metrics(run_instrument(SCRIPT, d, audio=r.audio, extra=["--render", "r"]))
            s = [0.0] * int(2.0 * x.rate) + list(x.samples) + [0.0] * int(2.0 * x.rate)
            padded = os.path.join(d, "padded.wav")
            write_wav(padded, s, x.rate)
            b = metrics(run_instrument(SCRIPT, d, audio=padded, extra=["--render", "r"]))
            self.assertEqual(total(b, "internal_pauses"), total(a, "internal_pauses"))


if __name__ == "__main__":
    unittest.main()
