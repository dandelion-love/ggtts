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
            a = run_instrument(SCRIPT, d, events=r.trace)
            moved = os.path.join(d, "moved.jsonl")
            write_lines(moved, shift_clock(read_lines(r.trace)))
            self.assertEqual(a["status"], "ok")
            self.assertGreater(total(metrics(a), "seconds"), 0)
            self.assertEqual(metrics(a), metrics(run_instrument(SCRIPT, d, events=moved)))

    def test_planted_truncation_is_reported_not_read_as_a_short_render(self):
        with Workdir() as d:
            r = fixture(d)
            lines = read_lines(r.trace)
            self.assertEqual(lines[-1]["ev"], "end")
            cut = os.path.join(d, "cut.jsonl")
            write_lines(cut, lines[:-1])
            b = run_instrument(SCRIPT, d, events=cut)
            self.assertEqual(b["status"], "truncated")
            self.assertIsNone(total(metrics(b), "seconds"))
            self.assertIsNone(total(metrics(b), "frames"))


if __name__ == "__main__":
    unittest.main()
