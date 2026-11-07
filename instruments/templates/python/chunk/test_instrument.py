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
            self.assertEqual(metrics(a), metrics(run_instrument(SCRIPT, d, events=moved)))

    def test_planted_cap_stop_and_runaway_are_seen(self):
        with Workdir() as d:
            r = fixture(d)
            lines = read_lines(r.trace)
            a = metrics(run_instrument(SCRIPT, d, events=r.trace))
            end = next(l for l in lines if l["ev"] == "chunk_end" and l["stop"] != "cap" and not l["verdict"])
            end["stop"], end["verdict"] = "cap", "runaway"
            bad = os.path.join(d, "planted.jsonl")
            write_lines(bad, lines)
            b = metrics(run_instrument(SCRIPT, d, events=bad))
            self.assertEqual(total(b, "cap_stops"), total(a, "cap_stops") + 1)
            self.assertEqual(total(b, "flagged_draws"), total(a, "flagged_draws") + 1)
            self.assertEqual(total(b, "draws"), total(a, "draws"))


if __name__ == "__main__":
    unittest.main()
