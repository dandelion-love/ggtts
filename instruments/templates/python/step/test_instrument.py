import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))

from ggtts_instrument.audio import read_audio  # noqa: E402
from ggtts_instrument.testing import (  # noqa: E402
    Workdir, fixture, metrics, read_lines, run_instrument, shift_clock, write_lines, write_wav)

SCRIPT = os.path.join(HERE, "instrument.py")


class Selftest(unittest.TestCase):
    def test_report_identifies_or_declares_unknown_stop_semantics(self):
        with Workdir() as d:
            r = fixture(d)
            lines = read_lines(r.trace)
            lines[0].setdefault("flags", {})["p_stop_semantics"] = "qwen-sampler-weights/1"
            tagged = os.path.join(d, "tagged.jsonl")
            write_lines(tagged, lines)
            result = run_instrument(SCRIPT, d, events=tagged)
            self.assertEqual(result["provenance"]["config"]["p_stop_semantics"], "qwen-sampler-weights/1")
            lines[0]["flags"].pop("p_stop_semantics")
            unknown = os.path.join(d, "old.jsonl")
            write_lines(unknown, lines)
            result = run_instrument(SCRIPT, d, events=unknown)
            self.assertEqual(result["provenance"]["config"]["p_stop_semantics"], "unknown")

    def test_noop_reads_as_no_change(self):
        with Workdir() as d:
            r = fixture(d)
            a = run_instrument(SCRIPT, d, events=r.trace)
            moved = os.path.join(d, "moved.jsonl")
            write_lines(moved, shift_clock(read_lines(r.trace)))
            b = run_instrument(SCRIPT, d, events=moved)
            self.assertEqual(a["status"], "ok")
            self.assertEqual(metrics(a), metrics(b))

    def test_planted_early_stop_pressure_is_seen(self):
        with Workdir() as d:
            r = fixture(d)
            lines = read_lines(r.trace)
            a = metrics(run_instrument(SCRIPT, d, events=r.trace))
            step = next(l for l in lines if l["ev"] == "step" and l["frame"] == 1)
            step["p_stop"] = 0.95
            bad = os.path.join(d, "planted.jsonl")
            write_lines(bad, lines)
            b = metrics(run_instrument(SCRIPT, d, events=bad))
            key = lambda n: next(k for k in a if k[0] == n)
            self.assertEqual(b[key("early_stop_frames")], a[key("early_stop_frames")] + 1)
            self.assertEqual(b[key("p_stop_early_max")], 0.95)

    def test_a_trace_without_end_is_truncated(self):
        with Workdir() as d:
            r = fixture(d)
            cut = os.path.join(d, "cut.jsonl")
            write_lines(cut, read_lines(r.trace)[:-1])
            self.assertEqual(run_instrument(SCRIPT, d, events=cut)["status"], "truncated")

    def test_missing_step_data_is_not_a_success(self):
        with Workdir() as d:
            r = fixture(d)
            original = read_lines(r.trace)
            cases = (
                [line for line in original if line["ev"] != "step"],
                [line if line["ev"] != "step" else
                 {k: v for k, v in line.items() if k != "p_stop"} for line in original],
                [line for line in original if line["ev"] != "step" or line["chunk"] != 0],
            )
            for i, lines in enumerate(cases):
                path = os.path.join(d, "missing-%d.jsonl" % i)
                write_lines(path, lines)
                result = run_instrument(SCRIPT, d, events=path)
                self.assertEqual(result["status"], "skipped")
                self.assertIn("p_stop", result["error"])

    def test_one_frame_draw_has_no_early_sample_but_is_valid(self):
        with Workdir() as d:
            r = fixture(d)
            lines = read_lines(r.trace)
            kept = set()
            one = []
            for line in lines:
                if line["ev"] == "step":
                    if line["chunk"] in kept:
                        continue
                    kept.add(line["chunk"])
                one.append(line)
            path = os.path.join(d, "one-frame.jsonl")
            write_lines(path, one)
            result = run_instrument(SCRIPT, d, events=path)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(all(value is None for value in metrics(result).values()))


if __name__ == "__main__":
    unittest.main()
