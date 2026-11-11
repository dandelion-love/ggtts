import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from ggtts_instrument.testing import Workdir, synthetic, read_lines, write_lines, run_instrument, metrics  # noqa: E402

SCRIPT = os.path.join(HERE, "instrument.py")


class TimingTest(unittest.TestCase):
    def trace(self, directory):
        render = synthetic(directory)
        lines = read_lines(render.trace)
        lines[0]["format"] = "ggtts-trace/2"
        out = []
        for row in lines:
            if row["ev"] == "chunk_end":
                for stage, ms in (("generate", 4), ("decode", 3), ("judge", 1)):
                    out.append({"ev": "timing", "stage": stage, "chunk": row["chunk"],
                                "draw": row["draw"], "ms": ms, "t_ms": row["t_ms"]})
            if row["ev"] == "end":
                out.extend([{"ev": "timing", "stage": "first_audio", "ms": 12, "t_ms": row["t_ms"]},
                            {"ev": "timing", "stage": "render", "ms": 20, "t_ms": row["t_ms"]}])
            out.append(row)
        path = os.path.join(directory, "timing.jsonl")
        write_lines(path, out)
        return path, out

    def test_complete_and_clock_shift_noop(self):
        with Workdir() as d:
            path, rows = self.trace(d)
            first = run_instrument(SCRIPT, d, events=path)
            self.assertEqual(first["status"], "ok")
            vals = {key[0]: value for key, value in metrics(first).items()}
            self.assertEqual(vals["ttfa_ms"], 12)
            self.assertEqual(vals["generate_ms_total"], 8)
            for row in rows:
                row["t_ms"] += 1000
            write_lines(path, rows)
            self.assertEqual(metrics(first), metrics(run_instrument(SCRIPT, d, events=path)))

    def test_missing_first_write_and_stage_are_incomplete(self):
        with Workdir() as d:
            path, rows = self.trace(d)
            for stage in ("first_audio", "decode"):
                planted = [row for row in rows if row.get("stage") != stage]
                write_lines(path, planted)
                self.assertEqual(run_instrument(SCRIPT, d, events=path)["status"], "skipped")

    def test_planted_slow_render_changes_rtf(self):
        with Workdir() as d:
            path, rows = self.trace(d)
            before = run_instrument(SCRIPT, d, events=path)
            for row in rows:
                if row.get("stage") == "render":
                    row["ms"] *= 2
            write_lines(path, rows)
            after = run_instrument(SCRIPT, d, events=path)
            before_rtf = next(v for k, v in metrics(before).items() if k[0] == "render_rtf")
            after_rtf = next(v for k, v in metrics(after).items() if k[0] == "render_rtf")
            self.assertAlmostEqual(after_rtf, 2 * before_rtf)

    def test_explicit_stage_only_has_no_latency_or_rtf(self):
        with Workdir() as d:
            path, rows = self.trace(d)
            write_lines(path, [row for row in rows if row.get("stage") not in ("first_audio", "render")])
            report = run_instrument(SCRIPT, d, events=path, extra=["--stage-only", "true"])
            self.assertEqual(report["status"], "ok")
            vals = {key[0]: value for key, value in metrics(report).items()}
            self.assertIsNone(vals["ttfa_ms"])
            self.assertIsNone(vals["render_rtf"])
            self.assertEqual(vals["generate_ms_total"], 8)


if __name__ == "__main__":
    unittest.main()
