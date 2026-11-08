import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from ggtts_instrument.testing import Workdir, run_instrument, write_lines  # noqa: E402

SCRIPT = os.path.join(HERE, "instrument.py")


def trace():
    return [
        {"ev": "trace", "t_ms": 0, "format": "ggtts-trace/2", "engine": "qwen3-tts",
         "flags": {"p_stop_semantics": "qwen-sampler-weights/1", "codec_eos_token": 2150}},
        {"ev": "chunk_begin", "t_ms": 1, "chunk": 0, "draw": 0, "seed": 42},
        {"ev": "step", "t_ms": 2, "chunk": 0, "frame": 0, "token": 17, "p_stop": 0.01},
        {"ev": "step", "t_ms": 3, "chunk": 0, "frame": 1, "token": 2150, "p_stop": 0.9},
        {"ev": "chunk_end", "t_ms": 4, "chunk": 0, "draw": 0, "frames": 1, "stop": "eos"},
        {"ev": "end", "t_ms": 5, "frames": 1, "seconds": 0.1},
    ]


def invoke(root, name, lines):
    directory = os.path.join(root, name)
    os.mkdir(directory)
    path = os.path.join(directory, "trace.jsonl")
    write_lines(path, lines)
    result = run_instrument(SCRIPT, directory, events=path)
    output = os.path.join(directory, "stop-probability.jsonl")
    if os.path.isfile(output):
        with open(output, encoding="utf8") as source:
            return result, [json.loads(line) for line in source]
    return result, []


class StopTest(unittest.TestCase):
    def test_multiple_draws_and_cap_preserve_step_attribution(self):
        with Workdir() as d:
            lines = trace()
            lines[-1:-1] = [
                {"ev": "chunk_begin", "t_ms": 5, "chunk": 0, "draw": 1, "seed": 43},
                {"ev": "step", "t_ms": 6, "chunk": 0, "frame": 0, "token": 18, "p_stop": 0.2},
                {"ev": "chunk_end", "t_ms": 7, "chunk": 0, "draw": 1, "frames": 1, "stop": "cap"},
            ]
            lines[-1]["t_ms"] = 8
            result, rows = invoke(d, "draws", lines)
            self.assertEqual(result["status"], "ok")
            self.assertEqual([(r["draw"], r["step"], r["terminal_eos"]) for r in rows],
                             [(0, 0, False), (0, 1, True), (1, 0, False)])

    def test_noop_and_planted_terminal_probability(self):
        with Workdir() as d:
            lines = trace()
            base, rows = invoke(d, "base", lines)
            _, same = invoke(d, "same", lines)
            self.assertEqual(base["status"], "ok")
            self.assertEqual(rows, same)
            self.assertEqual([r["terminal_eos"] for r in rows], [False, True])
            self.assertEqual(rows[-1]["p_stop"], 0.9)
            self.assertEqual(len(base["artifacts"]), 1)
            changed = trace()
            changed[3]["p_stop"] = 0.5
            _, planted = invoke(d, "plant", changed)
            self.assertEqual(planted[-1]["p_stop"], 0.5)
            self.assertEqual(planted[0], rows[0])

    def test_missing_terminal_or_semantics_is_not_complete(self):
        with Workdir() as d:
            no_terminal = [event for event in trace() if not (event["ev"] == "step" and event["token"] == 2150)]
            missing = trace()
            del missing[0]["flags"]["codec_eos_token"]
            unmarked = trace()
            unmarked[0]["flags"] = {}
            kitten = trace()
            kitten[0]["engine"] = "kitten-tts-2"
            seed = trace()
            del seed[1]["seed"]
            frames = trace()
            del frames[4]["frames"]
            truncated = trace()[:-1]
            for name, lines, status in (("lost", no_terminal, "skipped"), ("missing", missing, "skipped"),
                                        ("old", unmarked, "skipped"), ("kitten", kitten, "skipped"),
                                        ("seed", seed, "skipped"), ("frames", frames, "skipped"),
                                        ("truncated", truncated, "truncated")):
                result, rows = invoke(d, name, lines)
                self.assertEqual(result["status"], status, name)
                self.assertEqual(rows, [], name)

    def test_noncontiguous_and_nonfinite_values_are_refused(self):
        with Workdir() as d:
            gap = trace()
            gap[3]["frame"] = 2
            invalid = trace()
            invalid[2]["p_stop"] = 1.2
            nonfinite = trace()
            nonfinite[2]["p_stop"] = float("nan")
            for name, lines in (("gap", gap), ("invalid", invalid), ("nan", nonfinite)):
                result, rows = invoke(d, name, lines)
                self.assertEqual(result["status"], "skipped")
                self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
