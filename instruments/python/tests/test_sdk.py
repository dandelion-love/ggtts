"""The SDK's own checks: events in, audio in, an envelope out, and a crash reported rather than lost.

Run with `python -m unittest discover -s instruments/python/tests`. With `GGTTS_FIXTURE_DIR` set (the cargo harness
sets it) the FLAC reader is also checked sample for sample against the fake engine's render.
"""

import io
import json
import os
import struct
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from ggtts_instrument import (  # noqa: E402
    AudioError, Envelope, EnvelopeError, TraceError, check, iter_events, read_audio, read_trace, run)
from ggtts_instrument.testing import Workdir, synthetic, write_lines, write_wav  # noqa: E402

SHA = "a" * 64


def envelope():
    e = Envelope("x", "1", "run-1", hooks=["step"], config={"k": 1}, audio="raw")
    e.metric("m", 1.5, "s", "lower", interval=(1.0, 2.0)).metric("d", None, "1", "neutral", render="r", chunk=0)
    e.doc["artifacts"].append({"path": "a.csv", "type": "text/csv", "sha256": SHA})
    e.doc["provenance"]["inputs"].append({"path": "t.jsonl", "sha256": SHA})
    e.doc["provenance"]["seeds"].append({"render": "r", "chunk": 0, "seed": 7})
    return e.doc


class EnvelopeCheck(unittest.TestCase):
    def test_a_built_envelope_is_valid(self):
        self.assertEqual(check(envelope()), [])

    def test_previous_format_and_unknown_state(self):
        doc = envelope()
        doc["format"] = "ggtts-report/1"
        self.assertEqual(check(doc), [])
        doc["provenance"]["audio"] = "unknown"
        self.assertNotEqual(check(doc), [])
        doc["format"] = "ggtts-report/2"
        self.assertEqual(check(doc), [])

    def test_planted_defects_are_each_refused(self):
        def mut(f):
            d = json.loads(json.dumps(envelope()))
            f(d)
            return d

        plants = {
            "no status": lambda d: d.pop("status"),
            "a status not in the list": lambda d: d.update(status="done"),
            "the next format": lambda d: d.update(format="ggtts-report/3"),
            "an upper-case name": lambda d: d.update(name="X"),
            "a metric without a unit": lambda d: d["metrics"][0].pop("unit"),
            "a direction not in the list": lambda d: d["metrics"][0].update(direction="up"),
            "a value as a string": lambda d: d["metrics"][0].update(value="1.5"),
            "a value left out instead of null": lambda d: d["metrics"][1].pop("value"),
            "an interval low above high": lambda d: d["metrics"][0].update(interval=[2.0, 1.0]),
            "a chunk without its render": lambda d: d["metrics"][1].pop("render"),
            "an artifact without a hash": lambda d: d["artifacts"][0].pop("sha256"),
            "no identity": lambda d: d.pop("identity"),
            "no run": lambda d: d["identity"].pop("run"),
            "no provenance": lambda d: d.pop("provenance"),
            "no config": lambda d: d["provenance"].pop("config"),
            "a model hash left out instead of null": lambda d: d["provenance"].pop("model_sha256"),
            "no audio state": lambda d: d["provenance"].pop("audio"),
            "a negative seed": lambda d: d["provenance"]["seeds"][0].update(seed=-1),
            "an input with a short hash": lambda d: d["provenance"]["inputs"][0].update(sha256="abc"),
            "no log field": lambda d: d.pop("log"),
            "an undeclared field": lambda d: d.update(rtf=0.3),
        }
        for name, f in plants.items():
            self.assertNotEqual(check(mut(f)), [], name)

    def test_every_reason_is_named(self):
        d = envelope()
        d["status"] = "done"
        d["metrics"][0]["direction"] = "up"
        d["provenance"].pop("audio")
        self.assertEqual(len(check(d)), 3, check(d))

    def test_write_refuses_an_invalid_envelope_and_writes_nothing(self):
        with Workdir() as w:
            e = Envelope("x", "1", "run-1")
            e.metric("m", 1.0, "", "lower")
            out = os.path.join(w, "env.json")
            with self.assertRaises(EnvelopeError):
                e.write(out)
            self.assertFalse(os.path.exists(out))


class Events(unittest.TestCase):
    def test_a_trace_reads_and_knows_it_is_complete(self):
        with Workdir() as w:
            t = read_trace(synthetic(w).trace)
            self.assertTrue(t.complete)
            self.assertEqual(t.header["format"], "ggtts-trace/1")
            self.assertEqual(len(t.draws()), 2)

    def test_what_is_not_a_trace_is_refused(self):
        bad = {
            "no header": ['{"ev":"step","t_ms":0}'],
            "the next format": ['{"ev":"trace","format":"ggtts-trace/3","t_ms":0}'],
            "no t_ms": ['{"ev":"trace","format":"ggtts-trace/1"}'],
            "not JSON": ['{"ev":"trace","format":"ggtts-trace/1","t_ms":0}', "{not json"],
            "a second header": ['{"ev":"trace","format":"ggtts-trace/1","t_ms":0}',
                                '{"ev":"trace","format":"ggtts-trace/1","t_ms":1}'],
            "empty": [],
        }
        for name, lines in bad.items():
            with self.assertRaises(TraceError, msg=name):
                read_trace(io.StringIO("\n".join(lines) + "\n"))

    def test_a_trace_without_end_is_incomplete_not_short(self):
        with Workdir() as w:
            r = synthetic(w)
            with open(r.trace, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertFalse(read_trace(io.StringIO("\n".join(lines[:-1]) + "\n")).complete)

    def test_events_are_read_live_from_a_pipe_as_they_arrive(self):
        with Workdir() as w:
            with open(synthetic(w).trace, encoding="utf-8") as f:
                lines = f.read().splitlines()
            rfd, wfd = os.pipe()
            seen = []
            first_seen = threading.Event()

            def writer():
                with os.fdopen(wfd, "w", encoding="utf-8") as out:
                    out.write(lines[0] + "\n")
                    out.flush()
                    # The reader must see the header before the rest is written.
                    self.assertTrue(first_seen.wait(10), "the header was not delivered live")
                    for l in lines[1:]:
                        out.write(l + "\n")

            t = threading.Thread(target=writer)
            t.start()
            with os.fdopen(rfd, encoding="utf-8") as src:
                for ev in iter_events(src):
                    seen.append(ev["ev"])
                    first_seen.set()
            t.join()
            self.assertEqual(seen[0], "trace")
            self.assertEqual(seen[-1], "end")
            self.assertEqual(len(seen), len(lines))


class Audio(unittest.TestCase):
    def test_wav_round_trips_exactly(self):
        with Workdir() as w:
            s = [0.0, 0.5, -0.5, 0.25, -1.0, 32767 / 32768.0]
            p = os.path.join(w, "a.wav")
            write_wav(p, s, 24000)
            a = read_audio(p)
            self.assertEqual((a.samples, a.rate), (s, 24000))

    def test_a_streamed_wav_of_unknown_length_reads_to_the_end(self):
        with Workdir() as w:
            p = os.path.join(w, "s.wav")
            pcm = struct.pack("<4h", 1, 2, 3, 4)
            with open(p, "wb") as f:
                f.write(b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE")
                f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16))
                f.write(b"data" + struct.pack("<I", 0xFFFFFFFF) + pcm)
            self.assertEqual(len(read_audio(p).samples), 4)

    def test_what_is_not_audio_is_refused(self):
        with Workdir() as w:
            p = os.path.join(w, "x.bin")
            with open(p, "wb") as f:
                f.write(b"not audio at all")
            with self.assertRaises(AudioError):
                read_audio(p)

    @unittest.skipUnless(os.environ.get("GGTTS_FIXTURE_DIR"), "needs the fake engine's render (cargo harness)")
    def test_ggtts_flac_decodes_sample_for_sample(self):
        d = os.environ["GGTTS_FIXTURE_DIR"]
        with open(os.path.join(d, "audio.s16le"), "rb") as f:
            raw = f.read()
        want = [x / 32768.0 for x in struct.unpack("<%dh" % (len(raw) // 2), raw)]
        a = read_audio(os.path.join(d, "audio.flac"), prefer_soundfile=False)
        self.assertEqual(len(a.samples), len(want))
        # The first differing sample, not a list diff: difflib over 10^5 samples would take minutes to fail.
        bad = next((i for i, (x, y) in enumerate(zip(a.samples, want)) if x != y), None)
        self.assertIsNone(bad, "sample %s: decoded %r, written %r" % (
            bad, a.samples[bad] if bad is not None else None, want[bad] if bad is not None else None))
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            self.assertEqual(a.rate, json.load(f)["rate"])


class Cli(unittest.TestCase):
    def test_decoder_identity_is_recorded_by_the_decoding_child(self):
        with Workdir() as w:
            audio, out = os.path.join(w, "input.wav"), os.path.join(w, "env.json")
            write_wav(audio, [0.0, 0.5, -0.5], 24000)
            def measure(ctx, env):
                read_audio(ctx.audio)
                env.input(ctx.audio)
            self.assertEqual(run(measure, "x", "1", [], argv=["--audio", audio, "--envelope", out]), 0)
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            records = env["provenance"]["config"]["ggtts_audio_decoders"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["kind"], "ggtts-python-wav")
            self.assertEqual(len(records[0]["sha256"]), 64)
            self.assertIn({"path": "decoder:ggtts-python-wav", "sha256": records[0]["sha256"]},
                          env["provenance"]["inputs"])
            self.assertNotIn(w, json.dumps(records))
            self.assertEqual(check(env), [])

    def test_unknown_and_ambiguous_native_identity_never_invents_a_hash(self):
        from ggtts_instrument.audio import _decoder
        for paths in ([], ["not-a-real-file"], ["one", "two"]):
            with patch("ggtts_instrument.audio._loaded_sndfile_paths", return_value=paths):
                self.assertEqual(_decoder("libsndfile"), {"kind": "libsndfile", "sha256": None})

    def test_real_native_flac_decoder_identifies_its_loaded_library(self):
        try:
            import soundfile
        except ImportError:
            self.skipTest("soundfile is optional")
        from ggtts_instrument.audio import _loaded_sndfile_paths, sha256_of
        if not (sys.platform.startswith("linux") or sys.platform == "win32"):
            self.skipTest("native module enumeration unavailable on this platform")
        with Workdir() as w:
            audio, out = os.path.join(w, "input.flac"), os.path.join(w, "env.json")
            soundfile.write(audio, [0.0, 0.5, -0.5], 24000, format="FLAC", subtype="PCM_16")
            self.assertEqual(run(lambda c, e: read_audio(c.audio), "x", "1", [],
                                 argv=["--audio", audio, "--envelope", out]), 0)
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            paths = _loaded_sndfile_paths()
            self.assertEqual(len(paths), 1, "actual loaded decoder must be unambiguous")
            digest = sha256_of(paths[0])
            self.assertEqual(env["provenance"]["config"]["ggtts_audio_decoders"],
                             [{"kind": "libsndfile", "sha256": digest}])
            self.assertIn({"path": "decoder:libsndfile", "sha256": digest}, env["provenance"]["inputs"])
            self.assertNotIn(paths[0], json.dumps(env))
            unknown = os.path.join(w, "unknown.json")
            with patch("ggtts_instrument.audio._loaded_sndfile_paths", return_value=[]):
                self.assertEqual(run(lambda c, e: read_audio(c.audio), "x", "1", [],
                                     argv=["--audio", audio, "--envelope", unknown]), 0)
            with open(unknown, encoding="utf-8") as f:
                missing = json.load(f)
            self.assertEqual(missing["provenance"]["config"]["ggtts_audio_decoders"],
                             [{"kind": "libsndfile", "sha256": None}])
            self.assertFalse(any(x["path"].startswith("decoder:") for x in missing["provenance"]["inputs"]))

    def test_real_instrument_child_records_native_decoder(self):
        try:
            import soundfile
        except ImportError:
            self.skipTest("soundfile is optional")
        if not (sys.platform.startswith("linux") or sys.platform == "win32"):
            self.skipTest("native module resolution unavailable on this platform")
        with Workdir() as w:
            audio, out = os.path.join(w, "input.flac"), os.path.join(w, "env.json")
            soundfile.write(audio, [0.0, 0.5, -0.5] * 1000, 24000, format="FLAC", subtype="PCM_16")
            script = os.path.abspath(os.path.join(HERE, "..", "..", "examples", "python", "internal-pauses", "instrument.py"))
            result = subprocess.run([sys.executable, script, "--audio", audio, "--envelope", out],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            decoder = env["provenance"]["config"]["ggtts_audio_decoders"][0]
            self.assertEqual(decoder["kind"], "libsndfile")
            self.assertIsNotNone(decoder["sha256"])
            self.assertIn({"path": "decoder:libsndfile", "sha256": decoder["sha256"]},
                          env["provenance"]["inputs"])
            self.assertEqual(check(env), [])

    def test_a_crash_is_written_as_a_failed_envelope(self):
        with Workdir() as w:
            out = os.path.join(w, "env.json")

            def boom(ctx, env):
                env.metric("m", 1.0, "1", "lower")
                raise RuntimeError("planted crash")

            stderr, sys.stderr = sys.stderr, io.StringIO()
            try:
                code = run(boom, "x", "1", ["step"], argv=["--envelope", out, "--run", "r1", "--k", "v"])
            finally:
                sys.stderr = stderr
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            self.assertEqual(code, 1)
            self.assertEqual(env["status"], "failed")
            self.assertIn("planted crash", env["error"])
            self.assertEqual(env["metrics"], [], "a crash leaves no half-measured value")
            self.assertEqual(env["provenance"]["config"]["k"], "v")
            self.assertEqual(check(env), [])

    def test_the_no_op_measure_is_ok_and_records_its_defaults(self):
        with Workdir() as w:
            out = os.path.join(w, "env.json")
            self.assertEqual(run(lambda c, e: None, "x", "1", [], argv=["--envelope", out]), 0)
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            self.assertEqual(env["status"], "ok")
            self.assertEqual(env["provenance"]["config"]["run_default"], "local")

    def test_a_flag_without_a_value_is_refused(self):
        with self.assertRaises(SystemExit):
            run(lambda c, e: None, "x", "1", [], argv=["--envelope"])


if __name__ == "__main__":
    unittest.main()

