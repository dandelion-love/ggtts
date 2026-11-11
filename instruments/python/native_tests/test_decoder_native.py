"""Optional real libsndfile checks; run with soundfile and numpy installed."""
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from ggtts_instrument import check, read_audio, run
from ggtts_instrument.testing import Workdir


class NativeDecoder(unittest.TestCase):
    def test_real_native_flac_decoder_identifies_its_loaded_library(self):
        try:
            import soundfile
        except ImportError:
            self.skipTest("soundfile is optional")
        from ggtts_instrument.audio import _loaded_sndfile_paths, sha256_of
        with patch("ggtts_instrument.audio._bound_module_path", side_effect=["first", "second", "first"]):
            self.assertEqual(_loaded_sndfile_paths("input.flac"), [], "mixed bound modules must be unknown")
        if sys.platform == "win32":
            with patch("ggtts_instrument.audio._bound_module_path", return_value="module") as module:
                _loaded_sndfile_paths("input.flac")
                want = int(soundfile._ffi.cast("uintptr_t", soundfile._snd.sf_wchar_open))
                self.assertEqual(module.call_args_list[0].args[0], want)
            with patch("ggtts_instrument.audio._bound_module_path", return_value="module") as module:
                _loaded_sndfile_paths(b"input.flac")
                want = int(soundfile._ffi.cast("uintptr_t", soundfile._snd.sf_open))
                self.assertEqual(module.call_args_list[0].args[0], want)
        if not (sys.platform.startswith("linux") or sys.platform == "win32"):
            self.skipTest("native module enumeration unavailable on this platform")
        with Workdir() as w:
            audio, out = os.path.join(w, "input.flac"), os.path.join(w, "env.json")
            soundfile.write(audio, [0.0, 0.5, -0.5], 24000, format="FLAC", subtype="PCM_16")
            self.assertEqual(run(lambda c, e: read_audio(c.audio), "x", "1", [],
                                 argv=["--audio", audio, "--envelope", out]), 0)
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            paths = _loaded_sndfile_paths(audio)
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

