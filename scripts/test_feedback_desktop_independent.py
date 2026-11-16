"""Independent desktop host tests using an actual synthetic #475 render."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import feedback_desktop as host
from scripts.feedback_page import input_digest

ROOT = Path(__file__).resolve().parents[1]

class DesktopIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cargo = Path.home() / ".cargo/bin/cargo"
        if not cargo.is_file():
            cargo = Path("cargo")
        subprocess.run([str(cargo), "build", "--quiet", "-p", "ggtts-cli", "--bin", "ggtts-render"],
                       cwd=ROOT, check=True)
        cls.binary = ROOT / "target/debug" / ("ggtts-render.exe" if os.name == "nt" else "ggtts-render")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ggtts-feedback-desktop-test-")
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.audio = self.dir / "read.flac"
        self.page = self.dir / "page.html"
        self.companion = Path(str(self.page) + ".provenance.json")
        run = subprocess.run([str(self.binary), "--engine", "fake", "--text", "A\u4e2d\U00020000\u6587.",
                              "--out", str(self.audio), "--feedback-page", str(self.page)],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.source = host.verify(self.page, self.audio)
        self.result = {
            "format": "ggtts-feedback-result/1", "id": self.source["id"],
            "input_sha256": input_digest(self.source), "note": "read note",
            "items": [{"id": i["id"], "marks": [0], "rating": "accept", "note": "item note"}
                      for i in self.source["items"]],
        }
        self.result_path = self.dir / "result.json"
        self.result_path.write_text(json.dumps(self.result), encoding="utf-8")

    def test_real_generated_page_opens_and_result_round_trips_next_to_audio(self):
        with patch.object(host.webbrowser, "open", return_value=True) as browser:
            host.open_page(self.page, self.audio)
            browser.assert_called_once_with(self.page.resolve().as_uri())
        destination = host.save(self.page, self.audio, self.result_path)
        self.assertEqual(destination, self.dir / "read.flac.feedback.json")
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), self.result)
        self.assertEqual(host.verify(self.page, self.audio), self.source)

    def test_untrusted_page_never_opens(self):
        self.page.write_text(self.page.read_text(encoding="utf-8").replace(
            "<title>Listening feedback</title>", "<title>altered</title>"), encoding="utf-8")
        with patch.object(host.webbrowser, "open") as browser:
            with self.assertRaises(ValueError):
                host.open_page(self.page, self.audio)
            browser.assert_not_called()

    def test_tampered_audio_or_companion_rejected(self):
        with self.audio.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(ValueError):
            host.verify(self.page, self.audio)
        self.audio.write_bytes(self.audio.read_bytes()[:-6])
        meta = json.loads(self.companion.read_text(encoding="utf-8"))
        meta["input_sha256"] = "0" * 64
        self.companion.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError):
            host.verify(self.page, self.audio)

    def test_planted_chunk_evidence_defects_are_rejected(self):
        original = json.loads(self.companion.read_text(encoding="utf-8"))
        for field, value in [
            ("text_sha256", "0" * 64), ("audio_sha256", "0" * 64),
            ("start", 1), ("end", original["output_samples"] + 1),
            ("draw", -1), ("seed", -1), ("id", "wrong"),
        ]:
            bad = copy.deepcopy(original)
            bad["chunks"][0][field] = value
            self.companion.write_text(json.dumps(bad), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                host.verify(self.page, self.audio)
        self.companion.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(host.verify(self.page, self.audio), self.source)

    def test_malformed_page_and_result_fail_visibly(self):
        self.result_path.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            host.save(self.page, self.audio, self.result_path)
        self.assertFalse((self.dir / "read.flac.feedback.json").exists())
        self.page.write_bytes(b"not an html page")
        with self.assertRaises(ValueError):
            host.verify(self.page, self.audio)

    def test_invalid_and_wrong_input_result_never_overwrite_existing(self):
        destination = self.dir / "read.flac.feedback.json"
        destination.write_text("existing", encoding="utf-8")
        for bad in ("index", "digest"):
            result = copy.deepcopy(self.result)
            if bad == "index": result["items"][0]["marks"] = [999]
            else: result["input_sha256"] = "0" * 64
            self.result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                host.save(self.page, self.audio, self.result_path)
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing")
        self.result_path.write_text(json.dumps(self.result), encoding="utf-8")
        with self.assertRaises(FileExistsError):
            host.save(self.page, self.audio, self.result_path)
        self.assertEqual(destination.read_text(encoding="utf-8"), "existing")

    def test_planted_publish_failure_leaves_no_result(self):
        destination = self.dir / "read.flac.feedback.json"
        with patch("scripts.feedback_page.os.link", side_effect=OSError("planted write failure")):
            with self.assertRaises(OSError):
                host.save(self.page, self.audio, self.result_path)
        self.assertFalse(destination.exists())
        self.assertFalse(any(p.name.startswith(".feedback-") for p in self.dir.iterdir()))
        self.assertTrue(self.page.exists())
        self.assertTrue(self.audio.exists())

if __name__ == "__main__":
    unittest.main()
