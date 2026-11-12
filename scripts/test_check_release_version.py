"""Tests for check_release_version.py: each refusal fires, and a matching directory passes.

Run: python3 scripts/test_check_release_version.py (CI runs it in the selection job, on every run). The packed
binaries are stand-in shell scripts that print a --version, so this needs a POSIX shell.
"""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from check_release_version import main

TAG = "v0.1.0-rc.999"
SHA = "0123456789abcdef0123456789abcdef01234567"
NAMES = ("ggtts-render", "ggtts-measure", "ggtts-kitten-convert", "ggtts-kitten")


@unittest.skipIf(os.name == "nt", "the stand-in binaries are shell scripts")
class ReleaseVersionTest(unittest.TestCase):
    def packed(self, d, commit=SHA, tag=TAG, versions=None, listed=NAMES, info=None):
        """A packed directory whose binaries print `<name> <tag>`, `ggtts <commit> (...)` and a ggml line."""
        versions = versions or {}
        for name in NAMES:
            t, c = versions.get(name, (tag, commit))
            exe = Path(d, name)
            exe.write_text(f"#!/bin/sh\necho '{name} {t}'\necho 'ggtts {c} (https://example.invalid/ggtts)'\n"
                           "echo 'ggml af97976c7810cdabb1863172f31c432dab767de7 (v0.16.0+7, x)'\n")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        meta = {"ggtts": TAG, "ggtts_commit": SHA, "binaries": list(listed)} if info is None else info
        Path(d, "BUILDINFO.json").write_text(json.dumps(meta))
        return d

    def check(self, d):
        try:
            main(d, TAG)
        except ValueError as error:
            return str(error)
        return None

    def test_a_matching_directory_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self.check(self.packed(d)))

    def test_every_listed_binary_is_held_to_the_tag_and_the_commit(self):
        with tempfile.TemporaryDirectory() as d:
            self.packed(d, versions={"ggtts-kitten": (TAG, "f" * 40)})
            self.assertIn("ggtts-kitten reports ggtts commit 'ffff", self.check(d))
        with tempfile.TemporaryDirectory() as d:
            self.packed(d, versions={"ggtts-kitten-convert": ("v0.1.0-rc.998", SHA)})
            self.assertIn("ggtts-kitten-convert reports 'ggtts-kitten-convert v0.1.0-rc.998'", self.check(d))

    def test_a_build_without_a_commit_is_refused(self):
        # build.rs prints `unknown` when it finds no commit; that can never match BUILDINFO's SHA.
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("reports ggtts commit 'unknown'", self.check(self.packed(d, commit="unknown")))
        with tempfile.TemporaryDirectory() as d:
            info = {"ggtts": TAG, "ggtts_commit": "unknown", "binaries": list(NAMES)}
            self.assertIn("ggtts_commit is 'unknown', expected a full commit SHA", self.check(self.packed(d, info=info)))

    def test_a_listed_binary_that_is_missing_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            self.packed(d)
            Path(d, "ggtts-kitten").unlink()
            self.assertEqual(self.check(d), f"ggtts-kitten is listed in BUILDINFO binaries or required, and is not in {d}")
        with tempfile.TemporaryDirectory() as d:
            # ggtts-render and ggtts-measure are required even when BUILDINFO leaves them out.
            self.packed(d, listed=("ggtts-kitten",))
            Path(d, "ggtts-measure").unlink()
            self.assertIn("ggtts-measure is listed in BUILDINFO binaries or required", self.check(d))

    def test_wrong_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("BUILDINFO ggtts is 'v0'", self.check(self.packed(d, info={"ggtts": "v0"})))
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("ggtts_commit is None", self.check(self.packed(d, info={"ggtts": TAG})))


if __name__ == "__main__":
    unittest.main()
