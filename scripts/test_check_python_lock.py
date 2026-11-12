#!/usr/bin/env python3
"""The lock drift check refuses a drifted binding lockfile, and passes the committed pair."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_python_lock import ROOT, drift, main, packages  # noqa: E402

ROOT_LOCK = """version = 4

[[package]]
name = "libm"
version = "0.2.16"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aa"

[[package]]
name = "windows-sys"
version = "0.52.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "windows-sys"
version = "0.59.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
dependencies = [
 "libm",
]

[[package]]
name = "ggtts-core"
version = "0.0.1"
"""

PY_LOCK = """version = 4

[[package]]
name = "ggtts-core"
version = "0.0.1"

[[package]]
name = "libm"
version = "0.2.16"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "pyo3"
version = "0.25.1"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "windows-sys"
version = "0.59.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
"""


class LockDrift(unittest.TestCase):
    def test_parses_every_package_and_version(self):
        self.assertEqual(packages(ROOT_LOCK),
                         {"libm": {"0.2.16"}, "windows-sys": {"0.52.0", "0.59.0"}, "ggtts-core": {"0.0.1"}})

    def test_agreeing_lockfiles_pass(self):
        # A subset of the main lockfile's versions, and crates only the binding needs, are fine.
        self.assertEqual(drift(ROOT_LOCK, PY_LOCK), [])

    def test_a_drifted_version_is_refused_naming_the_crate(self):
        planted = PY_LOCK.replace('name = "libm"\nversion = "0.2.16"', 'name = "libm"\nversion = "0.2.15"')
        self.assertEqual(drift(ROOT_LOCK, planted),
                         ["libm: the binding resolves 0.2.15; the main lockfile resolves 0.2.16"])
        planted = PY_LOCK.replace('version = "0.59.0"', 'version = "0.60.2"')
        self.assertEqual(len(drift(ROOT_LOCK, planted)), 1)

    def test_an_empty_lockfile_is_refused(self):
        self.assertTrue(drift(ROOT_LOCK, "version = 4\n"))
        self.assertTrue(drift("", PY_LOCK))

    def test_the_committed_lockfiles_agree(self):
        self.assertEqual(main([]), 0)
        planted = (ROOT / "crates/ggtts-py/Cargo.lock").read_text(encoding="utf-8")
        name = next(n for n in packages(planted) if n in packages((ROOT / "Cargo.lock").read_text(encoding="utf-8")))
        version = sorted(packages(planted)[name])[0]
        bad = planted.replace(f'name = "{name}"\nversion = "{version}"', f'name = "{name}"\nversion = "999.0.0"', 1)
        self.assertNotEqual(drift((ROOT / "Cargo.lock").read_text(encoding="utf-8"), bad), [])


if __name__ == "__main__":
    unittest.main()
