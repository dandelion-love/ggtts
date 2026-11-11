"""Tests for check_glibc_floor.py: each refusal fires, and a binary within the floor passes.

Run: python3 scripts/test_check_glibc_floor.py (CI runs it in the selection job, on every run).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_glibc_floor import FLOOR, evaluate, main, needed_versions, symbols_by_index

SCRIPT = Path(__file__).resolve().parent / "check_glibc_floor.py"

# readelf --version-info --wide and --dyn-syms --wide, cut down from a binary built on Ubuntu 24.04 (the v0.1.0-rc.8
# Linux CPU ggtts-render, which does not start on Ubuntu 22.04).
VERSION_INFO_2404 = """\
Version needs section '.gnu.version_r' contains 3 entries:
 Addr: 0x0000000000003298  Offset: 0x003298  Link: 7 (.dynstr)
  000000: Version: 1  File: libstdc++.so.6  Cnt: 2
  0x0010:   Name: CXXABI_1.3.9  Flags: none  Version: 9
  0x0020:   Name: GLIBCXX_3.4.26  Flags: none  Version: 8
  0x0030: Version: 1  File: libgcc_s.so.1  Cnt: 1
  0x0040:   Name: GCC_4.2.0  Flags: none  Version: 7
  0x0050: Version: 1  File: libc.so.6  Cnt: 4
  0x0060:   Name: GLIBC_2.39  Flags: none  Version: 6
  0x0070:   Name: GLIBC_2.38  Flags: none  Version: 5
  0x0080:   Name: GLIBC_2.34  Flags: none  Version: 4
  0x0090:   Name: GLIBC_2.2.5  Flags: none  Version: 3
"""
DYN_SYMS_2404 = """\
Symbol table '.dynsym' contains 6 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __isoc23_strtol@GLIBC_2.38 (5)
     2: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND pidfd_spawnp@GLIBC_2.39 (6)
     3: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND pidfd_getpid@GLIBC_2.39 (6)
     4: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND dlopen@GLIBC_2.34 (4)
     5: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND _Znam@GLIBCXX_3.4 (2)
"""


class FloorTest(unittest.TestCase):
    def test_parses_needs_and_their_symbols(self):
        needs = needed_versions(VERSION_INFO_2404)
        self.assertIn(("libc.so.6", "GLIBC_2.38", "5"), needs)
        self.assertIn(("libstdc++.so.6", "GLIBCXX_3.4.26", "8"), needs)
        self.assertEqual(len(needs), 7)
        symbols = symbols_by_index(DYN_SYMS_2404)
        self.assertEqual(sorted(symbols["6"]), ["pidfd_getpid", "pidfd_spawnp"])
        self.assertEqual(symbols["5"], ["__isoc23_strtol"])

    def test_a_2404_build_is_refused_naming_the_version_and_symbols(self):
        problems, highest = evaluate("ggtts-render", VERSION_INFO_2404, DYN_SYMS_2404, FLOOR)
        self.assertEqual(highest["GLIBC"], ("2.39", "libc.so.6"))
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("GLIBC_2.39 from libc.so.6 exceeds the floor GLIBC_2.35 (needed by: pidfd_getpid, pidfd_spawnp)",
                      problems[0])
        self.assertIn("GLIBC_2.38 from libc.so.6 exceeds the floor GLIBC_2.35 (needed by: __isoc23_strtol)",
                      problems[1])

    def test_the_same_binary_within_a_raised_floor_passes(self):
        # The no-op: nothing above the floor reads as nothing wrong.
        floor = dict(FLOOR, GLIBC="2.39")
        self.assertEqual(evaluate("ggtts-render", VERSION_INFO_2404, DYN_SYMS_2404, floor)[0], [])

    def test_libstdcxx_is_held_to_its_floor_too(self):
        floor = dict(FLOOR, GLIBC="2.39", GLIBCXX="3.4.25")
        problems = evaluate("ggtts-render", VERSION_INFO_2404, DYN_SYMS_2404, floor)[0]
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("GLIBCXX_3.4.26 from libstdc++.so.6 exceeds the floor GLIBCXX_3.4.25", problems[0])

    def test_an_unknown_namespace_is_refused(self):
        text = VERSION_INFO_2404.replace("Name: GCC_4.2.0", "Name: GLIBC_PRIVATE")
        problems = evaluate("x", text, DYN_SYMS_2404, dict(FLOOR, GLIBC="2.39"))[0]
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("needs GLIBC_PRIVATE from libgcc_s.so.1, a version namespace with no floor here", problems[0])
        text = VERSION_INFO_2404.replace("File: libgcc_s.so.1", "File: libcudart.so.12").replace(
            "Name: GCC_4.2.0", "Name: libcudart.so.12")
        self.assertIn("a version namespace with no floor here",
                      evaluate("x", text, DYN_SYMS_2404, dict(FLOOR, GLIBC="2.39"))[0][0])

    def test_nothing_read_is_never_a_pass(self):
        # A binary needing no GLIBC version (unread, static, or readelf printed nothing) is refused, not passed.
        for text in ("", "Version symbols section '.gnu.version' contains 2 entries:\n"):
            problems = evaluate("x", text, "", FLOOR)[0]
            self.assertEqual(len(problems), 1)
            self.assertIn("needs no GLIBC version at all", problems[0])

    def test_command_line_on_real_files(self):
        if not shutil.which("readelf"):
            self.skipTest("readelf is not installed")
        elf = os.path.realpath(shutil.which("sh") or "/bin/sh")

        def run(*args):
            return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

        # Planted: an impossible floor turns a real dynamically linked binary red, naming the version.
        result = run("--floor", "GLIBC=2.0", elf)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("exceeds the floor GLIBC_2.0", result.stderr)
        self.assertIn("not the Ubuntu 22.04 floor", result.stderr)
        # The no-op: a floor above anything glibc defines passes the same binary.
        result = run("--floor", "GLIBC=99", elf)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as d:
            text = Path(d, "README.md")
            text.write_text("not a binary\n")
            result = run(str(text))
            self.assertEqual(result.returncode, 1)
            self.assertIn("not an ELF file", result.stderr)
            result = run(str(Path(d, "absent")))
            self.assertEqual(result.returncode, 1)
            self.assertIn("No such file", result.stderr)
        with self.assertRaises(SystemExit):
            main(["--floor", "GLIBC", elf])
        with self.assertRaises(SystemExit):
            main(["--floor", "NOPE=1", elf])

    def test_packed_checks_exactly_what_pack_listed(self):
        if not shutil.which("readelf"):
            self.skipTest("readelf is not installed")
        elf = os.path.realpath(shutil.which("sh") or "/bin/sh")

        def run(*args):
            return subprocess.run([sys.executable, str(SCRIPT), "--floor", "GLIBC=99", *args],
                                  capture_output=True, text=True)

        with tempfile.TemporaryDirectory() as d:
            for name in ("ggtts-render", "ggtts-kitten"):
                shutil.copy(elf, Path(d, name))
            Path(d, "README.md").write_text("not a binary\n")
            info = Path(d, "BUILDINFO.json")
            info.write_text('{"binaries": ["ggtts-render", "ggtts-kitten"]}')
            result = run("--packed", d)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("2 checked", result.stderr)
            # Planted: a packed ELF the list leaves out is refused, naming it; it can never go unchecked.
            info.write_text('{"binaries": ["ggtts-render"]}')
            result = run("--packed", d)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("ggtts-kitten is an ELF file that BUILDINFO.json's binaries leaves out", result.stderr)
            # A listed binary that is absent, and a missing or empty list, are refused too.
            info.write_text('{"binaries": ["ggtts-render", "ggtts-kitten", "ggtts-measure"]}')
            result = run("--packed", d)
            self.assertEqual(result.returncode, 1)
            self.assertIn("names ggtts-measure, which is not in the directory", result.stderr)
            for text in ('{"binaries": []}', "{}"):
                info.write_text(text)
                self.assertEqual(run("--packed", d).returncode, 1)
            info.unlink()
            self.assertIn("cannot read the packed binaries list", run("--packed", d).stderr)
        with self.assertRaises(SystemExit):
            main(["--packed", "x", elf])
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
