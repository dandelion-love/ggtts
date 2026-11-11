#!/usr/bin/env python3
"""Tests for check_pe_imports.py, on PE images built here: it reads what a file imports, passes Windows and the VC++
runtime, passes a DLL shipped beside the file, and refuses everything else, naming it. Runs on any host."""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pe_imports import NotPE, imports, main  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "check_pe_imports.py"


def pe(dlls, delayed=()):
    """A minimal PE32+ image with one section holding an import table for `dlls` and a delay-import table."""
    va, raw = 0x1000, 0x200
    body = bytearray()
    names = {}
    tables = 20 * (len(dlls) + 1) + 32 * (len(delayed) + 1)
    for name in list(dlls) + list(delayed):
        names[name] = va + tables + len(body)
        body += name.encode("ascii") + b"\0"
    section = bytearray()
    for name in dlls:
        section += struct.pack("<IIIII", 0, 0, 0, names[name], 0)
    section += bytes(20)
    delay_rva = va + len(section)
    for name in delayed:
        section += struct.pack("<IIIIIIII", 1, names[name], 0, 0, 0, 0, 0, 0)
    section += bytes(32)
    section += body
    image = bytearray(raw)
    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x40)
    image[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", image, 0x44, 0x8664, 1, 0, 0, 0, 240, 0x22)
    optional = 0x58
    struct.pack_into("<H", image, optional, 0x20B)
    directories = optional + 112
    struct.pack_into("<II", image, directories + 8, va, 20 * (len(dlls) + 1))
    if delayed:
        struct.pack_into("<II", image, directories + 8 * 13, delay_rva, 32 * (len(delayed) + 1))
    table = optional + 240
    image[table:table + 8] = b".idata\0\0"
    struct.pack_into("<IIII", image, table + 8, len(section), va, len(section), raw)
    return bytes(image) + bytes(section)


class Imports(unittest.TestCase):
    def test_reads_import_and_delay_import_tables(self):
        self.assertEqual(imports(pe(["KERNEL32.dll", "VCRUNTIME140.dll"], ["bcrypt.dll"])),
                         ["KERNEL32.dll", "VCRUNTIME140.dll", "bcrypt.dll"])

    def test_not_a_pe_is_refused(self):
        with self.assertRaises(NotPE):
            imports(b"\x7fELF" + bytes(100))


class Command(unittest.TestCase):
    def run_check(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_packed_directory(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "lib").mkdir()
            Path(d, "ggtts-render.exe").write_bytes(pe(["KERNEL32.dll", "api-ms-win-crt-heap-l1-1-0.dll",
                                                        "VCOMP140.DLL", "MSVCP140.dll"]))
            Path(d, "lib", "ggtts_ffi.dll").write_bytes(pe(["kernel32.dll", "ntdll.dll", "VCRUNTIME140.dll"]))
            Path(d, "README.md").write_text("not a binary\n")
            result = self.run_check("--packed", d)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("2 checked", result.stderr)
            # Planted: a DLL that is neither Windows nor the VC++ runtime, and not shipped, is refused, naming it.
            Path(d, "lib", "ggtts_ffi.dll").write_bytes(pe(["kernel32.dll", "ggml.dll"]))
            result = self.run_check("--packed", d)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("ggtts_ffi.dll: needs ggml.dll", result.stderr)
            # The no-op: the same DLL shipped beside it passes.
            Path(d, "lib", "ggml.dll").write_bytes(pe(["kernel32.dll"]))
            self.assertEqual(self.run_check("--packed", d).returncode, 0)
            # --deny takes a name out of the allowed lists, so a real requirement can be planted.
            result = self.run_check("--deny", "vcomp140.dll", "--packed", d)
            self.assertEqual(result.returncode, 1)
            self.assertIn("needs VCOMP140.DLL", result.stderr)
            Path(d, "vcomp140.dll").write_bytes(pe(["kernel32.dll"]))
            self.assertEqual(self.run_check("--deny", "vcomp140.dll", "--packed", d).returncode, 0)

    def test_nothing_read_is_never_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("holds no .exe or .dll", self.run_check("--packed", d).stderr)
            Path(d, "x.dll").write_bytes(b"\x7fELF" + bytes(100))
            result = self.run_check("--packed", d)
            self.assertEqual(result.returncode, 1)
            self.assertIn("not a PE image", result.stderr)
            Path(d, "x.dll").write_bytes(pe(["VCRUNTIME140.dll"]))
            result = self.run_check("--packed", d)
            self.assertEqual(result.returncode, 1)
            self.assertIn("imports neither kernel32.dll nor ntdll.dll", result.stderr)
        with self.assertRaises(SystemExit):
            main([])

    def test_a_real_image_when_given(self):
        # GGTTS_PE_SAMPLE names a real .exe or .dll (CI's Windows package-smoke passes the packed library).
        sample = os.environ.get("GGTTS_PE_SAMPLE")
        if not sample:
            self.skipTest("GGTTS_PE_SAMPLE is not set")
        names = [n.lower() for n in imports(Path(sample).read_bytes())]
        self.assertIn("kernel32.dll", names)


if __name__ == "__main__":
    unittest.main()
