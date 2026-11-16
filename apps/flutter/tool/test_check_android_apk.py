import importlib.util
import pathlib
import struct
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

MODULE = pathlib.Path(__file__).with_name("check_android_apk.py")
spec = importlib.util.spec_from_file_location("apk_gate", MODULE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def elf(machine, tail=b"fresh"):
    data = bytearray(120)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 3, machine)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<I", data, 64, 1)
    struct.pack_into("<Q", data, 64 + 48, 16384)
    return bytes(data) + tail


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.native = self.root / "native"
        self.contents = {}
        for abi, machine in gate.ABIS.items():
            data = elf(machine)
            folder = self.native / abi
            folder.mkdir(parents=True)
            (folder / "libggtts_ffi.so").write_bytes(data)
            self.contents[f"lib/{abi}/libggtts_ffi.so"] = data
        self.exports = set(gate.REQUIRED)

    def run_gate(self, entries=None):
        apk = self.root / "test.apk"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(apk, "w") as archive:
                for name, data in entries if entries is not None else self.contents.items():
                    archive.writestr(name, data)
        def run(cmd, **kwargs):
            if cmd[0] == "nm":
                return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(
                    "000001 T " + name for name in self.exports))
            pathlib.Path(cmd[3]).write_bytes(pathlib.Path(cmd[4]).read_bytes())
            return subprocess.CompletedProcess(cmd, 0)
        with patch.object(gate.subprocess, "run", side_effect=run):
            gate.verify(apk, self.native, "nm", "strip")

    def test_positive(self):
        self.run_gate()

    def test_missing_and_extra_and_duplicate_abi_refused(self):
        entries = list(self.contents.items())
        for bad in [entries[:1], entries + [("lib/armeabi-v7a/libggtts_ffi.so", elf(40))],
                    entries + [entries[0]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_gate(bad)

    def test_stale_and_wrong_arch_and_bad_elf_refused(self):
        key = "lib/arm64-v8a/libggtts_ffi.so"
        for data in (elf(183, b"stale"), elf(62), b"not ELF"):
            bad = dict(self.contents)
            bad[key] = data
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.run_gate(list(bad.items()))

    def test_4k_alignment_and_missing_load_are_refused(self):
        key = "lib/arm64-v8a/libggtts_ffi.so"
        for replacement in (4096, 0):
            bad = dict(self.contents)
            data = bytearray(bad[key])
            if replacement:
                struct.pack_into("<Q", data, 64 + 48, replacement)
            else:
                struct.pack_into("<I", data, 64, 2)
            bad[key] = bytes(data)
            with self.assertRaises(ValueError):
                self.run_gate(list(bad.items()))

    def test_missing_export_refused(self):
        self.exports.remove("ggtts_render_opts")
        with self.assertRaisesRegex(ValueError, "ggtts_render_opts"):
            self.run_gate()

    def test_fresh_input_missing_refused(self):
        (self.native / "x86_64" / "libggtts_ffi.so").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_gate()


if __name__ == "__main__":
    unittest.main()
