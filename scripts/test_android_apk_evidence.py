"""Host regressions for the APK's packaged JNI evidence export gate.

Run with ANDROID_NDK_HOME pointing at an installed NDK:
    python3 -m unittest scripts.test_android_apk_evidence -v
No Android device or model is used.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NDK = os.environ.get("ANDROID_NDK_HOME")
TOOL = Path(NDK) / "toolchains/llvm/prebuilt/linux-x86_64/bin/clang" if NDK else None
ENTRY = "lib/arm64-v8a/libggtts_ffi.so"
SYMBOL = "Java_ggtts_Native_renderStreamEvidence"
NOTICES = [
    "META-INF/ggtts/LICENSE", "META-INF/ggtts/THIRD_PARTY.md", "META-INF/ggtts/BUILDINFO.json",
    "META-INF/ggtts/licenses/Apache-2.0.txt", "META-INF/ggtts/licenses/ggml-LICENSE.txt",
    "META-INF/ggtts/licenses/android-crates.txt", "META-INF/ggtts/licenses/libc++-LICENSE.txt",
]

@unittest.skipUnless(TOOL and TOOL.is_file(), "ANDROID_NDK_HOME with Linux NDK clang required")
class PackagedEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def library(self, source, target="aarch64-linux-android24"):
        c = self.work / "fixture.c"
        lib = self.work / "fixture.so"
        c.write_text(source)
        subprocess.run([str(TOOL), "--target=" + target, "-shared", "-nostdlib", "-o", str(lib), str(c)],
                       check=True, capture_output=True)
        return lib.read_bytes()

    def apk(self, binary, duplicate=False):
        apk = self.work / "fixture.apk"
        with zipfile.ZipFile(apk, "w") as z:
            for name in NOTICES:
                z.writestr(name, "fixture")
            z.writestr(ENTRY, binary)
            if duplicate:
                z.writestr(ENTRY, binary)
        return apk

    def check(self, apk):
        return subprocess.run(["bash", str(ROOT / "scripts/android_build.sh"), "check-apk", str(apk)],
                              env=os.environ.copy(), text=True, capture_output=True)

    def test_defined_export_in_packaged_arm64_passes(self):
        result = self.check(self.apk(self.library("void " + SYMBOL + "(void) {}")))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("packages the AArch64 JNI evidence export", result.stderr)

    def test_missing_undefined_or_lookalike_export_refused(self):
        cases = [
            "void Java_ggtts_Native_loadWithModels(void) {}",
            "extern void " + SYMBOL + "(void); void call_it(void) { " + SYMBOL + "(); }",
            "void " + SYMBOL + "_suffix(void) {}",
        ]
        for source in cases:
            with self.subTest(source=source):
                result = self.check(self.apk(self.library(source)))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("lacks defined JNI export " + SYMBOL, result.stderr)

    def test_elf32_with_aarch64_machine_is_refused_for_class(self):
        # Plant an ELF32 ARM shared object but change only e_machine to AArch64.
        # This isolates the class check from the architecture and DYN checks.
        lib = bytearray(self.library("void " + SYMBOL + "(void) {}", "armv7a-linux-androideabi24"))
        self.assertEqual(lib[4], 1)  # EI_CLASS = ELFCLASS32
        lib[18:20] = (183).to_bytes(2, "little")  # EM_AARCH64
        probe = self.work / "wrong-class.so"
        probe.write_bytes(lib)
        readelf = TOOL.with_name("llvm-readelf")
        header = subprocess.run([str(readelf), "-h", str(probe)], check=True, text=True, capture_output=True).stdout
        self.assertIn("ELF32", header)
        self.assertIn("AArch64", header)
        self.assertIn("DYN", header)
        result = self.check(self.apk(bytes(lib)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an ELF64 AArch64 shared ELF", result.stderr)

    def test_bad_elf_wrong_arch_and_duplicate_entry_refused(self):
        cases = [
            self.apk(b"not an ELF").read_bytes(),
            self.apk(self.library("void " + SYMBOL + "(void) {}", "x86_64-linux-android24")).read_bytes(),
            self.apk(self.library("void " + SYMBOL + "(void) {}"), duplicate=True).read_bytes(),
        ]
        for content in cases:
            self.work.joinpath("case.apk").write_bytes(content)
            result = self.check(self.work / "case.apk")
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == "__main__":
    unittest.main()
