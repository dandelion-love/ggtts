#!/usr/bin/env python3
"""Fail closed unless the APK carries the fresh Flutter C ABI for exactly both supported ABIs."""
import argparse
import pathlib
import struct
import subprocess
import tempfile
import zipfile

ABIS = {"arm64-v8a": 183, "x86_64": 62}
REQUIRED = frozenset({
    "ggtts_abi_version", "ggtts_load_opts", "ggtts_render_opts", "ggtts_pcm_free",
    "ggtts_free", "ggtts_last_error", "ggtts_options_schema", "ggtts_capabilities",
    "ggtts_string_free",
})


def check_elf(data, machine):
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
        raise ValueError("expected little-endian ELF64")
    elf_type, elf_machine = struct.unpack_from("<HH", data, 16)
    if elf_type != 3 or elf_machine != machine:
        raise ValueError("incorrect ELF type or architecture")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    if phentsize < 56 or not phnum or phoff < 64 or phoff + phentsize * phnum > len(data):
        raise ValueError("invalid ELF program header table")
    loads = 0
    for index in range(phnum):
        offset = phoff + index * phentsize
        if struct.unpack_from("<I", data, offset)[0] != 1:
            continue
        loads += 1
        file_offset, virtual_address = struct.unpack_from("<QQ", data, offset + 8)
        alignment = struct.unpack_from("<Q", data, offset + 48)[0]
        if (alignment < 16384 or alignment & (alignment - 1) or
                file_offset % alignment != virtual_address % alignment):
            raise ValueError("ELF LOAD segment is not 16 KB aligned")
    if not loads:
        raise ValueError("ELF has no LOAD segments")


def verify(apk, native_dir, nm, strip):
    with zipfile.ZipFile(apk) as archive, tempfile.TemporaryDirectory(prefix="ggtts-apk-check-") as temp:
        entries = [i for i in archive.infolist() if i.filename.startswith("lib/") and
                   i.filename.endswith("/libggtts_ffi.so")]
        expected = {f"lib/{abi}/libggtts_ffi.so" for abi in ABIS}
        if len(entries) != len(expected) or {i.filename for i in entries} != expected:
            raise ValueError("APK must contain exactly one Flutter library for each supported ABI")
        for entry in entries:
            abi = entry.filename.split("/")[1]
            data = archive.read(entry)
            check_elf(data, ABIS[abi])
            packaged = pathlib.Path(temp) / (abi + ".so")
            packaged.write_bytes(data)
            symbols = subprocess.run([str(nm), "-D", "--defined-only", str(packaged)],
                                     check=True, capture_output=True, text=True).stdout
            exported = {line.split()[-1] for line in symbols.splitlines() if line.split()}
            missing = REQUIRED - exported
            if missing:
                raise ValueError(f"{abi} missing defined C exports: {', '.join(sorted(missing))}")
            fresh = pathlib.Path(native_dir) / abi / "libggtts_ffi.so"
            check_elf(fresh.read_bytes(), ABIS[abi])
            stripped = pathlib.Path(temp) / (abi + "-fresh.so")
            subprocess.run([str(strip), "--strip-unneeded", "-o", str(stripped), str(fresh)],
                           check=True, capture_output=True)
            if stripped.read_bytes() != data:
                raise ValueError(f"{abi} APK library does not match fresh native build")
            print(f"{abi}: ELF64, 16 KB LOAD alignment, required exports and fresh stripped bytes verified")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("apk", type=pathlib.Path)
    p.add_argument("native_dir", type=pathlib.Path)
    p.add_argument("nm", type=pathlib.Path)
    p.add_argument("strip", type=pathlib.Path)
    args = p.parse_args()
    try:
        verify(args.apk, args.native_dir, args.nm, args.strip)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Flutter APK native gate failed: {exc}\n")


if __name__ == "__main__":
    main()
