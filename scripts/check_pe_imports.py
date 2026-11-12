#!/usr/bin/env python3
"""Refuse a Windows release directory whose executables or DLLs need a DLL that neither Windows nor the archive has.

    python3 scripts/check_pe_imports.py [--deny NAME ...] --packed DIR
    python3 scripts/check_pe_imports.py [--deny NAME ...] PE...

For each .exe and .dll (with --packed, every one anywhere under DIR) it reads the DLLs the file imports, from its
import and delay-import tables, and passes each that is

  * a Windows system library (kernel32.dll, ntdll.dll, the api-ms-win-* API sets, and the rest of SYSTEM below),
  * the Microsoft Visual C++ runtime (VCRUNTIME, MSVCP, VCOMP, CONCRT 140: the "Visual C++ Redistributable", which
    the README names as a requirement of the Windows archive and which the archive does not ship), or
  * a file in the same directory as the importing file, which Windows loads first.

Anything else fails, naming the file and the DLL: the archive would ship a binary that does not start on a machine
without that DLL. `--deny NAME` removes NAME from the two lists above, so CI can plant a requirement that the archive
does not meet and require red, then copy a file of that name beside the binary and require green.

It cannot silently pass: a file that is not a PE image, a PE with no import of kernel32.dll or ntdll.dll (it was not
read), and a --packed DIR with no .exe or .dll all fail. The parser is plain Python with no dependency, so it runs on
any host; CI runs it on the Windows archive, and tests it with built images (scripts/test_check_pe_imports.py).
"""
import argparse
import os
import struct
import sys

SYSTEM = {
    "advapi32.dll", "bcrypt.dll", "bcryptprimitives.dll", "crypt32.dll", "dbghelp.dll", "gdi32.dll", "kernel32.dll",
    "kernelbase.dll", "ncrypt.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "powrprof.dll", "psapi.dll",
    "secur32.dll", "setupapi.dll", "shell32.dll", "shlwapi.dll", "synchronization.dll", "user32.dll", "userenv.dll",
    "version.dll", "winmm.dll", "ws2_32.dll", "mswsock.dll", "iphlpapi.dll", "msvcrt.dll", "ucrtbase.dll",
}
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-win-")
VC_RUNTIME = {
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcomp140.dll", "concrt140.dll",
}


class NotPE(ValueError):
    pass


def imports(data):
    """The DLL names a PE image imports (import and delay-import tables), in table order."""
    if data[:2] != b"MZ" or len(data) < 0x40:
        raise NotPE("not a PE image (no MZ header)")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise NotPE("not a PE image (no PE signature)")
    sections, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        directories = optional + 112
    elif magic == 0x10B:
        directories = optional + 96
    else:
        raise NotPE(f"unknown optional header magic {magic:#x}")
    table = optional + optional_size
    spans = []
    for i in range(sections):
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, table + 40 * i + 8)
        spans.append((va, max(vsize, rsize), raw))

    def offset(rva):
        for va, size, raw in spans:
            if va <= rva < va + size:
                return raw + rva - va
        raise NotPE(f"RVA {rva:#x} is in no section")

    def name_at(rva):
        start = offset(rva)
        end = data.index(b"\0", start)
        return data[start:end].decode("ascii")

    names = []
    # Data directory 1: the import table, 20-byte descriptors ending in a zero one; the name RVA is at +12.
    # Data directory 13: the delay-import table, 32-byte descriptors; the name RVA is at +4.
    for index, size, at in ((1, 20, 12), (13, 32, 4)):
        rva = struct.unpack_from("<I", data, directories + 8 * index)[0]
        if not rva:
            continue
        cursor = offset(rva)
        while True:
            descriptor = data[cursor:cursor + size]
            if len(descriptor) < size or not any(descriptor):
                break
            names.append(name_at(struct.unpack_from("<I", descriptor, at)[0]))
            cursor += size
    return names


def check(path, deny):
    """(problems, imports) for one file."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        names = imports(data)
    except (NotPE, struct.error, ValueError, UnicodeDecodeError) as error:
        return [f"{path}: {error}"], []
    lower = [n.lower() for n in names]
    problems = []
    if not {"kernel32.dll", "ntdll.dll"} & set(lower):
        problems.append(f"{path}: imports neither kernel32.dll nor ntdll.dll; refusing to report it complete")
    beside = {n.lower() for n in os.listdir(os.path.dirname(os.path.abspath(path)))}
    for name in names:
        key = name.lower()
        if key in deny:
            known = False
        else:
            known = key in SYSTEM or key.startswith(SYSTEM_PREFIXES) or key in VC_RUNTIME
        if not known and key not in beside:
            problems.append(f"{path}: needs {name}, which is neither a Windows system library nor the Visual C++ "
                            "runtime, and is not beside it in the archive")
    return problems, names


def packed(directory):
    found = []
    for parent, _, files in os.walk(directory):
        found += [os.path.join(parent, f) for f in sorted(files) if f.lower().endswith((".exe", ".dll"))]
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deny", action="append", default=[], metavar="NAME",
                        help="treat NAME as neither a system library nor the VC++ runtime (for a planted check)")
    parser.add_argument("--packed", metavar="DIR", help="check every .exe and .dll under a packed release directory")
    parser.add_argument("files", nargs="*", metavar="PE")
    args = parser.parse_args(argv)
    if bool(args.packed) == bool(args.files):
        parser.error("give either PE files or --packed DIR")
    deny = {n.lower() for n in args.deny}
    problems = []
    files = args.files
    if args.packed:
        files = packed(args.packed)
        if not files:
            problems.append(f"{args.packed}: holds no .exe or .dll")
    for path in files:
        try:
            found, names = check(path, deny)
        except OSError as error:
            found, names = [f"{path}: {error.strerror}"], []
        print(f"{path}: imports {', '.join(names) or 'nothing read'}", file=sys.stderr)
        problems += found
    for problem in problems:
        print("FAIL " + problem, file=sys.stderr)
    if problems:
        return 1
    print(f"ok: every import is Windows, the Visual C++ runtime, or in the archive ({len(files)} checked)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
