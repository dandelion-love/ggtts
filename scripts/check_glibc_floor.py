#!/usr/bin/env python3
"""Refuse a Linux release binary that needs a newer C or C++ runtime than the stated floor.

    python3 scripts/check_glibc_floor.py [--floor NAME=VERSION ...] ELF...

The floor is what Ubuntu 22.04 ships (glibc 2.35, and the GCC 12 runtime libraries from jammy-updates: libstdc++,
libgcc_s, libgomp). For each ELF it reads the symbol versions the binary needs from the dynamic loader's point of
view (`readelf --version-info`, the `.gnu.version_r` section: one entry per library and version), takes the highest
in each namespace (GLIBC, GLIBCXX, CXXABI, GCC, GOMP, OMP), and fails when any exceeds the floor, naming the
namespace, the version, the library and the symbols that need it.

It cannot silently pass. It fails when an argument is not an ELF file, when readelf is missing or fails, when a
binary needs no GLIBC version at all (it was not read, or it is not the dynamically linked binary this was written
for), and when a binary needs a version namespace the floor does not name (GLIBC_PRIVATE, or a library added to the
link without a floor for it). `--floor NAME=VERSION` replaces one namespace's floor; CI uses it to plant an
impossible floor and require red.
"""
import argparse
import os
import re
import subprocess
import sys

# Ubuntu 22.04 (jammy, with jammy-updates): libc6 2.35, and libstdc++6, libgcc-s1 and libgomp1 12.3.0. These are the
# highest version each of those libraries defines, so a binary within them starts there.
FLOOR = {
    "GLIBC": "2.35",
    "GLIBCXX": "3.4.30",
    "CXXABI": "1.3.13",
    "GCC": "12.0.0",
    "GOMP": "5.1",
    "OMP": "5.1",
}
FLOOR_NAME = "Ubuntu 22.04"

VERSIONED = re.compile(r"^([A-Z][A-Z_]*?)_(\d+(?:\.\d+)*)$")


def parse_version(text):
    return tuple(int(part) for part in text.split("."))


def needed_versions(readelf_text):
    """[(library, version name, version index)] from `readelf --version-info --wide`'s .gnu.version_r section."""
    needs, in_needs, library = [], False, None
    for line in readelf_text.splitlines():
        if line.startswith("Version needs section"):
            in_needs = True
            continue
        if line.startswith(("Version symbols section", "Version definition section")):
            in_needs = False
            continue
        if not in_needs:
            continue
        match = re.search(r"\bFile: (\S+)", line)
        if match:
            library = match.group(1)
            continue
        match = re.search(r"\bName: (\S+)\s.*\bVersion: (\d+)", line)
        if match:
            needs.append((library, match.group(1), match.group(2)))
    return needs


def symbols_by_index(dynsym_text):
    """{version index: [symbol, ...]} for the undefined dynamic symbols, from `readelf --dyn-syms --wide`."""
    result = {}
    for line in dynsym_text.splitlines():
        match = re.match(r"^\s*\d+:(?:\s+\S+){5}\s+UND\s+([^@\s]+)@\S+\s+\((\d+)\)", line)
        if match:
            result.setdefault(match.group(2), []).append(match.group(1))
    return result


def readelf(*args):
    env = dict(os.environ, LC_ALL="C")
    return subprocess.run(["readelf", *args], check=True, capture_output=True, text=True, env=env).stdout


def check(path, floor):
    """(problems, {namespace: (highest version, library)}) for one file."""
    with open(path, "rb") as handle:
        if handle.read(4) != b"\x7fELF":
            return [f"{path}: not an ELF file"], {}
    return evaluate(path, readelf("--version-info", "--wide", path), readelf("--dyn-syms", "--wide", path), floor)


def evaluate(path, version_info, dyn_syms, floor):
    """check() on readelf's two outputs, so a test can hand it any binary's."""
    needs = needed_versions(version_info)
    symbols = symbols_by_index(dyn_syms)
    problems, highest = [], {}
    for library, name, index in needs:
        match = VERSIONED.match(name)
        if not match or match.group(1) not in floor:
            problems.append(f"{path}: needs {name} from {library}, a version namespace with no floor here "
                            f"(floors: {', '.join(sorted(floor))})")
            continue
        namespace, version = match.groups()
        if namespace not in highest or parse_version(version) > parse_version(highest[namespace][0]):
            highest[namespace] = (version, library)
        if parse_version(version) > parse_version(floor[namespace]):
            users = ", ".join(sorted(symbols.get(index, []))[:8]) or "symbols not listed"
            problems.append(f"{path}: {name} from {library} exceeds the floor {namespace}_{floor[namespace]} "
                            f"(needed by: {users})")
    if "GLIBC" not in highest:
        problems.append(f"{path}: needs no GLIBC version at all; refusing to report it within the floor")
    return problems, highest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--floor", action="append", default=[], metavar="NAME=VERSION",
                        help="replace one namespace's floor (for a planted check)")
    parser.add_argument("binaries", nargs="+", metavar="ELF")
    args = parser.parse_args(argv)
    floor = dict(FLOOR)
    for item in args.floor:
        name, sep, version = item.partition("=")
        if not sep or name not in FLOOR or not re.fullmatch(r"\d+(\.\d+)*", version):
            parser.error(f"--floor wants NAME=VERSION with NAME one of {', '.join(sorted(FLOOR))}, not {item!r}")
        floor[name] = version
    print("floor: " + ", ".join(f"{name}_{floor[name]}" for name in sorted(floor))
          + ("" if floor == FLOOR else f" (not the {FLOOR_NAME} floor: overridden)"), file=sys.stderr)
    problems = []
    for path in args.binaries:
        try:
            found, highest = check(path, floor)
        except FileNotFoundError as error:
            found, highest = [f"{path}: {error.strerror}: {error.filename}"], {}
        except subprocess.CalledProcessError as error:
            found, highest = [f"{path}: readelf failed: {error.stderr.strip()}"], {}
        summary = ", ".join(f"{name}_{highest[name][0]} ({highest[name][1]})" for name in sorted(highest))
        print(f"{path}: highest needed: {summary or 'none'}", file=sys.stderr)
        problems += found
    for problem in problems:
        print("FAIL " + problem, file=sys.stderr)
    if problems:
        return 1
    print(f"ok: every binary is within the floor ({len(args.binaries)} checked)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
