#!/usr/bin/env python3
"""Licence notices for everything linked into the shipped binaries, generated and checked.

    scripts/third_party_notices.py --write                 # regenerate licenses/rust-crates.txt and python-crates.txt
    scripts/third_party_notices.py --check [FILE]          # exit 1 unless FILE (default: licenses/rust-crates.txt)
                                                           # is complete and current
    scripts/third_party_notices.py --check-python [FILE]   # the same for the wheel (default: licenses/python-crates.txt)

The crates are the normal (not build, not dev) dependencies of the packages that build the shipped binaries
(ggtts-render, ggtts-measure, and the Kitten tools ggtts-kitten-convert and ggtts-kitten) and the shipped C library
(ggtts-ffi's libggtts_ffi.so / ggtts_ffi.dll), resolved by `cargo metadata --locked` for every feature set a release is built
with, on every target an archive is released for. Each crate's name, version and licence expression come from its manifest, and its licence
texts from the files in its source directory in the cargo registry. Nothing is installed and nothing is written by
hand: a crate added to Cargo.lock without a regenerated notice fails --check, which CI and the pack step run.

The Python wheel (crates/ggtts-py, its own workspace and lockfile) has its own list, licenses/python-crates.txt: the
normal dependencies of its native module, resolved from crates/ggtts-py/Cargo.toml with the features the release
builds the wheel with (`python`, `pyo3/extension-module`, and `ggml` on Linux or `bindgen` on Windows): the engine
crates the archives link, plus PyO3 and its dependencies. scripts/python_wheel.sh packs it into the wheel.

--check also holds licenses/ggml-LICENSE.txt, the licence of the statically linked ggml, to the copy in the ggml
submodule when the submodule is checked out.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "licenses" / "rust-crates.txt"
PYTHON_NOTICES = ROOT / "licenses" / "python-crates.txt"
PYTHON_MANIFEST = ROOT / "crates" / "ggtts-py" / "Cargo.toml"
# The Python wheel's native module, ggtts-py's cdylib, with every feature set release.yml builds a wheel with.
PYTHON_LIBRARIES = ("ggtts_py",)
PYTHON_FEATURE_SETS = (("python", "pyo3/extension-module", "ggml"), ("python", "pyo3/extension-module", "bindgen"))
GGML_NOTICE = ROOT / "licenses" / "ggml-LICENSE.txt"
GGML_SOURCE = ROOT / "crates" / "ggtts-ggml-sys" / "ggml" / "LICENSE"
# The binaries `scripts/package_and_smoke.sh pack` copies into an archive. The Kitten tools are built from
# ggtts-kitten, which ggtts-render links too, so they add no crate of their own.
SHIPPED_BINARIES = ("ggtts-render", "ggtts-measure", "ggtts-kitten-convert", "ggtts-kitten")
# The C library pack copies into lib/: ggtts-ffi's cdylib (without its `jni` feature, which no archive ships).
SHIPPED_LIBRARIES = ("ggtts_ffi",)
# Every feature set a shipped binary or the library is built with: the default, and release.yml's CPU, Windows and
# CUDA builds (the CUDA archive's library is built against CPU ggml, as in the CPU archive).
FEATURE_SETS = ((), ("ggtts-cli/ggml", "ggtts-ffi/ggml"), ("ggtts-cli/bindgen", "ggtts-ffi/ggml"),
                ("ggtts-cli/cuda",))
# The targets archives are released for. A dependency that no released target compiles is not linked.
TARGETS = ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc")
# LICENSE, LICENSE-MIT, MIT-LICENSE, COPYING, NOTICE and the like; never a source file.
LICENCE_FILE = re.compile(r"(licen[sc]e|copying|notice|copyright)", re.IGNORECASE)
NOT_A_LICENCE = (".rs", ".toml", ".json", ".yml", ".yaml", ".lock")
CRATE_LINE = re.compile(r"^([A-Za-z0-9_-]+) (\S+) \| ")


def metadata(features, target, manifest=None):
    cmd = ["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", target]
    if manifest is not None:
        cmd += ["--manifest-path", str(manifest)]
    if features:
        cmd += ["--features", ",".join(features)]
    return json.loads(subprocess.check_output(cmd, cwd=ROOT))


def linked_packages(python=False):
    """Every third-party package reachable from a shipped binary's package through normal dependencies (with
    `python`, from the wheel's native module)."""
    found = {}
    binaries, libraries = ((), PYTHON_LIBRARIES) if python else (SHIPPED_BINARIES, SHIPPED_LIBRARIES)
    feature_sets = PYTHON_FEATURE_SETS if python else FEATURE_SETS
    for features, target in ((f, t) for f in feature_sets for t in TARGETS):
        meta = metadata(features, target, PYTHON_MANIFEST if python else None)
        packages = {p["id"]: p for p in meta["packages"]}
        nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
        roots = [p["id"] for p in meta["packages"] if p["id"] in meta["workspace_members"]
                 and any(("bin" in t["kind"] and t["name"] in binaries)
                         or ("cdylib" in t["kind"] and t["name"] in libraries) for t in p["targets"])]
        shipped = {t["name"] for r in roots for t in packages[r]["targets"] if {"bin", "cdylib"} & set(t["kind"])}
        if not set(binaries + libraries) <= shipped:
            raise ValueError("no workspace package builds "
                             + ", ".join(sorted(set(binaries + libraries) - shipped)))
        seen, stack = set(roots), list(roots)
        while stack:
            for dep in nodes[stack.pop()]["deps"]:
                if dep["pkg"] not in seen and any(k["kind"] is None for k in dep["dep_kinds"]):
                    seen.add(dep["pkg"])
                    stack.append(dep["pkg"])
        for pid in seen:
            package = packages[pid]
            if package["source"] is not None:
                found[(package["name"], package["version"])] = package
    return [found[key] for key in sorted(found)]


def licence_texts(package):
    directory = Path(package["manifest_path"]).parent
    files = sorted((f for f in directory.iterdir() if f.is_file() and LICENCE_FILE.search(f.name)
                    and not f.name.lower().endswith(NOT_A_LICENCE)),
                   key=lambda f: f.name.lower())
    texts = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.strip("\n").split("\n")) + "\n"
        texts.append((f.name, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], text))
    return texts


def generate(python=False):
    crates, texts, users, problems = [], {}, {}, []
    for package in linked_packages(python):
        name = f"{package['name']} {package['version']}"
        files = licence_texts(package)
        if not package.get("license"):
            problems.append(f"{name} declares no licence expression")
        if not files:
            problems.append(f"{name} has no licence file in its source")
        for _, key, text in files:
            texts[key] = text
            users.setdefault(key, []).append(name)
        crates.append(f"{name} | {package.get('license') or 'UNKNOWN'} | "
                      + ", ".join(f"{filename} [{key}]" for filename, key, _ in files))
    if problems:
        raise ValueError("; ".join(problems))
    if python:
        title = ["Licences of the Rust crates linked into the ggtts Python wheel's native module (ggtts._native, "
                 "built from crates/ggtts-py with PyO3)"]
    else:
        title = [f"Licences of the Rust crates linked into the ggtts binaries ({', '.join(SHIPPED_BINARIES)})"
                 f" and the C library ({', '.join(SHIPPED_LIBRARIES)})"]
    out = title + [
           "",
           "Generated by scripts/third_party_notices.py from `cargo metadata --locked` and each crate's own source.",
           "Do not edit: run `python3 scripts/third_party_notices.py --write` after changing Cargo.lock.",
           "Each line is a crate, its version, its licence expression, and its licence files with the id of their",
           "text below. Identical texts are printed once.",
           "",
           "== Crates =="]
    out += crates
    for key in sorted(texts, key=lambda k: (users[k][0], k)):
        out += ["", f"== Text [{key}], for: {', '.join(users[key])} ==", "", texts[key].rstrip("\n")]
    return "\n".join(out) + "\n"


def read(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def check(path, python=False):
    """Every reason the notices are not complete; empty when they are."""
    problems = []
    expected = generate(python)
    actual = read(path) if path.is_file() else ""
    if not actual:
        problems.append(f"{path} is missing or empty")
    have = {m.group(1) + " " + m.group(2) for m in map(CRATE_LINE.match, actual.split("\n")) if m}
    need = {m.group(1) + " " + m.group(2) for m in map(CRATE_LINE.match, expected.split("\n")) if m}
    for crate in sorted(need - have):
        problems.append(f"no notice for linked crate {crate}")
    for crate in sorted(have - need):
        problems.append(f"a notice for {crate}, which is not linked")
    if not problems and actual != expected:
        problems.append(f"{path} is not what the generator writes")
    if not GGML_NOTICE.is_file() or not read(GGML_NOTICE).strip():
        problems.append(f"{GGML_NOTICE} (the licence of the statically linked ggml) is missing or empty")
    elif GGML_SOURCE.is_file() and read(GGML_NOTICE) != read(GGML_SOURCE):
        problems.append(f"{GGML_NOTICE} differs from the ggml submodule's LICENSE")
    return problems


def main(argv):
    if argv == ["--write"]:
        NOTICES.write_text(generate(), encoding="utf-8", newline="\n")
        PYTHON_NOTICES.write_text(generate(python=True), encoding="utf-8", newline="\n")
        print(f"wrote {PYTHON_NOTICES}", file=sys.stderr)
        if GGML_SOURCE.is_file():
            GGML_NOTICE.write_text(read(GGML_SOURCE), encoding="utf-8", newline="\n")
        print(f"wrote {NOTICES}", file=sys.stderr)
        return 0
    if argv[:1] in (["--check"], ["--check-python"]) and len(argv) <= 2:
        python = argv[0] == "--check-python"
        given = Path(argv[1]) if len(argv) == 2 else None
        problems = check(given or (PYTHON_NOTICES if python else NOTICES), python)
        for problem in problems:
            print("third-party notices: " + problem, file=sys.stderr)
        if problems:
            print("third-party notices: run `python3 scripts/third_party_notices.py --write` and commit the result",
                  file=sys.stderr)
            return 1
        print("third-party notices: every crate linked into the shipped "
              + ("Python wheel" if python else "binaries and library") + " has its notice",
              file=sys.stderr)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"third-party notices: {exc}")
