#!/usr/bin/env python3
"""Make a synthetic two-layer talker from the pinned tiny cut, for CPU tests only.

Prints a newly created temporary GGUF path. This is not a speech-quality model.
Input must match fixtures/tiny/SOURCE's CUT_SHA256. No output path is accepted.
"""
import argparse
import hashlib
import os
from pathlib import Path
import struct
import tempfile

from cut_tiny_fixture import Reader, TALKER_COUNT, UINT32

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/ggtts-qwen3-tts/fixtures/tiny/SOURCE"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make(source):
    entries = dict(line.split("=", 1) for line in SOURCE.read_text().splitlines()
                   if "=" in line and not line.startswith("#"))
    if digest(source) != entries["CUT_SHA256"]:
        raise ValueError("input does not match the pinned one-layer tiny cut")
    with source.open("rb") as f:
        r = Reader(f)
        if r.raw(4) != b"GGUF" or r.u32() != 3:
            raise ValueError("expected GGUF version 3")
        nt, nk = r.u64(), r.u64()
        kv, alignment, found_count = [], 32, False
        for _ in range(nk):
            key, ty = r.string(), r.u32()
            val = r.value_bytes(ty)
            if key.decode() == TALKER_COUNT:
                if found_count or ty != UINT32 or val != struct.pack("<I", 1):
                    raise ValueError("expected exactly one talker layer")
                val, found_count = struct.pack("<I", 2), True
            if key == b"general.alignment":
                if ty != UINT32:
                    raise ValueError("invalid alignment type")
                alignment = struct.unpack("<I", val)[0]
            kv.append((key, ty, val))
        if not found_count or alignment == 0 or alignment & (alignment - 1):
            raise ValueError("missing layer count or invalid alignment")
        infos = []
        for _ in range(nt):
            name, nd = r.string(), r.u32()
            if not 1 <= nd <= 4:
                raise ValueError("invalid tensor dimensions")
            infos.append((name, nd, r.raw(nd * 8), r.u32(), r.u64()))
        pad = lambda n: (n + alignment - 1) // alignment * alignment
        start = pad(f.tell())
        f.seek(0, os.SEEK_END)
        size = f.tell() - start
        ordered = sorted(infos, key=lambda t: t[4])
        if len({t[0] for t in infos}) != len(infos) or len({t[4] for t in infos}) != len(infos):
            raise ValueError("duplicate tensor name or offset")
        lengths = {}
        for i, t in enumerate(ordered):
            end = ordered[i + 1][4] if i + 1 < len(ordered) else size
            if t[4] % alignment or t[4] >= end or end > size:
                raise ValueError("invalid tensor offsets")
            lengths[t[0]] = end - t[4]
        prefix = b"talker.blk.0."
        originals = [t for t in infos if t[0].startswith(prefix)]
        if not originals or any(t[0].startswith(b"talker.blk.1.") for t in infos):
            raise ValueError("expected layer zero and no layer one")
        duplicates = [(b"talker.blk.1." + t[0][len(prefix):], *t[1:]) for t in originals]
        output = infos + duplicates
        offsets, at = {}, 0
        for name, _, _, _, off in output:
            original = prefix + name[len(b"talker.blk.1."):] if name.startswith(b"talker.blk.1.") else name
            offsets[name] = at
            at += pad(lengths[original])
        directory = Path(tempfile.mkdtemp(prefix="ggtts-two-layer-"))
        target = directory / "synthetic-two-layer.gguf"
        try:
            with target.open("xb") as o:
                o.write(b"GGUF" + struct.pack("<IQQ", 3, len(output), len(kv)))
                for key, ty, val in kv:
                    o.write(struct.pack("<Q", len(key)) + key + struct.pack("<I", ty) + val)
                for name, nd, dims, ty, _ in output:
                    o.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", nd) + dims)
                    o.write(struct.pack("<IQ", ty, offsets[name]))
                o.write(b"\0" * (pad(o.tell()) - o.tell()))
                base = o.tell()
                for name, _, _, _, off in output:
                    original = prefix + name[len(b"talker.blk.1."):] if name.startswith(b"talker.blk.1.") else name
                    o.write(b"\0" * (base + offsets[name] - o.tell()))
                    f.seek(start + off)
                    left = lengths[original]
                    while left:
                        block = f.read(min(left, 1 << 20))
                        if not block:
                            raise ValueError("input truncated while copying")
                        o.write(block)
                        left -= len(block)
        except BaseException:
            target.unlink(missing_ok=True)
            directory.rmdir()
            raise
        return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="pinned one-layer tiny talker GGUF")
    args = ap.parse_args()
    try:
        print(make(args.source))
    except (OSError, ValueError, KeyError) as e:
        ap.exit(1, f"fixture refused: {e}\n")


if __name__ == "__main__":
    main()
