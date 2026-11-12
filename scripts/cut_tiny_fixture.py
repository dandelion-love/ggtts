#!/usr/bin/env python3
"""Cut the tiny real-model fixture from the published Qwen3-TTS talker GGUF.

    python3 scripts/cut_tiny_fixture.py IN.gguf OUT.gguf [--talker-layers 1] [--code-pred-layers 1]

The output keeps every metadata key and every tensor of the input, byte for byte, except:
  * `qwen3-tts.talker.block_count` and `qwen3-tts.code_pred.block_count` are set to the layer counts asked for;
  * the tensors of the dropped layers (`talker.blk.N.*` and `code_pred.blk.N.*` for N at or above those counts) are
    left out, and the remaining tensors' data offsets are recomputed.

Nothing is re-quantised or converted, so every kept tensor is the published one. The output is deterministic: the
same input and arguments give the same bytes, and `scripts/tiny_fixture.sh` checks its sha256 against the committed
one. Standard library only.

A model cut this way cannot speak: it is for checks of identity (the same tokens and samples with and without an
instrument, at a fixed seed), never for judging audio.
"""
import argparse
import re
import struct
import sys

TALKER_COUNT = "qwen3-tts.talker.block_count"
CP_COUNT = "qwen3-tts.code_pred.block_count"
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY, UINT32 = 8, 9, 4


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            sys.exit("unexpected end of file")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        n = self.u64()
        return self.raw(n)

    def value_bytes(self, ty):
        """The value's bytes as stored, without its type tag."""
        if ty in SCALAR:
            return self.raw(SCALAR[ty])
        if ty == STRING:
            s = self.string()
            return struct.pack("<Q", len(s)) + s
        if ty == ARRAY:
            et, n = self.u32(), self.u64()
            body = b"".join(self.value_bytes(et) for _ in range(n))
            return struct.pack("<IQ", et, n) + body
        sys.exit(f"unknown GGUF value type {ty}")


def layer_of(name, prefix):
    m = re.match(re.escape(prefix) + r"\.blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--talker-layers", type=int, default=1)
    ap.add_argument("--code-pred-layers", type=int, default=1)
    a = ap.parse_args()

    with open(a.inp, "rb") as f:
        r = Reader(f)
        if r.raw(4) != b"GGUF":
            sys.exit(f"{a.inp}: not a GGUF file")
        version = r.u32()
        if version != 3:
            sys.exit(f"{a.inp}: GGUF version {version}, expected 3")
        n_tensors, n_kv = r.u64(), r.u64()
        kv, alignment = [], 32
        for _ in range(n_kv):
            key = r.string()
            ty = r.u32()
            val = r.value_bytes(ty)
            k = key.decode()
            if k in (TALKER_COUNT, CP_COUNT):
                if ty != UINT32:
                    sys.exit(f"{k} is type {ty}, expected uint32")
                have = struct.unpack("<I", val)[0]
                want = a.talker_layers if k == TALKER_COUNT else a.code_pred_layers
                if not 0 < want <= have:
                    sys.exit(f"{k}: {want} layers asked, the input has {have}")
                val = struct.pack("<I", want)
            if k == "general.alignment":
                alignment = struct.unpack("<I", val)[0]
            kv.append((key, ty, val))
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = r.raw(8 * nd)
            ty = r.u32()
            off = r.u64()
            infos.append((name, nd, dims, ty, off))
        header_end = f.tell()
        data_start = (header_end + alignment - 1) // alignment * alignment
        f.seek(0, 2)
        file_end = f.tell()

        # A tensor's stored size is the distance to the next tensor's offset (or to the end of the file).
        by_off = sorted(infos, key=lambda t: t[4])
        size = {}
        for i, t in enumerate(by_off):
            nxt = by_off[i + 1][4] if i + 1 < len(by_off) else file_end - data_start
            size[t[0]] = nxt - t[4]

        def kept(name):
            n = name.decode()
            tl, cl = layer_of(n, "talker"), layer_of(n, "code_pred")
            return not ((tl is not None and tl >= a.talker_layers) or (cl is not None and cl >= a.code_pred_layers))

        keep = [t for t in infos if kept(t[0])]
        if len(keep) == len(infos):
            sys.exit("nothing to cut: the layer counts asked for are the input's")

        def pad(n):
            return (n + alignment - 1) // alignment * alignment

        new_off, at = {}, 0
        for t in keep:
            new_off[t[0]] = at
            at += pad(size[t[0]])

        with open(a.out, "wb") as o:
            o.write(b"GGUF" + struct.pack("<IQQ", 3, len(keep), len(kv)))
            for key, ty, val in kv:
                o.write(struct.pack("<Q", len(key)) + key + struct.pack("<I", ty) + val)
            for name, nd, dims, ty, _ in keep:
                o.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", nd) + dims)
                o.write(struct.pack("<IQ", ty, new_off[name]))
            o.write(b"\0" * (pad(o.tell()) - o.tell()))
            base = o.tell()
            for name, _, _, _, off in keep:
                o.write(b"\0" * (base + new_off[name] - o.tell()))
                f.seek(data_start + off)
                left = size[name]
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    o.write(chunk)
                    left -= len(chunk)
    print(
        f"{a.out}: {len(keep)} of {n_tensors} tensors, talker layers {a.talker_layers}, "
        f"code predictor layers {a.code_pred_layers}"
    )


if __name__ == "__main__":
    main()
