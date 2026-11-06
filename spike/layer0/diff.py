"""Compare two raw f32 outputs of the layer: byte identity, the first differing element, and the largest gap.

Usage: python3 diff.py A.bin B.bin HIDDEN_SIZE
Exit status 0 when the files are byte-identical, 1 otherwise.
"""

import struct
import sys


def main() -> int:
    a_path, b_path, hidden = sys.argv[1], sys.argv[2], int(sys.argv[3])
    a = open(a_path, "rb").read()
    b = open(b_path, "rb").read()
    if len(a) != len(b):
        print(f"DIFFERENT sizes: {len(a)} and {len(b)} bytes")
        return 1
    n = len(a) // 4
    x = struct.unpack(f"<{n}f", a)
    y = struct.unpack(f"<{n}f", b)
    differing = [i for i in range(n) if a[4 * i : 4 * i + 4] != b[4 * i : 4 * i + 4]]
    if not differing:
        print(f"IDENTICAL: {len(a)} bytes, {n} floats")
        return 0
    first = differing[0]
    max_abs = max(abs(x[i] - y[i]) for i in differing)
    print(
        f"DIFFERENT: {len(differing)}/{n} floats differ, max abs diff {max_abs:.6g}, "
        f"first at token {first // hidden} element {first % hidden} ({x[first]!r} vs {y[first]!r})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
