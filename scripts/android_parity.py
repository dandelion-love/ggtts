#!/usr/bin/env python3
"""Compare two renders of the same request on two machines (here: an Android phone and desktop CPU).

Each side is a directory holding, for a run NAME, what `ggtts-render --story ... --seed S --trace NAME.trace.jsonl
--trace-level 2 --stream NAME.raw.wav --stream-raw` wrote. Compared, in order of how early a difference shows:

* the trace header: engine, model and codec sha256 (a mismatch means the comparison is void, not a parity result);
* each chunk's seed (from the story path, independent of the machine);
* every frame's codebook-0 token, where the engine traces steps (Qwen3-TTS does; Kitten-TTS 2 does not);
* each chunk's frame count and stop reason;
* the raw stream's samples (16-bit, before the level step): equal or not, and how far apart where they overlap.

    android_parity.py compare A_DIR B_DIR NAME...     # one line per run, and a verdict: exit 0 identical, 1 differ
    android_parity.py frames DIR_A DIR_B GLOB         # frames per chunk across many draws, side by side, with an
                                                      # exact permutation test on each run's total frames
    android_parity.py --self-test                     # identical files read as identical; each planted change is caught

Bit-exact output is not promised across CPU architectures: ggml's ARM and x86 kernels round differently. When the
tokens differ, `frames` compares the two machines' distributions over seeds instead (a stated distribution parity).
"""

import glob
import itertools
import json
import os
import statistics
import struct
import sys
import tempfile


def read_trace(path):
    header, chunks, steps = None, {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            ev = e.get("ev")
            if ev == "trace":
                header = e
            elif ev == "chunk_begin":
                chunks.setdefault(e["chunk"], {})["seed"] = e["seed"]
            elif ev == "chunk_end":
                c = chunks.setdefault(e["chunk"], {})
                c["frames"], c["stop"] = e["frames"], e["stop"]
            elif ev == "step":
                steps[(e["chunk"], e["frame"])] = e["token"]
    if header is None:
        raise ValueError(f"{path}: no trace header")
    return header, chunks, steps


def read_stream_wav(path):
    """The samples of a ggtts stream WAV (16-bit mono; its length fields are 0xffffffff, so read to the end)."""
    data = open(path, "rb").read()
    i = data.find(b"data")
    if data[:4] != b"RIFF" or i < 0:
        raise ValueError(f"{path}: not a WAV stream")
    body = data[i + 8:]
    body = body[: len(body) - len(body) % 2]
    return struct.unpack(f"<{len(body) // 2}h", body)


def compare(a_dir, b_dir, name):
    ha, ca, sa = read_trace(os.path.join(a_dir, f"{name}.trace.jsonl"))
    hb, cb, sb = read_trace(os.path.join(b_dir, f"{name}.trace.jsonl"))
    r = {"name": name}
    for k in ("engine", "model_sha256", "codec_sha256"):
        if ha.get(k) != hb.get(k):
            r["void"] = f"{k} differs: {ha.get(k)} vs {hb.get(k)}"
            return r
    r["engine"] = ha["engine"]
    r["seeds_equal"] = [ca[c].get("seed") for c in sorted(ca)] == [cb[c].get("seed") for c in sorted(cb)]
    r["frames"] = {"a": [ca[c].get("frames") for c in sorted(ca)], "b": [cb[c].get("frames") for c in sorted(cb)]}
    r["stops"] = {"a": [ca[c].get("stop") for c in sorted(ca)], "b": [cb[c].get("stop") for c in sorted(cb)]}
    if sa or sb:
        keys = sorted(set(sa) | set(sb))
        same = [k for k in keys if sa.get(k) == sb.get(k)]
        first = next((k for k in keys if sa.get(k) != sb.get(k)), None)
        r["tokens"] = {
            "frames_a": len(sa), "frames_b": len(sb), "equal": len(same),
            "first_diff": None if first is None else {"chunk": first[0], "frame": first[1],
                                                       "a": sa.get(first), "b": sb.get(first)},
        }
        # Per chunk: how many leading frames agree before the first difference.
        lead = {}
        for c in sorted(set(k[0] for k in keys)):
            n = 0
            while (c, n) in sa and sa.get((c, n)) == sb.get((c, n)):
                n += 1
            lead[c] = n
        r["tokens"]["leading_equal_per_chunk"] = lead
    else:
        r["tokens"] = None  # the engine traces no steps
    wa = read_stream_wav(os.path.join(a_dir, f"{name}.raw.wav"))
    wb = read_stream_wav(os.path.join(b_dir, f"{name}.raw.wav"))
    n = min(len(wa), len(wb))
    diffs = [abs(x - y) for x, y in zip(wa[:n], wb[:n])]
    first_s = next((i for i, d in enumerate(diffs) if d), None)
    r["samples"] = {"a": len(wa), "b": len(wb), "identical": wa == wb, "first_diff_sample": first_s,
                    "max_abs_diff_overlap": max(diffs) if diffs else 0}
    identical = (r["seeds_equal"] and r["frames"]["a"] == r["frames"]["b"] and r["samples"]["identical"]
                 and (r["tokens"] is None or r["tokens"]["first_diff"] is None))
    r["identical"] = identical
    return r


def permutation_p(xa, xb):
    """Exact two-sided permutation p-value for the difference of means of two small samples: the share of every
    relabelling of the pooled values whose mean difference is at least as large as the one observed."""
    pooled, n = xa + xb, len(xa)
    observed = abs(statistics.mean(xa) - statistics.mean(xb))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        hits += abs(statistics.mean(a) - statistics.mean(b)) >= observed - 1e-12
    return hits / total


def frames_table(dir_a, dir_b, pattern):
    """Frames per chunk for every run matching PATTERN that both sides have, and each side's mean and spread."""
    rows, per = [], {"a": {}, "b": {}}
    for path in sorted(glob.glob(os.path.join(dir_a, pattern + ".trace.jsonl"))):
        name = os.path.basename(path)[: -len(".trace.jsonl")]
        other = os.path.join(dir_b, name + ".trace.jsonl")
        if not os.path.exists(other):
            continue
        _, ca, _ = read_trace(path)
        _, cb, _ = read_trace(other)
        fa = [ca[c]["frames"] for c in sorted(ca)]
        fb = [cb[c]["frames"] for c in sorted(cb)]
        rows.append((name, fa, fb))
        for side, fr in (("a", fa), ("b", fb)):
            for c, f in enumerate(fr):
                per[side].setdefault(c, []).append(f)
    summary = {}
    totals_a, totals_b = [sum(fa) for _, fa, _ in rows], [sum(fb) for _, _, fb in rows]
    if rows:
        summary["total_frames"] = {
            "a": totals_a, "b": totals_b,
            "a_mean": statistics.mean(totals_a), "b_mean": statistics.mean(totals_b),
            "permutation_p": permutation_p(totals_a, totals_b),
        }
    for c in sorted(per["a"]):
        xa, xb = per["a"][c], per["b"].get(c, [])
        summary[c] = {
            "a_mean": statistics.mean(xa), "b_mean": statistics.mean(xb),
            "a_sd": statistics.pstdev(xa), "b_sd": statistics.pstdev(xb),
            "a_range": [min(xa), max(xa)], "b_range": [min(xb), max(xb)], "n": len(xa),
        }
    return rows, summary


def self_test():
    def write(d, name, tokens, samples, engine="qwen3-tts"):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.trace.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ev": "trace", "engine": engine, "model_sha256": "m", "codec_sha256": "c"}) + "\n")
            f.write(json.dumps({"ev": "chunk_begin", "chunk": 0, "seed": 7}) + "\n")
            for i, t in enumerate(tokens):
                f.write(json.dumps({"ev": "step", "chunk": 0, "frame": i, "token": t}) + "\n")
            f.write(json.dumps({"ev": "chunk_end", "chunk": 0, "frames": len(tokens), "stop": "eos"}) + "\n")
        body = struct.pack(f"<{len(samples)}h", *samples)
        hdr = b"RIFF\xff\xff\xff\xffWAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        open(os.path.join(d, f"{name}.raw.wav"), "wb").write(hdr + b"data\xff\xff\xff\xff" + body)

    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    write(a, "same", [1, 2, 3], [0, 5, -5])
    write(b, "same", [1, 2, 3], [0, 5, -5])
    write(a, "tok", [1, 2, 3], [0, 5, -5])
    write(b, "tok", [1, 9, 3], [0, 5, -5])
    write(a, "pcm", [1, 2, 3], [0, 5, -5])
    write(b, "pcm", [1, 2, 3], [0, 5, -4])
    write(a, "len", [1, 2, 3], [0, 5, -5])
    write(b, "len", [1, 2, 3, 4], [0, 5, -5, 1])
    ok = True
    r = compare(a, b, "same")
    ok &= r["identical"] and r["samples"]["identical"]
    r = compare(a, b, "tok")
    ok &= (not r["identical"]) and r["tokens"]["first_diff"] == {"chunk": 0, "frame": 1, "a": 2, "b": 9}
    r = compare(a, b, "pcm")
    ok &= (not r["identical"]) and r["samples"]["max_abs_diff_overlap"] == 1 and r["samples"]["first_diff_sample"] == 2
    r = compare(a, b, "len")
    ok &= (not r["identical"]) and r["frames"] == {"a": [3], "b": [4]}
    write(b, "void", [1, 2, 3], [0, 5, -5], engine="kitten-tts-2")
    write(a, "void", [1, 2, 3], [0, 5, -5])
    ok &= "void" in compare(a, b, "void")
    # The permutation test: two samples from one distribution are not told apart; two far apart are.
    ok &= permutation_p([10, 12, 11, 13], [12, 11, 10, 13]) > 0.5
    ok &= permutation_p([10, 11, 12, 10, 11, 12], [40, 41, 42, 40, 41, 42]) < 0.01
    print("android_parity: self-test " + ("passed: identical reads identical; a changed token, sample, frame count "
                                          "and engine are each caught; the permutation test tells far samples apart and "
                                          "near ones not" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    if len(argv) >= 4 and argv[0] == "compare":
        rc = 0
        for name in argv[3:]:
            r = compare(argv[1], argv[2], name)
            print(json.dumps(r, ensure_ascii=False))
            rc |= 0 if r.get("identical") else 1
        return rc
    if len(argv) == 4 and argv[0] == "frames":
        rows, summary = frames_table(argv[1], argv[2], argv[3])
        for name, fa, fb in rows:
            print(f"{name}\ta={fa}\tb={fb}")
        print(json.dumps(summary))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
