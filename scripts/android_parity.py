#!/usr/bin/env python3
"""Compare two renders of the same request on two machines (here: an Android phone and desktop CPU).

Each side is a directory holding, for a run NAME, what `ggtts-render --story ... --trace NAME.trace.jsonl
--trace-level 2 --stream NAME.raw.wav --stream-raw` wrote. Compared, in order of how early a difference shows:

* the trace header: engine, model and codec sha256 (a mismatch means the comparison is void, not a parity result);
* each chunk's seed (from the story path: the story's title and the chunk's text, independent of the machine);
* every frame's codebook-0 token, where the engine traces steps (Qwen3-TTS does; Kitten-TTS 2 does not);
* each chunk's frame count and stop reason;
* the raw stream's samples (16-bit, before the level step): equal or not, and how far apart where they overlap.

    android_parity.py compare A_DIR B_DIR NAME...     # one line per run, and a verdict: exit 0 identical, 1 differ
    android_parity.py frames A_DIR B_DIR GLOB         # frames per chunk across many draws, side by side; the
                                                      # difference of means with its 95% permutation interval
    android_parity.py runlog A_DIR B_DIR GLOB         # per run: each chunk's seed, and frames and stop on each side
    android_parity.py --self-test                     # identical reads identical; each planted change is caught alone

`frames` and `runlog` refuse (exit 2) when nothing matches GLOB, when a run is on one side only, when a run's two
sides have different chunk counts, or when they are different draws (different chunk seeds).

Bit-exact output is not promised across CPU architectures: ggml's ARM and x86 kernels round differently. When the
tokens differ, `frames` compares the two machines' distributions over draws instead (a stated distribution parity).
"""

import glob
import itertools
import json
import os
import statistics
import struct
import sys
import tempfile


class Refused(ValueError):
    """A comparison that cannot be made as asked; never read as a pass."""


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
            r["identical"] = False
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
    # Every field compared is part of the verdict: a difference in any one of them makes the runs differ.
    r["checks"] = {
        "seeds": r["seeds_equal"],
        "frames": r["frames"]["a"] == r["frames"]["b"],
        "stops": r["stops"]["a"] == r["stops"]["b"],
        "tokens": r["tokens"] is None or r["tokens"]["first_diff"] is None,
        "samples": r["samples"]["identical"],
    }
    r["identical"] = all(r["checks"].values())
    return r


def permutation_p(xa, xb):
    """Exact two-sided permutation p-value for the difference of means of two small samples: the share of every
    relabelling of the pooled values whose mean difference is at least as large as the one observed."""
    pooled, n = list(xa) + list(xb), len(xa)
    observed = abs(statistics.mean(xa) - statistics.mean(xb))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        hits += abs(statistics.mean(a) - statistics.mean(b)) >= observed - 1e-12
    return hits / total


def shift_interval(xa, xb, level=0.05, step=0.1):
    """The 95% permutation interval for the shift d = mean(a) - mean(b): every d for which a shifted by -d is not
    told apart from b (permutation p >= level), scanned in steps of `step`."""
    d0 = statistics.mean(xa) - statistics.mean(xb)
    span = (max(list(xa) + list(xb)) - min(list(xa) + list(xb))) * 2 + 1
    inside = []
    k = -int(span / step)
    while k * step <= span:
        d = d0 + k * step
        if permutation_p([x - d for x in xa], xb) >= level:
            inside.append(d)
        k += 1
    return [round(min(inside), 1), round(max(inside), 1)] if inside else None


def matched_runs(dir_a, dir_b, pattern):
    """(name, chunks_a, chunks_b) for every run matching `pattern`, on both sides, refusing what cannot compare."""
    names_a = {os.path.basename(p)[: -len(".trace.jsonl")] for p in glob.glob(os.path.join(dir_a, pattern + ".trace.jsonl"))}
    names_b = {os.path.basename(p)[: -len(".trace.jsonl")] for p in glob.glob(os.path.join(dir_b, pattern + ".trace.jsonl"))}
    if not names_a and not names_b:
        raise Refused(f"no run matches {pattern!r} in either directory")
    if names_a != names_b:
        raise Refused(f"runs on one side only: {sorted(names_a ^ names_b)}")
    runs = []
    for name in sorted(names_a):
        _, ca, _ = read_trace(os.path.join(dir_a, name + ".trace.jsonl"))
        _, cb, _ = read_trace(os.path.join(dir_b, name + ".trace.jsonl"))
        if sorted(ca) != sorted(cb):
            raise Refused(f"{name}: {len(ca)} chunks on one side, {len(cb)} on the other")
        if [ca[c].get("seed") for c in sorted(ca)] != [cb[c].get("seed") for c in sorted(cb)]:
            raise Refused(f"{name}: the two sides are different draws (chunk seeds differ)")
        runs.append((name, ca, cb))
    return runs


def frames_table(dir_a, dir_b, pattern):
    """Frames per chunk for every run matching PATTERN, and the difference of the two sides' mean total frames,
    with its exact permutation p and 95% permutation interval."""
    runs = matched_runs(dir_a, dir_b, pattern)
    rows = [(name, [ca[c]["frames"] for c in sorted(ca)], [cb[c]["frames"] for c in sorted(cb)]) for name, ca, cb in runs]
    totals_a, totals_b = [sum(fa) for _, fa, _ in rows], [sum(fb) for _, _, fb in rows]
    diff = statistics.mean(totals_a) - statistics.mean(totals_b)
    summary = {"total_frames": {
        "a": totals_a, "b": totals_b,
        "a_mean": statistics.mean(totals_a), "b_mean": statistics.mean(totals_b),
        "diff": round(diff, 2), "diff_pct_of_b": round(100 * diff / statistics.mean(totals_b), 1),
        "permutation_p": permutation_p(totals_a, totals_b),
        "interval_95": shift_interval(totals_a, totals_b),
        "n": len(rows),
    }}
    for c in sorted(set(i for _, fa, _ in rows for i in range(len(fa)))):
        xa, xb = [fa[c] for _, fa, _ in rows], [fb[c] for _, _, fb in rows]
        summary[str(c)] = {"a_mean": statistics.mean(xa), "b_mean": statistics.mean(xb),
                           "a_range": [min(xa), max(xa)], "b_range": [min(xb), max(xb)]}
    return rows, summary


def run_log(dir_a, dir_b, pattern):
    """One line per run and chunk: the chunk's seed, and its frames and stop on each side."""
    lines = []
    for name, ca, cb in matched_runs(dir_a, dir_b, pattern):
        for c in sorted(ca):
            lines.append(f"{name}\tchunk {c}\tseed {ca[c]['seed']}\ta {ca[c].get('frames')} {ca[c].get('stop')}"
                         f"\tb {cb[c].get('frames')} {cb[c].get('stop')}")
    return lines


def self_test():
    def write(d, name, tokens, samples, engine="qwen3-tts", model="m", codec="c", seed=7, frames=None, stop="eos"):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.trace.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ev": "trace", "engine": engine, "model_sha256": model, "codec_sha256": codec}) + "\n")
            f.write(json.dumps({"ev": "chunk_begin", "chunk": 0, "seed": seed}) + "\n")
            for i, t in enumerate(tokens):
                f.write(json.dumps({"ev": "step", "chunk": 0, "frame": i, "token": t}) + "\n")
            f.write(json.dumps({"ev": "chunk_end", "chunk": 0, "frames": len(tokens) if frames is None else frames,
                                "stop": stop}) + "\n")
        body = struct.pack(f"<{len(samples)}h", *samples)
        hdr = b"RIFF\xff\xff\xff\xffWAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        open(os.path.join(d, f"{name}.raw.wav"), "wb").write(hdr + b"data\xff\xff\xff\xff" + body)

    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    base = dict(tokens=[1, 2, 3], samples=[0, 5, -5])
    write(a, "same", **base)
    write(b, "same", **base)
    # Each planted case changes one thing only; each must be caught by its own check.
    cases = {
        "tok": (dict(base), dict(base, tokens=[1, 9, 3]), "tokens"),
        "pcm": (dict(base), dict(base, samples=[0, 5, -4]), "samples"),
        "frames": (dict(base), dict(base, frames=4), "frames"),
        "seed": (dict(base), dict(base, seed=8), "seeds"),
        "stop": (dict(base), dict(base, stop="cap"), "stops"),
    }
    ok = compare(a, b, "same")["identical"] is True
    for name, (wa, wb, check) in cases.items():
        write(a, name, **wa)
        write(b, name, **wb)
        r = compare(a, b, name)
        failed = [k for k, v in r["checks"].items() if not v]
        ok &= r["identical"] is False and failed == [check]
    for name, field in (("void-model", dict(model="m2")), ("void-codec", dict(codec="c2")),
                        ("void-engine", dict(engine="kitten-tts-2"))):
        write(a, name, **base)
        write(b, name, **dict(base, **field))
        r = compare(a, b, name)
        ok &= "void" in r and r["identical"] is False and list(field)[0] in r["void"]
    # The permutation test and its interval: two samples from one distribution are not told apart; far ones are.
    ok &= permutation_p([10, 12, 11, 13], [12, 11, 10, 13]) > 0.5
    ok &= permutation_p([10, 11, 12, 10, 11, 12], [40, 41, 42, 40, 41, 42]) < 0.01
    iv = shift_interval([10, 11, 12, 10, 11, 12], [40, 41, 42, 40, 41, 42])
    ok &= iv is not None and iv[0] < -30 < iv[1] < 0
    # `frames` refuses what it cannot compare.
    e = os.path.join(tmp, "empty")
    os.makedirs(e, exist_ok=True)
    for args, says in (((e, e, "x-*"), "no run matches"), ((a, e, "same"), "one side only")):
        try:
            frames_table(*args)
            ok = False
        except Refused as err:
            ok &= says in str(err)
    c1, c2 = os.path.join(tmp, "c1"), os.path.join(tmp, "c2")
    write(c1, "run", **base)
    write(c2, "run", **base)
    with open(os.path.join(c2, "run.trace.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ev": "chunk_begin", "chunk": 1, "seed": 9}) + "\n")
        f.write(json.dumps({"ev": "chunk_end", "chunk": 1, "frames": 2, "stop": "eos"}) + "\n")
    try:
        frames_table(c1, c2, "run")
        ok = False
    except Refused as err:
        ok &= "chunks" in str(err)
    # Different draws (chunk seeds differ) are refused, by frames and by runlog.
    d1, d2 = os.path.join(tmp, "d1"), os.path.join(tmp, "d2")
    write(d1, "run", **base)
    write(d2, "run", **dict(base, seed=8))
    for f in (frames_table, run_log):
        try:
            f(d1, d2, "run")
            ok = False
        except Refused as err:
            ok &= "different draws" in str(err)
    print("android_parity: self-test " + ("passed: identical reads identical; a changed token, sample, frame count, "
                                          "seed and stop are each caught alone by their own check; a different "
                                          "model, codec or engine voids the comparison; the permutation test and its "
                                          "interval tell far samples apart and near ones not; frames refuses no "
                                          "runs, one-sided runs and unequal chunk counts, and frames and runlog "
                                          "refuse different draws" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    try:
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
        if len(argv) == 4 and argv[0] == "runlog":
            print("\n".join(run_log(argv[1], argv[2], argv[3])))
            return 0
    except Refused as err:
        print(f"android_parity: refused: {err}", file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
