#!/usr/bin/env python3
"""Refuse a line added to the conformance lists since a base revision, so both lists only shrink.

    python3 scripts/check_conformance_base.py --base REF
    python3 scripts/check_conformance_base.py --self-test

The two lists are conformance/gaps.txt (the known gaps of the capability conformance test) and
conformance/render-path-exceptions.txt (the files that may call an engine directly). The conformance test
(crates/ggtts-cli/tests/conformance.rs) fails on a gap or a call that is not listed; this check fails on a listed line
that is not in the same file at REF, so a gap or an exception can be removed and never added. A pull request passes
its base, for example `--base origin/rc`. A file that does not exist at REF is reported and not compared (the change
that adds it). Blank lines and `#` comments are not entries. Exit 0 when nothing was added, 1 when something was,
2 on a usage or git error.
"""

import argparse
import subprocess
import sys

LISTS = ("conformance/gaps.txt", "conformance/render-path-exceptions.txt")


def entries(text):
    """The entries of a list file: its lines less blank lines and comments, line ends removed."""
    return [l.rstrip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def added(base_text, head_text):
    """The entries of head_text that base_text does not have, in head_text's order."""
    base = set(entries(base_text))
    return [l for l in entries(head_text) if l not in base]


def at(ref, path):
    """The file at ref, or None when it does not exist there."""
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True)
    if r.returncode != 0:
        exists = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], capture_output=True)
        if exists.returncode != 0:
            sys.exit(f"check_conformance_base: {ref} is not a commit")
        return None
    return r.stdout.decode("utf-8")


def check(ref):
    refused = []
    for path in LISTS:
        base = at(ref, path)
        with open(path, encoding="utf-8") as f:
            head = f.read()
        if base is None:
            print(f"{path}: not in {ref}, nothing to compare (the change that adds it)")
            continue
        new = added(base, head)
        removed = len(set(entries(base)) - set(entries(head)))
        print(f"{path}: {len(entries(head))} entries, {removed} removed since {ref}, {len(new)} added")
        for line in new:
            refused.append(f"{path}: added since {ref}: {line!r}")
    for r in refused:
        print(f"error: {r}", file=sys.stderr)
    if refused:
        print("These lists only shrink: carry or refuse the option, or render through the shared pipeline, instead.",
              file=sys.stderr)
    return 1 if refused else 0


def self_test():
    base = "# comment\nc-abi\tobserve.trace\t#280\nbook\trender.edge_s\t#283\n"
    assert added(base, base) == []
    assert added(base, "c-abi\tobserve.trace\t#280\n") == [], "a removed line is allowed"
    assert added(base, base + "\n# another comment\n") == [], "comments and blank lines are not entries"
    planted = base + "render-story\tload.planted\t#283\n"
    assert added(base, planted) == ["render-story\tload.planted\t#283"], "an added line is refused"
    changed = base.replace("#280", "#291")
    assert added(base, changed) == ["c-abi\tobserve.trace\t#291"], "a line changed is a line added"
    print("check_conformance_base: self-test passed (an added line and a changed line are refused)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", help="the revision the lists may only shrink from, for example origin/rc")
    p.add_argument("--self-test", action="store_true", help="show the check refuses an added line")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.base:
        p.error("give --base REF or --self-test")
    return check(a.base)


if __name__ == "__main__":
    sys.exit(main())
