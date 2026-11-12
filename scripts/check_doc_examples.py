#!/usr/bin/env python3
"""Hold the binding docs' code to the example files CI compiles and runs.

Every C, Python or Kotlin code block in docs/bindings/*.md must be a copy of an example file under
docs/bindings/examples/, named by a marker on the line before the block:

    <!-- example: examples/c/first_render.c -->            the whole file
    <!-- example: examples/c/stream.c lines=20-34 -->      lines 20 to 34 of it, as numbered from 1

The block must equal the file (or the lines) exactly. A code block in one of those languages with no marker, a marker
naming a file that does not exist, and an example file no page shows are each refused. Shell and other blocks are
not checked: they are commands, not programs. CI compiles and runs the files themselves (scripts/c_examples.sh, the
Kotlin binding job, crates/ggtts-py/tests/test_doc_examples.py), so a block that matches its file cannot rot.

    python3 scripts/check_doc_examples.py [DOCS_DIR]    # default: docs/bindings
    python3 scripts/check_doc_examples.py --self-test   # each refusal fires on a planted page, and a good page passes
"""
import re
import sys
import tempfile
from pathlib import Path

CHECKED = {"c", "python", "kotlin"}
MARKER = re.compile(r"^<!-- example: (\S+)(?: lines=(\d+)-(\d+))? -->$")
FENCE = re.compile(r"^```(\S*)\s*$")


def blocks(text):
    """(language, first line number, code, marker line or None) for each fenced block."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(lines) and lines[j] != "```":
            j += 1
        if j == len(lines):
            raise ValueError(f"line {start + 1}: a code block that is never closed")
        marker = lines[start - 1] if start > 0 and MARKER.match(lines[start - 1]) else None
        yield m.group(1), start + 1, "\n".join(lines[start + 1:j]), marker
        i = j + 1


def check(docs):
    docs = Path(docs)
    problems, shown = [], set()
    pages = sorted(docs.glob("*.md"))
    if not pages:
        return [f"no pages in {docs}"]
    for page in pages:
        try:
            found = list(blocks(page.read_text(encoding="utf-8")))
        except ValueError as e:
            problems.append(f"{page.name}: {e}")
            continue
        for lang, line, code, marker in found:
            if lang not in CHECKED:
                continue
            where = f"{page.name}:{line}"
            if marker is None:
                problems.append(f"{where}: a {lang} block with no '<!-- example: ... -->' marker before it")
                continue
            m = MARKER.match(marker)
            rel, first, last = m.group(1), m.group(2), m.group(3)
            path = docs / rel
            if not path.is_file():
                problems.append(f"{where}: the marker names {rel}, which does not exist")
                continue
            shown.add(path.resolve())
            want = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
            if first:
                file_lines = want.split("\n")
                a, b = int(first), int(last)
                if not 1 <= a <= b <= len(file_lines):
                    problems.append(f"{where}: lines={a}-{b} is outside {rel} ({len(file_lines)} lines)")
                    continue
                want = "\n".join(file_lines[a - 1:b])
            if code != want:
                problems.append(f"{where}: the block is not {rel}" + (f" lines {first}-{last}" if first else "")
                                + ": copy the file into the page again")
    examples = docs / "examples"
    for f in sorted(examples.rglob("*")) if examples.is_dir() else []:
        if f.is_file() and f.suffix in (".c", ".py", ".kt") and f.resolve() not in shown:
            problems.append(f"{f.relative_to(docs)}: an example no page shows")
    return problems


def self_test():
    ok = True

    def expect(name, files, fragment):
        nonlocal ok
        with tempfile.TemporaryDirectory() as d:
            for rel, text in files.items():
                p = Path(d, rel)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            got = check(d)
        if fragment is None and got:
            print(f"self-test FAILED: {name}: refused a good page: {got}", file=sys.stderr)
            ok = False
        elif fragment is not None and not any(fragment in g for g in got):
            print(f"self-test FAILED: {name}: expected a refusal with {fragment!r}, got {got}", file=sys.stderr)
            ok = False
        else:
            print(f"self-test ok: {name}", file=sys.stderr)

    src = "int main(void) {\n    return 0;\n}\n"
    good = "# C\n\n<!-- example: examples/c/a.c -->\n```c\n" + src + "```\n\n```sh\ncc a.c\n```\n"
    expect("a page that copies its example passes", {"c.md": good, "examples/c/a.c": src}, None)
    expect("a slice of the file passes", {
        "c.md": good + "\n<!-- example: examples/c/a.c lines=2-2 -->\n```c\n    return 0;\n```\n",
        "examples/c/a.c": src}, None)
    expect("a block that differs from its file is refused",
           {"c.md": good.replace("return 0", "return 1"), "examples/c/a.c": src}, "the block is not examples/c/a.c")
    expect("a block with no marker is refused",
           {"c.md": good.replace("<!-- example: examples/c/a.c -->\n", ""), "examples/c/a.c": src}, "no '<!-- example")
    expect("a marker naming a missing file is refused",
           {"c.md": good.replace("a.c -->", "b.c -->"), "examples/c/a.c": src}, "which does not exist")
    expect("an example no page shows is refused",
           {"c.md": good, "examples/c/a.c": src, "examples/python/b.py": "print(1)\n"}, "an example no page shows")
    expect("a slice outside the file is refused", {
        "c.md": good + "\n<!-- example: examples/c/a.c lines=2-9 -->\n```c\n    return 0;\n```\n",
        "examples/c/a.c": src}, "is outside")
    return ok


def main():
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    docs = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "docs" / "bindings")
    problems = check(docs)
    for p in problems:
        print(f"check_doc_examples: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"check_doc_examples: every C, Python and Kotlin block in {docs} is a copy of its example file",
          file=sys.stderr)


if __name__ == "__main__":
    main()
