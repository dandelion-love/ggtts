#!/usr/bin/env python3
"""Generate and validate self-contained ggtts feedback pages."""
import argparse
import base64
import binascii
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "feedback" / "page.html"
MARKER = "<!-- GGTTS_INPUT -->"
INPUT_FORMAT = "ggtts-feedback-input/1"
RESULT_FORMAT = "ggtts-feedback-result/1"
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
AUDIO = "data:audio/flac;base64,"
RATINGS = {"accept", "reject", "unsure"}

def cjk(char):
    n = ord(char)
    return any(lo <= n <= hi for lo, hi in (
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        (0x20000, 0x2FA1F), (0x30000, 0x323AF)))

def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly {', '.join(keys)}")

def _id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("invalid stable id")

def validate_input(doc):
    _object(doc, ("format", "id", "items"), "input")
    if doc["format"] != INPUT_FORMAT:
        raise ValueError("unsupported input format")
    _id(doc["id"])
    items = doc["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a nonempty list")
    seen = set()
    for item in items:
        _object(item, ("id", "text", "audio"), "item")
        _id(item["id"])
        if item["id"] in seen:
            raise ValueError("duplicate item id")
        seen.add(item["id"])
        if not isinstance(item["text"], str) or not item["text"] or any(0xD800 <= ord(c) <= 0xDFFF for c in item["text"]):
            raise ValueError("text must be nonempty Unicode scalar text")
        uri = item["audio"]
        if not isinstance(uri, str) or not uri.startswith(AUDIO):
            raise ValueError("audio must be an inlined FLAC data URI")
        try:
            data = base64.b64decode(uri[len(AUDIO):], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 audio") from exc
        if len(data) < 42 or data[:4] != b"fLaC" or data[4] & 0x7f != 0 or data[5:8] != b"\x00\x00\x22":
            raise ValueError("audio lacks a complete FLAC STREAMINFO header")
        stream = int.from_bytes(data[18:26], "big")
        if (stream >> 44) == 0 or (stream & ((1 << 36) - 1)) == 0:
            raise ValueError("FLAC sample rate and sample count must be nonzero")
    return doc

def validate_result(result, source):
    validate_input(source)
    _object(result, ("format", "id", "note", "items"), "result")
    if result["format"] != RESULT_FORMAT or result["id"] != source["id"]:
        raise ValueError("result format or input id mismatch")
    if not isinstance(result["note"], str) or not isinstance(result["items"], list):
        raise ValueError("invalid result note or items")
    expected = {i["id"]: i for i in source["items"]}
    seen = set()
    for item in result["items"]:
        _object(item, ("id", "marks", "rating", "note"), "result item")
        ident = item["id"]
        if not isinstance(ident, str) or ident not in expected or ident in seen:
            raise ValueError("unknown or duplicate result item")
        seen.add(ident)
        if not isinstance(item["rating"], str) or item["rating"] not in RATINGS or not isinstance(item["note"], str) or not isinstance(item["marks"], list):
            raise ValueError("invalid rating, note or marks")
        count = sum(cjk(c) for c in expected[ident]["text"])
        marks = item["marks"]
        if any(type(n) is not int or n < 0 or n >= count for n in marks) or len(set(marks)) != len(marks):
            raise ValueError("invalid CJK mark index")
    if seen != set(expected):
        raise ValueError("missing result item")
    return result

def generate(source, template=None):
    validate_input(source)
    page = (template if template is not None else TEMPLATE.read_text(encoding="utf-8"))
    if page.count(MARKER) != 1:
        raise ValueError("template must contain exactly one input marker")
    data = json.dumps(source, ensure_ascii=True, separators=(",", ":")).replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    return page.replace(MARKER, '<script id="feedback-input" type="application/json">' + data + '</script>')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("input", type=Path)
    build.add_argument("output", type=Path)
    check = sub.add_parser("check-result")
    check.add_argument("input", type=Path)
    check.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        if args.command == "build":
            page = generate(source)
            with args.output.open("x", encoding="utf-8", newline="\n") as out:
                out.write(page)
        else:
            validate_result(json.loads(args.result.read_text(encoding="utf-8")), source)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"feedback: {exc}\n")

if __name__ == "__main__":
    main()
