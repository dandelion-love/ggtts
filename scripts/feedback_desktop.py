#!/usr/bin/env python3
"""Open a verified offline feedback page and save validated results beside its audio."""
import argparse
import base64
import hashlib
import json
import re
import webbrowser
from pathlib import Path

try:
    from scripts.feedback_page import generate, input_digest, publish, validate_result
except ModuleNotFoundError:
    from feedback_page import generate, input_digest, publish, validate_result

SCRIPT = re.compile(r'<script id="feedback-input" type="application/json">(.*?)</script>', re.DOTALL)

def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify(page: Path, audio: Path):
    """Check exact generated page and its companion against the selected audio."""
    raw = page.read_bytes()
    try:
        text = raw.decode("utf-8")
        found = SCRIPT.findall(text)
        if len(found) != 1:
            raise ValueError("page must contain exactly one feedback input")
        source = json.loads(found[0])
        expected = generate(source).encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("invalid feedback page") from exc
    if raw != expected:
        raise ValueError("page is not the trusted generated template")
    sidecar = Path(str(page) + ".provenance.json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("invalid companion")
        render = meta.get("render")
        audio_meta = render.get("audio") if isinstance(render, dict) else None
        actual = _hash_file(audio)
        if (meta.get("format") != "ggtts-feedback-provenance/1"
                or meta.get("id") != source["id"]
                or meta.get("input_sha256") != input_digest(source)
                or meta.get("output_sha256") != actual
                or not isinstance(audio_meta, dict)
                or audio_meta.get("sha256") != actual):
            raise ValueError("companion, input or audio identity does not match")
    except (json.JSONDecodeError, UnicodeError, TypeError, KeyError) as exc:
        raise ValueError("invalid feedback companion") from exc
    chunks = meta.get("chunks")
    rate = meta.get("sample_rate")
    total = meta.get("output_samples")
    if (not isinstance(chunks, list) or len(chunks) != len(source["items"])
            or type(rate) is not int or rate <= 0 or type(total) is not int or total <= 0):
        raise ValueError("incomplete chunk provenance")
    def flac_info(header):
        if len(header) < 42 or header[:4] != b"fLaC" or header[4] & 0x7f != 0 or header[5:8] != b"\x00\x00\x22":
            raise ValueError("invalid FLAC header")
        packed = int.from_bytes(header[18:26], "big")
        return packed >> 44, packed & ((1 << 36) - 1)
    with audio.open("rb") as f:
        output_info = flac_info(f.read(42))
    if output_info != (rate, total):
        raise ValueError("output audio header conflicts with companion")
    end = 0
    for i, (item, record) in enumerate(zip(source["items"], chunks)):
        if not isinstance(record, dict):
            raise ValueError("invalid chunk provenance")
        start, stop = record.get("start"), record.get("end")
        if (record.get("id") != item["id"] or type(record.get("chunk")) is not int or record["chunk"] != i
                or type(start) is not int or type(stop) is not int
                or start < end or start >= stop or stop > total
                or (i == 0 and start != 0)
                or type(record.get("draw")) is not int or record["draw"] < 0
                or type(record.get("seed")) is not int or record["seed"] < 0
                or record.get("stop") not in ("eos", "cap")):
            raise ValueError("invalid chunk identity or span")
        clip = base64.b64decode(item["audio"].split(",", 1)[1], validate=True)
        if (record.get("text_sha256") != hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                or record.get("audio_sha256") != hashlib.sha256(clip).hexdigest()
                or flac_info(clip) != (rate, stop - start)):
            raise ValueError("chunk content conflicts with companion")
        end = stop
    if end != total:
        raise ValueError("unmapped trailing output audio")
    return source

def open_page(page: Path, audio: Path):
    verify(page, audio)
    if not webbrowser.open(page.resolve().as_uri()):
        raise OSError("no browser accepted the feedback page")

def save(page: Path, audio: Path, result: Path):
    source = verify(page, audio)
    feedback = json.loads(result.read_text(encoding="utf-8"))
    validate_result(feedback, source)
    destination = audio.with_name(audio.name + ".feedback.json")
    content = json.dumps(feedback, ensure_ascii=True, indent=2) + "\n"
    publish(destination, content)
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    op = sub.add_parser("open")
    op.add_argument("page", type=Path)
    op.add_argument("--audio", required=True, type=Path)
    sp = sub.add_parser("save")
    sp.add_argument("page", type=Path)
    sp.add_argument("result", type=Path)
    sp.add_argument("--audio", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "open":
            open_page(args.page, args.audio)
            print("Opened verified offline feedback page.")
        else:
            print(f"Saved validated feedback to {save(args.page, args.audio, args.result)}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"feedback desktop: {exc}\n")

if __name__ == "__main__":
    main()
