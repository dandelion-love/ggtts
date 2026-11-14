#!/usr/bin/env python3
"""A stand-in for an ASR judge worker, for tests: it speaks the worker protocol and hears no audio.

`--model DIR` names a directory holding `answers.json`: `{chunk text: [transcript of its 1st read, 2nd, ...]}`, the
last repeating. A text it does not list is read back as itself, except that `{"first_read_loses": N}` drops N
characters from the first read of every text. `{"never_ready": "why"}` in the file makes it refuse to start with
that reason. Each request is appended to `DIR/requests.jsonl`, so a test can see what was asked.
"""
import argparse
import json
import os
import sys

# The protocol is UTF-8 JSON lines whatever the platform's default: Windows starts Python with its console code
# page on stdin and stdout, which cannot carry Chinese.
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
path = os.path.join(a.model, "answers.json")
answers = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
if "never_ready" in answers:
    print(json.dumps({"error": answers["never_ready"]}), flush=True)
    sys.exit(2)
print(json.dumps({"ready": True}), flush=True)
seen = {}
with open(os.path.join(a.model, "requests.jsonl"), "a", encoding="utf-8") as log:
    for line in sys.stdin:
        req = json.loads(line)
        log.write(json.dumps({"text": req.get("text"), "device": a.device, "audio_exists": os.path.isfile(req["audio"])},
                             ensure_ascii=False) + "\n")
        log.flush()
        text = req.get("text", "")
        n = seen.get(text, 0)
        seen[text] = n + 1
        script = answers.get(text)
        lose = answers.get("first_read_loses", 0)
        if script:
            said = script[min(n, len(script) - 1)]
        elif lose and n == 0:
            # Drop `lose` CJK characters after the second one: a clause lost on the first draw only.
            cjk = [i for i, c in enumerate(text) if "㐀" <= c <= "鿿"]
            gone = set(cjk[2:2 + lose])
            said = "".join(c for i, c in enumerate(text) if i not in gone)
        else:
            said = text
        print(json.dumps({"text": said}, ensure_ascii=False), flush=True)
