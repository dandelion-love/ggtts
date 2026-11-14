#!/usr/bin/env python3
"""asr-qwen as a render's judge: one process per render, the model loaded once, one draw read back per request.

ggtts starts it with `--model DIR --device cpu|cuda` (`--judge asr-qwen,model=DIR`) and speaks JSON lines:

* first line out: `{"ready": true}` once the model is loaded, or `{"error": "..."}` and exit;
* then, for each line in, `{"audio": PATH, "text": TEXT}`: one line out, `{"text": TRANSCRIPT}` or
  `{"error": "..."}`. `text` is what the draw was asked to say; this worker does not use it.

End of input ends the process. Nothing is written to stdout but those lines: the model's own output is suppressed,
since it can contain the speech it heard. The model is a local Qwen3-ASR snapshot, loaded offline, as the `asr-qwen`
instrument loads it (see its README for where to obtain one and what to record about it).
"""
import argparse
import contextlib
import io
import json
import os
import sys


def say(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load(model_dir, device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import torch
        from qwen_asr import Qwen3ASRModel

        torch.set_num_threads(4)
        return Qwen3ASRModel.from_pretrained(
            str(model_dir), dtype=torch.float32, device_map=device, max_inference_batch_size=1,
            local_files_only=True,
        )


def read(model, audio):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = model.transcribe([str(audio)], language="Cantonese")
    if not isinstance(result, list) or not result or not isinstance(getattr(result[0], "text", None), str):
        raise ValueError("malformed result")
    return result[0].text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    if not os.path.isfile(os.path.join(a.model, "config.json")):
        say({"error": "the model directory has no config.json: give a local Qwen3-ASR snapshot"})
        return 2
    try:
        model = load(a.model, a.device)
    except Exception as e:  # noqa: BLE001 - the reason goes back to ggtts, not the speech
        say({"error": f"the model did not load ({type(e).__name__})"})
        return 2
    say({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            say({"text": read(model, json.loads(line)["audio"])})
        except Exception as e:  # noqa: BLE001
            say({"error": f"the draw was not read ({type(e).__name__})"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
