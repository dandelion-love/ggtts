#!/usr/bin/env python3
"""The PyTorch side of the Kitten-TTS 2 engine: reference dumps, reference-clip preparation and the S3Gen decode.

This script is test and bridge tooling. It is not part of the Rust build and nothing in ggtts links it. It needs a
Python environment with `torch`, `transformers`, `safetensors` and `kittenml` (Apache-2.0), and it runs on the CPU
only: every GPU is hidden before torch loads.

Subcommands:
  logits     teacher-forced LM forward in float32 on fixed inputs; writes the inputs, the hidden state after every
             layer and the logits, for the ggml graph's parity check.
  tokenize   the Hugging Face tokenizer's ids for some texts, for the Rust tokenizer's check.
  reference  a reference clip and its transcript to the conditioning the engine needs: S3 tokens of the clip,
             the 512-dim speaker embedding and the transcript. A built-in voice can be named instead of a clip.
  decode     S3 codec tokens to a 24 kHz waveform through Kitten's S3Gen (chatterbox-turbo, meanflow, 2 steps),
             conditioned on the reference clip. Writes float32 little-endian samples.

Outputs never go into the repository: weights, dumps and audio from a Kitten model are under the model's licence.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch is imported

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def low_priority(threads: int) -> None:
    """Idle priority and a bounded thread count, so a reference run never competes with anything else."""
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x40)  # IDLE_PRIORITY_CLASS
        else:
            os.nice(19)
    except Exception:
        pass
    import torch

    torch.set_num_threads(threads)


def load_config(repo: Path) -> dict:
    return json.loads((repo / "config.json").read_text(encoding="utf-8"))


def voice_spec(repo: Path, config: dict, name: str) -> dict:
    voices_path = repo / config.get("voices", "voices/voices.json")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    spec = voices[name]
    root = voices_path.parent
    art = np.load(root / spec["artifacts"])
    return {
        "transcript": spec["transcript"],
        "reference_wav": str(root / spec["reference"]),
        "reference_tokens": [int(t) for t in art["reference_tokens"].tolist()],
        "embedding": [float(x) for x in np.asarray(art["embedding"], np.float32).ravel()],
    }


def tokenizer_for(repo: Path, config: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(repo / config.get("lm_dir", "lm")))


def build_prompt(config: dict, tok, ref: dict, text: str) -> list[int]:
    from kittenml.kittentts2.prompt import build_generation_prompt, build_reference_prefix
    from kittenml.kittentts2.tokens import TokenMap

    tm = TokenMap.from_config(config)
    ref_ids = tok.encode(ref["transcript"].strip(), add_special_tokens=False)
    prefix = build_reference_prefix(tm, ref_ids, ref["reference_tokens"])
    return build_generation_prompt(tm, tok.encode(text, add_special_tokens=False), reference_prefix=prefix)


def cmd_logits(a) -> int:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    low_priority(a.threads)
    repo = Path(a.repo)
    config = load_config(repo)
    tok = tokenizer_for(repo, config)
    ref = voice_spec(repo, config, a.voice)
    prompt = build_prompt(config, tok, ref, a.text)
    base = config["token_map"]["audio_id_base"]
    forced = [base + t for t in ref["reference_tokens"][: a.forced]]
    ids = prompt + forced

    state = load_file(a.weights)
    lm_config = AutoConfig.from_pretrained(str(repo / config.get("lm_dir", "lm")))
    model = AutoModelForCausalLM.from_config(lm_config, dtype=torch.float32, attn_implementation="eager")
    spk = {k[len("spk_proj."):]: v.float() for k, v in state.items() if k.startswith("spk_proj.")}
    lm_state = {k: v.float() for k, v in state.items() if not k.startswith("spk_proj.")}
    missing, _ = model.load_state_dict(lm_state, strict=False)
    model.tie_weights()
    if [m for m in missing if m != "lm_head.weight"]:
        raise SystemExit(f"missing weights: {missing[:5]}")
    model.eval()
    head = torch.nn.Sequential(torch.nn.Linear(512, lm_config.hidden_size), torch.nn.LayerNorm(lm_config.hidden_size))
    head.load_state_dict(spk, strict=True)
    head.eval()

    emb_raw = torch.tensor(ref["embedding"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        proj = head(emb_raw)  # [1, hidden]
        x = model.get_input_embeddings()(torch.tensor([ids]))
        x = torch.cat([proj.unsqueeze(1), x[:, 1:, :]], dim=1)
        out = model(inputs_embeds=x, output_hidden_states=True, use_cache=False)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hs = torch.stack([h[0] for h in out.hidden_states]).float().numpy()  # [n_layers + 1, T, hidden]
    logits = out.logits[0, -a.last :, :].float().numpy()
    hs.astype("<f4").tofile(out_dir / "hidden.f32")
    logits.astype("<f4").tofile(out_dir / "logits.f32")
    np.asarray(ref["embedding"], "<f4").tofile(out_dir / "speaker_raw.f32")
    proj[0].float().numpy().astype("<f4").tofile(out_dir / "speaker_proj.f32")
    meta = {
        "weights": str(a.weights),
        "text": a.text,
        "voice": a.voice,
        "ids": ids,
        "prompt_len": len(prompt),
        "forced": a.forced,
        "hidden_shape": list(hs.shape),
        "logits_shape": list(logits.shape),
        "logits_last": a.last,
        "dtype": "float32",
        "torch": torch.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "ids"}))
    return 0


def cmd_tokenize(a) -> int:
    repo = Path(a.repo)
    config = load_config(repo)
    tok = tokenizer_for(repo, config)
    texts = json.loads(Path(a.texts).read_text(encoding="utf-8"))
    out = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_reference(a) -> int:
    low_priority(a.threads)
    repo = Path(a.repo)
    config = load_config(repo)
    if a.voice:
        ref = voice_spec(repo, config, a.voice)
    else:
        from kittenml.kittentts2.speaker import SpeakerEncoder
        from kittenml.kittentts2.vocoder import S3Codec

        if not (a.transcript and a.transcript.strip()):
            raise SystemExit("a reference clip needs its transcript (--transcript)")
        codec = S3Codec(device="cpu")
        enc = SpeakerEncoder(repo / config.get("speaker_embedding", "speaker/model.safetensors"), device="cpu")
        ref = {
            "transcript": a.transcript.strip(),
            "reference_wav": str(Path(a.wav).resolve()),
            "reference_tokens": [int(t) for t in codec.encode_reference(a.wav)],
            "embedding": [float(x) for x in enc(a.wav)],
        }
    Path(a.out).write_text(json.dumps(ref), encoding="utf-8")
    print(json.dumps({"transcript": ref["transcript"], "reference_tokens": len(ref["reference_tokens"])}))
    return 0


def cmd_decode(a) -> int:
    import torch

    low_priority(a.threads)
    from kittenml.kittentts2.vocoder import S3Codec

    ref = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    tokens = json.loads(Path(a.tokens).read_text(encoding="utf-8"))
    torch.manual_seed(a.seed)
    codec = S3Codec(device="cpu")
    cond = codec.reference_conditioning(ref["reference_wav"])
    wav = codec.decode(cond, tokens)
    if wav is None:
        raise SystemExit("no valid audio tokens")
    np.asarray(wav, "<f4").tofile(a.out)
    print(json.dumps({"samples": int(len(wav)), "sample_rate": codec.sample_rate}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("logits")
    p.add_argument("--repo", required=True, help="the Kitten-TTS 2 model directory (config, tokenizer, voices)")
    p.add_argument("--weights", required=True, help="an LM safetensors in Kitten's key layout")
    p.add_argument("--voice", default="Bruno")
    p.add_argument("--text", default="The little lighthouse kept its lamp burning all through the stormy night.")
    p.add_argument("--forced", type=int, default=24, help="reference audio tokens appended, teacher-forced")
    p.add_argument("--last", type=int, default=32, help="positions whose logits are written")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("tokenize")
    p.add_argument("--repo", required=True)
    p.add_argument("--texts", required=True, help="a JSON list of strings")
    p.add_argument("--out", required=True)
    p = sub.add_parser("reference")
    p.add_argument("--repo", required=True)
    p.add_argument("--voice", help="a built-in voice instead of a clip")
    p.add_argument("--wav")
    p.add_argument("--transcript")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("decode")
    p.add_argument("--reference", required=True, help="the JSON `reference` wrote")
    p.add_argument("--tokens", required=True, help="a JSON list of S3 token ids (0..6560)")
    p.add_argument("--out", required=True, help="float32 little-endian samples at 24 kHz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    return {"logits": cmd_logits, "tokenize": cmd_tokenize, "reference": cmd_reference, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
