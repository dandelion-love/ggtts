#!/usr/bin/env python3
"""Record the PyTorch reference for steering: the talker's final hidden state with and without a forward hook that
adds a vector at the output of one layer.

    python3 scripts/steer_reference.py TALKER.gguf OUT_DIR [--layer 0] [--scale 0.5] [--tokens 5] [--seed 1]

The reference is built from the same bytes the engine reads. Every talker layer and the output norm are read from the
GGUF and dequantised to f32 (the `gguf` package), and loaded into `transformers`' Qwen3 decoder (`Qwen3Model`), which
is the layer the Qwen3-TTS talker uses. A seeded random prefill of `--tokens` rows goes in as `inputs_embeds` at
positions 0 onward. With every position equal across the three M-RoPE sections, the talker's multimodal rope is the
plain rope Qwen3 applies, so no rope change is needed.

The steering vector is seeded random, scaled to the standard deviation of that layer's unsteered output, and applied
by `register_forward_hook` on `layers[layer]`: the hook returns the block's output plus `scale * vector`, in f32. That
is what the engine's add node does.

OUT_DIR gets three files:
  * `steer.safetensors`: the steering file, one tensor `talker.layer.<layer>`;
  * `reference.safetensors`: `input` (tokens x hidden), `hidden.plain` and `hidden.steered` (the final hidden state
    after the output norm, every row);
  * `reference.json`: the talker's sha256, the layer, the scale, the seed and the versions used.

CPU only. Needs torch, transformers, safetensors and gguf.
"""
import argparse
import hashlib
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import safetensors  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen3Config, Qwen3Model  # noqa: E402

# GGUF tensor name (after `talker.blk.<i>.`) to the Qwen3 decoder layer's parameter.
LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def field(reader, *keys):
    for k in keys:
        f = reader.fields.get(k)
        if f is not None:
            return f.contents()
    raise SystemExit(f"the GGUF has none of {keys}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("talker")
    ap.add_argument("out_dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    r = GGUFReader(a.talker)
    h = int(field(r, "qwen3-tts.talker.embedding_length"))
    n_layers = int(field(r, "qwen3-tts.talker.block_count"))
    if not 0 <= a.layer < n_layers:
        raise SystemExit(f"--layer {a.layer} is out of range: the talker has layers 0 to {n_layers - 1}")
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=h,
        intermediate_size=int(field(r, "qwen3-tts.talker.feed_forward_length")),
        num_hidden_layers=n_layers,
        num_attention_heads=int(field(r, "qwen3-tts.talker.attention.head_count")),
        num_key_value_heads=int(field(r, "qwen3-tts.talker.attention.head_count_kv")),
        head_dim=int(field(r, "qwen3-tts.talker.attention.key_length")),
        rms_norm_eps=float(field(r, "qwen3-tts.talker.attention.layer_norm_rms_epsilon")),
        rope_theta=float(field(r, "qwen3-tts.talker.rope.freq_base")),
        max_position_embeddings=4096,
        attention_bias=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model = Qwen3Model(config).to(torch.float32).eval()

    state = {}
    for t in r.tensors:
        name = t.name
        if name == "talker.output_norm.weight":
            key = "norm.weight"
        elif name.startswith("talker.blk."):
            i, rest = name[len("talker.blk."):].split(".", 1)
            if rest not in LAYER_NAMES:
                continue
            key = f"layers.{i}.{LAYER_NAMES[rest]}"
        else:
            continue
        w = dequantize(t.data, t.tensor_type).astype(np.float32)
        # GGUF lists ne0 (the input dimension) first; the data is row-major (out, in), as torch holds it.
        state[key] = torch.from_numpy(w.reshape(list(reversed([int(x) for x in t.shape]))).copy())
    want = {k for k in model.state_dict() if not k.startswith("embed_tokens.")}
    missing = sorted(want - state.keys())
    if missing:
        raise SystemExit(f"the GGUF lacks {missing[:4]}")
    model.load_state_dict(state, strict=False)

    g = torch.Generator().manual_seed(a.seed)
    x = torch.randn(1, a.tokens, h, generator=g, dtype=torch.float32) * 0.05

    with torch.no_grad():
        # The layer's unsteered output sets the vector's size.
        seen = {}

        def capture(_module, _inputs, output):
            seen["out"] = output[0] if isinstance(output, tuple) else output

        handle = model.layers[a.layer].register_forward_hook(capture)
        plain = model(inputs_embeds=x).last_hidden_state[0].clone()
        handle.remove()
        size = float(seen["out"].std())
        vector = (torch.randn(h, generator=g, dtype=torch.float32) * size).contiguous()

        def steer(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + a.scale * vector,) + tuple(output[1:])
            return output + a.scale * vector

        handle = model.layers[a.layer].register_forward_hook(steer)
        steered = model(inputs_embeds=x).last_hidden_state[0].clone()
        handle.remove()

    os.makedirs(a.out_dir, exist_ok=True)
    save_file({f"talker.layer.{a.layer}": vector}, os.path.join(a.out_dir, "steer.safetensors"))
    save_file(
        {"input": x[0].contiguous(), "hidden.plain": plain.contiguous(), "hidden.steered": steered.contiguous()},
        os.path.join(a.out_dir, "reference.safetensors"),
    )
    meta = {
        "talker_sha256": sha256(a.talker),
        "layer": a.layer,
        "scale": a.scale,
        "tokens": a.tokens,
        "seed": a.seed,
        "input_std": 0.05,
        "vector_std": size,
        "steering_effect_max_abs": float((steered - plain).abs().max()),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "gguf": getattr(gguf, "__version__", "unknown"),
        },
    }
    with open(os.path.join(a.out_dir, "reference.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
