#!/usr/bin/env python3
"""Optional offline Qwen3-ASR instrument for already completed audio."""
import contextlib
import hashlib
import importlib.metadata
import io
import os
import sys
import unicodedata
from pathlib import Path

try:
    import ggtts_instrument  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'python'))

from ggtts_instrument import read_audio, run
from ggtts_instrument.audio import sha256_of
from ggtts_instrument.device import select_device, verify_model_device

NAME, VERSION = 'asr-qwen', '0.2.0'


def normalize(text):
    return ''.join(ch for ch in unicodedata.normalize('NFKC', text).casefold() if ch.isalnum())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def model_identity(model):
    """Hash every local snapshot file, including filenames and weight shards."""
    files = sorted((p for p in model.rglob('*') if p.is_file()), key=lambda p: p.relative_to(model).as_posix())
    if not (model / 'config.json').is_file() or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('local ASR model is incomplete')
    h = hashlib.sha256()
    for path in files:
        name = path.relative_to(model).as_posix().encode('utf-8')
        h.update(len(name).to_bytes(8, 'big'))
        h.update(name)
        h.update(bytes.fromhex(sha256_of(str(path))))
    return h.hexdigest()


def transcribe(model_dir, audio, device='cpu'):
    # Suppress third-party output and error text: either can contain private speech.
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torch
            from qwen_asr import Qwen3ASRModel
            torch.set_num_threads(4)
            model = Qwen3ASRModel.from_pretrained(
                str(model_dir), dtype=torch.float32, device_map=device, max_inference_batch_size=1,
                local_files_only=True,
            )
            verify_model_device(model, device)
            result = model.transcribe([str(audio)], language='Cantonese')
        if not isinstance(result, list) or not result or not isinstance(getattr(result[0], 'text', None), str):
            raise ValueError('malformed result')
        if not normalize(result[0].text):
            raise ValueError('empty result')
        return result[0].text
    except Exception:
        raise RuntimeError('ASR unavailable or returned invalid output') from None


def measure(ctx, env):
    model_arg = ctx.config.pop('model_dir', '')
    reference_arg = ctx.config.pop('reference_file', '')
    device_arg = ctx.config.pop('device', None)
    # The SDK's implicit render id is the audio filename stem, which may be private.
    if 'render_default' in ctx.config:
        ctx.render = 'render'
        ctx.config['render_default'] = 'render'
    unexpected = set(ctx.config) - {'render_default', 'run_default'}
    if unexpected:
        ctx.config.clear()
        raise ValueError('unsupported instrument configuration')
    if not ctx.audio or not model_arg or not reference_arg:
        raise ValueError('completed audio, local model and reference are required')
    model, ref = Path(model_arg), Path(reference_arg)
    if not model.is_dir() or not ref.is_file():
        raise ValueError('local model directory and reference file are required')
    try:
        audio = read_audio(ctx.audio)
        if not audio.samples:
            raise ValueError('empty audio')
        reference = normalize(ref.read_text(encoding='utf-8'))
        if not reference:
            raise ValueError('empty reference')
        model_hash = model_identity(model)
        audio_hash, reference_hash = sha256_of(ctx.audio), sha256_of(str(ref))
    except Exception:
        raise ValueError('audio, reference or local model could not be read') from None
    env.doc['provenance']['model_sha256'] = model_hash
    env.doc['provenance']['inputs'].extend([
        {'path': 'completed-audio', 'sha256': audio_hash},
        {'path': 'reference-text', 'sha256': reference_hash},
    ])
    ctx.config['requested_device'] = device_arg if device_arg is not None else os.environ.get('GGTTS_INSTRUMENT_DEVICE', 'cpu')
    requested, device = select_device(device_arg)
    ctx.config.update({'normalization': 'unicode-nfkc-casefold-alphanumeric-v1', 'language': 'Cantonese',
                       'requested_device': requested, 'threads': 4})
    try:
        ctx.config['qwen_asr_version'] = importlib.metadata.version('qwen-asr')
    except importlib.metadata.PackageNotFoundError:
        ctx.config['qwen_asr_version'] = 'unavailable'
    hyp = normalize(transcribe(model, ctx.audio, device))
    ctx.config['device'] = device
    if not hyp:
        raise ValueError('empty normalized ASR result')
    env.metric('character_error_rate', distance(reference, hyp) / len(reference), '1', 'neutral', render=ctx.render)


if __name__ == '__main__':
    sys.exit(run(measure, NAME, VERSION, hooks=['artifact.audio'], audio_state='unknown'))
