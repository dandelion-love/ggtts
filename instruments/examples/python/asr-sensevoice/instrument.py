#!/usr/bin/env python3
"""Optional offline SenseVoice instrument on completed audio; no quality verdict."""
import contextlib
import importlib.metadata
import io
import os
import re
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

NAME, VERSION = 'asr-sensevoice', '0.2.1'
_TAG = re.compile(r'<\|[^|]*\|>')


def normalize(text):
    text = unicodedata.normalize('NFKC', _TAG.sub('', text)).casefold()
    return ''.join(ch for ch in text if ch.isalnum())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def transcribe(model_dir, audio, device='cpu'):
    # Keep third-party model output and exceptions out of logs: it can include private speech.
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['MODELSCOPE_OFFLINE'] = '1'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from funasr import AutoModel
            model = AutoModel(model=str(model_dir), device=device, disable_update=True, ncpu=4)
            verify_model_device(model, device)
            result = model.generate(input=str(audio), language='yue', use_itn=False)
        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise ValueError('malformed result')
        text = result[0].get('text')
        if not isinstance(text, str) or not normalize(text):
            raise ValueError('empty result')
        return text
    except Exception:
        raise RuntimeError('ASR unavailable or returned invalid output') from None


def measure(ctx, env):
    # The SDK normally records all config values. Paths may be private, so remove them
    # before even the failure path writes an envelope; the files are identified by hash.
    model_arg = ctx.config.pop('model_dir', '')
    ref_arg = ctx.config.pop('reference_file', '')
    device_arg = ctx.config.pop('device', None)
    if 'render_default' in ctx.config:
        ctx.render = 'render'
        ctx.config['render_default'] = 'render'
    unexpected = set(ctx.config) - {'render_default', 'run_default'}
    if unexpected:
        ctx.config.clear()
        raise ValueError('unsupported instrument configuration')
    if not ctx.audio:
        raise ValueError('completed audio is required')
    model = Path(model_arg)
    ref = Path(ref_arg)
    if not model.is_dir() or not ref.is_file():
        raise ValueError('local model directory and reference file are required')
    # The official FunASR snapshot maps these files via configuration.json.
    # Some local variants also contain tokens.json; preserve its hash when present.
    model_files = ('model.pt', 'config.yaml', 'configuration.json', 'am.mvn', 'chn_jpn_yue_eng_ko_spectok.bpe.model')
    if any(not (model / name).is_file() for name in model_files):
        raise ValueError('local model is incomplete')
    if (model / 'tokens.json').is_file():
        model_files += ('tokens.json',)
    audio = read_audio(ctx.audio)
    if not audio.samples:
        raise ValueError('empty audio')
    reference = normalize(ref.read_text(encoding='utf-8'))
    if not reference:
        raise ValueError('empty reference')
    # SDK input() stores basenames, which may themselves contain private text.
    env.doc['provenance']['inputs'].extend([
        {'path': 'audio', 'sha256': audio.sha256},
        {'path': 'reference', 'sha256': sha256_of(ref)},
    ])
    for name in model_files:
        env.input(str(model / name))
    ctx.config['normalization'] = 'unicode-nfkc-casefold-alphanumeric-v1'
    ctx.config['language'] = 'yue'
    ctx.config['requested_device'] = device_arg if device_arg is not None else os.environ.get('GGTTS_INSTRUMENT_DEVICE', 'cpu')
    requested, device = select_device(device_arg)
    ctx.config['requested_device'] = requested
    ctx.config['threads'] = 4
    ctx.config['itn'] = False
    try:
        ctx.config['funasr_version'] = importlib.metadata.version('funasr')
    except importlib.metadata.PackageNotFoundError:
        ctx.config['funasr_version'] = 'unavailable'
    hyp = normalize(transcribe(model, ctx.audio, device))
    ctx.config['device'] = device
    if not hyp:
        raise ValueError('empty normalized ASR result')
    env.metric('character_error_rate', distance(reference, hyp) / len(reference), '1', 'neutral', render=ctx.render)


if __name__ == '__main__':
    sys.exit(run(measure, NAME, VERSION, hooks=['artifact.audio'], audio_state='unknown'))
