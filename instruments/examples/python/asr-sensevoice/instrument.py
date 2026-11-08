#!/usr/bin/env python3
"""Optional CPU SenseVoice instrument on completed audio; no quality verdict."""
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

NAME, VERSION = 'asr-sensevoice', '0.1.0'
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


def transcribe(model_dir, audio):
    # Keep third-party model output and exceptions out of logs: it can include private speech.
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['MODELSCOPE_OFFLINE'] = '1'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from funasr import AutoModel
            model = AutoModel(model=str(model_dir), device='cpu', disable_update=True, ncpu=4)
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
    # SenseVoice requires these files; never download a model implicitly.
    model_files = ('model.pt', 'config.yaml', 'am.mvn', 'tokens.json', 'chn_jpn_yue_eng_ko_spectok.bpe.model')
    if any(not (model / name).is_file() for name in model_files):
        raise ValueError('local model is incomplete')
    audio = read_audio(ctx.audio)
    if not audio.samples:
        raise ValueError('empty audio')
    reference = normalize(ref.read_text(encoding='utf-8'))
    if not reference:
        raise ValueError('empty reference')
    env.input(ctx.audio).input(str(ref))
    for name in model_files:
        env.input(str(model / name))
    ctx.config['normalization'] = 'unicode-nfkc-casefold-alphanumeric-v1'
    ctx.config['language'] = 'yue'
    ctx.config['device'] = 'cpu'
    ctx.config['threads'] = 4
    ctx.config['itn'] = False
    try:
        ctx.config['funasr_version'] = importlib.metadata.version('funasr')
    except importlib.metadata.PackageNotFoundError:
        ctx.config['funasr_version'] = 'unavailable'
    hyp = normalize(transcribe(model, ctx.audio))
    if not hyp:
        raise ValueError('empty normalized ASR result')
    env.metric('character_error_rate', distance(reference, hyp) / len(reference), '1', 'neutral', render=ctx.render)


if __name__ == '__main__':
    sys.exit(run(measure, NAME, VERSION, hooks=['artifact.audio'], audio_state='unknown'))
