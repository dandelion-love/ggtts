import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('asr_sensevoice', Path(__file__).with_name('instrument.py'))
instrument = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrument)


class FakeEnvelope:
    def __init__(self):
        self.inputs, self.metrics = [], []

    def input(self, path):
        self.inputs.append(path)
        return self

    def metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))


class TestInstrument(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        for name in ('model.pt', 'config.yaml', 'am.mvn', 'tokens.json', 'chn_jpn_yue_eng_ko_spectok.bpe.model'):
            (self.model / name).write_bytes(b'synthetic')
        self.reference = self.root / 'ref.txt'
        self.reference.write_text('abcdef', encoding='utf-8')
        self.audio = self.root / 'audio.wav'
        with wave.open(str(self.audio), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000); f.writeframes(b'\x00\x00' * 24)
    def ctx(self):
        return SimpleNamespace(audio=str(self.audio), render='test', config={'model_dir': str(self.model),
                                                                            'reference_file': str(self.reference)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_noop_and_planted_omission(self):
        with patch.object(instrument, 'transcribe', return_value='abcdef'):
            one, two = FakeEnvelope(), FakeEnvelope()
            instrument.measure(self.ctx(), one); instrument.measure(self.ctx(), two)
        self.assertEqual(one.metrics, two.metrics)
        self.assertEqual(one.metrics[0][0][1], 0)
        with patch.object(instrument, 'transcribe', return_value='abc'):
            bad = FakeEnvelope(); instrument.measure(self.ctx(), bad)
        self.assertEqual(bad.metrics[0][0][1], 0.5)
        self.assertEqual(len(one.inputs), 7)

    def test_missing_model_audio_or_reference_and_empty_asr_fail(self):
        with patch.object(instrument, 'transcribe', return_value=''):
            with self.assertRaises(ValueError): instrument.measure(self.ctx(), FakeEnvelope())
        self.reference.unlink()
        with self.assertRaises(ValueError): instrument.measure(self.ctx(), FakeEnvelope())
        self.reference.write_text('abc', encoding='utf-8')
        (self.model / 'model.pt').unlink()
        with self.assertRaises(ValueError): instrument.measure(self.ctx(), FakeEnvelope())
        (self.model / 'model.pt').write_bytes(b'x')
        self.audio.unlink()
        with self.assertRaises((OSError, ValueError)): instrument.measure(self.ctx(), FakeEnvelope())
        self.audio.write_bytes(b'bad')
        with self.assertRaises((OSError, ValueError)): instrument.measure(self.ctx(), FakeEnvelope())

    def test_envelope_excludes_reference_and_private_paths(self):
        out = self.root / 'result.json'
        args = ['--audio', str(self.audio), '--envelope', str(out), '--model-dir', str(self.model),
                '--reference-file', str(self.reference)]
        with patch.object(instrument, 'transcribe', return_value='abcdef'):
            self.assertEqual(instrument.run(instrument.measure, instrument.NAME, instrument.VERSION,
                                            ['artifact.audio'], audio_state='unknown', argv=args), 0)
        content = out.read_text(encoding='utf-8')
        data = json.loads(content)
        self.assertNotIn('abcdef', content)
        self.assertNotIn(str(self.root), content)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(data['provenance']['audio'], 'unknown')

    def test_invalid_recognizer_result_fails_closed(self):
        for response in ([], [{}], [{'text': ''}], [{'text': '<|yue|>'}]):
            recognizer = SimpleNamespace(generate=lambda **kwargs: response)
            fake = SimpleNamespace(AutoModel=lambda **kwargs: recognizer)
            with patch.dict(sys.modules, {'funasr': fake}):
                with self.assertRaisesRegex(RuntimeError, 'invalid output'):
                    instrument.transcribe(self.model, self.audio)


if __name__ == '__main__':
    unittest.main()
