import importlib.util
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('asr_qwen', Path(__file__).with_name('instrument.py'))
instrument = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrument)


class FakeEnvelope:
    def __init__(self):
        self.doc = {'provenance': {'model_sha256': None, 'inputs': []}}
        self.metrics = []

    def metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))


class TestInstrument(unittest.TestCase):
    def test_unavailable_cuda_records_request_without_claiming_use(self):
        out = self.root / 'unavailable.json'
        args = ['--audio', str(self.audio), '--envelope', str(out), '--model-dir', str(self.model),
                '--reference-file', str(self.reference)]
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.dict(os.environ, {'GGTTS_INSTRUMENT_DEVICE': 'cuda'}):
            code = instrument.run(instrument.measure, instrument.NAME, instrument.VERSION,
                                  ['artifact.audio'], audio_state='unknown', argv=args)
        config = json.loads(out.read_text(encoding='utf-8'))['provenance']['config']
        self.assertEqual(code, 1)
        self.assertEqual(config['requested_device'], 'cuda')
        self.assertNotIn('device', config)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / 'private-model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}', encoding='utf-8')
        (self.model / 'model.safetensors').write_bytes(b'synthetic weights')
        self.reference = self.root / 'private-reference.txt'
        self.reference.write_text('abcdef', encoding='utf-8')
        self.audio = self.root / 'private-audio.wav'
        with wave.open(str(self.audio), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(b'\x00\x00' * 24)

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self):
        return SimpleNamespace(audio=str(self.audio), render='test', config={
            'model_dir': str(self.model), 'reference_file': str(self.reference)})

    def test_noop_and_planted_omission(self):
        with patch.object(instrument, 'transcribe', return_value='abcdef'):
            a, b = FakeEnvelope(), FakeEnvelope()
            instrument.measure(self.ctx(), a)
            instrument.measure(self.ctx(), b)
        self.assertEqual(a.metrics, b.metrics)
        self.assertEqual(a.metrics[0][0][1], 0)
        with patch.object(instrument, 'transcribe', return_value='abc'):
            planted = FakeEnvelope()
            instrument.measure(self.ctx(), planted)
        self.assertEqual(planted.metrics[0][0][1], 0.5)
        self.assertEqual(a.doc['provenance']['model_sha256'], b.doc['provenance']['model_sha256'])
        (self.model / 'model.safetensors').write_bytes(b'changed')
        self.assertNotEqual(a.doc['provenance']['model_sha256'], instrument.model_identity(self.model))

    def test_missing_files_and_empty_hypothesis_fail(self):
        with patch.object(instrument, 'transcribe', return_value=''):
            with self.assertRaises(ValueError):
                instrument.measure(self.ctx(), FakeEnvelope())
        self.reference.unlink()
        with self.assertRaises(ValueError):
            instrument.measure(self.ctx(), FakeEnvelope())
        self.reference.write_text('abc', encoding='utf-8')
        (self.model / 'model.safetensors').unlink()
        with self.assertRaises(ValueError):
            instrument.measure(self.ctx(), FakeEnvelope())
        (self.model / 'model.safetensors').write_bytes(b'x')
        self.audio.unlink()
        with self.assertRaises(ValueError):
            instrument.measure(self.ctx(), FakeEnvelope())
        self.audio.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            instrument.measure(self.ctx(), FakeEnvelope())

    def test_real_envelope_hides_text_and_paths_on_success_and_failure(self):
        for output in ('abcdef', ''):
            out = self.root / ('ok.json' if output else 'bad.json')
            args = ['--audio', str(self.audio), '--envelope', str(out), '--model-dir', str(self.model),
                    '--reference-file', str(self.reference)]
            with patch.object(instrument, 'transcribe', return_value=output):
                code = instrument.run(instrument.measure, instrument.NAME, instrument.VERSION,
                                      ['artifact.audio'], audio_state='unknown', argv=args)
            text = out.read_text(encoding='utf-8')
            doc = json.loads(text)
            self.assertEqual(code, 0 if output else 1)
            self.assertEqual(doc['status'], 'ok' if output else 'failed')
            self.assertNotIn('abcdef', text)
            self.assertNotIn(str(self.root), text)
            self.assertNotIn('private-', text)

    def test_recognizer_config_is_offline_cpu_and_empty_fails(self):
        calls = {}

        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(transcribe=lambda audio, language: [SimpleNamespace(text='abc')])

        torch = SimpleNamespace(float32='f32', set_num_threads=lambda n: calls.update(threads=n))
        with patch.dict(sys.modules, {'torch': torch, 'qwen_asr': SimpleNamespace(Qwen3ASRModel=FakeModel)}):
            self.assertEqual(instrument.transcribe(self.model, self.audio), 'abc')
        self.assertEqual(calls['device_map'], 'cpu')
        self.assertTrue(calls['local_files_only'])
        self.assertEqual(calls['threads'], 4)

    def test_malformed_recognizer_output_fails_closed(self):
        torch = SimpleNamespace(float32='f32', set_num_threads=lambda n: None)
        for result in ([], [{}], [SimpleNamespace(text=None)], [SimpleNamespace(text=12)]):
            model = SimpleNamespace(transcribe=lambda *args, **kwargs: result)
            factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
            with patch.dict(sys.modules, {'torch': torch, 'qwen_asr': SimpleNamespace(Qwen3ASRModel=factory)}):
                with self.assertRaisesRegex(RuntimeError, 'invalid output'):
                    instrument.transcribe(self.model, self.audio)


if __name__ == '__main__':
    unittest.main()
