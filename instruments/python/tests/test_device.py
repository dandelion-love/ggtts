import os
import sys
import types
import unittest
from unittest.mock import patch

from ggtts_instrument.device import select_device, verify_model_device


class DeviceTest(unittest.TestCase):
    def test_precedence_and_auto(self):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {"torch": fake}), patch.dict(os.environ, {"GGTTS_INSTRUMENT_DEVICE": "cuda"}):
            self.assertEqual(select_device(), ("cuda", "cuda"))
            self.assertEqual(select_device("cpu"), ("cpu", "cpu"))
            self.assertEqual(select_device("auto"), ("auto", "cuda"))

    def test_unavailable_cuda_fails_and_auto_falls_back(self):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaises(RuntimeError):
                select_device("cuda")
            self.assertEqual(select_device("auto"), ("auto", "cpu"))
        with self.assertRaises(ValueError):
            select_device("bogus")

    def test_cuda_placement_must_be_proved(self):
        class Model:
            def __init__(self, device):
                self.device = device
            def parameters(self):
                return iter([types.SimpleNamespace(device=types.SimpleNamespace(type=self.device))])
        verify_model_device(Model("cuda"), "cuda")
        with self.assertRaises(RuntimeError):
            verify_model_device(Model("cpu"), "cuda")
        with self.assertRaises(RuntimeError):
            verify_model_device(object(), "cuda")
        verify_model_device(object(), "cpu")
