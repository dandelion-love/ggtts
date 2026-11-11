"""ggtts_instrument: write a ggtts event instrument in Python.

An event instrument is a separate program that the runner starts after a render. It reads the render's event
stream (`ggtts-trace/2` or `/1`) and its audio, and writes one envelope (`ggtts-report/2`). This package needs only the
standard library:

- `events`: read a trace from a file, or live from a pipe;
- `audio`: read the render's audio (WAV, and the FLAC ggtts writes);
- `envelope`: build an envelope and check it before it is written;
- `cli`: the command line the runner passes, and `run()`, which turns any crash into a written `failed` envelope.

A template using all four is in `instruments/templates/python/`. See `docs/instruments/hooks.md` for what each hook
carries.
"""

from .audio import Audio, AudioError, read_audio
from .cli import Context, parse_args, run
from .device import select_device, verify_model_device
from .envelope import Envelope, EnvelopeError, check
from .events import Trace, TraceError, iter_events, read_trace

__version__ = "0.1.0"

__all__ = [
    "Audio",
    "AudioError",
    "Context",
    "Envelope",
    "EnvelopeError",
    "Trace",
    "TraceError",
    "check",
    "iter_events",
    "parse_args",
    "read_audio",
    "read_trace",
    "run",
    "select_device",
    "verify_model_device",
]
