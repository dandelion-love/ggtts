"""ggtts for Python: load a text-to-speech engine and render, with the settings of ggtts's C-ABI.

    import ggtts

    with ggtts.load() as tts:                 # Qwen3-TTS, its default model (downloaded once into the cache)
        audio = tts.render("今日天氣好好。")
        audio.write_flac("out.flac")
        samples = audio.samples               # float32: a numpy array when numpy is installed

Every option of ggtts's options table reaches Python through ``options``: a dict (or a JSON string) shaped as the
C-ABI's options documents, given to ``load`` (the ``load`` section, and ``render`` defaults) and to each render::

    with ggtts.load("qwen3-tts", options={"render": {"edge_s": 0.5}}) as tts:
        audio = tts.render("今日天氣好好。", options={"render": {"steer": "calm.safetensors"}})

Errors are ``GgttsError``, with ``code`` (one of the ``ERR_*`` constants, the C-ABI's codes), ``message`` and
``key``. A refused option is an ``OptionError``, which is a ``GgttsError`` and a ``ValueError``, with ``key`` the
option's schema key (``render.steer``). A panic inside ggtts never crosses into Python: it is a ``GgttsError`` with
``ERR_PANIC``. The GIL is released while an engine loads and renders.
"""

from __future__ import annotations

import array
import json
import os
import queue
import threading
from typing import Any, Callable, Iterator, List, Mapping, Optional, Union

from . import _native
from ._native import (
    COMMIT,
    DEFAULT_SEED,
    ENGINES,
    ERR_ARG,
    ERR_BUSY,
    ERR_ENGINE,
    ERR_HANDLE,
    ERR_PANIC,
    ERR_STOPPED,
    OK,
    VERSION,
    GgttsError,
)

try:  # numpy is optional: without it, samples are an array.array of float32.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by the fallback test
    _np = None

__all__ = [
    "Audio", "TTS", "GgttsError", "OptionError", "load", "options_schema", "write_flac", "DEFAULT_SEED", "ENGINES", "VERSION", "COMMIT",
    "OK", "ERR_ARG", "ERR_HANDLE", "ERR_ENGINE", "ERR_PANIC", "ERR_STOPPED", "ERR_BUSY",
]

__version__ = VERSION

PathLike = Union[str, "os.PathLike[str]"]
#: An options document: a dict shaped as the C-ABI's (``{"render": {"edge_s": 0.5}}``), or the same as a JSON string.
Options = Union[Mapping[str, Any], str]


class OptionError(GgttsError, ValueError):
    """An option refused: an unknown key, a value of the wrong type, an option the engine or this binding does not
    take, or one given both as an argument and in ``options``. ``key`` is the option's schema key (``render.steer``),
    or None when the whole document is refused (not JSON, not an object)."""


def _document(options: Optional[Options]) -> Optional[str]:
    """An options document as the JSON string the library reads."""
    if options is None or isinstance(options, str):
        return options
    if isinstance(options, Mapping):
        return json.dumps(options)
    raise TypeError(f"options must be a dict or a JSON string, not {type(options).__name__}")


def _refused(e: GgttsError, options: Optional[str]) -> GgttsError:
    """``e`` as an ``OptionError`` when it refuses an option: it names one, or a document was given and the call
    failed with ``ERR_ARG``. Any other error is ``e`` as it is."""
    key = getattr(e, "key", None)
    if key is None and not (options is not None and getattr(e, "code", None) == ERR_ARG):
        return e
    err = OptionError(*e.args)
    err.code, err.message, err.key = e.code, e.message, key
    return err


def _call(f, doc: Optional[str], /, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except GgttsError as e:
        refused = _refused(e, doc)
        if refused is e:
            raise
        raise refused from None


def _path(p: Optional[PathLike]) -> Optional[str]:
    return None if p is None else os.fspath(p)


def _float32_bytes(samples) -> bytes:
    """Samples (an Audio, a numpy array, an array.array, bytes of float32, or a sequence of floats) as float32 bytes."""
    if isinstance(samples, Audio):
        return bytes(samples.pcm)
    if isinstance(samples, (bytes, bytearray, memoryview)):
        return bytes(samples)
    if _np is not None and isinstance(samples, _np.ndarray):
        return _np.ascontiguousarray(samples, dtype=_np.float32).tobytes()
    if isinstance(samples, array.array) and samples.typecode == "f":
        return samples.tobytes()
    return array.array("f", samples).tobytes()


class Audio:
    """Mono float32 samples and their rate."""

    __slots__ = ("pcm", "sample_rate")

    def __init__(self, pcm: bytearray, sample_rate: int):
        #: The samples as native-endian float32 bytes.
        self.pcm = pcm
        self.sample_rate = sample_rate

    @property
    def samples(self):
        """The samples: a float32 numpy array when numpy is installed, else an ``array.array('f')``."""
        if _np is not None:
            return _np.frombuffer(self.pcm, dtype=_np.float32)
        out = array.array("f")
        out.frombytes(bytes(self.pcm))
        return out

    def numpy(self):
        """The samples as a float32 numpy array (needs numpy)."""
        if _np is None:
            raise ImportError("numpy is not installed: use Audio.samples or Audio.pcm")
        return _np.frombuffer(self.pcm, dtype=_np.float32)

    def to_bytes(self) -> bytes:
        """The samples as native-endian float32 bytes."""
        return bytes(self.pcm)

    def __len__(self) -> int:
        return len(self.pcm) // 4

    @property
    def duration(self) -> float:
        """Seconds."""
        return len(self) / self.sample_rate if self.sample_rate else 0.0

    def write_flac(self, path: PathLike) -> str:
        """Write as 16-bit mono FLAC (the suffix is forced to ``.flac``); returns the path written."""
        return _native.write_flac(os.fspath(path), bytes(self.pcm), self.sample_rate)

    def __repr__(self) -> str:
        return f"Audio({len(self)} samples, {self.sample_rate} Hz, {self.duration:.2f} s)"


class TTS:
    """A loaded engine. Use it as a context manager, or call ``close()``."""

    def __init__(self, engine: "_native.Engine"):
        self._engine = engine

    # What was loaded.
    engine = property(lambda self: self._engine.engine, doc="The engine name, e.g. ``qwen3-tts``.")
    speaker = property(lambda self: self._engine.speaker, doc="The speaker in effect, or None for the model's own.")
    instruction = property(lambda self: self._engine.instruction, doc="The style instruction, or None.")
    device = property(lambda self: self._engine.device, doc="``cpu`` or ``cuda``.")
    closed = property(lambda self: self._engine.closed)

    @property
    def info(self) -> dict:
        """The engine's family, compute backend and parity level."""
        e = self._engine
        return {"family": e.family, "backend": e.backend, "parity": e.parity}

    @property
    def model_files(self) -> List[dict]:
        """Each model file: ``role``, ``path``, and ``source`` (``path`` if given, ``cache`` for the default)."""
        return [{"role": r, "path": p, "source": s} for r, p, s in self._engine.model_files]

    def capabilities(self) -> dict:
        """What the loaded engine supports, as the library states it (the C-ABI's ``ggtts_capabilities``, format
        ``ggtts-capabilities/1``): ``family``, ``instruction`` (and ``no_instruction_why``), ``steering``,
        ``speakers`` and ``speaker_names`` (the model's speakers), ``devices``, ``hooks`` and ``step_events``."""
        return json.loads(self._engine.capabilities())

    def render(self, text: str, seed: Optional[int] = None, options: Optional[Options] = None) -> Audio:
        """Render ``text`` (one paragraph per non-empty line), as the C-ABI's ``ggtts_render`` does.

        ``seed`` is the salt of the story seed every chunk's sampler seed is derived from (the document's
        ``render.seed_salt``), ``DEFAULT_SEED`` when neither it nor ``options`` gives one. ``options`` is a render
        document (``ggtts_render_opts``'s), laid over the load's ``render`` defaults: steering, an edge, redraws, a
        trace. A load option in it, or one this binding does not take, is an ``OptionError``.
        """
        doc = _document(options)
        pcm, rate = _call(self._engine.render, doc, text, seed, doc)
        return Audio(pcm, rate)

    def render_stream(
        self,
        text: str,
        on_chunk: Callable[[Audio], Optional[bool]],
        seed: Optional[int] = None,
        options: Optional[Options] = None,
    ) -> bool:
        """Render ``text``, calling ``on_chunk(audio)`` with each chunk, in order, as soon as it is ready.

        ``on_chunk`` returning ``False`` stops the render; an exception it raises stops it and is raised here.
        Returns True when the whole text was rendered, False when stopped. The chunks are the render's own samples
        before its final level step, so their concatenation has the render's length, not its exact values.
        ``seed`` and ``options`` are as ``render``'s.
        """
        doc = _document(options)
        return _call(self._engine.render_stream, doc, text, lambda pcm, rate: on_chunk(Audio(pcm, rate)), seed, doc)

    def stream(self, text: str, seed: Optional[int] = None, options: Optional[Options] = None) -> Iterator[Audio]:
        """Iterate over the chunks of a render, in order. Stop early with ``break`` (or ``close()`` the generator):
        the render stops after the chunk it is generating. ``seed`` and ``options`` are as ``render``'s."""
        doc = _document(options)
        chunks: "queue.Queue" = queue.Queue(maxsize=1)
        stop = threading.Event()
        done = object()
        failure: list = []

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    chunks.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def run():
            try:
                self._engine.render_stream(text, lambda pcm, rate: put(Audio(pcm, rate)), seed, doc)
            except GgttsError as e:
                failure.append(_refused(e, doc))
            except BaseException as e:  # noqa: BLE001 - handed to the consumer
                failure.append(e)
            finally:
                put(done)

        worker = threading.Thread(target=run, name="ggtts-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = chunks.get()
                if item is done:
                    break
                yield item
            if failure:
                raise failure[0]
        finally:
            stop.set()
            worker.join()

    def close(self) -> None:
        """Drop the engine; later calls raise ``GgttsError`` with ``ERR_HANDLE``."""
        self._engine.close()

    def __enter__(self) -> "TTS":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else self.device
        return f"<ggtts.TTS {self.engine} speaker={self.speaker!r} ({state})>"


def load(
    engine: Optional[str] = None,
    model: Optional[PathLike] = None,
    codec: Optional[PathLike] = None,
    voice: Optional[PathLike] = None,
    speaker: Optional[str] = None,
    instruction: Optional[str] = None,
    device: Optional[str] = None,
    threads: Optional[int] = None,
    offline: Optional[bool] = None,
    progress: Optional[Callable[[str], None]] = None,
    options: Optional[Options] = None,
) -> TTS:
    """Load an engine, with the settings of the C-ABI's ``ggtts_load_with``, or an options document.

    engine:      ``qwen3-tts`` (the default unless ``options`` names one), ``kitten-tts-2``, or ``fake`` (a test
                 engine that needs no files).
    model:       the model GGUF (Qwen3-TTS: the talker; Kitten-TTS 2: the LM). For Qwen3-TTS, a model or codec not
                 given is the pinned default model's, from the per-user cache (``GGTTS_CACHE_DIR`` overrides where),
                 downloaded and verified first unless ``offline``. With the default model and no speaker, the speaker
                 is the default model's default speaker.
    codec:       the codec GGUF (Kitten-TTS 2: the S3Gen GGUF).
    voice:       Kitten-TTS 2's reference voice JSON (``ggtts-kitten prepare`` writes it).
    speaker:     a Qwen3-TTS speaker name (case is ignored). An unknown one is refused with the model's speakers.
    instruction: a Qwen3-TTS style instruction ("Speak slowly and softly."); a checkpoint that takes none refuses it.
    device:      ``cpu`` (the default) or ``cuda`` (``cuda`` needs a build with CUDA).
    threads:     CPU threads; None for 4.
    offline:     never download: a default model file missing from the cache is an error naming it.
    progress:    called with each download message; by default they go to stderr.
    options:     a load document, the C-ABI's ``ggtts_load_opts`` document as a dict or JSON string: the ``load``
                 section (every argument above has its key there, and ``cache_dir`` too) and ``render`` defaults
                 every render on the engine starts from. The whole document is checked against the engine before
                 anything is read.

    A setting the engine does not take is refused (``ERR_ARG``), never ignored; an empty string is no setting. A value
    given both as an argument and in ``options`` is an ``OptionError`` naming both.
    """
    doc = _document(options)
    native = _call(
        _native.load,
        doc,
        engine,
        model=_path(model),
        codec=_path(codec),
        voice=_path(voice),
        speaker=speaker,
        instruction=instruction,
        device=device,
        threads=threads,
        offline=offline,
        options=doc,
        progress=progress,
    )
    return TTS(native)


def options_schema() -> dict:
    """The JSON Schema of an options document, as the library states it (the C-ABI's ``ggtts_options_schema``): every
    option ``options=`` may give, with its type, default and phase. No schema file ships with the package."""
    return json.loads(_native.options_schema())


def write_flac(path: PathLike, samples, sample_rate: Optional[int] = None) -> str:
    """Write float32 mono ``samples`` (an Audio, numpy array, array.array, float32 bytes or a sequence of floats) as
    16-bit FLAC at ``sample_rate`` (an Audio's own rate when not given). The suffix is forced to ``.flac``; returns
    the path written."""
    if sample_rate is None:
        if not isinstance(samples, Audio):
            raise TypeError("write_flac needs a sample_rate unless samples is an Audio")
        sample_rate = samples.sample_rate
    return _native.write_flac(os.fspath(path), _float32_bytes(samples), sample_rate)
