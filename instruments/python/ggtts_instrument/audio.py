"""Read a render's audio as mono float samples in [-1, 1].

- **WAV:** 16-bit PCM or 32-bit float. A streamed WAV whose sizes are 0xFFFFFFFF (`ggtts-render --stream`) is read to
  the end of the file.
- **FLAC:** read with `soundfile` when it is installed, which handles any FLAC. Without it, this module decodes the
  subset ggtts writes itself (CONSTANT and VERBATIM subframes, any bit depth, mono or independent channels), so the
  standard library is enough for ggtts's own output. A FLAC that uses predictive subframes needs `soundfile`, and the
  error says so.

Several channels are averaged to one.
"""

import hashlib
import os
import struct
import sys
from contextvars import ContextVar

_DECODERS = ContextVar("ggtts_audio_decoders", default=None)


def _start_decoder_capture():
    records = []
    return records, _DECODERS.set(records)


def _stop_decoder_capture(token):
    _DECODERS.reset(token)


def _loaded_sndfile_paths():
    """Resolve the modules backing the open/read/close functions soundfile uses.

    A mapped library with a matching filename alone is insufficient: another
    library might be bound. Resolve the address actually bound by soundfile.
    """
    import soundfile
    paths = []
    for symbol in ("sf_open", "sf_readf_float", "sf_close"):
        address = int(soundfile._ffi.cast("uintptr_t", getattr(soundfile._snd, symbol)))
        path = _bound_module_path(address)
        if not path:
            return []
        paths.append(path)
    return [paths[0]] if len({os.path.normcase(os.path.realpath(p)) for p in paths}) == 1 else []


def _bound_module_path(address):
    import ctypes
    if not address:
        return None
    if sys.platform.startswith("linux"):
        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p)]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        if dladdr(ctypes.c_void_p(address), ctypes.byref(info)) and info.filename:
            path = os.fsdecode(info.filename)
            return path if os.path.isabs(path) else None
        return None
    if sys.platform == "win32":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = wintypes.HMODULE()
        get_handle = kernel.GetModuleHandleExW
        get_handle.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.HMODULE)]
        get_handle.restype = wintypes.BOOL
        if not get_handle(0x4 | 0x2, ctypes.c_void_p(address), ctypes.byref(handle)):
            return None
        name = kernel.GetModuleFileNameW
        name.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        name.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        n = name(handle, buf, len(buf))
        return buf.value if n and n < len(buf) - 1 and os.path.isabs(buf.value) else None
    return None


def _decoder(kind):
    """Identify decoder code without exposing local paths; never guess on ambiguity."""
    digest = None
    try:
        if kind == "libsndfile":
            paths = _loaded_sndfile_paths()
            if len(paths) == 1:
                digest = sha256_of(paths[0])
        else:
            digest = sha256_of(__file__)
    except (OSError, ValueError, AttributeError, TypeError, ImportError, OverflowError):
        pass
    record = {"kind": kind, "sha256": digest}
    records = _DECODERS.get()
    if records is not None:
        records.append(record)
    return record


class AudioError(ValueError):
    """The file is not audio this module can read."""


class Audio:
    def __init__(self, samples, rate, path, sha256, decoder=None):
        self.samples = samples
        self.rate = rate
        self.path = path
        self.sha256 = sha256
        self.decoder = decoder

    @property
    def seconds(self):
        return len(self.samples) / float(self.rate) if self.rate else 0.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _mix(channels):
    if len(channels) == 1:
        return channels[0]
    n = len(channels)
    return [sum(s) / n for s in zip(*channels)]


# --- WAV ---------------------------------------------------------------------------------------------------------


def _read_wav(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[body:body + 16])
        elif cid == b"data":
            end = len(data) if size == 0xFFFFFFFF else min(len(data), body + size)
            pcm = data[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise AudioError("a WAV file needs `fmt ` and `data` chunks")
    tag, channels, rate, _, _, bits = fmt
    if tag == 1 and bits == 16:
        n = len(pcm) // 2
        flat = [x / 32768.0 for x in struct.unpack("<%dh" % n, pcm[:n * 2])]
    elif tag == 3 and bits == 32:
        n = len(pcm) // 4
        flat = list(struct.unpack("<%df" % n, pcm[:n * 4]))
    else:
        raise AudioError("WAV format %d with %d bits is not read here (16-bit PCM or 32-bit float)" % (tag, bits))
    flat = flat[:len(flat) - len(flat) % channels]
    return _mix([flat[c::channels] for c in range(channels)]), rate


# --- FLAC, the subset ggtts writes ------------------------------------------------------------------------------


class _Bits:
    def __init__(self, data, pos):
        self.data, self.bit = data, pos * 8

    def read(self, n):
        v = 0
        for _ in range(n):
            byte = self.data[self.bit >> 3]
            v = (v << 1) | ((byte >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        return v

    def signed(self, n):
        v = self.read(n)
        return v - (1 << n) if n and v & (1 << (n - 1)) else v

    def utf8(self):
        first = self.read(8)
        extra = 0
        mask = 0x80
        while first & mask:
            extra += 1
            mask >>= 1
        v = first & (mask - 1) if extra else first
        for _ in range(max(0, extra - 1)):
            v = (v << 6) | (self.read(8) & 0x3F)
        return v

    def align(self):
        self.bit = (self.bit + 7) & ~7


_BLOCK = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608}
_DEPTH = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}
_RATE = {1: 88200, 2: 176400, 3: 192000, 4: 8000, 5: 16000, 6: 22050, 7: 24000, 8: 32000, 9: 44100, 10: 48000,
         11: 96000}


def _read_flac(data):
    if data[:4] != b"fLaC":
        raise AudioError("not a FLAC file")
    pos, info, last = 4, None, False
    while not last:
        hdr = data[pos]
        last, kind = bool(hdr & 0x80), hdr & 0x7F
        size = int.from_bytes(data[pos + 1:pos + 4], "big")
        if kind == 0:
            b = _Bits(data, pos + 4 + 10)
            info = (b.read(20), b.read(3) + 1, b.read(5) + 1, b.read(36))
        pos += 4 + size
    if info is None:
        raise AudioError("FLAC without STREAMINFO")
    rate, nch, depth, total = info
    channels = [[] for _ in range(nch)]
    while pos + 2 <= len(data):
        b = _Bits(data, pos)
        if b.read(15) != 0x7FFC:
            raise AudioError("lost FLAC frame sync at byte %d" % pos)
        b.read(1)
        bs_code, rate_code, chan, depth_code = b.read(4), b.read(4), b.read(4), b.read(3)
        b.read(1)
        b.utf8()
        if bs_code == 6:
            block = b.read(8) + 1
        elif bs_code == 7:
            block = b.read(16) + 1
        elif bs_code >= 8:
            block = 256 << (bs_code - 8)
        elif bs_code in _BLOCK:
            block = _BLOCK[bs_code]
        else:
            raise AudioError("reserved FLAC block size")
        if rate_code == 12:
            b.read(8)
        elif rate_code in (13, 14):
            b.read(16)
        if chan >= 8:
            raise AudioError("this FLAC uses stereo decorrelation; install soundfile to read it")
        frame_depth = _DEPTH.get(depth_code, depth) if depth_code else depth
        b.read(8)  # header CRC-8
        for c in range(chan + 1):
            b.read(1)
            kind = b.read(6)
            wasted = 0
            if b.read(1):
                wasted = 1
                while not b.read(1):
                    wasted += 1
            d = frame_depth - wasted
            if kind == 0:
                channels[c].extend([b.signed(d) << wasted] * block)
            elif kind == 1:
                if d == 16 and b.bit % 8 == 0 and not wasted:
                    start = b.bit >> 3
                    channels[c].extend(struct.unpack(">%dh" % block, data[start:start + 2 * block]))
                    b.bit += 16 * block
                else:
                    channels[c].extend(b.signed(d) << wasted for _ in range(block))
            else:
                raise AudioError("this FLAC uses predictive subframes, which only ggtts's own subset reader "
                                 "skips; install soundfile (pip install soundfile) to read it")
        b.align()
        pos = (b.bit >> 3) + 2  # frame CRC-16
    scale = float(1 << (depth - 1))
    mono = _mix([[x / scale for x in ch] for ch in channels])
    if total and len(mono) != total:
        raise AudioError("FLAC decoded %d samples, STREAMINFO says %d" % (len(mono), total))
    return mono, rate


def _soundfile(path):
    try:
        import soundfile  # optional
    except ImportError:
        return None
    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return _mix([list(map(float, data[:, c])) for c in range(data.shape[1])]), int(rate)


def read_audio(path, prefer_soundfile=True):
    """Read `path` as mono float samples. Raises `AudioError` with the fix when it cannot."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"RIFF":
        samples, rate = _read_wav(data)
        decoder = _decoder("ggtts-python-wav")
    elif data[:4] == b"fLaC":
        got = _soundfile(path) if prefer_soundfile else None
        samples, rate = got if got is not None else _read_flac(data)
        decoder = _decoder("libsndfile" if got is not None else "ggtts-python-flac")
    else:
        raise AudioError("%s is neither WAV nor FLAC" % path)
    return Audio(samples, rate, str(path), sha256_of(path), decoder)
