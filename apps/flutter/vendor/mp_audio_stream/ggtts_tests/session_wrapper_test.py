import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])
lib.ma_stream_session_new.restype = ctypes.c_uint64
for name in ("ma_stream_session_stop", "ma_stream_session_uninit",
             "ma_stream_session_pause", "ma_stream_session_resume",
             "ma_stream_session_software_empty", "ma_stream_session_state"):
    f = getattr(lib, name)
    f.argtypes = [ctypes.c_uint64]
    f.restype = ctypes.c_int64
lib.ma_stream_session_init.argtypes = [ctypes.c_uint64] + [ctypes.c_int64] * 5
lib.ma_stream_session_init.restype = ctypes.c_int64
lib.ma_stream_init.argtypes = [ctypes.c_int64] * 5
lib.ma_stream_init.restype = ctypes.c_int64
lib.ma_stream_session_push.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
lib.ma_stream_session_push.restype = ctypes.c_int64

first = lib.ma_stream_session_new()
assert first != 0
assert lib.ma_stream_session_new() == 0
assert lib.ma_stream_session_state(first) == 0
assert lib.ma_stream_session_stop(first) == 0
assert lib.ma_stream_session_state(first) == 3
assert lib.ma_stream_session_init(first, 1, 0, 1, 24000, 0) == -1
assert lib.ma_stream_session_push(first, None, 0) == -3
assert lib.ma_stream_init(1, 0, 1, 24000, 0) == -9
assert lib.ma_stream_session_uninit(first) == 0
second = lib.ma_stream_session_new()
assert second > first
assert lib.ma_stream_session_state(first) == -1
assert lib.ma_stream_session_state(second) == 0
assert lib.ma_stream_session_stop(first) == -1
assert lib.ma_stream_session_pause(first) == -1
assert lib.ma_stream_session_uninit(first) == -1
assert lib.ma_stream_session_stop(second) == 0
assert lib.ma_stream_session_uninit(second) == 0
print("exported session wrappers passed without opening an audio device")
