// No physical backend is compiled. The production path must reject the null sink.
#include "../src/mp_audio_stream.h"
#include "../src/miniaudio/miniaudio.h"
#include <cassert>
#include <cstdio>

int main() {
    ma_backend backends[MA_BACKEND_COUNT];
    size_t count = 0;
    assert(ma_get_enabled_backends(backends, MA_BACKEND_COUNT, &count) == MA_SUCCESS);
    assert(count == 1 && backends[0] == ma_backend_null);
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto id = ma_stream_session_new();
        assert(id != 0);
        assert(ma_stream_session_init(id, 48000, 0, 1, 24000, 0) == -10);
        assert(ma_stream_session_state(id) == 0);
        assert(ma_stream_session_uninit(id) == 0);
    }
    std::puts("production null backend refused and cleaned up");
}
