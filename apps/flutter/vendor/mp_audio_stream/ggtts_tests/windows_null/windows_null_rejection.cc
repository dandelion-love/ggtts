#include "../../src/mp_audio_stream.h"
#include "../../src/miniaudio/miniaudio.h"
#include <cstdio>

int main() {
    ma_backend backends[MA_BACKEND_COUNT];
    size_t count = 0;
    if (ma_get_enabled_backends(backends, MA_BACKEND_COUNT, &count) != MA_SUCCESS ||
        count != 1 || backends[0] != ma_backend_null) {
        std::fprintf(stderr, "test did not force the null backend\n");
        return 2;
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        const auto session = ma_stream_session_new();
        if (!session) return 3;
        const auto result = ma_stream_session_init(session, 48000, 0, 1, 24000, 0);
        if (result != -4) {
            std::fprintf(stderr, "null backend must fail initialization; got %lld\n", static_cast<long long>(result));
            ma_stream_session_uninit(session);
            return 1;
        }
        if (ma_stream_session_state(session) != 0 || ma_stream_session_uninit(session) != 0) {
            std::fprintf(stderr, "failed initialization must leave no active playback session\n");
            return 4;
        }
    }
    if (ma_stream_init(48000, 0, 1, 24000, 0) != -4) return 5;
    ma_stream_uninit();
    return 0;
}
