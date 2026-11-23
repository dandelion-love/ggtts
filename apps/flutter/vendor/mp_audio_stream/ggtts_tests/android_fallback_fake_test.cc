// Test production session wrappers using fake device/context calls and real ring/control.
// The only compiled miniaudio backend is null; fake calls never open any device.
#include "../src/mp_audio_stream.h"
#include "../src/miniaudio/miniaudio.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum class Failure { none, context, fallback_init, fallback_start, non_aaudio_start, stop_failure };
static Failure failure = Failure::none;
static bool plant = false;
static bool device_live = false;
static bool context_live = false;
static ma_context default_context;
static std::vector<std::string> events;

extern "C" ma_result ggtts_test_ma_context_init(const ma_backend* backends, ma_uint32 count,
                                 const ma_context_config*, ma_context* context) {
    events.emplace_back("context_init");
    assert(count == 1 && backends && backends[0] == ma_backend_opensl);
    assert(!device_live && !context_live);
    if (failure == Failure::context) return MA_ERROR;
    std::memset(context, 0, sizeof(*context));
    context->backend = ma_backend_opensl;
    context_live = true;
    return MA_SUCCESS;
}
extern "C" ma_result ggtts_test_ma_context_uninit(ma_context*) {
    events.emplace_back("context_uninit");
    assert(context_live && !device_live);
    context_live = false;
    return MA_SUCCESS;
}
extern "C" ma_result ggtts_test_ma_device_init(ma_context* context, const ma_device_config*, ma_device* device) {
    assert(!device_live);
    if (!context) {
        events.emplace_back("default_init");
        std::memset(&default_context, 0, sizeof(default_context));
        default_context.backend = failure == Failure::non_aaudio_start ? ma_backend_null : ma_backend_aaudio;
        context = &default_context;
    } else {
        events.emplace_back("fallback_init");
        assert(context_live && context->backend == ma_backend_opensl);
        if (failure == Failure::fallback_init) return MA_ERROR;
    }
    std::memset(device, 0, sizeof(*device));
    device->pContext = context;
    device_live = true;
    return MA_SUCCESS;
}
extern "C" ma_result ggtts_test_ma_device_start(ma_device* device) {
    assert(device_live && device && device->pContext);
    if (device->pContext->backend == ma_backend_opensl) {
        events.emplace_back("fallback_start");
        return failure == Failure::fallback_start ? MA_ERROR : MA_SUCCESS;
    }
    events.emplace_back("default_start");
    return plant ? MA_SUCCESS : MA_ERROR;
}
extern "C" ma_result ggtts_test_ma_device_stop(ma_device*) {
    assert(device_live);
    events.emplace_back("stop");
    return failure == Failure::stop_failure ? MA_ERROR : MA_SUCCESS;
}
extern "C" void ggtts_test_ma_device_uninit(ma_device*) {
    assert(device_live);
    events.emplace_back("device_uninit");
    device_live = false;
}

static void clear(Failure next) {
    assert(!device_live && !context_live);
    events.clear();
    failure = next;
}
static void eq(const std::vector<std::string>& expected) {
    if (events != expected) {
        for (const auto& e : events) std::fprintf(stderr, "%s ", e.c_str());
        std::fputc('\n', stderr);
    }
    assert(events == expected);
}
static void failed(Failure mode, const std::vector<std::string>& expected) {
    clear(mode);
    const auto id = ma_stream_session_new();
    assert(id != 0);
    assert(ma_stream_session_init(id, 64, 0, 1, 24000, 0) == -5);
    eq(expected);
    assert(!device_live && !context_live);
    assert(ma_stream_session_state(id) == 0);
    assert(ma_stream_session_uninit(id) == 0);
}

int main(int argc, char**) {
    plant = argc > 1;
    ma_backend backends[MA_BACKEND_COUNT];
    size_t count = 0;
    assert(ma_get_enabled_backends(backends, MA_BACKEND_COUNT, &count) == MA_SUCCESS);
    assert(count == 1 && backends[0] == ma_backend_null);

    clear(Failure::none);
    const auto first = ma_stream_session_new();
    assert(first);
    assert(ma_stream_session_init(first, 64, 0, 1, 24000, 0) == 0);
    eq({"default_init", "default_start", "device_uninit",
        "context_init", "fallback_init", "fallback_start"});
    float samples[] = {0.25f, -0.5f};
    assert(ma_stream_session_push(first, samples, 2) == 0);
    assert(ma_stream_session_pause(first) == 0);
    assert(ma_stream_session_resume(first) == 0);
    assert(ma_stream_session_stop(first) == 0);
    assert(ma_stream_session_push(first, samples, 2) == -3);
    assert(ma_stream_session_uninit(first) == 0);
    assert(!device_live && !context_live);
    eq({"default_init", "default_start", "device_uninit",
        "context_init", "fallback_init", "fallback_start",
        "stop", "fallback_start", "stop", "device_uninit", "context_uninit"});

    failed(Failure::context, {"default_init", "default_start", "device_uninit", "context_init"});
    failed(Failure::fallback_init, {"default_init", "default_start", "device_uninit", "context_init", "fallback_init", "context_uninit"});
    failed(Failure::fallback_start, {"default_init", "default_start", "device_uninit", "context_init", "fallback_init", "fallback_start", "device_uninit", "context_uninit"});
    failed(Failure::non_aaudio_start, {"default_init", "default_start", "device_uninit"});

    clear(Failure::stop_failure);
    const auto failed_stop = ma_stream_session_new();
    assert(failed_stop);
    assert(ma_stream_session_init(failed_stop, 64, 0, 1, 24000, 0) == 0);
    assert(ma_stream_session_stop(failed_stop) == -1);
    assert(ma_stream_session_uninit(failed_stop) == -1);
    assert(!device_live && !context_live);
    eq({"default_init", "default_start", "device_uninit", "context_init",
        "fallback_init", "fallback_start", "stop", "stop", "device_uninit", "context_uninit"});

    clear(Failure::none);
    const auto next = ma_stream_session_new();
    assert(next > first);
    assert(ma_stream_session_init(next, 64, 0, 1, 24000, 0) == 0);
    assert(ma_stream_session_stop(first) == -1);
    assert(ma_stream_session_state(next) == 1);
    assert(ma_stream_session_stop(next) == 0);
    assert(ma_stream_session_uninit(next) == 0);
    assert(!device_live && !context_live);
    std::puts("Android fallback fake device tests passed");
}
