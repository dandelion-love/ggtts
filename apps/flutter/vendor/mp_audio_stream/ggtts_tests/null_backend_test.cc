// Exercises the production wrappers and callback with only miniaudio's null backend.
#include "../src/mp_audio_stream.h"
#include "../src/miniaudio/miniaudio.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

static std::mutex observed_mutex;
static std::vector<float> observed;
extern "C" void ggtts_test_observe(const float* samples, unsigned count) {
    std::lock_guard<std::mutex> lock(observed_mutex);
    observed.insert(observed.end(), samples, samples + count);
}
static bool wait_empty(uint64_t id) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ma_stream_session_software_empty(id) == 1) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}
int main(int argc, char**) {
    ma_backend backends[MA_BACKEND_COUNT];
    size_t count = 0;
    assert(ma_get_enabled_backends(backends, MA_BACKEND_COUNT, &count) == MA_SUCCESS);
    assert(count == 1 && backends[0] == ma_backend_null);
    auto first = ma_stream_session_new();
    assert(first && ma_stream_session_init(first, 48000, 0, 1, 24000, 0) == 0);
    std::vector<float> input(12000, 0.25f);
    std::fill(input.begin() + 6000, input.end(), 0.75f);
    assert(ma_stream_session_push(first, input.data(), input.size()) == 0);
    assert(ma_stream_session_pause(first) == 0);
    assert(ma_stream_session_state(first) == 2);
    assert(ma_stream_session_push(first, input.data(), 1) == -2);
    size_t paused;
    { std::lock_guard<std::mutex> lock(observed_mutex); paused = observed.size(); }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    { std::lock_guard<std::mutex> lock(observed_mutex); assert(observed.size() == paused); }
    assert(ma_stream_session_resume(first) == 0);
    assert(wait_empty(first));
    assert(ma_stream_session_pause(first) == 0);
    {
        std::lock_guard<std::mutex> lock(observed_mutex);
        assert(observed.size() == input.size());
        if (argc > 1) observed.insert(observed.begin(), 0.0f); // planted silence must fail the sample oracle
        assert(observed == input);
    }
    // Resume, queue a long tail, then stop while paused. It must never reach a later session.
    assert(ma_stream_session_resume(first) == 0);
    std::vector<float> tail(24000, 0.5f);
    assert(ma_stream_session_push(first, tail.data(), tail.size()) == 0);
    assert(ma_stream_session_pause(first) == 0);
    assert(ma_stream_session_stop(first) == 0);
    assert(ma_stream_session_state(first) == 3);
    assert(ma_stream_session_push(first, tail.data(), tail.size()) == -3);
    assert(ma_stream_session_uninit(first) == 0);
    { std::lock_guard<std::mutex> lock(observed_mutex); observed.clear(); }
    auto second = ma_stream_session_new();
    assert(second > first);
    assert(ma_stream_session_init(second, 48000, 0, 1, 24000, 0) == 0);
    assert(ma_stream_session_stop(first) == -1);
    std::vector<float> next(6000, 0.125f);
    assert(ma_stream_session_push(second, next.data(), next.size()) == 0);
    assert(wait_empty(second));
    assert(ma_stream_session_pause(second) == 0);
    { std::lock_guard<std::mutex> lock(observed_mutex); assert(observed == next); }
    assert(ma_stream_session_stop(second) == 0);
    assert(ma_stream_session_uninit(second) == 0);
    std::puts("null backend session integration passed");
}
