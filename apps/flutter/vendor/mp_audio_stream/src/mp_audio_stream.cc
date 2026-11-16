#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include "./miniaudio/miniaudio.h"

#include "mp_audio_stream.h"
#include "ggtts_audio_control.h"
#include <mutex>

#define DEVICE_FORMAT       ma_format_f32

// fade duration to avoid pop noise when the buffer is exhausted / recovered
#define FADE_SEC            0.005f

typedef struct {
    ma_device device;

    // lock-free single producer (push) / single consumer (data_callback) ring buffer
    ma_pcm_rb rb;
    bool rb_initialized;

    ma_uint32 channels;

    bool is_exhaust;
    ma_uint32 exhaust_recover_size;

    ma_uint32 exhaust_count;
    ma_uint32 full_count;

    // for fade-out on exhaust and fade-in on recovery
    float last[MA_MAX_CHANNELS];
    float gain;
    float gain_step;
    float decay;
    bool fade;

} _ctx_t;

_ctx_t * _ctx = NULL;
static bool device_initialized = false;

class ControlledDevice : public GgttsAudioControl::Backend {
public:
    bool start() override { return _ctx && ma_device_start(&_ctx->device) == MA_SUCCESS; }
    bool stop() override { return _ctx && ma_device_stop(&_ctx->device) == MA_SUCCESS; }
    void clear() override { if (_ctx && _ctx->rb_initialized) ma_pcm_rb_reset(&_ctx->rb); }
    void destroy() override {
        if (_ctx && device_initialized) {
            ma_device_uninit(&_ctx->device);
            device_initialized = false;
        }
        if (_ctx && _ctx->rb_initialized) {
            ma_pcm_rb_uninit(&_ctx->rb);
            _ctx->rb_initialized = false;
        }
    }
    uint64_t queued() override { return _ctx && _ctx->rb_initialized ? ma_pcm_rb_available_read(&_ctx->rb) : 0; }
};
static ControlledDevice controlled_device;
static GgttsAudioControl control(controlled_device);
static GgttsAudioLifecycle lifecycle;
int64_t ma_stream_pause() { std::lock_guard<std::mutex> lock(lifecycle.mutex()); return lifecycle.legacyAllowed() && control.pause() ? 0 : -1; }
int64_t ma_stream_resume_controlled() { std::lock_guard<std::mutex> lock(lifecycle.mutex()); return lifecycle.legacyAllowed() && control.resume() ? 0 : -1; }
int64_t ma_stream_stop_controlled() { return lifecycle.stop(control); }
int64_t ma_stream_software_empty() { return control.softwareEmpty() ? 1 : 0; }
int64_t ma_stream_control_state() { return static_cast<int64_t>(control.state()); }


// reads up to `frame_count` frames from the ring buffer, returns the number of frames read
static ma_uint32 rb_read(float* out, ma_uint32 frame_count)
{
    ma_uint32 total = 0;
    while (total < frame_count) {
        ma_uint32 frames = frame_count - total;
        void* p;
        if (ma_pcm_rb_acquire_read(&_ctx->rb, &frames, &p) != MA_SUCCESS || frames == 0) {
            break;
        }
        memcpy(&out[total * _ctx->channels], p, frames * _ctx->channels * sizeof(float));
        ma_pcm_rb_commit_read(&_ctx->rb, frames);
        total += frames;
    }
    return total;
}

// writes `frame_count` frames into the ring buffer (the caller checks the available space)
static void rb_write(const float* in, ma_uint32 frame_count)
{
    ma_uint32 total = 0;
    while (total < frame_count) {
        ma_uint32 frames = frame_count - total;
        void* p;
        if (ma_pcm_rb_acquire_write(&_ctx->rb, &frames, &p) != MA_SUCCESS || frames == 0) {
            break;
        }
        memcpy(p, &in[total * _ctx->channels], frames * _ctx->channels * sizeof(float));
        ma_pcm_rb_commit_write(&_ctx->rb, frames);
        total += frames;
    }
}

// processes `frame_count` frames of `out` in place: the first `read_frames` frames,
// already read from the ring buffer, are faded in, and the rest are filled by
// decaying the last output value toward zero
static void write_frames(float* out, ma_uint32 frame_count, ma_uint32 read_frames)
{
    const ma_uint32 channels = _ctx->channels;

    // without fade: keep as is, and fill zero when exhausted
    const bool fade = _ctx->fade;
    const float decay = fade ? _ctx->decay : 0.0f;

    for (ma_uint32 f = 0; f < frame_count; f++) {
        if (f < read_frames) {
            const float gain = fade ? _ctx->gain : 1.0f;
            for (ma_uint32 c = 0; c < channels; c++) {
                _ctx->last[c] = *out * gain;
                *out++ = _ctx->last[c];
            }
            _ctx->gain += _ctx->gain_step;
            if (_ctx->gain > 1.0f) _ctx->gain = 1.0f;
        } else {
            for (ma_uint32 c = 0; c < channels; c++) {
                _ctx->last[c] *= decay;
                *out++ = _ctx->last[c];
            }
            _ctx->gain = 0.0f;
        }
    }
}

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frame_count)
{
#ifdef MP_AUDIO_STREAM_DEBUG
    printf("callback: frameCount:%d available:%d\n", frame_count, ma_pcm_rb_available_read(&_ctx->rb));
#endif
    float * out = (float *)pOutput;

    ma_uint32 plyable_size = ma_pcm_rb_available_read(&_ctx->rb) * _ctx->channels;

    if (_ctx->is_exhaust && _ctx->exhaust_recover_size > plyable_size) {
        write_frames(out, frame_count, 0);
        _ctx->exhaust_count++;
        return;
    }

    _ctx->is_exhaust = false;
    ma_uint32 read_frames = rb_read(out, frame_count);
    // fade in the read frames, and fade out the rest if any
    write_frames(out, frame_count, read_frames);

    if (read_frames < frame_count) {
        _ctx->is_exhaust = true;
        _ctx->exhaust_count++;
    }
}

static int64_t push_locked(float* buf, int64_t length) {
    return control.push([&]() -> int64_t {
#ifdef MP_AUDIO_STREAM_DEBUGB
    printf("push: length:%lld available:%u\n", (long long)length, ma_pcm_rb_available_read(&_ctx->rb));
    for (int i=0; i<100; i+=10) {
        for (int j=0; j<10; j++) {
            unsigned char *b = (unsigned char *)(&buf[i+j]);
            printf("%02x%02x%02x%02x %f ", *(b+3), *(b+2), *(b+1), *(b+0), buf[i+j]);
        }
        printf("\n");
    }
    fflush(stdout);
#endif

    // reject partial frames, same as web, and lengths beyond the buffer's 32-bit frame count
    if (length < 0 || length % _ctx->channels != 0 || length / _ctx->channels > UINT32_MAX) {
        return -1;
    }
    ma_uint32 frames = (ma_uint32)(length / _ctx->channels);

    // ignore if no buffer remains
    if (ma_pcm_rb_available_write(&_ctx->rb) < frames) {
        _ctx->full_count++;
        return -1;
    }

    rb_write(buf, frames);

    return 0;
    });
}

int64_t ma_stream_stat_exhaust_count() {
    return _ctx->exhaust_count;
}

int64_t ma_stream_stat_full_count() {
    return _ctx->full_count;
}

void ma_stream_stat_reset() {
    _ctx->full_count = 0;
    _ctx->exhaust_count = 0;
}

int64_t ma_stream_uninit_checked() {
    std::lock_guard<std::mutex> lock(lifecycle.mutex());
    if (!lifecycle.legacyAllowed()) return -1;
    if (!_ctx || !device_initialized) return 0;
    return control.teardown() ? 0 : -1;
}

// Preserve the upstream ABI; new callers use the checked variant.
void ma_stream_uninit() { (void)ma_stream_uninit_checked(); }

static int64_t init_locked(int64_t max_buffer_size, int64_t keep_buffer_size, int64_t channels, int64_t sample_rate, int64_t fade_on_exhaust)
{
    // miniaudio takes 32-bit values, so reject the ones out of range
    if (channels < 1 || channels > MA_MAX_CHANNELS
        || sample_rate < 1 || sample_rate > UINT32_MAX
        || max_buffer_size < channels || max_buffer_size > UINT32_MAX
        || keep_buffer_size < 0 || keep_buffer_size > UINT32_MAX) {
        printf("Invalid parameters.\n");
        return -7;
    }
    const ma_uint32 ch = (ma_uint32)channels;
    const ma_uint32 rate = (ma_uint32)sample_rate;

    if (control.state() == GgttsAudioControl::State::running ||
        control.state() == GgttsAudioControl::State::paused) return -8;
    if (_ctx == NULL) {
        _ctx = (_ctx_t *)calloc(1,sizeof(_ctx_t));
        if (!_ctx) return -6;

        _ctx->rb_initialized = false;
        _ctx->is_exhaust = false;
        _ctx->exhaust_recover_size = 10 * 1024;
        _ctx->exhaust_count = 0;
        _ctx->full_count = 0;
    } else if (device_initialized) {
        ma_device_uninit(&_ctx->device);
        device_initialized = false;
    }

    GgttsInitGuard init_guard([] { controlled_device.destroy(); });
    ma_device_config deviceConfig;

    deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = DEVICE_FORMAT;
    deviceConfig.playback.channels = ch;
    deviceConfig.sampleRate        = rate;
    deviceConfig.dataCallback      = data_callback;

    if (ma_device_init(NULL, &deviceConfig, &_ctx->device) != MA_SUCCESS) {
        printf("Failed to open playback device.\n");
        return -4;
    }
    device_initialized = true;

#ifdef MP_AUDIO_STREAM_DEBUG
    printf("Device Name: %s\n", _ctx->device.playback.name);
#endif

    _ctx->exhaust_recover_size = (ma_uint32)keep_buffer_size;

    // the device is not started yet, so the callback does not touch the buffer here
    if (_ctx->rb_initialized) {
        ma_pcm_rb_uninit(&_ctx->rb);
        _ctx->rb_initialized = false;
    }

    if (ma_pcm_rb_init(DEVICE_FORMAT, ch, (ma_uint32)max_buffer_size / ch, NULL, NULL, &_ctx->rb) != MA_SUCCESS) {
        printf("Failed to allocate buffer.\n");
        return -6;
    }
    _ctx->rb_initialized = true;

    _ctx->channels = ch;

    memset(_ctx->last, 0, sizeof(_ctx->last));
    _ctx->gain = 0.0f;
    _ctx->gain_step = 1.0f / (FADE_SEC * (float)rate);
    _ctx->decay = expf(-1.0f / (FADE_SEC * (float)rate));
    _ctx->fade = fade_on_exhaust != 0;

    if (ma_device_start(&_ctx->device) != MA_SUCCESS) {
        printf("Failed to start playback device.\n");
        return -5;
    }

    if (!control.begin()) return -8;
    init_guard.commit();
    return 0;
}

// Legacy admission is separate from session-aware reads.
int64_t ma_stream_push(float* buf, int64_t length) {
    std::lock_guard<std::mutex> lock(lifecycle.mutex());
    if (!lifecycle.legacyAllowed()) return -3;
    return push_locked(buf, length);
}

int64_t ma_stream_init(int64_t max_buffer_size, int64_t keep_buffer_size, int64_t channels, int64_t sample_rate, int64_t fade_on_exhaust) {
    std::lock_guard<std::mutex> lock(lifecycle.mutex());
    if (!lifecycle.legacyAllowed()) return -9;
    return init_locked(max_buffer_size, keep_buffer_size, channels, sample_rate, fade_on_exhaust);
}

uint64_t ma_stream_session_new() { return lifecycle.reserve(); }
int64_t ma_stream_session_init(uint64_t id, int64_t max_buffer_size, int64_t keep_buffer_size, int64_t channels, int64_t sample_rate, int64_t fade_on_exhaust) {
    return lifecycle.initialize(id, [&] { return init_locked(max_buffer_size, keep_buffer_size, channels, sample_rate, fade_on_exhaust); });
}
int64_t ma_stream_session_pause(uint64_t id) { return lifecycle.invokeSession(id, [&] { return control.pause() ? 0 : -1; }); }
int64_t ma_stream_session_resume(uint64_t id) { return lifecycle.invokeSession(id, [&] { return control.resume() ? 0 : -1; }); }
int64_t ma_stream_session_stop(uint64_t id) { return lifecycle.stopSession(id, control); }
int64_t ma_stream_session_uninit(uint64_t id) {
    return lifecycle.release(id, [&] { return control.teardown() ? 0 : -1; });
}

int64_t ma_stream_session_push(uint64_t id, float* buf, int64_t length) {
    return lifecycle.pushSession(id, [&] { return push_locked(buf, length); });
}
int64_t ma_stream_session_software_empty(uint64_t id) {
    return lifecycle.invokeSession(id, [&] { return control.softwareEmpty() ? 1 : 0; });
}

int64_t ma_stream_session_state(uint64_t id) { return lifecycle.sessionState(id, control); }
