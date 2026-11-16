#ifdef __cplusplus
#ifdef WIN32
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C" __attribute__((visibility("default"))) __attribute__((used))
#endif // WIN32
    #include <cstdint>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
#else // __cplusplus - Objective-C or other C platform
    #define EXPORT extern
    #include "stdint.h"
    #include "stdio.h"
    #include "stdlib.h"
    #include "string.h"
#endif


EXPORT
int64_t ma_stream_init(int64_t max_buffer_size, int64_t keep_buffer_size, int64_t channels, int64_t sample_rate, int64_t fade_on_exhaust);

EXPORT
void ma_stream_uninit(void);

EXPORT
int64_t ma_stream_push(float*, int64_t);

EXPORT
int64_t ma_stream_stat_exhaust_count(void);

EXPORT
int64_t ma_stream_stat_full_count(void);

EXPORT
void ma_stream_stat_reset(void);

// Returns 0 on success. Control calls may block while the device transitions.
EXPORT int64_t ma_stream_pause(void);
EXPORT int64_t ma_stream_resume_controlled(void);
EXPORT int64_t ma_stream_stop_controlled(void);
// 1 means the software ring is empty, not that the hardware has played its buffer.
EXPORT int64_t ma_stream_software_empty(void);
// idle=0, running=1, paused=2, stopped=3.
EXPORT int64_t ma_stream_control_state(void);

// Always releases the device; returns -1 when its explicit stop failed.
EXPORT int64_t ma_stream_uninit_checked(void);
