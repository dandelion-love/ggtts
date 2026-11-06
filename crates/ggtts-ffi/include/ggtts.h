/* ggtts C-ABI. See docs/android/binding-api.md. Every int32_t function returns GGTTS_OK (0) or a negative code and
 * leaves the calling thread's message in ggtts_last_error(); a panic inside comes back as GGTTS_ERR_PANIC (an abort or
 * a crash in C code still ends the process). Handles are ids, never reused: a call with a freed id or 0 is
 * GGTTS_ERR_HANDLE. No call waits on another: a second call on an engine that is rendering, including one from its own
 * stream callback, is GGTTS_ERR_BUSY, and ggtts_free never waits for a render. */
#ifndef GGTTS_H
#define GGTTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGTTS_OK 0
#define GGTTS_ERR_ARG (-1)
#define GGTTS_ERR_HANDLE (-2)
#define GGTTS_ERR_ENGINE (-3)
#define GGTTS_ERR_PANIC (-4)
#define GGTTS_ERR_STOPPED (-5)
#define GGTTS_ERR_BUSY (-6)

/* Return 0 to go on, anything else to stop the render (GGTTS_ERR_STOPPED). */
typedef int32_t (*ggtts_piece_fn)(void *user, const float *samples, size_t len, uint32_t rate);

/* engine: "qwen3-tts", "kitten-tts-2" or "fake". codec and voice may be NULL where the engine needs none. */
int32_t ggtts_load(const char *engine, const char *model, const char *codec, const char *voice, int32_t threads,
                   uint64_t *handle);
int32_t ggtts_render(uint64_t handle, const char *text, uint64_t seed, float **pcm, size_t *len, uint32_t *rate);
int32_t ggtts_render_stream(uint64_t handle, const char *text, uint64_t seed, ggtts_piece_fn on_piece, void *user);
void ggtts_pcm_free(float *pcm, size_t len);
int32_t ggtts_free(uint64_t handle);
const char *ggtts_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
