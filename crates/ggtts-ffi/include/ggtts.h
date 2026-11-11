/* ggtts C-ABI. See docs/android/binding-api.md. Every int32_t function returns GGTTS_OK (0) or a negative code and
 * leaves the calling thread's message in ggtts_last_error(); a panic inside comes back as GGTTS_ERR_PANIC (an abort or
 * a crash in C code still ends the process). Handles are ids, never reused: a call with a freed id or 0 is
 * GGTTS_ERR_HANDLE. No call waits on another: a second call on an engine that is rendering, including one from its own
 * stream callback, is GGTTS_ERR_BUSY, and ggtts_free never waits for a render.
 *
 * Versioning: GGTTS_ABI_VERSION goes up by one whenever a function, a struct or a code is added, and an exported
 * function never changes its signature or its behaviour. ggtts_abi_version() is the version of the library actually
 * loaded; a caller built against a newer header checks it before calling a function the library may not have.
 *   1  ggtts_load, ggtts_render, ggtts_render_stream, ggtts_pcm_free, ggtts_free, ggtts_last_error
 *   2  ggtts_load_with and ggtts_settings (a Qwen3-TTS speaker and style instruction), ggtts_abi_version */
#ifndef GGTTS_H
#define GGTTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGTTS_ABI_VERSION 2

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

/* Voice settings for ggtts_load_with (version 2). Set size to sizeof(ggtts_settings); start from
 * GGTTS_SETTINGS_INIT so a field you do not set is NULL. Later versions accept any earlier sizeof(ggtts_settings) and
 * treat the fields it lacks as unset; a size this library does not know is GGTTS_ERR_ARG. Strings are UTF-8 and
 * NUL-terminated, owned by the caller and read only during the call. A NULL or empty string is no setting. Both are
 * Qwen3-TTS settings: another engine refuses either one with GGTTS_ERR_ARG.
 *   speaker      a speaker the model has, case ignored (for example "hk_voice_777"); NULL: the speaker ggtts_load
 *                uses ("hk_cantonese_speaker"). An unknown speaker is GGTTS_ERR_ENGINE, and the message lists the
 *                model's speakers.
 *   instruction  a style instruction for every chunk (for example "Speak slowly and softly."); NULL: none. A
 *                checkpoint that takes none (Base, or custom_voice below 1.7B) refuses it with GGTTS_ERR_ENGINE. */
typedef struct ggtts_settings {
    size_t size;
    const char *speaker;
    const char *instruction;
} ggtts_settings;

#define GGTTS_SETTINGS_INIT {sizeof(ggtts_settings), NULL, NULL}

/* ggtts_load with voice settings. settings may be NULL; with neither setting, the engine and its output are
 * ggtts_load's. A refused setting writes no handle. Example:
 *   ggtts_settings s = GGTTS_SETTINGS_INIT;
 *   s.speaker = "hk_voice_777";
 *   s.instruction = "Speak slowly and softly.";
 *   if (ggtts_load_with("qwen3-tts", talker, codec, NULL, 4, &s, &handle) != GGTTS_OK)
 *       fprintf(stderr, "%s\n", ggtts_last_error()); */
int32_t ggtts_load_with(const char *engine, const char *model, const char *codec, const char *voice, int32_t threads,
                        const ggtts_settings *settings, uint64_t *handle);
uint32_t ggtts_abi_version(void);
int32_t ggtts_render(uint64_t handle, const char *text, uint64_t seed, float **pcm, size_t *len, uint32_t *rate);
int32_t ggtts_render_stream(uint64_t handle, const char *text, uint64_t seed, ggtts_piece_fn on_piece, void *user);
void ggtts_pcm_free(float *pcm, size_t len);
int32_t ggtts_free(uint64_t handle);
const char *ggtts_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
