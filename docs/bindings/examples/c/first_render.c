/* first_render.c: load an engine, render one line, write it to a WAV file.
 *
 *   ./first_render                                  Qwen3-TTS with its default model (downloaded once, 2.33 GB)
 *   ./first_render qwen3-tts talker.gguf codec.gguf  Qwen3-TTS with model files you already have
 *   ./first_render kitten-tts-2 lm.gguf s3gen.gguf voice.json
 *   ./first_render fake                             the test engine: no files, a tone instead of speech
 *
 * It writes first_render.wav. */
#include <stdio.h>
#include <stdlib.h>

#include "ggtts.h"

/* Write mono float samples as a 16-bit PCM WAV file. Returns 0 on success. */
static int write_wav(const char *path, const float *samples, size_t len, uint32_t rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data = (uint32_t)(len * 2);
    unsigned char h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                           16, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 16, 0,
                           'd', 'a', 't', 'a', 0, 0, 0, 0};
    uint32_t fields[4][2] = {{4, 36 + data}, {24, rate}, {28, rate * 2}, {40, data}};
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++) h[fields[i][0] + b] = (unsigned char)(fields[i][1] >> (8 * b));
    fwrite(h, 1, sizeof h, f);
    for (size_t i = 0; i < len; i++) {
        float x = samples[i] > 1.0f ? 1.0f : samples[i] < -1.0f ? -1.0f : samples[i];
        int16_t s = (int16_t)(x * 32767.0f);
        unsigned char le[2] = {(unsigned char)(s & 0xff), (unsigned char)((s >> 8) & 0xff)};
        fwrite(le, 1, 2, f);
    }
    return fclose(f) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *engine = argc > 1 ? argv[1] : "qwen3-tts";
    const char *model = argc > 2 ? argv[2] : NULL; /* NULL: the default model (qwen3-tts only) */
    const char *codec = argc > 3 ? argv[3] : NULL;
    const char *voice = argc > 4 ? argv[4] : NULL; /* kitten-tts-2's prepared reference voice */

    /* The settings below are ABI version 3. A library older than this header refuses them, so say so plainly. */
    if (ggtts_abi_version() < 3) {
        fprintf(stderr, "this libggtts_ffi is ABI version %u; this program needs 3\n", (unsigned)ggtts_abi_version());
        return 1;
    }

    ggtts_settings settings = GGTTS_SETTINGS_INIT; /* every field unset: CPU, no speaker, no download */
    settings.allow_download = 1;                   /* fetch the default model files if the cache lacks them */

    uint64_t tts = 0;
    if (ggtts_load_with(engine, model, codec, voice, 4, &settings, &tts) != GGTTS_OK) {
        fprintf(stderr, "load failed: %s\n", ggtts_last_error());
        return 1;
    }

    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    /* UTF-8 text; one paragraph per line. 20260905 is the seed the other bindings use by default. */
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82"; /* 今日天氣好好。 */
    if (ggtts_render(tts, text, 20260905, &pcm, &len, &rate) != GGTTS_OK) {
        fprintf(stderr, "render failed: %s\n", ggtts_last_error());
        ggtts_free(tts);
        return 1;
    }

    int written = write_wav("first_render.wav", pcm, len, rate);
    printf("%lu samples at %u Hz (%.2f s)%s\n", (unsigned long)len, (unsigned)rate, (double)len / rate,
           written == 0 ? ", written to first_render.wav" : "; could not write first_render.wav");

    ggtts_pcm_free(pcm, len); /* the buffer is the library's: free it with ggtts_pcm_free, never free() */
    ggtts_free(tts);
    return written == 0 ? 0 : 1;
}
