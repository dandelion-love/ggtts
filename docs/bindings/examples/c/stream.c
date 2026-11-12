/* stream.c: render piece by piece, stop early, and see the threading rules and error codes at work.
 *
 *   ./stream            Qwen3-TTS with its default model (it must already be in the cache: this never downloads)
 *   ./stream fake       the test engine
 *   ./stream qwen3-tts talker.gguf codec.gguf
 */
#include <stdio.h>

#include "ggtts.h"

struct listener {
    uint64_t tts;      /* the engine, so the callback can try to call back into it */
    int pieces;        /* pieces received so far */
    int stop_after;    /* ask to stop after this many pieces */
    int32_t reentered; /* what a render from inside the callback returned */
};

/* Called once per kept chunk, in order, on the thread that called ggtts_render_stream. The samples belong to the
 * library and are valid only during this call: copy them (or hand them to an audio device) before returning. */
static int32_t on_piece(void *user, const float *samples, size_t len, uint32_t rate) {
    struct listener *l = user;
    (void)samples;
    l->pieces++;
    printf("piece %d: %lu samples at %u Hz\n", l->pieces, (unsigned long)len, (unsigned)rate);

    /* One render at a time per engine: calling back into it from here is refused, never a deadlock. */
    float *pcm = NULL;
    size_t n = 0;
    uint32_t r = 0;
    l->reentered = ggtts_render(l->tts, "x", 1, &pcm, &n, &r);

    return l->pieces >= l->stop_after; /* 0: go on; anything else: stop */
}

int main(int argc, char **argv) {
    const char *engine = argc > 1 ? argv[1] : "qwen3-tts";
    const char *model = argc > 2 ? argv[2] : NULL;
    const char *codec = argc > 3 ? argv[3] : NULL;

    ggtts_settings settings = GGTTS_SETTINGS_INIT; /* allow_download stays 0: a missing default file is an error */
    uint64_t tts = 0;
    int32_t code = ggtts_load_with(engine, model, codec, NULL, 4, &settings, &tts);
    if (code != GGTTS_OK) {
        /* GGTTS_ERR_ENGINE here names a missing default file and its URL: run first_render once, or pass paths. */
        fprintf(stderr, "load failed (%d): %s\n", (int)code, ggtts_last_error());
        return 1;
    }

    /* Three paragraphs, so three or more pieces: 今日天氣好好。 / 我哋去海邊行下。 / 行完之後，去食飯。 */
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82\n"
                       "\xe6\x88\x91\xe5\x93\x8b\xe5\x8e\xbb\xe6\xb5\xb7\xe9\x82\x8a\xe8\xa1\x8c\xe4\xb8\x8b\xe3\x80\x82\n"
                       "\xe8\xa1\x8c\xe5\xae\x8c\xe4\xb9\x8b\xe5\xbe\x8c\xef\xbc\x8c\xe5\x8e\xbb\xe9\xa3\x9f\xe9\xa3\xaf"
                       "\xe3\x80\x82";
    struct listener l = {tts, 0, 2, GGTTS_OK};
    code = ggtts_render_stream(tts, text, 20260905, on_piece, &l);
    int ok = 1;
    if (code != GGTTS_ERR_STOPPED) {
        fprintf(stderr, "expected GGTTS_ERR_STOPPED after %d pieces, got %d: %s\n", l.stop_after, (int)code,
                ggtts_last_error());
        ok = 0;
    }
    if (l.reentered != GGTTS_ERR_BUSY) {
        fprintf(stderr, "expected GGTTS_ERR_BUSY from the callback, got %d\n", (int)l.reentered);
        ok = 0;
    }
    printf("stopped after %d pieces (code %d); a render from the callback returned %d (GGTTS_ERR_BUSY)\n", l.pieces,
           (int)code, (int)l.reentered);

    /* Text with nothing to say is an argument error, refused before rendering. */
    code = ggtts_render_stream(tts, "\xef\xbc\x9f\xef\xbc\x81", 1, on_piece, &l); /* "？！" */
    printf("punctuation only: %d (GGTTS_ERR_ARG): %s\n", (int)code, ggtts_last_error());
    ok &= code == GGTTS_ERR_ARG;

    /* A freed handle is never reused: every later call with it is GGTTS_ERR_HANDLE, not a crash. */
    ggtts_free(tts);
    code = ggtts_render_stream(tts, text, 1, on_piece, &l);
    printf("after ggtts_free: %d (GGTTS_ERR_HANDLE): %s\n", (int)code, ggtts_last_error());
    ok &= code == GGTTS_ERR_HANDLE;
    return ok ? 0 : 1;
}
