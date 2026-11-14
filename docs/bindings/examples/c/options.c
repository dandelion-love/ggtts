/* options.c: load and render with options documents, the JSON form of every option (C-ABI version 4).
 *
 *   ./options           the test engine
 *   ./options fake      the same
 */
#include <stdio.h>

#include "ggtts.h"

struct counter {
    int pieces;
};

static int32_t count(void *user, const float *samples, size_t len, uint32_t rate) {
    struct counter *c = user;
    (void)samples;
    (void)len;
    (void)rate;
    c->pieces++;
    return 0;
}

int main(int argc, char **argv) {
    (void)argv;
    if (argc > 1 && argv[1][0] != 'f') {
        fprintf(stderr, "this example runs on the fake engine only\n");
        return 1;
    }
    if (ggtts_abi_version() < 4) {
        fprintf(stderr, "this library is C-ABI version %u: the options documents need 4\n", ggtts_abi_version());
        return 1;
    }
    /* The load document: the load section, and render defaults every render on the handle starts from. */
    uint64_t tts = 0;
    int32_t code = ggtts_load_opts("{\"load\": {\"engine\": \"fake\"}, \"render\": {\"seed_salt\": \"7\"}}", &tts);
    if (code != GGTTS_OK) {
        fprintf(stderr, "load failed (%d): %s\n", (int)code, ggtts_last_error());
        return 1;
    }
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82"; /* 今日天氣好好。 */
    int ok = 1;

    /* A whole read: no output.stream, so on_piece is NULL and the samples come back in pcm. NULL is the empty
     * document: the load's defaults as they are. */
    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    code = ggtts_render_opts(tts, text, "{\"render\": {\"edge_s\": 0.2}}", NULL, NULL, &pcm, &len, &rate);
    printf("whole read: %d, %lu samples at %u Hz\n", (int)code, (unsigned long)len, (unsigned)rate);
    ok &= code == GGTTS_OK && len > 0;
    ggtts_pcm_free(pcm, len);

    /* A stream: output.stream "raw" hands each kept chunk to on_piece before the level step; pcm may be NULL. */
    struct counter c = {0};
    code = ggtts_render_opts(tts, text, "{\"output\": {\"stream\": \"raw\"}}", count, &c, NULL, NULL, NULL);
    printf("stream: %d, %d pieces\n", (int)code, c.pieces);
    ok &= code == GGTTS_OK && c.pieces > 0;

    /* Refused before anything is rendered, each GGTTS_ERR_ARG, the message naming the option. */
    const char *refused[] = {
        "{\"render\": {\"no_such_option\": 1}}",                 /* not an option */
        "{\"load\": {\"threads\": 2}}",                         /* a load option, in a render document */
        "{\"render\": {\"steer\": \"steer.safetensors\"}}",     /* the fake engine cannot steer */
        "{\"render\": ",                                        /* not JSON */
    };
    for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
        code = ggtts_render_opts(tts, text, refused[i], NULL, NULL, &pcm, &len, &rate);
        printf("refused: %d: %s\n", (int)code, ggtts_last_error());
        ok &= code == GGTTS_ERR_ARG;
    }
    ggtts_free(tts);
    return ok ? 0 : 1;
}
