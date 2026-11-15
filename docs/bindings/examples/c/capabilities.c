/* capabilities.c: what the library and a loaded engine take, read from the library (C-ABI version 5).
 *
 *   ./capabilities           the test engine
 *   ./capabilities fake      the same
 */
#include <stdio.h>
#include <string.h>

#include "ggtts.h"

int main(int argc, char **argv) {
    (void)argv;
    if (argc > 1 && argv[1][0] != 'f') {
        fprintf(stderr, "this example runs on the fake engine only\n");
        return 1;
    }
    if (ggtts_abi_version() < 5) {
        fprintf(stderr, "this library is C-ABI version %u: introspection needs 5\n", ggtts_abi_version());
        return 1;
    }
    int ok = 1;

    /* The options schema, as built into the library: static, never freed. */
    const char *schema = ggtts_options_schema();
    printf("options schema: %lu bytes\n", (unsigned long)strlen(schema));
    ok &= strstr(schema, "\"$schema\"") != NULL;

    uint64_t tts = 0;
    if (ggtts_load_opts("{\"load\": {\"engine\": \"fake\"}}", &tts) != GGTTS_OK) {
        fprintf(stderr, "load failed: %s\n", ggtts_last_error());
        return 1;
    }

    /* The loaded engine's capability statement, as JSON: free it with ggtts_string_free. */
    char *caps = NULL;
    int32_t code = ggtts_capabilities(tts, &caps);
    printf("capabilities: %d: %s\n", (int)code, caps ? caps : "(none)");
    ok &= code == GGTTS_OK && caps != NULL && strstr(caps, "\"family\":\"fake\"") != NULL;
    ggtts_string_free(caps);

    /* A refused option names its key, so a binding needs no message parsing; another error leaves it NULL. */
    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    const char *text = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xe3\x80\x82"; /* 今日天氣好好。 */
    code = ggtts_render_opts(tts, text, "{\"render\": {\"steer\": \"steer.safetensors\"}}", NULL, NULL, &pcm, &len,
                             &rate);
    const char *key = ggtts_last_error_key();
    printf("refused: %d, key %s\n", (int)code, key ? key : "(none)");
    ok &= code == GGTTS_ERR_ARG && key != NULL && strcmp(key, "render.steer") == 0;
    code = ggtts_render_opts(tts, NULL, NULL, NULL, NULL, &pcm, &len, &rate);
    key = ggtts_last_error_key();
    printf("no text: %d, key %s\n", (int)code, key ? key : "(none)");
    ok &= code == GGTTS_ERR_ARG && key == NULL;

    ggtts_free(tts);
    /* A freed handle has no capabilities. */
    caps = NULL;
    ok &= ggtts_capabilities(tts, &caps) == GGTTS_ERR_HANDLE && caps == NULL;
    return ok ? 0 : 1;
}
