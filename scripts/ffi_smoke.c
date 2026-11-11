/* The C library smoke: built by scripts/package_and_smoke.sh against a packed release's include/ggtts.h and its
 * library (lib/), and run with only that library on the loader's path. It checks that the library loaded is the
 * header's ABI version, then loads the fake engine through ggtts_load_with and renders one line through ggtts_render.
 * Any failure prints a line starting "ffi smoke: FAILED" and exits 1. The version is the header's GGTTS_ABI_VERSION,
 * never a number written here, so the smoke follows the header as it moves. */
#include <stdio.h>

#include "ggtts.h"

/* "今日天氣好好，我哋去公園行下。" in UTF-8, escaped so the compiler's source encoding cannot change it. */
static const char TEXT[] = "\xe4\xbb\x8a\xe6\x97\xa5\xe5\xa4\xa9\xe6\xb0\xa3\xe5\xa5\xbd\xe5\xa5\xbd\xef\xbc\x8c"
                           "\xe6\x88\x91\xe5\x93\x8b\xe5\x8e\xbb\xe5\x85\xac\xe5\x9c\x92\xe8\xa1\x8c\xe4\xb8\x8b"
                           "\xe3\x80\x82";

int main(void) {
    uint32_t loaded = ggtts_abi_version();
    if (loaded != (uint32_t)GGTTS_ABI_VERSION) {
        fprintf(stderr, "ffi smoke: FAILED: the library's ggtts_abi_version() is %u, the header's GGTTS_ABI_VERSION is %u\n",
                (unsigned)loaded, (unsigned)GGTTS_ABI_VERSION);
        return 1;
    }
    ggtts_settings settings = GGTTS_SETTINGS_INIT;
    uint64_t handle = 0;
    int32_t code = ggtts_load_with("fake", NULL, NULL, NULL, 1, &settings, &handle);
    if (code != GGTTS_OK || handle == 0) {
        fprintf(stderr, "ffi smoke: FAILED: ggtts_load_with(\"fake\") returned %d: %s\n", (int)code, ggtts_last_error());
        return 1;
    }
    float *pcm = NULL;
    size_t len = 0;
    uint32_t rate = 0;
    code = ggtts_render(handle, TEXT, 7, &pcm, &len, &rate);
    if (code != GGTTS_OK || pcm == NULL || rate == 0 || len < rate) {
        fprintf(stderr, "ffi smoke: FAILED: ggtts_render returned %d with %lu samples at %u Hz: %s\n", (int)code,
                (unsigned long)len, (unsigned)rate, ggtts_last_error());
        return 1;
    }
    ggtts_pcm_free(pcm, len);
    code = ggtts_free(handle);
    if (code != GGTTS_OK) {
        fprintf(stderr, "ffi smoke: FAILED: ggtts_free returned %d: %s\n", (int)code, ggtts_last_error());
        return 1;
    }
    printf("ffi smoke: ok: ABI version %u, the fake engine rendered %lu samples at %u Hz\n", (unsigned)loaded,
           (unsigned long)len, (unsigned)rate);
    return 0;
}
