#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int32_t (*piece_fn)(void *, const float *, size_t, uint32_t);
uint32_t ggtts_abi_version(void) { return 6; }
const char *ggtts_options_schema(void) { return "{\"schema\":\"fixture\"}"; }
const char *ggtts_last_error(void) { return "stopped"; }
int32_t ggtts_load_opts(const char *doc, uint64_t *h) { (void)doc; *h = 7; return 0; }
int32_t ggtts_free(uint64_t h) { (void)h; return 0; }
void ggtts_pcm_free(float *pcm, size_t n) { (void)n; free(pcm); }
void ggtts_string_free(char *s) { free(s); }
int32_t ggtts_capabilities(uint64_t h, char **out) {
  (void)h; const char *s = "{\"speakers\":true,\"speaker_names\":[\"fixture\"]}";
  *out = malloc(strlen(s)+1); strcpy(*out, s); return 0;
}
int32_t ggtts_render_opts(uint64_t h, const char *text, const char *doc, piece_fn cb,
                          void *user, float **pcm, size_t *len, uint32_t *rate) {
  (void)h; (void)text;
  if (!cb || user || pcm || len || rate) return -1;
  float samples[] = {0.25f, -0.5f};
  if (doc && strstr(doc, "bad_rate")) return cb(user, samples, 2, 0) ? -5 : 0;
  if (doc && strstr(doc, "bad_length")) return cb(user, NULL, 1, 24000) ? -5 : 0;
  if (doc && strstr(doc, "bad_huge")) return cb(user, samples, (size_t)0x80000000, 24000) ? -5 : 0;
  if (cb(user, samples, 2, 24000)) return -5;
  samples[0] = 0.75f; samples[1] = 1.0f;
  if (cb(user, samples, 2, 24000)) return -5;
  return 0;
}

static uint64_t fixture_token = 0;
static int fixture_cancelled = 0;
int32_t ggtts_cancel_new(uint64_t *out) { fixture_cancelled = 0; *out = ++fixture_token; return 0; }
int32_t ggtts_cancel_signal(uint64_t id) { if (id != fixture_token) return -2; fixture_cancelled = 1; return 0; }
int32_t ggtts_cancel_free(uint64_t id) { if (id != fixture_token) return -2; fixture_token = 0; return 0; }
int32_t ggtts_render_opts_cancellable(uint64_t h, const char *text, const char *doc, uint64_t token, piece_fn cb, void *user) {
  if (token != fixture_token) return -2;
  if (fixture_cancelled) return -5;
  return ggtts_render_opts(h,text,doc,cb,user,NULL,NULL,NULL);
}
