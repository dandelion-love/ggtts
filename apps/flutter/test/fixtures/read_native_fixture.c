#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <string.h>

typedef int32_t (*piece_fn)(void *, const float *, size_t, uint32_t);
static atomic_int mode, stage, cancelled, pushes, paused_pushes, state, buffer_samples;
static atomic_ulong token_id, session_id, next_token, next_session;
void fake_mode(int m) { atomic_store(&mode,m); atomic_store(&stage,0); atomic_store(&pushes,0); atomic_store(&paused_pushes,0); }
int fake_stage(void) { return atomic_load(&stage); }
int fake_pushes(void) { return atomic_load(&pushes); }
int fake_buffer_samples(void) { return atomic_load(&buffer_samples); }
int fake_paused_pushes(void) { return atomic_load(&paused_pushes); }
#ifndef FAKE_ABI
#define FAKE_ABI 6
#endif
uint32_t ggtts_abi_version(void) { return FAKE_ABI; }
const char *ggtts_options_schema(void) { return "{}"; }
const char *ggtts_last_error(void) { return "fixture invalid options"; }
int32_t ggtts_load_opts(const char *doc, uint64_t *h) { (void)doc; *h=7; return 0; }
int32_t ggtts_free(uint64_t h) { (void)h; return 0; }
void ggtts_pcm_free(float *p, size_t n) { (void)n; free(p); }
void ggtts_string_free(char *p) { free(p); }
int32_t ggtts_capabilities(uint64_t h, char **p) { (void)h; *p=strdup(atomic_load(&mode)<2 ? "{\"cancellable\":true}" : "{}"); return 0; }
int32_t ggtts_render_opts(uint64_t h,const char*t,const char*d,piece_fn cb,void*u,float**p,size_t*n,uint32_t*r) { (void)h;(void)t;(void)d;(void)p;(void)n;(void)r; if(atomic_load(&mode)==3) return -1; float samples[2048]={0}; if(cb(u,samples,2048,24000)) return -5; if(atomic_load(&mode)==4 && cb(u,samples,2048,16000)) return -5; return 0; }
#if FAKE_ABI >= 6
int32_t ggtts_cancel_new(uint64_t *out) { uint64_t id=atomic_fetch_add(&next_token,1)+1; atomic_store(&cancelled,0); atomic_store(&token_id,id); *out=id; return 0; }
int32_t ggtts_cancel_signal(uint64_t id) { if (id!=atomic_load(&token_id)) return -2; atomic_store(&cancelled,1); return 0; }
int32_t ggtts_cancel_free(uint64_t id) { if (id!=atomic_load(&token_id)) return -2; atomic_store(&token_id,0); return 0; }
int32_t ggtts_render_opts_cancellable(uint64_t h,const char*t,const char*d,uint64_t id,piece_fn cb,void*u) {
  (void)h;(void)t;(void)d;
  if(id!=atomic_load(&token_id)) return -2;
  atomic_store(&stage,1);
  if(atomic_load(&mode)==0) {
    while(!atomic_load(&cancelled)) usleep(1000);
    return -5;
  }
  float samples[2048]={0};
  if(cb(u,samples,2048,24000)) return -5;
  return atomic_load(&cancelled) ? -5 : 0;
}
#endif
uint64_t ma_stream_session_new(void) {
  if(atomic_load(&session_id)!=0) return 0;
  uint64_t id=atomic_fetch_add(&next_session,1)+1;
  atomic_store(&state,0); atomic_store(&session_id,id); return id;
}
int64_t ma_stream_session_state(uint64_t id) { return id==atomic_load(&session_id) ? atomic_load(&state) : -1; }
int64_t ma_stream_session_init(uint64_t id,int64_t a,int64_t b,int64_t c,int64_t d,int64_t e) {
  (void)b;(void)c;(void)d;(void)e; atomic_store(&buffer_samples, (int)a);
  if(id!=atomic_load(&session_id)||atomic_load(&state)!=0) return -1;
  atomic_store(&state,1); return 0;
}
int64_t ma_stream_session_pause(uint64_t id) {
  if(id!=atomic_load(&session_id)||atomic_load(&state)!=1) return -1;
  atomic_store(&state,2); return 0;
}
int64_t ma_stream_session_resume(uint64_t id) {
  if(id!=atomic_load(&session_id)||atomic_load(&state)!=2) return -1;
  atomic_store(&state,1); return 0;
}
int64_t ma_stream_session_stop(uint64_t id) {
  if(id!=atomic_load(&session_id)) return -1;
  atomic_store(&state,3); return 0;
}
int64_t ma_stream_session_uninit(uint64_t id) {
  if(id!=atomic_load(&session_id)) return -1;
  atomic_store(&session_id,0); return 0;
}
int64_t ma_stream_session_software_empty(uint64_t id) {
  return id==atomic_load(&session_id)&&atomic_load(&state)==1 ? 1 : -1;
}
int64_t ma_stream_session_push(uint64_t id,float *p,int64_t n) {
  (void)p;(void)n;
  if(id!=atomic_load(&session_id)||atomic_load(&state)==3) return -3;
  if(atomic_load(&state)==2) { atomic_fetch_add(&paused_pushes,1); return -2; }
  atomic_fetch_add(&pushes,1); return atomic_load(&mode)>=2 ? 0 : -1;
}
