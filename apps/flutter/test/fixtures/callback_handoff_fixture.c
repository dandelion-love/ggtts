#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
typedef int32_t (*piece_fn)(void*,const float*,size_t,uint32_t);
static float samples[32];
static int sample_count, init_count, release_count, first_handed_off, pushes;
static uint64_t session;
void fixture_reset(void) { sample_count=init_count=release_count=first_handed_off=pushes=0; session=0; }
int fixture_count(void) { return sample_count; }
float fixture_sample(int n) { return n>=0 && n<sample_count ? samples[n] : -999; }
int fixture_init_count(void) { return init_count; }
int fixture_release_count(void) { return release_count; }
int fixture_first_handed_off(void) { return first_handed_off; }
uint32_t ggtts_abi_version(void) { return 5; }
const char *ggtts_options_schema(void) { return "{}"; }
const char *ggtts_last_error(void) { return "fixture"; }
int32_t ggtts_load_opts(const char *doc,uint64_t *h) { (void)doc; *h=1; return 0; }
int32_t ggtts_free(uint64_t h) { (void)h; return 0; }
void ggtts_pcm_free(float *p,size_t n) { (void)n; free(p); }
void ggtts_string_free(char *p) { free(p); }
int32_t ggtts_capabilities(uint64_t h,char **p) { (void)h; *p=strdup("{}"); return 0; }
int32_t ggtts_render_opts(uint64_t h,const char*t,const char*d,piece_fn cb,void*u,float**p,size_t*n,uint32_t*r) {
 (void)h;(void)t;(void)p;(void)n;(void)r;
 const float a[]={0.25f,-0.5f}, b[]={0.75f,1.0f};
 if(cb(u,a,2,24000)) return -5;
 first_handed_off=(sample_count==2);
 if(cb(u,b,2,d && strstr(d,"bad_rate") ? 32000 : 24000)) return -5;
 return 0;
}
int64_t ma_stream_init(int64_t a,int64_t b,int64_t c,int64_t d,int64_t e) { (void)a;(void)b;(void)c;(void)d;(void)e; init_count++; return 0; }
int64_t ma_stream_push(float *p,int64_t n) {
 pushes++;
#ifdef PLANT_DROP_SECOND
 if(pushes==2) return 0;
#endif
 if(n<0 || sample_count+n>32) return -1;
 memcpy(samples+sample_count,p,n*sizeof(float)); sample_count+=(int)n; return 0;
}
void ma_stream_uninit(void) { release_count++; session=0; }
int64_t ma_stream_stat_exhaust_count(void) { return 0; }
int64_t ma_stream_stat_full_count(void) { return 0; }
void ma_stream_stat_reset(void) {}
uint64_t ma_stream_session_new(void) { session=1; return session; }
int64_t ma_stream_session_init(uint64_t id,int64_t a,int64_t b,int64_t c,int64_t d,int64_t e) { if(id!=session) return -1; return ma_stream_init(a,b,c,d,e); }
int64_t ma_stream_session_push(uint64_t id,float*p,int64_t n) { return id==session ? ma_stream_push(p,n) : -3; }
int64_t ma_stream_session_pause(uint64_t id) { return id==session?0:-1; }
int64_t ma_stream_session_resume(uint64_t id) { return id==session?0:-1; }
int64_t ma_stream_session_stop(uint64_t id) { return id==session?0:-1; }
int64_t ma_stream_session_uninit(uint64_t id) { if(id!=session) return -1; ma_stream_uninit(); return 0; }
int64_t ma_stream_session_software_empty(uint64_t id) { return id==session?1:-1; }
int64_t ma_stream_session_state(uint64_t id) { return id==session?1:-1; }
