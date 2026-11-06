// Reference harness: layer 0 of the Qwen3-TTS talker, built the way qwen3-tts.cpp builds it.
//
// Source: https://github.com/Danmoreng/qwen3-tts.cpp at 16bb5afcd06311031c72a8488f8d59660dc2fb46 (MIT,
// Copyright (c) the qwen3-tts.cpp authors). Every graph op below is transcribed, in order, from
//   src/transformer/transformer_graph_talker.cpp:39-172   build_prefill_forward_graph (loop body, il = 0)
// and every input is filled as in
//   src/transformer/transformer_runtime.cpp:101-205       TTSTransformer prefill: n_ctx, positions, M-RoPE, mask
// Weights are loaded as in
//   src/gguf_loader.cpp:303-430                           load_tensor_data_from_file (CPU buffer, fread at offsets)
//   src/transformer/transformer_loader.cpp:266            ggml_backend_sched_new(..., QWEN3_TTS_MAX_NODES, false, true)
//   src/transformer/transformer_cache.cpp:84-140          init_kv_cache (F16, [head_dim, n_kv_head, n_ctx])
// The model's own metadata keys are read as in src/transformer/transformer_loader_metadata.cpp:109-265.
//
// The layer's output is inpL after the loop body (transformer_graph_talker.cpp:170), the residual stream that
// feeds layer 1. It is named "layer0_out" and written as raw little-endian f32, [hidden_size, n_tokens].
//
// Usage: layer0_ref MODEL.gguf OUT.bin [--threads N] [--tokens T] [--seed S] [--iters N] [--warmup N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int MAX_NODES = 16384; // QWEN3_TTS_MAX_NODES, transformer_state_internal.h:75

static void die(const char * msg) {
    fprintf(stderr, "layer0_ref: %s\n", msg);
    exit(1);
}

// splitmix64; each value is a 24-bit integer scaled into [-1, 1), so it is exact in f32 on every platform.
static void seeded_input(uint64_t seed, std::vector<float> & out) {
    uint64_t s = seed;
    for (auto & v : out) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const int32_t r = (int32_t) (z >> 40) - (1 << 23);
        v = (float) r / (float) (1 << 23);
    }
}

static uint32_t get_u32(gguf_context * g, const char * key) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) die(key);
    return gguf_get_val_u32(g, i);
}

static float get_f32(gguf_context * g, const char * key) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) die(key);
    return gguf_get_val_f32(g, i);
}

struct Layer {
    ggml_tensor * attn_norm, * attn_q, * attn_k, * attn_v, * attn_q_norm, * attn_k_norm, * attn_output;
    ggml_tensor * ffn_norm, * ffn_gate, * ffn_up, * ffn_down;
};

struct Hparams {
    int hidden_size, n_head, n_kv_head, head_dim;
    float eps, rope_theta;
    int mrope_section[3];
};

static ggml_cgraph * build_layer0(ggml_context * ctx0, const Hparams & hp, const Layer & layer,
                                  ggml_tensor * k_cache, ggml_tensor * v_cache, int n_ctx, int n_tokens,
                                  int n_past) {
    const int n_head = hp.n_head, n_kv_head = hp.n_kv_head, head_dim = hp.head_dim;
    const float eps = hp.eps, rope_theta = hp.rope_theta;
    const int n_kv_pad = std::min<int>(n_ctx, GGML_PAD(n_past + n_tokens, 256)); // :49

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, MAX_NODES, false); // :58

    ggml_tensor * inp_prefill_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.hidden_size, n_tokens); // :60
    ggml_set_name(inp_prefill_embd, "inp_prefill_embd");
    ggml_set_input(inp_prefill_embd);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens); // :64
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_tokens); // :70 (use_mrope)
    ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
    ggml_set_input(inp_mrope_pos);

    ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv_pad, n_tokens); // :75
    ggml_set_name(inp_mask, "inp_mask");
    ggml_set_input(inp_mask);

    ggml_tensor * cur = inp_prefill_embd; // :79
    ggml_tensor * inpL = cur;

    int mrope_sections[GGML_MROPE_SECTIONS] = { hp.mrope_section[0], hp.mrope_section[1], hp.mrope_section[2], 0 }; // :82

    cur = ggml_rms_norm(ctx0, inpL, eps); // :87
    cur = ggml_mul(ctx0, cur, layer.attn_norm);

    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur); // :90
    ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

    Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens); // :94
    Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

    Qcur = ggml_rms_norm(ctx0, Qcur, eps); // :99
    Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);

    Kcur = ggml_rms_norm(ctx0, Kcur, eps); // :104
    Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_mrope_pos, nullptr, // :109
                           head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                           rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Kcur = ggml_rope_multi(ctx0, Kcur, inp_mrope_pos, nullptr, // :113
                           head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                           rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * k_cache_2d = ggml_view_2d(ctx0, k_cache, head_dim * n_kv_head, n_ctx, k_cache->nb[2], 0); // :129
    ggml_tensor * v_cache_2d = ggml_view_2d(ctx0, v_cache, head_dim * n_kv_head, n_ctx, v_cache->nb[2], 0);

    ggml_tensor * Kcur_2d = ggml_view_2d(ctx0, Kcur, head_dim * n_kv_head, n_tokens, Kcur->nb[2], 0); // :132
    ggml_tensor * Vcur_2d = ggml_view_2d(ctx0, Vcur, head_dim * n_kv_head, n_tokens, Vcur->nb[2], 0);

    ggml_tensor * k_updated = ggml_set_rows(ctx0, k_cache_2d, Kcur_2d, inp_pos); // :135
    ggml_tensor * v_updated = ggml_set_rows(ctx0, v_cache_2d, Vcur_2d, inp_pos);

    ggml_build_forward_expand(gf, k_updated); // :138
    ggml_build_forward_expand(gf, v_updated);

    ggml_tensor * K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_kv_pad, k_cache->nb[1], k_cache->nb[2], 0); // :141
    ggml_tensor * V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_kv_pad, v_cache->nb[1], v_cache->nb[2], 0);

    ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3); // :149
    K = ggml_permute(ctx0, K, 0, 2, 1, 3);
    V = ggml_permute(ctx0, V, 0, 2, 1, 3);

    ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, inp_mask, 1.0f / sqrtf((float) head_dim), 0.0f, 0.0f); // :153
    ggml_flash_attn_ext_set_prec(KQV_fa, GGML_PREC_F32);
    cur = ggml_cont_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);

    cur = ggml_mul_mat(ctx0, layer.attn_output, cur); // :158
    cur = ggml_add(ctx0, cur, inpL);
    ggml_tensor * inpFF = cur;

    cur = ggml_rms_norm(ctx0, inpFF, eps); // :162
    cur = ggml_mul(ctx0, cur, layer.ffn_norm);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur); // :165
    ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

    cur = ggml_swiglu_split(ctx0, gate, up); // :168

    // ffn_down_matmul, :28-35: an F16 weight is cast to F32 first; the published talker is Q8_0.
    ggml_tensor * w_down = layer.ffn_down;
    if (w_down->type == GGML_TYPE_F16) {
        w_down = ggml_cast(ctx0, w_down, GGML_TYPE_F32);
    }
    cur = ggml_mul_mat(ctx0, w_down, cur); // :170

    inpL = ggml_add(ctx0, cur, inpFF); // :172

    ggml_set_name(inpL, "layer0_out");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);
    return gf;
}

int main(int argc, char ** argv) {
    if (argc < 3) die("usage: layer0_ref MODEL.gguf OUT.bin [--threads N] [--tokens T] [--seed S] [--iters N] [--warmup N]");
    const char * model_path = argv[1];
    const char * out_path = argv[2];
    int n_threads = 4, n_tokens = 16, iters = 0, warmup = 10;
    uint64_t seed = 20261106;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--threads") n_threads = atoi(argv[i + 1]);
        else if (a == "--tokens") n_tokens = atoi(argv[i + 1]);
        else if (a == "--seed") seed = strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--iters") iters = atoi(argv[i + 1]);
        else if (a == "--warmup") warmup = atoi(argv[i + 1]);
        else die("unknown flag");
    }

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) die("cannot open GGUF");

    Hparams hp;
    hp.hidden_size = (int) get_u32(g, "qwen3-tts.talker.embedding_length");
    hp.n_head = (int) get_u32(g, "qwen3-tts.talker.attention.head_count");
    hp.n_kv_head = (int) get_u32(g, "qwen3-tts.talker.attention.head_count_kv");
    hp.head_dim = (int) get_u32(g, "qwen3-tts.talker.attention.key_length");
    hp.eps = get_f32(g, "qwen3-tts.talker.attention.layer_norm_rms_epsilon");
    hp.rope_theta = get_f32(g, "qwen3-tts.talker.rope.freq_base");
    {
        const int64_t i = gguf_find_key(g, "qwen3-tts.talker.rope.mrope_section");
        if (i < 0) die("this harness covers the M-RoPE talker only");
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g, i);
        for (int k = 0; k < 3; ++k) hp.mrope_section[k] = d[k];
    }

    // Weights: layer 0 only, in a CPU buffer, read at their file offsets.
    static const char * names[11] = {
        "attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output",
        "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
    };
    ggml_init_params wp = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * t[11];
    for (int k = 0; k < 11; ++k) {
        std::string name = std::string("talker.blk.0.") + names[k] + ".weight";
        ggml_tensor * m = ggml_get_tensor(meta, name.c_str());
        if (!m) die(name.c_str());
        t[k] = ggml_new_tensor(wctx, m->type, ggml_n_dims(m), m->ne);
        ggml_set_name(t[k], name.c_str());
    }
    Layer layer = { t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10] };

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) die("no CPU backend");
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        FILE * f = fopen(model_path, "rb");
        if (!f) die("cannot open model for reading");
        const size_t data_offset = gguf_get_data_offset(g);
        std::vector<uint8_t> buf;
        for (int k = 0; k < 11; ++k) {
            const int64_t id = gguf_find_tensor(g, ggml_get_name(t[k]));
            const size_t nbytes = gguf_get_tensor_size(g, id);
            if (nbytes != ggml_nbytes(t[k])) die("tensor size mismatch");
            buf.resize(nbytes);
            if (fseeko(f, (off_t) (data_offset + gguf_get_tensor_offset(g, id)), SEEK_SET) != 0) die("seek");
            if (fread(buf.data(), 1, nbytes, f) != nbytes) die("read");
            ggml_backend_tensor_set(t[k], buf.data(), 0, nbytes);
        }
        fclose(f);
    }

    // KV cache: n_ctx as the runtime picks it for a fresh prefill (transformer_runtime.cpp:101-103).
    const int n_ctx = std::max<int>(256, n_tokens + 16);
    ggml_init_params cp = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * cctx = ggml_init(cp);
    ggml_tensor * k_cache = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, hp.head_dim, hp.n_kv_head, n_ctx);
    ggml_tensor * v_cache = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, hp.head_dim, hp.n_kv_head, n_ctx);
    ggml_set_name(k_cache, "k_cache_0");
    ggml_set_name(v_cache, "v_cache_0");
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    ggml_backend_buffer_clear(cbuf, 0);

    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, MAX_NODES, false, true);

    std::vector<uint8_t> compute_meta(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false));

    std::vector<float> input((size_t) hp.hidden_size * n_tokens);
    seeded_input(seed, input);
    std::vector<float> output((size_t) hp.hidden_size * n_tokens);

    const int n_past = 0;
    auto forward = [&]() {
        ggml_init_params p = { compute_meta.size(), compute_meta.data(), true };
        ggml_context * ctx0 = ggml_init(p);
        ggml_cgraph * gf = build_layer0(ctx0, hp, layer, k_cache, v_cache, n_ctx, n_tokens, n_past);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) die("alloc graph");

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_prefill_embd"), input.data(), 0, input.size() * sizeof(float));

        std::vector<int32_t> pos(n_tokens); // transformer_runtime.cpp:150-161
        for (int i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

        std::vector<int32_t> mpos((size_t) n_tokens * 4); // :164-180
        for (int i = 0; i < n_tokens; ++i) {
            const int32_t p = n_past + i;
            mpos[i] = p;
            mpos[i + n_tokens] = p;
            mpos[i + n_tokens * 2] = p;
            mpos[i + n_tokens * 3] = 0;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_mrope_pos"), mpos.data(), 0, mpos.size() * sizeof(int32_t));

        ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask"); // :183-205
        const int32_t n_kv_pad = (int32_t) inp_mask->ne[0];
        std::vector<ggml_fp16_t> mask((size_t) n_tokens * n_kv_pad, ggml_fp32_to_fp16(-INFINITY));
        for (int32_t q = 0; q < n_tokens; ++q) {
            const int32_t last_k = std::min<int32_t>(n_past + q, n_kv_pad - 1);
            for (int32_t k = 0; k <= last_k; ++k) mask[(size_t) q * n_kv_pad + k] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) die("compute");
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer0_out"), output.data(), 0, output.size() * sizeof(float));
        ggml_backend_sched_reset(sched);
        ggml_free(ctx0);
    };

    forward();
    FILE * fo = fopen(out_path, "wb");
    if (!fo || fwrite(output.data(), sizeof(float), output.size(), fo) != output.size()) die("write output");
    fclose(fo);
    fprintf(stderr, "layer0_ref: wrote %zu bytes (%d tokens x %d), threads=%d seed=%llu\n",
            output.size() * sizeof(float), n_tokens, hp.hidden_size, n_threads, (unsigned long long) seed);

    if (iters > 0) {
        for (int i = 0; i < warmup; ++i) forward();
        std::vector<double> ms(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            forward();
            ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        double sum = 0;
        for (double v : ms) sum += v;
        std::sort(ms.begin(), ms.end());
        printf("timing lang=cpp threads=%d tokens=%d iters=%d mean_ms=%.4f median_ms=%.4f min_ms=%.4f\n",
               n_threads, n_tokens, iters, sum / iters, ms[iters / 2], ms[0]);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cctx);
    ggml_free(wctx);
    ggml_backend_free(backend);
    gguf_free(g);
    ggml_free(meta);
    return 0;
}
