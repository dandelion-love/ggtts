//! Layer 0 of the Qwen3-TTS talker, built in Rust through the raw ggml bindings.
//!
//! The graph is the one qwen3-tts.cpp builds, op for op and in the same order. Citations are to
//! https://github.com/Danmoreng/qwen3-tts.cpp at 16bb5afcd06311031c72a8488f8d59660dc2fb46 (MIT):
//!   src/transformer/transformer_graph_talker.cpp  build_prefill_forward_graph, loop body for il = 0
//!   src/transformer/transformer_runtime.cpp       the prefill inputs: n_ctx, positions, M-RoPE positions, mask
//!   src/gguf_loader.cpp                           weights in a CPU buffer, read at their file offsets
//!   src/transformer/transformer_cache.cpp         the F16 KV cache
//! `reference/layer0_ref.cpp` is the C++ side of the comparison and carries the same citations.
//!
//! Usage: layer0 MODEL.gguf OUT.bin [--threads N] [--tokens T] [--seed S] [--iters N] [--warmup N] [--plant NAME]
//!
//! `--plant` builds a deliberately wrong graph, so the comparison can be shown to turn red:
//!   rope-neox     ggml_rope_ext with GGML_ROPE_TYPE_NEOX in place of ggml_rope_multi with GGML_ROPE_TYPE_MROPE
//!   rope-normal   ggml_rope_ext with GGML_ROPE_TYPE_NORMAL (adjacent pairs) in place of the M-RoPE
//!   swap-gate-up  ggml_swiglu_split(up, gate) in place of ggml_swiglu_split(gate, up)
//!   norm-order    the attention norm weight applied before ggml_rms_norm rather than after it

use ggml_sys as g;
use std::ffi::{c_void, CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ptr;
use std::time::Instant;

const MAX_NODES: usize = 16384; // QWEN3_TTS_MAX_NODES, transformer_state_internal.h:75

#[derive(Clone, Copy, PartialEq)]
enum Plant {
    None,
    RopeNeox,
    RopeNormal,
    SwapGateUp,
    NormOrder,
}

struct Hparams {
    hidden_size: i64,
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    eps: f32,
    rope_theta: f32,
    mrope_section: [i32; 3],
}

struct Layer {
    attn_norm: *mut g::ggml_tensor,
    attn_q: *mut g::ggml_tensor,
    attn_k: *mut g::ggml_tensor,
    attn_v: *mut g::ggml_tensor,
    attn_q_norm: *mut g::ggml_tensor,
    attn_k_norm: *mut g::ggml_tensor,
    attn_output: *mut g::ggml_tensor,
    ffn_norm: *mut g::ggml_tensor,
    ffn_gate: *mut g::ggml_tensor,
    ffn_up: *mut g::ggml_tensor,
    ffn_down: *mut g::ggml_tensor,
}

fn die(msg: &str) -> ! {
    eprintln!("layer0: {msg}");
    std::process::exit(1);
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// splitmix64; each value is a 24-bit integer scaled into [-1, 1), exact in f32. Same as the C++ harness.
fn seeded_input(seed: u64, out: &mut [f32]) {
    let mut s = seed;
    for v in out.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let r = (z >> 40) as i32 - (1 << 23);
        *v = r as f32 / (1 << 23) as f32;
    }
}

unsafe fn get_u32(gg: *mut g::gguf_context, key: &str) -> i64 {
    let i = g::gguf_find_key(gg, cstr(key).as_ptr());
    if i < 0 {
        die(key);
    }
    g::gguf_get_val_u32(gg, i) as i64
}

unsafe fn get_f32(gg: *mut g::gguf_context, key: &str) -> f32 {
    let i = g::gguf_find_key(gg, cstr(key).as_ptr());
    if i < 0 {
        die(key);
    }
    g::gguf_get_val_f32(gg, i)
}

unsafe fn named_input(ctx: *mut g::ggml_context, t: *mut g::ggml_tensor, name: &CStr) -> *mut g::ggml_tensor {
    g::ggml_set_name(t, name.as_ptr());
    g::ggml_set_input(t);
    let _ = ctx;
    t
}

/// transformer_graph_talker.cpp:39-172, il = 0, use_mrope = true. Line numbers in the comments.
#[allow(clippy::too_many_arguments)]
unsafe fn build_layer0(
    ctx0: *mut g::ggml_context,
    hp: &Hparams,
    layer: &Layer,
    k_cache: *mut g::ggml_tensor,
    v_cache: *mut g::ggml_tensor,
    n_ctx: i64,
    n_tokens: i64,
    n_past: i64,
    plant: Plant,
) -> *mut g::ggml_cgraph {
    let (n_head, n_kv_head, head_dim) = (hp.n_head, hp.n_kv_head, hp.head_dim);
    let (eps, rope_theta) = (hp.eps, hp.rope_theta);
    let n_kv_pad = n_ctx.min(g::pad_to(n_past + n_tokens, 256)); // :49

    let gf = g::ggml_new_graph_custom(ctx0, MAX_NODES, false); // :58

    let inp_prefill_embd = named_input(
        ctx0,
        g::ggml_new_tensor_2d(ctx0, g::GGML_TYPE_F32, hp.hidden_size, n_tokens),
        c"inp_prefill_embd",
    ); // :60
    let inp_pos = named_input(
        ctx0,
        g::ggml_new_tensor_1d(ctx0, g::GGML_TYPE_I32, n_tokens),
        c"inp_pos",
    ); // :64
    let inp_mrope_pos = named_input(
        ctx0,
        g::ggml_new_tensor_1d(ctx0, g::GGML_TYPE_I32, 4 * n_tokens),
        c"inp_mrope_pos",
    ); // :70
    let inp_mask = named_input(
        ctx0,
        g::ggml_new_tensor_2d(ctx0, g::GGML_TYPE_F16, n_kv_pad, n_tokens),
        c"inp_mask",
    ); // :75

    let mut cur;
    let inp_l = inp_prefill_embd; // :79-80

    let mut mrope_sections: [i32; g::GGML_MROPE_SECTIONS as usize] =
        [hp.mrope_section[0], hp.mrope_section[1], hp.mrope_section[2], 0]; // :82

    if plant == Plant::NormOrder {
        cur = g::ggml_mul(ctx0, inp_l, layer.attn_norm);
        cur = g::ggml_rms_norm(ctx0, cur, eps);
    } else {
        cur = g::ggml_rms_norm(ctx0, inp_l, eps); // :87
        cur = g::ggml_mul(ctx0, cur, layer.attn_norm); // :88
    }

    let mut q_cur = g::ggml_mul_mat(ctx0, layer.attn_q, cur); // :90
    let mut k_cur = g::ggml_mul_mat(ctx0, layer.attn_k, cur); // :91
    let mut v_cur = g::ggml_mul_mat(ctx0, layer.attn_v, cur); // :92

    q_cur = g::ggml_reshape_3d(ctx0, q_cur, head_dim, n_head, n_tokens); // :94
    k_cur = g::ggml_reshape_3d(ctx0, k_cur, head_dim, n_kv_head, n_tokens); // :95
    v_cur = g::ggml_reshape_3d(ctx0, v_cur, head_dim, n_kv_head, n_tokens); // :96

    q_cur = g::ggml_rms_norm(ctx0, q_cur, eps); // :99
    q_cur = g::ggml_mul(ctx0, q_cur, layer.attn_q_norm); // :100
    k_cur = g::ggml_rms_norm(ctx0, k_cur, eps); // :104
    k_cur = g::ggml_mul(ctx0, k_cur, layer.attn_k_norm); // :105

    let hd = head_dim as i32;
    match plant {
        Plant::RopeNeox | Plant::RopeNormal => {
            // The planted defect: the reference's non-M-RoPE branch (:117-123), or adjacent-pair rotation.
            let mode = if plant == Plant::RopeNeox {
                g::GGML_ROPE_TYPE_NEOX
            } else {
                g::GGML_ROPE_TYPE_NORMAL
            } as i32;
            q_cur = g::ggml_rope_ext(
                ctx0,
                q_cur,
                inp_pos,
                ptr::null_mut(),
                hd,
                mode,
                0,
                rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            k_cur = g::ggml_rope_ext(
                ctx0,
                k_cur,
                inp_pos,
                ptr::null_mut(),
                hd,
                mode,
                0,
                rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
        }
        _ => {
            let mode = g::GGML_ROPE_TYPE_MROPE as i32;
            q_cur = g::ggml_rope_multi(
                ctx0,
                q_cur,
                inp_mrope_pos,
                ptr::null_mut(),
                hd,
                mrope_sections.as_mut_ptr(),
                mode,
                0,
                rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ); // :109-111
            k_cur = g::ggml_rope_multi(
                ctx0,
                k_cur,
                inp_mrope_pos,
                ptr::null_mut(),
                hd,
                mrope_sections.as_mut_ptr(),
                mode,
                0,
                rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ); // :113-115
        }
    }

    let k_cache_2d = g::ggml_view_2d(ctx0, k_cache, head_dim * n_kv_head, n_ctx, (*k_cache).nb[2], 0); // :129
    let v_cache_2d = g::ggml_view_2d(ctx0, v_cache, head_dim * n_kv_head, n_ctx, (*v_cache).nb[2], 0); // :130
    let k_cur_2d = g::ggml_view_2d(ctx0, k_cur, head_dim * n_kv_head, n_tokens, (*k_cur).nb[2], 0); // :132
    let v_cur_2d = g::ggml_view_2d(ctx0, v_cur, head_dim * n_kv_head, n_tokens, (*v_cur).nb[2], 0); // :133

    let k_updated = g::ggml_set_rows(ctx0, k_cache_2d, k_cur_2d, inp_pos); // :135
    let v_updated = g::ggml_set_rows(ctx0, v_cache_2d, v_cur_2d, inp_pos); // :136
    g::ggml_build_forward_expand(gf, k_updated); // :138
    g::ggml_build_forward_expand(gf, v_updated); // :139

    let mut k = g::ggml_view_3d(
        ctx0,
        k_cache,
        head_dim,
        n_kv_head,
        n_kv_pad,
        (*k_cache).nb[1],
        (*k_cache).nb[2],
        0,
    ); // :141
    let mut v = g::ggml_view_3d(
        ctx0,
        v_cache,
        head_dim,
        n_kv_head,
        n_kv_pad,
        (*v_cache).nb[1],
        (*v_cache).nb[2],
        0,
    ); // :145

    let q = g::ggml_permute(ctx0, q_cur, 0, 2, 1, 3); // :149
    k = g::ggml_permute(ctx0, k, 0, 2, 1, 3); // :150
    v = g::ggml_permute(ctx0, v, 0, 2, 1, 3); // :151

    let kqv_fa = g::ggml_flash_attn_ext(ctx0, q, k, v, inp_mask, 1.0 / (head_dim as f32).sqrt(), 0.0, 0.0); // :153
    g::ggml_flash_attn_ext_set_prec(kqv_fa, g::GGML_PREC_F32); // :155
    cur = g::ggml_cont_2d(ctx0, kqv_fa, n_head * head_dim, n_tokens); // :156

    cur = g::ggml_mul_mat(ctx0, layer.attn_output, cur); // :158
    cur = g::ggml_add(ctx0, cur, inp_l); // :159
    let inp_ff = cur; // :160

    cur = g::ggml_rms_norm(ctx0, inp_ff, eps); // :162
    cur = g::ggml_mul(ctx0, cur, layer.ffn_norm); // :163

    let gate = g::ggml_mul_mat(ctx0, layer.ffn_gate, cur); // :165
    let up = g::ggml_mul_mat(ctx0, layer.ffn_up, cur); // :166

    cur = if plant == Plant::SwapGateUp {
        g::ggml_swiglu_split(ctx0, up, gate)
    } else {
        g::ggml_swiglu_split(ctx0, gate, up) // :168
    };

    // ffn_down_matmul, :28-35: an F16 weight is cast to F32 first. The published talker is Q8_0.
    let mut w_down = layer.ffn_down;
    if (*w_down).type_ == g::GGML_TYPE_F16 {
        w_down = g::ggml_cast(ctx0, w_down, g::GGML_TYPE_F32);
    }
    cur = g::ggml_mul_mat(ctx0, w_down, cur); // :170

    let out = g::ggml_add(ctx0, cur, inp_ff); // :172
    g::ggml_set_name(out, c"layer0_out".as_ptr());
    g::ggml_set_output(out);
    g::ggml_build_forward_expand(gf, out);
    gf
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        die("usage: layer0 MODEL.gguf OUT.bin [--threads N] [--tokens T] [--seed S] [--iters N] [--warmup N] [--plant NAME]");
    }
    let (model_path, out_path) = (&args[1], &args[2]);
    let (mut n_threads, mut n_tokens, mut iters, mut warmup, mut seed) = (4i32, 16i64, 0usize, 10usize, 20261106u64);
    let mut plant = Plant::None;
    let mut i = 3;
    while i + 1 < args.len() {
        let v = &args[i + 1];
        match args[i].as_str() {
            "--threads" => n_threads = v.parse().unwrap(),
            "--tokens" => n_tokens = v.parse().unwrap(),
            "--seed" => seed = v.parse().unwrap(),
            "--iters" => iters = v.parse().unwrap(),
            "--warmup" => warmup = v.parse().unwrap(),
            "--plant" => {
                plant = match v.as_str() {
                    "rope-neox" => Plant::RopeNeox,
                    "rope-normal" => Plant::RopeNormal,
                    "swap-gate-up" => Plant::SwapGateUp,
                    "norm-order" => Plant::NormOrder,
                    _ => die("unknown plant"),
                }
            }
            _ => die("unknown flag"),
        }
        i += 2;
    }

    unsafe {
        let mut meta: *mut g::ggml_context = ptr::null_mut();
        let gp = g::gguf_init_params {
            no_alloc: true,
            ctx: &mut meta,
        };
        let model_c = cstr(model_path);
        let gg = g::gguf_init_from_file(model_c.as_ptr(), gp);
        if gg.is_null() {
            die("cannot open GGUF");
        }

        let mut hp = Hparams {
            hidden_size: get_u32(gg, "qwen3-tts.talker.embedding_length"),
            n_head: get_u32(gg, "qwen3-tts.talker.attention.head_count"),
            n_kv_head: get_u32(gg, "qwen3-tts.talker.attention.head_count_kv"),
            head_dim: get_u32(gg, "qwen3-tts.talker.attention.key_length"),
            eps: get_f32(gg, "qwen3-tts.talker.attention.layer_norm_rms_epsilon"),
            rope_theta: get_f32(gg, "qwen3-tts.talker.rope.freq_base"),
            mrope_section: [0; 3],
        };
        let mi = g::gguf_find_key(gg, c"qwen3-tts.talker.rope.mrope_section".as_ptr());
        if mi < 0 {
            die("this program covers the M-RoPE talker only");
        }
        let d = g::gguf_get_arr_data(gg, mi) as *const i32;
        for k in 0..3 {
            hp.mrope_section[k] = *d.add(k);
        }

        // Weights: layer 0 only, in a CPU buffer, read at their file offsets (gguf_loader.cpp:303-430).
        const NAMES: [&str; 11] = [
            "attn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_q_norm",
            "attn_k_norm",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ];
        let wctx = g::ggml_init(g::ggml_init_params {
            mem_size: 16 * g::ggml_tensor_overhead(),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        let mut t = [ptr::null_mut::<g::ggml_tensor>(); 11];
        for (k, n) in NAMES.iter().enumerate() {
            let name = cstr(&format!("talker.blk.0.{n}.weight"));
            let m = g::ggml_get_tensor(meta, name.as_ptr());
            if m.is_null() {
                die(&format!("missing talker.blk.0.{n}.weight"));
            }
            t[k] = g::ggml_new_tensor(wctx, (*m).type_, g::ggml_n_dims(m), (*m).ne.as_ptr());
            g::ggml_set_name(t[k], name.as_ptr());
        }
        let layer = Layer {
            attn_norm: t[0],
            attn_q: t[1],
            attn_k: t[2],
            attn_v: t[3],
            attn_q_norm: t[4],
            attn_k_norm: t[5],
            attn_output: t[6],
            ffn_norm: t[7],
            ffn_gate: t[8],
            ffn_up: t[9],
            ffn_down: t[10],
        };

        let mut backend = g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
        if backend.is_null() {
            die("no CPU backend");
        }
        g::ggml_backend_cpu_set_n_threads(backend, n_threads);

        let wbuf = g::ggml_backend_alloc_ctx_tensors(wctx, backend);
        g::ggml_backend_buffer_set_usage(wbuf, g::GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        {
            let mut f = File::open(model_path).unwrap_or_else(|_| die("cannot open model for reading"));
            let data_offset = g::gguf_get_data_offset(gg);
            let mut buf = Vec::new();
            for &tk in &t {
                let id = g::gguf_find_tensor(gg, g::ggml_get_name(tk));
                let nbytes = g::gguf_get_tensor_size(gg, id);
                if nbytes != g::ggml_nbytes(tk) {
                    die("tensor size mismatch");
                }
                buf.resize(nbytes, 0u8);
                f.seek(SeekFrom::Start(
                    (data_offset + g::gguf_get_tensor_offset(gg, id)) as u64,
                ))
                .unwrap();
                f.read_exact(&mut buf).unwrap();
                g::ggml_backend_tensor_set(tk, buf.as_ptr() as *const c_void, 0, nbytes);
            }
        }

        // KV cache: n_ctx as the runtime picks it for a fresh prefill (transformer_runtime.cpp:101-103), F16
        // as init_kv_cache makes it (transformer_cache.cpp:86, :128-136).
        let n_ctx = 256i64.max(n_tokens + 16);
        let cctx = g::ggml_init(g::ggml_init_params {
            mem_size: 2 * g::ggml_tensor_overhead(),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        let k_cache = g::ggml_new_tensor_3d(cctx, g::GGML_TYPE_F16, hp.head_dim, hp.n_kv_head, n_ctx);
        let v_cache = g::ggml_new_tensor_3d(cctx, g::GGML_TYPE_F16, hp.head_dim, hp.n_kv_head, n_ctx);
        g::ggml_set_name(k_cache, c"k_cache_0".as_ptr());
        g::ggml_set_name(v_cache, c"v_cache_0".as_ptr());
        let cbuf = g::ggml_backend_alloc_ctx_tensors(cctx, backend);
        g::ggml_backend_buffer_clear(cbuf, 0);

        let sched = g::ggml_backend_sched_new(&mut backend, ptr::null_mut(), 1, MAX_NODES, false, true); // transformer_loader.cpp:266

        let mut compute_meta =
            vec![0u8; g::ggml_tensor_overhead() * MAX_NODES + g::ggml_graph_overhead_custom(MAX_NODES, false)];

        let hidden = hp.hidden_size as usize;
        let nt = n_tokens as usize;
        let mut input = vec![0f32; hidden * nt];
        seeded_input(seed, &mut input);
        let mut output = vec![0f32; hidden * nt];
        let n_past = 0i64;

        let mut forward = |output: &mut [f32]| {
            let ctx0 = g::ggml_init(g::ggml_init_params {
                mem_size: compute_meta.len(),
                mem_buffer: compute_meta.as_mut_ptr() as *mut c_void,
                no_alloc: true,
            });
            let gf = build_layer0(ctx0, &hp, &layer, k_cache, v_cache, n_ctx, n_tokens, n_past, plant);
            if !g::ggml_backend_sched_alloc_graph(sched, gf) {
                die("alloc graph");
            }
            let get = |name: &CStr| g::ggml_graph_get_tensor(gf, name.as_ptr());
            // An input the graph does not use is absent from it, and ggml_backend_tensor_set aborts on a null
            // tensor; the reference checks each input the same way (transformer_runtime.cpp:150, :165, :184).
            let set = |name: &CStr, data: *const c_void, n: usize| {
                let t = get(name);
                if !t.is_null() {
                    g::ggml_backend_tensor_set(t, data, 0, n);
                }
            };

            g::ggml_backend_tensor_set(
                get(c"inp_prefill_embd"),
                input.as_ptr() as *const c_void,
                0,
                input.len() * 4,
            );

            let pos: Vec<i32> = (0..n_tokens).map(|i| (n_past + i) as i32).collect(); // transformer_runtime.cpp:150-161
            set(c"inp_pos", pos.as_ptr() as *const c_void, pos.len() * 4);

            let mut mpos = vec![0i32; nt * 4]; // :164-180
            for i in 0..nt {
                let p = (n_past as usize + i) as i32;
                mpos[i] = p;
                mpos[i + nt] = p;
                mpos[i + nt * 2] = p;
                mpos[i + nt * 3] = 0;
            }
            set(c"inp_mrope_pos", mpos.as_ptr() as *const c_void, mpos.len() * 4);

            let inp_mask = get(c"inp_mask"); // :183-205
            let n_kv_pad = (*inp_mask).ne[0] as usize;
            let mut mask = vec![g::ggml_fp32_to_fp16(f32::NEG_INFINITY); nt * n_kv_pad];
            let zero = g::ggml_fp32_to_fp16(0.0);
            for q in 0..nt {
                let last_k = (n_past as usize + q).min(n_kv_pad - 1);
                for k in 0..=last_k {
                    mask[q * n_kv_pad + k] = zero;
                }
            }
            g::ggml_backend_tensor_set(inp_mask, mask.as_ptr() as *const c_void, 0, mask.len() * 2);

            if g::ggml_backend_sched_graph_compute(sched, gf) != g::GGML_STATUS_SUCCESS {
                die("compute");
            }
            g::ggml_backend_tensor_get(
                get(c"layer0_out"),
                output.as_mut_ptr() as *mut c_void,
                0,
                output.len() * 4,
            );
            g::ggml_backend_sched_reset(sched);
            g::ggml_free(ctx0);
        };

        forward(&mut output);
        let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        File::create(out_path)
            .and_then(|mut f| f.write_all(&bytes))
            .unwrap_or_else(|_| die("write output"));
        eprintln!(
            "layer0: wrote {} bytes ({} tokens x {}), threads={} seed={}",
            bytes.len(),
            n_tokens,
            hidden,
            n_threads,
            seed
        );

        if iters > 0 {
            for _ in 0..warmup {
                forward(&mut output);
            }
            let mut ms = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                forward(&mut output);
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let mean = ms.iter().sum::<f64>() / iters as f64;
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "timing lang=rust threads={} tokens={} iters={} mean_ms={:.4} median_ms={:.4} min_ms={:.4}",
                n_threads,
                n_tokens,
                iters,
                mean,
                ms[iters / 2],
                ms[0]
            );
        }

        g::ggml_backend_sched_free(sched);
        g::ggml_backend_buffer_free(cbuf);
        g::ggml_backend_buffer_free(wbuf);
        g::ggml_free(cctx);
        g::ggml_free(wctx);
        g::ggml_backend_free(backend);
        g::gguf_free(gg);
        g::ggml_free(meta);
    }
}
