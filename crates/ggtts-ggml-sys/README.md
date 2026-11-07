# ggtts-ggml-sys

Raw, unsafe bindings to [ggml](https://github.com/ggml-org/ggml) at `af97976c7810cdabb1863172f31c432dab767de7`,
with ggml built from source by CMake, CPU only, statically linked. Promoted from the layer-0 spike, which showed
Rust over these bindings reproduces a Qwen3-TTS talker layer byte for byte against the reference on CPU.

## Using it

```toml
ggtts-ggml-sys = { workspace = true, features = ["ggml"] }
```

```bash
git submodule update --init crates/ggtts-ggml-sys/ggml
cargo build -p ggtts-ggml-sys --features ggml   # needs cmake and a C/C++ compiler; not libclang
```

| Feature | What it does |
|---|---|
| (none, the default) | Nothing is built or linked. The crate holds only the pin, `pad_to` and the rope mode numbers |
| `ggml` | Builds ggml from the submodule with CMake (Release, static, `-fPIC`, `GGML_NATIVE` at ggml's default) and links `ggml`, `ggml-cpu`, `ggml-base` and, on Linux, `stdc++`, `gomp`, `m`. Uses the committed bindings |
| `bindgen` | Implies `ggml`, and generates the bindings at build time from `wrapper.h`. Needs libclang |

| Variable | Effect |
|---|---|
| `GGML_SRC` | Build this ggml tree instead of the submodule. It must be the pinned commit |
| `GGML_LIB_DIR` | Link prebuilt `libggml.a`, `libggml-cpu.a`, `libggml-base.a` from here instead of building. A parity check points it at the reference's own ggml build, so both sides link the same compiled kernels |
| `CMAKE_BUILD_PARALLEL_LEVEL` | CMake's build parallelism |

## Why the feature gate, and why committed bindings

Two choices, made so CI stays green without a workflow change:

* **The ggml build is behind `ggml`, off by default.** CI checks out without submodules, and its build job also runs
  on Windows, where this binding has not been built. With the feature off, `cargo build --workspace`, clippy and
  the tests build the rest of the workspace exactly as before. Engines that need ggml gate their ggml code behind
  their own `ggml` feature. A CI job that runs `git submodule update --init` and builds `--features ggml` on Linux
  is the next CI change; it is described in the pull request rather than made here.
* **The bindings are committed for `x86_64-unknown-linux-gnu`** (`src/bindings/`), so the `ggml` feature needs
  cmake and a compiler but not libclang. libclang was the hard part of the spike's build: a bare `libclang.so`
  without its resource headers fails on `stdbool.h`. Gating bindgen alone behind a feature would still have left
  every consumer to provide libclang. Other targets build with `--features bindgen`, and the build says so if
  they try without it. `scripts/regenerate-bindings.sh` rewrites the committed file after the pin or `wrapper.h`
  moves; with libclang from pip, for example:

  ```bash
  LIBCLANG_PATH=<venv>/lib/python3.10/site-packages/clang/native \
  BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/11/include \
  crates/ggtts-ggml-sys/scripts/regenerate-bindings.sh
  ```

## What in ggml's C API fights Rust

From the spike:

* **Errors abort the process.** `GGML_ASSERT` calls `abort()`, and a panic cannot unwind through it. Setting an
  input the graph does not use (absent, so `ggml_graph_get_tensor` returns null) kills the program inside
  `ggml_backend_tensor_set`. Validate before calling.
* **Lifetimes are implicit.** A tensor belongs to its `ggml_context`, its data to a backend buffer, and a graph's
  metadata to a caller-provided buffer reused on every call. None of it is in the types.
* **Function-like macros vanish.** `GGML_PAD` is re-implemented here as `pad_to` (the name `ggml_pad` is an op).
* **Array parameters decay.** `ggml_rope_multi`'s `int sections[4]` arrives as `*mut c_int`, unchecked.
* **Static linking order** is `ggml`, `ggml-cpu`, `ggml-base`, then the C++ runtime and OpenMP; Rust links
  position-independent executables, so ggml is built with `-fPIC`.
* **`GGML_NATIVE`** (ggml's default) compiles for the build machine's CPU: right for a parity check, wrong for a
  release binary.
