# Windows CUDA build and local package

This is a source-build path, not a released Windows CUDA archive. It has been
checked for compilation and package startup; a real GPU render and a published
archive have not yet been verified. The current release workflow produces the
CUDA archive for Linux, not Windows.

Use a 64-bit Windows environment with a compatible NVIDIA GPU and driver, the
Microsoft Visual C++ x64 Redistributable, Rust pinned by `rust-toolchain.toml`,
Visual Studio C++ build tools, CMake, Ninja, libclang and a CUDA 12 toolkit. Set
`LIBCLANG_PATH` to the directory containing libclang and start a Visual Studio
developer shell. The build below targets NVIDIA compute capabilities 8.6 and
8.9; support for other architectures has not been established. The portable
CPU part requires AVX2, FMA, F16C, BMI2 and OS-enabled AVX state.

For a toolkit installed in the usual location, PowerShell can build the two
executables as follows:

```powershell
git submodule update --init crates/ggtts-ggml-sys/ggml
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3'
$env:CUDA_HOME = $env:CUDA_PATH
$env:GGML_PORTABLE = '1'
$env:GGML_CUDA_ARCH = '86;89'
$env:CMAKE_GENERATOR = 'Ninja'
cargo build --release --locked -p ggtts-cli --features ggtts-cli/bindgen,ggtts-cli/cuda
```

The Windows toolkit's `bin` directory must be on `PATH` to run the unpackaged
build. To prepare an application directory with the CUDA runtime and cuBLAS
DLLs, first use the existing packer, then add the toolkit files:

```powershell
bash scripts/package_and_smoke.sh pack target/release/ggtts-render.exe dist/ggtts-windows-cuda
powershell -NoProfile -File scripts/package_windows_cuda.ps1 `
  -PackageDirectory dist/ggtts-windows-cuda -CudaRoot $env:CUDA_PATH
```

The helper includes `cudart64_12.dll`, `cublas64_12.dll`,
`cublasLt64_12.dll`, the toolkit's EULA and a SHA-256 manifest. It does not
include `nvcuda.dll`: the installed NVIDIA driver supplies that library. The
Microsoft Visual C++ x64 Redistributable remains a system prerequisite. The
helper does not install either prerequisite, archive the directory or publish
it. Use the existing archive/smoke workflow after packaging; an offline startup
does not demonstrate that CUDA executes on a GPU.

NVIDIA lists the CUDA runtime and cuBLAS components in its
[CUDA 12.3 EULA](https://docs.nvidia.com/cuda/archive/12.3.2/eula/index.html),
subject to its distribution terms. Microsoft describes its redistributable
requirements in [Redistribute Visual C++ files](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170).
The CUDA 12.3 [release notes](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-toolkit-release-notes/)
give Windows driver 527.41 as the minimum for minor-version compatibility,
subject to the documented compatibility limitations. Check the terms and the
driver requirements for the exact toolkit used by any release.
