#!/usr/bin/env bash
# Build and smoke the same Linux CUDA archive for selected CI and for releases.
set -euo pipefail
: "${TAG:?set TAG}" "${GITHUB_SHA:?set GITHUB_SHA}"
export GGML_CUDA_ARCH='86;89' GGML_PORTABLE=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export GGTTS_RELEASE_VERSION="$TAG"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
export PATH="$CUDA_HOME/bin:$PATH"
cargo build --release --locked -p ggtts-cli --features ggtts-cli/cuda
bin=${CARGO_TARGET_DIR:-target}/release/ggtts-render
# The Kitten tools (conversion, and `prepare` of a reference voice) need no GPU. Built against CPU ggml in a target
# directory of their own, they add a few MB; linked against ggml-cuda, which is static, each would carry a second
# copy of the CUDA kernels. pack copies them from beside ggtts-render.
# The C library (ggtts-ffi) is built there too, against CPU ggml: its engines run on the CPU backend whatever ggml is
# built with (the binding has no device choice yet), so a CUDA-linked copy would carry kernels it never runs and need
# the CUDA driver and runtime to load. This archive's library is the CPU archive's, built the same way.
kitten_target=${KITTEN_CPU_TARGET_DIR:-target-kitten-cpu}
CARGO_TARGET_DIR="$kitten_target" cargo build --release --locked -p ggtts-kitten -p ggtts-ffi \
    --features ggtts-kitten/ggml,ggtts-ffi/ggml
cp "$kitten_target/release/ggtts-kitten" "$kitten_target/release/ggtts-kitten-convert" \
    "$kitten_target/release/libggtts_ffi.so" "$(dirname "$bin")/"
# These are machine-code images, not just a CMake configuration claim.
cuobjdump --list-elf "$bin" > cuda-images.txt
grep -q sm_86 cuda-images.txt
grep -q sm_89 cuda-images.txt
src=crates/ggtts-core/src/lib.rs
commit=$(sed -n 's/^ *pub const COMMIT: &str = "\([0-9a-f]\{40\}\)";$/\1/p' "$src")
describe=$(sed -n 's/^ *pub const DESCRIBE: &str = "\(.*\)";$/\1/p' "$src")
test ${#commit} -eq 40
name="ggtts-${TAG}-x86_64-linux-cuda-ggml-${commit:0:7}"
mkdir -p "dist/$name"
binaries=$(bash scripts/package_and_smoke.sh pack "$bin" "dist/$name")
libraries=$(bash scripts/package_and_smoke.sh libraries "dist/$name")
# The C library carries no CUDA: it must load without the driver or the CUDA runtime.
if readelf -d "dist/$name/lib/libggtts_ffi.so" | grep -E 'NEEDED.*(libcuda|libcudart|libcublas)'; then
    echo "lib/libggtts_ffi.so links CUDA; it is meant to be the CPU library" >&2
    exit 1
fi
# Hosted runners have no driver. The stub permits CPU/fake-engine smoke tests;
# it is never packaged and must never replace a real driver on a user's machine.
stub_dir=$(mktemp -d)
trap 'rm -rf "$stub_dir"' EXIT
ln -s "$CUDA_HOME/lib64/stubs/libcuda.so" "$stub_dir/libcuda.so.1"
export LD_LIBRARY_PATH="${GGTTS_CUDA_DRIVER_DIR:-$stub_dir}:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
engines=$(bash scripts/buildinfo_engines.sh "dist/$name/ggtts-render" qwen3-tts kitten-tts-2)
cat > "dist/$name/BUILDINFO.json" <<EOF
{
  "ggtts": "${TAG}",
  "ggtts_commit": "${GITHUB_SHA}",
  "ggml_commit": "${commit}",
  "ggml_describe": "${describe}",
  "rustc": "$(rustc -V)",
  "target": "x86_64-linux",
  "binaries": ${binaries},
  "libraries": ${libraries},
  "library_features": ["cpu", "ggml"],
  "features": ["cpu", "ggml", "cuda"],
  "cuda_toolkit": "$(nvcc --version | sed -n 's/.*release \([^,]*\),.*/\1/p')",
  "cuda_architectures": ["sm_86", "sm_89"],
  "engines": ${engines},
  "built_by": "${GITHUB_SERVER_URL:-local}/${GITHUB_REPOSITORY:-local}/actions/runs/${GITHUB_RUN_ID:-local}"
}
EOF
"dist/$name/ggtts-render" --version | tee "dist/$name/VERSION.txt"
grep -q "$commit" "dist/$name/VERSION.txt"
(cd dist && bash ../scripts/package_and_smoke.sh archive "$name" tar.gz)
rm -rf "dist/$name"
bash scripts/package_and_smoke.sh smoke "dist/$name.tar.gz" "$TAG"
