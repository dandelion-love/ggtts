[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$Diagnostics,
    [string]$TargetDir
)
# Requires Rust x86_64-pc-windows-msvc, Flutter, Visual Studio C++ tools,
# CMake and libclang (set LIBCLANG_PATH if needed). CUDA builds also require
# Ninja and a Visual C++ developer environment.
$ErrorActionPreference = 'Stop'
$app = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = (Resolve-Path (Join-Path $app '../..')).Path
if (-not $TargetDir) {
    $suffix = if ($Cuda) { 'target/flutter-windows-cuda' } else { 'target/flutter-windows' }
    $TargetDir = Join-Path $root $suffix
}
$native = Join-Path $app 'native/windows-x64'
$target = 'x86_64-pc-windows-msvc'
$features = if ($Cuda) { 'ggtts-ffi/cuda,ggtts-ggml-sys/bindgen' } else { 'ggtts-ffi/ggml,ggtts-ggml-sys/bindgen' }
$runtime = @()
$previousGenerator = $env:CMAKE_GENERATOR
$previousCudaHome = $env:CUDA_HOME
if ($Cuda) {
    if (-not $env:CUDA_PATH) { throw 'CUDA_PATH must name the installed CUDA toolkit.' }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Use an x64 Native Tools Command Prompt for the CUDA build.' }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) { throw 'Ninja is required for the CUDA build.' }
    $runtime = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll') | ForEach-Object { Join-Path $env:CUDA_PATH "bin/$_" }
    foreach ($file in $runtime) { if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "CUDA runtime missing: $file" } }
    $env:CMAKE_GENERATOR = 'Ninja'
    $env:CUDA_HOME = $env:CUDA_PATH
}
try {
    Push-Location $root
    try {
        & cargo build --release --locked --target $target --target-dir $TargetDir -p ggtts-ffi -p ggtts-ggml-sys --features $features
        if ($LASTEXITCODE -ne 0) { throw "Native build failed ($LASTEXITCODE)." }
        New-Item -ItemType Directory -Force -Path $native | Out-Null
        Copy-Item -LiteralPath (Join-Path $TargetDir "$target/release/ggtts_ffi.dll") -Destination $native -Force
        Push-Location $app
        try {
            $flutterArgs = @('build', 'windows', '--release')
            if ($Diagnostics) { $flutterArgs += '--dart-define=GGTTS_DIAGNOSTICS=true' }
            & flutter @flutterArgs
            if ($LASTEXITCODE -ne 0) { throw "Flutter build failed ($LASTEXITCODE)." }
        } finally { Pop-Location }
    } finally { Pop-Location }
    $bundle = Join-Path $app 'build/windows/x64/runner/Release'
    if ($Cuda) {
        Copy-Item -LiteralPath $runtime -Destination $bundle -Force
    } else {
        # Flutter reuses its output directory; remove only DLLs from a prior CUDA build.
        foreach ($name in @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
            $file = Join-Path $bundle $name
            if (Test-Path -LiteralPath $file -PathType Leaf) { Remove-Item -LiteralPath $file -Force }
        }
    }
    Write-Host "Windows bundle: $bundle"
} finally {
    $env:CMAKE_GENERATOR = $previousGenerator
    $env:CUDA_HOME = $previousCudaHome
}
