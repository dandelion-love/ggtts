[CmdletBinding()]
param()
# Requires Rust x86_64-pc-windows-msvc, Flutter, Visual Studio C++ tools,
# CMake and libclang (set LIBCLANG_PATH if needed).
$ErrorActionPreference = 'Stop'
$app = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = (Resolve-Path (Join-Path $app '../..')).Path
$targetDir = Join-Path $root 'target/flutter-windows'
$native = Join-Path $app 'native/windows-x64'
$target = 'x86_64-pc-windows-msvc'
Push-Location $root
try {
    & cargo build --release --locked --target $target --target-dir $targetDir -p ggtts-ffi -p ggtts-ggml-sys --features 'ggtts-ffi/ggml,ggtts-ggml-sys/bindgen'
    if ($LASTEXITCODE -ne 0) { throw "Native build failed ($LASTEXITCODE)." }
    New-Item -ItemType Directory -Force -Path $native | Out-Null
    Copy-Item -LiteralPath (Join-Path $targetDir "$target/release/ggtts_ffi.dll") -Destination $native -Force
    Push-Location $app
    try {
        & flutter build windows --release
        if ($LASTEXITCODE -ne 0) { throw "Flutter build failed ($LASTEXITCODE)." }
    } finally { Pop-Location }
} finally { Pop-Location }
Write-Host "Windows bundle: $app/build/windows/x64/runner/Release"
