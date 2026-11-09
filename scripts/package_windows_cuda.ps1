[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackageDirectory,
    [string]$CudaRoot = $env:CUDA_PATH
)

# Add only CUDA toolkit redistributable candidates to an already packed Windows
# application.  The NVIDIA driver and Microsoft runtime remain prerequisites.
# The release workflow must archive and smoke this directory separately.
$ErrorActionPreference = 'Stop'
if (-not $CudaRoot) { throw 'Set CUDA_PATH or pass -CudaRoot to the installed CUDA toolkit.' }
$package = (Resolve-Path -LiteralPath $PackageDirectory).Path
$toolkit = (Resolve-Path -LiteralPath $CudaRoot).Path
foreach ($name in @('ggtts-render.exe', 'ggtts-measure.exe', 'LICENSE', 'README.md')) {
    if (-not (Test-Path -LiteralPath (Join-Path $package $name) -PathType Leaf)) {
        throw "Package is missing $name. Run package_and_smoke.sh pack first."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $package 'instruments') -PathType Container)) {
    throw 'Package is missing instruments/. Run package_and_smoke.sh pack first.'
}
$files = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
foreach ($name in $files) {
    if (-not (Test-Path -LiteralPath (Join-Path $toolkit "bin\$name") -PathType Leaf)) {
        throw "CUDA toolkit is missing $name."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $toolkit 'EULA.txt') -PathType Leaf)) {
    throw 'CUDA toolkit is missing EULA.txt.'
}
foreach ($name in ($files + @('NVIDIA-CUDA-EULA.txt', 'CUDA-RUNTIME.json'))) {
    if (Test-Path -LiteralPath (Join-Path $package $name)) { throw "Refusing to overwrite $name." }
}
$hashes = [ordered]@{}
foreach ($name in $files) {
    $source = Join-Path $toolkit "bin\$name"
    Copy-Item -LiteralPath $source -Destination (Join-Path $package $name)
    $hashes[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $package $name)).Hash.ToLowerInvariant()
}
Copy-Item -LiteralPath (Join-Path $toolkit 'EULA.txt') -Destination (Join-Path $package 'NVIDIA-CUDA-EULA.txt')
$metadata = [ordered]@{
    schema = 'ggtts-cuda-runtime/1'
    toolkit_files_sha256 = $hashes
    eula_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $package 'NVIDIA-CUDA-EULA.txt')).Hash.ToLowerInvariant()
    prerequisites = @('Compatible NVIDIA driver providing nvcuda.dll', 'Microsoft Visual C++ x64 Redistributable')
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $package 'CUDA-RUNTIME.json') -Encoding UTF8
