$ErrorActionPreference = 'Stop'
$root = Join-Path $env:TEMP ('ggtts-cuda-pack-test-' + [guid]::NewGuid().ToString('N'))
$package = Join-Path $root 'package'
$toolkit = Join-Path $root 'toolkit'
New-Item -ItemType Directory -Force -Path (Join-Path $package 'instruments'), (Join-Path $toolkit 'bin') | Out-Null
foreach ($name in @('ggtts-render.exe', 'ggtts-measure.exe', 'LICENSE', 'README.md')) {
    Set-Content -LiteralPath (Join-Path $package $name) -Value 'fixture'
}
foreach ($name in @('cudart64_12.dll', 'cublas64_12.dll')) {
    Set-Content -LiteralPath (Join-Path $toolkit "bin\$name") -Value $name
}
Set-Content -LiteralPath (Join-Path $toolkit 'EULA.txt') -Value 'fixture licence'
$script = Join-Path $PSScriptRoot 'package_windows_cuda.ps1'
$failed = $false
try { & $script -PackageDirectory $package -CudaRoot $toolkit } catch { $failed = $true }
if (-not $failed) { throw 'Missing CUDA component was accepted.' }
if (Test-Path -LiteralPath (Join-Path $package 'cudart64_12.dll')) { throw 'Missing input caused a partial package.' }
Set-Content -LiteralPath (Join-Path $toolkit 'bin\cublasLt64_12.dll') -Value 'third fixture'
& $script -PackageDirectory $package -CudaRoot $toolkit
$metadata = Get-Content -Raw -LiteralPath (Join-Path $package 'CUDA-RUNTIME.json') | ConvertFrom-Json
foreach ($name in @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $package $name)).Hash.ToLowerInvariant()
    if ($metadata.toolkit_files_sha256.$name -ne $actual) { throw "Wrong recorded hash: $name" }
}
if (Test-Path -LiteralPath (Join-Path $package 'nvcuda.dll')) { throw 'The NVIDIA driver was bundled.' }
$failed = $false
try { & $script -PackageDirectory $package -CudaRoot $toolkit } catch { $failed = $true }
if (-not $failed) { throw 'Existing package files were overwritten.' }
Write-Output 'Windows CUDA package checks passed.'
