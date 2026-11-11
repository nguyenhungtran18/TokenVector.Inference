# Smoke test: call the TokenVector.Inference.dll from .NET.
# Prereq: DLL built (run packaging\pack.ps1 first), plus the runtime DLLs
# from the compiler repo (default D:\TokenVector\Release\runtime).
param(
    [string]$RuntimeNativeDir = "D:\TokenVector\Release\runtime"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dll = Join-Path $root "packaging\lib\net10.0\TokenVector.Inference.dll"

if (-not (Test-Path $dll)) {
    Write-Host "[FAIL] Missing $dll - run packaging\pack.ps1 first (it builds the DLL from source)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $RuntimeNativeDir)) {
    Write-Host "[FAIL] Runtime native dir not found: $RuntimeNativeDir" -ForegroundColor Red
    Write-Host "       Pass -RuntimeNativeDir <path> (must contain tkv_simd_native.dll)." -ForegroundColor Red
    exit 1
}

$out = Join-Path $PSScriptRoot "bin\Debug\net10.0"
Write-Host "[1/2] Build smoke app..." -ForegroundColor Cyan
& dotnet build (Join-Path $PSScriptRoot "smoke.csproj") -v q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/2] Copy runtime DLLs + run..." -ForegroundColor Cyan
Copy-Item (Join-Path $RuntimeNativeDir "tkv_simd_native.dll") $out -Force
Copy-Item (Join-Path $RuntimeNativeDir "System.Numerics.Vectors.dll") $out -Force -ErrorAction SilentlyContinue
& dotnet (Join-Path $out "smoke.dll")
exit $LASTEXITCODE
