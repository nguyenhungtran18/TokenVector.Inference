# build_m1.ps1 — build shim tkv_simd_native.dll (M1 OpenBLAS) vao out\m1
# Cach dung (tu goc repo):
#   powershell -File tools/simd_native/build_m1.ps1
# Sau do copy m1blas.dll (OpenBLAS cua scipy) + System.Numerics.Vectors.dll
# vao cung thu muc — exe .tkv nap 2 DLL do khi chay tu out\m1.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$out = Join-Path $root "out\m1"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$src = Join-Path $root "tools\simd_native\tkv_simd_native_m1.c"
$dll = Join-Path $out "tkv_simd_native.dll"
cmd /c "call `"$vcvars`" >nul 2>&1 && cl /nologo /LD /O2 /MT /Fe:`"$dll`" `"$src`" /link /NOIMPLIB"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $dll)) {
    Write-Host "LOI: build shim that bai" -ForegroundColor Red
    exit 1
}
# m1blas.dll = OpenBLAS LP64 cua scipy (cblas_sgemm int32), doi ten de shim nap duoc
$py = "C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\scipy.libs"
$blas = Get-ChildItem -Path $py -Filter "libscipy_openblas-*.dll" | Select-Object -First 1
if (-not $blas) { Write-Host "LOI: khong thay OpenBLAS trong scipy.libs" -ForegroundColor Red; exit 1 }
Copy-Item -LiteralPath $blas.FullName (Join-Path $out "m1blas.dll") -Force
Copy-Item -LiteralPath (Join-Path $root "out\System.Numerics.Vectors.dll") $out -Force
Write-Host "OK: $dll + m1blas.dll + System.Numerics.Vectors.dll -> $out" -ForegroundColor Green
