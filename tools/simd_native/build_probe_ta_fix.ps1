# build_probe_ta_fix.ps1 - build probe tach pha lmhead_bwd_fused
# #include nguyen tkv_simd_native.c nen khong can DLL; nhung file do dung
# windows.h + immintrin nen phai build bang MSVC /arch:AVX2.
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 /Fe:probe_ta_fix.exe probe_ta_fix.c D:/TokenVector/Release/tools/simd_native/tkv_bwd_ops.c D:/TokenVector/Release/tools/simd_native/tkv_flash_attn.c"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }
Write-Host "OK -> probe_ta_fix.exe"
