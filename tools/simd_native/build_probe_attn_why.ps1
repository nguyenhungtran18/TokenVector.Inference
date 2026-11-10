# build_probe_attn_why.ps1 - build probe_attn_why.exe (khong can DLL, nap dong)
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 probe_attn_why.c /Fe:probe_attn_why.exe"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }
Write-Host "OK -> probe_attn_why.exe"
