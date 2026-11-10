# build_probe_attn_save_run.ps1 - build probe_attn_save_run.exe (PROBE)
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 probe_attn_save_run.c /Fe:probe_attn_save_run.exe"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }
Copy-Item probe_attn_save_run.exe ..\..\out\probe_attn_save_run.exe -Force
Write-Host "OK -> out\probe_attn_save_run.exe"
