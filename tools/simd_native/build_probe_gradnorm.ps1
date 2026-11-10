# build_probe_gradnorm.ps1 - build probe_gradnorm.exe (nap DLL dong)
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 probe_gradnorm.c /Fe:probe_gradnorm.exe"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }
Write-Host "OK -> probe_gradnorm.exe"
