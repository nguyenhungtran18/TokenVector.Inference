# build_probe_attn_save.ps1 - build probe_attn_save.dll (PROBE, khong phai production)
# Chay tu thu muc nay: powershell -ExecutionPolicy Bypass -File build_probe_attn_save.ps1
$ErrorActionPreference = 'Stop'

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Khong tim thay vswhere.exe - can Visual Studio" }
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "Khong tim thay vcvars64.bat trong $vs" }

$torchDir = (& python -c "import torch,os;print(os.path.dirname(torch.__file__))").Trim()
if (-not (Test-Path $torchDir)) { throw "Khong tim thay torch (python -c import torch)" }
$inc = Join-Path $torchDir "include"
$lib = Join-Path $torchDir "lib"
Write-Host "torch: $torchDir"

$base = "/nologo /O2 /arch:AVX2 /std:c++20 /EHsc /MD /LD"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl $base /I`"$inc`" /I`"$inc\torch\csrc\api\include`" probe_attn_save.cpp /Fo:probe_attn_save.obj /link /LIBPATH:`"$lib`" torch.lib torch_cpu.lib c10.lib /OUT:probe_attn_save.dll"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }

$dich = Join-Path (Resolve-Path "..\..\out") "probe_attn_save.dll"
Copy-Item probe_attn_save.dll $dich -Force
Write-Host "OK -> $dich"
