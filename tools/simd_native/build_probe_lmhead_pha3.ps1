# build_probe_lmhead_pha3.ps1 - build probe_lmhead_pha3.exe (PROBE)
# Can libtorch (ATen) + scipy-openblas. Chay tu thu muc nay.
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "Khong tim thay vcvars64.bat" }

$torchDir = (& python -c "import torch,os;print(os.path.dirname(torch.__file__))").Trim()
$inc = Join-Path $torchDir "include"
$lib = Join-Path $torchDir "lib"
Write-Host "torch: $torchDir"

$base = "/nologo /O2 /arch:AVX2 /std:c++20 /EHsc /MD"
cmd /c "call `"$vcvars`" > nul 2>&1 && cl $base /I`"$inc`" /I`"$inc\torch\csrc\api\include`" probe_lmhead_pha3.cpp /Fe:probe_lmhead_pha3.exe /link /LIBPATH:`"$lib`" torch.lib torch_cpu.lib c10.lib"
if ($LASTEXITCODE -ne 0) { throw "build that bai" }

$dich = Join-Path (Resolve-Path "..\..\out") "probe_lmhead_pha3.exe"
Copy-Item probe_lmhead_pha3.exe $dich -Force
Write-Host "OK -> $dich"
