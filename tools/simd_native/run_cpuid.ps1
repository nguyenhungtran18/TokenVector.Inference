# run_cpuid.ps1 - dich test_cpuid.c KHONG co /arch (intrinsics chi can /D) va chay.
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
Push-Location $PSScriptRoot
try {
    cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 test_cpuid.c /Fe:test_cpuid.exe && test_cpuid.exe"
    exit $LASTEXITCODE
} finally { Pop-Location }
