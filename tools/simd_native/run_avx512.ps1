# run_avx512.ps1 - dich /arch:AVX512 roi chay; exit code != 0 nghia la khong dung duoc.
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
Push-Location $PSScriptRoot
try {
    cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX512 test_avx512.c /Fe:test_avx512.exe"
    if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED (compiler khong ho tro /arch:AVX512)"; exit 2 }
    & ".\test_avx512.exe"
    Write-Host "exit=$LASTEXITCODE"
    exit $LASTEXITCODE
} finally { Pop-Location }
