# run_peak.ps1 - dich /arch:AVX2 va chay test_peak (do tran AVX2 1 luong).
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
Push-Location $PSScriptRoot
try {
    cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 /fp:precise test_peak.c /Fe:test_peak.exe"
    if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED"; exit 2 }
    & ".\test_peak.exe"
    exit $LASTEXITCODE
} finally { Pop-Location }
