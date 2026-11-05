# run_bench_micro.ps1 - build /O2 /arch:AVX2 /fp:fast (giong build.ps1) va chay.
param([string[]]$Args2)
$ErrorActionPreference = 'Stop'
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
Push-Location $PSScriptRoot
try {
    cmd /c "call `"$vcvars`" > nul 2>&1 && cl /nologo /O2 /arch:AVX2 /fp:fast bench_micro.c /Fe:bench_micro.exe"
    if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED"; exit 2 }
    & ".\bench_micro.exe" @Args2
    exit $LASTEXITCODE
} finally { Pop-Location }
