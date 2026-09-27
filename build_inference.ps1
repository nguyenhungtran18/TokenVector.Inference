# TokenVector.Inference - Pure TokenVector Build, Test & Benchmark Pipeline
#
# Repo thuan TokenVector: khong con C#, khong dotnet. Mọi build/test chay
# qua tkvc + tools\run_cases.ps1.
#
# Cach dung:
#   powershell -File build_inference.ps1
#   powershell -File build_inference.ps1 -SkipTests
#   powershell -File build_inference.ps1 -SkipBenchmarks
#   powershell -File build_inference.ps1 -Compiler "D:\path\to\tkvc.exe" -WorkDir $env:TEMP
param(
    [switch]$SkipTests = $false,
    [switch]$SkipBenchmarks = $false,
    [string]$Compiler = "D:\TokenVector\3.code\dist\tkvc_patched.exe",
    [string]$WorkDir = $env:TEMP,
    [int]$BenchmarkTimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  TokenVector.Inference Pure-TKV Pipeline                                  " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Build cac entry point chuan (theo README). KHONG build le tung module:
# tkvc co bug tu-phat sinh lai import-copy stub (ghi zero, lam hong build
# sau). Chi build entry points da tai lieu hoa.
Write-Host "`n[1/3] Building entry points (inference_tests, gap_tests, compare_rivals)..." -ForegroundColor Yellow
$entries = @(
    @{ Src = "tv\tests\inference_tests.tkv"; Out = "inference_tests.exe" },
    @{ Src = "tv\tests\gap_tests.tkv"; Out = "gap_tests.exe" },
    @{ Src = "tv\benchmarks\compare_rivals.tkv"; Out = "compare_rivals.exe" }
)
foreach ($e in $entries) {
    $src = Join-Path $root $e.Src
    $out = Join-Path $WorkDir $e.Out
    & $Compiler build $src --out $out 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $out)) {
        Write-Host "LOI: bien dich that bai: $($e.Src)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  $($e.Src) OK." -ForegroundColor Green
}
# Guard: phat hien import-copy bi zero (bug tu-phat sinh cua tkvc).
# PowerShell 5.1: doc byte bang -Encoding Byte (khong co -AsByteStream).
$zeroed = @()
foreach ($tkv in (Get-ChildItem -Path (Join-Path $root "tv") -Recurse -Filter *.tkv)) {
    if ($tkv.Length -eq 0) { continue }
    $head = Get-Content -LiteralPath $tkv.FullName -Encoding Byte -ReadCount 64 -TotalCount 64
    $hasZero = @($head | Where-Object { $_ -eq 0 }).Count -gt 0
    if ($hasZero) { $zeroed += $tkv.FullName }
    if ($zeroed.Count -ge 5) { break }
}
if ($zeroed) {
    Write-Host "CANH BAO: phat hien file .tkv bi zero (bug tkvc), khoi phuc tu git:" -ForegroundColor Red
    $zeroed | ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Red }
    exit 1
}

# 2. Run tests (26 inference cases + 12 gap cases, converted from xUnit).
if (-not $SkipTests) {
    Write-Host "`n[2/3] Running Pure-TKV Test Suites..." -ForegroundColor Yellow
    & (Join-Path $root "tools\run_cases.ps1") -Compiler $Compiler -WorkDir $WorkDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LOI: test suite that bai." -ForegroundColor Red
        exit 1
    }
}

# 3. Benchmarks (regression gates, pure .tkv). Dung exe da build o [1/3].
# Luu y: pure .tkv chay scalar (khong SIMD nhu C#), bo benchmark nay co the
# mat >10 phut. Dung -SkipBenchmarks de bo qua, hoac -BenchmarkTimeoutSeconds
# de gio han thoi gian.
if (-not $SkipBenchmarks) {
    Write-Host "`n[3/3] Running benchmarks (compare_rivals)..." -ForegroundColor Yellow
    Write-Host "  Gio han: $BenchmarkTimeoutSeconds giay (pure .tkv khong SIMD, cham hon C# nhieu lan)." -ForegroundColor DarkGray
    $benchExe = Join-Path $WorkDir "compare_rivals.exe"
    $p = Start-Process -FilePath $benchExe -NoNewWindow -PassThru
    # Cache handle: khong co $p.Handle thi .NET khong giu lai handle cua
    # process da ket thuc va $p.ExitCode tra ve rong (bi bao "benchmark that bai"
    # du process chay thanh cong).
    $null = $p.Handle
    if (-not $p.WaitForExit($BenchmarkTimeoutSeconds * 1000)) {
        try { $p.Kill() } catch { }
        Write-Host "LOI: benchmark vuot gio han $BenchmarkTimeoutSeconds giay (da dung)." -ForegroundColor Red
        Write-Host "      Pure .tkv scalar cham hon C# SIMD nhieu lan; dung -SkipBenchmarks de build nhanh." -ForegroundColor DarkGray
        exit 1
    }
    if ($p.ExitCode -ne 0) {
        Write-Host "LOI: benchmark that bai (exit $($p.ExitCode))." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n==========================================================================" -ForegroundColor Green
Write-Host "  TokenVector.Inference Pipeline Completed Successfully!                   " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
