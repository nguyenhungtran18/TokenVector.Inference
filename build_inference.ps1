# TokenVector.Inference - Automated Build, Test, Benchmark, and Packaging Script
param(
    [string]$Configuration = "Release",
    [switch]$SkipTests = $false,
    [switch]$SkipBenchmarks = $false
)

$ErrorActionPreference = "Stop"
$DotnetExe = "C:\Users\Admin\.dotnet\dotnet.exe"
if (-not (Test-Path $DotnetExe)) {
    $DotnetExe = (Get-Command dotnet -ErrorAction SilentlyContinue).Source
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  TokenVector.Inference Build & Packaging Pipeline ($Configuration)       " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Build Core Library
Write-Host "`n[1/4] Building TokenVector.Inference Core..." -ForegroundColor Yellow
& $DotnetExe build "TokenVector.Inference.csproj" -c $Configuration

# 2. Run Tests
if (-not $SkipTests) {
    Write-Host "`n[2/4] Running xUnit Test Suite..." -ForegroundColor Yellow
    & $DotnetExe test "tests\TokenVector.Inference.Tests\TokenVector.Inference.Tests.csproj" -c $Configuration --no-build
}

# 3. Package NuGet and Artifacts
Write-Host "`n[3/4] Packaging Library (.dll, .xml, .snupkg, .nupkg) into dist/..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "dist" | Out-Null
& $DotnetExe pack "TokenVector.Inference.csproj" -c $Configuration --output "dist"

Copy-Item "bin\$Configuration\net8.0\TokenVector.Inference.dll" -Destination "dist\" -Force
Copy-Item "bin\$Configuration\net8.0\TokenVector.Inference.xml" -Destination "dist\" -Force

# 4. Optional Benchmarks
if (-not $SkipBenchmarks) {
    Write-Host "`n[4/4] Executing Benchmark Suite..." -ForegroundColor Yellow
    & $DotnetExe run --project "benchmarks\TokenVector.Inference.Benchmarks\TokenVector.Inference.Benchmarks.csproj" -c $Configuration --no-build
}

Write-Host "`n==========================================================================" -ForegroundColor Green
Write-Host "  TokenVector.Inference Pipeline Completed Successfully!                   " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
