# Script dong goi TokenVector.Inference thanh .nupkg.
# Bao gom: (a) source (.tkv), (b) managed assembly TokenVector.Inference.dll
# build tu source bang compiler tu-host (bundle dependency TokenVector 1.0.12).
# Cach dung:
#   powershell -File packaging\pack.ps1
# Push:
#   nuget push packaging\TokenVector.Inference.1.0.1.nupkg -ApiKey <KEY> -Source https://api.nuget.org/v3/index.json
$ErrorActionPreference = "Stop"
$packRoot = $PSScriptRoot
$root = Split-Path -Parent $packRoot
Set-Location $root

$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
$ilasm = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\ilasm.exe"
$work = Join-Path $env:TEMP "tkv_pack_infer"
$libDir = Join-Path $packRoot "lib\net10.0"

Write-Host "[1/4] Build managed assembly tu source..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $work | Out-Null
New-Item -ItemType Directory -Force -Path $libDir | Out-Null

$dll = Join-Path $libDir "TokenVector.Inference.dll"
$il = Join-Path $work "TokenVector.Inference.il"
$nugetCli = Join-Path $env:TEMP "tkv_pack\nuget.exe"

& $tkvc tv/tests/_lib_root.tkv $dll main 2>&1 | Out-Null
if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] tkvc build that bai." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $dll)) {
    Write-Host "[FAIL] khong tao duoc $dll" -ForegroundColor Red
    exit 1
}

# Patch assembly name + version trong IL roi asm lai bang ilasm.
$ilSrc = Join-Path $libDir "TokenVector.Inference.il"
if (Test-Path $ilSrc) {
    $t = Get-Content $ilSrc -Raw
    $t = $t.Replace(".assembly TKVApp {}", ".assembly TokenVector.Inference`n{`n  .ver 1:0:1:0`n}")
    $t = $t.Replace(".module TKVApp.exe", ".module TokenVector.Inference.dll")
    Set-Content $ilSrc -Value $t -Encoding UTF8
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ilasm /DLL /OUTPUT=$dll $ilSrc | Out-Null
    $ilasmRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($ilasmRc -ne 0) {
        Write-Host "[FAIL] ilasm that bai (exit $ilasmRc)." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[WARN] khong thay IL tai $ilSrc - giu nguyen DLL do tkvc sinh ra." -ForegroundColor Yellow
}

if (-not (Test-Path $nugetCli)) {
    Write-Host "[2/4] Tai nuget.exe..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path $nugetCli) | Out-Null
    Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nugetCli -UseBasicParsing
}

Write-Host "[3/4] Dong goi..." -ForegroundColor Cyan
& $nugetCli pack (Join-Path $packRoot "TokenVector.Inference.nuspec") -OutputDirectory $packRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] nuget pack that bai (exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

$nupkg = Get-ChildItem -Path $packRoot -Filter "*.nupkg" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($nupkg) {
    Write-Host "[4/4] [SUCCESS] $($nupkg.FullName)" -ForegroundColor Green
    Write-Host "Kich thuoc: $([math]::Round($nupkg.Length / 1MB, 2)) MB" -ForegroundColor Green
    Write-Host "Push: nuget push $($nupkg.FullName) -ApiKey <KEY> -Source https://api.nuget.org/v3/index.json" -ForegroundColor Yellow
}
