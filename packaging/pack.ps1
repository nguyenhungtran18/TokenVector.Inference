# Script tu dong dong goi TokenVector.Inference (source package) thanh .nupkg.
# Cách dùng:
#   powershell -File packaging\pack.ps1
# Sau khi build xong, push thủ công:
#   nuget push packaging\TokenVector.Inference.1.0.1.nupkg -ApiKey <KEY> -Source https://api.nuget.org/v3/index.json
$ErrorActionPreference = "Stop"
$packRoot = $PSScriptRoot
$nugetRoot = $packRoot
$nugetCli = Join-Path $env:TEMP "tkv_pack\nuget.exe"

if (-not (Test-Path $nugetCli)) {
    Write-Host "Dang tai nuget.exe..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path $nugetCli) | Out-Null
    Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nugetCli -UseBasicParsing
}

Write-Host "Bat dau dong goi TokenVector.Inference (source package)..." -ForegroundColor Cyan
& $nugetCli pack (Join-Path $nugetRoot "TokenVector.Inference.nuspec") -OutputDirectory $nugetRoot

# PHAI kiem tra exit code TRUOC khi bao SUCCESS - xem note trong compiler repo.
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] nuget pack that bai (exit $LASTEXITCODE) - khong tao duoc .nupkg moi." -ForegroundColor Red
    exit 1
}

$nupkg = Get-ChildItem -Path $nugetRoot -Filter "*.nupkg" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($nupkg) {
    Write-Host "`n[SUCCESS] Da tao package: $($nupkg.FullName)" -ForegroundColor Green
    Write-Host "Kich thuoc: $([math]::Round($nupkg.Length / 1MB, 2)) MB" -ForegroundColor Green
    Write-Host "`nLenh push (thay <YOUR_NUGET_API_KEY>):" -ForegroundColor Yellow
    Write-Host "nuget push $($nupkg.FullName) -ApiKey <YOUR_NUGET_API_KEY> -Source https://api.nuget.org/v3/index.json" -ForegroundColor White
}
