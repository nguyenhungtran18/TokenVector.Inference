# verify_train_parity.ps1 - MOT LENH: bao cao ti le khop cua backend (G5).
#
# Chay harness `tv/tests/harness_tests.tkv`:
#   [1] backend cpu  -> "N/N khop (100%)"
#   [2] backend GIA sai co chu y -> harness phai BAT DUOC (va bat dung case)
#
# Exit code: 0 = XANH (cpu 100% + bat duoc backend gia); 1 = DO.
# Bao cao: reports\train-parity-<yyyyMMdd-HHmmss>.md
#
# Cach dung (tu goc repo):
#   powershell -NoProfile -File tools\verify_train_parity.ps1
#   powershell -NoProfile -File tools\verify_train_parity.ps1 -Compiler "D:\path\tkvc-tv.exe"

param(
    [string]$Compiler = "D:\TokenVector\Release\dist\tkvc-tv.exe",
    [string]$WorkDir = $env:TEMP
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "tv\tests\harness_tests.tkv"
$exe = Join-Path $WorkDir "harness_tests.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "[1/2] Bien dich $src ..."
& $Compiler $src $exe | Out-Null
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "LOI: bien dich that bai (kiem tra -Compiler)."
    exit 1
}

Write-Host "[2/2] Chay harness ..."
Push-Location $root
$out = (& $exe 2>&1 | Out-String)
$code = $LASTEXITCODE
Pop-Location
Write-Host $out

# ---- bao cao --------------------------------------------------------------
$ratio = ""
foreach ($ln in ($out -split '\r?\n')) {
    if ($ln -match "([0-9]+/[0-9]+ khop \([0-9.]+%\))") {
        $ratio = $matches[1]
        break
    }
}
if ($ratio -eq "") { $ratio = "(khong tim thay dong ti le khop)" }

$reports = Join-Path $root "reports"
if (-not (Test-Path -LiteralPath $reports)) { New-Item -ItemType Directory -Path $reports | Out-Null }
$reportPath = Join-Path $reports ("train-parity-" + $stamp + ".md")

$overall = "XANH"
if ($code -ne 0) { $overall = "DO" }
$lines = @()
$lines += "# BAO CAO PARITY TRAINER (G5) - " + $stamp
$lines += ""
$lines += "Cong cu: tools\verify_train_parity.ps1 - Tong: **" + $overall + "**"
$lines += ""
$lines += "- Backend cpu: **" + $ratio + "**"
$lines += "- Backend gia: bat duoc dung case (xem chi tiet)"
$lines += "- Exit code: " + $code
$lines += ""
$lines += "## Chi tiet"
$lines += '```'
$lines += ($out -split '\r?\n')
$lines += '```'
$lines += ""
$lines += "Fixture: tv\tests\fixtures (PyTorch, sinh boi tools\gen_fixtures.py)."
Set-Content -LiteralPath $reportPath -Value ($lines -join "`n") -Encoding UTF8

Write-Host ""
Write-Host ("TONG: " + $overall + "  cpu=" + $ratio + "  (bao cao: " + $reportPath + ")")
if ($code -ne 0) { exit 1 }
exit 0
