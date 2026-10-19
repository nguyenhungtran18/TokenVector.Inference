# verify_gpt2_124m.ps1 — PARITY GPT-2 124M THAT (fixture NGOAI REPO).
#
# Mo khoa BLOCKER 124M trong HANDOFF.md 0.6. Thiet ke:
# docs/handoff-gpt2-124m-fixture-design.md
#
# Cach dung (tu goc repo):
#   powershell -NoProfile -File tools\verify_gpt2_124m.ps1
#   powershell -NoProfile -File tools\verify_gpt2_124m.ps1 -Regen         # sinh lai fixture
#   powershell -NoProfile -File tools\verify_gpt2_124m.ps1 -FixDir D:\tv124m
#
# Fixture mac dinh: tv\tests\fixtures-124m (trong repo, .gitignore — khong
# push len GitHub duoc vi file 119-231MB vuot gioi han 100MB khong co LFS).
#
# - Fixture ~2.3 GB text (shard 15M gia tri/man - file >~594MB bi .NET
#   OutOfMemoryException luc split, da do 19-10-2026) nam NGOAI repo.
# - Gate doc duong dan qua pointer tv\tests\g2_full_dir.txt (argv() khong
#   hoat dong tren build tkvc hien tai - da probe: argc=0).
# - Khong co python/torch/transformers va chua co fixture -> SKIP (exit 0).
# - Exit: 0 = khong FAIL (SKIP duoc phep), 1 = co FAIL.

param(
    [string]$Python = "python",
    [string]$FixDir = "",
    [switch]$Regen
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
# Mac dinh: thu muc 124m trong repo (da .gitignore) — khong dung %TEMP% nua
# (TEMP hay bi don boi he dieu hanh -> gate SKIP bat ngo). -FixDir van cho
# phep chi vao thu muc khac.
if ($FixDir -eq "") { $FixDir = Join-Path $root "tv\tests\fixtures-124m" }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reports = Join-Path $root "reports"
if (-not (Test-Path -LiteralPath $reports)) { New-Item -ItemType Directory -Path $reports | Out-Null }
$reportPath = Join-Path $reports ("gpt2-124m-" + $stamp + ".md")
$pointer = Join-Path $root "tv\tests\g2_full_dir.txt"
$compiler = "D:\TokenVector\Release\dist\tkvc-tv.exe"

$NL = "`n"
$BT = [string][char]96
$FO = $NL + $BT + $BT + $BT + "`n"
$FC = $NL + $BT + $BT + $BT

$lines = @()
$results = @()
function Add-Result($id, $status, $detail) {
    $script:results += [pscustomobject]@{ Id = $id; Status = $status; Detail = $detail }
    $script:lines += ("| " + $id + " | " + $status + " | " + $detail + " |")
    Write-Host ("  -> [" + $id + "] " + $status + " " + $detail)
}

Push-Location $root
try {

# ==== [1/4] FIXTURE (gen ngoai repo) ==================================
Write-Host "[1/4] Fixture 124M: kiem tra/sinh tai $FixDir ..."
$hasFix = (Test-Path -LiteralPath (Join-Path $FixDir "g2_shards.txt")) -and
          (Test-Path -LiteralPath (Join-Path $FixDir "g2_grad_wte.txt.0"))
if ($hasFix -and -not $Regen) {
    Add-Result "1-fixture" "PASS" ("da co tai " + $FixDir)
} else {
    $pyCmd = Get-Command $Python -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Add-Result "1-fixture" "SKIP" ("khong tim thay python: " + $Python + " (va khong co fixture san)")
    } else {
        & $Python -c "import torch, transformers" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Add-Result "1-fixture" "SKIP" "thieu torch/transformers (chay duoc o may cai day du)"
        } else {
            $out = (& $Python (Join-Path $root "tools\gen_gpt2_fixture.py") --full --out $FixDir 2>&1 | Out-String)
            if ($LASTEXITCODE -ne 0) {
                Add-Result "1-fixture" "FAIL" ("gen_gpt2_fixture --full loi exit=" + $LASTEXITCODE)
                $lines += ($FO + (($out -split "`n") | Select-Object -Last 15) -join "`n" + $FC)
            } else {
                Add-Result "1-fixture" "PASS" ("da sinh tai " + $FixDir)
            }
        }
    }
}

# ==== [2/4] POINTER ====================================================
Write-Host "[2/4] Ghi pointer $pointer ..."
if (Test-Path -LiteralPath $pointer) { Remove-Item -Force -LiteralPath $pointer }
$ok = $true
if (Test-Path -LiteralPath (Join-Path $FixDir "g2_shards.txt")) {
    Set-Content -LiteralPath $pointer -Value $FixDir -NoNewline -Encoding ASCII
    Add-Result "2-pointer" "PASS" $FixDir
} else {
    Set-Content -LiteralPath $pointer -Value "MISSING" -NoNewline -Encoding ASCII
    $ok = $false
    Add-Result "2-pointer" "SKIP" "fixture chua co -> gate se SKIP"
}

# ==== [3/4] BIEN DICH + CHAY GATE ======================================
Write-Host "[3/4] Bien dich + chay gate gpt2_full_tests ..."
if (-not (Test-Path -LiteralPath $compiler)) {
    Add-Result "3-gate" "SKIP" ("khong tim thay compiler: " + $compiler)
} else {
    $exe = Join-Path $env:TEMP ("gpt2_full_tests_" + $stamp + ".exe")
    $cLog = (& $compiler "tv\tests\gpt2_full_tests.tkv" $exe "main" 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exe)) {
        Add-Result "3-gate" "FAIL" "compile loi"
        $lines += ($FO + $cLog.Trim() + $FC)
    } else {
        $gOut = (& $exe 2>&1 | Out-String)
        $gExit = $LASTEXITCODE
        $lines += ($FO + $gOut.Trim() + $FC)
        $m = [regex]::Match($gOut, "cases=(\d+) fails=(\d+)")
        if ($gExit -ne 0) {
            Add-Result "3-gate" "FAIL" ("gate exit=" + $gExit + " " + $m.Value)
        } elseif ($gOut -match "\[SKIP\]") {
            Add-Result "3-gate" "SKIP" "gate SKIP (fixture thieu)"
        } elseif ($m.Success -and [int]$m.Groups[2].Value -eq 0) {
            Add-Result "3-gate" "PASS" ("PARITY 124M " + $m.Value)
        } else {
            Add-Result "3-gate" "FAIL" $m.Value
        }
        Remove-Item -Force -LiteralPath $exe -ErrorAction SilentlyContinue
    }
}

# ==== [4/4] GATE NHO KHONG HOI QUY =====================================
Write-Host "[4/4] Gate nho (gpt2_tests) khong hoi quy ..."
$exe2 = Join-Path $env:TEMP ("gpt2_small_" + $stamp + ".exe")
$c2 = (& $compiler "tv\tests\gpt2_tests.tkv" $exe2 "main" 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exe2)) {
    Add-Result "4-small" "FAIL" "compile loi"
} else {
    $o2 = (& $exe2 2>&1 | Out-String)
    if ($o2 -match "cases=(\d+) fails=(\d+)" -and [int]$matches[2] -eq 0) {
        Add-Result "4-small" "PASS" $matches[0]
    } else {
        Add-Result "4-small" "FAIL" (($o2 -split "`n" | Select-Object -Last 2) -join " ")
    }
    Remove-Item -Force -LiteralPath $exe2 -ErrorAction SilentlyContinue
}

} finally {
    Pop-Location
}

# ==== BAO CAO ==========================================================
$fail = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
$skip = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
$overall = "XANH"
if ($fail -gt 0) { $overall = "DO (" + $fail + " FAIL)" }
elseif ($skip -gt 0) { $overall = "XANH CO SKIP (" + $skip + ")" }

$report = @()
$report += "# BAO CAO PARITY GPT-2 124M - " + $stamp
$report += ""
$report += "Cong cu: tools\verify_gpt2_124m.ps1 - Tong: **" + $overall + "**"
$report += ""
$report += "| Buoc | Trang thai | Chi tiet |"
$report += "|---|---|---|"
$report += $lines | Where-Object { $_ -match "^\| " }
$report += ""
$report += "## Chi tiet"
$report += $lines | Where-Object { $_ -notmatch "^\| " }
$report += ""
$report += "Thiet ke: docs\handoff-gpt2-124m-fixture-design.md - fixture ~2.3 GB text shard 15M/man"
$report += "(nguong .NET ~594MB: file 594MB OutOfMemory luc split, 533MB pass - do 19-10-2026)."
New-Item -ItemType Directory -Force -Path $reports | Out-Null
Set-Content -LiteralPath $reportPath -Value ($report -join "`n") -Encoding UTF8
Write-Host ""
Write-Host ("TONG: " + $overall + "  (bao cao: " + $reportPath + ")")
if ($fail -gt 0) { exit 1 }
exit 0
