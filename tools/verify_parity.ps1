# verify_parity.ps1 - XAC NHAN LAI KET QUA PARITY - 3 lap doc lap.
#
# De xuat trong docs/accumulation-order-vs-pytorch-2026-10-18.md (muc 5).
#   [1/3] FIXTURE ORACLE : sinh lai fixture tu PyTorch vao thu muc tam,
#                          diff o moi dong voi ban da check-in.
#                          Phat hien: fixture drift, torch upgrade doi bai doi chieu.
#   [2/3] BANG THU TU TICH LUY: chay benchmark\probe_accumulation_order.py,
#                          so cac so bit-diff voi BASELINE da chot.
#                          Phat hien: MKL/torch dispatch doi (AVX2 -> AVX512...).
#   [3/3] GATE TOLERANCE  : chay tools\run_cases.ps1 (test tu kiem tol 1e-5).
#                          Phat hien: regression code .tkv.
#
# Exit code: 0 = khong FAIL (SKIP duoc phep, co bao ro); 1 = co it nhat 1 FAIL.
# Bao cao: reports\parity-<yyyyMMdd-HHmmss>.md
#
# Cach dung (tu goc repo):
#   powershell -NoProfile -File tools\verify_parity.ps1
#   powershell -NoProfile -File tools\verify_parity.ps1 -Python "C:\py\python.exe"
#   powershell -NoProfile -File tools\verify_parity.ps1 -SkipRegen   # bo qua [1/3]
#
# [1/3] va [2/3] can python + torch (+transformers cho fixture GPT-2);
# thieu thi tu SKIP co ly do; [3/3] van chay (khong phu thuoc python).

param(
    [string]$Python = "python",
    [switch]$SkipRegen
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reports = Join-Path $root "reports"
if (-not (Test-Path -LiteralPath $reports)) { New-Item -ItemType Directory -Path $reports | Out-Null }
$reportPath = Join-Path $reports ("parity-" + $stamp + ".md")

# Baseline da chot trong docs/accumulation-order-vs-pytorch-2026-10-18.md (muc 1)
# tren may do: torch 2.14.0+cpu, MKL 2026.1, AVX2. Doi so = moi truong doi.
$BASE_MM_SEQ  = 1499   # mm [8,64]@[64,256] tuan tu vs torch
$BASE_MM_4X4  = 8      # mm [4,4] tuan tu vs torch
$BASE_MEANSEQ = "True" # mean(64) tuan tu khac torch
$BASE_EXPDR   = 0      # exp f32: torch vs double-round
$BASE_SQRT    = 0      # sqrt f32 khac 0/64

$lines = @()
$results = @()

# Fence markdown bang bien (khong de tay: `"`n````" se bi backtick thu 3 escape
# dau nhay dong -> chuoi khong dong duoc, loi parser da gap khi thu).
$NL = "`n"
$BT = [string][char]96
$FO = $NL + $BT + $BT + $BT + "`n"
$FC = $NL + $BT + $BT + $BT

function Add-Result($id, $status, $detail) {
    $script:results += [pscustomobject]@{ Id = $id; Status = $status; Detail = $detail }
    $script:lines += ("| " + $id + " | " + $status + " | " + $detail + " |")
    Write-Host ("  -> [" + $id + "] " + $status + " " + $detail)
}

Push-Location $root
try {

# ==== [1/3] FIXTURE ORACLE ==============================================
Write-Host "[1/3] Fixture oracle: sinh lai tu PyTorch + diff voi ban check-in ..."
if ($SkipRegen) {
    Add-Result "1-fixture" "SKIP" "(-SkipRegen)"
} else {
    $pyCmd = Get-Command $Python -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Add-Result "1-fixture" "SKIP" "khong tim thay python: $Python"
    } else {
        & $Python -c "import torch, transformers" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Add-Result "1-fixture" "SKIP" "thieu torch/transformers (chay duoc o may cai day du)"
        } else {
            $tmp = Join-Path $env:TEMP ("parity_regen_" + $stamp)
            if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force $tmp }
            $out1 = (& $Python (Join-Path $root "tools\gen_fixtures.py") --out $tmp 2>&1 | Out-String)
            $e1 = $LASTEXITCODE
            $out2 = (& $Python (Join-Path $root "tools\gen_gpt2_fixture.py") --out $tmp 2>&1 | Out-String)
            $e2 = $LASTEXITCODE
            if ($e1 -ne 0 -or $e2 -ne 0) {
                Add-Result "1-fixture" "FAIL" ("sinh fixture loi (gen_fixtures=" + $e1 + " gen_gpt2=" + $e2 + ") xem log: " + $tmp)
                $tail = (($out1 + $out2) -split "`n" | Select-Object -Last 15) -join "`n"
                $lines += ($FO + $tail + $FC)
            } else {
                $cmp = (& $Python (Join-Path $root "tools\compare_fixtures.py") (Join-Path $root "tv\tests\fixtures") $tmp 2>&1 | Out-String)
                $e3 = $LASTEXITCODE
                $lines += ($FO + $cmp.Trim() + $FC)
                Write-Host ($cmp -replace "\s+", " ").Trim()
                if ($e3 -eq 0) {
                    Add-Result "1-fixture" "PASS" "fixture check-in khop 100% voi ban sinh lai moi"
                } else {
                    Add-Result "1-fixture" "FAIL" ("fixture drift, xem bao cao. Tam: " + $tmp)
                }
                if ($e3 -eq 0 -and (Test-Path -LiteralPath $tmp)) {
                    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
                }
            }
        }
    }
}

# ==== [2/3] BANG THU TU TICH LUY ========================================
Write-Host "[2/3] Bang thu tu tich luys: chay probe_accumulation_order.py ..."
$pyCmd2 = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pyCmd2) {
    Add-Result "2-order" "SKIP" "khong tim thay python: $Python"
} else {
    $probeOut = (& $Python (Join-Path $root "benchmark\probe_accumulation_order.py") 2>&1 | Out-String)
    $probeExit = $LASTEXITCODE
    $lines += ($FO + $probeOut.Trim() + $FC)
    if ($probeExit -ne 0) {
        Add-Result "2-order" "FAIL" ("probe loi exit=" + $probeExit)
    } else {
        $mismatches = @()
        $m = [regex]::Match($probeOut, "matmul \[8,64\]@\[64,256\]: bit-diff elements = (\d+) / 2048")
        if (-not $m.Success -or [int]$m.Groups[1].Value -ne $BASE_MM_SEQ) {
            $mismatches += ("mm-seq=" + $(if ($m.Success) { $m.Groups[1].Value } else { "?" }) + " (base " + $BASE_MM_SEQ + ")")
        }
        $m = [regex]::Match($probeOut, "matmul \[4,4\]: bit-diff = (\d+)")
        if (-not $m.Success -or [int]$m.Groups[1].Value -ne $BASE_MM_4X4) {
            $mismatches += ("mm-4x4=" + $(if ($m.Success) { $m.Groups[1].Value } else { "?" }) + " (base " + $BASE_MM_4X4 + ")")
        }
        $m = [regex]::Match($probeOut, "mean\(64\):.*bitdiff=(\w+)")
        if (-not $m.Success -or $m.Groups[1].Value -ne $BASE_MEANSEQ) {
            $mismatches += ("mean-seq=" + $(if ($m.Success) { $m.Groups[1].Value } else { "?" }) + " (base " + $BASE_MEANSEQ + ")")
        }
        $m = [regex]::Match($probeOut, "torch vs double-round bitdiff=(\d+)")
        if (-not $m.Success -or [int]$m.Groups[1].Value -ne $BASE_EXPDR) {
            $mismatches += ("exp-doubleround=" + $(if ($m.Success) { $m.Groups[1].Value } else { "?" }) + " (base " + $BASE_EXPDR + ")")
        }
        $m = [regex]::Match($probeOut, "sqrt f32\s+: bitdiff=(\d+)/64")
        if (-not $m.Success -or [int]$m.Groups[1].Value -ne $BASE_SQRT) {
            $mismatches += ("sqrt=" + $(if ($m.Success) { $m.Groups[1].Value } else { "?" }) + " (base " + $BASE_SQRT + ")")
        }
        $avx = [regex]::Match($probeOut, "CPU capability usage: (\w+)")
        $cpu = "unknown"
        if ($avx.Success) { $cpu = $avx.Groups[1].Value }
        if ($mismatches.Count -eq 0) {
            Add-Result "2-order" "PASS" ("khop baseline (torch/MKL tai van, CPU " + $cpu + ")")
        } else {
            Add-Result "2-order" "FAIL" ("ket qua doi so baseline, moi truong da doi: " + ($mismatches -join "; "))
        }
    }
}

# ==== [3/3] GATE TOLERANCE ==============================================
Write-Host "[3/3] Gate tolerance: chay run_cases.ps1 (cac test tu kiem 1e-5) ..."
$rcLog = Join-Path $env:TEMP ("parity_runcases_" + $stamp + ".log")
& powershell -NoProfile -File (Join-Path $root "tools\run_cases.ps1") *> $rcLog
$rcExit = $LASTEXITCODE
$rcText = ""
if (Test-Path -LiteralPath $rcLog) { $rcText = Get-Content -Raw -LiteralPath $rcLog }
$total = ""
if ($rcText -match "TOTAL: pass=\d+ fail=\d+ crash=\d+") { $total = $matches[0] }
$g2 = ""
if ($rcText -match "GPT-2 cases=\d+ fails=\d+") { $g2 = $matches[0] }
if ($rcText) {
    $key = (($rcText -split "`r?`n" | Select-String "TOTAL:|PASS \(16|cases=.*fails=" | ForEach-Object { $_.Line.Trim() }) -join "`n")
    $lines += ($FO + $key + $FC)
}
if ($rcExit -eq 0) {
    Add-Result "3-gates" "PASS" ($total + " | " + $g2)
} else {
    Add-Result "3-gates" "FAIL" ("run_cases exit=" + $rcExit + " " + $total)
}

} finally {
    Pop-Location
}

# ==== BAO CAO ============================================================
$fail = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
$skip = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
$overall = "XANH"
if ($fail -gt 0) { $overall = "DO (" + $fail + " FAIL)" }
elseif ($skip -gt 0) { $overall = "XANH CO SKIP (" + $skip + ")" }

$report = @()
$report += "# BAO CAO PARITY - " + $stamp
$report += ""
$report += "Cong cu: tools\verify_parity.ps1 - Tong: **" + $overall + "**"
$report += ""
$report += "| Lap | Trang thai | Chi tiet |"
$report += "|---|---|---|"
$report += $lines | Where-Object { $_ -match "^\| " }
$report += ""
$report += "## Chi tiet"
$report += $lines | Where-Object { $_ -notmatch "^\| " }
$report += ""
$report += "Ref: docs\accumulation-order-vs-pytorch-2026-10-18.md - baseline do tren torch 2.14.0+cpu / MKL 2026.1 / AVX2."
New-Item -ItemType Directory -Force -Path $reports | Out-Null
Set-Content -LiteralPath $reportPath -Value ($report -join "`n") -Encoding UTF8
Write-Host ""
Write-Host ("TONG: " + $overall + "  (bao cao: " + $reportPath + ")")
if ($fail -gt 0) { exit 1 }
exit 0
