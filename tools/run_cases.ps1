# Chay tung test case trong MOT process rieng.
#
# LY DO (2026-09-26): co case nem AccessViolationException. TokenVector
# KHONG bat duoc loai nay bang 'except Exception' (chi bat duoc Exception),
# nen chay ca bo trong 1 process se bi 1 case lam treo - tat ca case phia
# sau khong bao gio chay duoc. Chay moi case trong 1 process rieng thi
# crash chi doc 1 case do.
#
# Cach dung:
#   powershell -File tools\run_cases.ps1
#   powershell -File tools\run_cases.ps1 -Compiler "D:\path\to\tkvc-tv.exe"
#
# (2026-10-15) Compiler da duoc viet lai (self-host). CLI moi KHONG con
# subcommand `build` va KHONG con `--out`:
#   tkvc-tv.exe <nguon.tkv> <ra.exe> [ten_ham_vao]
param(
    [string]$Compiler = "D:\TokenVector\Release\dist\tkvc-tv.exe",
    [string]$WorkDir = $env:TEMP
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "tv\tests\inference_tests.tkv"
$exe = Join-Path $WorkDir "inference_cases.exe"

Write-Host "[1/3] Bien dich $src ..."
& $Compiler $src $exe | Out-Null
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "LOI: bien dich that bai."
    exit 1
}

$names = @(
    "generate_loop_text", "writer_modelproto", "constant_fold_translate",
    "gathernd_scatter", "range_eye_cumsum", "mish_prelu_instancenorm",
    "speculative_all_accept", "tools_parse_validate", "autograd_trains_step",
    "grammar_prefix_mask", "if_loop_end_to_end", "awq_asym_and_fp16",
    "autograd_gradcheck_and_adam", "pack_mask_and_preprocess", "lstm_unrolled",
    "resize_nearest_einsum", "qlinear_conv_matmulinteger", "lora_forward_and_step",
    "hal_matmul_relu", "string_tensor_pack", "string_ops_side_table",
    "scan_run_driver", "prefix_skip_plan", "beam_log_softmax_tree",
    "vit_graph_builds", "gguf_k_quant_gate"
)

Write-Host "[2/3] Chay $($names.Count) case ..."
$rows = @()
for ($i = 1; $i -le $names.Count; $i++) {
    $out = (& $exe $i 2>&1 | Out-String)
    if ($out -match "\[PASS\]") {
        $status = "PASS"
    } elseif ($out -match "\[FAIL\]") {
        $status = "FAIL"
    } else {
        $status = "CRASH"
    }
    $detail = ""
    if ($status -eq "FAIL" -and $out -match "\[FAIL\][^\r\n]*") {
        $detail = ($matches[0] -replace "\[FAIL\]\s*", "" -replace "\s+", " ")
    } elseif ($status -eq "CRASH" -and $out -match "(Unhandled Exception:[^\r\n]*)") {
        $detail = ($matches[1] -replace "\s+", " ")
    }
    $rows += [pscustomobject]@{
        Idx    = $i
        Name   = $names[$i - 1]
        Status = $status
        Detail = $detail
    }
}

$rows | ForEach-Object { "{0,-5} {1,-32} {2}" -f $_.Idx, $_.Name, $_.Status }
""
$p = @($rows | Where-Object { $_.Status -eq "PASS" }).Count
$f = @($rows | Where-Object { $_.Status -eq "FAIL" }).Count
$c = @($rows | Where-Object { $_.Status -eq "CRASH" }).Count
Write-Host "TOTAL: pass=$p fail=$f crash=$c"
""
Write-Host "Chi tiet loi:"
$rows | Where-Object { $_.Status -ne "PASS" } |
    ForEach-Object { "  {0}. {1}`n     {2}" -f $_.Idx, $_.Name, $_.Detail }

# [3/3] gap_tests: mirror xUnit (protobuf, calibrate, quantize, fp8, batcher).
# Chay rieng 1 process (giong main suite) de crash khong lan.
$gapSrc = Join-Path $root "tv\tests\gap_tests.tkv"
$gapExe = Join-Path $WorkDir "gap_cases.exe"
Write-Host ""
Write-Host "[3/3] Bien dich + chay gap_tests ..."
& $Compiler $gapSrc $gapExe | Out-Null
if (-not (Test-Path -LiteralPath $gapExe)) {
    Write-Host "LOI: bien dich gap_tests that bai."
    exit 1
}
$gapOut = (& $gapExe 2>&1 | Out-String)
if ($gapOut -match "gap tests: PASS") {
    Write-Host "gap tests: PASS (16/16)"
} else {
    Write-Host "gap tests: FAIL"
    Write-Host ($gapOut -replace "\s+", " ").Trim()
    exit 1
}

# [4/4] parity_tests: doi chieu thuat toan voi PyTorch bang fixture tinh san.
# Fixture sinh boi tools\gen_fixtures.py (chay tren may co torch), check-in san
# vao tv\tests\fixtures\ nen test khong can torch. Xem docs/ROADMAP_TRAIN.md muc 5.
$parSrc = Join-Path $root "tv\tests\parity_tests.tkv"
$parExe = Join-Path $WorkDir "parity_cases.exe"
Write-Host ""
Write-Host "[4/4] Bien dich + chay parity_tests (doi chieu PyTorch) ..."
& $Compiler $parSrc $parExe | Out-Null
if (-not (Test-Path -LiteralPath $parExe)) {
    Write-Host "LOI: bien dich parity_tests that bai."
    exit 1
}
$parOut = (& $parExe 2>&1 | Out-String)
Write-Host ($parOut -replace "\s+", " ").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "parity tests: FAIL"
    exit 1
}

# [5/5] grad_tests: doi chieu BACKWARD tung op voi PyTorch autograd (fixture
# tinh san). Xem docs/ROADMAP_TRAIN.md G1.
$grSrc = Join-Path $root "tv\tests\grad_tests.tkv"
$grExe = Join-Path $WorkDir "grad_cases.exe"
Write-Host ""
Write-Host "[5/5] Bien dich + chay grad_tests (gradient vs PyTorch) ..."
& $Compiler $grSrc $grExe | Out-Null
if (-not (Test-Path -LiteralPath $grExe)) {
    Write-Host "LOI: bien dich grad_tests that bai."
    exit 1
}
$grOut = (& $grExe 2>&1 | Out-String)
Write-Host ($grOut -replace "\s+", " ").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "grad tests: FAIL"
    exit 1
}

# [6/6] tape_tests: backward TONG HOP qua chuoi op nhieu tang.
$tpSrc = Join-Path $root "tv\tests\tape_tests.tkv"
$tpExe = Join-Path $WorkDir "tape_cases.exe"
Write-Host ""
Write-Host "[6/6] Bien dich + chay tape_tests (backward tong hop) ..."
& $Compiler $tpSrc $tpExe | Out-Null
if (-not (Test-Path -LiteralPath $tpExe)) {
    Write-Host "LOI: bien dich tape_tests that bai."
    exit 1
}
$tpOut = (& $tpExe 2>&1 | Out-String)
Write-Host ($tpOut -replace "\s+", " ").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "tape tests: FAIL"
    exit 1
}

# [7/7] gc_tests: gradcheck bang SAI PHAY — kiem chung backward doc lap
# voi PyTorch (so voi dao phan cua chinh ham forward).
$gcSrc = Join-Path $root "tv\tests\gc_tests.tkv"
$gcExe = Join-Path $WorkDir "gc_cases.exe"
Write-Host ""
Write-Host "[7/7] Bien dich + chay gc_tests (gradcheck sai phay) ..."
& $Compiler $gcSrc $gcExe | Out-Null
if (-not (Test-Path -LiteralPath $gcExe)) {
    Write-Host "LOI: bien dich gc_tests that bai."
    exit 1
}
$gcOut = (& $gcExe 2>&1 | Out-String)
Write-Host ($gcOut -replace "\s+", " ").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "gc tests: FAIL"
    exit 1
}

# [8/8] gpt2_tests: doi chieu KIEN TRUC GPT-2 (forward + hidden tung tang +
# loss) voi fixture sinh boi tools\gen_gpt2_fixture.py tu HF `gpt2` (G2).
$g2Src = Join-Path $root "tv\tests\gpt2_tests.tkv"
$g2Exe = Join-Path $WorkDir "gpt2_cases.exe"
Write-Host ""
Write-Host "[8/8] Bien dich + chay gpt2_tests (GPT-2 vs HF) ..."
& $Compiler $g2Src $g2Exe | Out-Null
if (-not (Test-Path -LiteralPath $g2Exe)) {
    Write-Host "LOI: bien dich gpt2_tests that bai."
    exit 1
}
$g2Out = (& $g2Exe 2>&1 | Out-String)
Write-Host ($g2Out -replace "\s+", " ").Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "gpt2 tests: FAIL"
    exit 1
}



