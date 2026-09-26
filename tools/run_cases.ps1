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
#   powershell -File tools\run_cases.ps1 -Compiler "D:\path\to\tkvc.exe"
param(
    [string]$Compiler = "D:\TokenVector\3.code\dist\tkvc_patched.exe",
    [string]$WorkDir = $env:TEMP
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "tv\tests\inference_tests.tkv"
$exe = Join-Path $WorkDir "inference_cases.exe"

Write-Host "[1/2] Bien dich $src ..."
& $Compiler build $src --out $exe | Out-Null
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

Write-Host "[2/2] Chay $($names.Count) case ..."
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
