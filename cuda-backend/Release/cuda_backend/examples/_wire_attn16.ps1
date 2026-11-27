$ErrorActionPreference='Stop'
cd 'D:\TokenVector.Inference\cuda-backend\Release\cuda_backend\examples'
$p='ilgpt2_trainN_clean.tkv'
$t=Get-Content $p -Raw
$t=$t.Replace("`r`n","`n")
function Rep2($s,$old,$new,$tag){
  $n=[regex]::Matches($s,[regex]::Escape($old)).Count
  Write-Host ($tag+' hits='+$n)
  if($n -ne 1){ throw $tag }
  return $s.Replace($old,$new)
}
# extern for strided batched GEMM
$a='    {"name": "cublas_gemm_ex", "dll": "cublas.dll", "symbol": "cublasGemmEx",
     "convention": "cdecl",
     "params": ["i64", "i32", "i32", "i32", "i32", "i32", "i64", "i64", "i32",
                "i32", "i64", "i32", "i32", "i64", "i64", "i32", "i32", "i32", "i32"],
     "returns": "i32"},
]'
$ext='    {"name": "cublas_gemm_strided_batched_ex", "dll": "cublas.dll", "symbol": "cublasGemmStridedBatchedEx",
     "convention": "cdecl",
     "params": ["i64", "i32", "i32", "i32", "i32", "i32", "i64", "i64", "i32", "i32",
                "i64", "i64", "i32", "i32", "i64", "i64", "i64", "i32", "i32", "i64",
                "i32", "i32", "i32"], "returns": "i32"},
]'
$t=Rep2 $t $a ($a.Substring(0,$a.Length-1)+$ext) 'extern'
# splice functions
$snip=Get-Content '_attn16_snippet.tkv' -Raw
$a='def wrand(i: "i32", j: "i32", s: "f32") -> "f32":'
$t=Rep2 $t $a ($snip+$a) 'snippet'
# init handles + constants after loss3 init
$a='    dGI: "i64" = read_i64(cell, 0)'
$t=Rep2 $t $a ($a+@'

    a16h: "i64" = attn16_init()
    if a16h == 0:
        print("ATTN16 INIT FAIL")
        return 1
    sff16: "i64" = read_i64(a16h, 0)
    sbk16: "i64" = read_i64(a16h, 8)
    s64T: "i64" = 64 * T
    sATT: "i64" = T * T
    hal16 = alloc_bytes(4)
    wh = write_f32(hal16, 0, 0.125)
'@) 'init handles'
# enlarge attention buffers
foreach($v in @('dS','dP','gP','gS')){
  $old='    rc = cuda_malloc(cell, sb)
    '+$v+' = read_i64(cell, 0)'
  $new='    rc = cuda_malloc(cell, 16 * sb)
    '+$v+' = read_i64(cell, 0)'
  $t=Rep2 $t $old $new ('enlarge '+$v)
}
$old='        rc = cuda_malloc(cell, sb)
        wh = write_i64(AP, ll * 8, read_i64(cell, 0))'
$t=Rep2 $t $old '        rc = cuda_malloc(cell, 16 * sb)
        wh = write_i64(AP, ll * 8, read_i64(cell, 0))' 'enlarge AP'
# forward attention
$old='            rc = gemm16(ch, 0, 1, T, T, H, hal, fpA, T, fpB, T, zero, dS, T)'
$t=Rep2 $t $old '            rc = gemm16b(ch, 0, 1, T, T, 64, hal16, fpA, T, s64T, fpB, T, s64T, zero, dS, T, sATT, 16)' 'fwd scores'
$t=Rep2 $t '            rc = sf_fwd(sff, dS, pd, T)' '            rc = sf16_fwd(sff16, dS, pd, T, 16)' 'fwd softmax'
$old='            rc = toh(th, pd, fpA, sTT)
            rc = toh(th, vd, fpB, sQH)
            rc = gemm16(ch, 0, 0, T, H, T, one, fpA, T, fpB, T, zero, ad, T)'
$new='            rc = toh(th, pd, fpA, 16 * sTT)
            rc = toh(th, vd, fpB, sQH)
            rc = gemm16b(ch, 0, 0, T, 64, T, one, fpA, T, sATT, fpB, T, s64T, zero, ad, T, s64T, 16)'
$t=Rep2 $t $old $new 'fwd av'
# backward attention
$old='            rc = gemm16(ch, 0, 1, T, T, H, one, fpA, T, fpB, T, zero, gP, T)
            rc = toh(th, pd, fpA, sTT)
            rc = toh(th, gA, fpB, sQH)
            rc = gemm16(ch, 1, 0, T, H, T, one, fpA, T, fpB, T, zero, gV, T)'
$new='            rc = gemm16b(ch, 0, 1, T, T, 64, one, fpA, T, s64T, fpB, T, s64T, zero, gP, T, sATT, 16)
            rc = toh(th, pd, fpA, 16 * sTT)
            rc = toh(th, gA, fpB, sQH)
            rc = gemm16b(ch, 1, 0, T, 64, T, one, fpA, T, sATT, fpB, T, s64T, zero, gV, T, s64T, 16)'
$t=Rep2 $t $old $new 'bwd gp gv'
$t=Rep2 $t '            rc = sf_bwd(sbk, gP, pd, gS, T)' '            rc = sf16_bwd(sbk16, gP, pd, gS, T, 16)' 'bwd softmax'
$old='            rc = toh(th, gS, fpA, sTT)
            rc = toh(th, kd, fpB, sQH)
            rc = gemm16(ch, 0, 0, T, H, T, hal, fpA, T, fpB, T, zero, gQ, T)'
$new='            rc = toh(th, gS, fpA, 16 * sTT)
            rc = toh(th, kd, fpB, sQH)
            rc = gemm16b(ch, 0, 0, T, 64, T, hal16, fpA, T, sATT, fpB, T, s64T, zero, gQ, T, s64T, 16)'
$t=Rep2 $t $old $new 'bwd gq'
$old='            rc = toh(th, gS, fpA, sTT)
            rc = toh(th, qd, fpB, sQH)
            rc = gemm16(ch, 1, 0, T, H, T, hal, fpA, T, fpB, T, zero, gK, T)'
$new='            rc = toh(th, gS, fpA, 16 * sTT)
            rc = toh(th, qd, fpB, sQH)
            rc = gemm16b(ch, 1, 0, T, 64, T, hal16, fpA, T, sATT, fpB, T, s64T, zero, gK, T, s64T, 16)'
$t=Rep2 $t $old $new 'bwd gk'
Set-Content $p -Value $t -NoNewline -Encoding UTF8
& '..\..\dist\tkvc-tv.exe' $p 'ilgpt2_trainN_head16.exe' run --emit-il 2>&1 | Select-Object -First 6
