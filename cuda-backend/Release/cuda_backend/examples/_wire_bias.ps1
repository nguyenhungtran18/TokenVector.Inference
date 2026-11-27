$ErrorActionPreference='Stop'
cd 'D:\TokenVector.Inference\cuda-backend\Release\cuda_backend\examples'
$p='ilgpt2_real10.tkv'
$t=Get-Content $p -Raw
$t=$t.Replace("`r`n","`n")
function Rep2($s,$old,$new,$tag){
  $n=[regex]::Matches($s,[regex]::Escape($old)).Count
  Write-Host ($tag+' hits='+$n)
  if($n -ne 1){ throw $tag }
  return $s.Replace($old,$new)
}
# splice bias functions
$snip=Get-Content '_bias_snippet.tkv' -Raw
$a='def wrand(i: "i32", j: "i32", s: "f32") -> "f32":'
$t=Rep2 $t $a ($snip+$a) 'snippet'
# arrays
$a='    ALG2 = alloc_bytes(msz)'
$t=Rep2 $t $a ($a+@'

    ABQ = alloc_bytes(msz)
    ABO = alloc_bytes(msz)
    AB1 = alloc_bytes(msz)
    AB2 = alloc_bytes(msz)
'@) 'bias arrays'
# init handles after a16h init
$a='    sbk16: "i64" = read_i64(a16h, 8)'
$t=Rep2 $t $a ($a+@'

    bh: "i64" = bias_init()
    if bh == 0:
        print("BIAS INIT FAIL")
        return 1
    bhadd: "i64" = read_i64(bh, 0)
    bhgrad: "i64" = read_i64(bh, 8)
    h3: "i64" = 3 * hb
    h4: "i64" = 4 * hb
'@) 'bias handles'
# lsz includes 6 bias vectors
$t=Rep2 $t '    lsz: "i64" = 4 * whb + w1b + w2b + 4 * hb' '    lsz: "i64" = 4 * whb + w1b + w2b + 10 * hb' 'lsz'
# init loop: bias device buffers after LN weights
$a='        wh = cuda_h2d(read_i64(ALB2, ll * 8), hg + wo, hb, 1)'
$t=Rep2 $t $a ($a+@'

        rc = cuda_malloc(cell, 3 * hb)
        wh = write_i64(ABQ, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ABQ, ll * 8), hg + wo, 3 * hb, 1)
        wo = wo + 3 * hb
        rc = cuda_malloc(cell, hb)
        wh = write_i64(ABO, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ABO, ll * 8), hg + wo, hb, 1)
        wo = wo + hb
        rc = cuda_malloc(cell, 4 * hb)
        wh = write_i64(AB1, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(AB1, ll * 8), hg + wo, 4 * hb, 1)
        wo = wo + 4 * hb
        rc = cuda_malloc(cell, hb)
        wh = write_i64(AB2, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(AB2, ll * 8), hg + wo, hb, 1)
'@) 'bias buffers'
# forward pointer reads
$a='            lg2: "i64" = read_i64(ALG2, ll * 8)'
$t=Rep2 $t $a ($a+"`n            bq: `"i64`" = read_i64(ABQ, ll * 8)`n            bo: `"i64`" = read_i64(ABO, ll * 8)`n            bf1: `"i64`" = read_i64(AB1, ll * 8)`n            bf2: `"i64`" = read_i64(AB2, ll * 8)") 'bias ptrs fwd'
# forward bias adds after the 4 GEMMs
$old='            rc = gemm16(ch, 0, 0, T, 3 * H, H, one, fpA, T, fpQKV, H, zero, qd, T)'
$t=Rep2 $t $old ($old+"`n            rc = bias_add(bhadd, qd, bq, T, 3 * T * H)") 'bias qkv'
$old='            rc = gemm16(ch, 0, 0, T, H, H, one, fpA, T, fpB, H, zero, dO, T)'
$t=Rep2 $t $old ($old+"`n            rc = bias_add(bhadd, dO, bo, T, T * H)") 'bias o'
$old='            rc = gemm16(ch, 0, 0, T, H, F, one, fpA, T, fpB, F, zero, dO2, T)'
$t=Rep2 $t $old ($old+"`n            rc = bias_add(bhadd, dO2, bf2, T, T * H)") 'bias mlp2'
$old='            rc = gemm16(ch, 0, 0, T, F, H, one, fpA, T, fpB, H, zero, zd, T)'
$t=Rep2 $t $old ($old+"`n            rc = bias_add(bhadd, zd, bf1, T, T * F)") 'bias mlp1'
# backward pointer reads (second occurrence of lg2 assignment)
$a='            lg2 = read_i64(ALG2, ll * 8)'
$t=Rep2 $t $a ($a+"`n            bq = read_i64(ABQ, ll * 8)`n            bo = read_i64(ABO, ll * 8)`n            bf1 = read_i64(AB1, ll * 8)`n            bf2 = read_i64(AB2, ll * 8)") 'bias ptrs bwd'
# bias gradients after weight-gradient GEMMs
$old='            rc = gemm16(ch, 1, 0, H, F, T, one, fpA, T, fpB, T, zero, g1d, H)'
$t=Rep2 $t $old ($old+"`n            rc = bias_grad_k(bhgrad, g1d, bf1, T, F)") 'bias grad w1'
$old='            rc = gemm16(ch, 1, 0, H, H, T, one, fpA, T, fpB, T, zero, gWo, H)'
$t=Rep2 $t $old ($old+"`n            rc = bias_grad_k(bhgrad, gWo, bo, T, H)") 'bias grad wo'
$old='            rc = gemm16(ch, 1, 0, H, H, T, one, fpA, T, fpB, T, zero, gqd, H)'
$t=Rep2 $t $old ($old+"`n            rc = bias_grad_k(bhgrad, gqd, bq, T, H)`n            rc = bias_grad_k(bhgrad, gkd, bq + h3, T, H)`n            rc = bias_grad_k(bhgrad, gvd, bq + h3 + h3, T, H)") 'bias grad qkv'
$old='            rc = gemm16(ch, 1, 0, F, H, T, one, fpA, T, fpB, T, zero, g2d, F)'
$t=Rep2 $t $old ($old+"`n            rc = bias_grad_k(bhgrad, g2d, bf2, T, H)") 'bias grad w2'
Set-Content $p -Value $t -NoNewline -Encoding UTF8
& '..\..\dist\tkvc-tv.exe' $p 'ilgpt2_real10.exe' run --emit-il 2>&1 | Select-Object -First 6
