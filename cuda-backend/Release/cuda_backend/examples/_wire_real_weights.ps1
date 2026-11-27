$ErrorActionPreference='Stop'
cd 'D:\TokenVector.Inference\cuda-backend\Release\cuda_backend\examples'
$p='ilgpt2_trainN_head16.tkv'
$t=Get-Content $p -Raw
$t=$t.Replace("`r`n","`n")
function Rep2($s,$old,$new,$tag){
  $n=[regex]::Matches($s,[regex]::Escape($old)).Count
  Write-Host ($tag+' hits='+$n)
  if($n -ne 1){ throw $tag }
  return $s.Replace($old,$new)
}
# 1) libc file externs
$a=']'
$ext='    {"name": "c_fopen", "dll": "libc.dll", "symbol": "fopen",
     "convention": "cdecl", "params": ["i64", "i64"], "returns": "i64"},
    {"name": "c_fread", "dll": "libc.dll", "symbol": "fread",
     "convention": "cdecl", "params": ["i64", "i64", "i64", "i64"], "returns": "i64"},
    {"name": "c_fclose", "dll": "libc.dll", "symbol": "fclose",
     "convention": "cdecl", "params": ["i64"], "returns": "i32"},
]'
$i=$t.IndexOf('__tkv_extern_pinvoke__ = [')
$j=$t.IndexOf("`n]`n",$i)
if($j -lt 0){ throw 'extern list end' }
$extLines=$ext.Substring(0,$ext.LastIndexOf(']'))
$t=$t.Substring(0,$j+1)+$extLines+$t.Substring($j+1)
# 2) per-layer LN arrays
$a='    AA = alloc_bytes(msz)'
$t=Rep2 $t $a ($a+@'

    ALG1 = alloc_bytes(msz)
    ALB1 = alloc_bytes(msz)
    ALG2 = alloc_bytes(msz)
    ALB2 = alloc_bytes(msz)
'@) 'LN arrays'
# 3) final LN device buffers after dbet allocation
$a='    dbet = read_i64(cell, 0)'
$t=Rep2 $t $a ($a+@'

    rc = cuda_malloc(cell, hb)
    dFG: "i64" = read_i64(cell, 0)
    rc = cuda_malloc(cell, hb)
    dFB: "i64" = read_i64(cell, 0)
'@) 'final LN buffers'
# 4) read whole weight file before layer init loop
$a='    ll: "i32" = 0
    while ll < NL:
        rc = cuda_malloc(cell, whb)'
$read=@'
    lsz: "i64" = 4 * whb + w1b + w2b + 4 * hb
    hg: "i64" = alloc_bytes(qb + 2 * hb + NL * lsz + wh2)
    fpb = b"/kaggle/working/gpt2m_weights.bin"
    fpath = alloc_bytes(64)
    fp: "i32" = 0
    while fp < len(fpb):
        wh = write_u8(fpath, fp, fpb[fp])
        fp = fp + 1
    wh = write_u8(fpath, fp, 0)
    mode = alloc_bytes(4)
    wh = write_u8(mode, 0, 114)
    wh = write_u8(mode, 1, 0)
    fh: "i64" = c_fopen(fpath, mode)
    if fh == 0:
        print("WEIGHT FILE OPEN FAIL")
        return 1
    nr: "i64" = c_fread(hg, 1, qb + 2 * hb + NL * lsz + wh2, fh)
    print("WEIGHT BYTES", nr)
    rc = c_fclose(fh)
    wh = cuda_h2d(dX, hg, qb, 1)
    wh = cuda_h2d(dFG, hg + qb, hb, 1)
    wh = cuda_h2d(dFB, hg + qb + hb, hb, 1)
    wo: "i64" = 0
'@
$t=Rep2 $t $a ($read+"`n"+$a) 'weight file read'
# 5) replace random weight init with h2d from hg, plus per-layer LN device buffers
$start=$t.IndexOf('        hwq: "i64" = alloc_bytes(whb)')
$end=$t.IndexOf('        wh = cuda_h2d(w2d, hw2, w2b, 1)',$start)
if($start -lt 0 -or $end -lt 0){ throw 'weight init block' }
$end=$t.IndexOf("`n",$end)+1
$new=@'
        wo = qb + 2 * hb + ll * lsz
        wh = cuda_h2d(wqd, hg + wo, whb, 1)
        wo = wo + whb
        wh = cuda_h2d(wkd, hg + wo, whb, 1)
        wo = wo + whb
        wh = cuda_h2d(wvd, hg + wo, whb, 1)
        wo = wo + whb
        wh = cuda_h2d(wod, hg + wo, whb, 1)
        wo = wo + whb
        wh = cuda_h2d(w1d, hg + wo, w1b, 1)
        wo = wo + w1b
        wh = cuda_h2d(w2d, hg + wo, w2b, 1)
        wo = wo + w2b
        rc = cuda_malloc(cell, hb)
        wh = write_i64(ALG1, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ALG1, ll * 8), hg + wo, hb, 1)
        wo = wo + hb
        rc = cuda_malloc(cell, hb)
        wh = write_i64(ALB1, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ALB1, ll * 8), hg + wo, hb, 1)
        wo = wo + hb
        rc = cuda_malloc(cell, hb)
        wh = write_i64(ALG2, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ALG2, ll * 8), hg + wo, hb, 1)
        wo = wo + hb
        rc = cuda_malloc(cell, hb)
        wh = write_i64(ALB2, ll * 8, read_i64(cell, 0))
        wh = cuda_h2d(read_i64(ALB2, ll * 8), hg + wo, hb, 1)
'@
$t=$t.Substring(0,$start)+$new+"`n"+$t.Substring($end)
# 6) head weight from file
$start=$t.IndexOf('    hWh = alloc_bytes(wh2)')
$end=$t.IndexOf('    rc = cuda_malloc(cell, wh2)',$start)
if($start -lt 0 -or $end -lt 0){ throw 'head weight block' }
$t=$t.Substring(0,$start)+'    wh = cuda_h2d(dWh, hg + qb + 2 * hb + NL * lsz, wh2, 1)'+"`n"+$t.Substring($end)
# careful: dWh is allocated AFTER this point; move h2d after allocation
# undo by swapping: remove the inserted line and insert after dWh allocation
$bad='    wh = cuda_h2d(dWh, hg + qb + 2 * hb + NL * lsz, wh2, 1)'
$t=$t.Replace($bad,'')
$a='    dWh = read_i64(cell, 0)'
$t=Rep2 $t $a ($a+"`n    wh = cuda_h2d(dWh, hg + qb + 2 * hb + NL * lsz, wh2, 1)") 'head weight h2d'
$t=Rep2 $t '    wh = cuda_h2d(dWh, hWh, wh2, 1)' '' 'remove old head h2d'
# 7) per-layer LN pointers in forward/backward
$a='            s2d = read_i64(AS2, ll * 8)'
$k=$t.IndexOf($a)
if($k -lt 0){ throw 'LN ptrs fwd' }
$ins="`n            lg1: `"i64`" = read_i64(ALG1, ll * 8)`n            lb1: `"i64`" = read_i64(ALB1, ll * 8)`n            lg2: `"i64`" = read_i64(ALG2, ll * 8)`n            lb2: `"i64`" = read_i64(ALB2, ll * 8)"
$t=$t.Substring(0,$k+$a.Length)+$ins+$t.Substring($k+$a.Length)
Write-Host 'LN ptrs fwd hits=1'
$t=Rep2 $t 'rc = ln_fwd(lnf, xin, dgam, dbet, y1d, xh1d, s1d, T, H)' 'rc = ln_fwd(lnf, xin, lg1, lb1, y1d, xh1d, s1d, T, H)' 'ln1 fwd'
$t=Rep2 $t 'rc = ln_fwd(lnf, dX2, dgam, dbet, y2d, xh2d, s2d, T, H)' 'rc = ln_fwd(lnf, dX2, lg2, lb2, y2d, xh2d, s2d, T, H)' 'ln2 fwd'
$a2='            s2d = read_i64(AS2, ll * 8)'
$k1=$t.IndexOf($a2)
$k2=$t.IndexOf($a2,$k1+1)
if($k2 -lt 0){ throw 'bwd s2d' }
$ins="`n            lg1 = read_i64(ALG1, ll * 8)`n            lb1 = read_i64(ALB1, ll * 8)`n            lg2 = read_i64(ALG2, ll * 8)`n            lb2 = read_i64(ALB2, ll * 8)"
$t=$t.Substring(0,$k2+$a2.Length)+$ins+$t.Substring($k2+$a2.Length)
Write-Host 'bwd s2d hits=1'
$t=Rep2 $t 'rc = ln_bwd(lnb, xh2d, gY2, dgam, s2d, gDX2, T, H)' 'rc = ln_bwd(lnb, xh2d, gY2, lg2, s2d, gDX2, T, H)' 'ln2 bwd'
$t=Rep2 $t 'rc = ln_bwd(lnb, xh1d, gDX2, dgam, s1d, gY1, T, H)' 'rc = ln_bwd(lnb, xh1d, gDX2, lg1, s1d, gY1, T, H)' 'ln1 bwd'
# 8) final LN before head
$t=Rep2 $t '        rc = toh(th, dOUT, fpA, sQH)' @'
        rc = ln_fwd(lnf, dOUT, dFG, dFB, y1d, xh1d, s1d, T, H)
        rc = toh(th, y1d, fpA, sQH)
'@ 'final LN'
Set-Content 'ilgpt2_real.tkv' -Value $t -NoNewline -Encoding UTF8
& '..\..\dist\tkvc-tv.exe' 'ilgpt2_real.tkv' 'ilgpt2_real.exe' run --emit-il 2>&1 | Select-Object -First 8
