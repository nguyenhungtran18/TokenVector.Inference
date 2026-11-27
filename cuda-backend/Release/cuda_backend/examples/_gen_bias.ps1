$ErrorActionPreference='Stop'
cd 'D:\TokenVector.Inference\cuda-backend\Release\cuda_backend\examples'
$f=Get-Content '_bias.ptx' -Raw
$f=$f.Replace("`r`n","`n").Replace('\','\\').Replace("`n",'\n').Replace('"','\"')
$tpl=@'
def bias_init() -> "i64":
    PTX = b"__PTX__"
    F1 = b"biasadd_kernel\x00"
    F2 = b"biasgrad_kernel\x00"
    pb = alloc_bytes(len(PTX) + 1)
    kk: "i32" = 0
    while kk < len(PTX):
        wh: "i64" = write_u8(pb, kk, PTX[kk])
        kk = kk + 1
    wh = write_u8(pb, kk, 0)
    mc = alloc_bytes(8)
    rc: "i32" = cu_module_load(mc, pb)
    if rc != 0:
        print("BIAS MOD FAIL", rc)
        return 0
    mod: "i64" = read_i64(mc, 0)
    h = alloc_bytes(16)
    kk = 0
    while kk < len(F1):
        wh = write_u8(pb, kk, F1[kk])
        kk = kk + 1
    wh = write_u8(pb, kk, 0)
    fc = alloc_bytes(8)
    rc = cu_module_fn(fc, mod, pb)
    if rc != 0:
        print("BIAS FN1 FAIL", rc)
        return 0
    wh = write_i64(h, 0, read_i64(fc, 0))
    kk = 0
    while kk < len(F2):
        wh = write_u8(pb, kk, F2[kk])
        kk = kk + 1
    wh = write_u8(pb, kk, 0)
    rc = cu_module_fn(fc, mod, pb)
    if rc != 0:
        print("BIAS FN2 FAIL", rc)
        return 0
    wh = write_i64(h, 8, read_i64(fc, 0))
    return h

def bias_add(fnh: "i64", c: "i64", b: "i64", t: "i32", n: "i32") -> "i32":
    a0 = alloc_bytes(8)
    wh: "i64" = write_i64(a0, 0, c)
    a1 = alloc_bytes(8)
    wh = write_i64(a1, 0, b)
    a2 = alloc_bytes(4)
    wh = write_i32(a2, 0, t)
    a3 = alloc_bytes(4)
    wh = write_i32(a3, 0, n)
    pl = alloc_bytes(32)
    wh = write_i64(pl, 0, a0)
    wh = write_i64(pl, 8, a1)
    wh = write_i64(pl, 16, a2)
    wh = write_i64(pl, 24, a3)
    g: "i32" = (n + 255) >> 8
    rc: "i32" = cu_launch(fnh, g, 1, 1, 256, 1, 1, 0, 0, pl, 0)
    if rc != 0:
        print("BIAS ADD FAIL", rc)
    return rc

def bias_grad_k(fnh: "i64", gmat: "i64", b: "i64", t: "i32", rows: "i32") -> "i32":
    a0 = alloc_bytes(8)
    wh: "i64" = write_i64(a0, 0, gmat)
    a1 = alloc_bytes(8)
    wh = write_i64(a1, 0, b)
    a2 = alloc_bytes(4)
    wh = write_i32(a2, 0, t)
    a3 = alloc_bytes(4)
    wh = write_i32(a3, 0, rows)
    pl = alloc_bytes(32)
    wh = write_i64(pl, 0, a0)
    wh = write_i64(pl, 8, a1)
    wh = write_i64(pl, 16, a2)
    wh = write_i64(pl, 24, a3)
    g: "i32" = (rows + 255) >> 8
    rc: "i32" = cu_launch(fnh, g, 1, 1, 256, 1, 1, 0, 0, pl, 0)
    if rc != 0:
        print("BIAS GRAD FAIL", rc)
    return rc

'@
$out=$tpl.Replace('__PTX__',$f)
Set-Content '_bias_snippet.tkv' -Value $out -NoNewline -Encoding UTF8
Write-Output ('snippet chars='+$out.Length)
