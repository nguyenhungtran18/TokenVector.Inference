$ErrorActionPreference='Stop'
cd 'D:\TokenVector.Inference\cuda-backend\Release\cuda_backend\examples'
$f=Get-Content '_attn16.ptx' -Raw
$f=$f.Replace("`r`n","`n").Replace('\','\\').Replace("`n",'\n').Replace('"','\"')
$tpl=@'
def attn16_init() -> "i64":
    PTX = b"__PTX__"
    F1 = b"sf16_kernel\x00"
    F2 = b"sbk16_kernel\x00"
    pb = alloc_bytes(len(PTX) + 1)
    kk: "i32" = 0
    while kk < len(PTX):
        wh: "i64" = write_u8(pb, kk, PTX[kk])
        kk = kk + 1
    wh = write_u8(pb, kk, 0)
    mc = alloc_bytes(8)
    rc: "i32" = cu_module_load(mc, pb)
    if rc != 0:
        print("ATTN16 MOD FAIL", rc)
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
        print("ATTN16 FN1 FAIL", rc)
        return 0
    wh = write_i64(h, 0, read_i64(fc, 0))
    kk = 0
    while kk < len(F2):
        wh = write_u8(pb, kk, F2[kk])
        kk = kk + 1
    wh = write_u8(pb, kk, 0)
    rc = cu_module_fn(fc, mod, pb)
    if rc != 0:
        print("ATTN16 FN2 FAIL", rc)
        return 0
    wh = write_i64(h, 8, read_i64(fc, 0))
    return h

def sf16_fwd(fnh: "i64", s: "i64", p: "i64", tn: "i32", nh: "i32") -> "i32":
    a0 = alloc_bytes(8)
    wh: "i64" = write_i64(a0, 0, s)
    a1 = alloc_bytes(8)
    wh = write_i64(a1, 0, p)
    a2 = alloc_bytes(4)
    wh = write_i32(a2, 0, tn)
    pl = alloc_bytes(24)
    wh = write_i64(pl, 0, a0)
    wh = write_i64(pl, 8, a1)
    wh = write_i64(pl, 16, a2)
    rc: "i32" = cu_launch(fnh, 1, nh, 1, tn, 1, 1, 0, 0, pl, 0)
    if rc != 0:
        print("SF16 FWD FAIL", rc)
    return rc

def sf16_bwd(fnh: "i64", dp: "i64", p: "i64", ds: "i64", tn: "i32", nh: "i32") -> "i32":
    a0 = alloc_bytes(8)
    wh: "i64" = write_i64(a0, 0, dp)
    a1 = alloc_bytes(8)
    wh = write_i64(a1, 0, p)
    a2 = alloc_bytes(8)
    wh = write_i64(a2, 0, ds)
    a3 = alloc_bytes(4)
    wh = write_i32(a3, 0, tn)
    pl = alloc_bytes(32)
    wh = write_i64(pl, 0, a0)
    wh = write_i64(pl, 8, a1)
    wh = write_i64(pl, 16, a2)
    wh = write_i64(pl, 24, a3)
    rc: "i32" = cu_launch(fnh, 1, nh, 1, tn, 1, 1, 0, 0, pl, 0)
    if rc != 0:
        print("SF16 BWD FAIL", rc)
    return rc

def gemm16b(handle: "i64", ta: "i32", tb: "i32", m: "i32", n: "i32", k: "i32", alpha: "i64", a: "i64", lda: "i32", sa: "i64", b: "i64", ldb: "i32", sb: "i64", beta: "i64", c: "i64", ldc: "i32", sc: "i64", batch: "i32") -> "i32":
    rc: "i32" = cublas_gemm_strided_batched_ex(handle, ta, tb, m, n, k, alpha, a, 2, lda, sa, b, 2, ldb, sb, beta, c, 0, ldc, sc, batch, 74, -1)
    if rc != 0:
        print("GEMM16B FAIL", rc)
    return rc

'@
$out=$tpl.Replace('__PTX__',$f)
Set-Content '_attn16_snippet.tkv' -Value $out -NoNewline -Encoding UTF8
Write-Output ('snippet chars='+$out.Length)
