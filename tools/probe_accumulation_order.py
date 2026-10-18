# -*- coding: utf-8 -*-
# Xac dinh thu tu tich luy THUC TE cua torch CPU cho tung loai phep toan,
# so voi tich luy tuan tu k=0..K-1 bang float32 (giong vong lap .tkv).
import numpy as np
import torch

print(torch.__config__.show()[:600])
torch.manual_seed(0)
rng = np.random.RandomState(0)

def seq_dot_rows(X, W):
    # y[i,o] = sum_k x[i,k]*w[k,o]  tich luy tuan tu, f32 moi buoc
    n, k = X.shape
    k2, o = W.shape
    assert k == k2
    Y = np.zeros((n, o), dtype=np.float32)
    for i in range(n):
        for oo in range(o):
            acc = np.float32(0)
            for kk in range(k):
                acc = np.float32(acc + np.float32(X[i, kk]) * np.float32(W[kk, oo]))
            Y[i, oo] = acc
    return Y

def seq_mean_rows(X):
    n, h = X.shape
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = np.float32(0)
        for j in range(h):
            acc = np.float32(acc + np.float32(X[i, j]))
        out[i] = np.float32(acc / np.float32(h))
    return out

# ---- 1. matmul [8,64]@[64,256] (giong LM head/linear GPT-2 nho) ----
X = rng.randn(8, 64).astype(np.float32)
W = rng.randn(64, 256).astype(np.float32)
tX, tW = torch.tensor(X), torch.tensor(W)
tY = tX @ tW
nY = seq_dot_rows(X, W)
bitdiff = (tY.numpy().view(np.int32) != nY.view(np.int32)).sum()
print("matmul [8,64]@[64,256]: bit-diff elements = %d / %d  max_abs=%.3e"
      % (bitdiff, nY.size, np.abs(tY.numpy() - nY).max()))
if bitdiff:
    idx = np.argwhere(tY.numpy().view(np.int32) != nY.view(np.int32))[0]
    i, o = int(idx[0]), int(idx[1])
    print("  vi tri dau tien [%d,%d]: torch=%.9g seq=%.9g" % (i, o, tY[i, o], nY[i, o]))

# ---- 2. matmul nho [4,4]@[4,4] ----
A = rng.randn(4, 4).astype(np.float32)
B = rng.randn(4, 4).astype(np.float32)
tb = torch.tensor(A) @ torch.tensor(B)
nb = seq_dot_rows(A, B)
print("matmul [4,4]: bit-diff = %d" % (tb.numpy().view(np.int32) != nb.view(np.int32)).sum())

# ---- 3. mean/sum 64 phan tu ----
V = rng.randn(64).astype(np.float32)
tm = torch.tensor(V).mean()
sm = np.float32(0)
for j in range(64):
    sm = np.float32(sm + np.float32(V[j]))
sm = np.float32(sm / np.float32(64))
print("mean(64): torch=%.9g seq=%.9g bitdiff=%s"
      % (tm.item(), sm, tm.numpy().view(np.int32) != np.float32(sm).view(np.int32)))

# ---- 4. exp f32 ----
vals = np.array([-3.3, -1.0, 0.0, 0.5, 1.7, 3.3], dtype=np.float32)
te = torch.exp(torch.tensor(vals))
ne = np.exp(vals.astype(np.float32))          # np exp f32
de = np.exp(vals.astype(np.float64)).astype(np.float32)  # double roi tron ve f32 (.NET Math.Exp)
print("exp f32: torch vs np_f32 bitdiff=%d ; torch vs double-round bitdiff=%d"
      % ((te.numpy().view(np.int32) != ne.view(np.int32)).sum(),
         (te.numpy().view(np.int32) != de.view(np.int32)).sum()))

# ---- 5. ung vien thu tu tich luy cho MKL sgemm ----
def cand_fma_seq(X, W):
    n, k = X.shape
    k2, o = W.shape
    Y = np.zeros((n, o), dtype=np.float32)
    for i in range(n):
        for oo in range(o):
            acc = 0.0  # f64
            for kk in range(k):
                p = float(X[i, kk]) * float(W[kk, oo])   # exact trong f64
                acc = acc + p                            # ≈ fma (thuong exact)
            Y[i, oo] = np.float32(acc)
    return Y

def cand_fma_4acc(X, W):
    # 4 accumulators lien thap (k%4), tong cuoi ((g0+g1)+(g2+g3))
    n, k = X.shape
    k2, o = W.shape
    Y = np.zeros((n, o), dtype=np.float32)
    for i in range(n):
        for oo in range(o):
            g = [0.0, 0.0, 0.0, 0.0]
            for kk in range(k):
                p = float(X[i, kk]) * float(W[kk, oo])
                g[kk % 4] += p
            Y[i, oo] = np.float32((g[0] + g[1]) + (g[2] + g[3]))
    return Y

def cand_f64sum(X, W):
    Y64 = X.astype(np.float64) @ W.astype(np.float64)
    return Y64.astype(np.float32)

tY_np = tY.numpy()
for name, fn in [("fma_seq", cand_fma_seq), ("fma_4acc", cand_fma_4acc), ("f64sum", cand_f64sum)]:
    Y = fn(X, W)
    bd = (tY_np.view(np.int32) != Y.view(np.int32)).sum()
    print("%-10s: bit-diff=%d/%d max_abs=%.3e" % (name, bd, Y.size, np.abs(tY_np - Y).max()))

npY = X @ W
bd = (tY_np.view(np.int32) != npY.view(np.int32)).sum()
print("np-BLAS   : bit-diff=%d/%d" % (bd, npY.size))

# ---- 6. ung vien cho mean(64) ----
def m_f64(v):
    return np.float32(float(v.astype(np.float64).sum()) / 64.0)
def m_tree8(v):
    # 8 lane AVX2-like: sum moi 8 phan tu roi cong 8 ket qua
    acc8 = np.zeros(8, dtype=np.float32)
    for j in range(64):
        acc8[j % 8] = np.float32(acc8[j % 8] + np.float32(v[j]))
    s = np.float32(np.float32(np.float32(acc8[0]+acc8[1]) + np.float32(acc8[2]+acc8[3]))
                   + np.float32(np.float32(acc8[4]+acc8[5]) + np.float32(acc8[6]+acc8[7])))
    return np.float32(s / np.float32(64))
for name, fn in [("mean_f64", m_f64), ("mean_tree8", m_tree8)]:
    r = fn(V)
    print("%-11s: bitdiff=%s val=%.9g (torch=%.9g)" %
          (name, tm.numpy().view(np.int32) != np.float32(r).view(np.int32), r, tm.item()))

# ---- 7. sqrt / erf (elementwise) ----
vs = np.abs(rng.randn(64)).astype(np.float32)
ts = torch.sqrt(torch.tensor(vs))
ns = np.sqrt(vs)
print("sqrt f32  : bitdiff=%d/64" % (ts.numpy().view(np.int32) != ns.view(np.int32)).sum())
import math
ve = (rng.randn(64) * 1.5).astype(np.float32)
te2 = torch.erf(torch.tensor(ve))
ne2 = np.array([np.float32(math.erf(float(v))) for v in ve])  # libm double roi tron
print("erf       : bitdiff=%d/64 (torch vs libm-double-round)" %
      (te2.numpy().view(np.int32) != ne2.view(np.int32)).sum())
