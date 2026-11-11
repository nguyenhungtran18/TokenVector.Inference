# Reference PyTorch cho tv/tests/_gm.tkv — cung 3 shape, T=1 va T=14.
# Chay: python benchmark/probe_gemm_shape.py [seq]
import sys
import time

import torch
import torch.nn.functional as F

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
h, ih = 768, 3072


def best(fn, runs=3):
    fn()
    b = 1e18
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        b = min(b, (time.perf_counter() - t0) * 1000.0)
    return b


def gfl(ms, m, n, k):
    return (2.0 * m * n * k) / (ms / 1000.0) / 1e9


for nt in (1, 14):
    torch.set_num_threads(nt)
    x = torch.randn(seq, h)
    w1 = torch.randn(ih, h)
    go = torch.randn(seq, ih)
    w2 = torch.randn(h, ih)
    a2 = torch.randn(seq, ih)

    A = best(lambda: F.linear(x, w1))
    B = best(lambda: F.linear(go, w2))
    C = best(lambda: a2.t() @ go)

    print("PT T=%d A m=%d n=%d k=%d = %.1f ms  %.1f GFLOP/s"
          % (nt, seq, ih, h, A, gfl(A, seq, ih, h)))
    print("PT T=%d B m=%d n=%d k=%d = %.1f ms  %.1f GFLOP/s"
          % (nt, seq, h, ih, B, gfl(B, seq, h, ih)))
    print("PT T=%d C m=%d n=%d k=%d = %.1f ms  %.1f GFLOP/s"
          % (nt, ih, h, seq, C, gfl(C, ih, h, seq)))
    print("PT T=%d TOTAL3 = %.1f ms" % (nt, A + B + C))
