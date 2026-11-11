# P7b — DO torch.matmul GEMM vs tkv simd_gemm (cung 3 shape, cung session)
#
# Boi canh: 3 bucket GEMM (ATTN+MLP+LHEAD) = 24.1s = 66% buoc @seq1024. tkv
# da lam tot phan wrapper (probe BH: builder 1.4-3.2x, wave-12 3.2x). Con lai:
# KHOANG CACH GIUA simd_gemm NATIVE VA torch.matmul CUNG SHAPE la bao nhieu?
# Day la 3 shape that cua buoc 124M (giong probe BH S3/S4/S5/S1).
#
# S1 dx1      [1024, 3072] = gy[1024,3072] @ w1[3072,768]^T   k=3072  m*n*k
# S3 dxf      [1024, 768]  = gg[1024,50257] @ wte[50257,768]   k=50257
# S4 dwte     [50257, 768] = ggT[50257,1024] @ xf[1024,768]   k=1024
# S5 logits   [1024, 50257]= xf[1024,768] @ wteT[768,50257]   k=768
# (S5 la transpose-nhan: torch dung @, tkv dung smm_dot — cung flop huu ich.)
#
# Chay (tu goc repo):  python benchmark/p7b_gemm_torch.py
import statistics
import sys
import time

import torch

WARMUP = 1
ITERS = 3


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def gflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e6)


def bench(tag, m, n, k, mk_a, mk_b, trans_b=False):
    torch.manual_seed(7)
    a = mk_a()
    b = mk_b()
    if trans_b:
        ms = timeit(lambda: a @ b.T)  # torch @ khong auto-transpose (khac numpy)
    else:
        ms = timeit(lambda: a @ b)
    print("P7b torch %s [%dx%dx%d]: %.1f ms  %.1f GFLOPS"
          % (tag, m, n, k, ms, gflops(m, n, k, ms)))
    sys.stdout.flush()


def main():
    print("P7b env: torch %s threads=%d (torch.matmul f32)"
          % (torch.__version__, torch.get_num_threads()))
    sys.stdout.flush()

    # S1 dx1: [1024,3072] @ [3072,768]
    bench("S1 dx1", 1024, 768, 3072,
          lambda: torch.randn(1024, 3072), lambda: torch.randn(3072, 768))

    # S3 dxf: [1024,50257] @ [50257,768]
    bench("S3 dxf", 1024, 768, 50257,
          lambda: torch.randn(1024, 50257), lambda: torch.randn(50257, 768))

    # S4 dwte: [50257,1024] @ [1024,768]
    bench("S4 dwte", 50257, 768, 1024,
          lambda: torch.randn(50257, 1024), lambda: torch.randn(1024, 768))

    # S5 logits (A @ B^T): [1024,768] @ [768,50257]
    bench("S5 logits", 1024, 50257, 768,
          lambda: torch.randn(1024, 768), lambda: torch.randn(50257, 768),
          trans_b=True)

    print("P7b torch GEMM done")


if __name__ == "__main__":
    main()
