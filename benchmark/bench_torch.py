# BENCH PYTORCH — doi chieu tv/tests/simd_bench*.tkv (cung config, cung data)
#   A: linear 768->2304 fwd+bwd, 256 token        (GEMM lon — giong gate simd [9])
#   B: GPT-2 nho  2/4/64  vocab256 seq8 train step (giong gate simd [8])
#   C: GPT-2 124M (12/12/768) train step, cung 8 token fixture
#   D: GPT-2 124M o seq thuc (P4): python benchmark/bench_torch.py d [seq=1024]
# Chay (tu goc repo):  python benchmark/bench_torch.py [a|b|c|d|all] [seq]
# Hop dong so: fp32, dropout=0, AdamW eps=1e-8 betas=(0.9,0.999) wd=0 — cung
# tham so hop dong cua tkv (_mk_opt). Moi lan do = forward + backward (+ step).
import sys
import os
import time
import statistics

import torch
import torch.nn.functional as F

FIX = "tv/tests/fixtures/"


def load_ints(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.extend(int(x) for x in ln.split())
    return out


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


def report(tag, ts, extra=""):
    print("BENCH %s: median=%.1f ms  mean=%.1f  min=%.1f  max=%.1f  n=%d  %s"
          % (tag, statistics.median(ts), statistics.fmean(ts), min(ts), max(ts),
             len(ts), extra))
    sys.stdout.flush()


def tier_a():
    torch.manual_seed(7)
    x = torch.randn(256, 768, requires_grad=True)
    w = torch.randn(2304, 768, requires_grad=True)  # F.linear weight = [out, in]
    b = torch.randn(2304, requires_grad=True)
    gy = torch.randn(256, 2304)

    def step():
        y = F.linear(x, w, b)
        torch.autograd.grad(y, (x, w, b), grad_outputs=gy)

    report("A torch linear fwd+bwd 256x768->2304", timeit(step, 1, 10))


def _train_bench(cfg, ids, pos, vocab, warmup, iters, tag):
    torch.manual_seed(7)
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel(cfg)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.0025, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)

    def loss_fn():
        logits = model(ids, position_ids=pos).logits
        return F.cross_entropy(logits.view(-1, vocab), ids.view(-1))

    l0 = loss_fn().item()

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()

    ts = timeit(step, warmup, iters)
    report(tag, ts, "first_loss=%.4f params=%d" % (l0, npar))


def tier_b():
    from transformers import GPT2Config
    ids_l = load_ints(FIX + "g2_ids.txt")
    pos_l = load_ints(FIX + "g2_pos.txt")
    ids = torch.tensor([ids_l], dtype=torch.long)
    pos = torch.tensor([pos_l], dtype=torch.long)
    cfg = GPT2Config(vocab_size=256, n_positions=32, n_embd=64, n_head=4,
                     n_layer=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    _train_bench(cfg, ids, pos, 256, 2, 10,
                 "B torch GPT-2 small train step seq8 (2/4/64, vocab256)")


def tier_c():
    from transformers import GPT2Config
    ids_l = load_ints(FIX + "g2_ids.txt")
    pos_l = load_ints(FIX + "g2_pos.txt")
    ids = torch.tensor([ids_l], dtype=torch.long)
    pos = torch.tensor([pos_l], dtype=torch.long)
    cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_head=12,
                     n_layer=12, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    _train_bench(cfg, ids, pos, 50257, 1, 3,
                 "C torch GPT-2 124M train step seq8 (12/12/768)")


def tier_d(seq=1024):
    """124M o seq thuc (P4) — CUNG token voi tv/tests/simd_bench_124m_seq.tkv
    (ids[i] = (i*37+11) % 50257, pos[i] = i) de so sanh apple-to-apple.
    Do 2 phan: fwd+bwd (doi chieu tkv mode 0) va full step (mode 1).

    Thiet bi: `TKV_BENCH_DEVICE=cpu|cuda` (mac dinh cpu de giu so lieu cu).
    Tren may co GPU, chay CA HAI de tach ro "CPU apples-to-apples" va
    "PyTorch tren GPU" (2026-11-12: notebook Colab truoc day gan nhan sai la
    "PyTorch CUDA" nhung script khong he chuyen tensor/model sang CUDA)."""
    from transformers import GPT2Config
    from transformers import GPT2LMHeadModel
    dev = os.environ.get("TKV_BENCH_DEVICE", "cpu").strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        print("BENCH: TKV_BENCH_DEVICE=cuda nhung CUDA khong co -> dung cpu")
        dev = "cpu"
    ids_l = [(i * 37 + 11) % 50257 for i in range(seq)]
    pos_l = list(range(seq))
    ids = torch.tensor([ids_l], dtype=torch.long).to(dev)
    pos = torch.tensor([pos_l], dtype=torch.long).to(dev)
    cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_head=12,
                     n_layer=12, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(cfg).to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.0025, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)

    def loss_fn():
        logits = model(ids, position_ids=pos).logits
        return F.cross_entropy(logits.view(-1, 50257), ids.view(-1))

    l0 = loss_fn().item()

    def fb():
        if dev == "cuda":
            torch.cuda.synchronize()
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        if dev == "cuda":
            torch.cuda.synchronize()

    def step():
        if dev == "cuda":
            torch.cuda.synchronize()
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()

    report("D torch GPT-2 124M fwd+bwd seq%d dev=%s" % (seq, dev), timeit(fb, 1, 2),
           "loss=%.4f params=%d" % (l0, npar))
    report("D torch GPT-2 124M full step seq%d dev=%s" % (seq, dev), timeit(step, 1, 2),
           "loss=%.4f params=%d" % (l0, npar))


def tier_e(seq=8):
    """PHASE A — doi chuan 2 phia (docs/archive/PLAN_TO_PARITY.md): tach tung phase
    (zero / fwd / bwd / opt) va chay ca 1T + 14T de so bucket voi CJ8 tkv.
    Goi: python benchmark/bench_torch.py e <seq>

    `TKV_BENCH_THREADS=2` de chi chay 1 cau hinh thread (vd tren Colab 2 vCPU,
    chay 1T + 14T qua lau; 2 thread = dung so thread cua .tkv GEMM pool de so
    tung pha apples-to-apples)."""
    from transformers import GPT2Config
    from transformers import GPT2LMHeadModel
    threads = [int(x) for x in
               os.environ.get("TKV_BENCH_THREADS", "1,14").split(",") if x.strip()]
    ids_l = [(i * 37 + 11) % 50257 for i in range(seq)]
    pos_l = list(range(seq))
    ids = torch.tensor([ids_l], dtype=torch.long)
    pos = torch.tensor([pos_l], dtype=torch.long)
    cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_head=12,
                     n_layer=12, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    for nt in threads:
        torch.set_num_threads(nt)
        torch.manual_seed(7)
        model = GPT2LMHeadModel(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0025, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.0)

        def loss_fn():
            logits = model(ids, position_ids=pos).logits
            return F.cross_entropy(logits.view(-1, 50257), ids.view(-1))

        # warmup 2 buoc (thu cho nong + init opt state)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()

        zs, fs, bs, os_ = [], [], [], []
        for _ in range(3):
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            t1 = time.perf_counter()
            loss = loss_fn()
            t2 = time.perf_counter()
            loss.backward()
            t3 = time.perf_counter()
            opt.step()
            t4 = time.perf_counter()
            zs.append((t1 - t0) * 1000.0)
            fs.append((t2 - t1) * 1000.0)
            bs.append((t3 - t2) * 1000.0)
            os_.append((t4 - t3) * 1000.0)
        report("E torch zero_grad seq%d T=%d" % (seq, nt), zs)
        report("E torch fwd(+CE) seq%d T=%d" % (seq, nt), fs)
        report("E torch bwd seq%d T=%d" % (seq, nt), bs)
        report("E torch opt.step seq%d T=%d" % (seq, nt), os_)
        tot = [a + b + c + d for a, b, c, d in zip(zs, fs, bs, os_)]
        report("E torch TOTAL seq%d T=%d" % (seq, nt), tot)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    # chap nhan "all" hoac chuoi chu giup hop ("ab", "a", "bc", ...)
    tiers = {"a", "b", "c", "d", "e"} if which == "all" else set(which)
    unknown = tiers - {"a", "b", "c", "d", "e"}
    if unknown:
        print("BENCH unknown tier(s): %s — dung a|b|c|d|all" % sorted(unknown))
        sys.exit(2)
    print("BENCH env: torch %s threads=%d"
          % (torch.__version__, torch.get_num_threads()))
    sys.stdout.flush()
    if "a" in tiers:
        tier_a()
    if "b" in tiers:
        tier_b()
    if "c" in tiers:
        tier_c()
    if "d" in tiers:
        tier_d(seq)
    if "e" in tiers:
        tier_e(seq)
    print("BENCH done")


if __name__ == "__main__":
    main()
