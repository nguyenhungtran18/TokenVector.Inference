# BENCH PYTORCH — doi chieu tv/tests/simd_bench*.tkv (cung config, cung data)
#   A: linear 768->2304 fwd+bwd, 256 token        (GEMM lon — giong gate simd [9])
#   B: GPT-2 nho  2/4/64  vocab256 seq8 train step (giong gate simd [8])
#   C: GPT-2 124M (12/12/768) train step, cung 8 token fixture
# Chay (tu goc repo):  python tools/bench_torch.py [a|b|c|all]
# Hop dong so: fp32, dropout=0, AdamW eps=1e-8 betas=(0.9,0.999) wd=0 — cung
# tham so hop dong cua tkv (_mk_opt). Moi lan do = forward + backward (+ step).
import sys
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


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    # chap nhan "all" hoac chuoi chu giup hop ("ab", "a", "bc", ...)
    tiers = {"a", "b", "c"} if which == "all" else set(which)
    unknown = tiers - {"a", "b", "c"}
    if unknown:
        print("BENCH unknown tier(s): %s — dung a|b|c|all" % sorted(unknown))
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
    print("BENCH done")


if __name__ == "__main__":
    main()
