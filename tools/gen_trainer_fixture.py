#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture G4 TRAINER: AdamW + warmup/cosine + clip norm + grad accumulation.

Implementation THAM CHIEU (PyTorch) — phai doi xung CHINH XAC tung buoc voi
`tv/training/trainer.tkv`:

  Model:  MLP 4 -> 8 (ReLU) -> 3, cross-entropy mean.
          W1 [8,4] row-major, b1 [8], W2 [3,8] row-major, b2 [3]  (67 tham so)
  Data:   20 mau (x 4 chieu, nhan 0..2), co dinh, sinh bang seed.
  Order:  moi epoch xao tron 20 mau bang LCG 31-bit (Fisher-Yates):
            state' = (state * 1103515245 + 12345) % 2^31
          draw thu d lay perm[d % 20]; het 20 -> xao lai tu state hien tai.
  Step:   loss = mean(loss_sample) tren `accum` mau; backward; clip;
          AdamW (decoupled weight decay) voi lr cua buoc do.
  LR:     s = 0..total-1 (0-based)
            s <  warmup : lr = lr_max * (s + 1) / warmup
            s >= warmup : prog = (s - warmup) / (total - warmup)
                          lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(pi*prog))
  Clip:   PyTorch clip_grad_norm_: norm = L2(L2 tung tham so, ...);
          coef = clip / (norm + 1e-6); chi nhan khi coef < 1.

Chay:  python tools\\gen_trainer_fixture.py
"""

import os
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("CANH BAO: khong tim thay PyTorch.")
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tv", "tests", "fixtures", "train_cases.txt")

IN_D = 4
HID = 8
OUT_D = 3
N_PARAM = HID * IN_D + HID + OUT_D * HID + OUT_D   # 67

WARMUP = 5
TOTAL = 400
LR_MAX = 0.1
LR_MIN = 0.005
WD = 0.01
B1 = 0.9
B2 = 0.999
EPS = 1e-8
CLIP = 1.0
ACCUM = 2
N_SAMPLES = 20
SNAP_STEPS = [1, 5, 20, 40, 100, 200, 400]

LCG_A = 1103515245
LCG_C = 12345
LCG_M = 1 << 31


def rng_step(state):
    return (state * LCG_A + LCG_C) % LCG_M


def lr_at(s):
    if s < WARMUP:
        return LR_MAX * float(s + 1) / float(WARMUP)
    prog = float(s - WARMUP) / float(TOTAL - WARMUP)
    import math
    return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * prog))


def write_ints(f, vals):
    f.write(" ".join(str(int(v)) for v in vals) + "\n")


def main():
    torch.manual_seed(2026)
    torch.set_grad_enabled(True)

    def mk(a, b, scale):
        return (torch.randn(a, b, dtype=torch.float32) * scale).reshape(-1)

    w1 = mk(HID, IN_D, 0.5)
    b1 = torch.zeros(HID, dtype=torch.float32)
    w2 = mk(OUT_D, HID, 0.5)
    b2 = torch.zeros(OUT_D, dtype=torch.float32)

    torch.manual_seed(7)
    xs = torch.randn(N_SAMPLES, IN_D, dtype=torch.float32)
    ys = torch.randint(0, OUT_D, (N_SAMPLES,), dtype=torch.int64)
    # dam bao ca 3 lop xuat hien
    ys[0] = 0
    ys[1] = 1
    ys[2] = 2

    params = [w1, b1, w2, b2]
    # Snapshot TRUOC khi train — optimizer sua tai cho (in-place) nen neu doc
    # lai w1/b1/w2/b2 sau vong lap se ra trong so CUOI, khong phai init.
    init_vals = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)

    opt = torch.optim.AdamW(params, lr=LR_MAX, betas=(B1, B2),
                            eps=EPS, weight_decay=WD)

    # Thu tu mau: LCG 31-bit Fisher-Yates moi epoch.
    state = 42
    perm = list(range(N_SAMPLES))
    i = N_SAMPLES - 1
    while i > 0:
        state = rng_step(state)
        j = state % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
        i -= 1
    draws = 0

    def next_index():
        nonlocal state, perm, draws
        if draws > 0 and draws % N_SAMPLES == 0:
            perm = list(range(N_SAMPLES))
            i2 = N_SAMPLES - 1
            while i2 > 0:
                state = rng_step(state)
                j2 = state % (i2 + 1)
                perm[i2], perm[j2] = perm[j2], perm[i2]
                i2 -= 1
        idx = perm[draws % N_SAMPLES]
        draws += 1
        return idx

    perm_init = list(perm)
    p_init = torch.cat([p.detach().reshape(-1) for p in params]).clone()

    steps = []
    snaps = {}
    for s in range(TOTAL):
        lr = lr_at(s)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        losses = []
        for _ in range(ACCUM):
            idx = next_index()
            x = xs[idx:idx + 1]
            y = ys[idx:idx + 1]
            h = torch.relu(x @ w1.reshape(HID, IN_D).T + b1)
            z = h @ w2.reshape(OUT_D, HID).T + b2
            l = F.cross_entropy(z, y)
            (l / ACCUM).backward()
            losses.append(float(l.item()))
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        step_loss = sum(losses) / float(ACCUM)
        steps.append((lr, step_loss, float(gn)))
        if (s + 1) in SNAP_STEPS:
            snaps[s + 1] = torch.cat([p.detach().reshape(-1) for p in params]).clone()

    # eval loss tren toan bo 20 mau
    with torch.no_grad():
        h = torch.relu(xs @ w1.reshape(HID, IN_D).T + b1)
        z = h @ w2.reshape(OUT_D, HID).T + b2
        eval_loss = float(F.cross_entropy(z, ys).item())
        pred = z.argmax(dim=1)
        acc = float((pred == ys).float().mean().item())
        logits = z.clone()

    flat_final = torch.cat([p.detach().reshape(-1) for p in params]).clone()

    # resume: 3 buoc -> checkpoint -> 3 buoc, phai bang chay lien 6 buoc
    def train_n(p_start, n, st0, perm0, draw0):
        pv = p_start.clone()
        w1r = pv[0:HID * IN_D].clone().requires_grad_(True)
        b1r = pv[HID * IN_D:HID * IN_D + HID].clone().requires_grad_(True)
        o1 = HID * IN_D + HID
        w2r = pv[o1:o1 + OUT_D * HID].clone().requires_grad_(True)
        b2r = pv[o1 + OUT_D * HID:].clone().requires_grad_(True)
        ps = [w1r, b1r, w2r, b2r]
        op = torch.optim.AdamW(ps, lr=LR_MAX, betas=(B1, B2), eps=EPS, weight_decay=WD)
        st = st0
        pm = list(perm0)
        dr = draw0
        for s in range(n):
            for g in op.param_groups:
                g["lr"] = lr_at(s)
            op.zero_grad(set_to_none=True)
            for _ in range(ACCUM):
                if dr > 0 and dr % N_SAMPLES == 0:
                    pm = list(range(N_SAMPLES))
                    i3 = N_SAMPLES - 1
                    while i3 > 0:
                        st = rng_step(st)
                        j3 = st % (i3 + 1)
                        pm[i3], pm[j3] = pm[j3], pm[i3]
                        i3 -= 1
                idx = pm[dr % N_SAMPLES]
                dr += 1
                hh = torch.relu(xs[idx:idx + 1] @ w1r.reshape(HID, IN_D).T + b1r)
                zz = hh @ w2r.reshape(OUT_D, HID).T + b2r
                (F.cross_entropy(zz, ys[idx:idx + 1]) / ACCUM).backward()
            torch.nn.utils.clip_grad_norm_(ps, CLIP)
            op.step()
        return torch.cat([p.detach().reshape(-1) for p in ps]).clone()

    # chay LIEN 6 buoc tu dau (state/perm/draws ban dau) -> moc resume
    resume_ref = train_n(p_init, 6, 42, perm_init, 0)

    lines = []
    lines.append("# G4 trainer fixture - sinh boi tools\\gen_trainer_fixture.py")
    lines.append("# (PyTorch reference, doi xung voi tv/training/trainer.tkv)")
    lines.append("#sec params")
    lines.append("in=%d hid=%d out=%d n_param=%d warmup=%d total=%d lr_max=%.9g "
                 "lr_min=%.9g wd=%.9g b1=%.9g b2=%.9g eps=%.9g clip=%.9g "
                 "accum=%d n_samples=%d"
                 % (IN_D, HID, OUT_D, N_PARAM, WARMUP, TOTAL, LR_MAX, LR_MIN, WD,
                    B1, B2, EPS, CLIP, ACCUM, N_SAMPLES))
    lines.append("#sec init")
    for p in init_vals:
        lines.append(" ".join("%.9g" % float(v) for v in p.reshape(-1).tolist()))
    lines.append("#sec x")
    for r in xs.tolist():
        lines.append(" ".join("%.9g" % float(v) for v in r))
    lines.append("#sec y")
    write_ints_to = " ".join(str(int(v)) for v in ys.tolist())
    lines.append(write_ints_to)
    lines.append("#sec step_lr")
    lines.append(" ".join("%.9g" % v for v, _, _ in steps))
    lines.append("#sec step_loss")
    lines.append(" ".join("%.9g" % v for _, v, _ in steps))
    lines.append("#sec step_gnorm")
    lines.append(" ".join("%.9g" % v for _, _, v in steps))
    for s in SNAP_STEPS:
        lines.append("#sec snap%d" % s)
        lines.append(" ".join("%.9g" % float(v) for v in snaps[s].tolist()))
    lines.append("#sec final")
    lines.append(" ".join("%.9g" % float(v) for v in flat_final.tolist()))
    lines.append("#sec resume_ref6")
    lines.append(" ".join("%.9g" % float(v) for v in resume_ref.tolist()))
    lines.append("#sec eval_loss")
    lines.append("%.9g" % eval_loss)
    lines.append("#sec eval_acc")
    lines.append("%.9g" % acc)
    lines.append("#sec logits")
    lines.append(" ".join("%.9g" % float(v) for v in logits.reshape(-1).tolist()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("-> %s (steps=%d params=%d eval_loss=%.6g acc=%.3f)"
          % (os.path.relpath(OUT, ROOT).replace("\\", "/"), TOTAL, N_PARAM,
             eval_loss, acc))
    print("   step1 loss=%.6g lr=%.6g gnorm=%.6g"
          % (steps[0][1], steps[0][0], steps[0][2]))


if __name__ == "__main__":
    main()
