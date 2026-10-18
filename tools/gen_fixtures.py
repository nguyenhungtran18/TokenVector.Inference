#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture so chuan cho TokenVector.Inference tu PyTorch.

Day la CONG CU PHAT TRIEN (khong phai runtime) - chay tren may co torch:
    python tools\gen_fixtures.py

Ket qua (file van ban trong tv\tests\fixtures\) duoc check-in vao repo de
khong phai phu thuoc torch khi chay test.

Format fixture (text, de doc duoc tu .tkv bang read_file + split):
    dong 1:  "# n=<so phan tu>  <ten case>"
    dong 2:  cac gia tri cach nhau bang 1 dau cach, 9 chu so nghia trong f32
"""

import os
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("CANH BAO: khong tim thay PyTorch. Fixture cu khong duoc tao moi.")
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "tv", "tests", "fixtures")


def fmt(vals):
    return " ".join("%.9g" % float(v) for v in vals)


def write_case(name, header, values):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# n=%d  %s\n" % (len(values), header))
        f.write(fmt(values) + "\n")
    print("  -> %s  (n=%d)" % (os.path.relpath(path, ROOT), len(values)))
    return path


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    print("torch", torch.__version__)
    meta = {"torch": torch.__version__, "device": "cpu", "dtype": "float32"}

    # --- 1. GELU: ca 2 bien theo dung nhu PyTorch ---
    # PyTorch mac dinh la 'none' = erf chinh xac. 'tanh' la bien the xap xi.
    # Repo TokenVector hien chi co bien the 'tanh' -> can ca 2 de bat sai so.
    xs = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.1, 0.0,
          0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    x = torch.tensor(xs, dtype=torch.float32)
    write_case("gelu_x", "input", x.tolist())
    write_case("gelu_y_erf", "torch.nn.functional.gelu(x, approximate='none')",
               F.gelu(x, approximate="none").tolist())
    write_case("gelu_y_tanh", "torch.nn.functional.gelu(x, approximate='tanh')",
               F.gelu(x, approximate="tanh").tolist())

    # --- 2. Softmax hang ---
    row = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.5, 0.0]
    r = torch.tensor([row], dtype=torch.float32)
    sm = F.softmax(r, dim=-1).reshape(-1)
    write_case("softmax_row_x", "input", r.reshape(-1).tolist())
    write_case("softmax_row_y", "torch.softmax(dim=-1)", sm.tolist())

    # --- 3. RMSNorm (khop 3.3 cu ROADMAP_TRAIN) ---
    xs2 = [0.5, -1.25, 2.0, 0.0, 3.5, -0.75, 1.0, -2.5]
    g = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    v = torch.tensor(xs2, dtype=torch.float32)
    gg = torch.tensor(g, dtype=torch.float32)
    eps = 1e-6
    # tinh trong float32 nhu torch: (x * rsqrt(mean(x^2) + eps)) * gamma
    ms = v.pow(2).mean()
    out = v * torch.rsqrt(ms + eps) * gg
    write_case("rmsnorm_x", "input", v.tolist())
    write_case("rmsnorm_y", "x * rsqrt(mean(x^2)+1e-6) * gamma", out.tolist())

    # --- 4. Softmax + cross-entropy 1 chuoi ---
    logits = [2.0, 1.0, 0.5, -0.5, 3.0, 0.25]
    lg = torch.tensor([logits], dtype=torch.float32)
    target = torch.tensor([2], dtype=torch.int64)
    loss = F.cross_entropy(lg, target, reduction="mean")
    write_case("ce_logits", "input", lg.reshape(-1).tolist())
    write_case("ce_loss", "F.cross_entropy(reduction='mean')", [loss.item()])

    print("done. meta =", meta)


GRAD_CASES = {}


def gen_grads():
    """Sinh fixture GRADIENT cho tung op (forward + backward deu doi chieu)."""
    torch.set_grad_enabled(True)
    torch.manual_seed(1)

    def w(name, vals):
        write_case("g_" + name, "gradient case", vals)

    # --- matmul: A(3x4) @ B(5x4) -> C(3x5) ---
    m, n, k = 3, 5, 4
    A = torch.randn(m, k, dtype=torch.float32)
    B = torch.randn(n, k, dtype=torch.float32)
    C = A @ B.t()
    GY = torch.randn(m, n, dtype=torch.float32)
    w("mm_A", A.reshape(-1).tolist())
    w("mm_B", B.reshape(-1).tolist())
    w("mm_GY", GY.reshape(-1).tolist())
    w("mm_C", C.reshape(-1).tolist())
    w("mm_dA", (GY @ B).reshape(-1).tolist())
    w("mm_dB", (GY.t() @ A).reshape(-1).tolist())

    # --- add voi broadcast b (8) x (3) ---
    a = torch.randn(8, dtype=torch.float32)
    b = torch.randn(3, dtype=torch.float32)
    gy = torch.randn(8, dtype=torch.float32)
    w("add_a", a.tolist())
    w("add_b", b.tolist())
    w("add_gy", gy.tolist())
    w("add_da", gy.tolist())
    ga = torch.zeros(8, dtype=torch.float32)
    gb = torch.zeros(3, dtype=torch.float32)
    for i in range(8):
        ga[i] = gy[i]
        gb[i % 3] += gy[i]
    w("add_db", gb.tolist())

    # --- mul voi broadcast ---
    w("mul_a", a.tolist())
    w("mul_b", b.tolist())
    w("mul_gy", gy.tolist())
    w("mul_da", (gy * b.repeat(3)[:8]).tolist())
    gb2 = torch.zeros(3, dtype=torch.float32)
    for i in range(8):
        gb2[i % 3] += gy[i] * a[i]
    w("mul_db", gb2.tolist())

    # --- activation: cung mot dau vao cho nhieu op ---
    x0 = torch.tensor([-4.0, -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 4.0,
                       0.7, -0.9, 1.3, -3.1, 0.05, 5.0, -5.0], dtype=torch.float32)
    xg = torch.randn(16, dtype=torch.float32)
    w("act_x", x0.tolist())
    w("act_gy", xg.tolist())

    def act_grad(fn):
        xi = x0.clone().requires_grad_(True)
        return torch.autograd.grad(fn(xi), xi, xg)[0].tolist()

    w("act_dgelu_erf", act_grad(lambda z: F.gelu(z, approximate="none")))
    w("act_dgelu_tanh", act_grad(lambda z: F.gelu(z, approximate="tanh")))
    w("act_dsilu", act_grad(F.silu))
    w("act_drelu", act_grad(F.relu))

    # --- rmsnorm / layernorm ---
    xn = torch.randn(8, dtype=torch.float32) * 2.0
    gyn = torch.randn(8, dtype=torch.float32)
    w("norm_x", xn.tolist())
    w("norm_gy", gyn.tolist())
    for nm, fn in (("rms", F.rms_norm), ("ln", F.layer_norm)):
        xi = xn.clone().requires_grad_(True)
        eps = 1e-6
        if nm == "rms":
            out = xi * torch.rsqrt(xi.pow(2).mean() + eps)
        else:
            out = F.layer_norm(xi, (8,), eps=eps)
        g = torch.autograd.grad(out, xi, gyn)[0]
        w("d" + nm, g.tolist())
        w("y" + nm, out.detach().tolist())

    # --- softmax 2 hang ---
    xs = torch.randn(2, 5, dtype=torch.float32)
    gys = torch.randn(2, 5, dtype=torch.float32)
    xi = xs.clone().requires_grad_(True)
    out = F.softmax(xi, dim=-1)
    gs = torch.autograd.grad(out, xi, gys)[0]
    w("sm_x", xs.reshape(-1).tolist())
    w("sm_gy", gys.reshape(-1).tolist())
    w("sm_y", out.detach().reshape(-1).tolist())
    w("sm_g", gs.reshape(-1).tolist())

    # --- cross entropy voi mask (target LUON la token hop le) ---
    # Quy uoc: vi tri mask==0 bi loai khoi ca tu so va mau so; target phai
    # la vi tri KHONG bi mask. Dung -1e30 de thay the (tuong duong -inf cho
    # softmax), roi lay gradient bang autograd de khong tu viet tay.
    lg = torch.randn(6, dtype=torch.float32)
    mask = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.int64)
    tgt = 1
    cnt = float(int(mask.sum()))
    lgi = lg.clone().requires_grad_(True)
    li = torch.where(mask == 1, lgi, torch.full_like(lgi, -1e30))
    loss = F.cross_entropy(li.unsqueeze(0), torch.tensor([tgt]), reduction="sum") / cnt
    gl = torch.autograd.grad(loss, lgi)[0]
    w("ce_glogits", lg.tolist())
    w("ce_gx", gl.tolist())
    w("ce_gmask", mask.tolist())
    w("ce_gloss", [loss.item()])

    # --- embedding voi chi so lap lai ---
    d = 3
    W = torch.randn(6, d, dtype=torch.float32)
    idx = torch.tensor([0, 2, 2, 5, 0], dtype=torch.int64)
    gyv = torch.randn(5, d, dtype=torch.float32)
    Wi = W.clone().requires_grad_(True)
    out = F.embedding(idx, Wi)
    gW = torch.autograd.grad(out, Wi, gyv)[0]
    w("emb_W", W.reshape(-1).tolist())
    w("emb_idx", [int(v) for v in idx.tolist()])
    w("emb_gy", gyv.reshape(-1).tolist())
    w("emb_gW", gW.reshape(-1).tolist())


def gen_chain():
    """Fixture cho CHUOI OP nhieu tang — dung de kiem tra tape backward tong hop.

    Day la thu ma forward rời không lo duoc: gradient phai chay qua nhieu tang,
    moi tang la mot op khac nhau, va gradient phai cong don ve dau vao dung.
    """
    torch.set_grad_enabled(True)
    torch.manual_seed(2)

    def w(name, vals):
        write_case("c_" + name, "chain case", vals)

    n = 12
    x = torch.randn(n, dtype=torch.float32)
    w1 = torch.randn(n, dtype=torch.float32)
    w2 = torch.randn(n, dtype=torch.float32)
    b1 = torch.randn(n, dtype=torch.float32)
    mask = torch.tensor([1] * 8 + [0, 0, 1, 1], dtype=torch.int64)
    target = 3
    eps = 1e-6

    x.requires_grad_(True)
    w1.requires_grad_(True)
    b1.requires_grad_(True)

    # 6 tang, xen ke nhau de gradient phai chay qua nhieu op khac nhau:
    # rms_norm -> mul -> gelu(erf) -> add -> layernorm -> silu -> CE
    h1 = torch.nn.functional.rms_norm(x, (n,), eps=eps) * w1
    h2 = torch.nn.functional.gelu(h1, approximate="none")
    h3 = h2 + b1
    h4 = torch.nn.functional.layer_norm(h3, (n,), eps=eps)
    h5 = torch.nn.functional.silu(h4)
    keep = (mask == 1)
    sel = torch.where(keep, h5, torch.full_like(h5, -1e30))
    cnt = float(int(mask.sum()))
    loss = (torch.logsumexp(sel, dim=0) - sel[target]) / cnt

    gs = torch.autograd.grad(loss, [x, w1, b1])
    w("x", x.detach().tolist())
    w("w1", w1.detach().tolist())
    w("b1", b1.detach().tolist())
    w("mask", [int(v) for v in mask.tolist()])
    w("loss", [loss.item()])
    w("h1", h1.detach().tolist())
    w("h2", h2.detach().tolist())
    w("h5", h5.detach().tolist())
    w("gx", gs[0].tolist())
    w("gw1", gs[1].tolist())
    w("gb1", gs[2].tolist())



if __name__ == "__main__":
    main()
    print("-- gradient fixtures --")
    gen_grads()
    print("-- chain fixtures (tape) --")
    gen_chain()







