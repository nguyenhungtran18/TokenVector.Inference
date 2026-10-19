#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh fixture cho GPT-2 reference (G2).

Tai sao dung GPT-2 THU NHO (n_embd=64, n_head=4, n_layer=2, vocab=256)?

Muc tieu cua G2 la CHUNG MINH THUAT TOAN, khong phai chung minh kha nang
chua 124M tham so. Kien truc giong HF `gpt2` y nguyen:
  - embedding wte + wpe
  - moi block: ln -> attn (QKV + causal mask + softmax) -> residual
               -> ln -> mlp (linear -> gelu erf -> linear) -> residual
  - ln cuoi + LM head TIE weight voi wte
  - activation_function='gelu' (erf chinh xac, khop numerics-contract)
  - dropout = 0 (muc 5 cua numerics-contract)

Thu nho chi de fixture nho (duoc check-in), con thuat toan thi y nguyen.

Chay:  python tools\\gen_gpt2_fixture.py
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
OUT = os.path.join(ROOT, "tv", "tests", "fixtures")

# --out <thu muc>: ghi ra thu muc khac (dung boi verify_parity.ps1).
if "--out" in sys.argv:
    _i = sys.argv.index("--out")
    OUT = os.path.abspath(sys.argv[_i + 1])

# ---- cau hinh GPT-2 thu nho (giong HF, chi giam kich thuoc) ---------------
# (Ban 124M that su bi chan: wte ~425 MB text khong check-in duoc, xem
# HANDOFF.md 0.6. Giu ban thu nho lam bang chung thuat toan.)
VOCAB = 256
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
MAX_POS = 32
SEQ = 8
EPS = 1e-6
SEED = 7


def write_case(name, header, values):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "g2_" + name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# n=%d  %s\n" % (len(values), header))
        f.write(" ".join("%.9g" % float(v) for v in values) + "\n")
    print("  -> g2_%-22s n=%d" % (name, len(values)))
    return path


def main():
    torch.set_grad_enabled(True)
    torch.manual_seed(SEED)

    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=MAX_POS,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        activation_function="gelu",      # erf chinh xac
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=EPS,
        use_cache=False,
        torch_dtype=torch.float32,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()

    # ---- token ids co dinh ----
    ids = torch.tensor([[3, 17, 42, 5, 200, 88, 12, 7]], dtype=torch.int64)
    B, T = ids.shape
    pos = torch.arange(T, dtype=torch.int64).unsqueeze(0)

    # ---- forward ----
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=pos)
        logits = out.logits                       # [B, T, VOCAB]
    loss = F.cross_entropy(logits[0, :, :].reshape(-1, VOCAB),
                           ids[0].reshape(-1), reduction="mean")

    # ---- lay weights (HF dung Conv1D: weight [in, out]) ----
    w = {}
    w["wte"] = model.transformer.wte.weight.detach()      # [VOCAB, N_EMBD]
    w["wpe"] = model.transformer.wpe.weight.detach()      # [MAX_POS, N_EMBD]
    for i in range(N_LAYER):
        b = model.transformer.h[i]
        w["ln1_g%d" % i] = b.ln_1.weight.detach()
        w["ln1_b%d" % i] = b.ln_1.bias.detach()
        w["ln2_g%d" % i] = b.ln_2.weight.detach()
        w["ln2_b%d" % i] = b.ln_2.bias.detach()
        w["attn_w%d" % i] = b.attn.c_attn.weight.detach()   # [N_EMBD, 3*N_EMBD]
        w["attn_b%d" % i] = b.attn.c_attn.bias.detach()     # [3*N_EMBD]
        w["proj_w%d" % i] = b.attn.c_proj.weight.detach()   # [N_EMBD, N_EMBD]
        w["proj_b%d" % i] = b.attn.c_proj.bias.detach()
        w["mlp_w1%d" % i] = b.mlp.c_fc.weight.detach()      # [N_EMBD, 4*N_EMBD]
        w["mlp_b1%d" % i] = b.mlp.c_fc.bias.detach()
        w["mlp_w2%d" % i] = b.mlp.c_proj.weight.detach()    # [4*N_EMBD, N_EMBD]
        w["mlp_b2%d" % i] = b.mlp.c_proj.bias.detach()
    w["lnf_g"] = model.transformer.ln_f.weight.detach()
    w["lnf_b"] = model.transformer.ln_f.bias.detach()
    # LM head TIE weight voi wte
    assert model.lm_head.weight.data_ptr() == w["wte"].data_ptr()

    # ---- hidden states tung tang (de khoanh vung loi) ----
    h = model.transformer.wte(ids) + model.transformer.wpe(pos)
    hs = [h.detach()]
    for i in range(N_LAYER):
        h = model.transformer.h[i](h)
        hs.append(h.detach())

    # ---- gradient cho vai tham so chon mau ----
    params = {
        "wte": model.transformer.wte.weight,
        "ln1_g0": model.transformer.h[0].ln_1.weight,
        "attn_w0": model.transformer.h[0].attn.c_attn.weight,
        "mlp_w1_0": model.transformer.h[0].mlp.c_fc.weight,
        "lnf_g": model.transformer.ln_f.weight,
    }
    for p in params.values():
        p.requires_grad_(True)
    out2 = model(input_ids=ids, position_ids=pos)
    loss2 = F.cross_entropy(out2.logits[0].reshape(-1, VOCAB), ids[0].reshape(-1),
                            reduction="mean")
    grads = torch.autograd.grad(loss2, list(params.values()))

    # ---- ghi fixture ----
    write_case("ids", "input token ids", ids.reshape(-1).tolist())
    write_case("pos", "position ids", pos.reshape(-1).tolist())
    write_case("loss", "cross_entropy mean", [loss.item()])
    write_case("logits", "logits [B,T,VOCAB]", logits.reshape(-1).tolist())
    for k, v in w.items():
        write_case("w_" + k, "weight " + k, v.reshape(-1).tolist())
    for i, hh in enumerate(hs):
        write_case("h%d" % i, "hidden sau block %d" % i, hh.reshape(-1).tolist())
    for i, (k, g) in enumerate(zip(params.keys(), grads)):
        write_case("grad_" + k, "grad " + k, g.reshape(-1).tolist())

    # ---- G2 bo sung: padding mask (attention_mask) ----------------------
    # Pad o CUOI (causal + padding). Query o vi tri pad van attend cac key hop
    # le (giong HF); loss chi tinh tren vi tri mask=1 (labels=-100).
    ids_p = torch.tensor([[3, 17, 42, 5, 200, 88, 0, 0]], dtype=torch.int64)
    mask_p = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.int64)
    pos_p = torch.arange(T, dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        out_p = model(input_ids=ids_p, position_ids=pos_p, attention_mask=mask_p)
        logits_p = out_p.logits
        # Hidden tung tang TRUOC ln_f: ban transformers 5.x chi tra
        # (embed, block0, ln_f) — thieu block cuoi. Nen tu chay tung block voi
        # mask 4D [B,1,T,T] = causal & padding (da kiem chung khop
        # hidden_states[1] cua HF bit-for-bit), roi ln_f rieng.
        _T = ids_p.shape[1]
        _causal = torch.tril(torch.ones(_T, _T, dtype=torch.bool))[None, None]
        _pmask = mask_p[:, None, None, :].to(torch.bool)
        _am4 = _causal & _pmask
        _h = model.transformer.wte(ids_p) + model.transformer.wpe(pos_p)
        _h = model.transformer.drop(_h)
        hs_p = [_h.detach()]
        for _i in range(N_LAYER):
            _h = model.transformer.h[_i](_h, attention_mask=_am4)
            hs_p.append(_h.detach())
        lnf_p = model.transformer.ln_f(_h).detach()
    valid = mask_p.reshape(-1).bool()
    lp = logits_p[0][valid]
    yp = ids_p[0][valid]
    loss_p = F.cross_entropy(lp, yp, reduction="mean")
    write_case("m_ids", "ids co padding", ids_p.reshape(-1).tolist())
    write_case("m_mask", "attention_mask", mask_p.reshape(-1).tolist())
    write_case("m_pos", "position ids", pos_p.reshape(-1).tolist())
    write_case("m_logits", "logits [T,VOCAB] co padding", logits_p.reshape(-1).tolist())
    write_case("m_loss", "CE mean chi tren vi tri mask=1", [loss_p.item()])
    for i, hh in enumerate(hs_p):
        write_case("m_h%d" % i, "hidden sau block %d (mask)" % i, hh.reshape(-1).tolist())
    write_case("m_lnf", "hidden sau ln_f (mask)", lnf_p.reshape(-1).tolist())

    print("done. config = vocab=%d n_embd=%d n_head=%d n_layer=%d seq=%d"
          % (VOCAB, N_EMBD, N_HEAD, N_LAYER, SEQ))


if __name__ == "__main__":
    main()
