"""Tim hieu vi sao TKV loss cao: PyTorch nhung CHI train cac tham so ma TKV train.

TKV hien cap nhat: 4 ma tran/层 (c_attn, c_proj, mlp.c_fc, mlp.c_proj) + bias
+ tied head. TKV KHONG cap nhat: LayerNorm gamma/beta (ln1/ln2/ln_f), wte embedding.

Ablation A = dung tap tham so cua TKV (LN + wte frozen, head frozen)
Ablation B = them tied head (wte) duoc train
Baseline   = train tat ca (ket qua da biet ~0.0001 sau 150 step)
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Config

N, H, V, T = 24, 1024, 50257, 512
d = np.load("/kaggle/working/gpt2m_ids.npz")
xi = torch.from_numpy(d["xb"].astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(d["yb"].astype(np.int64)).unsqueeze(0).cuda()
cfg = GPT2Config(n_layer=N, n_head=16, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)


def run(include_head, steps=150, lr0=1e-4):
    m = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()
    for p in m.parameters():
        p.requires_grad_(False)
    trainable = []
    for b in m.transformer.h:
        for name in ("attn.c_attn.weight", "attn.c_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight",
                     "attn.c_attn.bias", "attn.c_proj.bias",
                     "mlp.c_fc.bias", "mlp.c_proj.bias"):
            p = dict(b.named_parameters())[name]
            p.requires_grad_(True)
            trainable.append(p)
    if include_head:
        for p in m.lm_head.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=lr0, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)
    out = {}
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr0 * min(step / 10.0, 1.0)
        loss = m(input_ids=xi, labels=yi).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in (1, 10, 50, 100, 150):
            out[step] = float(loss.detach())
    del m, opt
    torch.cuda.empty_cache()
    return out


for inc in (False, True):
    r = run(inc)
    print("ABL head=%s  " % inc + "  ".join("s%d=%.4f" % (k, v) for k, v in r.items()),
          flush=True)
print("ABL done", flush=True)
