"""DIAG: so sanh tung buoc forward cua HF voi TKV (cung batch 512 token).

Tag dump (khop voi ilgpt2_real10.tkv):
  1 embedding   2 ln1   3 Q   4 attn-out   5 c_proj   6 x2
  7 ln2         8 z     9 gelu  10 o2   11 out(layer0)   12 ln_f(out layer23)
  20 Wq[0,:8]   21 Wk[0,:8]   22 Wv[0,:8]  (tu c_attn.weight)
"""
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Config

NL, H, NH, V, T = 24, 1024, 16, 50257, 512
NH, HD = 16, H // 16
d = np.load("/kaggle/working/gpt2m_ids.npz")
xb, yb = d["xb"], d["yb"]
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
m = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").eval()
xi = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(yb.astype(np.int64)).unsqueeze(0).cuda()

with torch.no_grad():
    loss = float(m(input_ids=xi, labels=yi).loss)
print("DIAG base(ids) %.4f" % loss, flush=True)

mask = torch.tril(torch.ones(T, T, device="cuda")).view(1, 1, T, T)


def dump(tag, x, n=8):
    v = x.reshape(-1, x.shape[-1])[0, :n].tolist()
    print("DUMP %d %s" % (tag, " ".join("%.6f" % q for q in v)), flush=True)


with torch.no_grad():
    h0 = m.transformer.wte(xi) + m.transformer.wpe(
        torch.arange(T, device="cuda").unsqueeze(0))
    blk = m.transformer.h[0]
    ln1 = blk.ln_1(h0)
    qkv = blk.attn.c_attn(ln1)
    q, k, v = qkv.split(H, dim=2)

    def sp(x):
        return x.view(1, T, NH, HD).permute(0, 2, 1, 3)

    att = (sp(q) @ sp(k).transpose(-2, -1)) / math.sqrt(HD)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = torch.softmax(att, dim=-1)
    y = (att @ sp(v)).permute(0, 2, 1, 3).contiguous().view(1, T, H)
    o = blk.attn.c_proj(y)
    x2 = h0 + o
    ln2 = blk.ln_2(x2)
    z = blk.mlp.c_fc(ln2)
    hh = 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))
    o2 = blk.mlp.c_proj(hh)
    out = x2 + o2
    qpre = q - blk.attn.c_attn.bias[:H].view(1, 1, H)
    for tag, t in ((1, h0), (2, ln1), (3, q), (4, y), (5, o), (6, x2),
                   (7, ln2), (8, z), (9, hh), (10, o2), (11, out)):
        dump(tag, t)
    hs = m(input_ids=xi, output_hidden_states=True).hidden_states
    dump(12, m.transformer.ln_f(hs[-1]))
    dump(23, qpre)
    import numpy as np16
    v16 = np16.float16(ln1[0, 0, :8].detach().cpu().numpy())
    print("DUMP 24 %s" % " ".join("%.6f" % float(x) for x in v16), flush=True)
    lg = m(input_ids=xi).logits
    print("DIAG logits[0,0:4] %.6f %.6f %.6f %.6f"
          % tuple(float(x) for x in lg[0, 0, :4]), flush=True)

wq = m.transformer.h[0].attn.c_attn.weight.detach().cpu().numpy()
v16w = np.float16(wq[0, :8])
print("DUMP 25 %s" % " ".join("%.6f" % float(x) for x in v16w), flush=True)
for tag, row in ((20, 0), (21, H), (22, 2 * H)):
    print("DUMP %d %s" % (tag, " ".join("%.6f" % v for v in wq[row, :8])), flush=True)
_v16 = np.float16(ln1[0, 0, :8].detach().cpu().numpy())
print("DUMP 26 %s" % " ".join("%.6f" % float(x) for x in _v16), flush=True)
print("DUMP 27 %s" % " ".join("%.6f" % float(x) for x in _v16), flush=True)
print("DUMP 30 538.000000 650.000000 762.000000 874.000000  (ref GEMM 4x4, B=A+16)", flush=True)
print("DIAG done", flush=True)
