"""DIAG: tim phep bien doi nao bien loss 9.04 (HF) thanh 23.11 (TKV).

Chay tren cung batch 512 token, cung trong so gpt2-medium.
Moi bien the duoc ap len model roi tra ve state goc.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2Config

NL, H, NH, V, T = 24, 1024, 16, 50257, 512
d = np.load("/kaggle/working/gpt2m_ids.npz")
xb, yb = d["xb"], d["yb"]
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
m = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").eval()
xi = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(yb.astype(np.int64)).unsqueeze(0).cuda()
with torch.no_grad():
    emb = m.transformer.wte(xi) + m.transformer.wpe(
        torch.arange(T, device="cuda").unsqueeze(0))
sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}


def restore():
    m.load_state_dict(sd0)


def loss(ids=None, embeds=None):
    with torch.no_grad():
        if embeds is not None:
            return float(m(inputs_embeds=embeds, labels=yi).loss)
        return float(m(input_ids=ids, labels=yi).loss)


def LNs():
    out = [m.transformer.ln_f] + [blk.ln_1 for blk in m.transformer.h] + \
          [blk.ln_2 for blk in m.transformer.h]
    return out


def biases():
    out = []
    for blk in m.transformer.h:
        out += [blk.attn.c_attn.bias, blk.attn.c_proj.bias,
                blk.mlp.c_fc.bias, blk.mlp.c_proj.bias]
    return out


print("DIAG base(ids)      %.4f" % loss(ids=xi), flush=True)
print("DIAG base(embeds)   %.4f" % loss(embeds=emb), flush=True)
with torch.no_grad():
    lg = m(input_ids=xi).logits
print("DIAG logits[0,0:4] %.6f %.6f %.6f %.6f"
      % tuple(float(x) for x in lg[0, 0, :4]), flush=True)
del lg
torch.cuda.empty_cache()

restore()
with torch.no_grad():
    m.transformer.ln_f.weight.fill_(1.0)
    m.transformer.ln_f.bias.fill_(0.0)
print("DIAG ln_f identity  %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    for ln in LNs():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)
print("DIAG all LN identity %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    for b in biases():
        b.zero_()
print("DIAG bias zero      %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    for b in biases():
        b.mul_(2.0)
print("DIAG bias x2        %.4f" % loss(ids=xi), flush=True)

restore()
for blk in m.transformer.h:
    blk.attn.scale = 1.0
print("DIAG attn scale 1.0 %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    e2 = m.transformer.wte(xi)
print("DIAG no wpe         %.4f" % loss(embeds=e2), flush=True)

restore()
for blk in m.transformer.h:
    blk.attn.bias = torch.zeros_like(blk.attn.bias)
print("DIAG no causal mask %.4f" % loss(ids=xi), flush=True)


class LNWrong(nn.Module):
    """LayerNorm tren truc T thay vi truc H."""

    def __init__(self, ln):
        super().__init__()
        self.w = ln.weight
        self.b = ln.bias

    def forward(self, x):
        mu = x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b


restore()
for blk in m.transformer.h:
    blk.ln_1 = LNWrong(blk.ln_1)
    blk.ln_2 = LNWrong(blk.ln_2)
m.transformer.ln_f = LNWrong(m.transformer.ln_f)
print("DIAG LN wrong axis  %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    m.transformer.wte.weight.normal_(0, 0.02)
print("DIAG head random    %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    for blk in m.transformer.h:
        blk.attn.c_proj.weight.mul_(0.0)
print("DIAG no attn out    %.4f" % loss(ids=xi), flush=True)

restore()
with torch.no_grad():
    for blk in m.transformer.h:
        blk.mlp.c_proj.weight.mul_(0.0)
print("DIAG no mlp out     %.4f" % loss(ids=xi), flush=True)

restore()
m2 = m.half()
with torch.no_grad():
    print("DIAG fp16 weights   %.4f" % float(
        m2(input_ids=xi, labels=yi).loss), flush=True)
del m2
torch.cuda.empty_cache()
print("DIAG done", flush=True)
