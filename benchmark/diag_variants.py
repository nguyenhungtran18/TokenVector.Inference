"""DIAG: kiem tra loss HF tren cung batch 512 token + in logits de so voi TKV.

In ra:
  DIAG base(ids)        loss cua gpt2-medium pretrained tren 512 token dau
  DIAG logits[0,0:4]    4 logit dau tien cua vi tri 0 (so sanh truc tiep voi TKV LG0)
  DIAG <bien the>       vai phep bien doi de tham chieu
"""
import numpy as np
import torch
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
sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}


def loss():
    with torch.no_grad():
        return float(m(input_ids=xi, labels=yi).loss)


def tryp(name, fn):
    try:
        fn()
        print("DIAG %-20s %.4f" % (name, loss()), flush=True)
    except Exception as e:  # noqa: BLE001
        print("DIAG %-20s SKIP %s" % (name, type(e).__name__), flush=True)
    m.load_state_dict(sd0)


print("DIAG base(ids)          %.4f" % loss(), flush=True)
with torch.no_grad():
    lg = m(input_ids=xi).logits
print("DIAG logits[0,0:4] %.6f %.6f %.6f %.6f"
      % tuple(float(x) for x in lg[0, 0, :4]), flush=True)
del lg
torch.cuda.empty_cache()


def lnf_identity():
    with torch.no_grad():
        m.transformer.ln_f.weight.fill_(1.0)
        m.transformer.ln_f.bias.fill_(0.0)


def all_ln_identity():
    with torch.no_grad():
        for ln in [m.transformer.ln_f] + [b.ln_1 for b in m.transformer.h] + \
                  [b.ln_2 for b in m.transformer.h]:
            ln.weight.fill_(1.0)
            ln.bias.fill_(0.0)


def bias_zero():
    with torch.no_grad():
        for b in m.transformer.h:
            b.attn.c_attn.bias.zero_()
            b.attn.c_proj.bias.zero_()
            b.mlp.c_fc.bias.zero_()
            b.mlp.c_proj.bias.zero_()


def no_attn_out():
    with torch.no_grad():
        for b in m.transformer.h:
            b.attn.c_proj.weight.zero_()


def no_mlp_out():
    with torch.no_grad():
        for b in m.transformer.h:
            b.mlp.c_proj.weight.zero_()


tryp("ln_f identity", lnf_identity)
tryp("all LN identity", all_ln_identity)
tryp("bias zero", bias_zero)
tryp("no attn out", no_attn_out)
tryp("no mlp out", no_mlp_out)
print("DIAG done", flush=True)
