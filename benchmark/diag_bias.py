"""In 8 gia tri dau cua moi vector bias/LN trong layer 0 de tim offset dang doc."""
import numpy as np
from transformers import AutoModelForCausalLM, GPT2Config

cfg = GPT2Config(n_layer=24, n_head=16, n_embd=1024, n_positions=1024,
                 vocab_size=50257, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
m = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).eval()
b = m.transformer.h[0]
items = [
    ("ln1w", b.ln_1.weight), ("ln1b", b.ln_1.bias),
    ("ln2w", b.ln_2.weight), ("ln2b", b.ln_2.bias),
    ("attnb", b.attn.c_attn.bias), ("projb", b.attn.c_proj.bias),
    ("fcb", b.mlp.c_fc.bias), ("mlpb", b.mlp.c_proj.bias),
    ("lnfw", m.transformer.ln_f.weight), ("lnfb", m.transformer.ln_f.bias),
]
for name, t in items:
    v = t.detach().cpu().numpy()[:8]
    print("CAND %-6s %s" % (name, " ".join("%.6f" % x for x in v)), flush=True)
