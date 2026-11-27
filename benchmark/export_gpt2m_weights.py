"""Export full pretrained GPT2-medium weights into the exact layout of
cuda-backend/.../ilgpt2_real10.tkv, plus a train/test split of real text.

Noi dung train: 4096 token dau cua tinyshakespeare (corpus that dang hoc).
Train batch = 512 token dau; test batch = 512 token tiep theo (held-out).
"""
import numpy as np
import torch
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

T, NL, H, NH, V = 512, 24, 1024, 16, 50257
F = 4 * H
OUT = "/kaggle/working/gpt2m_weights.bin"
IDS = "/kaggle/working/gpt2m_ids.npz"

cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).eval()
sd = model.state_dict()
f32 = lambda a: np.ascontiguousarray(a.detach().cpu().numpy(), dtype=np.float32)

tok = AutoTokenizer.from_pretrained("gpt2")
text = urllib.request.urlopen(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    timeout=60).read().decode("utf-8")
ids = np.array(tok.encode(text), dtype=np.int64)
xb, yb = ids[:T], ids[1:T + 1]                 # train
xt, yt = ids[T:2 * T], ids[T + 1:2 * T + 1]    # held-out test
np.savez(IDS, xb=xb, yb=yb, xt=xt, yt=yt)
print("CORPUS chars=%d tokens=%d | train 512 | held-out 512" % (len(text), len(ids)))

wte = f32(sd["transformer.wte.weight"])
wpe = f32(sd["transformer.wpe.weight"])[:T]
emb = (wte[xb] + wpe).copy()             # [T,H] -> column-major i+j*T

parts = [emb, f32(sd["transformer.ln_f.weight"]), f32(sd["transformer.ln_f.bias"])]
for i in range(NL):
    p = "transformer.h.%d." % i
    qkv = f32(sd[p + "attn.c_attn.weight"])
    parts += [qkv[0:H].T.copy(), qkv[H:2 * H].T.copy(), qkv[2 * H:3 * H].T.copy(),
              f32(sd[p + "attn.c_proj.weight"]).T.copy(),
              f32(sd[p + "mlp.c_fc.weight"]).T.copy(),
              f32(sd[p + "mlp.c_proj.weight"]).T.copy(),
              f32(sd[p + "ln_1.weight"]), f32(sd[p + "ln_1.bias"]),
              f32(sd[p + "ln_2.weight"]), f32(sd[p + "ln_2.bias"])]
parts.append(wte.T.copy())              # tied head, last

blob = b"".join(p.tobytes(order="F") for p in parts)
with open(OUT, "wb") as f:
    f.write(blob)
print("EXPORT", OUT, len(blob), "bytes")
