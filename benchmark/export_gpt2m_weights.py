"""Export full pretrained GPT2-medium weights into the exact layout of
cuda-backend/.../ilgpt2_real.tkv.

File layout (float32 little-endian, contiguous):
  [0 : qb]                 input embeddings for the fixed batch: emb(ids)+wpe, T*H
  [qb : qb+2*hb]           final layer-norm gamma, beta (H each)
  then for each of NL layers:
      Wq, Wk, Wv, Wo       H*H each
      W1                   H*F
      W2                   F*H
      LN1 gamma, LN1 beta, LN2 gamma, LN2 beta   H each

All matrices are transposed from HuggingFace [out,in] to the column-major
layout the .tkv kernels consume (buffer[i + j*rows]).
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Config

T, NL, H, NH, V = 512, 24, 1024, 16, 50257
F = 4 * H
OUT = "/kaggle/working/gpt2m_weights.bin"
IDS = "/kaggle/working/gpt2m_ids.npy"

cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).eval()
sd = model.state_dict()
f32 = lambda a: np.ascontiguousarray(a.detach().cpu().numpy(), dtype=np.float32)

# Fixed real batch: ids from tinyshakespeare (same across both sides).
import urllib.request
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
text = urllib.request.urlopen(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    timeout=60).read().decode("utf-8")
ids = np.array(tok.encode(text[:20000]), dtype=np.int64)[:T + 1]
xb, yb = ids[:-1], ids[1:]
np.save(IDS, np.stack([xb, yb]))

wte = f32(sd["transformer.wte.weight"])          # [V,H]
wpe = f32(sd["transformer.wpe.weight"])[:T]       # [T,H]
emb = (wte[xb] + wpe).T.copy()                    # [H,T] -> linear i+j*T

parts = [emb,
         f32(sd["transformer.ln_f.weight"]),
         f32(sd["transformer.ln_f.bias"])]
for i in range(NL):
    p = "transformer.h.%d." % i
    qkv = f32(sd[p + "attn.c_attn.weight"])        # [3H,H]
    parts.append(qkv[0:H].T.copy())                # Wq [H,H]
    parts.append(qkv[H:2*H].T.copy())              # Wk
    parts.append(qkv[2*H:3*H].T.copy())            # Wv
    parts.append(f32(sd[p + "attn.c_proj.weight"]).T.copy())
    parts.append(f32(sd[p + "mlp.c_fc.weight"]).T.copy())     # [H,F]
    parts.append(f32(sd[p + "mlp.c_proj.weight"]).T.copy())   # [F,H]
    parts.append(f32(sd[p + "ln_1.weight"]))
    parts.append(f32(sd[p + "ln_1.bias"]))
    parts.append(f32(sd[p + "ln_2.weight"]))
    parts.append(f32(sd[p + "ln_2.bias"]))
    head = f32(sd[p + "attn.c_attn.bias"])  # unused by this trainer, keep layout honest
# head (tied wte) last
parts.append(wte.T.copy())                        # [H,V]

blob = b"".join(p.tobytes(order="C") for p in parts)
with open(OUT, "wb") as f:
    f.write(blob)
print("EXPORT", OUT, len(blob), "bytes; ids", len(xb), "loss baseline ready")
