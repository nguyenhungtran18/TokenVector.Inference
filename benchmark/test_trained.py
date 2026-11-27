"""BƯỚC 3/3 - test noi dung da train AI.
Doc model da train o buoc 2, do loss tren tap held-out va sinh text tu prompt.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

NL, H, NH, V, T = 24, 1024, 16, 50257, 512
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg)
model.load_state_dict(torch.load("/kaggle/working/trained_gpt2m.pt", map_location="cpu"))
model = model.to("cuda").eval()

d = np.load("/kaggle/working/gpt2m_ids.npz")
xtr, ytr, xte, yte = (d[k] for k in ("xb", "yb", "xt", "yt"))
def loss_of(x, y):
    with torch.no_grad():
        return float(model(input_ids=torch.from_numpy(x.astype(np.int64)).unsqueeze(0).cuda(),
                           labels=torch.from_numpy(y.astype(np.int64)).unsqueeze(0).cuda()).loss)
print("TEST - trained AI", flush=True)
print("Noi dung hoc: tinyshakespeare, 4096 token; train = 512 dau, held-out = 512 tiep.", flush=True)
print("CE loss train batch   = %.4f" % loss_of(xtr, ytr), flush=True)
print("CE loss held-out      = %.4f  (baseline pretrained ~8.97)" % loss_of(xte, yte), flush=True)

tok = AutoTokenizer.from_pretrained("gpt2")
for prompt in ("ROMEO: ", "FIRST CITIZEN: ", "HAMLET: "):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=50256)
    print("PROMPT %-18s -> %s" % (repr(prompt), tok.decode(out[0].tolist()).replace("\n", " | ")), flush=True)
