"""BƯỚC 3/3 - test noi dung da train AI.

CUNG mot nhiem vu voi luc train: position t -> token t+1 (KHONG de HF shift lan 2).
Do CE tren tap train va held-out, roi sinh text tu prompt.
"""
import numpy as np
import torch
import torch.nn.functional as tnf
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

NL, H, NH, V, T = 24, 1024, 16, 50257, 512
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").eval()

d = np.load("/kaggle/working/gpt2m_ids.npz")
xtr, ytr, xte, yte = (d[k] for k in ("xb", "yb", "xt", "yt"))
xtr = torch.from_numpy(xtr.astype(np.int64)).unsqueeze(0).cuda()
ytr = torch.from_numpy(ytr.astype(np.int64)).unsqueeze(0).cuda()
xte = torch.from_numpy(xte.astype(np.int64)).unsqueeze(0).cuda()
yte = torch.from_numpy(yte.astype(np.int64)).unsqueeze(0).cuda()


def loss_of(x, y):
    with torch.no_grad():
        return float(tnf.cross_entropy(model(input_ids=x).logits[0], y[0]))


print("TEST - trained AI", flush=True)
print("Noi dung hoc: tinyshakespeare, 4096 token; train = 512 dau, held-out = 512 tiep.", flush=True)
print("BASELINE pretrained held-out CE = %.4f" % loss_of(xte, yte), flush=True)
model.load_state_dict(torch.load("/kaggle/working/trained_gpt2m.pt", map_location="cpu"))
model = model.to("cuda").eval()
print("CE loss train batch   = %.4f" % loss_of(xtr, ytr), flush=True)
print("CE loss held-out      = %.4f" % loss_of(xte, yte), flush=True)

tok = AutoTokenizer.from_pretrained("gpt2")
for prompt in ("ROMEO: ", "FIRST CITIZEN: ", "HAMLET: "):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=50256)
    print("PROMPT %-18s -> %s" % (repr(prompt), tok.decode(out[0].tolist()).replace("\n", " | ")), flush=True)
