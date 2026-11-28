"""So sanh tung diem mot cua backward: TKV vs PyTorch, tai step-1 (truoc update).

TKV duoc them cac dong DG* / DY10 trong ilgpt2_real10.tkv (chi step==1).
Torch dung hook de lay cung mot dai luong:
  DGD0  = dL/dlogits[:,0]
  DGO0  = dL/d(hidden cuoi) = grad_input cua ln_f
  DGQ23 = grad_output cua h[23].attn.c_attn, cot 0
  DGQ00 = grad_output cua h[0].attn.c_attn, cot 0
  DY10  = input cua h[0].attn.c_attn (= LN1 output), cot 0
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as tnf
from transformers import AutoModelForCausalLM, GPT2Config

BASE = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/TokenVectorInfer"
EXE = BASE + "/cuda-backend/Release/cuda_backend/examples/ilgpt2_real10.exe"
DOTNET = "/kaggle/working/dotnet/dotnet"
NL, H, NH, V, T, F = 24, 1024, 16, 50257, 512, 4096
IDX = list(range(0, T, 16))   # 32 vi tri

d = np.load("/kaggle/working/gpt2m_ids.npz")
xi = torch.from_numpy(d["xb"].astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(d["yb"].astype(np.int64)).unsqueeze(0).cuda()
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()

cap = {}
def cap_bw(tag):
    def fn(m, gi, go):
        cap[tag] = go[0][0, :, 0].detach().float().cpu().numpy()[IDX]
    return fn
def cap_lnf(tag):
    def fn(m, gi, go):
        cap[tag] = gi[0][0, :, 0].detach().float().cpu().numpy()[IDX]
    return fn
def cap_fwd(tag):
    def fn(m, inp, out):
        cap[tag] = inp[0][0, :, 0].detach().float().cpu().numpy()[IDX]
    return fn

model.transformer.ln_f.register_full_backward_hook(cap_lnf("DGO0"))
model.transformer.h[23].attn.c_attn.register_full_backward_hook(cap_bw("DGQ23"))
model.transformer.h[0].attn.c_attn.register_full_backward_hook(cap_bw("DGQ00"))
model.transformer.h[0].attn.c_attn.register_forward_hook(cap_fwd("DY10"))

logits = model(input_ids=xi).logits
logits.retain_grad()
tnf.cross_entropy(logits[0], yi[0]).backward()
cap["DGD0"] = logits.grad[0, :, 0].detach().float().cpu().numpy()[IDX]
print("TORCH ref loss %.4f" % float(tnf.cross_entropy(logits.detach()[0], yi[0])), flush=True)

cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_real10.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
with open("/kaggle/working/tkv_lr.bin", "wb") as f:
    f.write(np.float32(1e-4).tobytes())
r = subprocess.run([DOTNET, EXE], env=dict(os.environ), capture_output=True,
                   text=True, timeout=2400)
print("TKV rc=%d" % r.returncode, flush=True)

tags = ("DGD0", "DGO0", "DGQ23", "DGQ00", "DY10")
vals = {t: [] for t in tags}
for line in (r.stdout or "").splitlines():
    p = line.split()
    if len(p) == 2 and p[0] in vals:
        try:
            vals[p[0]].append(float(p[1]))
        except ValueError:
            pass

for t in tags:
    a = np.array(vals[t], dtype=np.float64)
    b = cap[t]
    if a.size != 32:
        print("DIAG %-6s MISSING (%d values)" % (t, a.size), flush=True)
        continue
    corr = float(np.corrcoef(a, b)[0, 1])
    print("DIAG %-6s n=%d corr=%+.5f  |a|=%.3e |b|=%.3e maxerr=%.3e"
          % (t, a.size, corr, np.abs(a).mean(), np.abs(b).mean(),
             np.max(np.abs(a - b))), flush=True)
    print("  tkv[:5] =" + " ".join("%+.4e" % v for v in a[:5]), flush=True)
    print("  ref[:5] =" + " ".join("%+.4e" % v for v in b[:5]), flush=True)
