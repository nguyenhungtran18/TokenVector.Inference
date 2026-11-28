"""BÆ¯á»šC 2/3 - train that: PyTorch eager vs TokenVector pure .tkv.
Chay sau export_gpt2m_weights.py. Luu model da train de cell test dung lai.
"""
import gc
import json
import os
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Config

BASE = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/TokenVectorInfer"
EXE = BASE + "/cuda-backend/Release/cuda_backend/examples/ilgpt2_real10.exe"
DOTNET = "/kaggle/working/dotnet/dotnet"
NL, H, NH, V, T = 24, 1024, 16, 50257, 512


d = np.load("/kaggle/working/gpt2m_ids.npz")
xb, yb = d["xb"], d["yb"]
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()
xi = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(yb.astype(np.int64)).unsqueeze(0).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
times, losses = [], []
for step in range(2 + 10):
    t0 = time.perf_counter()
    loss = model(input_ids=xi, labels=yi).loss
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if step >= 2:
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.detach()))
torch_ms = st.median(times)
print("TORCH train: median %.1f ms/step  loss %.4f -> %.4f"
      % (torch_ms, losses[0], losses[-1]), flush=True)
torch.save(model.state_dict(), "/kaggle/working/trained_gpt2m.pt")
print("SAVED /kaggle/working/trained_gpt2m.pt", flush=True)
del opt
gc.collect(); torch.cuda.empty_cache()

cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_real10.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
LRF = "/kaggle/working/tkv_lr.bin"
with open(LRF, "wb") as _f:
    _f.write(np.float32(0.0001).tobytes())
r = subprocess.run([DOTNET, EXE], env=dict(os.environ), capture_output=True, text=True, timeout=1200)
print("TKV rc=", r.returncode, flush=True)
if r.returncode != 0:
    print((r.stdout or "")[-2500:]); print((r.stderr or "")[:800]); sys.exit(1)
lines = (r.stdout or "").splitlines()
ms = [float(l.split()[1]) for l in lines if l.startswith("STEPMS ")]
tl = [float(l.split()[1]) for l in lines if l.startswith("LOSSF ")]
if not tl:
    sys.exit(1)
tkv_ms = st.median(ms[3:]) if len(ms) > 6 else st.median(ms)
print("TKV step losses: " + " -> ".join("%.4f" % x for x in tl), flush=True)
drops = [round(tl[i - 1] - tl[i], 4) for i in range(1, len(tl))]
print("TKV per-step drops: " + ", ".join("%.4f" % x for x in drops), flush=True)
end_loss, first_loss = tl[-1], tl[0]
print("TKV train: median %.1f ms/step  loss %.4f -> %.4f"
      % (tkv_ms, first_loss, end_loss), flush=True)
print("BENCH TABLE (no C++ bridge)", flush=True)
print("| implementation | weights | median ms/step | speedup | train loss |", flush=True)
print("| :--- | :--- | ---: | ---: | :--- |", flush=True)
print("| PyTorch eager | gpt2-medium pretrained | %.1f | 1.00x | %.4f -> %.4f |"
      % (torch_ms, losses[0], losses[-1]), flush=True)
print("| TokenVector pure .tkv | gpt2-medium pretrained | %.1f | %.2fx | %.4f -> %.4f |"
      % (tkv_ms, torch_ms / tkv_ms, first_loss, end_loss), flush=True)
