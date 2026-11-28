"""B_2/3 - train that: PyTorch eager vs TokenVector pure .tkv.

So sanh theo so UPDATE bang nhau: sau 12 update.
TKV in loss truoc moi update, nen step 13 = sau 12 update.
Doi chieu gradient dWq cua lop 0 (step 1) voi PyTorch de kiem tra backward.
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

# --- tham chieu: loss + gradient o TRANG THAI BAN DAU (truoc moi update) ---
model.zero_grad(set_to_none=True)
_l0 = model(input_ids=xi, labels=yi).loss
_l0.backward()
GQ_REF = (model.transformer.h[0].attn.c_attn.weight.grad[:256, 0]
          .detach().cpu().numpy().astype(np.float64))
model.zero_grad(set_to_none=True)
print("TORCH step0 loss %.4f" % float(_l0), flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
times, losses_all = [], []
for step in range(14):
    t0 = time.perf_counter()
    loss = model(input_ids=xi, labels=yi).loss
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    losses_all.append(float(loss.detach()))
    if step >= 2:
        times.append((time.perf_counter() - t0) * 1000.0)
torch_ms = st.median(times)
print("TORCH trace: " + " -> ".join("%.4f" % x for x in losses_all), flush=True)
print("TORCH train: median %.1f ms/step  loss sau 12 update %.4f"
      % (torch_ms, losses_all[12]), flush=True)
torch.save(model.state_dict(), "/kaggle/working/trained_gpt2m.pt")
print("SAVED /kaggle/working/trained_gpt2m.pt", flush=True)
del opt
gc.collect(); torch.cuda.empty_cache()

cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_real10.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
LRF = "/kaggle/working/tkv_lr.bin"

runs = []
for lrv in (1e-4, 2e-4, 3e-4):
    with open(LRF, "wb") as _f:
        _f.write(np.float32(lrv).tobytes())
    r = subprocess.run([DOTNET, EXE], env=dict(os.environ), capture_output=True,
                       text=True, timeout=2400)
    lines = (r.stdout or "").splitlines()
    tl = [float(l.split()[1]) for l in lines if l.startswith("LOSSF ")]
    ms = [float(l.split()[1]) for l in lines if l.startswith("STEPMS ")]
    gq = [float(l.split()[1]) for l in lines if l.startswith("GQ0 ")]
    print("TKV lr=%.0e rc=%d" % (lrv, r.returncode), flush=True)
    if r.returncode != 0 or not tl:
        print((r.stdout or "")[-2000:]); print((r.stderr or "")[:800], flush=True)
        continue
    print("TKV trace lr=%.0e: " % lrv + " -> ".join("%.4f" % x for x in tl), flush=True)
    drops = [round(tl[i - 1] - tl[i], 4) for i in range(1, len(tl))]
    neg = sum(1 for x in drops if x < 0)
    print("TKV drops lr=%.0e: " % lrv + ", ".join("%.4f" % x for x in drops)
          + "  (so step tang: %d)" % neg, flush=True)
    if ms:
        print("TKV ms lr=%.0e median %.1f" % (lrv, st.median(ms)), flush=True)
    if len(gq) == 256:
        a = np.array(gq, dtype=np.float64)
        b = GQ_REF
        print("GRADCHK dWq[0:256,0] corr=%.5f  maxerr=%.3e  |tkv|=%.3e  |ref|=%.3e  "
              "ratio=%.4f" % (float(np.corrcoef(a, b)[0, 1]),
                              float(np.max(np.abs(a - b))),
                              float(np.linalg.norm(a)), float(np.linalg.norm(b)),
                              float(np.linalg.norm(a) / np.linalg.norm(b))), flush=True)
        print("GRADCHK tkv[:6]=" + " ".join("%.3e" % v for v in a[:6]), flush=True)
        print("GRADCHK ref[:6]=" + " ".join("%.3e" % v for v in b[:6]), flush=True)
    runs.append((lrv, tl, st.median(ms) if ms else 0.0, neg))

if not runs:
    sys.exit(1)
best = min(runs, key=lambda z: z[1][-1])
print("BENCH TABLE (no C++ bridge) - 12 update moi ben", flush=True)
print("| implementation | weights | median ms/step | speedup | loss sau 12 update |", flush=True)
print("| :--- | :--- | ---: | ---: | :--- |", flush=True)
print("| PyTorch eager (lr 5e-5, all params) | gpt2-medium pretrained | %.1f | 1.00x | %.4f |"
      % (torch_ms, losses_all[12]), flush=True)
print("| TokenVector .tkv (lr %.0e) | gpt2-medium pretrained | %.1f | %.2fx | %.4f |"
      % (best[0], best[2], torch_ms / best[2], best[1][-1]), flush=True)
print("BEST lr=%.0e  loss %.4f  steps tang %d" % (best[0], best[1][-1], best[3]), flush=True)
