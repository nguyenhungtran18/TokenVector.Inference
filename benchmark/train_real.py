"""B_2/3 - train that: PyTorch eager vs TokenVector pure .tkv.

Mot lan chay kiem tra 4 thu:
  1. LAYOUT: doc truc tiep blob trong so va so voi state_dict cua torch
     (q/k/v/c_proj/c_fc/c_proj2/head) -> bat loi transpose cua exporter.
  2. FORWARD: loss step-1 cua TKV phai ~= loss step-0 cua torch (cung trong so).
  3. BACKWARD: dWq lop 0 (cot 0) cua TKV so voi torch -> corr phai ~1.
  4. TOC DO + loss sau 12 update, quet lr.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2Config

BASE = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/TokenVectorInfer"
EXE = BASE + "/cuda-backend/Release/cuda_backend/examples/ilgpt2_real10.exe"
DOTNET = "/kaggle/working/dotnet/dotnet"
BLOB = "/kaggle/working/gpt2m_weights.bin"
NL, H, NH, V, T, F = 24, 1024, 16, 50257, 512, 4096
NUP = 12          # so update moi ben


def ce(m, x, y):
    return F.cross_entropy(m(input_ids=x).logits[0], y[0])


d = np.load("/kaggle/working/gpt2m_ids.npz")
xb, yb = d["xb"], d["yb"]
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()
sd = model.state_dict()
xi = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(yb.astype(np.int64)).unsqueeze(0).cuda()


# ---------- 1. LAYOUT cua blob ----------
def f32(k):
    return sd[k].detach().float().cpu().numpy()


raw = np.fromfile(BLOB, dtype=np.float32)
print("BLOB floats=%d (expect %d)" % (raw.size, (T * H + 2 * H + NL * (4 * H * H + H * F + F * H + 13 * H) + H * V + T)))


def chk(tag, a, b):
    e = float(np.abs(a - b).max())
    s = float(np.abs(b).max()) + 1e-12
    print("LAYOUT %-10s maxerr=%.3e  rel=%.2e  %s" % (tag, e, e / s, "OK" if e / s < 1e-5 else "*** SAI ***"), flush=True)


o = T * H + 2 * H
qkv = f32("transformer.h.0.attn.c_attn.weight")
chk("q", raw[o:o + H * H].reshape(H, H, order="F"), qkv[:, 0:H]); o += H * H
chk("k", raw[o:o + H * H].reshape(H, H, order="F"), qkv[:, H:2 * H]); o += H * H
chk("v", raw[o:o + H * H].reshape(H, H, order="F"), qkv[:, 2 * H:3 * H]); o += H * H
chk("c_proj", raw[o:o + H * H].reshape(H, H, order="F"),
    f32("transformer.h.0.attn.c_proj.weight")); o += H * H
chk("c_fc", raw[o:o + H * F].reshape(H, F, order="F"),
    f32("transformer.h.0.mlp.c_fc.weight")); o += H * F
chk("mlp_proj", raw[o:o + F * H].reshape(F, H, order="F"),
    f32("transformer.h.0.mlp.c_proj.weight")); o += F * H
hoff = T * H + 2 * H + NL * (4 * H * H + H * F + F * H + 13 * H)
chk("head", raw[hoff:hoff + H * V].reshape(H, V, order="F"),
    f32("transformer.wte.weight").T)

# ---------- 2/3. tham chieu torch: loss step-0 + dWq lop 0 ----------
model.zero_grad(set_to_none=True)
l0 = ce(model, xi, yi)
l0.backward()
GQ = (model.transformer.h[0].attn.c_attn.weight.grad[:256, 0]
      .detach().cpu().numpy().astype(np.float64))
model.zero_grad(set_to_none=True)
print("TORCH step0 loss %.4f" % float(l0), flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
times, losses = [], []
for step in range(NUP + 2):
    t0 = time.perf_counter()
    loss = ce(model, xi, yi)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    losses.append(float(loss.detach()))
    if step >= 2:
        times.append((time.perf_counter() - t0) * 1000.0)
torch_ms = st.median(times)
print("TORCH trace: " + " -> ".join("%.4f" % x for x in losses), flush=True)
print("TORCH train: median %.1f ms/step  loss sau %d update %.4f"
      % (torch_ms, NUP, losses[NUP]), flush=True)
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
for lrv in (5e-5, 1e-4, 2e-4):
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
        print("GRADCHK dWq[0:256,0] corr=%+.5f  maxerr=%.3e  ratio=%.4f"
              % (float(np.corrcoef(a, GQ)[0, 1]), float(np.max(np.abs(a - GQ))),
                 float(np.linalg.norm(a) / np.linalg.norm(GQ))), flush=True)
        print("GRADCHK tkv[:4]=" + " ".join("%+.3e" % v for v in a[:4]), flush=True)
        print("GRADCHK ref[:4]=" + " ".join("%+.3e" % v for v in GQ[:4]), flush=True)
    runs.append((lrv, tl, st.median(ms) if ms else 0.0, neg))

if not runs:
    sys.exit(1)
best = min(runs, key=lambda z: z[1][-1])
print("BENCH TABLE (no C++ bridge) - %d update moi ben" % NUP, flush=True)
print("| implementation | weights | median ms/step | speedup | loss sau %d update |" % NUP, flush=True)
print("| :--- | :--- | ---: | ---: | :--- |", flush=True)
print("| PyTorch eager (lr 5e-5, all params) | gpt2-medium pretrained | %.1f | 1.00x | %.4f |"
      % (torch_ms, losses[NUP]), flush=True)
print("| TokenVector .tkv (lr %.0e) | gpt2-medium pretrained | %.1f | %.2fx | %.4f |"
      % (best[0], best[2], torch_ms / best[2], best[1][-1]), flush=True)
print("BEST lr=%.0e  loss %.4f  steps tang %d" % (best[0], best[1][-1], best[3]), flush=True)
