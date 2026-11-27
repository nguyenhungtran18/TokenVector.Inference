"""Real training benchmark: full pretrained GPT2-medium weights on both sides.

Chay tren Kaggle T4 sau khi clone repo nay:
  python benchmark/export_gpt2m_weights.py
  python benchmark/bench_real_train.py /kaggle/working/TokenVectorInfer
"""
import copy
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
T, NL, H, NH, V = 512, 24, 1024, 16, 50257
MEASURED = 10

pair = np.load("/kaggle/working/gpt2m_ids.npy")
xb_np, yb_np = pair[0], pair[1]
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()
with torch.no_grad():
    xi = torch.from_numpy(xb_np.astype(np.int64)).unsqueeze(0).cuda()
    yi = torch.from_numpy(yb_np.astype(np.int64)).unsqueeze(0).cuda()
    loss_before = float(model(input_ids=xi, labels=yi).loss)
print("HF pretrained initial loss on batch = %.4f" % loss_before, flush=True)

# ---- PyTorch eager training, same weights, same batch ----
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
times, losses = [], []
for step in range(2 + MEASURED):
    t0 = time.perf_counter()
    loss = model(input_ids=xi, labels=yi).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if step >= 2:
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.detach()))
torch_ms = st.median(times)
print("TORCH real-train median %.1f ms/step loss %.4f -> %.4f"
      % (torch_ms, losses[0], losses[-1]), flush=True)
torch_after = losses[-1]
# keep trained model for the final test cell (in-memory only within this process)
del opt
gc.collect(); torch.cuda.empty_cache()

# ---- TokenVector pure .tkv, loads the same pretrained weights from binary ----
cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_real10.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
env = dict(os.environ)
r = subprocess.run([DOTNET, EXE], env=env, capture_output=True, text=True, timeout=1200)
print("TKV rc=", r.returncode, flush=True)
if r.returncode != 0:
    print((r.stdout or "")[-2500:]); print((r.stderr or "")[:800]); sys.exit(1)
ms = [float(l.split()[1]) for l in (r.stdout or "").splitlines() if l.startswith("STEPMS ")]
losses_tkv = [float(l.split()[1]) for l in (r.stdout or "").splitlines() if l.startswith("LOSSF ")]
tkv_ms = st.median(ms[3:]) if len(ms) > 6 else st.median(ms)
print("TKV   real-train median %.1f ms/step loss %.4f -> %.4f"
      % (tkv_ms, losses_tkv[0], losses_tkv[-1]), flush=True)

print("FINAL REAL-TRAIN BENCH", flush=True)
print("| implementation | weights | median ms/step | speedup |", flush=True)
print("| :--- | :--- | ---: | ---: |", flush=True)
print("| torch eager | gpt2-medium pretrained | %.1f | 1.00x |" % torch_ms, flush=True)
print("| TokenVector pure .tkv | gpt2-medium pretrained | %.1f | %.2fx |"
      % (tkv_ms, torch_ms / tkv_ms), flush=True)
print("| C++ bridge (old) | random init | 187.8 | — |", flush=True)

# ---- final test of the trained AI ----
with torch.no_grad():
    p = model.generate(input_ids=xi[:, :8], max_new_tokens=24,
                       do_sample=False, pad_token_id=50256)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
print("FINAL TEST (PyTorch trained model generation):", flush=True)
print(tok.decode(p[0].tolist()), flush=True)
print("FINAL TEST loss: HF before %.4f -> after %.4f ; TKV before %.4f -> after %.4f"
      % (loss_before, torch_after, losses_tkv[0], losses_tkv[-1]), flush=True)
