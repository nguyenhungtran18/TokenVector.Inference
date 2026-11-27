"""Final head-to-head: PyTorch eager vs pure-TokenVector GPT2-medium train step.

Chay tren Kaggle T4, trong cung session:
    python benchmark/bench_final_torch_tkv.py /kaggle/working/TokenVectorInfer

Khong torch trong duong chay TokenVector; torch chi la doi chieu.
In ra bang median(ms)/step, toc do nhan va loss cua moi ben.
"""
import json
import os
import statistics as st
import subprocess
import sys
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

BASE = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/TokenVectorInfer"
DOTNET = "/kaggle/working/dotnet/dotnet"
EXE = BASE + "/cuda-backend/Release/cuda_backend/examples/ilgpt2_long.exe"
T, V, NL, H, NH = 512, 50257, 24, 1024, 16
MEASURED = 10

print("BENCH GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)

# ---- PyTorch eager reference: same config, same T, same optimizer ----
torch.manual_seed(7)
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = GPT2LMHeadModel(cfg).to("cuda").train()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
xb = torch.randint(0, V, (1, T), device="cuda")
yb = torch.randint(0, V, (1, T), device="cuda")
torch_times, torch_losses = [], []
for step in range(2 + MEASURED):
    t0 = time.perf_counter()
    loss = model(input_ids=xb, labels=yb).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if step >= 2:
        torch_times.append((time.perf_counter() - t0) * 1000.0)
        torch_losses.append(float(loss))
torch_ms = st.median(torch_times)
print("TORCH median %.1f ms/step  loss %.4f -> %.4f"
      % (torch_ms, torch_losses[0], torch_losses[-1]), flush=True)
del model, opt, xb, yb
import gc
gc.collect()
torch.cuda.empty_cache()

# ---- TokenVector final build: 10 steps, warm-up in steps 1..2 ----
cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_long.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
env = dict(os.environ)
r = subprocess.run([DOTNET, EXE], env=env, capture_output=True, text=True, timeout=1200)
print("TKV rc=", r.returncode, flush=True)
if r.returncode != 0:
    print((r.stderr or "")[:800])
    print((r.stdout or "")[-1500:])
    sys.exit(1)
ms = [float(l.split()[1]) for l in (r.stdout or "").splitlines()
      if l.startswith("STEPMS ")]
losses = [float(l.split()[1]) for l in (r.stdout or "").splitlines()
          if l.startswith("LOSSF ")]
steady = ms[3:] if len(ms) > 6 else ms
tkv_ms = st.median(steady)
print("TKV   median %.1f ms/step (steps 4..%d)  loss %.4f -> %.4f"
      % (tkv_ms, len(ms), losses[0], losses[-1]), flush=True)

print("FINAL BENCH TABLE", flush=True)
print("| implementation | median ms/step | vs torch |", flush=True)
print("| :--- | ---: | ---: |", flush=True)
print("| torch eager (fp32, AdamW) | %.1f | 1.00x |" % torch_ms, flush=True)
print("| TokenVector pure .tkv (final) | %.1f | %.2fx |"
      % (tkv_ms, torch_ms / tkv_ms), flush=True)
print("| C++ bridge (old baseline) | 187.8 | 2.36x |", flush=True)
