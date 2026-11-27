"""Real training benchmark + final test on held-out Shakespeare text.

Collateral:
  export_gpt2m_weights.py  -> gpt2m_weights.bin + gpt2m_ids.npz
  ilgpt2_real10.exe        -> pure .tkv trainer with pretrained weights

Chay: python benchmark/bench_real_train.py /kaggle/working/TokenVectorInfer
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
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

BASE = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/TokenVectorInfer"
EXE = BASE + "/cuda-backend/Release/cuda_backend/examples/ilgpt2_real10.exe"
DOTNET = "/kaggle/working/dotnet/dotnet"
NL, H, NH, V, T = 24, 1024, 16, 50257, 512

d = np.load("/kaggle/working/gpt2m_ids.npz")
xb, yb, xt, yt = (d[k] for k in ("xb", "yb", "xt", "yt"))
cfg = GPT2Config(n_layer=NL, n_head=NH, n_embd=H, n_positions=1024,
                 vocab_size=V, layer_norm_epsilon=1e-6,
                 resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium", config=cfg).to("cuda").train()
xi = torch.from_numpy(xb.astype(np.int64)).unsqueeze(0).cuda()
yi = torch.from_numpy(yb.astype(np.int64)).unsqueeze(0).cuda()
xe = torch.from_numpy(xt.astype(np.int64)).unsqueeze(0).cuda()
ye = torch.from_numpy(yt.astype(np.int64)).unsqueeze(0).cuda()
with torch.no_grad():
    hf_before_train = float(model(input_ids=xi, labels=yi).loss)
    hf_before_test = float(model(input_ids=xe, labels=ye).loss)
print("HF pretrained loss: train %.4f  held-out %.4f" % (hf_before_train, hf_before_test), flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=0.0)
times, train_losses = [], []
for step in range(2 + 10):
    t0 = time.perf_counter()
    loss = model(input_ids=xi, labels=yi).loss
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if step >= 2:
        times.append((time.perf_counter() - t0) * 1000.0)
        train_losses.append(float(loss.detach()))
torch_ms = st.median(times)
model.eval()
with torch.no_grad():
    hf_after_test = float(model(input_ids=xe, labels=ye).loss)
model.train()
print("TORCH train median %.1f ms/step  train %.4f -> %.4f  held-out %.4f -> %.4f"
      % (torch_ms, train_losses[0], train_losses[-1], hf_before_test, hf_after_test), flush=True)

tok = AutoTokenizer.from_pretrained("gpt2")
prompt = "ROMEO: "
with torch.no_grad():
    gen = model.generate(input_ids=tok(prompt, return_tensors="pt").input_ids.cuda(),
                         max_new_tokens=28, do_sample=False, pad_token_id=50256)
generated = tok.decode(gen[0].tolist())
del opt
gc.collect(); torch.cuda.empty_cache()

# ---- TokenVector pure .tkv, same pretrained weights ----
cfg_rt = {"runtimeOptions": {"tfm": "net10.0",
          "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}}}
with open(os.path.join(os.path.dirname(EXE), "ilgpt2_real10.runtimeconfig.json"), "w") as f:
    json.dump(cfg_rt, f)
r = subprocess.run([DOTNET, EXE], env=dict(os.environ), capture_output=True, text=True, timeout=1200)
print("TKV rc=", r.returncode, flush=True)
if r.returncode != 0:
    print((r.stdout or "")[-2500:]); print((r.stderr or "")[:800]); sys.exit(1)
lines = (r.stdout or "").splitlines()
ms = [float(l.split()[1]) for l in lines if l.startswith("STEPMS ")]
tkv_losses = [float(l.split()[1]) for l in lines if l.startswith("LOSSF ")]
tkv_bytes = [l for l in lines if l.startswith("WEIGHT BYTES")]
tkv_ms = st.median(ms[3:]) if len(ms) > 6 else st.median(ms)
print("TKV   train median %.1f ms/step  loss %.4f -> %.4f  %s"
      % (tkv_ms, tkv_losses[0], tkv_losses[-1], tkv_bytes[0] if tkv_bytes else ""), flush=True)

print("FINAL REAL-TRAIN BENCH", flush=True)
print("| implementation | weights | median ms/step | speedup | train loss |", flush=True)
print("| :--- | :--- | ---: | ---: | :--- |", flush=True)
print("| torch eager | gpt2-medium pretrained | %.1f | 1.00x | %.4f -> %.4f |"
      % (torch_ms, train_losses[0], train_losses[-1]), flush=True)
print("| TokenVector pure .tkv | gpt2-medium pretrained | %.1f | %.2fx | %.4f -> %.4f |"
      % (tkv_ms, torch_ms / tkv_ms, tkv_losses[0], tkv_losses[-1]), flush=True)
print("| C++ bridge (old, random init) | — | 187.8 | — | — |", flush=True)

print("FINAL TEST - trained AI", flush=True)
print("Corpus: tinyshakespeare (4096 tokens); train = first 512, held-out = next 512.", flush=True)
print("Held-out CE loss: before %.4f -> after %.4f (PyTorch)" % (hf_before_test, hf_after_test), flush=True)
print("Prompt '%s' -> %s" % (prompt, generated.replace("\n", " | ")), flush=True)
print("TKV train loss %.4f -> %.4f (should drop, not stay at ln(50257)=10.8249)"
      % (tkv_losses[0], tkv_losses[-1]), flush=True)
