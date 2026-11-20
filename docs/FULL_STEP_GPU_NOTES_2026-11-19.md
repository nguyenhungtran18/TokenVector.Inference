# Full-step GPU session notes — 2026-11-19

## Standing context
- TokenVector main: `1768eb1` (clean). Inference: `9091d5f` (clean except stray file, see below).
- Notebook: `demo/colab_bench_ai_gpu.ipynb`, cell 3 does `rm -rf` + fresh clone; cell 5 build `.so`; cell 7D = fullstep bench via ctypes; cell 9 = torch GPT2 eager reference.
- Colab T4 16 GB, torch `2.14.1+cu130`, Python 3.13. Bench = random weights (seed 7), T=2048, 3 steps.
- Random weights => loss is NOT comparable to torch ref (ours flat 10.82->10.75 because warmup steps 1-2, lr=0.0025 cosine/1000; torch ref has no warmup). Speed + mem + util only.
- Native (non-ATen) path must stay EXACT — untouched. GPU fullstep is in `D:\TokenVector\Release\tools\simd_native\torch_fused.cpp`.
- Don't commit half-done debug work; verify numerics before claiming speedups; report if session unhealthy.

## What happened this session (2026-11-19)
1. Diagnosed `mat1/mat2` dtype mismatch crash (`Half != Float`) in `TkvFullStep124m`:
   - `.to(Float)` casts on attention recompute path (dQ/dK/dV) and `ctx`; matmuls with fp16 params.
   - Fix: uniform fp16 for all fullstep mm/addmm via helpers `half_mm2` / `half_mat2` (cast no-op when already Half, print `XDBG` only on mismatch), dQ/dK/dV kept Half, removed stray `ctx.to(Float)`.
   - Also fixed silent stale-weights bug: AdamW updated `g_p_all` (fp32 master) but the fp16 `g_p[i]` copies were never refreshed; added fp16 sync loop after `g_p_all.sub_(...)` (D2D cast, no launch to Python).
2. Colab run (`colab_bench_ai_gpu (37).ipynb`): build rc=0, worker rc=0, STDERR clean, no XDBG mismatch lines.
   - STEP 0/1/2 = 709.1 / 246.6 / 247.2 ms; FULLSTEP median **247.2 ms**; util 85%; peak 5843 MiB.
   - Torch ref (cell 9): STEP 0/1/2 = 852.9 / 34.0 / 29.8 ms; median **34.0 ms**; mem 5200 MiB; util 100%.
   - => ~**7.3x slower** than eager torch. Step 0 (~709 ms) includes ~250-460 ms lazy kernel load; median of 3 hides it (same for torch ref: 852.9).
3. GPU bridge per-op history: ~3345 ms/step (loss 17.11). Native CPU fullstep: ~15653 ms/step (loss 25.454). fp32 HEAD before fp16 work: `0cde600`.

## Decision (user approved)
**Next step: autograd backward for the CUDA fullstep** (fwd stays identical; bwd via `loss_t.backward()`; keeps fused ATen LN-bwd / flash-mem-efficient bwd / cuBLAS GEMMs like eager torch). Manual backward stays for CPU path only.

## Autograd rewrite plan (tomorrow)
Edit `D:\TokenVector\Release\tools\simd_native\torch_fused.cpp`, function `TkvFullStep124m`:

1. Add `#include <c10/core/GradMode.h>` if not already transitively included (use `c10::NoGradGuard`).
2. After `const double inv = ...` (just before `/* ---- forward, save activations ---- */`):
   ```cpp
   const bool do_bwd = (run_mode != 1) && g_grad_buf && g_grad_len > 0;
   const bool use_autograd = want_cuda() && do_bwd;
   if (use_autograd)
       for (auto& p : g_p) if (p.defined() && !p.requires_grad()) p.set_requires_grad(true);
   ```
3. Keep `Tensor` for loss:
   ```cpp
   Tensor loss_t = -(lp.gather(1, tgt).squeeze(1)).mean();
   float loss_val = loss_t.item<float>();
   ```
   early return stays: `if (run_mode == 1 || !g_grad_buf) return loss_val;`
4. **Hoist** the `auto put = [&](int64_t idx, const Tensor& grad) {...}` lambda to right after the early return; DELETE the duplicate later at `/* write grads in ps order */`.
5. Replace `/* ---- backward (no autograd) ---- */` region with:
   ```cpp
   if (use_autograd) {
       xin.clear(); a_in.clear(); m_in.clear(); ctx_s.clear();
       u_s.clear(); g1_s.clear(); qkv_s.clear();   // graph keeps what it needs
       lp.reset();
       loss_t.backward();
       for (size_t i = 0; i < g_p.size(); ++i) {
           const Tensor& gr = g_p[i].grad();
           if (gr.defined()) put((int64_t)i, gr);   // order == flat layout (wte,wpe,lnfg,lnfb,12x12)
           else fprintf(stderr, "XDBG autograd: no grad for param %zu\n", i);
       }
       for (auto& p : g_p) if (p.grad().defined()) p.mutable_grad() = Tensor();
   } else {
       /* ... entire existing manual backward, unchanged ... */
   }
   ```
   - Manual branch must remain byte-identical (native CPU path).
   - Everything between `auto P = lp.exp();` and the early-free block before `/* AdamW` goes inside the `else {` ... `}`.
   - The `d_emb`, `g_ln1g...` vectors, `dwte/dwpe/dlnfg/dlnfb` only exist in the else; `put` calls stay inside else.
6. Wrap the **post-AdamW fp16 sync loop** (`g_p[i].copy_(g_p_all.narrow(0, off, n));`) in `c10::NoGradGuard` (or `g_p[i].detach().copy_(...)`) — in-place copy on a leaf with requires_grad=true throws under autograd.
7. `full_ln_fwd` layer_norm calls are autograd-friendly as-is; no changes needed.

## Verify tomorrow
- Balance of braces; `git diff` sanity; build via Colab cell 5 -> cell 7D.
- Expect: build rc=0; step time close to torch ref + fp32 AdamW overhead; if OOM on T=2048 graph, fall back to incremental fused-LN-bwd route instead.
- If slower or errors: revert to manual path by keeping `use_autograd` behind env/flag, or fall back to fp32 HEAD `0cde600`. Do NOT claim speedup without measured median from cell 7D.

## Stray/unrelated files
- `D:\TokenVector.Inference\torch_fused.cpp` — 0 bytes, untracked, created ~21:43 tonight (almost certainly an artifact of an earlier wrong-path shell command of mine). Not content; user can delete after confirming. Not deleted.
- `D:\TokenVector\Release\tools\simd_native\torch_cute.cpp` — unused, untouched.

## Requirements reminders
- Native mode EXACT; native path never touched.
- Numbers must carry timestamp/HEAD; report if session unhealthy; notebook stays fault-tolerant; no commit while debug is half-done.
- Reference numbers today: native CPU 15653 ms/step; per-op bridge ~3345 ms/step; fullstep GPU 247.2 ms (1768eb1, 2026-11-19); torch eager 34.0 ms.

## 2026-11-20 — autograd rewrite iter 1 + debug

- `591deff` (autograd bwd) Colab run `colab_bench_ai_gpu (38).ipynb`: build rc=0, worker rc=0;
  STEP 0/1/2 = 802.6 / 225.9 / 228.0 ms; **FULLSTEP median 228.0 ms** (vs 247.2 manual, torch 32.9);
  mem 6097 MiB (+254 graph overhead); util 87%.
- Loss 10.8203125 / 10.8203125 / 10.78125 — no crash, numerics plausible (random weights + warmup).
- BUG: STDERR `XDBG autograd: no grad for param 0` and `param 1` (wte/wpe). Others 2..147 have grads.
  If left unfixed, AdamW reads garbage from `g_grad_dev` for entries 0/1 (at::empty) — correctness risk.
- `75f7944` added: print after `loss_t.backward()` of `lt.rg`, `g_p[0..2].requires_grad()`,
  `.grad().defined()`, `.grad().numel()`; zero_() fallback in put() for undefined grads (safety only).
- Next: user reruns cell 5 -> 7D. From debug line infer root cause:
  - `rg=0` -> set_requires_grad didn't stick.
  - `rg=1, gdef=0` -> backward didn't accumulate (graph disconnected for index_select path).
  - `p2` has grad -> autograd works for mm/LN leaves; issue isolated to embedding path.
- Perf still ~7x behind eager; after correctness is fixed, revisit (CUDA graphs / lower graph overhead).

### 2026-11-20 root cause found (commit `62b85a5`)
- Debug (notebook 39): STEP 0 `p0/p1(rg=1,gdef=0,n=0)` but p2 OK; STEP 1/2 `gdef=1` with
  n=38597376 (wte) / 1572864 (wpe). So only FIRST step missed embedding grads.
- Cause: `set_requires_grad(true)` was placed AFTER `wte = g_p[0].view(...)` / `wpe = ...`.
  Step 0 built those views while params still had rg=false => graph disconnected for embedding;
  from step 1 on, rg=true persisted from step 0 so views were connected. Other params are viewed
  INSIDE the layer loop (after the set block), hence always had grads.
- Fix: moved `set_requires_grad` block BEFORE creating wte/wpe views. Debug prints + zero_()
  fallback kept for one verification run; remove prints after confirmation.
- Notebook 39 numbers (pre-fix): 686.6 / 230.8 / 230.8 ms median 230.8, mem 6097 MiB, util 78%.
- VERIFIED notebook 40 (`62b85a5`): all 3 steps `p0/p1 gdef=1` (wte 38597376, wpe 1572864),
  no "no grad" lines; loss 10.8203125 / 10.8046875 / 10.75 — matches manual path exactly.
  Timing 958.3 / 265.5 / 224.3 (median 265.5, noisy), mem 6097 MiB, util 88%.
- `87cec59` removed grad debug prints, added phase timing: `XDBG timing fwd/bwd/put/adam/total ms`
  (chrono wall time; notebook uses CUDA_LAUNCH_BLOCKING=1 so this tracks GPU time).
- Notebook 41 timing (steady step 1/2): fwd 49.6/40.5, bwd 110.6/100.5, put 21.3/14.1,
  adam 72.5/78.0, total 254/233 ms; median 254.3; mem 6097 MiB, util 82%.
  Torch ref total 32.9 ms. Biggest costs: bwd ~105 ms, adam ~75 ms.
- `9cf756a` fused AdamW via `g_p_all.addcdiv_(g_m_dev, denom, -lr/bc1)` (removes 3 full-size
  temporaries; decoupled WD) and added `sdpa=` sub-timing inside fwd to identify attention backend cost.
- Next: rerun cell 5 -> 7D; compare adam and sdpa numbers.

### 2026-11-20 BENCH METHODOLOGY BUG (critical)
- Notebook 42 (`9cf756a`): fwd 39.8/39.5 (sdpa only 6.8/6.9!), bwd 98.2/108.6,
  put 15.6/13.8, **adam 47.9/47.9** (was ~75 -> fused addcdiv_ saved ~27ms),
  total 201.5/209.9, median **210.2 ms**; mem **4185 MiB** (was 6097, AdamW temporaries gone),
  util 68%.
- SDPA is NOT the bottleneck (6.8ms of 40ms fwd) — attention backend fine.
- **The "7.3x slower than torch" claim was INVALID**: cell 7D worker sets
  `CUDA_LAUNCH_BLOCKING=1` (real GPU time), but torch ref worker (cell 9) does NOT
  set it and does NOT call `torch.cuda.synchronize()` — its 32.9 ms is kernel
  enqueue time only, not GPU completion. Apples-to-oranges.
- Fixed notebook repo: `b6597e9` added `torch.cuda.synchronize()` after
  `opt.zero_grad()`, before `dt = ...`, in torch ref worker.
- Next: rerun cell 5 -> 7D -> cell 9 with synced torch ref; only then compare.
  Our 210.2 ms is real GPU time; if torch synced is ~200 ms we are at parity.
  Do NOT repeat the "7x" claim until fair numbers exist.
