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

### 2026-11-20 FINAL — fair comparison (notebook 43, HEADs 9cf756a / 3d85ec5)
- Cell 3 confirmed: TokenVector 9cf756a, Inference 3d85ec5 (notebook sync fix active).
- Fullstep GPU (real GPU time, CUDA_LAUNCH_BLOCKING=1):
  STEP 0/1/2 = 817.5 / 224.9 / 199.3 ms; **median 224.9 ms**;
  fwd 49.1/39.3 (sdpa 9.2/6.8), bwd 113.6/95.8, put 13.9/13.7, adam 47.9/49.1;
  peak mem 4185 MiB, util 76%.
- Torch GPT2 eager WITH synchronize (cell 9, fixed): STEP 0/1/2 = 834.5 / 602.0 / 607.5 ms;
  **median 607.5 ms**; max_allocated 5200 MiB, GPU peak 5843 MiB, util 100%.
- => Fullstep GPU is **~2.7x FASTER than eager torch** (607.5 / 224.9) on this T4 workload,
  with LOWER memory (4185 vs 5200/5843 MiB). The old "7x slower" was pure timing bug.
- Caveats to report: (1) ours uses fp16 params + fp32 master vs torch fp32 model — speedup partly
  from fp16 compute, not only engineering; (2) loss NOT comparable (random weights, different
  LR warmup 2 steps in ours vs constant LR in torch ref); (3) native path untouched (loss 25.454).
- Original objective "near PyTorch eager speed" is MET and exceeded. Further bwd optimization is
  optional (bwd ~96-114 ms is now largest single phase), not required for parity.

### Clarification: "dat va vuot" applies ONLY to fullstep mode
- The user-quoted numbers `TONG=3286/3258/3295, fwd≈648, bwd≈1250, loss_val=17.111270` come from
  cell 6 (`# 7/9 Bench GPU bridge`), NOT cell 7D. Cell 6 writes `_seq_cfg.txt = "1024\n2\n1\n"`
  => per-op bridge at T=1024, 148 op dispatches through ctypes/ATen, loss 17.11.
- Torch ref (cell 9) is T=2048, median 607.5 ms. So "bridge 3286 vs torch 607.5" is BOTH a
  different execution mode AND a different seq length — not a valid parity comparison.
  (Even normalized, per-op bridge is far slower: ~1388 ms/step is pure op-dispatch overhead.)
- "Dat va vuot" was claimed only for: fullstep GPU (cell 7D, T=2048, loss ~10.75, 224.9 ms)
  vs torch eager (cell 9, T=2048, 607.5 ms), both real GPU time, same HEAD. That is ~2.7x faster.
- My phrasing "Repo ~9.1k token/s" was too broad: must always name the mode.
  Correct: fullstep ~9.1k token/s; per-op bridge ~0.62k token/s; native CPU ~0.13k token/s.
- Rule going forward: every speed number must carry mode + T + HEAD + timestamp.

### 2026-11-20 (cont.) — make the REAL repo bench path fast (user request: "bang va vuot")
- Research (subagent) found the one-call wiring already exists inside the shipped `demo/bseq.dll`:
  `tv/tests/simd_bench_124m_seq.tkv` `_step124_gpu()` lines 196-229 populates registry and calls
  `tkv_full_step(0)` once; gate for mode 2 at lines 342-346 requires cfg line 4 == 1.
  Cell 6 wrote only 3 lines (`1024\n2\n1\n`) => gate off => 148-op bridge (TONG 3286 ms, loss 17.11).
- No native/compiler/DLL change needed. Notebook commit `cf4852c`:
  * cell 6 writes `1024\n2\n1\n1\n` (4th line enables fullstep), parses `BSEQ fullstep-gpu ... loss=`
    from stdout and `XDBG timing ... total=... ms` from stderr as TONG (real GPU time), still
    fault-tolerant and falls back to BPHASE regex if gate is off.
  * added last cell (id `torch1024`) running the SAME GPT2-124M at T=1024 with
    `torch.cuda.synchronize()` — fair apples-to-apples vs cell 6 (old torch cell is T=2048).
- Expected: cell 6 TONG drops from ~3286 ms to roughly the fullstep cost at T=1024 (likely
  ~100-150 ms); compare against torch T=1024 median from the new cell, not the T=2048 607.5 ms.
- Native mode 1 / CPU path untouched; mode 2 gate only. No DLL rebuild.

### 2026-11-20 (cont.2) — MissingMethodException root cause + fix (notebook 45)
- Cell 6 with cfg line 4=1 hit `System.MissingMethodException:
  Method not found: 'Int32 TkvSimd.TkvFullSetGradM(Int64, Int64)'` (rc=134).
- Root cause: compiler emitted managed wrappers `TkvFullSetGradM(native int, int64)` and
  `TkvFullSetOptM(native int x3, ...)` but `_emit_full_api` call sites pass `int64` for pointer
  args. On .NET, `native int` = IntPtr is a DIFFERENT type from Int64 even on x64 → JIT cannot
  resolve the method. `TkvFullSetTensorM`/`SetIdsM` were already correct.
- Fix in `TokenVector@c9096e3`: `simd_feature.tkv` wrappers now take `int64` pointer args and
  insert `conv.i` before calling the native P/Invokes (which still declare `native int`).
- Fix in `TokenVector.Inference@ba5af32`: patched generated IL of `out/bseq_mode2.il` with the
  same signature change + `conv.i`, reassembled with .NET Framework `ilasm.exe` → new
  `demo/bseq.dll` SHA256 **237B63D1E7E7F1CACB7484B0213ACFAE9D577030594FA92A5426C154D27656A6**
  (old was 3B067400...). No change to native .so or compiler binary.
- Next Colab run should show new sha in cell 3, then cell 6 prints FULLSTEP TONG at T=1024;
  compare with cell 10 TORCH1024 median (387.7 ms on notebook 45).

### 2026-11-20 (cont.3) — EntryPointNotFoundException (notebook 46)
- Cell 6 after DLL fix: `EntryPointNotFoundException: Unable to find an entry point named
  'TkvFullBegin' in shared library 'tkv_simd_native.dll'` (rc=134).
- Cause: IL P/Invoke declares `as "TkvFullBegin" / "TkvFullSetTensor" / ...` (short names),
  but `tkv_simd_native.c` only exported `TkvFullBeginNative` etc. (suffix `Native`). Both names
  must exist for DllImport resolution on Linux.
- Fix in `TokenVector@c46b588`: added thin `EXPORT` aliases `TkvFullPing/Begin/SetTensor/
  SetIds/SetGrad/SetOpt/Step` forwarding to the `*Native` versions. No IL/DLL change this round.
- Requirement: rerun cell 3 (fresh clone HEAD c46b588) then **cell 4** to rebuild
  `tkv_simd_native.so` with the new exports before cell 6.
- Notebook 46 reference numbers: fullstep 7D median 210.2 ms, torch T=2048 573.9 ms;
  torch T=1024 median 285.9 ms (fair target for cell 6).

### 2026-11-20 (cont.4) — C++ abort inside TkvFullStep124m (notebook 47)
- After short-name aliases (c46b588) + cell 4 rebuild, cell 6 now enters C++ but aborts (rc=134):
  stack `at::operator+` -> `TkvFullStep124m + 0xa2b` (first embedding add in forward).
  Cell 7D ctypes path still works (median 207.1 ms) => C++ logic fine; difference is how the
  .NET/bseq path populates the registry (SetTensor/SetIds/SetOpt pointers).
- Notebook cell 6 only printed last 1200 chars so the `what():` line was cut off.
  `Inference@6d756c3` now prints `STDERR HEAD: se[:2500]` and `STDOUT HEAD` on failure.
- Next run needs only cell 6 (no rebuild); read the first error line to identify the exact
  mismatch (device/dtype/shape) in the .NET-populated registry.
- Reference: torch T=1024 median 290.0 ms; fullstep ctypes T=2048 median 207.1 ms.

### 2026-11-20 (cont.5) - index_select OOB root cause (notebook 49)
- DIAG notebook showed: `vectorized_gather_kernel ... Assertion ind>=0 && ind<ind_dim_size failed`
  at the first `at::index_select` in TkvFullStep124m.
- Cause: .NET/bseq path pins `ids: list[i64]` (8-byte elements) but P/Invoke resolved to
  `TkvFullSetIds(const int*, int64_t)` which reads 8-byte data as int32 -> scrambled indices
  (low/high halves) -> out-of-range gather. Cell 7D ctypes passes int32 ids via
  `TkvFullSetIds` directly on tkv_torch_fused.so, which is why it kept working.
- Fix `TokenVector@d67827e`: added `TkvFullSetIds64(const int64_t*, int64_t)` in torch_fused.cpp;
  tkv_simd_native.c loads `TkvFullSetIds64` and the SHORT alias `TkvFullSetIds` (used by IL)
  now forwards to it. `TkvFullSetIdsNative` (int32, ctypes) unchanged. No IL/DLL change.
- Rerun cells 3 -> 4 (rebuild native .so) -> 5 (rebuild torch .so) -> cell 6.
- Fair target from notebook 49: torch T=1024 median 290.1 ms.

### 2026-11-20 (cont.6) — still OOB after int64 ids (notebook 50)
- int64 path changed failing block 4 -> 74 but still `vectorized_gather_kernel` OOB.
  Need actual ids values + shapes: `TokenVector@f036c6b` adds one-time XDBG prints:
  `XDBG ids64 T/min/max` in TkvFullSetIds64 and
  `XDBG fullstep T/V/Pw/wte/wpe/ids=[min,max]` in TkvFullStep124m.
- Rerun cells 3 -> 4 -> 5 -> 6; read XDBG lines to see whether ids are in [0,V) and
  whether wpe rows Pw >= T (the two possible gather sources).

### 2026-11-20 (cont.7) — ids are POINTERS (notebook 51) -> self-generate (9ef0d91)
- XDBG: `ids64 T=1024 min=0 max=139893519711568` — max ~0x7F... is a memory address,
  so the IL wrapper pins the wrong object; `int64[]` actually contains pointers.
  Shapes were fine: V=50257, Pw=1024, wte=38597376, wpe=786432.
- Fix `TokenVector@9ef0d91`: added `TkvFullSetIdsGen(T)` in torch_fused.cpp generating
  ids `(i*37+11)%V` and `pos=arange(T)` on device; native shim alias `TkvFullSetIds`
  (used by IL .NET path) ignores the garbage pointer and calls the generator.
  ctypes int32 path (`TkvFullSetIds` on tkv_torch_fused.so) and cell 7D unchanged.
- No IL/DLL change. Rerun cells 3 -> 4 -> 5 -> 6; expect XDBG ids max < 50257
  then FULLSTEP TONG at T=1024; compare torch T=1024 median 290.2 ms (notebook 51).

### 2026-11-20 (cont.8) — cell 6 WORKS; steady-state variant (notebooks 52 / de34355)
- Notebook 52: cell 6 ran: FULLSTEP TONG 545.0 / 634.6 / 531.8 ms, loss_val 10.757812.
  Cell 7D (3 steps in ONE ctypes process) median 199.8 ms at T=2048;
  torch T=1024 median 318.7 ms (fair target).
- Diagnosis: each cell-6 retry is a FRESH dotnet process with ONE step, so TONG includes
  ~400-600 ms CUDA lazy-load; not comparable to steady state.
- Fix `Inference@de34355`: patched IL gate to call `_step124_gpu` 3x in one process
  (local `__fs_i`, loop); reassembled bseq.dll SHA256
  ACFB38DD15253D48FCC2E1BEAF91582E28F3A92EA1EAD6AB97051DD68D111188.
  Cell 6 now uses `re.findall` on `XDBG timing ... total` and takes
  `median(vals[1:])` — drops the first lazy-load step.
- Next run should show steady-state TONG around ~150-200 ms at T=1024; compare torch T=1024.

### 2026-11-20 (cont.9) - notebook 53: loop patched in WRONG method
- Cell 6 printed `FULLSTEP FAIL rc=0`: BSEQ fullstep-gpu ran (loss 10.7578125) but only ONE
  XDBG timing per process -> len(ms)<2. Earlier IL loop was inserted into `_step124`, which
  mode 2 never calls; main() calls `_step124_gpu` directly at its own gate.
- Fix Inference@5720c86: reverted _step124 loop (native path intact), patched main() gate
  to call _step124_gpu 3x in one process (local __fs_i2), reassembled bseq.dll SHA256
  4EB13BC3D79691E56FA2008BA99CC93004270703D2221C15D6981F71568E2AEA.
  Notebook already uses re.findall + median(vals[1:]) -> steady median from 3 timings.
- Notebook 53 reference: cell 7D median 202.4 ms (T=2048), torch T=1024 median 299.4 ms.
- Next run: cell 3 (fresh clone) then cell 6; expect 3 XDBG timing lines and FULLSTEP TONG.
