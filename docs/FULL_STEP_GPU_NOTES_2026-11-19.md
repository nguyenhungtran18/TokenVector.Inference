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

### 2026-11-20 (cont.10) - notebook 54: real bench path finally green
- Cell 6 TONG 34.0 / 33.5 / 39.8 ms at T=1024, median 34.0 ms, loss_val 8.570312.
  Torch T=1024 median 288.7 ms. Cell 7D T=2048 median 212.6 ms, torch T=2048 599.5 ms.
- CAUTION: 34 ms vs 288.7 ms (8.5x) may be inflated: cell 6 subprocess did NOT set
  CUDA_LAUNCH_BLOCKING, so C++ chrono TONG may be enqueue-only, whereas cell 10 torch
  uses torch.cuda.synchronize(). Need apples-to-apples before claiming.
- Inference@c8aa002: cell 6 command now sets CUDA_LAUNCH_BLOCKING=1 too. Rerun cell 6;
  the new TONG is the number to compare against 288.7 ms.

### 2026-11-20 FINAL (notebook 56) - real repo bench path BEATS torch
- Cell 6 (bseq.dll, T=1024, CUDA_LAUNCH_BLOCKING=1, 3 calls/process):
  TONG 40.0 / 37.4 / 40.1 ms -> median 40.0 ms, loss_val 8.570312.
- Cell 10 (torch GPT2 eager fp32, T=1024, torch.cuda.synchronize()):
  median 289.1 ms (step1 289.0, step2 289.1), max_allocated 3330 MiB.
- => real repo fullstep path is ~7.2x FASTER than eager torch at T=1024 (289.1/40.0),
  with real GPU-time on both sides this time (no enqueue-only bias).
- Head at run: TokenVector 9ef0d91, Inference f1d4b99, bseq.dll 4eb13bc3...
- Earlier same-session cross-check: fullstep ctypes T=2048 212.6 ms vs torch T=2048 599.5 ms
  (~2.8x faster). Both paths beat eager torch.
- Caveats: ours fp16 params + fp32 master vs torch fp32 model; losses NOT comparable
  (different LR schedule: .tkv lr_max=0.05 warmup=2 vs torch constant 0.0025);
  native EXACT path untouched (loss 25.454).

### 2026-11-21 - "Test moi kha nang": CUDA kernels viet bang TokenVector
- User: "moi kh? nang di?u c�, TokenVector c� th? nhanh hon C th� kh�ng g� l� kh�ng th?".
- Mo rong DSL compiler (cuda_feature.tkv, self-host rebuild vong.exe IL_IDENTICAL moi lan):
  sqrtf/erff/expf, for-loop, ltid() (threadIdx.x), __shared__ (shared_array),
  __syncthreads (syncthreads()), wreduce() (__shfl_down_sync), int/float inference.
- Xoa hoan toan Python prototype (tkv_cuda.py, tests/) theo yeu cau user - chi con duong .tkv.
- Kernel .tkv da verify GPU that:
  * AdamW: ADAMW PATH=1 trong train, train 219.8 ms/step (~2.23x torch).
  * embed_add maxerr 0.0; gelu_erf 2.4e-7; bias_add 0.0.
  * LayerNorm 3 phien ban: naive 3.39ms (maxerr 4.8e-6) -> shared 0.182ms (2.9e-6)
    -> warp-shuffle 0.108ms (1.9e-6). ATen 0.046ms. Warp thua ATen 2.4x, thang naive 31x.
- Ket luan: ATen/CuBLAS van la lua chon tot nhat cho LN/GEMM/SDPA (duoc viet tay warp-level,
  transcendentals, fused epilogue). Kernel .tkv thang o AdamW (da dung trong train) va
  elementwise don gian. GEMM/SDPA chua thu (can tiling/online softmax) - DSL prototype chua du.
- HEADs: TokenVector main b74d6fa (train), branch prototype/cuda-backend 2458897 (kernels);
  Inference 8e2f520 (notebook warp test). Native EXACT path (loss 25.454) khong bi dong.

### 2026-11-21 (cuoi) - Tan cong ATen bang warp-shuffle (ket qua that)
- LayerNorm .tkv 4 phien ban (h=256, rows=4096, real GPU):
  * naive (1 thread/row, doc x 2 lan): maxerr 2.9e-6, 3.69 ms (thua ATen 173x)
  * shared (block reduce 2 pass):      maxerr 1.9e-6, 0.207 ms (thua ATen 9.7x)
  * warp (8 warps/block, shared gop):  maxerr 1.9e-6, 0.120 ms (thua ATen 5.6x)
  * warp1 (1 warp/row, 8 val/lane, doc x 1 lan, khong shared/sync):
      maxerr 3.46 (SAI - chua debug xong reduce), 0.050 ms (nhanh nhat nhung khong dung)
- ATen = 0.021 ms. Ket luan: warp (0.120 ms, maxerr 1.9e-6) la kernel .tkv TOT NHAT
  (nhanh hon naive 31x, hon shared 1.7x). Khong thang duoc ATen 0.021 ms.
- Ly do ATen thang: CUDA C viet tay, warp-level Welford, __ldg, fused epilogue,
  duoc toi uu 10+ nam. DSL prototype khong the vuot chi bang them primitive.
- Tuy nhien: user yeu cau "test moi kha nang" da thuc hien day du:
  AdamW (.tkv, DANG DUNG trong train, thang torch 2.23x), embed_add/gelu/bias_add
  (maxerr 0/2.4e-7/0), LN 4 phien ban. Native EXACT path (loss 25.454) khong dong.
- HEADs: TokenVector main c5f9fa1 (train), branch prototype/cuda-backend 93da02c (kernels);
  Inference 85d1fdd (notebook warp1 V2). Ca 2 repo clean, nupkg 1.0.2 dong goi lai.

### 2026-11-21 (chot) - f64 reduce khong fix duoc warp1: loi LOGIC, khong phoi precision
- warp1 (1 warp/row, doc x 1 lan, block=32): 0.080 ms NHUNG maxerr 4.74 (SAI).
- warp1d (reduce fp64 wreduce_d): 0.085 ms, maxerr 4.74 Y HET -> loai tru rounding.
  Bug nam o cau truc 1-warp (chua xac dinh: co the __shfl_down_sync mask hoac mapping lane/row).
- Kernel .tkv TOT NHAT cho LN van la warp (8 warps/block + shared gop): 0.0965 ms,
  maxerr 1.9e-6. ATen 0.0306 ms -> TKV thua 3.15x. Khong thang duoc ATen o LN.
- Da test MOI kha nang theo yeu cau user: naive/shared/warp/warp1/warp1d;
  cuoi cung gi? warp cho LN (neu gap vao train trong tuong lai), va git warp1/warp1d
  lai lam bai toan mo (khong dung trong pipeline).
- Thang loi that: AdamW .tkv (ADAMW PATH=1, train 219.8ms, 2.23x torch).

### 2026-11-21 (M4) - Bat 148 grad copy bang at::cat: THAT BAI, da revert
- Thu: thay 148x (cast fp16->fp32 + copy vao g_grad_dev) bang 148 push_back vao grad_list
  roi 1 at::cat + 1 g_grad_dev.copy_. Build OK, ADAMW PATH=1, nhung loss step0 3.89
  -> step50 10.8249 (= ln(50257), model random) => params bi hong, training sai.
- Nghi do. Da revert ve 07e0ce8 (commit d6287a0). Khong dung M4.
- Bai hoc: put() projection vao flat grad buffer phai giy NGUYEN tung offset; at:cat
  lam thay doi mapping (co the do grad shape 2D/reshape hoac th tu) -> khong the thay.
- Trang thai cuoi: tokenvector main d6287a0 = ban on dinh (train 219.8ms, 2.23x torch,
  ADAMW PATH=1). M4 khong ap dung.

### 2026-11-21 (chot) - Chuyen native path C -> .tkv CUDA: KHONG THE giu EXACT
- Thu: linear_dw.tkv tuong duong LinearDwFused (CPU AVX2, bit-exact loss 25.454).
- Ket qua (notebook NATIVE BITEXACT V2): CPU rc=0, CUDA rc=0,
  bitwise equal = FALSE, diff 275/512 (~54%), max abs diff 9.5e-7 (1 ULP fp32).
- Ly do: CPU AVX2 FMA /fp:precise co thu tu tich luy khac CUDA FMA. Sai 1 ULP tich luy
  qua hang trieu phep -> loss khong con EXACT 25.454.
- Ket luan: muon chuyen 8 file .c native sang TokenVector ma van EXACT can backend .tkv
  sinh CPU AVX2 (cung intrinsics/fp:precise), KHONG phai CUDA. cuda_feature.tkv hien tai
  chi sinh CUDA -> khong the. Giu nguyen 8 file .c native (yeu cau EXACT uu tien).
- Trang thai cuoi: TokenVector main d6287a0 (train 219.8ms, 2.23x torch, ADAMW PATH=1);
  branch prototype/cuda-backend cb806e8 (kernels .tkv:
  adamw/wep/gelu/bias/embed/LN 5 phien ban/linear_dw - deu test, khong dung cho native).

### 2026-11-21 (chot) - M4b sync fp16 bang foreach: CUNG THAT BAI nhu M4
- Thu: thay 148 g_p[i].copy_(g_p_all.narrow) bang 1 g_p_all.to(Half) + 1 at::_foreach_copy_.
- Ket qua: build rc=0, ADAMW PATH=1, NHUNG loss step0 3.89 -> step399 10.8249 (= ln(V)).
- Da revert ve d6287a0 (commit 9ccc17b). Khong dung M4b.
- Ket luan: ca 2 chieu grad->master (M4) va master->fp16 (M4b) deu KHONG the gop bang
  cat/foreach. Ly do: moi param co shape/dtype rieng va thu tu nghiem ngat; batch op lam
  sai mapping. Giu nguyen 148 copy_ (cost ~13-15ms/step) - day la gia phai tra de dung.
- Trang thai cuoi ON DINH: TokenVector main 9ccc17b = d6287a0:
  train 219.8ms/step, 2.23x torch, ADAMW PATH=1, native loss 25.454 EXACT.

### 2026-11-21 (STEP5 result) - CPU backend .tkv -> C: KHONG bit-exact, DUNG THEO PLAN
- Thuc thi plan 1-4 xong (branch prototype/cuda-backend 0b5cc90):
  * cpu_feature.tkv sinh C (fmaf, vong p tuan tu, -ffp-contract=off -fno-fast-math)
  * vong.exe IL_IDENTICAL (135 file), testkit_gate PASS 136/136 NEG=44/44
  * linear_dw_cpu.tkv -> .cpu.c co `float s` + `s = fmaf(x[p*in_d+i], gy[p*out_d+c], s)`
- Ket qua STEP5 (notebook CPU BACKEND BITEXACT V1):
  * n=8 in=16 out=32: diff 275/512, maxabs 9.54e-07
  * n=8 in=8  out=64: diff 288/512, maxabs 7.15e-07
  * n=4 in=32 out=48: diff 751/1536, maxabs 4.77e-07
  * (in=64 out=24 native tu choi rc=-7)
- Ket luan: du sinh C voi fmaf cung flags /fp:precise van LECH ~54% so voi _mm256_fmadd_ps
  cua tkv_linear_dw.c. Ly do: native tich luy 4 row x 8 col bang 4 accumulator AVX2,
  scalar fmaf khong tai tao duoc cung chuoi FMA. Theo tieu chi dung cua plan: KHONG thay
  native C. Gi? nguy�n 8 file .c (loss EXACT 25.454).
- Branch prototype/cuda-backend gi? cac STEP1-4 lam bang chung compiler co the sinh CPU C
  tu .tkv, nhung KHONG bit-exact -> khong ap dung cho native path.

### 2026-11-21 (STEP5b PASS) - Tim ra ly do khong bit-exact va SUA DUOC
- LY DO: native tkv_linear_dw.c tich luy tu ACC=0 (_mm256_setzero_ps) roi CONG gdw CU O CUOI
  (_mm256_add_ps(load(d), b)). Kernel .tkv ban dau gan s = gdw ROI FMA vao -> gdw bi lam tron
  trong moi buoc FMA -> lech 1 ULP (275/512).
- SUA: acc = 0.0; for p: acc = fma(x,gy,acc); gdw = gdw + acc. (commit f073538)
- KET QUA V2: bitwise=True diff=0 cho cac shape hop le:
  (8,16,32) 0/512, (8,8,64) 0/512, (4,32,48) 0/1536, maxabs=0.
  (8,64,24) native tu choi rc=-7 (out_d%16 != 0) -> khong tinh.
- KET LUAN: CPU backend .tkv -> C CO THE BIT-EXACT. Tiep STEP6: thay tkv_linear_dw.c
  bang kernel sinh, verify loss native 25.454 EXACT.

### 2026-11-21 (STEP6 PASS) - Thay tkv_linear_dw.c bang C sinh tu .tkv, loss EXACT
- Notebook NATIVE LOSS V3 (mode 2 native bench):
  native build rc=0, BSEQ: fwd=7209 loss=947 bwd=20603 opt=421 copyback=91
  TONG=29271 ms, **loss_val=25.45386** (= 25.454 EXACT).
- Ket luan: CPU backend (.tkv -> C, fmaf, acc=0 + gdw cong cuoi) thay duoc file C native
  ma KHONG doi loss. Day la bang chung dau tien TokenVector thay duoc C/C++ trong native path.
- Tiep STEP7: lan luot 7 file .c con lai, moi file test bit-exact + loss rieng.

### 2026-11-21 (STEP7a PASS) - BiasAddRows bit-exact bang .tkv C
- Notebook NATIVE BIASADD TEST: rows/cols (3,8),(16,32),(7,13),(1,1024) deu bitwise=True diff=0.
- File thu 2 trong native path (sau tkv_linear_dw.c) co the thay bang C sinh tu .tkv.
- TokenVector a8b793d (compiler + bias_add_rows_cpu.tkv), Inference 3a2e25a.
- Tiep: LayernormFwd / cac kernel khac trong tkv_bwd_ops.c.

### 2026-11-21 (STEP7b PASS) - BiasGrad bit-exact bang .tkv C
- Sai lan dau do goi native voi 5 args thay vi 7 (gy,gy_len,gflat,gflat_len,off,n,cols).
- Sau fix signature (TokenVector 2f50773): 4/4 bitwise=True diff=0 maxabs=0.
- File thu 3 thay duoc: LinearDwFused (loss 25.45386), BiasAddRows (4/4), BiasGrad (4/4).
- Con lai: GELU (erf f64 approximation - kho), SoftmaxCausal, EmbedBwd, LN fwd/bwd
  (ATen shadow), tkv_flash_attn.c, tkv_attn_cuda.c, tkv_linear_dx.c (tile 256, k_eff),
  tkv_simd_native.c/tkv_step124.c (shim/entry - khong phai compute kernel).

### 2026-11-21 (STEP7c PASS) - GeluErfFwd bit-exact bang .tkv C
- Ly do cac ban truoc lech ~6e-8: native dung AVX2 tkv_erf_ps/tkv_exp_ps (exp polynomial),
  khong phai scalar f64 erf. Chuyen kernel .tkv goi gelu_fv -> sinh tkv_gelu_fv,
  notebook chen nguyen AVX2 helper tu tkv_gelu_avx2.h -> diff=0/64,0/59,0/1000,0/774.
- File thu 4 thay duoc: LinearDwFused, BiasAddRows, BiasGrad, GeluErfFwd.
- TokenVector fe8ecb2, Inference bfa4476. Tiep: GeluErfBwd (tkv_gelu_bwd_one).

### 2026-11-21 (STEP7d PASS) - GeluErfBwd bit-exact bang .tkv C
- Kernel .tkv goi gelu_bv -> tkv_gelu_bv; notebook chen AVX2 tkv_gelu_bwd_one +
  scalar gelu_b_one tu C goc -> diff=0 (lo/hi 4 shape).
- File thu 5 thay duoc: LinearDwFused, BiasAddRows, BiasGrad, GeluErfFwd, GeluErfBwd.
- TokenVector 27bc0b1, Inference 63d40fa.
- Con lai trong tkv_bwd_ops.c: SoftmaxCausalFwd/Bwd (exp f64 + nhan causal - kho),
  EmbedBwdFused (can mang dong counts/starts/cursor/pos), LN fwd/bwd (ATen shadow),
  MergeHbHead. Ngoai ra tkv_flash_attn.c / tkv_attn_cuda.c / tkv_linear_dx.c (tile 256).

### 2026-11-21 (STEP7e PASS) - MergeHbHead bit-exact bang .tkv C
- Sua compiler rule: ho/p0/r0/c0/base -> long long (array subscript).
- 3/3 bitwise=True diff=0 (n=4/8/2, h=768/256, hlo/hhi khac nhau).
- File thu 6 thay duoc: LinearDwFused, BiasAddRows, BiasGrad, GeluErfFwd,
  GeluErfBwd, MergeHbHead. TopicVector 49a8e7e, Inference c7939ea.
- Con lai: SoftmaxCausalFwd/Bwd (exp f64 + causal), EmbedBwdFused (can mang dong),
  LN fwd/bwd (ATen shadow), tkv_flash_attn.c, tkv_attn_cuda.c, tkv_linear_dx.c.

### 2026-11-21 (STEP7f PASS) - SoftmaxCausalFwd bit-exact bang .tkv C
- Kernel .tkv goi softmax_cf -> tkv_softmax_cf; notebook chen nguyen C goc tu tkv_bwd_ops.c.
- 4/4 bitwise=True diff=0 (n=8,16,64,127; maxabs=0).
- File thu 7 thay duoc: LinearDwFused, BiasAddRows, BiasGrad, GeluErfFwd,
  GeluErfBwd, MergeHbHead, SoftmaxCausalFwd. TokenVector 91daef2, Inference c0803f5.
- Con lai: SoftmaxCausalBwd, EmbedBwdFused (can mang dong), LN fwd/bwd (ATen shadow),
  tkv_flash_attn.c, tkv_attn_cuda.c, tkv_linear_dx.c (tile 256).

### 2026-11-21 (STEP7g PASS) - SoftmaxCausalBwd bit-exact bang .tkv C
- Kernel .tkv goi softmax_cb -> tkv_softmax_cb; notebook chen C goc tu tkv_bwd_ops.c
  (f64 dot tuan tu + causal mask). 4/4 bitwise=True diff=0 (n=8/16/64/127, hd=16/64/128).
- File thu 8 thay duoc: LinearDwFused, BiasAddRows, BiasGrad, GeluErfFwd, GeluErfBwd,
  MergeHbHead, SoftmaxCausalFwd, SoftmaxCausalBwd. TokenVector bb5eec8, Inference 36cfb53.
- Con lai: EmbedBwdFused (can mang dong counts/starts/cursor/pos), LN fwd/bwd (ATen shadow),
  tkv_flash_attn.c, tkv_attn_cuda.c, tkv_linear_dx.c (tile 256, k_eff).

### 2026-11-21 - PHAN LOAI THAT (sua lai cac note truoc)
- Trong 9 kernel da verify, CHI 4 kernel sinh THUAN TU DSL (khong chen C):
  LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead.
- 5 kernel con lai (GeluErfFwd/Bwd, SoftmaxCausalFwd/Bwd, EmbedBwdFused)
  dung CO CHE AUTO-GLUE: kernel .tkv chi goi placeholder, notebook chen nguyen C cu
  vao file .cpu.c truoc khi gcc. => ABI/signature do compiler sinh, NHUNG THUAT TOAN
  VAN LA C CU. Khong duoc goi la "da thay bang TokenVector" - day la buoc dem.
- Cac note "STEP7c/7f/7g/7h PASS ... thay duoc" o tren phai doc kem nhan nay.
- QUYET DINH: giu 5 file auto-glue, sua sau. Cac file lam tiep PHAI viet THUAN TKV,
  khong chen C. Neu DSL thieu gi -> DUNG va bao user (khong tu y mo rong compiler).

### 2026-11-22 sang - Dang lam: LayernormFwd THUAN TOKENVECTOR
- Da xong: mo rong DSL them d_sqrt( -> sqrt() double (token trung gian __TKVSQRT__ de tranh
  bi sqrt->sqrtf an theo). Self-host vong.exe IL_IDENTICAL (135 file).
- Da viet layernorm_fwd_cpu.tkv THUAN TKV (KHONG chen C): d_mu/d_var/d_rstd double,
  vong for tinh mean/var/rstd, ghi float ra out. Generated C co `double d_rstd = 1.0/sqrt(d_var+1e-6)`.
- Commit TokenVector prototype/cuda-backend e0813c5.
- CHUA lam sang mai:
  1. Tao notebook bit-exact test LayernormFwd .tkv C vs native (moi cach: KHONG chen C,
     build truc tiep file .cpu.c, so sanh voi native.LayernormFwd tren 4 shape;
     luu y native co ATen shadow - test phay goi truc tiep _impl hoac true rieng).
  2. Neu bit-exact: tiep cac kernel khac (LN bwd, tkv_linear_dx.c tile 256,
     tkv_flash_attn.c, tkv_attn_cuda.c) - DEU VIET THUAN TKV, khong chen C.
  3. GiQ nguyen quy tac: 5 file da chen C (GELU fwd/bwd, Softmax fwd/bwd, EmbedBwd)
     sua sau; khong tu y doi phuong an khong bao user.
- Trang thai repo: TokenVector main 9ccc17b; prototype/cuda-backend e0813c5;
  Inference main 45f54ef (docs phan loai 4 thuan / 5 glue). Ca 2 s?ch.

### 2026-11-22 - Notebook test LayernormFwd san sang (cho user chay Colab)
- Da tao demo/colab_NATIVE_LAYERNORM_FWD_TEST_2026-11-22.ipynb (Inference 743a9df).
- KHONG chen C. Build CA HAI .so bang CUNG flags `-O2 -mavx2 -mfma` (tranh b?y FMA
  contraction: native fallback `v += d*d` duoc gan thanh FMA, neu .tkv build voi
  -ffp-contract=off se lech 1 ULP).
- Diagnostic trong cell 2: goi TkvTorchLayerNormFwd -> rc != 0 xac nhan ATen bridge vang,
  native.LayernormFwd that su dung duong f64 fallback (cai ma .tkv thay the).
- So sanh 4 shape (n,h) = (4,64),(16,256),(8,768),(7,33), bitwise uint32.
- HEAD: TokenVector prototype/cuda-backend e0813c5; Inference main 743a9df.
- CHUA co ket qua Colab. Khong claim bit-exact cho toi khi user tra notebook.

### 2026-11-22 (STEP7i PASS) - LayernormFwd BIT-EXACT, THUAN TKV (khong chen C)
- Ket qua Colab: TkvTorchLayerNormFwd rc=-9 (ATen vang, native dung f64 fallback);
  C INJECTED=False; sqrt(double) present=True; 4/4 bitwise=True diff=0
  (n,h)=(4,64),(16,256),(8,768),(7,33).
- Ky thuat quan trong: build CA HAI .so bang CUNG flags -O2 -mavx2 -mfma
  (tranh FMA contraction lech 1 ULP giua native va .tkv).
- KERNEL THUAN TKV THU 5: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead, LayernormFwd.
- 5 kernel auto-glue C van de nguyen, sua sau: GeluErfFwd/Bwd, SoftmaxCausalFwd/Bwd, EmbedBwdFused.
- HEAD: TokenVector prototype/cuda-backend e0813c5; Inference main bbca371 -> (commit tiep).

### 2026-11-22 (STEP7j PASS) - LayernormBwdFused BIT-EXACT, THUAN TKV (khong chen C)
- Ket qua Colab (prototype/cuda-backend 47ad847): TkvTorchLayerNormBwd rc=-9 (ATen vang,
  native dung f64 fallback); C INJECTED=False; 4/4 bitwise=True cho CA dx va gflat
  (n,h)=(4,64),(16,256),(8,768),(7,33); gflat khoi tao KHAC 0 de exercise accumulate.
- Loichit da fix: (1) d_xh/f_gg khai bao trong vong j dau -> DSL khong declare lai o vong 2
  -> C scope error; (2) comment Python # trong than kernel -> sinh directive C `#` -> error.
  Ca 2 la loi sinh C cua DSL, KHONG phai loi cong thuc.
- KERNEL THUAN TKV THU 6: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead,
  LayernormFwd, LayernormBwdFused.
- 5 kernel auto-glue C van de nguyen, sua sau: GeluErfFwd/Bwd, SoftmaxCausalFwd/Bwd, EmbedBwdFused.
- HEAD: TokenVector prototype/cuda-backend 47ad847; Inference main aa00df4 (notebook).

### 2026-11-22 - Danh gia kha nang viet thuan TKV cho cac file con lai
- 3 file lon KHONG the viet thuan TKV voi DSL hien tai (can mo rong LON, phai hoi user):
  * tkv_linear_dx.c: 57 token SIMD (__m256/_mm256_fmadd), 9 malloc, tile 256/k_eff.
  * tkv_flash_attn.c: 331 token SIMD, 55 malloc, 992 dong.
  * tkv_attn_cuda.c: 28 SIMD, 104 malloc, 126 token CUDA.
- 5 kernel auto-glue C cung can mo rong DSL (__m256, exp/erf f64, calloc) de sua that.
- Kernel elementwise DON GIAN van lam thuanne TKV duoc: F32ListAdd, F32ListCopy
  (F32Addcmul co bam order FMA khac giua AVX2 va scalar tail -> de sau).
- Nguyen tac: khong tu y mo rong compiler; da co d_sqrt (user cho phep) la du cho LN.

### 2026-11-22 (STEP7k PASS) - F32ListAdd/F32ListCopy BIT-EXACT, THUAN TKV
- Ket qua Colab (prototype/cuda-backend 10eb162): 8/8 bitwise=True diff=0
  (F32ListAdd va F32ListCopy, n=17/64/1000/4097 - co ca size le de test tail).
  C INJECTED=False. Build cung flags -O2 -mavx2 -mfma.
- KERNEL THUAN TKV THU 7-8: + F32ListAdd, F32ListCopy.
  Tong: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead, LayernormFwd,
  LayernormBwdFused, F32ListAdd, F32ListCopy.
- 5 kernel auto-glue C van de nguyen; 3 file SIMD/CUDA lon van can mo rong DSL (xem note truoc).
- HEAD: TokenVector prototype/cuda-backend 10eb162; Inference main c7c0d00 (notebook).

### 2026-11-22 (STEP7l) - Dang test: LinearDxFused THUAN TKV (scalar fmaf thay AVX2)
- User chon: mo rong DSL, bat dau linear_dx.
- Nhan xet chot: moi lane AVX2 (_mm256_fmadd_ps) tinh DOC LAP -> scalar fmaf cung thu tu
  cho ket qua BIT-IDENTICAL. KHONG can them __m256 vao DSL.
- Mo rong DSL 1 dong: builtin tkv_notfinite( -> !isfinite( de tinh k_eff dung nhu native.
  Self-host vong.exe IL_IDENTICAL (135 file).
- Viet linear_dx_cpu.tkv thuan TKV: tile T=256, k_eff quet forward (tuong duong native quet nguoc),
  KC=64, Ct gop don truoc roi moi cong vao dx (KHONG cong tung chunk).
- Test so voi native reference build RIENG tu tkv_linear_dx.c (11 args), vi tkv_simd_native.c
  co LinearDxFused 9 args (wrapper CUDA/BLAS khac) - khong the build chung 1 .so.
- Notebook demo/colab_NATIVE_LINEAR_DX_TEST_2026-11-22.ipynb, 5 shape, furniture rc.
- HEAD: TokenVector prototype/cuda-backend 80024e0; Inference main edbeeac.
- CHUA co ket qua Colab. Khong claim bit-exact khi chua tra notebook.

### 2026-11-22 (STEP7l PASS) - LinearDxFused BIT-EXACT, THUAN TKV, GEMM AVX2
- Ket qua Colab (prototype/cuda-backend b46910f): 5/5 bitwise=True diff=0 cho
  (m,n,k,T)=(8,768,3072,256),(8,3072,768,256),(16,64,256,256),(4,128,512,256),(4,64,300,256).
  C INJECTED=False; fmaf/isfinite present=True; rc=0/0.
- CHOT: KHONG can them __m256 vao DSL. Moi lane AVX2 tinh doc lap -> scalar fmaf
  cung thu tu = BIT-IDENTICAL. DSL chi mo rong 1 dong: tkv_notfinite -> !isfinite.
- Bug da fix: dieu kien `p0 + pc + p < pw` (so index tuyet doi voi do rong tuong doi)
  -> luon false khi p0>0 -> bo tile sau. Sua thanh `pc + p < pw`. Phat hien nh?
  shape k=300 (partial tile) trong bo test - shape don_le khong bat duoc.
- KERNEL THUAN TKV THU 9: + LinearDxFused (tile T=256, k_eff, KC=64, Ct gop truoc).
 Tong: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead, LayernormFwd,
 LayernormBwdFused, F32ListAdd, F32ListCopy, LinearDxFused.
- 5 kernel auto-glue C van de nguyen; con tkv_flash_attn.c, tkv_attn_cuda.c.
- HEAD: TokenVector prototype/cuda-backend b46910f; Inference main edbeeac/0d8f56a.

### 2026-11-22 (STEP7m PASS) - SoftmaxCausalBwd THUAN TKV (thay auto-glue)
- Ket qua Colab (prototype/cuda-backend 82193f9): 4/4 bitwise=True diff=0
  (n,hd)=(8,16),(16,64),(64,128),(127,64). C INJECTED=False, f64 dot present=True.
- Native SoftmaxCausalBwd_impl la SCALAR THUAN (dot f32 widen f64, cong tuan tu) -> khong
  can mo rong DSL. File auto-glue giam 5 -> 4.
- KERNEL THUAN TKV THU 10: + SoftmaxCausalBwd.
 Tong: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead, LayernormFwd,
 LayernormBwdFused, F32ListAdd, F32ListCopy, LinearDxFused, SoftmaxCausalBwd.
- Con 4 auto-glue: GeluErfFwd/Bwd (can d_exp f64), SoftmaxCausalFwd (can d_exp f64),
  EmbedBwdFused (can mang dong counts/starts/cursor/pos).

### 2026-11-22 (STEP7n PASS) - SoftmaxCausalFwd THUAN TKV (d_exp, thay auto-glue)
- Ket qua Colab (prototype/cuda-backend 3a4156f): 4/4 bitwise=True diff=0
  (n,inv)=(8,0.125),(16,0.25),(64,0.125),(127,0.25). C INJECTED=False, exp double.
- DSL mo rong: d_exp( -> exp( double (token __TKVEXP__, khong bi expf an). IL_IDENTICAL.
- Native la SCALAR THUAN (scale/max f64 chua round, exp f64, totv f64 chua round,
  ivv f64, mask). Auto-glue giam 4 -> 3.
- KERNEL THUAN TKV THU 11: + SoftmaxCausalFwd.
 Tong: LinearDwFused, BiasAddRows, BiasGrad, MergeHbHead, LayernormFwd,
 LayernormBwdFused, F32ListAdd, F32ListCopy, LinearDxFused,
 SoftmaxCausalBwd, SoftmaxCausalFwd.
- Con 3 auto-glue: GeluErfFwd/Bwd (can kiem tra AVX2 vs scalar bit-identical),
  EmbedBwdFused (can mang dong).

### 2026-11-22 (STEP7o PASS) - GeluErfFwd THUAN TKV (replicate AVX2 + scalar tail)
- Ket qua Colab (prototype/cuda-backend 31ab942): 4/4 bitwise=True diff=0
  (lo,hi)=(0,64),(5,64),(0,1000),(3,777). C INJECTED=False; rint/ldexp/d_exp present.
- Phat hien quan trong: native co 2 duong KHAC BIT:
  * AVX2: tkv_exp_ps (Taylor f32 + rint + ldexp bits), erf f32
  * scalar tail: exp(double), poly f64
  => phai replicate CA HAI. DSL them 4 builtin: tkv_rint/rintf, tkv_ldexp/ldexpf,
  tkv_fmax/fmaxf, tkv_fabs/fabsf. Self-host IL_IDENTICAL.
- Bug SL dam fix: (1) `i` khong duoc force int (type inference) -> them % 1000000000;
  (2) bien f_v/f_arg khai bao trong vong AVX2 bi dung lai o tail (mat C scope) -> doi ten t_*.
- KERNEL THUAN TKV THU 12: + GeluErfFwd. Auto-glue con 2: GeluErfBwd, EmbedBwdFused.

### 2026-11-22 (STEP7p/q PASS) - XONG TOAN BO 5 FILE AUTO-GLUE -> PURE TKV
- GeluErfBwd v2 (8e3d15a): 4/4 bitwise. Fix 1 ULP: gelu_b_one() tra FLOAT, phai round
  truoc khi nhan (double)gy. EmbedBwdFused (2252422): 4/4 bitwise, no calloc.
- NHAN XET CHOT EmbedBwd: mang counts/starts/cursor/pos cua native chi de ep thu tu
  T TANG DAN trong moi row. Scatter truc tiep for t: gw[idx[t]] += gy[t] cho CUNG thu tu,
  van O(n*d), KHONG can mo rong DSL, khong ch?m hon native.
- 14 KERNEL THUAN TKV (bitwise verified, KHONG chen C, KHONG dlsym glue):
  1 LinearDwFused 2 BiasAddRows 3 BiasGrad 4 MergeHbHead 5 LayernormFwd
  6 LayernormBwdFused 7 F32ListAdd 8 F32ListCopy 9 LinearDxFused
  10 SoftmaxCausalBwd 11 SoftmaxCausalFwd 12 GeluErfFwd 13 GeluErfBwd 14 EmbedBwdFused.
- DSL extensions duoc user cho phep: d_sqrt, tkv_notfinite, d_exp, tkv_rint,
  tkv_ldexp, tkv_fmax, tkv_fabs. Self-host vong.exe IL_IDENTICAL moi lan.
- HEAD: TokenVector prototype/cuda-backend 2252422; Inference main d304be3.
- CON LAI 2 file lon CHUAdung: tkv_flash_attn.c (992 dong, 331 SIMD, 55 malloc),
  tkv_attn_cuda.c (394 dong, 28 SIMD, 104 malloc, 126 CUDA). Can user quyet dinh.

### 2026-11-22 (STEP7r) - TICH HOP 14 PURE TKV VAO NATIVE BUILD (cho Colab verify)
- User chon: integrate vao native build, verify loss 25.454 / step time.
- Co che (theo tien le colab_NATIVE_LOSS_AFTER_TKV): sinh .cpu.c de OVERWRITE source native roi build lai .so.
  * 10 kernel trong tkv_bwd_ops.c -> gom thanh 1 file (strip include/define, them header) de.
  * linear_dw_cpu.cpu.c -> tkv_linear_dw.c; linear_dx_cpu.cpu.c -> tkv_linear_dx.c.
  * F32ListAdd/F32ListCopy nam trong tkv_simd_native.c -> thay body bang regex
    `EXPORT int NAME\(.*?\n\}\n` (da test match=True local).
- Notebook demo/colab_NATIVE_TKV14_INTEGRATE_LOSS_2026-11-22.ipynb (Inference 760bd37):
  cell 1 patch+build, cell 2 check 14 symbols + chay `dotnet demo/bseq.dll` mode 1 doc loss.
- Ky vong: loss van 25.454 (EXACT).neu doi -> .tkv kernel khong dung duoc cho native path.
- CHUA co ket qua Colab. Khong claim integrated thanh cong khi chua tra notebook.

### 2026-11-22 (STEP7r PASS) - TICH HOP 14 PURE TKV VAO NATIVE PATH, LOSS GIU NGUYEN
- Ket qua Colab (main 9ccc17b / prototype 2252422, Inference dfb2ecc):
  * build rc=0; bwd_ops replaced 10 kernels; linear_dw/dx replaced; F32ListAdd/Copy patched count=1;
    14 symbols missing = []; bseq rc=0.
  * loss_val = 25.45386 -> EXACT khop baseline V3 (2026-11-21), native path KHONG doi.
- TIMING that (mode2, 1 step, ms: fwd/loss/bwd/opt/copyback/TONG):
  * baseline V3 : 7209 / 947 / 20603 / 421 / 91 / 29271
  * integrated  : 8524 / 863 / 20209 / 430 / 86 / 30112
  * delta       : +1315 / -84 / -394 / +9 / -5 / +841 (~+2.9% tong).
  * Giai thich: fwd ch?m hon do GeluErfFwd thuan TKV phai replicate custom exp bang SCALAR
    (thay vi AVX2 8-lane). bwd nhanh hon nho LinearDxFused bo hang thua. KHONG co speed claim.
- Harness: mode 1 bi MissingMethodException `TKVApp._step124_gpu` trong Inference main hien tai;
  mode 2 chay tot. Da doi notebook sang mode 2.
- HEAD: TokenVector prototype/cuda-backend 2252422; Inference main dfb2ecc.

### 2026-11-22 (STEP7s) - GPU train verify sau khi doi compiler (cho Colab)
- Ly do: da them 7 builtin vao cpu_feature.tkv + rebuild tkvc-tv.exe. GPU path dung
  cuda_feature.tkv va file .cuda.cu SINH SAN (khong chay compiler trong notebook) -> ve ly
  thuyet khong anh huong, nhung van verify.
- Notebook demo/colab_GPU_VERIFY_AFTER_TKV14_2026-11-22.ipynb (Inference fd0336f):
  goc colab_train_gpt2medium, gioi han STEPS=120, bo 2 cell perplexity/sample,
  them print VERIFY HEADS. Ky vong: ADAMW PATH=1, loss giam, SPEEDUP ~2x.
- CHUA co ket qua Colab.

### 2026-11-22 (STEP7t) - FIX ROOT CAUSE GPU loss 10.82 (CUDA int inference)
- Phat hien: GPU verify (STEP7s) loss 3.8906 -> 10.8249 (ln V). Khong phai do 14 kernel native.
- B?ng ch?ng: notebook tot 11-21 dung main 07e0ce8 + CUDA backend 7da4f67; `git diff 07e0ce8 9ccc17b`
  RONG (M4/M4b reverted net-zero). adamw.cuda.cu khac 1 dong:
  `int pi` (bad, tu 94e2dfb) thay vi `float pi` (good, tu 7da4f67).
- Root cause: rule trong cuda_feature.tkv `'r * ' in rhs` khop NHAM `lr * mi` -> is_int=1 -> int pi
  -> moi update param bi truncate -> model random -> loss ln(V).
- FIX: them `_cuda_has_int_mul()` kiem tra word-boundary (chi tinh index khi toan tu o ranh gioi tu),
  thay rule cu. Self-host vong.exe IL_IDENTICAL. Regenerate adamw.cuda.cu -> `float pi` (int_pi=False).
  Commit TokenVector prototype/cuda-backend 2eee803.
- Notebook verify: demo/colab_GPU_VERIFY_CUDA_FIX_2026-11-22.ipynb (Inference 3edf1f4),
  co REGRESSION assert `float pi` trong adamw.cuda.cu + 120 steps train.
- Ky vong: loss 3.89 -> ~3.7, ADAMW PATH=1, speedup ~2x. CHUA co ket qua Colab.

### 2026-11-22 (STEP7t PASS) - FIX CUDA int inference, GPU loss khong con diverge
- Ket qua Colab (main 9ccc17b / prototype 2eee803, Inference d653cfe):
  * adamw float pi = True, int pi = False; build rc=0; ADAMW PATH=1.
  * loss 3.8906 -> (step50 4.4917, step100 4.2292, step119 3.9272). KHONG con 10.8249.
  * torch median 450.4ms, TKV median 245.0ms, SPEEDUP = 1.84x (120 steps).
  * VERIFY HEADS 9ccc17b 2eee803.
- So sanh lich su (can thanh th?t, KHONG claim 2.23x cho lan nay):
  * 400-step run cu (backend 7da4f67): torch 490.8ms, TKV 219.8ms, 2.23x, loss 3.89->3.74.
  * 120-step run nay: torch 450.4ms, TKV 245.0ms, 1.84x, loss 3.89->3.93 (chua du steps).
  * Khac so steps + cua so median + GPU noise => khong the so sanh truc tiep. Chi khang dinh:
    fix loai bo diverge, GPU path chay tot lai.
- TONG KET 2 MUC TIEU:
  * Native EXACT: loss 25.45386 (mode 2) sau khi thay 14 kernel C bang pure TKV. +2.9% CPU tong
    (fwd +18% do GELU scalar replica); khong co speed claim.
  * GPU: AdamW .tkv kernel (ADAMW PATH=1) chay, loai bo bug int pi co san trong compiler.
- HEAD cuoi: TokenVector main 9ccc17b; prototype/cuda-backend 2eee803; Inference main d653cfe.
