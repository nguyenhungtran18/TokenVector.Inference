# Benchmark Guide — GPT-2 124M full training step

This guide reproduces the numbers quoted in `README.md` for TokenVector.Inference.
All numbers below were measured **2026-11-20** on a Colab **Tesla T4 16 GB**,
Python 3.13, torch `2.14.1+cu130`, with the following HEADs:

- `TokenVector`: `9ef0d91` (native shim + compiler wrapper fixes)
- `TokenVector.Inference`: `f1d4b99` (notebook, `bseq.dll` SHA256 `4eb13bc3…`)

## 1. Results (real GPU time on both sides)

| Path | seq | TokenVector | PyTorch eager | Ratio |
|---|---:|---:|---:|---:|
| `bseq.dll` fullstep (cell `# 7/9`) | 1024 | **40.0 ms** | 289.1 ms | **~7.2× faster** |
| ctypes fullstep (cell `7D/9`) | 2048 | **212.6 ms** | 599.5 ms | **~2.8× faster** |

Timing fairness: the repo worker runs with `CUDA_LAUNCH_BLOCKING=1`; the torch
worker calls `torch.cuda.synchronize()` after each `opt.step()`. Neither side is
enqueue-time-only.

## 2. Files involved

- Notebook: `demo/colab_bench_ai_gpu.ipynb` (or the dated copies
  `colab_bench_fullstep_FIXED_2026-11-20.ipynb` /
  `colab_bench_fullstep_DIAG_2026-11-20.ipynb`).
- Config: `tv/tests/_seq_cfg.txt`, four lines: `seq`, `mode`, `steps`, `fullstep_flag`.
  Line 4 = `1` routes mode 2 to `_step124_gpu()` -> one `TkvFullStep124m` call.
- Bench source: `tv/tests/simd_bench_124m_seq.tkv`.
- Native bridge: `Release/tools/simd_native/torch_fused.cpp` (`TkvFullStep124m`)
  and `tkv_simd_native.c` (P/Invoke forwarders).

## 3. Steps to reproduce

1. Open the notebook in Colab (T4 GPU runtime).
2. Cell `1/9`: verify GPU/torch. Cell `2/9`: install .NET 10 SDK.
3. Cell `3/9`: `rm -rf` + fresh clone of both repos; verify printed HEADs and
   `demo/bseq.dll sha256: 4eb13bc3d79691e56fa2008ba99cc93004270703d2221c15d6981f71568e2aea`.
4. Cell `4/9`: build `tkv_simd_native.so` + `tkv_step124.so`.
5. Cell `5/9`: build `tkv_torch_fused.so` (must print `build rc = 0` and `ctypes load OK`).
6. Cell `# 7/9`: fullstep via `bseq.dll`, T=1024, three calls in one process.
   Expected lines:
   ```
   1: FULLSTEP TONG=  40.0 ms | loss_val=8.570312 (T=1024)
   2: FULLSTEP TONG=  37.4 ms | loss_val=8.570312 (T=1024)
   3: FULLSTEP TONG=  40.1 ms | loss_val=8.570312 (T=1024)
   TONG median : 40.0 ms
   ```
   The first call includes CUDA lazy-load; the notebook reports the median of
   the last two.
7. Cell `7D/9`: ctypes fullstep, T=2048. Expected `FULLSTEP median ms≈200`.
8. Cell `8/9`: torch eager reference, T=2048 (synchronized).
9. Last cell `# 10/9`: torch eager reference, T=1024 (synchronized) — the fair
   comparison target for step 6.

## 4. How to interpret

- `TONG` = total wall time of one full training step (forward + loss + backward
  + AdamW), excluding host-side process startup.
- The GPU fullstep uses **fp16 parameters with an fp32 master copy and fp32
  moments**; the torch reference is an fp32 model. Part of the speed-up comes
  from fp16 tensor-core compute.
- **Loss values are not comparable** across modes: random weights seed 7,
  different LR schedules (`.tkv` `lr_max=0.05`, warmup 2 steps vs constant
  `0.0025` in the torch reference), and different numerics.
- The native CPU EXACT path (loss `25.454`) is untouched and is a separate,
  non-GPU code path.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FULLSTEP FAIL rc=134` with `MissingMethodException: TkvFullSetGradM` | stale `bseq.dll`; re-clone HEAD with the int64 wrapper fix |
| `EntryPointNotFoundException: TkvFullBegin` | stale `tkv_simd_native.so`; re-run cell 4 |
| `vectorized_gather_kernel index out of bounds` | stale native shim; ids must be self-generated (`TkvFullSetIdsGen`) |
| `TONG ≈ 500-600 ms` | first step CUDA lazy-load; use the median of 2nd/3rd call |
| Ratios look ~7x better than expected | check that cell 6 really sets `CUDA_LAUNCH_BLOCKING=1` |

## 6. Historical reference

- Native CPU fullstep @seq1024: `4016 ms` vs torch `4061.1 ms` (0.989×, 2026-11-11).
- Per-op GPU bridge (148 op dispatches): `~3286 ms/step` — superseded by fullstep.
- Full journal: `docs/FULL_STEP_GPU_NOTES_2026-11-19.md`, `docs/HANDOFF.md`.
