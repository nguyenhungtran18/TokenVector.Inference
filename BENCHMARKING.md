# How to Benchmark TokenVector.Inference vs PyTorch

Practical recipe for apples-to-apples comparison on the GPT-2 124M training step
(the benchmark we publish). Data journal: `BENCHMARKS.md` (latest: §9.22).
Results directory: `benchmark/` (gitignored, except the two files of each recorded run).

## What you need

- Reference compiler (self-hosted): `D:\TokenVector\Release\dist\tkvc-tv.exe`
  (override with `$tkvc` anywhere below).
- .NET 10 runtime on `PATH` (compiled binaries need `dotnet <exe>` and a
  `.runtimeconfig.json` next to them — `tkvc-tv` does **not** emit one).
- Python + `torch 2.14+cpu` + `transformers` (for the PyTorch side).
- ~8 GB free RAM; machine Xeon E5-2695 v3 (14C/28T) is the reference host.

> **Measured 2026-11-11 (same session, §9.22):** full step @seq1024 =
> **4016 ms** (TKV) vs **4061.1 ms** (torch) → **0.989×**, loss EXACT.

## One-time setup

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
# ensure the config file exists and pick today's cell:
Set-Content tv/tests/_seq_cfg.txt "1024`n1`n1`n1"   # seq / mode / steps / warmup
```

`_seq_cfg.txt` fields: `seq` (8/256/1024), `mode` (`0` = fwd+bwd,
`1` = full step, `2` = 5-phase breakdown), `steps`, `warmup`.

## Build the TKV bench binary

```powershell
& $tkvc tv/tests/simd_bench_124m_seq.tkv benchmark/bseq.exe main
# runtimeconfig is required or you get "hostpolicy.dll not found":
Set-Content benchmark/bseq.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
```

## Measure TKV

```powershell
# single config — best-of-3 (run 3x, take the min):
dotnet benchmark/bseq.exe
# output example:
#   BSEQ full-step simd seq=1024 ms=4016.0 loss=10.760399 -> 13.442921
# mode 0 prints:  BSEQ fwd=... bwd=... total=...
# mode 2 prints:  BPHASE seq=1024 fwd=1016.0 loss=109.0 bwd=2328.0 opt=453.0 copyback=141.0 TONG=4047.0 loss_val=...
```

Matrix driver (TKV best-of-3 + torch both tiers, logs to `benchmark/`):

```powershell
powershell -File tools/_bench_matrix.ps1
```

## Measure PyTorch (always re-measure, same session)

```powershell
python tools/bench_torch.py d 1024   # tier d: fwd+bwd and full step
python tools/bench_torch.py e 1024   # tier e: zero/fwd/bwd/opt breakdown, T=1 and T=14
```

**Never quote a cross-session torch number.** §9.21 borrowed
§9.18's `4130.6`; the same-session figure turned out to be `4061.1`.
Every published table states whether it is same-session or cross-session.

## Reading the results

| TKV output | Meaning | Pair with |
|---|---|---|
| `BSEQ fwd=… bwd=… total=…` (mode 0) | fwd+bwd, no optimizer | torch tier d `fb()` |
| `BSEQ full-step simd` (mode 1) | one full training step | torch tier d `step()` |
| `BPHASE … TONG=…` (mode 2) | 5 phases, min-of-2 | torch tier e phases |
| `BENCH E torch … seqN T=14` | torch phases | BPHASE |

Loss must print **EXACT** values: `10.572291` / `10.756927` / `10.760399`
(seq8/256/1024). If loss differs, stop — something is wrong.

## Adding a new run to the journal

1. Run the matrix above; save the raw log to
   `benchmark/_bench_matrix_<tag>_<yyyyMMdd_HHmmss>.log`.
2. Append a `§9.x` section to `BENCHMARKS.md` with: host CPU, compiler path +
   sha256, torch version + threads, same/cross-session note, tables of TKV vs
   torch (mode 1 best-of-3, mode 2 TONG, tier `d`/`e`), loss EXACT line, note
   on variance (`mode 1` is noisy — `mode 2` is the number to trust).
3. Note it in `HANDOFF.md` (append-only).

## Gotchas

- `dotnet benchmark/bseq.exe` fails with `hostpolicy.dll not found` →
  missing `*.runtimeconfig.json` next to the exe (see §9.17).
- `OMP: Warning #182` is harmless (OpenMP affinity).
- RAM: full step @124M keeps ~7.2 GB resident; close torch before/after to
  avoid page-file noise; do not run torch and TKV concurrently.
- `tv/tests/_seq_cfg.txt` is rewritten by the scripts — restore it after
  experiments.
- `mode 1` swings ±5% between runs; use `mode 2` for phase decisions.
- Do not benchmark with a compiler build that lacks the native AVX2/FMA
  exports (`tkv_simd_native.dll`) — the fallback scalar path is ~2× slower.
