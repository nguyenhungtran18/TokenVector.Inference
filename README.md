# 🚀 TokenVector.Inference: Pure-`.tkv` Deep Learning Runtime & Training Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc--tv-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)

<img src="assets/tokenvector-logo.png" alt="TokenVector logo" width="96" height="96">

**TokenVector.Inference** is a deep learning runtime written in **TokenVector
(`.tkv`)** — inference (graph optimization, quantization, micro-serving) **and**
a GPT-2 124M training engine whose step time is now at parity with PyTorch
(measured 2026-11-11, see below).

> **Current status, verified 2026-11-11:**
> - GPT-2 124M full train step @seq1024: **4016 ms** vs PyTorch 2.14.0+cpu
>   (same session, 14 threads) **4061.1 ms** → **0.989×**, loss **EXACT**
>   (`10.760399`), `gate_f32.sh` **135/135**, `trainer_tests` **15/15**,
>   `gpt2_full_tests` **33/33**. Details: `benchmark/BENCHMARKS.md` §9.21–§9.22.
> - Sources are `.tkv`; the optimizing pipeline emits IL that P/Invokes the
>   companion native SIMD library (`tkv_simd_native.dll`, built by the
>   compiler repo) for hot kernels. The CPU path is the supported path.
> - Known gaps: seq8/seq256 full step still slightly behind torch
>   (1.067×/1.027×); working set ~7.2 GB @seq1024; mode-1 timing is noisy —
>   use mode-2 (phase breakdown) for decisions.
>
> `docs/HANDOFF.md` is the running engineering journal (append-only). Every known
> gap, and every wrong turn taken to find them, is documented there.

## What ships today

| Area | State |
|---|---|
| Inference runtime (graph, PTQ INT8/FP8, ONNX reader, batcher) | Working, tested; see `docs/TEST_REPORT.md` |
| GPT-2 124M training (fwd/bwd/AdamW/clip/checkpoint) | Working; see `benchmark/BENCHMARKS.md` §9 |
| Native SIMD kernels (GEMM, fused bwd, AdamW) | Shipped via compiler repo (`tkv_simd_native.dll`) |
| GPU / NPU offload | **Unavailable** — CPU only |

## Requirements

- **TokenVector compiler**, current self-hosted build (referred to here as
  `$tkvc`). The reference build is `D:\TokenVector\Release\dist\tkvc-tv.exe`.
- **.NET 10 runtime** to run the compiled executables. `tkvc-tv` does **not**
  emit a `.runtimeconfig.json` next to the output — without it you get
  `hostpolicy.dll not found`. Create one:

  ```json
  {"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}
  ```

  (or run via `dotnet <app>.exe` with the file present, cf. `benchmark/BENCHMARKS.md` §9.17).

## Repository layout

```
tv/            TokenVector sources (engine)
  runtime/     graph runtime, session, arena, SIMD matmul (.tkv)
  training/    autograd, AdamW/clip, GPT-2 124M reference (.tkv)
  tests/       gate suites (fixtures not committed; see below)
  tools/       CLI, trainer demo
tools/         host-side verification: gate_f32.sh (135/135), run_cases.ps1,
               verify_*.ps1 (parity), fixture generators (gen_*.py)
benchmark/     benchmark tooling & scratch: bench_torch.py (PyTorch reference),
               _bench_matrix.ps1 (matrix runner), probes, logs;
               binaries & dumps gitignored, scripts & logs tracked
docs/          engineering docs;  docs/archive/ = completed/superseded plans
packaging/     NuGet packaging (TokenVector.Inference.nuspec, pack.ps1) — build via packaging/pack.ps1
reports/       dated parity & benchmark reports
benchmark/BENCHMARKING.md step-by-step benchmark guide (EN + VI)
benchmark/BENCHMARKS.md  master benchmark journal (current: §9.22)
docs/HANDOFF.md     engineering journal (append-only)
```

GPT-2 124M parity fixtures (~2.3 GB) are **not in git**; regenerate with
`python tools/gen_gpt2_fixture.py` (see `docs/handoff-gpt2-124m-fixture-design.md`).

## Quick start

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"

# Inference quickstart (graph -> session -> one forward pass)
& $tkvc tv/examples/quickstart.tkv benchmark/quickstart.exe main
Set-Content benchmark/quickstart.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/quickstart.exe      # prints: Y[0] = 2.25  /  quickstart: OK
```

The API is graph-based; there is no `tv.inference.load_onnx()` and no
`NDArray`. See `USER_GUIDE.md` for the API walkthrough and
`docs/TOKENVECTOR_SYNTAX_SPEC.md` for the language rules that bite (module-level
constants are not importable; record `__init__` is skipped; `TkvInt` …).

## Build & test

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"

# Inference case suites (each case in its own process)
powershell -File tools\run_cases.ps1 -Compiler $tkvc

# f32 kernel gate, 10 suites, 135 assertions
bash tools/gate_f32.sh

# Trainer gate (MLP, AdamW vs PyTorch reference): 15/15 expected
& $tkvc tv/tests/trainer_tests.tkv benchmark\trainer_tests.exe main
dotnet benchmark\trainer_tests.exe
```

## Benchmark vs PyTorch

```powershell
# PyTorch side (needs python + torch 2.14+cpu + transformers)
python benchmark/bench_torch.py d 1024   # fwd+bwd and full step
python benchmark/bench_torch.py e 1024   # phase breakdown T=1 / T=14

# TokenVector side (124M, _seq_cfg.txt picks seq/mode/steps/warmup)
Set-Content tv/tests/_seq_cfg.txt "1024`n1`n1`n1"
& $tkvc tv/tests/simd_bench_124m_seq.tkv benchmark/bseq.exe main
dotnet benchmark/bseq.exe
```

Latest same-session results (2026-11-11, `benchmark/BENCHMARKS.md` §9.22):

| metric | TKV | PyTorch | TKV/torch |
|---|---:|---:|---:|
| full step @seq1024 | **4016** | 4061.1 | **0.989×** |
| full step @seq256 | 1594 | 1551.4 | 1.027× |
| full step @seq8 | 1047 | 981.4 | 1.067× |
| fwd+bwd @seq1024 | 3375.0 | 3395.3 | **0.994×** |
| opt.step @seq1024 (phase) | **453** | 646.4 | **−193 ms** |

## Docs map

- `benchmark/BENCHMARKS.md` — benchmark journal (§9.22 = latest, same-session vs torch)
- `docs/HANDOFF.md` — engineering journal, every investigation and wrong turn
- `USER_GUIDE.md` / `USER_GUIDE_VI.md` — API user guide
- `TOKENVECTOR_SYNTAX_SPEC(_VI).md` — language syntax rules
- `docs/COMPILER-NOTES-2026-10-16.md` — compiler team notes (2026-10-16)
- `docs/numerics-contract.md` — fp32 numerics contract (what is/ isn't bit-exact vs PyTorch)
- `docs/COMPILER_TEAM_REQUEST*.md` — requests to the compiler team (#1–#14)
- `docs/archive/` — completed/superseded plans (do not execute)
- `RELEASE_NOTES(_VI).md` — release history
- `docs/ROADMAP_TRAIN.md` — training roadmap (campaign complete)

## Ecosystem

**TokenVector** is a next-generation AI/HPC ecosystem built on the `.tkv`
language and `tkvc` compiler:

- [`TokenVector`](https://github.com/nguyenhungtran18/TokenVector) — language & compiler
- [`TokenVector.Numerics`](https://github.com/nguyenhungtran18/TokenVector.Numerics) — NDArray / linear algebra
- [`TokenVector.Vision`](https://github.com/nguyenhungtran18/TokenVector.Vision) — 2D & 3D volumetric vision
- [`TokenVector.Text`](https://github.com/nguyenhungtran18/TokenVector.Text) — tokenization, embeddings, LLM preprocessing
- [`TokenVector.Data`](https://github.com/nguyenhungtran18/TokenVector.Data) — double-buffered data pipeline
- **TokenVector.Inference** — this repo: inference runtime + GPT-2 124M training
