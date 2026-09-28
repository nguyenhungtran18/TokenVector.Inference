# 🚀 TokenVector.Inference: Ultra Low-Latency Model Serving & Quantization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

<img src="assets/tokenvector-logo.svg" alt="TokenVector logo" width="96" height="96">

**TokenVector.Inference** is a pure **TokenVector (`.tkv`)** deep learning graph runtime — graph optimization, precision quantization, and a micro-batcher — built for the TokenVector AI ecosystem.

> **Current status, verified 2026-09-28:** 26/26 cases in `tv/tests/inference_tests.tkv` and 16/16 in `tv/tests/gap_tests.tkv` pass. The runtime is **100% TokenVector with no native code in this package** — no C/C++, no P/Invoke, no companion DLL. CPU execution works, TokenVector SIMT is a deterministic simulation, and CUDA/OpenCL/OpenVINO are honestly reported as unavailable.
>
> Read the caveats before relying on it: the tensor arena is **not** zero-GC (see below), there is **no SIMD**, and several entry points are stubs. `HANDOFF.md` lists every known gap; `SIMD_NOTES.md` covers vectorization.


### Backend status

| Backend | Status |
|---|---|
| CPU | Available — pure `.tkv` scalar kernels (`hal.tkv`) |
| TokenVector SIMT | Deterministic lane simulation (`simt.tkv`) |
| CUDA | **Unavailable** — no GPU backend in the pure-TokenVector runtime |
| OpenCL | **Unavailable** — no GPU backend in the pure-TokenVector runtime |
| OpenVINO NPU | **Unavailable** — no NPU backend in the pure-TokenVector runtime |

Training (`tv/training/autograd.tkv`) runs on the CPU path only: forward, backward, gradient check and Adam are all pure `.tkv`. It has never depended on a GPU backend.

---

## 🌐 The TokenVector AI Ecosystem Overview

**TokenVector** is a next-generation AI and High-Performance Computing (HPC) ecosystem built on the **TokenVector language (`.tkv`)** and its **`tkvc` compiler**, designed for high-performance CPU inference with **0% Garbage Collection overhead (Zero-GC)**:

```
                         ╔══════════════════════════════════════════════╗
                         ║         THE TOKENVECTOR AI ECOSYSTEM         ║
                         ╚══════════════════════════════════════════════╝
                                                │
         ┌────────────────────────┬─────────────┴────────────┬────────────────────────┐
         ▼                        ▼                          ▼                        ▼
┌──────────────────┐    ┌──────────────────┐       ┌──────────────────┐     ┌──────────────────┐
│TokenVector.Vision│    │TokenVector.Numerics│     │ TokenVector.Text │     │ TokenVector.Data │
│(Computer Vision  │    │  (Linear Algebra │       │(Text, Tokenizer, │     │ (Data Pipeline & │
│  2D & 3D Voxel)  │    │  & NDArray Core) │       │ Embeddings & LLM)│     │  Double-Buffer)  │
└────────┬─────────┘    └────────┬─────────┘       └────────┬─────────┘     └────────┬─────────┘
         │                       │                          │                        │
         └───────────────────────┼──────────────────────────┴────────────────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ TokenVector.Inference  │
                     │  (Embedded Inference:  │
                     │  Serving, Opt & Quant) │
                     └────────────────────────┘
```

### Core Components in the Ecosystem:
1. **[`TokenVector`](https://github.com/nguyenhungtran18/TokenVector)**: The official TokenVector programming language (`.tkv`) and compiler (`tkvc`) with standard library (`stdlib/tv/inference`).
2. **[`TokenVector.Numerics`](https://github.com/nguyenhungtran18/TokenVector.Numerics)**: High-performance linear algebra and N-dimensional array (`NDArray<T>`) tensor core with zero-copy interop.
3. **[`TokenVector.Vision`](https://github.com/nguyenhungtran18/TokenVector.Vision)**: 2D & 3D volumetric vision engine, fused transforms, YOLO Letterbox, NMS, and ImagePainter.
4. **[`TokenVector.Text`](https://github.com/nguyenhungtran18/TokenVector.Text)**: High-speed text processing, tokenization (BPE, WordPiece, SentencePiece), embeddings, and LLM preprocessing.
5. **[`TokenVector.Data`](https://github.com/nguyenhungtran18/TokenVector.Data)**: Multi-threaded double-buffering prefetching pipeline eliminating I/O bottlenecks.
6. **[`TokenVector.Inference`](https://github.com/nguyenhungtran18/TokenVector.Inference)**: Ultra low-latency deep learning inference runtime, graph optimization passes, INT8/FP8 quantization, and in-process micro-serving — **100% `.tkv`**.

---

## 🌟 Key Capabilities & Highlights

1. **Tensor Liveness Analysis** (arena reuse, *not* zero-GC today):
   - `ExecutionGraph.analyze_liveness_and_allocate_offsets()` computes live ranges and reuses scratch slots.
   - The arena is `dict[i64, list[f32]]` — boxed element storage, and `native_arena_alloc` is still a stub returning `0`. So each tensor costs managed allocations. The 0-byte hot-path goal is **not** met; see `SIMD_NOTES.md` §7 for the fix.
2. **Zero-Copy Protobuf & ONNX Reader:**
   - Pure `.tkv` protobuf wire reader decoding ONNX binaries from raw byte lists without intermediate string/object garbage.
3. **Graph Optimization Pipeline:**
   - `ConstantFoldingPass`: Folds static reshape/transpose/permute into initializers ahead of time.
   - `DeadCodeEliminationPass`: Prunes disconnected operations and intermediate buffers.
   - `OperatorFusionPass`: Kernel fusion (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) for cache-friendly execution.
4. **Quantization Engine (INT8 & FP8):**
   - Static PTQ via MinMax, KL, percentile, and MSE calibration.
   - Symmetric & asymmetric INT8 plus grouped INT4 (AWQ-style) and per-channel GEMM.
   - FP8 E4M3/E5M2 and FP16 compute paths for modern Transformer architectures.
5. **In-process serving structures (not yet listening):**
   - `EmbeddedServer` and `DynamicBatcher` exist and the batcher's synchronous path works, but `http_listen()` is a stub returning `0` and `spawn()` creates no worker thread, so no socket is bound and no micro-batching happens across threads.
   - Paged KV + prefix skip are implemented and covered by tests.
6. **TokenVector-first runtime — no native code in this package:**
   - Core compute, graph execution, quantization and tests are `.tkv` sources compiled with `tkvc`.
   - **No C/C++ in the repository, no `__tkv_extern_pinvoke__`, no companion DLL.** Byte codecs and IEEE-754 float bit reinterpretation are implemented in `.tkv` (`tv/runtime/f32_bits.tkv`), so the ONNX writer needs nothing beside its own executable.
   - GPU/NPU offload would have to ship as a separate native utility, not as a library every executable must carry.

---

## 📊 Benchmark Suite

Latency/throughput numbers must be measured on this pure `.tkv` engine — do not quote numbers from any previous backend.

```powershell
tkvc build tv/benchmarks/compare_rivals.tkv
```

See `BENCHMARKS.md` / `BENCHMARKS_VI.md` for suite structure and `COMPETITOR_COMPARISON.md` for functional (non-speed) rival comparison.

---

## ⚡ Quick Start (`.tkv`)

Runnable example — `tv/examples/quickstart.tkv` builds a graph, turns it into a
session and runs one forward pass:

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
& $tkvc build tv/examples/quickstart.tkv --out quickstart.exe
.\quickstart.exe      # prints: Y[0] = 2.25  /  quickstart: OK
```

The API is graph-based; there is no `tv.inference.load_onnx()` and no `NDArray`:

```tokenvector
import tv.runtime.session
# Module-level constants are NOT importable - redeclare them (values from
# tv/runtime/session.tkv and tv/optimization/kernels.tkv).
OperatorType_FusedLinear = 2
ActivationFunction_Relu = 1

g = ExecutionGraph(1, "quickstart", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w          # list[f32] of length n*k
g.initializers["B"] = b          # list[f32] of length n
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("W", [8, 4])
g.get_or_create_tensor("B", [8])
g.get_or_create_tensor("Y", [1, 8])

node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")          # node.inputs.append("W"), append("B")
node.outputs.append("Y")
g.nodes.append(node)

session = session_from_graph(g, 1)   # 1 = run the optimization passes
y = session.run_single(x, "X", "Y")  # x is a flat list[f32]
session.close()
```

What is **not** wired up yet (stubs, do not put these in a quick start):

| API | State |
| :--- | :--- |
| `tv.inference.load_onnx()` | `tv.inference` is an auto-generated stub; the module does not exist |
| `tensor.zeros(...)` | no such symbol; tensors are plain `list[f32]` |
| `parse_file("m.onnx")` | `parser_read_bytes` is a stub returning `""`, so file reading is not implemented |
| `cli.tkv serve ...` | `argv()` is a stub returning `[]`, so the CLI always prints usage |
| `http_listen()` (embedded server) | stub returning `0`; nothing binds a socket |

---

## Build, Test, and Packaging

```powershell
# TokenVector compiler
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"

# Tests: 26 cases, each in its own process, then the 16-case gap suite
powershell -File tools\run_cases.ps1 -Compiler $tkvc
#   TOTAL: pass=26 fail=0 crash=0
#   gap tests: PASS (16/16)

# Build + test + benchmark in one go
powershell -File build_inference.ps1 -SkipBenchmarks
```

**Do not build modules one by one.** `tkvc build <single module>` rewrites the
generated dotted import copies, and has been observed to leave nine of them filled
with NUL bytes (restored with `git checkout HEAD -- <file>`). Build only entry
points:

```powershell
& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/tests/gap_tests.tkv        --out gap_tests.exe
& $tkvc build tv/examples/quickstart.tkv    --out quickstart.exe
& $tkvc build tv/benchmarks/compare_rivals.tkv --out compare_rivals.exe
```

The canonical sources are the files **not** named `tv.*.tkv` — those are generated
import copies, and after editing a canonical file you must copy it to every
matching `tv.*.tkv`.

The produced executables are self-contained: no DLL needs to sit next to them.
