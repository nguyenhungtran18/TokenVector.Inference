# 🚀 TokenVector.Inference: Ultra Low-Latency Model Serving & Quantization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

<img src="assets/tokenvector-logo.svg" alt="TokenVector logo" width="96" height="96">

**TokenVector.Inference** is an industrial-grade, **Zero-GC Allocation**, pure **TokenVector (`.tkv`)** deep learning model serving, graph optimization, and precision quantization engine designed for the TokenVector AI ecosystem.

> **Current release status:** 47 `.tkv` modules build clean with the packaged `tkvc`, `tv/tests/gap_tests.tkv` passes, and the runtime test suite runs. The runtime is **100% TokenVector with zero native dependencies** — no C/C++ sources, no P/Invoke, no companion DLL. CPU execution is available, TokenVector SIMT is a deterministic simulation, and CUDA/OpenCL/OpenVINO are reported honestly as unavailable.

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

1. **Zero-GC Allocation Hot Path:**
   - Pre-allocates a unified memory arena using **Tensor Liveness Analysis**.
   - Hot-path execution aims for **0 Byte** heap churn on the arena-reuse path.
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
5. **Ultra-Low Latency Embedded Serving:**
   - In-process REST engine (`EmbeddedServer`) with OpenAI-compatible API, SSE streaming, auth, limits, and Prometheus metrics.
   - Micro-batching / continuous batching over a session pool with paged KV + prefix skip.
6. **TokenVector-first runtime — zero native dependencies:**
   - Core compute, graph execution, serving, quantization and tests are `.tkv` sources compiled with `tkvc`.
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

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/transformer.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output shape:", output_tensor.shape)
session.close()
```

CLI:

```powershell
tkvc build tv/tools/cli.tkv
# then run the produced executable with serve|gen|export|calibrate|latency
```

---

## Build, Test, and Packaging

```powershell
# TokenVector compiler (packaged as compiler/tkvc_patched.exe)
$tkvc = ".\compiler\tkvc_patched.exe"

& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/benchmarks/compare_rivals.tkv
& $tkvc build tv/tests/gap_tests.tkv --out gap_tests.exe
.\gap_tests.exe
```

Build **every** module (the canonical sources are the files not named `tv.*.tkv` — those are generated import copies):

```powershell
$files = Get-ChildItem -Recurse -Filter *.tkv | Where-Object { $_.Name -notmatch '^tv\.' }
foreach ($f in $files) { & $tkvc build $f.FullName --target library --out out.dll }
```

The produced executables are self-contained: no DLL needs to sit next to them.
