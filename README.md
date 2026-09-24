# 🚀 TokenVector.Inference: Ultra Low-Latency Model Serving & Quantization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

**TokenVector.Inference** is an industrial-grade, **Zero-GC Allocation**, pure **TokenVector (`.tkv`)** deep learning model serving, graph optimization, and precision quantization engine designed for the TokenVector AI ecosystem.

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
6. **100% TokenVector (`.tkv`):**
   - Entire engine and tests are `.tkv` sources compiled with `tkvc`; no external native runtime DLLs required for the core path.

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

## 🛠️ Build, Test, and Packaging

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/tests/inference_tests.tkv
tkvc build tv/benchmarks/compare_rivals.tkv
tkvc build stdlib/tv/inference/inference.tkv --target library
```
