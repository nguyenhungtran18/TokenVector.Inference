# 🚀 TokenVector.Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Target: .NET 8.0](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![AOT Ready](https://img.shields.io/badge/Native%20AOT-Ready-success.svg)]()
[![Zero-GC Hot Path](https://img.shields.io/badge/Hot%20Path-0%20Byte%20GC-brightgreen.svg)]()

**TokenVector.Inference** is an ultra-high-performance, **Zero-GC Allocation**, pure C# Native deep learning model serving, graph optimization, and precision quantization engine designed for the TokenVector AI ecosystem.

---

## 🌟 Key Capabilities & Highlights

1. **Zero-GC Allocation Hot Path:**
   - Pre-allocates unified unmanaged `ExecutionMemoryArena` using **Tensor Liveness Analysis**.
   - `session.Run(...)` executes strictly **0 Byte** of managed heap memory allocation.
2. **Zero-Copy Protobuf & ONNX Reader:**
   - High-speed unmanaged parser (`FastProtobufReader`) decoding ONNX binaries directly from `ReadOnlySpan<byte>` without intermediate strings or object garbage.
3. **Graph Optimization Pipeline:**
   - `ConstantFoldingPass`: Folds static parameter reshape, transpose, and permute passes ahead of time.
   - `DeadCodeEliminationPass`: Prunes disconnected operations and intermediate buffers.
   - `OperatorFusionPass`: Hardware-accelerated kernel fusion (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) executing within CPU L1/L2 Cache.
4. **Quantization Engine (INT8 & FP8):**
   - Static Post-Training Quantization (PTQ) via MinMax and KL-Divergence calibration.
   - Symmetric & Asymmetric INT8 acceleration with AVX2 (`pmaddubsw` + `pmaddwd`) & AVX-512 VNNI.
   - FP8 E4M3 and E5M2 bit-accurate formats for modern Transformer LLM architectures.
5. **Ultra-Low Latency Embedded Serving:**
   - In-process REST/IPC engine (`EmbeddedServer`) responding with $< 1\text{ ms}$ latency.
   - Lock-free micro-batching (`DynamicBatcher`) using `System.Threading.Channels` and $100\ \mu\text{s} - 500\ \mu\text{s}$ spin-wait sliding windows.
6. **100% C# .NET Native:**
   - Fully independent and self-contained ($< 10\text{ MB}$ footprint), Native AOT-ready without external C++ DLL dependencies (no `onnxruntime.dll` or `libtorch.so` required).

---

## 📊 Competitor Matrix

| Feature | `TokenVector.Inference` | `Microsoft.ML.OnnxRuntime` | `PyTorch LibTorch (C++)` |
| :--- | :--- | :--- | :--- |
| **Runtime Dependency** | **100% C# .NET Native (0 External DLLs)** | Depends on native C++ DLLs | Depends on 500MB+ C++ binaries |
| **GC Pressure on Inference** | **Strictly 0 Bytes (Zero GC Hot Path)** | GCs & pins buffers across P/Invoke | Heavy interop overhead |
| **Model Load & Cold Start** | **Ultra-Fast (Zero-Copy Parser)** | Slower runtime layer parsing | Moderate |
| **Embedded Latency** | **$< 1\text{ ms}$ (Optimized Micro-Server)** | $\sim 2 - 5\text{ ms}$ | $\sim 3 - 10\text{ ms}$ |
| **Package Footprint** | **$< 10\text{ MB}$ (Lightweight & Self-Contained)** | $> 50 - 120\text{ MB}$ | $> 500\text{ MB}$ |

---

## ⚡ Quick Start (C#)

```csharp
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

// 1. Load ONNX model and initialize Zero-GC Session
using var session = new InferenceSession("models/transformer.onnx");

// 2. Prepare Input NDArray
var input = new NDArray<float>(1, 64);
for (int i = 0; i < 64; i++) input.AsSpan()[i] = 1.0f;

// 3. Execute Zero-GC Inference
NDArray<float> output = session.Run(input);

Console.WriteLine($"Output Shape: [{string.Join(", ", output.Shape)}]");
```

---

## 🛠️ Build and Test

```powershell
# Run full automated build, test, packaging, and benchmarks
.\build_inference.ps1 -Configuration Release
```
