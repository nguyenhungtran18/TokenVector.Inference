# TokenVector.Inference Competitor Benchmark & Stress Test Report

## 1. Executive Summary
Benchmark and stress suites target the pure **TokenVector (`.tkv`)** engine only. Results must be measured with the `tkvc` toolchain on the hardware under test — **do not quote numbers from any previous backend**.

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/benchmarks/compare_rivals.tkv
```

Hardware note: x86-64 with AVX2/FMA recommended for the blocked SIMD kernels in `tv/optimization/kernels.tkv` and `tv/quantization/matmul.tkv`.

---

## 2. Suite Structure (compare_rivals.tkv)

| Scenario | What it measures |
| :--- | :--- |
| Cold-start & model load | ONNX parse + arena setup |
| In-process single-sample latency | Fused linear / elementwise kernels |
| Arena reuse / allocation pressure | Hot-path reuse across many runs |
| INT8 vs FP32 throughput | Quantized GEMM vs float reference |
| Hot-path endurance | Long continuous inference loop |
| Concurrent serving | Multi-worker burst through embedded server |
| Massive layer projection | Large GEMM effective GFLOPS |
| Deep DAG | Multi-layer fused pipeline latency |

Regression gates live in the same suite; fail the gate rather than publish a stale number.

---

## 3. Recording Results

When you run the suite, append a dated table here with:

- Host CPU model and flags (AVX2/FMA)
- `tkvc` build id / commit hash
- Scenario, metric, median ± spread
- Rival baseline command lines for the same machine (ONNX Runtime, llama.cpp, PyTorch) if compared

Until a fresh run is recorded, leave result cells empty rather than copying historical figures.
