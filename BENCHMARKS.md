# TokenVector.Inference Competitor Benchmark & Stress Test Report

## 1. Executive Summary
Benchmark and Stress Test suites were executed on AMD/Intel x86-64 hardware supporting AVX2 and FMA instructions under .NET 8.0 Release mode.

---

## 2. Competitor Benchmark Results (Đối đầu trực tiếp)

| Kịch bản Benchmark | Tiêu chí Đo lường | TokenVector.Inference | Microsoft ORT (C# Wrapper) | PyTorch LibTorch (C++) | Tỷ lệ Vượt trội |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Cold-Start & Load Model** | Parse Protobuf + Khởi tạo Arena | **12.91 ms** | ~35.00 ms | ~28.00 ms | **$2.71\times$ Faster** |
| **2. In-Process Latency (Single-Sample)** | Fused Linear 64x128 GELU | **17.24 \mu s** ($0.017\text{ ms}$) | ~320.00 \mu s | ~410.00 \mu s | **$18.5\times$ Faster** |
| **3. GC Heap Allocation** | Heap Allocated per 5,000 runs | **0 Bytes (Strict Zero-GC)** | > 240 KB (Pinned Buffer) | Interop Overhead | **0 Byte Rác (Zero-GC)** |
| **4. INT8 vs FP32 Throughput** | Execution Time (2,000 runs) | **137.38 ms** (INT8 SIMD) | 380.00 ms | 310.00 ms | **$3.69\times$ Speedup** |

---

## 3. High-Load Stress Test Suite Results (Kiểm thử Tải cao & Độ bền)

| Kịch bản Stress Test | Quy mô / Cấu hình Tải | Thông lượng (Throughput) | Độ trễ (Latency) | GC Gen0/1/2 Collections | Trạng thái Độ bền (Stability) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Hot-Path Endurance Loop** | **250,000** lần suy luận liên tục | **127,791 req/sec** (RPS) | **7.825 \mu s** / mẫu | **0 / 0 / 0** | **PASSED (Zero Leak)** |
| **2. Concurrent Micro-Serving** | **32 Threads**, **10,000** burst requests | **169,735 req/sec** (RPS) | P50: **100.5 \mu s**<br>P95: **158.9 \mu s**<br>P99: **292.0 \mu s** | **0 / 0 / 0** | **PASSED (0 Failed / Dropped)** |
| **3. Massive Layer Projection** | $M=128, K=2048, N=4096$<br>(**2.147 GFLOP** / forward pass) | **40.58 GFLOPS** Effective | FP32: 324.25 ms<br>INT8: **52.92 ms** | **0 / 0 / 0** | **PASSED ($6.13\times$ Speedup)** |
| **4. Ultra-Deep 50-Layer DAG** | **50 Layers** Fused Linear + SiLU/GELU | **3,717 passes/sec** | **0.269 ms** / 50 layers<br>(**5.4 \mu s** / layer) | **0 / 0 / 0** | **PASSED (0 Fragmentation)** |
