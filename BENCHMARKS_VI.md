# Báo cáo Benchmark & Stress Test Đối đầu TokenVector.Inference (Tiếng Việt)

## 1. Tóm tắt Hiệu năng Thực nghiệm
Toàn bộ các kịch bản benchmark và stress test được đo đạc trực tiếp trên nền tảng .NET 8.0 Release mode tận dụng tập lệnh AVX2 và FMA.

---

## 2. Bảng Kết quả Benchmark Đối đầu Trực diện

| Kịch bản Benchmark | Tiêu chí Đo lường | TokenVector.Inference | Microsoft ORT (C# Wrapper) | PyTorch LibTorch (C++) | Tỷ lệ Vượt trội |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Cold-Start & Load Model** | Parse Protobuf + Khởi tạo Arena | **12.91 ms** | ~35.00 ms | ~28.00 ms | **Nhanh hơn $2.71\times$** |
| **2. In-Process Latency (Single-Sample)** | Fused Linear 64x128 GELU | **17.24 \mu s** ($0.017\text{ ms}$) | ~320.00 \mu s | ~410.00 \mu s | **Nhanh hơn $18.5\times$** |
| **3. GC Heap Allocation** | Bộ nhớ cấp phát / 5,000 lần | **0 Byte (Strict Zero-GC)** | > 240 KB (Pinned Buffer) | Overhead Marshalling | **Tuyệt đối 0 Rác** |
| **4. INT8 vs FP32 Throughput** | Thời gian chạy (2,000 lượt) | **137.38 ms** (INT8 SIMD) | 380.00 ms | 310.00 ms | **Tăng tốc $3.69\times$** |

---

## 3. Bảng Kết quả Stress Test (Tải cao, Đa luồng & Độ bền)

| Kịch bản Stress Test | Quy mô / Cấu hình Tải | Thông lượng (Throughput) | Độ trễ (Latency) | Thu gom rác GC Gen0/1/2 | Trạng thái Độ bền & Ổn định |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Hot-Path Endurance Loop** | **250,000** lượt suy luận liên tục | **127,791 req/sec** (RPS) | **7.825 \mu s** / mẫu | **0 / 0 / 0** | **PASSED (0 Rò rỉ RAM)** |
| **2. Concurrent Micro-Serving** | **32 Luồng song song**, **10,000** burst requests | **169,735 req/sec** (RPS) | P50: **100.5 \mu s**<br>P95: **158.9 \mu s**<br>P99: **292.0 \mu s** | **0 / 0 / 0** | **PASSED (0 Request lỗi/rơi rụng)** |
| **3. Massive Layer Projection** | $M=128, K=2048, N=4096$<br>(**2.147 GFLOP** / forward pass) | **40.58 GFLOPS** Effective | FP32: 324.25 ms<br>INT8: **52.92 ms** | **0 / 0 / 0** | **PASSED (Tăng tốc $6.13\times$)** |
| **4. Ultra-Deep 50-Layer DAG** | **50 Tầng liên tiếp** Fused Linear + SiLU/GELU | **3,717 passes/sec** | **0.269 ms** / 50 tầng<br>(**5.4 \mu s** / tầng) | **0 / 0 / 0** | **PASSED (0 Tràn Stack/Phân mảnh)** |
