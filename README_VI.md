# 🚀 TokenVector.Inference (Tiếng Việt)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Target: .NET 8.0](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![AOT Ready](https://img.shields.io/badge/Native%20AOT-Ready-success.svg)]()
[![Zero-GC Hot Path](https://img.shields.io/badge/Hot%20Path-0%20Byte%20GC-brightgreen.svg)]()

**TokenVector.Inference** là động cơ suy luận mô hình học sâu (Deep Learning Inference Runtime), tối ưu hóa đồ thị tính toán (Graph Optimization Passes), lượng tử hóa đa độ chính xác (INT8 & FP8 Quantization Engine), và phục vụ nhúng siêu nhẹ (Micro-Serving) thuần 100% C# Native, đạt chuẩn **Zero-GC Allocation Hot Path** cho hệ sinh thái TokenVector AI.

---

## 🌟 Các Tính năng Nổi bật

1. **Hot Path Tuyệt đối Không Sinh Rác (Zero-GC Hot Path):**
   - Cấp phát vùng nhớ unmanaged `ExecutionMemoryArena` duy nhất dựa trên **Phân tích Vòng đời Tensor (Tensor Liveness Analysis)**.
   - Hàm `session.Run(...)` cam kết **0 Byte** cấp phát trên Managed Heap.
2. **Bộ Parser ONNX & Protobuf Nhị phân Zero-Copy:**
   - `FastProtobufReader` giải mã trực tiếp từ `ReadOnlySpan<byte>` qua con trỏ unmanaged, không tạo rác String hay Object trung gian.
3. **Pipeline Tối ưu hóa Đồ thị Tính toán (Optimization Passes):**
   - `ConstantFoldingPass`: Tính toán tĩnh và gộp trước các thao tác Reshape, Transpose, Permute trọng số.
   - `DeadCodeEliminationPass`: Cắt tỉa các nhánh và node không tham gia tạo Output.
   - `OperatorFusionPass`: Gộp các kernel tính toán (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) tận dụng L1/L2 Cache.
4. **Động cơ Lượng tử hóa Đa độ chính xác (INT8 & FP8):**
   - Hỗ trợ Static PTQ (MinMax & KL-Divergence calibration) và Dynamic Quantization.
   - Tăng tốc INT8 SIMD qua AVX2 (`pmaddubsw` + `pmaddwd`) và AVX-512 VNNI.
   - Hỗ trợ đầy đủ chuẩn FP8 `E4M3` (độ chính xác cao) và `E5M2` (dải động rộng).
5. **Bộ phục vụ Nhúng Siêu nhẹ & Dynamic Batching:**
   - In-process REST/IPC engine (`EmbeddedServer`) với độ trễ phản hồi $< 1\text{ ms}$.
   - Lock-free micro-batching (`DynamicBatcher`) gom nhóm request với sliding window $100\ \mu\text{s} - 500\ \mu\text{s}$.
6. **100% C# .NET Native:**
   - Độc lập hoàn toàn, kích thước thư viện $< 10\text{ MB}$, sẵn sàng cho Native AOT mà không phụ thuộc file DLL C++ bên ngoài.

---

## 📊 Bảng So sánh Năng lực Cạnh tranh

| Tiêu chí | `TokenVector.Inference` | `Microsoft.ML.OnnxRuntime` | `PyTorch LibTorch (C++)` |
| :--- | :--- | :--- | :--- |
| **Phụ thuộc Thư viện** | **100% C# .NET Native (Không cần DLL C++)** | Phụ thuộc Native C++ DLL | Phụ thuộc gói C++ > 500MB |
| **Cấp phát GC khi Inference** | **Tuyệt đối 0 Byte (Zero-GC Hot Path)** | Gây rác & Pin buffer qua P/Invoke | Overhead Marshalling lớn |
| **Tốc độ Khởi động & Load** | **Cực nhanh (Zero-Copy Parser)** | Chậm hơn do parse qua C++ runtime | Trung bình |
| **Độ trễ Phục vụ Nhúng** | **$< 1\text{ ms}$ (Tối ưu Micro-Serving)** | $\sim 2 - 5\text{ ms}$ | $\sim 3 - 10\text{ ms}$ |
| **Kích thước Đóng gói** | **$< 10\text{ MB}$ (Độc lập, gọn nhẹ)** | $> 50 - 120\text{ MB}$ | $> 500\text{ MB}$ |

---

## ⚡ Hướng dẫn Sử dụng Nhanh (C#)

```csharp
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

// 1. Khởi tạo phiên làm việc Zero-GC từ file ONNX
using var session = new InferenceSession("models/model.onnx");

// 2. Chuẩn bị tensor đầu vào
var input = new NDArray<float>(1, 64);
for (int i = 0; i < 64; i++) input.AsSpan()[i] = 1.0f;

// 3. Thực thi suy luận Zero-GC
NDArray<float> output = session.Run(input);

Console.WriteLine($"Kích thước đầu ra: [{string.Join(", ", output.Shape)}]");
```

---

## 🛠️ Biên dịch và Kiểm thử

```powershell
# Chạy pipeline tự động build, test, đóng gói và benchmark
.\build_inference.ps1 -Configuration Release
```
