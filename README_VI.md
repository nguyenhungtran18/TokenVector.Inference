# 🚀 TokenVector.Inference: Động cơ Phục vụ Suy luận Mô hình & Lượng tử hóa Siêu nhẹ

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

**TokenVector.Inference** là động cơ suy luận mô hình học sâu (Deep Learning Inference Runtime), tối ưu hóa đồ thị tính toán (Graph Optimization Passes), lượng tử hóa đa độ chính xác (INT8 & FP8 Quantization Engine), và phục vụ nhúng siêu nhẹ (Micro-Serving) **thuần 100% TokenVector (`.tkv`)**, đạt chuẩn **Zero-GC Allocation Hot Path** cho hệ sinh thái TokenVector AI.

---

## 🌐 Tổng quan Hệ sinh thái TokenVector AI

**TokenVector** là hệ sinh thái AI và Điện toán Hiệu năng cao (HPC) thế hệ mới dựa trên **ngôn ngữ TokenVector (`.tkv`)** và **trình biên dịch `tkvc`**, hướng tới suy luận CPU hiệu năng cao với **0% chi phí thu gom rác (Zero-GC)**:

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

### Các Thành phần Cốt lõi trong Hệ sinh thái:
1. **[`TokenVector`](https://github.com/nguyenhungtran18/TokenVector)**: Ngôn ngữ lập trình và trình biên dịch TokenVector chính thức (`.tkv`, `tkvc`) với thư viện chuẩn (`stdlib/tv/inference`).
2. **[`TokenVector.Numerics`](https://github.com/nguyenhungtran18/TokenVector.Numerics)**: Hạt nhân đại số tuyến tính và mảng đa chiều (`NDArray<T>`) với tương tác zero-copy.
3. **[`TokenVector.Vision`](https://github.com/nguyenhungtran18/TokenVector.Vision)**: Động cơ thị giác máy tính 2D & 3D Voxel, biến đổi ảnh fused, YOLO Letterbox, NMS và ImagePainter.
4. **[`TokenVector.Text`](https://github.com/nguyenhungtran18/TokenVector.Text)**: Xử lý văn bản tốc độ cao, tách từ (BPE, WordPiece, SentencePiece), embeddings và pipeline LLM.
5. **[`TokenVector.Data`](https://github.com/nguyenhungtran18/TokenVector.Data)**: Pipeline prefetching đa luồng bộ đệm kép loại bỏ nút thắt I/O.
6. **[`TokenVector.Inference`](https://github.com/nguyenhungtran18/TokenVector.Inference)**: Động cơ suy luận độ trễ cực thấp, pass tối ưu đồ thị, lượng tử hóa INT8/FP8 và micro-serving nhúng — **100% `.tkv`**.

---

## 🌟 Các Tính năng Nổi bật

1. **Hot Path Tuyệt đối Không Sinh Rác (Zero-GC Hot Path):**
   - Cấp phát vùng nhớ arena thống nhất theo **Phân tích Vòng đời Tensor (Tensor Liveness Analysis)**.
   - Đường hot path hướng tới **0 Byte** phân bổ heap nhờ tái sử dụng arena.
2. **Bộ Parser ONNX & Protobuf Nhị phân Zero-Copy:**
   - Reader protobuf thuần `.tkv` giải mã ONNX từ danh sách byte thô, không tạo chuỗi/đối tượng trung gian.
3. **Pipeline Tối ưu hóa Đồ thị Tính toán:**
   - `ConstantFoldingPass`: Gộp trước Reshape/Transpose/Permute tĩnh vào trọng số.
   - `DeadCodeEliminationPass`: Cắt tỉa node và buffer không tham gia tạo output.
   - `OperatorFusionPass`: Gộp kernel (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) tận dụng cache.
4. **Động cơ Lượng tử hóa Đa độ chính xác (INT8 & FP8):**
   - PTQ tĩnh (MinMax, KL, percentile, MSE) và lượng tử hóa động.
   - INT8 đối xứng/asymmetric, INT4 grouped (kiểu AWQ), GEMM per-channel.
   - FP8 `E4M3`/`E5M2` và đường compute FP16 cho kiến trúc Transformer.
5. **Bộ phục vụ Nhúng Siêu nhẹ & Batching:**
   - REST nhúng (`EmbeddedServer`) API tương thích OpenAI, SSE, auth, limits, Prometheus metrics.
   - Micro-batching / continuous batching trên pool session, paged KV + prefix skip.
6. **100% TokenVector (`.tkv`):**
   - Toàn bộ engine và test là nguồn `.tkv` biên dịch bằng `tkvc`; không cần DLL native ngoài cho đường lõi.

---

## 📊 Bộ Benchmark

Số latency/throughput **phải đo trên engine `.tkv` này** — không trích số của backend cũ.

```powershell
tkvc build tv/benchmarks/compare_rivals.tkv
```

Xem `BENCHMARKS.md` / `BENCHMARKS_VI.md` (cấu trúc suite) và `COMPETITOR_COMPARISON.md` (so sánh chức năng, không so tốc độ).

---

## ⚡ Hướng dẫn Sử dụng Nhanh (`.tkv`)

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/model.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output shape:", output_tensor.shape)
session.close()
```

CLI:

```powershell
tkvc build tv/tools/cli.tkv
# chạy file .exe sinh ra với serve|gen|export|calibrate|latency
```

---

## 🛠️ Biên dịch, Kiểm thử và Đóng gói

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/tests/inference_tests.tkv
tkvc build tv/benchmarks/compare_rivals.tkv
tkvc build stdlib/tv/inference/inference.tkv --target library
```
