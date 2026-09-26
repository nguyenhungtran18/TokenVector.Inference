# 🚀 TokenVector.Inference: Động cơ Phục vụ Suy luận Mô hình & Lượng tử hóa Siêu nhẹ

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

<img src="assets/tokenvector-logo.svg" alt="Logo TokenVector" width="96" height="96">

**TokenVector.Inference** là động cơ suy luận mô hình học sâu (Deep Learning Inference Runtime), tối ưu hóa đồ thị tính toán (Graph Optimization Passes), lượng tử hóa đa độ chính xác (INT8 & FP8 Quantization Engine), và phục vụ nhúng siêu nhẹ (Micro-Serving) **thuần 100% TokenVector (`.tkv`)**, đạt chuẩn **Zero-GC Allocation Hot Path** cho hệ sinh thái TokenVector AI.

> **Trạng thái bản phát hành:** 47 module `.tkv` build sạch bằng `tkvc` đi kèm, `tv/tests/gap_tests.tkv` chạy PASS, và bộ test runtime chạy được. Runtime **100% TokenVector, không phụ thuộc native** — không mã C/C++, không `__tkv_extern_pinvoke__`, không DLL kèm theo. CPU khả dụng, TokenVector SIMT là mô phỏng xác định, CUDA/OpenCL/OpenVINO được báo cáo trung thực là unavailable.

### Trạng thái backend

| Backend | Trạng thái |
|---|---|
| CPU | Khả dụng — kernel scalar thuần `.tkv` (`hal.tkv`) |
| TokenVector SIMT | Mô phỏng lane xác định (`simt.tkv`) |
| CUDA | **Unavailable** — runtime thuần TokenVector không có backend GPU |
| OpenCL | **Unavailable** — runtime thuần TokenVector không có backend GPU |
| OpenVINO NPU | **Unavailable** — runtime thuần TokenVector không có backend NPU |

Huấn luyện (`tv/training/autograd.tkv`) chạy trên đường CPU: forward, backward, gradcheck và Adam đều thuần `.tkv`. Nó chưa bao giờ phụ thuộc backend GPU.

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
6. **Runtime thuần TokenVector — không phụ thuộc native:**
   - Compute, graph execution, serving, lượng tử hóa và test là nguồn `.tkv` biên dịch bằng `tkvc`.
   - **Không mã C/C++ trong repo, không `__tkv_extern_pinvoke__`, không DLL kèm theo.** Byte codec và chuyển đổi bit IEEE-754 float được viết bằng `.tkv` (`tv/runtime/f32_bits.tkv`), nên ONNX writer không cần gì nào bên cạnh executable của nó.
   - Muốn có GPU/NPU thì phải đóng gói thành tiện ích native riêng, không phải thư viện mà mọi executable phải mang theo.

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

## Biên dịch, kiểm thử và đóng gói

```powershell
# TokenVector compiler (đi kèm trong gói: compiler/tkvc_patched.exe)
$tkvc = ".\compiler\tkvc_patched.exe"

& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/benchmarks/compare_rivals.tkv
& $tkvc build tv/tests/gap_tests.tkv --out gap_tests.exe
.\gap_tests.exe
```

Biên dịch **toàn bộ** module (nguồn canonical là các file không tên `tv.*.tkv` — đó là bản sao import được sinh ra):

```powershell
$files = Get-ChildItem -Recurse -Filter *.tkv | Where-Object { $_.Name -notmatch '^tv\.' }
foreach ($f in $files) { & $tkvc build $f.FullName --target library --out out.dll }
```

Executable sinh ra là tự chứa: không cần DLL nào nằm bên cạnh.
