# 🚀 TokenVector.Inference: Động cơ Phục vụ Suy luận Mô hình & Lượng tử hóa Siêu nhẹ

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Pure TokenVector](https://img.shields.io/badge/Backend-100%25%20.tkv-brightgreen.svg)]()

<img src="assets/tokenvector-logo.svg" alt="Logo TokenVector" width="96" height="96">

**TokenVector.Inference** là động cơ runtime đồ thị học sâu thuần **TokenVector (`.tkv`)** — tối ưu hóa đồ thị, lượng tử hóa đa độ chính xác, và micro-batcher — xây cho hệ sinh thái TokenVector AI.

> **Trạng thái, đã kiểm chứng 2026-09-28:** 26/26 case trong `tv/tests/inference_tests.tkv` và 16/16 trong `tv/tests/gap_tests.tkv` đều PASS. Runtime **100% TokenVector, không có mã native trong gói này** — không C/C++, không P/Invoke, không DLL kèm theo. CPU chạy được, TokenVector SIMT là mô phỏng xác định, CUDA/OpenCL/OpenVINO được báo cáo trung thực là unavailable.
>
> Hãy đọc các giới hạn trước khi dựa vào nó: arena tensor **chưa** zero-GC (xem bên dưới), **không có SIMD**, và nhiều entry point vẫn là stub. `HANDOFF.md` liệt kê mọi lỗ hổng đã biết; `SIMD_NOTES.md` nói về vector hoá.

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

1. **Phân tích vòng đời tensor** (tái sử dụng arena, *chưa* zero-GC):
   - `ExecutionGraph.analyze_liveness_and_allocate_offsets()` tính khoảng sống và tái dùng slot scratch.
   - Arena là `dict[i64, list[f32]]` — lưu trữ phần tử dạng hộp, và `native_arena_alloc` vẫn là stub trả `0`. Nên mỗi tensor vẫn tốn phân bổ managed. Mục tiêu 0 byte ở hot path **chưa** đạt; xem `SIMD_NOTES.md` §7.
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
5. **Cấu trúc phục vụ trong process (chưa listen):**
   - `EmbeddedServer` và `DynamicBatcher` đã có, và đường đồng bộ của batcher chạy được; nhưng `http_listen()` là stub trả `0` và `spawn()` không tạo worker thread, nên không có socket nào được bind và không có micro-batching đa luồng.
   - Paged KV + prefix skip đã hiện thực và có test.
6. **Runtime thuần TokenVector — không có mã native trong gói này:**
   - Compute, graph execution, lượng tử hóa và test là nguồn `.tkv` biên dịch bằng `tkvc`.
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

Ví dụ chạy được — `tv/examples/quickstart.tkv` dựng graph, tạo session và chạy
một forward pass:

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
& $tkvc build tv/examples/quickstart.tkv --out quickstart.exe
.\quickstart.exe      # in ra: Y[0] = 2.25  /  quickstart: OK
```

API làm việc trên **graph**; không có `tv.inference.load_onnx()` và không có `NDArray`:

```tokenvector
import tv.runtime.session
# Hằng số cấp module KHÔNG import được - phải khai báo lại (giá trị lấy từ
# tv/runtime/session.tkv và tv/optimization/kernels.tkv).
OperatorType_FusedLinear = 2
ActivationFunction_Relu = 1

g = ExecutionGraph(1, "quickstart", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w          # list[f32] dài n*k
g.initializers["B"] = b          # list[f32] dài n
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("W", [8, 4])
g.get_or_create_tensor("B", [8])
g.get_or_create_tensor("Y", [1, 8])

node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")          # node.inputs.append("W"), append("B")
node.outputs.append("Y")
g.nodes.append(node)

session = session_from_graph(g, 1)   # 1 = chạy các pass tối ưu
y = session.run_single(x, "X", "Y")  # x là list[f32] phẳng
session.close()
```

Những thứ **chưa** hoạt động (là stub, đừng đưa vào hướng dẫn nhanh):

| API | Trạng thái |
| :--- | :--- |
| `tv.inference.load_onnx()` | `tv.inference` là stub tự sinh; module này không tồn tại |
| `tensor.zeros(...)` | không có ký hiệu này; tensor là `list[f32]` |
| `parse_file("m.onnx")` | `parser_read_bytes` là stub trả `""`, chưa implement đọc file |
| `cli.tkv serve ...` | `argv()` là stub trả `[]`, CLI luôn chỉ in usage |
| `http_listen()` (embedded server) | stub trả `0`; không có socket nào được bind |

---

## Biên dịch, kiểm thử và đóng gói

```powershell
# TokenVector compiler
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"

# Test: 26 case (mỗi case một process riêng), rồi 16 case của gap suite
powershell -File tools\run_cases.ps1 -Compiler $tkvc
#   TOTAL: pass=26 fail=0 crash=0
#   gap tests: PASS (16/16)

# Build + test + benchmark trong một lệnh
powershell -File build_inference.ps1 -SkipBenchmarks
```

**Đừng biên dịch từng module một.** `tkvc build <một module>` sẽ ghi lại các bản
sao import dạng `tv.*.tkv`, và đã quan sát thấy 9 file bị ghi đầy byte NUL (khôi
phục bằng `git checkout HEAD -- <file>`). Chỉ build entry point:

```powershell
& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/tests/gap_tests.tkv        --out gap_tests.exe
& $tkvc build tv/examples/quickstart.tkv    --out quickstart.exe
& $tkvc build tv/benchmarks/compare_rivals.tkv --out compare_rivals.exe
```

Nguồn canonical là các file **không** tên `tv.*.tkv` — đó là bản sao import được
sinh ra; sau khi sửa file canonical phải copy sang mọi `tv.*.tkv` tương ứng.

Executable sinh ra là tự chứa: không cần DLL nào nằm bên cạnh.
