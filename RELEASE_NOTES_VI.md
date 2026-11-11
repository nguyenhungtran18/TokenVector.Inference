# Release Notes — TokenVector.Inference (Tiếng Việt)

## 2026-11-11 — trạng thái hiện tại (chưa đánh số bản phát hành)

> Bản ghi này thay cho việc phát hành v1.0.0 như ghi cuối trang. Các claim
> "Chưa có SIMD", "không DLL kèm theo", "`argv()` stub", "`http_listen()` stub"
> trong phần v1.0.0 bên dưới là **ảnh chụp 2026-09-28, đã lỗi thời**.

**Huấn luyện (GPT-2 124M, CPU):**
- Full train step @seq1024: **4016 ms** so với PyTorch 2.14.0+cpu cùng phiên **4061.1 ms** → **0.989×** (`benchmark/BENCHMARKS.md` §9.22); loss **EXACT** (`10.760399`).
- Request #14 (COMPILER_TEAM): `adamw_step_glm` nhận thêm tham số `coef` — `opt.step` 781 → ~438–469 ms; không còn đường clip chậm.
- Gate f32 **135/135** (`tools/gate_f32.sh`), `trainer_tests` **15/15**, `gpt2_full_tests` **33/33** (`benchmark/BENCHMARKS.md` §9.21).
- Numerics so với PyTorch: không hứa bit-exact ngoài nội bộ (matmul/reduction lệch 1–3 ulp; xem `docs/numerics-contract.md`). Loss khớp tuyệt đối trên fixture HF vì cùng cấu hình.

**Native & hiệu năng:**
- Hot kernels (GEMM, fused backward, AdamW) P/Invoke `tkv_simd_native.dll` (compiler repo `D:\TokenVector`, AVX2+FMA). Không có mã C/C++ trong repo này; đường scalar `.tkv` vẫn là tham chiếu ngữ nghĩa.
- Compiler tự phát lớp IL `TkvSimd` cho builtin `simd_*`.

**Docs:** README(-VI) viết lại hoàn toàn; `benchmark/BENCHMARKS_VI.md` thay bằng bản tóm tắt hiện hành; các kế hoạch đã xong chuyển vào `docs/archive/`.

**Còn lại:** seq8/seq256 chậm hơn torch một chút (1.067×/1.027×); arena chưa zero-GC; GPU/NPU không có; working set ~7.2 GB @seq1024.

---

## v1.0.0 — 2026-09-28 (historical)


## Tổng quan Phát hành
`TokenVector.Inference` là runtime đồ thị thuần TokenVector (`.tkv`) cho hệ sinh thái
TokenVector: tối ưu hóa đồ thị, lượng tử hóa INT8/INT4/FP8, các tiện ích sinh văn bản,
và micro-batcher trong process.

## Phần đang chạy
- **Các pass tối ưu hóa đồ thị:** `ConstantFoldingPass`, `QdqFoldPass`,
  `OperatorFusionPass` (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) và
  `DeadCodeEliminationPass`, tất cả thuần `.tkv`. Bật bằng `session_from_graph(graph, 1)`.
- **Phân tích vòng đời tensor:** tính khoảng sống và tái dùng slot scratch trong
  `analyze_liveness_and_allocate_offsets()`.
- **Lượng tử hóa:** PTQ tĩnh (MinMax / KL / percentile / MSE), INT8 đối xứng và
  asymmetric, INT4 grouped (kiểu AWQ), GEMM INT8, FP8 E4M3/E5M2, FP16.
- **Reader protobuf / ONNX:** thuần `.tkv`, scalar little-endian, tái hiện bit IEEE-754,
  chuỗi UTF-8, varint.
- **Stack xử lý văn bản:** sampling, beam search, speculative decoding, chat template,
  phân tích tool-call, sinh văn bản ràng buộc grammar.
- **Training:** autograd forward/backward, gradient check, Adam, LoRA.
- **Định dạng:** container GGUF với dequant Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K/Q4_K, SafeTensors.
- **Không có mã native trong gói này:** không C/C++, không P/Invoke, không DLL kèm theo.

## Phần chưa sẵn sàng
- **Chưa zero-GC.** Arena là `dict[i64, list[f32]]` và `native_arena_alloc` là stub trả
  `0`, nên tensor là phân bổ managed. `docs/SIMD_NOTES.md` §7 có cách sửa.
- **Chưa có SIMD.** Các vòng khối `VEC_W = 8` vẫn chạy scalar. `docs/SIMD_NOTES.md` nêu yêu
  cầu điều chỉnh compiler.
- **Chưa đọc được file ONNX.** `parser_read_bytes` là stub nên `parse_file` không trả
  gì. `parse_bytes` trên buffer trong bộ nhớ thì chạy được.
- **Chưa có HTTP server.** `http_listen()` là stub trả `0`.
- **Chưa nhận được tham số CLI.** `argv()` trả `[]`, nên `tv/tools/cli.tkv` luôn chỉ in usage.
- **Chưa có worker thread.** `spawn()` không tạo worker, nên `DynamicBatcher.enqueue`
  chạy inline và thực chất không gộp batch.
- **Chưa có backend GPU/NPU.** CUDA, OpenCL và OpenVINO được báo là unavailable.

## Xác minh
`tv/tests/inference_tests.tkv` 26/26 và `tv/tests/gap_tests.tkv` 16/16 tại 2026-09-28, chạy
bằng `powershell -File tools\run_cases.ps1`. Các lỗ hổng phủ biết liệt kê trong
`docs/TEST_REPORT_VI.md`.
