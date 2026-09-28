# Ghi chú Phát hành TokenVector.Inference - Phiên bản 1.0.0 (Tiếng Việt)

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
  `0`, nên tensor là phân bổ managed. `SIMD_NOTES.md` §7 có cách sửa.
- **Chưa có SIMD.** Các vòng khối `VEC_W = 8` vẫn chạy scalar. `SIMD_NOTES.md` nêu yêu
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
`TEST_REPORT_VI.md`.
