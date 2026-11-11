# Release Notes — TokenVector.Inference

## v1.0.1 — 2026-11-11

> Bản phát hành `TokenVector.Inference` 1.0.1 (NuGet). Bản này thay cho v1.0.0
> như ghi cuối trang. Các claim "No SIMD", "no companion DLL", "`argv()` stub",
> "`http_listen()` stub" trong phần v1.0.0 bên dưới là **ảnh chụp 2026-09-28, đã lỗi thời**.

**Đóng gói:** package kép: (1) nguồn `.tkv` đầy đủ (`src/tv/`, `src/tools/`,
`src/docs/`, `src/benchmarks`-docs) + (2) managed assembly `lib/net10.0/TokenVector.Inference.dll`
(build từ `.tkv` bằng compiler tu-host; 124+ public types: `ExecutionGraph`,
`InferenceSession`, `AdamW`, `Sampler`, `Tensor`…, assembly `TokenVector.Inference` v1.0.1.0).
**Dependency:** `TokenVector` ≥ 1.0.12 (compiler tu-host + native runtime `tkv_simd_native.dll`).
Package gồm: nguồn `.tkv` (`tv/`), công cụ verify (`tools/`), tài liệu kỹ thuật
(`docs/`), hướng dẫn benchmark (`BENCHMARKING*.md`), hướng dẫn sử dụng
(`USER_GUIDE*.md`). Logo mới (2026-10-18).


**Training (GPT-2 124M, CPU):**
- Full train step @seq1024: **4016 ms** vs PyTorch 2.14.0+cpu cùng phiên **4061.1 ms** → **0.989×** (`benchmark/BENCHMARKS.md` §9.22); loss **EXACT** (`10.760399`).
- Request #14 (COMPILER_TEAM): `adamw_step_glm` nhận thêm tham số `coef` — `opt.step` 781 → ~438–469 ms; không còn đường clip chậm (clip hoạt động hay không đều cùng một call).
- Gate f32 **135/135** (`tools/gate_f32.sh`), `trainer_tests` **15/15**, `gpt2_full_tests` **33/33** (`benchmark/BENCHMARKS.md` §9.21).
- Numerics so với PyTorch: không hứa bit-exact ngoài nội bộ (matmul/reduction lệch 1–3 ulp; xem `docs/numerics-contract.md`). Loss khớp tuyệt đối trên fixture HF vì cùng cấu hình.

**Native & hiệu năng:**
- Hot kernels (GEMM, fused backward, AdamW) P/Invoke `tkv_simd_native.dll` (compiler repo `D:\TokenVector`, AVX2+FMA). Không có mã C/C++ trong repo này; đường scalar `.tkv` vẫn là tham chiếu ngữ nghĩa.
- Compiler tự phát lớp IL `TkvSimd` cho builtin `simd_*`.

**Docs:** README(-VI) viết lại hoàn toàn; `benchmark/BENCHMARKS_VI.md` thay bằng bản tóm tắt hiện hành; các kế hoạch đã xong chuyển vào `docs/archive/`.

**Còn lại:** seq8/seq256 chậm hơn torch một chút (1.067×/1.027×); arena chưa zero-GC; GPU/NPU không có; working set ~7.2 GB @seq1024.

---

## v1.0.0 — 2026-09-28 (historical)


## Release Summary
`TokenVector.Inference` is the pure TokenVector (`.tkv`) graph runtime for the TokenVector
ecosystem: graph optimization, INT8/INT4/FP8 quantization, text-generation helpers, and an
in-process micro-batcher.

## What works
- **Graph optimization passes:** `ConstantFoldingPass`, `QdqFoldPass`, `OperatorFusionPass`
  (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`) and `DeadCodeEliminationPass`, all pure
  `.tkv`. Enable with `session_from_graph(graph, 1)`.
- **Tensor liveness analysis:** live ranges plus scratch-slot reuse in
  `analyze_liveness_and_allocate_offsets()`.
- **Quantization:** static PTQ (MinMax / KL / percentile / MSE), symmetric and asymmetric
  INT8, grouped INT4 (AWQ-style), INT8 GEMM, FP8 E4M3/E5M2, FP16.
- **Protobuf / ONNX wire reader:** pure `.tkv`, little-endian scalars, IEEE-754 float
  reinterpretation, UTF-8 strings, varints.
- **Text stack:** sampling, beam search, speculative decoding, chat templates, tool-call
  parsing, grammar-constrained generation.
- **Training:** autograd forward/backward, gradient check, Adam, LoRA.
- **Formats:** GGUF container with Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K/Q4_K dequant, SafeTensors.
- **Zero native code in this package:** no C/C++, no P/Invoke, no companion DLL.

## What is not ready yet
- **Not zero-GC.** The arena is `dict[i64, list[f32]]` and `native_arena_alloc` is a stub
  returning `0`, so tensors are managed allocations. `docs/SIMD_NOTES.md` §7 has the fix.
- **No SIMD.** `VEC_W = 8` blocked loops are scalar. `docs/SIMD_NOTES.md` has the compiler
  requirements.
- **No ONNX file loading.** `parser_read_bytes` is a stub, so `parse_file` returns nothing.
  `parse_bytes` on an in-memory buffer does work.
- **No HTTP server.** `http_listen()` is a stub returning `0`.
- **No CLI arguments.** `argv()` returns `[]`, so `tv/tools/cli.tkv` always prints usage.
- **No worker threads.** `spawn()` creates no worker, so `DynamicBatcher.enqueue` runs
  inline and does not actually batch.
- **No GPU/NPU backend.** CUDA, OpenCL and OpenVINO are reported as unavailable.

## Verification
`tv/tests/inference_tests.tkv` 26/26 and `tv/tests/gap_tests.tkv` 16/16 as of 2026-09-28,
run with `powershell -File tools\run_cases.ps1`. Known coverage gaps are listed in
`docs/TEST_REPORT.md`.
