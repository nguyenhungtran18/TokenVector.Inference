## 1.0.3 (2026-11-27)

- Merge toan bo CUDA/GPT2-medium trainer tu `TokenVector.CudaBackend` vao `cuda-backend/`.
- Ket qua verified: 176 ms/step tren Kaggle T4, nhanh hon bridge C++ 187.8 ms (~6%),
  loss hoi tu 11.18 -> 1.25 qua 10 buoc, khong C++/torch trong duong chay.
- Don dep cac file WIP/probe, cap nhat notebook Kaggle va docs.
# Release Notes â€” TokenVector.Inference

## v1.0.1 â€” 2026-11-11

> Báº£n phÃ¡t hÃ nh `TokenVector.Inference` 1.0.1 (NuGet). Báº£n nÃ y thay cho v1.0.0
> nhÆ° ghi cuá»‘i trang. CÃ¡c claim "No SIMD", "no companion DLL", "`argv()` stub",
> "`http_listen()` stub" trong pháº§n v1.0.0 bÃªn dÆ°á»›i lÃ  **áº£nh chá»¥p 2026-09-28, Ä‘Ã£ lá»—i thá»i**.

**ÄÃ³ng gÃ³i:** package kÃ©p: (1) nguá»“n `.tkv` Ä‘áº§y Ä‘á»§ (`src/tv/`, `src/tools/`,
`src/docs/`, `src/benchmarks`-docs) + (2) managed assembly `lib/net10.0/TokenVector.Inference.dll`
(build tá»« `.tkv` báº±ng compiler tu-host; 124+ public types: `ExecutionGraph`,
`InferenceSession`, `AdamW`, `Sampler`, `Tensor`â€¦, assembly `TokenVector.Inference` v1.0.1.0).
**Dependency:** `TokenVector` â‰¥ 1.0.12 (compiler tu-host + native runtime `tkv_simd_native.dll`).
Package gá»“m: nguá»“n `.tkv` (`tv/`), cÃ´ng cá»¥ verify (`tools/`), tÃ i liá»‡u ká»¹ thuáº­t
(`docs/`), hÆ°á»›ng dáº«n benchmark (`BENCHMARKING*.md`), hÆ°á»›ng dáº«n sá»­ dá»¥ng
(`USER_GUIDE*.md`). Logo má»›i (2026-10-18).


**Training (GPT-2 124M, CPU):**
- Full train step @seq1024: **4016 ms** vs PyTorch 2.14.0+cpu cÃ¹ng phiÃªn **4061.1 ms** â†’ **0.989Ã—** (`benchmark/BENCHMARKS.md` Â§9.22); loss **EXACT** (`10.760399`).
- Request #14 (COMPILER_TEAM): `adamw_step_glm` nháº­n thÃªm tham sá»‘ `coef` â€” `opt.step` 781 â†’ ~438â€“469 ms; khÃ´ng cÃ²n Ä‘Æ°á»ng clip cháº­m (clip hoáº¡t Ä‘á»™ng hay khÃ´ng Ä‘á»u cÃ¹ng má»™t call).
- Gate f32 **135/135** (`tools/gate_f32.sh`), `trainer_tests` **15/15**, `gpt2_full_tests` **33/33** (`benchmark/BENCHMARKS.md` Â§9.21).
- Numerics so vá»›i PyTorch: khÃ´ng há»©a bit-exact ngoÃ i ná»™i bá»™ (matmul/reduction lá»‡ch 1â€“3 ulp; xem `docs/numerics-contract.md`). Loss khá»›p tuyá»‡t Ä‘á»‘i trÃªn fixture HF vÃ¬ cÃ¹ng cáº¥u hÃ¬nh.

**Native & hiá»‡u nÄƒng:**
- Hot kernels (GEMM, fused backward, AdamW) P/Invoke `tkv_simd_native.dll` (compiler repo `D:\TokenVector`, AVX2+FMA). KhÃ´ng cÃ³ mÃ£ C/C++ trong repo nÃ y; Ä‘Æ°á»ng scalar `.tkv` váº«n lÃ  tham chiáº¿u ngá»¯ nghÄ©a.
- Compiler tá»± phÃ¡t lá»›p IL `TkvSimd` cho builtin `simd_*`.

**Docs:** README(-VI) viáº¿t láº¡i hoÃ n toÃ n; `benchmark/BENCHMARKS_VI.md` thay báº±ng báº£n tÃ³m táº¯t hiá»‡n hÃ nh; cÃ¡c káº¿ hoáº¡ch Ä‘Ã£ xong chuyá»ƒn vÃ o `docs/archive/`.

**CÃ²n láº¡i:** seq8/seq256 cháº­m hÆ¡n torch má»™t chÃºt (1.067Ã—/1.027Ã—); arena chÆ°a zero-GC; GPU/NPU khÃ´ng cÃ³; working set ~7.2 GB @seq1024.

---

## v1.0.0 â€” 2026-09-28 (historical)


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
  returning `0`, so tensors are managed allocations. `docs/SIMD_NOTES.md` Â§7 has the fix.
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

