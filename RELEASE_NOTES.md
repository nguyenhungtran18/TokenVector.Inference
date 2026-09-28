# TokenVector.Inference Release Notes - Version 1.0.0

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
  returning `0`, so tensors are managed allocations. `SIMD_NOTES.md` §7 has the fix.
- **No SIMD.** `VEC_W = 8` blocked loops are scalar. `SIMD_NOTES.md` has the compiler
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
`TEST_REPORT.md`.
