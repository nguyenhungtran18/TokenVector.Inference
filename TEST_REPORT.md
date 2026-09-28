# TokenVector.Inference Test Suite Report

## 1. Summary
- **Suites:** `tv/tests/inference_tests.tkv` (26 cases) and `tv/tests/gap_tests.tkv` (16 cases)
- **Target language:** TokenVector (`.tkv`)
- **Runner:** `powershell -File tools\run_cases.ps1`
- **Last verified:** 2026-09-28 — 26/26 pass, 16/16 pass, 0 crash

Each of the 26 cases runs in its **own process**. An `AccessViolationException` cannot be
caught in TokenVector, so a shared process would lose every case after the first crash.
`gap_tests` runs as a second process after the main suite; `run_cases.ps1` exits non-zero
if either suite fails.

## 2. What the 26 cases cover

| Area | Cases |
| :--- | :--- |
| Text generation | `generate_loop_text`, `grammar_prefix_mask`, `beam_log_softmax_tree`, `prefix_skip_plan`, `speculative_all_accept`, `tools_parse_validate` |
| ONNX parse / translate / write | `writer_modelproto`, `constant_fold_translate`, `gathernd_scatter`, `range_eye_cumsum`, `mish_prelu_instancenorm`, `resize_nearest_einsum`, `qlinear_conv_matmulinteger`, `lstm_unrolled`, `if_loop_end_to_end`, `scan_run_driver`, `pack_mask_and_preprocess`, `vit_graph_builds` |
| Quantization | `awq_asym_and_fp16`, `gguf_k_quant_gate`, `hal_matmul_relu` |
| Training / adapters | `autograd_trains_step`, `autograd_gradcheck_and_adam`, `lora_forward_and_step` |
| String tensors | `string_tensor_pack`, `string_ops_side_table` |

## 3. What the 16 gap cases cover

Added when the C# surface was ported: protobuf varint / IEEE float / UTF-8 string, MinMax
calibration, INT8 round-trip, FP8 E4M3 + E5M2 round-trip, batcher FIFO + empty sentinel,
SIMT matmul, dynamic-shape specialization, ViT weight remap, backend dispatch, gradient
clipping, plus four regression tests for hangs and for the optimization passes.

## 4. Known coverage gaps

- **No test asserts numbers through an optimized session except the two gap
  regressions.** Every other case builds its session with `session_from_graph(g, 0)`,
  which skips the optimization passes. A bug in a pass therefore passes the whole suite.
  Assert numeric results, not just `len(y)`, in any new session-level test.
- No MLP / ResNet / LLaMA-layer / 50-layer depth cases — those existed only in the deleted
  C# suite and have not been ported.
- No concurrency, endurance or zero-GC allocation case: `spawn()` creates no worker, so
  the batcher runs inline and cannot be stressed across threads.
- No test for the `raw_data` decoding path in `tv/onnx/models.tkv`.

## 5. How to run

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
powershell -File tools\run_cases.ps1 -Compiler $tkvc
# or build the suites directly
& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/tests/gap_tests.tkv        --out gap_tests.exe
```

Paste dated results (counts, duration, commit hash) below when a full run is captured.
Do not invent pass counts.
