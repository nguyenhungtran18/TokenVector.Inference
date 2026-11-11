# TokenVector.Inference Test Suite Report

> **2026-11-11 status:** gates that matter today are `tools/gate_f32.sh`
> (**135/135**, 10 f32 suites), `tv/tests/trainer_tests.tkv` (**15/15**),
> `tv/tests/gpt2_full_tests.tkv` (**33/33**, needs the out-of-repo 124M
> fixture — `python tools/gen_gpt2_fixture.py --full`), and
> `tools/run_cases.ps1` (**22 pass / 4 fail — identical to baseline**; the
> 4 failures are `System.Runtime.Intrinsics` missing under .NET Framework;
> `gap_tests` PASS when run under .NET 10, as `gate_f32.sh` does).
> Loss EXACT on all three sequences, determinism verified bit-for-bit.
> Details: `BENCHMARKS.md` §9.21. The 26/16 suites below remain from the
> 2026-09-28 port and are unchanged; `run_cases.ps1` still works but its
> counts predate the .NET 10 migration — do not quote them as current.

## 1. Summary (historical, 2026-09-28)

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
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
powershell -File tools\run_cases.ps1 -Compiler $tkvc
# f32 gate (the number that must stay 135/135):
bash tools/gate_f32.sh
# trainer gate (15/15):
& $tkvc tv/tests/trainer_tests.tkv out\trainer_tests.exe main
Set-Content benchmark/trainer_tests.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/trainer_tests.exe
```

Paste dated results (counts, duration, commit hash) below when a full run is captured.
Do not invent pass counts.
