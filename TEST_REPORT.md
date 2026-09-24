# TokenVector.Inference Test Suite Report

## 1. Summary
- **Suite:** `tv/tests/inference_tests.tkv`
- **Build:** `tkvc build tv/tests/inference_tests.tkv`
- **Target language:** TokenVector (`.tkv`)
- **Status:** record pass/fail from the `tkvc` run on this machine

## 2. Test Areas (suite modules)

| Area | What is asserted |
| :--- | :--- |
| Protobuf / ONNX parse | Varint, IEEE-754 float, length-delimited fields, model load |
| Graph passes | Constant fold, operator fusion, dead-code elimination |
| Quantization | MinMax/KL scale, INT8 roundtrip, FP8 E4M3/E5M2, INT8 GEMM vs float |
| Model accuracy | MLP / residual block / transformer layer numerical checks |
| Hot-path allocation | Arena reuse across repeated `run_single` calls |
| Serving / batching | Embedded server paths and batcher behavior |

## 3. How to run

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
.\inference_tests.exe
```

Paste dated results (counts, duration, commit hash) below when a full run is captured. Do not invent pass counts.
