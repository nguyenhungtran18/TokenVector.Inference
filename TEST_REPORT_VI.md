# Báo cáo Kiểm thử TokenVector.Inference (Tiếng Việt)

## 1. Tổng kết
- **Suite:** `tv/tests/inference_tests.tkv`
- **Build:** `tkvc build tv/tests/inference_tests.tkv`
- **Ngôn ngữ đích:** TokenVector (`.tkv`)
- **Trạng thái:** ghi pass/fail từ lần chạy `tkvc` trên máy này

## 2. Nhóm Kiểm thử (trong suite)

| Nhóm | Assert cái gì |
| :--- | :--- |
| Protobuf / ONNX parse | Varint, float IEEE-754, field length-delimited, load model |
| Graph passes | Constant fold, fusion, dead-code elimination |
| Quantization | Scale MinMax/KL, vòng lặp INT8, FP8 E4M3/E5M2, GEMM INT8 vs float |
| Độ chính xác model | MLP / residual block / transformer layer |
| Hot-path allocation | Arena reuse qua nhiều lần `run_single` |
| Serving / batching | Embedded server và batcher |

## 3. Cách chạy

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
.\inference_tests.exe
```

Dán kết quả có ngày tháng (số case, thời gian, commit) vào đây khi chạy đủ. Không bịa số pass.
