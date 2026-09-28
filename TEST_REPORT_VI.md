# Báo cáo Kiểm thử TokenVector.Inference (Tiếng Việt)

## 1. Tổng kết
- **Hai suite:** `tv/tests/inference_tests.tkv` (26 case) và `tv/tests/gap_tests.tkv` (16 case)
- **Ngôn ngữ đích:** TokenVector (`.tkv`)
- **Runner:** `powershell -File tools\run_cases.ps1`
- **Kiểm chứng gần nhất:** 2026-09-28 — 26/26 PASS, 16/16 PASS, 0 crash

Mỗi case trong 26 case chạy ở **process riêng**. `AccessViolationException` không bắt
được trong TokenVector, nên nếu dùng chung một process thì case đầu tiên crash sẽ làm
mất tất cả case còn lại. `gap_tests` chạy ở process thứ hai; `run_cases.ps1` trả về
khác 0 nếu bất kỳ suite nào fail.

## 2. 26 case phủ những gì

| Nhóm | Các case |
| :--- | :--- |
| Sinh văn bản | `generate_loop_text`, `grammar_prefix_mask`, `beam_log_softmax_tree`, `prefix_skip_plan`, `speculative_all_accept`, `tools_parse_validate` |
| ONNX parse / translate / write | `writer_modelproto`, `constant_fold_translate`, `gathernd_scatter`, `range_eye_cumsum`, `mish_prelu_instancenorm`, `resize_nearest_einsum`, `qlinear_conv_matmulinteger`, `lstm_unrolled`, `if_loop_end_to_end`, `scan_run_driver`, `pack_mask_and_preprocess`, `vit_graph_builds` |
| Lượng tử hóa | `awq_asym_and_fp16`, `gguf_k_quant_gate`, `hal_matmul_relu` |
| Training / adapter | `autograd_trains_step`, `autograd_gradcheck_and_adam`, `lora_forward_and_step` |
| String tensor | `string_tensor_pack`, `string_ops_side_table` |

## 3. 16 case gap phủ những gì

Thêm khi port mặt C# sang `.tkv`: protobuf varint / float IEEE / chuỗi UTF-8, hiệu
chuẩn MinMax, vòng INT8, vòng FP8 E4M3 + E5M2, FIFO của batcher + sentinel rỗng, matmul
SIMT, dynamic-shape, remap trọng số ViT, backend dispatch, gradient clipping, cộng 4 test
hồi quy cho lỗi treo và cho các pass tối ưu hóa.

## 4. Lỗ hổng phủ biết

- **Ngoài 2 test hồi quy, không test nào assert giá trị số qua session có bật tối ưu.**
  Tất cả case còn lại dựng session bằng `session_from_graph(g, 0)` — tắt pass. Vì vậy
  một lỗi trong pass vẫn khiến cả suite xanh. Hãy assert **giá trị**, không chỉ `len(y)`,
  trong mọi test session-level mới.
- Chưa có case MLP / ResNet / LLaMA-layer / độ sâu 50 tầng — chúng chỉ tồn tại trong bộ
  test C# đã xoá và chưa được port.
- Chưa có case concurrency, endurance hay cấp phát zero-GC: `spawn()` không tạo worker
  nên batcher chạy inline, không thể stress đa luồng.
- Chưa có test cho đường giải mã `raw_data` trong `tv/onnx/models.tkv`.

## 5. Cách chạy

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
powershell -File tools\run_cases.ps1 -Compiler $tkvc
# hoặc build trực tiếp
& $tkvc build tv/tests/inference_tests.tkv --out inference_tests.exe
& $tkvc build tv/tests/gap_tests.tkv        --out gap_tests.exe
```

Dán kết quả có ngày tháng (số case, thời gian, commit) vào đây khi chạy đủ. Không bịa số pass.
