# Báo cáo Benchmark & Stress Test Đối đầu TokenVector.Inference (Tiếng Việt)

## 1. Tóm tắt
Suite benchmark đo trên engine **thuần TokenVector (`.tkv`)**. Kết quả phải chạy bằng toolchain `tkvc` trên phần cứng đang thử — **không trích số của backend cũ**.

```powershell
# Compiler: D:\TokenVector\3.code\dist\tkvc.exe
tkvc build tv/benchmarks/compare_rivals.tkv
```

Lưu ý phần cứng: CPU x86-64 có AVX2/FMA khuyến nghị cho kernel blocked SIMD trong `tv/optimization/kernels.tkv` và `tv/quantization/matmul.tkv`.

---

## 2. Cấu trúc Suite (`compare_rivals.tkv`)

| Kịch bản | Đo cái gì |
| :--- | :--- |
| Cold-start & load model | Parse ONNX + khởi tạo arena |
| In-process latency (single-sample) | Fused linear / elementwise |
| Arena reuse / áp lực cấp phát | Reuse hot path qua nhiều lần chạy |
| INT8 vs FP32 throughput | GEMM lượng tử vs tham chiếu float |
| Hot-path endurance | Vòng suy luận liên tục dài |
| Concurrent serving | Burst đa worker qua embedded server |
| Massive layer projection | GEMM lớn, GFLOPS hiệu dụng |
| Deep DAG | Độ trễ pipeline nhiều lớp fused |

Regression gates nằm trong cùng suite; fail gate thay vì công bố số cũ.

---

## 3. Ghi nhận Kết quả

Khi chạy suite, bổ sung bảng có ngày tháng tại đây:

- Model CPU host và flag (AVX2/FMA)
- Commit / bản `tkvc`
- Kịch bản, chỉ số, median ± độ phân tán
- Lệnh baseline rival cùng máy (ONNX Runtime, llama.cpp, PyTorch) nếu so sánh

Chưa chạy lại thì để ô kết quả trống — không sao chép số lịch sử.
