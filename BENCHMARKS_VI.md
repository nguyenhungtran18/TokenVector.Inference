# Báo cáo Benchmark & Stress Test Đối đầu TokenVector.Inference (Tiếng Việt)

## 1. Tóm tắt
Suite benchmark đo trên engine **thuần TokenVector (`.tkv`)**. Kết quả phải chạy bằng
toolchain `tkvc` trên phần cứng đang thử — **không trích số của backend cũ**.

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
& $tkvc build tv/benchmarks/compare_rivals.tkv --out compare_rivals.exe
.\compare_rivals.exe
# hoặc chạy cả pipeline:
powershell -File build_inference.ps1 -BenchmarkTimeoutSeconds 300
```

**Đọc điều này trước khi trích số.** Tại thời điểm 2026-09-28, các hàm đo thời gian
trong `compare_rivals.tkv` (`now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`) vẫn
là stub trả `0`, nên suite in ra `0.0 ms`, `0 B` và giá trị `req/s` sinh ra từ phép chia
cho 0. **Hiện chưa con số thời gian nào trong suite này có ý nghĩa.** Banner
`QUICK-RUN SCALE` trong output cũng nói rõ điều đó. Hãy nối hàm đo thật trước khi ghi
kết quả.

Lưu ý phần cứng: **không có SIMD**. `tv/optimization/kernels.tkv` dùng vòng khối 8
rộng do tự viết (`VEC_W = 8`) nhưng vẫn chạy scalar, nên máy có AVX2/FMA chưa được lợi
gì. Xem `SIMD_NOTES.md` để biết lý do và compiler cần gì.

---

## 2. Cấu trúc Suite (`compare_rivals.tkv`)

Suite có sáu kịch bản: bốn benchmark, hai stress.

| Kịch bản | Đo cái gì |
| :--- | :--- |
| Cold-start & khởi tạo arena | Dựng graph + arena, qua `session_from_graph` |
| Latency single-sample trong process | Kernel fused linear / elementwise |
| Heap cho N lần suy luận | Áp lực cấp phát qua nhiều lần `run_single` |
| Throughput FP32 vs INT8 | GEMM lượng tử so với tham chiếu float |
| Hot-path endurance | Vòng suy luận liên tục, in bộ đếm GC |
| Burst đồng thời | Vòng `DynamicBatcher.enqueue`, in ok/err |

Số vòng cố ý nhỏ (latency 500, heap 500, throughput 200, endurance 2000, burst 500) để
suite chạy nhanh trong CI. Đây là smoke/regression gate, **không phải** phép so sánh
hiệu năng với baseline C# SIMD.

`stress_concurrent` in `ok=` và `err=`: nếu `err` khác 0 thì các request đã **lỗi** chứ
không phải chỉ chậm.

---

## 3. Ghi nhận Kết quả

Khi chạy suite, bổ sung bảng có ngày tháng tại đây:

- Model CPU host và flag
- Commit / bản `tkvc`
- Kịch bản, chỉ số, median ± độ phân tán
- Lệnh baseline rival cùng máy (ONNX Runtime, llama.cpp, PyTorch) nếu so sánh

Chưa chạy lại thì để ô kết quả trống — không sao chép số lịch sử. Áp dụng cùng quy tắc
cho các hằng số ORT/Torch viết cứng trong source: đó là literal trong file, không phải
số đo trên máy này.
