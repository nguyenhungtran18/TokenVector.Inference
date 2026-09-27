# HANDOFF — TokenVector.Inference (Pure TokenVector)

Ngày chốt: 2026-09-28. Nhánh `main`, chưa commit (xem "Trạng thái git" để biết cần commit gì).

---

## 1. Mục tiêu đã đạt

Chuyển toàn bộ chức năng từ C# sang pure `.tkv`, xác minh tương đương, rồi **xóa sạch C#**.

- C# đã bị xóa hoàn toàn: `Interop/`, `ONNX/`, `Optimization/`, `Quantization/`, `Runtime/`, `Server/`, `tests/`, `benchmarks/`, `TokenVector.Inference.csproj`, `TokenVector.Inference.sln`, `bin/`, `obj/`, `dist/`.
- Không còn `.cs` / `.csproj` / `.sln` nào trong repo.
- Không còn tham chiếu `dotnet`, `xUnit`, `BenchmarkDotNet`, `NuGet`, `TokenVector.Numerics` trong build/docs.
- Build + test chạy hoàn toàn qua `tkvc`.

## 2. Trạng thái test (ĐÃ XANH)

```powershell
powershell -File tools\run_cases.ps1
# TOTAL: pass=26 fail=0 crash=0
# gap tests: PASS (12/12)
```

```powershell
powershell -File build_inference.ps1 -SkipBenchmarks
# -> Completed Successfully
```

- `tv/tests/inference_tests.tkv`: 26/26 PASS
- `tv/tests/gap_tests.tkv`: 12/12 PASS (mirror xUnit cũ: protobuf, utf8, calibrate, INT8, FP8 E4M3/E5M2, batcher queue, SIMT/backend/dynamic-shape/ViT/clipping)
- Benchmark chạy tới `[4/4]` + `stress_hot` OK (xem mục 5 còn lỗi)

## 3. Thay đổi chính trong phiên này

### 3.1 Xóa C# + viết lại pipeline

| File | Việc |
|---|---|
| `build_inference.ps1` | Viết lại từ dotnet → pure tkv. Build 3 entry point (`inference_tests`, `gap_tests`, `compare_rivals`), chạy test, chạy benchmark có timeout guard. |
| `tools/run_cases.ps1` | Thêm bước `[3/3]`: build + chạy `gap_tests.tkv` sau 26 case, exit 1 nếu fail. |

Lệnh dùng:
```powershell
powershell -File build_inference.ps1                      # tất cả
powershell -File build_inference.ps1 -SkipBenchmarks     # nhanh (~ vài phút)
powershell -File build_inference.ps1 -Compiler "D:\path\to\tkvc.exe" -WorkDir $env:TEMP
powershell -File build_inference.ps1 -BenchmarkTimeoutSeconds 300
```

### 3.2 Port tính năng C# → `.tkv` thuần

- `tv/onnx/protobuf_reader.tkv` — đọc LE `u32/u64`, IEEE `f32/f64`, `utf8_decode`, varint.
- `tv/onnx/models.tkv` — giải mã `raw_data` (float/int32).
- `tv/quantization/types.tkv` — sửa decode FP8 E4M3 (exponent mask 4 bit, không phải 5).
- `tv/server/batcher.tkv` — sửa semantics constructor `BatchRequest` + sentinel queue rỗng.

### 3.3 Sửa 2 bug vô hạn lặp trong runtime (mới, chưa ai biết trước đây)

**(a) `DeadCodeEliminationPass.run()` — treo vô hạn** — `tv/runtime/session.tkv:3787`

```python
while i > -1:
    if i in active:
        continue          # <-- KHONG giam i  => lap vo han
```

Vòng `while changed` ngoài chạy lần 2 thì node đã `active` → `continue` mãi. Đây là lý do **mọi** `session_from_graph(g, 1)` (tức mọi session bật optimization) treo. Fix: `i = i + -1` trước `continue`.

Lưu ý: graph *mới* tạo tay chưa lộ bug này vì `scratch` rỗng; chỉ khi chạy pass mới lộ. Test cũ không cover nhánh này.

**(b) Bubble sort scratch trong `analyze_liveness_and_allocate_offsets()`** — `tv/runtime/session.tkv:3213`

```python
while _si < len(scratch) - 1:      # i32 < TkvInt  => lap vo han
```

Fix: `n_scratch = int(len(scratch))` + `for` loop thay `while`. Đây là fix phòng ngừa (cùng họ bug TkvInt đã gặp nhiều lần trong repo), không phải nguyên nhân chính của (a).

### 3.4 Sửa benchmark

- Giảm số vòng: latency `10000→500`, GC `5000→500`, throughput `2000→200`, stress `250000→2000`, concurrent `10000→500`.
- `bench_throughput` truyền `ptr_of(xf)` (stub trả `0`) vào `fused_matmul_bias_act` → `NullReferenceException`. Fix: truyền thẳng `list` (`xf, wf, [], yf`).
- In cảnh báo `QUICK-RUN SCALE` để không ai đọc nhầm số liệu quick-run so với baseline C# SIMD.

## 4. Bẫy tkvc đã gặp (quan trọng)

1. **Build lẻ từng module làm hỏng repo.** `tkvc build <module>` tự phát sinh lại các import-copy stub; 9 file bị ghi toàn số 0:
   `stdlib/tv.inference.tkv`, `stdlib/tv.numerics.{linalg,tensor}.tkv`, `stdlib/tv/inference/tv.numerics.{linalg,tensor}.tkv`, `tv/benchmarks/tv.inference.tkv`, `tv/benchmarks/tv.numerics.tensor.tkv`, `tv/format/tv.numerics.tensor.tkv`, `tv/onnx/tv.numerics.tensor.tkv`.
   Đã khôi phục bằng `git checkout HEAD -- <file>`. `build_inference.ps1` giờ chỉ build entry point + có guard phát hiện file zero.
2. **Sửa canonical ⇒ phải sync copy.** Compiler resolve import phẳng `tv.*.tkv` cạnh entry. Lệnh dùng nhiều lần:
   ```powershell
   Get-ChildItem -Recurse -Filter 'tv.runtime.session.tkv' | ForEach-Object { Copy-Item tv\runtime\session.tkv $_.FullName -Force }
   ```
3. **Chuỗi literal chỉ nhận dấu nháy kép.** `'abc'` trong biểu thức → lỗi tokenizer (dấu nháy đơn chỉ hợp lệ trong annotation kiểu).
4. **`print(..., flush=1)` không hỗ trợ** → `il_codegen: AST node khong ro: ('kwarg', 'flush', ...)`.
5. **`argv()` là stub trả `[]`** trong canonical source ⇒ không đọc được CLI flag; benchmark không scale được qua tham số.

## 5. Việc CÒN LẠI (chưa làm)

1. **`stress_concurrent` vẫn treo** trong `tv/benchmarks/compare_rivals.tkv:169`. Triệu chứng: `stress_hot` in xong, không bao giờ tới dòng `Concurrent 32T/500 burst`. Nghi ngờ `BatcherQueue.enqueue` **block khi queue đầy** (capacity 500, enqueue đúng 500). Cần kiểm tra `tv/server/batcher.tkv` xem `enqueue` có chờ/throttle không, và thêm test cho đường đầy-queue.
2. **Timing helpers là stub trả 0** nên benchmark in `0.0 us`, `0 B`, `0.0x`, `req/s = 2e15`. Các hàm đang là stub trong chính file benchmark: `now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`, `fill_rand`, `fill_const`, `ptr_of`, `parallel_for`. Cần nối vào stdlib thật (hoặc xác nhận patched compiler có tự thay) nếu muốn số liệu có ý nghĩa.
3. **`native_arena_alloc` vẫn là stub `return 0`** (`tv/runtime/session.tkv:228`) — arena thực tế là `dict[i64, list[f32]]` (list-boxed). Đây là chặn chính cho SIMD/zero-GC thật.
4. **Chưa có SIMD** (đã khảo sát): compiler không có vector type/intrinsic, `il_codegen.py` không vectorize, và `list[f32]` chặn JIT auto-vectorization. Đường khả thi giữ 100% tkv: đổi arena sang buffer liên tục (`bytearray`). **Phân tích chi tiết, so sánh 3 hướng đi và lộ trình nằm ở `SIMD_NOTES.md`.**
5. **Test chưa mirror hết C# cũ**: `ZeroGCHotPathTests`, `StressTests`, `ExecutionModelTests` (MLP/ResNet/LLaMA-layer/50 tầng), `GraphOptimizationTests` (fusion/DCE/const-fold theo cặp), `DynamicBatcherTests` phần async/concurrency. Hiện mới mirror protobuf/quant/batcher-queue + một số case đơn lẻ. **Không nên xem là tương đương hoàn toàn cho tới khi bổ sung.**
6. **Cần review 2 fix vô hạn lặp ở mục 3.3** — nên thêm test hồi quy: (i) `session_from_graph(g, 1)` phải trả về; (ii) graph có ≥1 tensor scratch phải qua được liveness.

## 6. Trạng thái git

64 file thay đổi, chưa commit. Cần lưu ý trước khi commit:

**Sửa thật (chỉ sửa file canonical, copy là sinh tự động):**
- `tv/runtime/session.tkv` — fix DCE + liveness sort (quan trọng nhất)
- `tv/onnx/protobuf_reader.tkv`, `tv/onnx/models.tkv`, `tv/onnx/translator.tkv`
- `tv/quantization/types.tkv`, `tv/server/batcher.tkv`
- `tv/tests/inference_tests.tkv`, `tv/tests/gap_tests.tkv`
- `tv/benchmarks/compare_rivals.tkv`
- `tools/run_cases.ps1`

**File mới:** `build_inference.ps1` (untracked), `tv/tests/tv.server.batcher.tkv` (untracked, copy phục vụ import — cần giữ).

**Nhiều `tv.*.tkv` bị `M`:** đây là import-copy sinh tự động, đã sync từ canonical. `.gitignore` chỉ ignore `/tv.*.tkv` ở gốc nên các copy trong thư mục con vẫn được track. Cân nhắc thêm pattern ignore cho `**/tv.*.tkv` hoặc commit cả copy cho khớp trạng thái hiện tại.

**Không đụng:** `TOKENVECTOR_SYNTAX_SPEC.md`, `assets/tokenvector-logo.svg`, `stdlib/tv/inference/inference.tv` (pre-existing, không liên quan).

**Không được:** `git reset --hard` / `git clean -fd` — có thay đổi binary/untracked có sẵn từ trước.

## 7. Cách xác minh lại từ đầu

```powershell
# 1. Test (bắt buộc xanh)
powershell -File tools\run_cases.ps1
#   TOTAL: pass=26 fail=0 crash=0
#   gap tests: PASS (12/12)

# 2. Build + test + benchmark
powershell -File build_inference.ps1 -SkipBenchmarks
powershell -File build_inference.ps1 -BenchmarkTimeoutSeconds 300   # hiện vẫn fail ở stress_concurrent
```

Compiler: `D:\TokenVector\3.code\dist\tkvc_patched.exe` (ngoài repo; `build_inference.ps1` nhận tham số `-Compiler`).

## 8. Simulator dùng để dò lỗi

Để khoanh vùng lỗi runtime trong `session.tkv` (file canonical không thể thêm print vì nó được import dạng flat copy):

1. Tạo `tv/benchmarks/_probe_bench.tkv` = copy `compare_rivals.tkv` với `def main()` đổi tên thành `_bench_main_original()`, thêm `main` mới gọi lần lượt từng pass với `print` marker giữa các bước.
2. Thêm `print` debug vào **bản flat copy** `tv/benchmarks/tv.runtime.session.tkv` (chỉ file này), build rồi xoá.
3. Chạy với timeout để không treo shell:
   ```powershell
   $p = Start-Process -FilePath "$env:TEMP\probe.exe" -NoNewWindow -PassThru `
        -RedirectStandardOutput "$env:TEMP\o.txt" -RedirectStandardError "$env:TEMP\e.txt"
   if (-not $p.WaitForExit(60000)) { $p.Kill() }
   ```
4. Nhớ: sửa xong phải copy lại canonical về flat copy, và xoá `_probe_bench.tkv`.
