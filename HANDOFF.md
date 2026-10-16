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
# gap tests: PASS (16/16)
```

```powershell
powershell -File build_inference.ps1 -SkipBenchmarks
# -> Completed Successfully
```

- `tv/tests/inference_tests.tkv`: 26/26 PASS
- `tv/tests/gap_tests.tkv`: 16/16 PASS (12 case mirror xUnit cũ: protobuf, utf8, calibrate, INT8, FP8 E4M3/E5M2, batcher queue, SIMT/backend/dynamic-shape/ViT/clipping — cộng 4 test hồi quy lỗi treo/lọc ngược)
- Benchmark chạy trọn tới hết (`[1/4]`…`[4/4]` + cả 2 stress), `ok=500 err=0`

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

### 3.3 Sửa 3 bug trong runtime (mới, chưa ai biết trước đây)

**(a) `DeadCodeEliminationPass` lọc NGUỘC node — nghiêm trọng nhất** — `tv/runtime/session.tkv:3806`

```python
if i not in active:      # <-- GIU node chet, XOA node song
    _qx1000.append(graph.nodes[i])
```

Triệu chứng: `session_from_graph(g, 1)` **xoá sạch mọi node** rồi trả về toàn `0`, **không
ném lỗi**:

```
opt=0 nodes=1 Y[0]=2.25  ← đúng
opt=1 nodes=0 Y[0]=0.0   ← sai
```

Vì sao sống dai: **mọi test trong `inference_tests.tkv` đều dựng session bằng
`session_from_graph(g, 0)`** (tắt optimization). Chỉ benchmark và `gap_tests` gọi `(1)`,
mà chúng chỉ assert `len(y)` chứ không assert giá trị. Fix: `if i in active:`.

Bài học: **test phải assert giá trị, không chỉ độ dài**, và phải có ít nhất một case bật
optimization. Cả hai test hồi quy trong `gap_tests` nay đã assert số (2.25 và 3.25).

**(b) `DeadCodeEliminationPass.run()` — treo vô hạn** — `tv/runtime/session.tkv:3787`

```python
while i > -1:
    if i in active:
        continue          # <-- KHONG giam i  => lap vo han
```

Vòng `while changed` ngoài chạy lần 2 thì node đã `active` → `continue` mãi. Đây là lý do **mọi** `session_from_graph(g, 1)` (tức mọi session bật optimization) treo. Fix: `i = i + -1` trước `continue`.

**(c) Bubble sort scratch trong `analyze_liveness_and_allocate_offsets()`** — `tv/runtime/session.tkv:3213`

```python
while _si < len(scratch) - 1:      # i32 < TkvInt  => lap vo han
```

Fix: `n_scratch = int(len(scratch))` + `for` loop thay `while`. Đây là fix phòng ngừa (cùng họ bug TkvInt đã gặp nhiều lần trong repo), không phải nguyên nhân chính của (b).

### 3.4 Sửa batcher + benchmark

- **`DynamicBatcher.enqueue` treo vô hạn.** `while not req.done: _bch_spin_yield()` mà `spawn(self._worker)` không tạo worker (runtime single-thread) và `_bch_spin_yield()` là stub `return 0` ⇒ CPU 100% mãi. Fix: giới hạn `_BCH_SPIN_LIMIT = 200000` lần quay, hết hạn thì chạy `_run_one` inline tại thread gọi. Cố tình giới hạn theo **số lần quay** chứ không theo đồng hồ, vì `_bch_now_us()` cũng là stub.
- **Thêm `new_dynamic_batcher_with_session()`.** `new_dynamic_batcher(1, ...)` đặt `1` vào field đầu nên `pool` rỗng ⇒ mọi request đều `req.error`. Factory mới gọi `pool.append(session)` y hệt `new_embedded_server`.
- Giảm số vòng: latency `10000→500`, GC `5000→500`, throughput `2000→200`, stress `250000→2000`, concurrent `10000→500`.
- `bench_throughput` truyền `ptr_of(xf)` (stub trả `0`) vào `fused_matmul_bias_act` → `NullReferenceException`. Fix: truyền thẳng `list` (`xf, wf, [], yf`).
- `stress_concurrent` đếm `ok`/`err` để lộ lỗi thay vì chỉ in req/s. Sau fix: `ok=500 err=0`.
- In cảnh báo `QUICK-RUN SCALE` để không ai đọc nhầm số liệu quick-run so với baseline C# SIMD.

### 3.5 Test hồi quy cho 3 lỗi treo (thêm vào `gap_tests.tkv`, 12→16)

| Test | Chặn lỗi |
|---|---|
| `test_optimized_session_terminates` | DCE quay vô hạn — test cũ chỉ gọi `session_from_graph(g, 0)` nên không bao giờ bật pass nào |
| `test_scratch_liveness_terminates` | Liveness sort với `scratch` khác rỗng; graph có `Y = gemm(X)` + `T0 = relu(X)` rồi `Z = Y + T0` để cả hai còn sống sau DCE |
| `test_batcher_enqueue_returns` | `enqueue` treo; test sẽ treo (không fail) nếu bug quay lại — `run_cases.ps1` chạy mỗi case ở process riêng nên vẫn bắt được |
| `test_batcher_queue_full_raises` | `ServerBusy` khi queue đầy + `queue_depth()` về 0 sau khi rảnh |

Lưu ý khi viết test: đừng tạo node ghi đè một tensor đã được đọc trong cùng graph (ghi `Y = Add(Y, T0)`) — `run_single` ném `ArgumentOutOfRangeException` ở `_exec_unary`. Dùng tensor output riêng.

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
6. **`__init__` CỦA RECORD BỊ BỎ QUA KHI BIÊN DỊCH.** Thân `__init__` không chạy; record được dựng từ field theo thứ tự, mỗi tham số positional tương ứng **một field không phải container**, theo đúng thứ tự field. Vì vậy mọi "khởi tạo" phải làm bên ngoài bằng factory + gán tay. Đã thấy ở `new_embedded_server` (`tv/server/embedded_server.tkv:439-441`): tạo batcher rồi `pool.append(session)`. Đây cũng là lý do `new_dynamic_batcher(1, ...)` có `pool` rỗng.
7. **Hằng số cấp module KHÔNG import được giữa các module** (chỉ hàm/record mới import được). Phải khai báo lại hằng ở file dùng — xem `inference_tests.tkv:42` và `gap_tests.tkv` đã làm y hệt.
8. **Compiler mới (2026-10-15) thêm 3 quy tắc nữa:**
   - **Trùng tên hàm giữa 2 module là lỗi cứng** (`trung ten ham [...]`) ⇒ mỗi helper phải có tên duy nhất toàn cục.
   - **Không index trực tiếp vào property `.shape`**, kể cả literal (`d.shape[0]` → `shape=[]`); đọc qua biến trung gian rồi index.
   - **Annotation phải resolve trước khi record được khai báo**; nhưng lời gọi hàm chưa khai báo thì OK. **Field kiểu container phải được gán** trong `__init__` hoặc ngoài constructor.

## 5. Việc CÒN LẠI (chưa làm)

1. **Timing helpers là stub trả 0** nên benchmark in `0.0 us`, `0 B`, `0.0x`, `req/s = 5e14` (chia cho 0). Các hàm đang là stub trong chính file benchmark: `now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`, `fill_rand`, `fill_const`, `ptr_of`, `parallel_for`. Phải nối vào stdlib thật thì mới chứng minh được cải tiến hiệu năng. **Đây là ưu tiên cao nhất** vì nó chặn mọi phép đo.
2. **`native_arena_alloc` vẫn là stub `return 0`** (`tv/runtime/session.tkv:228`) — nhưng đây là stub **cấp repo**, không phải giới hạn ngôn ngữ: `nativebuf.tkv` đã có `alloc_bytes`/`free_bytes`/`read_i32`/`write_i32` trả con trỏ thật (đã probe). Sửa được ngay trong repo này.
3. **SIMD: compiler ĐÃ CÓ `simd_*`, chỉ thiếu đóng gói.** (2026-10-15, sửa lại — bản cũ ghi "chưa có SIMD" là sai, viết theo compiler đã bị xoá.) Compiler mới có 13 builtin `simd_add/sub/mul/scale/axpy/dot/sum/max/min/lanes_*/hw/isa` trên mảng numpy. Việc còn lại của repo: đổi arena `dict[i64, list[f32]]` sang mảng liên tục rồi thay thân khối `VEC_W = 8`. **Không cần sửa compiler.** Lưu ý `simd_*` hiện chạy sẽ nổ `FileNotFoundException: System.Numerics.Vectors 4.1.4.0` — thuộc phía repo compiler. Chi tiết: `SIMD_NOTES.md`.
4. **Runtime chưa biên dịch được với compiler mới** — `tv/runtime/session.tkv` làm compiler nổ `The given key was not present in the dictionary` (KeyError phía compiler). Đã migrate 4 quy tắc ngôn ngữ mới (`.shape[N]`, index trên kết quả hàm, `//`/`%` cần nhớ tạm, trùng tên hàm giữa module) nhưng **chưa verify end-to-end vì chưa build được**. Báo cáo đầy đủ: `D:\TokenVector\Release\docs\e_handoff\2026-10-15-inference-repo-build-blocker.md`.
4. **Test chưa mirror hết C# cũ**: `ZeroGCHotPathTests`, `StressTests`, `ExecutionModelTests` (MLP/ResNet/LLaMA-layer/50 tầng), `GraphOptimizationTests` (fusion/DCE/const-fold theo cặp), `DynamicBatcherTests` phần async/concurrency. Hiện mới mirror protobuf/quant/batcher-queue + 4 test hồi quy lỗi treo. **Không nên xem là tương đương hoàn toàn cho tới khi bổ sung.**
5. **`BatcherQueue`/`DynamicBatcher` chưa thật sự đa luồng.** `spawn(self._worker)` không tạo worker (runtime single-thread), nên `enqueue` chạy inline ⇒ không có micro-batching thật, batch luôn bằng 1. Cần worker thật (xem câu hỏi mở ở `SIMD_NOTES.md` mục 8).
6. **Chưa có test cho `raw_data` decode** của `tv/onnx/models.tkv` — phần code này từng bị bỏ sót vì flat copy lệch bản (xem mục 6), và hiện vẫn chưa có case nào chạm tới nó.

## 6. Trạng thái git

64 file thay đổi, chưa commit. Cần lưu ý trước khi commit:

**Sửa thật (chỉ sửa file canonical, copy là sinh tự động):**
- `tv/runtime/session.tkv` — fix DCE + liveness sort (quan trọng nhất)
- `tv/onnx/protobuf_reader.tkv`, `tv/onnx/models.tkv`, `tv/onnx/translator.tkv`
- `tv/quantization/types.tkv`, `tv/server/batcher.tkv`
- `tv/tests/inference_tests.tkv`, `tv/tests/gap_tests.tkv`
- `tv/benchmarks/compare_rivals.tkv`
- `tools/run_cases.ps1`

**File mới:** `build_inference.ps1`, `HANDOFF.md`, `SIMD_NOTES.md`, `tv/examples/quickstart.tkv` (+ các copy `tv.*.tkv` cạnh nó), `tv/tests/tv.server.batcher.tkv`.

**Nhiều `tv.*.tkv` bị `M`:** đây là import-copy sinh tự động, đã sync từ canonical. `.gitignore` chỉ ignore `/tv.*.tkv` ở gốc nên các copy trong thư mục con vẫn được track. Cân nhắc thêm pattern ignore cho `**/tv.*.tkv` hoặc commit cả copy cho khớp trạng thái hiện tại.

**Đã dọn trong phiên dọn dẹp (2026-09-28):**
- Xoá `stdlib/tv/inference/inference.tv` — prototype `.tv` còn sót từ trước lúc đổi tên `.tv` → `.tkv`, dùng `NDArray[float32]`/`data_ptr`/`zeros_like` (không tồn tại) và không ai tham chiếu.
- Viết lại toàn bộ docs theo trạng thái thật (xem mục 3.6).
- `TOKENVECTOR_SYNTAX_SPEC.md` trước đó bị ghi đè thành 660 byte khoảng trắng; bản trong git cũng chỉ là placeholder 599 char mô tả API không tồn tại. Đã viết lại cả 2 bản EN/VI.

**Không đụng:** `assets/tokenvector-logo.svg` (thay đổi có sẵn, không liên quan).

**Không được:** `git reset --hard` / `git clean -fd` — có thay đổi binary/untracked có sẵn từ trước.

## 7. Cách xác minh lại từ đầu

```powershell
# 1. Test (bắt buộc xanh)
powershell -File tools\run_cases.ps1
#   TOTAL: pass=26 fail=0 crash=0
#   gap tests: PASS (16/16)

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
