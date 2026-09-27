# SIMD thuần TokenVector — ghi chú khảo sát & lộ trình

Ghi chú riêng. Ngày khảo sát: 2026-09-28. Liên quan: `HANDOFF.md` mục 5.4.
Phạm vi: repo `D:\TokenVector.Inference` + compiler `D:\TokenVector\3.code`.

---

## 1. Kết luận

**TokenVector hiện KHÔNG dùng được SIMD.** Không phải vì thiếu thuật toán, mà vì
cả 3 tầng đều chặn:

| Tầng | Trạng thái | Bằng chứng |
|---|---|---|
| Ngôn ngữ (vector type / intrinsic) | Không có | Grep `simd\|avx\|__m256\|vectoriz\|fma\|pshufb\|float32x\|span<` trong `build\pyinstaller_src\**\*.py` (27 file) → **0 match** |
| Compiler (auto-vectorization) | Không có pass nào | `compiler/il_codegen.py`, `il_core.py`, `il_dispatch.py`, `typed_dsl_parser.py` không có nhánh vectorize |
| Kiểu dữ liệu tensor | Boxed, chặn JIT | `ExecutionMemoryArena` = `dict[i64, list[f32]]` (`tv/runtime/session.tkv:99`) |
| Phép nạp 32 byte một lệnh | **Thiếu** | Không có `ldelema`/`Vector256`/`NativeMemory`/`Unsafe.` trong toàn bộ compiler |
| Bộ nhớ unmanaged | **Đã có** | `nativebuf.tkv`: `alloc_bytes` trả con trỏ thật — probe: `alloc_bytes(64) -> 6144304` |
| FFI sang native | **Có** | `il_features_library/ffi_feature.tkv` (`ctypes_cdll`, `c_call`) |

Tóm lại: phần còn thiếu **nhỏ hơn nhiều so với tưởng** — không phải cả tầng ngôn ngữ,
mà là (a) nâng `native_arena_alloc` từ stub lên `alloc_bytes` (đã có sẵn) và (b) thêm
một feature module vector với `v_load`/`v_store`. Chi tiết ở mục 7.

Đo hậu quả thực tế: `run_single` × 10 000 vòng không xong trong >10 phút; đã hạ
còn 500 vòng cho CI (`tv/benchmarks/compare_rivals.tkv`).

## 2. Vì sao `list[f32]` chặn SIMD (chi tiết quan trọng)

Compiler phát ra **.NET IL/CIL** (nhìn thấy rõ qua `pinvokeimpl("ucrtbase.dll")`
trong `ffi_feature.tkv`), nên code chạy trên CLR và do **RyuJIT** tối ưu.

RyuJIT chỉ vectorize được vòng lặp đơn giản trên **vùng nhớ liên tục** và không
alias. `List<float>` trong .NET là `float[]` đóng gói trong `object[]` — mỗi phần
tử là **một tham chiếu heap**. Vòng `y[i] = x[i]*w[i]` biên dịch thành:
nạp `object[]` → nạp `float&` → nạp `float` → nhân → ghi lại qua tham chiếu.
JIT không thể gom thành `vmulps`.

Hai hệ quả cụ thể trong repo này:

1. **Căn chỉnh 64 byte hiện chưa dùng được.** `ExecutionMemoryArena.__init__` có
   `(size_in_bytes + 63) // 64 * 64` với comment `# Align up to 64 bytes (AVX2 /
   AVX-512)` (`tv/runtime/session.tkv:106`), nhưng arena là `list[f32]` boxed nên
   căn chỉnh byte không tạo được vector load.
2. **CON TRỎ THẬT ĐÃ CÓ SẴN — kết luận cũ của tài liệu này là SAI.** `native_arena_alloc`
   trả `0` chỉ vì đó là **stub cấp repo**, không phải giới hạn của ngôn ngữ.
   `il_features_library/nativebuf.tkv` đã đăng ký 13 builtin quản lý bộ nhớ không
   GC, và đã probe thực tế (xem mục 7.1):
   `alloc_bytes(64) -> 6144304` (con trỏ thật), `write_i32`/`read_i32`/`read_u8`/
   `write_i64`/`read_i64` đều đúng giá trị, `free_bytes -> 0`.
   ⇒ Arena của inference **có thể cấp phát unmanaged ngay trong `.tkv`**, và đường
   FFI cũng không bị chặn. Điều còn thiếu không phải con trỏ, mà là phép **nạp 32
   byte một lệnh** (mục 7.3 / `C2`).
3. **Thiếu `read_f32`/`write_f32` trên native buffer.** `nativebuf` chỉ có
   `u8`/`i32`/`i64`. Đường scalar hiện phải đi qua `read_i32` rồi tự reinterpret —
   sai số bit với giá trị float.

## 3. Thêm một rào cản ít ai ngờ: `tv.numerics` là stub rỗng

`tv.numerics.tensor` và `tv.numerics.linalg` **không có source canonical trong repo
này** — chỉ có bản stub do `tkvc` tự sinh để resolve import:

```python
# tv/optimization/tv.numerics.tensor.tkv
# Stub module: tv.numerics.tensor (auto-generated for tkvc import resolution)
# Real tensor helpers live with callers.
def tensor_zeros(n: "i64") -> "list[f32]":
    ...
```

Lõi tensor thật nằm ở repo khác (`TokenVector.Numerics`) — chính repo mà dự án
C# đã xoá từng dùng làm NuGet package. Hệ quả: muốn có SIMD **thuần TokenVector**,
phần lõi vector hoá phải được viết **trong repo này**, không kỳ vọng nó có sẵn.

Lưu ý: rào cản này chỉ áp dụng nếu đi theo hướng dùng `tv.numerics`. Nếu chọn hướng
con trỏ unmanaged ở mục 7, arena và kernel nằm trong `tv/runtime` + `tv/quantization`
và không cần `tv.numerics` một chút nào.

## 4. Tín hiệu tích cực: idiom khối 8-lane đã có sẵn

`tv/optimization/kernels.tkv:8` đặt `VEC_W = 8` và dùng khối 8 rộng (vẫn scalar):

```python
VEC_W = 8
while j + VEC_W <= d:      # thân vector 8 lane
    for l in range(VEC_W):
        ...
    j = j + VEC_W
while j < d:               # đuôi dư
    ...
```

Xuất hiện ở các vòng dòng ~71-89, ~185-212, ~242-246. Đây chính là **điểm đổi từ
scalar sang SIMD thuần `.tkv`** ít sửa đổi nhất: giữ nguyên cấu trúc khối, chỉ thay
phép tính từng phần tử bằng phép vector. Không cần đổi thuật toán, không cần đổi
graph.

Lưu ý: idiom này **chỉ an toàn khi biên là `int`**. Mọi biên so sánh với `len(...)`
trả về `TkvInt` đã từng gây treo vô hạn (xem `HANDOFF.md` mục 3.3b và 4). Khi viết
lại vòng khối, phải ép `int` trước.

## 5. Ba hướng, so sánh

| | Hướng | Giữ 100% `.tkv` | Nỗ lực | Kỳ vọng |
|---|---|---|---|---|
| **A** | Arena dùng buffer liên tục ( unmanaged `alloc_bytes` hoặc `bytearray`) + giữ khối `VEC_W` | Có | Trung bình, trong repo này | Bỏ boxing, có con trỏ thật; JIT **có thể** tự vectorize vòng đơn giản; chưa phải SIMD do cấp |
| **B** | Thêm vector type + intrinsic vào compiler | Có | Trung bình, phải sửa `D:\TokenVector\3.code` | SIMD thật, khối 8-lane thành 1 lệnh vector |
| **C** | FFI sang DLL C biên dịch AVX2/AVX-512 | **Không** | Thấp | Nhanh nhất, nhưng đánh đổi tuyên bố kiến trúc |

Khuyến nghị: **A → B, C chỉ làm escape hatch.** Cần sửa đúng một thứ: nâng
`native_arena_alloc` từ stub lên `alloc_bytes` (đã có sẵn trong ngôn ngữ). Sau đó
chỉ còn thiếu ~1 builtin `v_load`/`v_store` ở tầng compiler (chi tiết ở mục 7).

## 6. Lộ trình gợi ý

### Giai đoạn 0 — Đo trước khi sửa (bắt buộc)
Benchmark hiện in `0.0 us`, `0 B`, `0.0x` vì các hàm thời gian là stub trả 0
(`now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`, `fill_rand`, `fill_const`,
`ptr_of`, `parallel_for` — định nghĩa ở cuối `compare_rivals.tkv`). **Chưa có số
liệu thật thì không chứng minh được cải tiến.** Nối các hàm này vào stdlib thật
trước.

### Giai đoạn 1 — Bỏ boxing khỏi arena (nâng stub lên bộ nhớ thật)
1. `native_arena_alloc(n)` → `alloc_bytes(n)`; `native_arena_free(p)` → `free_bytes(p)`
   (`tv/runtime/session.tkv:228`). Cả hai primitive **đã có sẵn** trong ngôn ngữ.
2. Giữ `fmap: dict[i64, list[f32]]` cho đường scalar, **thêm** con trỏ + offset cho
   đường vector. Rủi ro thấp hơn nhiều so với đổi toàn bộ arena sang byte.
3. Thêm `read_f32`/`write_f32` (thiếu trong `nativebuf`) hoặc chấp nhận đi qua
   `read_i32` cho tới khi compiler bổ sung.
4. Có test hồi quy: 26 case `inference_tests` + 16 case `gap_tests` phải xanh, và
   kết quả số học phải **giống hệt** trước/sau (không chỉ "chạy được").
5. Nhớ: `__init__` của record bị bỏ qua lúc biên dịch ⇒ phải khởi tạo qua
   `arena_create()`, không được dựa vào thân `__init__`.

### Giai đoạn 2 — Tận dụng khối `VEC_W` hiện có
1. Chuẩn hoá mọi vòng khối về biên `int`.
2. Tách thân vector và thân đuôi thành helper dùng chung để không nhân bản.
3. Kiểm chứng xem RyuJIT có vectorize hay không bằng cách **đo** (Giai đoạn 0), đừng
   đoán.
4. Khi có `v_load`/`v_store` (mục 7), thay phép tính từng phần tử trong thân khối.

### Giai đoạn 3 — Bổ sung compiler (chi tiết ở mục 7)

## 7. Yêu cầu điều chỉnh compiler

Tất cả đường dẫn dưới đây là file thật trong `D:\TokenVector\3.code`
(nguồn: `build\pyinstaller_src\`). Mỗi mục ghi **file cụ thể + việc + lý do**.

### 7.1 Đã có sẵn, KHÔNG cần làm lại (đã kiểm chứng)

| Có gì | Ở đâu | Ghi chú |
|---|---|---|
| Cấp phát bộ nhớ unmanaged, trả con trỏ 64-bit | `il_features_library/nativebuf.tkv` — `alloc_bytes`, `free_bytes` | Probe thực tế: `alloc_bytes(64) -> 6144304`, `free_bytes -> 0` |
| Đọc/ghi bộ nhớ theo offset | cùng file — `read_u8`/`write_u8`/`read_i32`/`write_i32`/`read_i64`/`write_i64` | Probe: `write_i32` rồi `read_i32` trả đúng `12345` và `-7` |
| Đường gọi DLL native | `il_features_library/ffi_feature.tkv` — `ctypes_cdll`, `c_call` | Cho hướng C |
| Cơ chế đăng ký dtype mới + builtin | `compiler/il_core.py` (`IL_SCALAR`) + `il_dispatch.register_expr_builtin` | Mẫu tham chiếu: `il_features_library/int_type.tkv` |
| Cơ chế chèn class IL một lần | `int_type.tkv` → `ensure_class(ctx)`, đẩy vào `ctx['extra_classes']` | Dùng khi cần tự định nghĩa struct |

**Hệ quả quan trọng:** không cần viết `native_arena_alloc` bằng FFI, không cần
kernel C để bắt đầu. Arena cấp phát được ngay bằng `alloc_bytes`.

### 7.2 `C1` — Đăng ký dtype vector

- **File:** `compiler/typed_dsl_parser.py:26` (`DTYPES`) và `compiler/il_core.py:9`
  (`IL_SCALAR`).
- **Việc:** thêm `f32x8` (sau đó mới tới `i32x8`, `f64x4`) vào `DTYPES`, và ánh xạ
  trong `IL_SCALAR` sang type **có sẵn của .NET**:
  ```
  IL_SCALAR['f32x8'] = 'valuetype [System.Runtime.Intrinsics]System.Runtime.Intrinsics.Vector256`1<float32>'
  ```
- **Lý do:** `DTYPES` là tập **đóng**; chưa thêm thì annotation `-> "f32x8"` không
  parse được. Grammar ở `typed_dsl_parser.py:11` chỉ nhận `DTYPE`.
- **Không làm:** đăng ký `f32x8` vào `IL_LDC_OP` (`il_core.py:25`). `ldc.*` không tạo
  được struct 32 byte ⇒ **cấm literal vector**, dùng `v_splat(x)`. Đây cũng là lý do
  `int_type.tkv` cố ý không đăng ký `int` vào `IL_LDC_OP`.
- **Kiểm tra:** `IL_NARROW_TO_I32` (`il_core.py:36`) chỉ dùng cho chỉ số — đừng thêm
  `f32x8` vào đó.

### 7.3 `C2` — Feature module mới cho phép vector

- **File mới:** `il_features_library/vector_type.tkv`, theo khuôn `int_type.tkv`
  (hàm `register_*_dtype()` gọi lúc nạp module + `register_expr_builtin` cho từng phép).
- **Việc:** đăng ký tối thiểu các builtin sau. Tên phải **duy nhất toàn cục** vì
  `il_dispatch.py:115` ném `ValueError` ngay lúc import nếu trùng tên.

| Builtin | Chữ ký | Ghi chú codegen |
|---|---|---|
| `v_splat` | `(x: f32) -> f32x8` | broadcast 1 giá trị |
| **`v_load`** | `(p: i64, off: i64) -> f32x8` | **mấu chốt** — nạp 32 byte tại địa chỉ native |
| **`v_store`** | `(p: i64, off: i64, v: f32x8) -> i64` | ghi 32 byte |
| `v_add` `v_sub` `v_mul` `v_div` | `(v, w: f32x8) -> f32x8` | |
| `v_fma` | `(a, b, c: f32x8) -> f32x8` | FMA gộp 2 phép, giảm 1 lệnh |
| `v_hsum` | `(v: f32x8) -> f32` | horizontal add — dùng ở đuôi khối |
| `v_lane` | `(v: f32x8, i: i32) -> f32` | rút 1 lane |
| `v_max` `v_min` | `(v, w: f32x8) -> f32x8` | |
| `v_any_gt` | `(v: f32x8, x: f32) -> i32` | trả 1 nếu có lane > x, cho nhánh điều kiện |

- **Codegen gợi ý cho `v_load`/`v_store`:** phát
  `System.Runtime.Intrinsics.Vector256::LoadUnsafe(ref byte*)` và `StoreUnsafe`.
  Dùng `LoadUnsafe` (không phải `Load`) để **cho phép địa chỉ lệch 4 byte** — kernel
  nhiều lúc chưa căn 32 byte. Nếu buộc dùng `Avx.LoadVector256` thì arena phải căn 32
  (hiện đã căn 64 nên vẫn được).

### 7.4 `C3` — Bổ sung `read_f32`/`write_f32` cho native buffer

- **File:** `il_features_library/nativebuf.tkv`.
- **Việc:** thêm `read_f32(p, off) -> f32` và `write_f32(p, off, v) -> i64`, cùng
  kiểu với `read_i32`/`write_i32` đã có.
- **Lý do:** hiện chỉ có `u8`/`i32`/`i64`. Đường scalar buộc phải `read_i32` rồi tự
  reinterpret ⇒ sai số bit với float.

### 7.5 `C4` — Cổng vào trong runtime của repo này (không thuộc compiler)

- **File:** `tv/runtime/session.tkv:228-236`.
- **Việc:** `native_arena_alloc` → `alloc_bytes`; `native_arena_free` → `free_bytes`.
- **Cảnh báo:** `__init__` của record bị compiler bỏ qua, nên mọi khởi tạo phải đi qua
  `arena_create()`; và phải giữ nguyên ngữ nghĩa zero-copy của `copy_from`/`copy_to`.

### 7.6 `C5` — Kiểm thử và nghiệm thu

- **Test compiler:** `D:\TokenVector\3.code\test\verify\vector_type_test.tkv` (thư mục
  test hiện có). Bắt buộc có: load/store round-trip, `v_hsum` khớp tổng scalar,
  `v_fma` khớp `a*b+c` tính bằng scalar, và kiểm tra **địa chỉ lệch** (offset 0..28).
- **Spike đo hiệu năng:** `D:\TokenVector\3.code\docs\spikes\` — tiền lệ là
  `spike_int_repr.py` mà `int_type.tkv` dẫn chiếu. Đo 1 vòng `v_mul` + `v_hsum` trên
  8 phần tử so với 8 phép nhân scalar, lặp đủ nhiều để đo.
- **Nghiệm thu ở repo này:** 26 case `inference_tests` + 16 case `gap_tests` xanh
  **và kết quả số học giống hệt**, không chỉ "chạy được".

### 7.7 Rủi ro khi làm

1. `Vector256` là value type **32 byte**. `il_codegen` giả định "một giá trị trên
   stack" cho mỗi biểu thức (`int_type.tkv` giải thích điều này khi chọn struct).
   Struct 32 byte làm tăng áp lực stack — cần đo, không giả định là miễn phí.
2. `System.Runtime.Intrinsics` cần target framework phù hợp. **Chưa xác minh**
   `tkvc_patched.exe` build cho `net8.0` hay target khác — phải kiểm tra trước khi
   viết codegen `Vector256`.
3. Không được kỳ vọng JIT tự vectorize thay cho `C2`. Nếu chỉ làm `C1` mà không
   có `v_load`/`v_store`, kết quả sẽ là "vector trên giấy" nhưng vẫn chạy scalar.
4. `register_expr_builtin` ném lỗi lúc **import** nếu trùng tên — phải grep trước
   khi chọn tên `v_*`.
5. Thứ tự nạp feature có ảnh hưởng (xem `plugin_loader.py`) — `IL_SCALAR` phải được
   đăng ký trước khi dùng.

## 8. Bẫy đã biết (rút từ phiên này)

1. **Đừng build lẻ từng module bằng `tkvc` khi đang lặp nhanh.** Compiler tự
   phát sinh lại import-copy stub; 9 file bị ghi toàn số 0 và phải khôi phục từ
   `git checkout HEAD --`. Đã có guard trong `build_inference.ps1`.
2. **Sửa canonical ⇒ sync copy.** Sửa `tv/runtime/session.tkv` mà quên copy sang
   `tv/*/tv.runtime.session.tkv` và `stdlib/tv/inference/` thì test vẫn xanh nhưng
   **đang test code cũ** — dễ tưởng fix đã hiệu quả.
3. **Biên kiểu `TkvInt` trong `while` là bẫy treo máy.** Mọi vòng khối vector hoá
   mới viết phải ép `int`.
4. **Chuỗi chỉ nhận dấu nháy kép** (`'x'` hỏng), **không có `print(..., flush=)`**,
   **`argv()` là stub trả `[]`** nên không truyền được flag.
5. **Đừng tin số liệu benchmark hiện tại.** Phần lớn là 0 do stub; phần `[stress]`
   in `req/s = 2e15` là hệ quả của việc chia cho 0.
6. **`__init__` của record bị bỏ qua lúc biên dịch.** Mọi phần khởi tạo phải nằm
   trong factory và gán tay. Điều này ảnh hưởng trực tiếp tới Giai đoạn 1: nếu thêm
   buffer liên tục và hi vọng `__init__` của `ExecutionMemoryArena` tự cấp phát,
   nó sẽ không chạy. Phải sửa `arena_create` và mọi call-site.
7. **Hằng số cấp module không import được giữa module** — mọi hằng dùng cho SIMD
   (vector width, tail mask) phải khai báo lại ở từng file.

## 9. Câu hỏi còn mở

Đã trả lời trong lúc viết tài liệu này:
- ~~TokenVector có con trỏ unmanaged thật không?~~ **Có** — `alloc_bytes`, đã probe.
- ~~Cần reinterpret bytes → f32 để làm hướng A không?~~ **Không cần** nếu đi đường
  con trỏ: `v_load`/`v_store` đọc thẳng 32 byte từ địa chỉ. Reinterpret chỉ cần cho
  đường scalar tạm thời, và `read_i32` đã đủ để dựng trước.

Còn lại:
- **`tkvc_patched.exe` build cho target framework nào?** `System.Runtime.Intrinsics`
  cần `net8.0` trở lên. Phải xác minh trước khi viết codegen `Vector256` (rủi ro 2 ở
  mục 7.7). Cách kiểm tra: xem target framework trong `tkv_compile.py` /
  `build\pyinstaller_src`, hoặc thử phát một lệnh `Vector256` và xem assembly sinh ra.
- **`bytearray`/`bytes` có map tới `byte[]` thật hay lại boxed?** Cần đọc
  `il_features_library/bytes_type.tkv` + `compiler/il_features/bytearray_type.py`.
  Nếu là `byte[]` thật thì còn một đường nữa: arena dùng mảng liên tục và để JIT
  tự vectorize, không cần đụng compiler.
- **`spawn()` là builtin gì và có tạo thread thật không?** Hiện nó **không** chạy
  worker ⇒ `DynamicBatcher` chạy inline. Nếu muốn chạy song song thật (multi-core ⇒
  tăng throughput), đây là câu hỏi cần trả lời trước, và nó cũng quyết định liệu có
  thể chia vòng khối `VEC_W` ra nhiều core.
- **Có thể bật `DOTNET_TieredCompilation`/`ReadyToRun` không?** Ảnh hưởng tới việc
  JIT có kịp tối ưu trong benchmark ngắn hay không.
