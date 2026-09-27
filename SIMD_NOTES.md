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
| Bộ nhớ unmanaged | **Không tồn tại** | `native_arena_alloc(...) -> 0` là stub (`tv/runtime/session.tkv:228`) |
| FFI sang native | **Có** | `il_features_library/ffi_feature.tkv` (`ctypes_cdll`, `c_call`) |

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

1. **Căn chỉnh 64 byte là vô nghĩa.** `ExecutionMemoryArena.__init__` có
   `(size_in_bytes + 63) // 64 * 64` với comment `# Align up to 64 bytes (AVX2 /
   AVX-512)` (`tv/runtime/session.tkv:106`). Căn chỉnh byte không tạo được vector
   load trên mảng boxed.
2. **Không lấy được con trỏ thật.** `native_arena_alloc` trả `0`; các hàm
   `native_arena_float_view` / `native_arena_int8_view` / `native_arena_int32_view`
   cũng là stub trả `[]`. Nghĩa là **con đường FFI cũng đang bị chặn**, vì không
   có địa chỉ nào để đưa cho kernel C.

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
| **A** | Arena dùng buffer liên tục (`bytearray` → `f32` view) + giữ khối `VEC_W` | Có | Cao, trong repo này | Bỏ boxing; JIT **có thể** tự vectorize vòng đơn giản; bỏ được chi phí 2–3×, chưa phải SIMD thật |
| **B** | Thêm vector type + intrinsic vào compiler (grammar, `il_codegen.py`, runtime helper) | Có | Rất cao, phải sửa `D:\TokenVector\3.code` | SIMD thật, khối 8-lane thành 1 lệnh vector |
| **C** | FFI sang DLL C biên dịch AVX2/AVX-512 | **Không** | Thấp (nếu đã có con trỏ thật) | Nhanh nhất, nhưng đánh đổi tuyên bố kiến trúc |

Hướng C hiện **bị chặn bởi chính `native_arena_alloc` stub** — phải làm A (hoặc
tương đương) trước thì C mới dùng được.

Khuyến nghị: **A trước, B ngược lên upstream compiler, C giữ làm escape hatch.**
Lý do: A là tiền đề kỹ thuật cho cả B lẫn C, và A nằm gọn trong repo này nên tự
làm được.

## 6. Lộ trình gợi ý

### Giai đoạn 0 — Đo trước khi sửa (bắt buộc)
Benchmark hiện in `0.0 us`, `0 B`, `0.0x` vì các hàm thời gian là stub trả 0
(`now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`, `fill_rand`, `fill_const`,
`ptr_of`, `parallel_for` — định nghĩa ở cuối `compare_rivals.tkv`). **Chưa có số
liệu thật thì không chứng minh được cải tiến.** Nối các hàm này vào stdlib thật
trước.

### Giai đoạn 1 — Bỏ boxing khỏi arena
1. Thay `fmap: dict[i64, list[f32]]` bằng buffer liên tục, ví dụ
   `fbuf: dict[i64, 'bytearray']` + `foff: dict[i64, i64]` (offset tính theo byte).
2. Cung cấp lớp đọc/ghi `f32` từ byte theo offset — cần **reinterpret**; đây là
   kỹ thuật chưa kiểm chứng là tồn tại trong TokenVector. Nếu không có, phương án
   thay thế là `list[f32]` phẳng dùng chung thay vì `dict` nhiều entry (giảm phân
   mảnh, chưa bỏ được boxing).
3. Giữ nguyên ngữ nghĩa zero-copy: `copy_from`/`copy_to`/`bind` không được sao chép
   toàn bộ tensor mỗi lần gọi.
4. Có test hồi quy: 26 case `inference_tests` + 12 case `gap_tests` phải xanh, và
   kết quả số học phải **giống hệt** trước/sau (không chỉ "chạy được").

### Giai đoạn 2 — Tận dụng khối `VEC_W` hiện có
1. Chuẩn hoá mọi vòng khối về biên `int`.
2. Tách thân vector và thân đuôi thành helper dùng chung để không nhân bản.
3. Kiểm chứng xem RyuJIT có vectorize hay không bằng cách **đo** (Giai đoạn 0), đừng
   đoán.

### Giai đoạn 3 — Đề xuất ngược lên compiler (không làm trong repo này)
Vector type `f32x8` + intrinsic `f32x8_load` / `f32x8_mul` / `f32x8_store`,
ánh xạ xuống IL dạng `System.Runtime.Intrinsics.X86.Avx`. Cần chỉnh:
`typed_dsl_parser.py` (grammar), `il_codegen.py` (codegen + dispatch),
`il_features_library/` (thêm feature file), và test trong `D:\TokenVector\3.code\test`.

## 7. Bẫy đã biết (rút từ phiên này)

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

## 8. Câu hỏi còn mở

- TokenVector có primitive **reinterpret bytes → f32** không? (`struct.unpack`,
  `memoryview`, hay tương đương). Không có thì Hướng A bị giới hạn nặng.
- `bytearray`/`bytes` trong TokenVector có map tới `byte[]` thật hay lại boxed?
  Cần đọc `il_features_library/bytes_type.tkv` + `compiler/il_features/bytearray_type.py`.
- Benchmark target là .NET 8 nào? Tính năng `Vector128/Vector256` của
  `System.Runtime.Intrinsics` có được bật mặc định không.
- Có `parallel_for` / thread thật trong runtime không, hay hiện cũng chỉ là stub?
