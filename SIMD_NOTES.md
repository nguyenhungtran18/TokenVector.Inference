# SIMD thuần TokenVector — khảo sát & lộ trình

> **ĐÃ SỬA 2026-10-15.** Bản đầu viết theo compiler cũ (`D:\TokenVector\3.code`, nay đã
> bị xoá) và kết luận **"TokenVector không dùng được SIMD" — kết luận đó SAI**.
> Compiler self-host mới **đã có sẵn `simd_*`**. Toàn bộ tài liệu viết lại theo
> compiler đang chạy: `D:\TokenVector\Release\dist\tkvc-tv.exe`
> (nguồn: `Release/build/selfhost/tv_gop.tkv`, `Release/compiler_src/*.tkv`).
> Lịch sử thay đổi: mục 9.

Liên quan: `HANDOFF.md`. Báo cáo phía compiler:
`D:\TokenVector\Release\docs\e_handoff\2026-10-15-inference-repo-build-blocker.md`.

---

## 1. Kết luận (đã kiểm chứng bằng chạy thật)

**TokenVector CÓ SIMD ở tầng ngôn ngữ.** Compiler tự sinh một lớp IL `TkvSimd` với
13 builtin, viết bằng `System.Numerics.Vector<T>`:

| Nhóm | Builtin |
|---|---|
| Phần tử | `simd_add`, `simd_sub`, `simd_mul`, `simd_scale`, `simd_axpy` |
| Thu gọn | `simd_dot`, `simd_sum`, `simd_max`, `simd_min` |
| Lane / phần cứng | `simd_lanes_f32`, `simd_lanes_f64`, `simd_hw`, `simd_isa` |

Cơ chế: `simd_*(a, b, o)` phát lệnh `TkvSimd::AddF32(a[], b[], o[])` … với hậu tố
theo kiểu phần tử (`_hau_to`: `f32→F32`, `f64→F64`, `i32→I32`, còn lại `I64`).
Ngoài ra có `System.Numerics.Vector4` (`_VEC4_BINOPS` với `op_Addition`,
`op_Subtraction`, `op_Multiply`, …) cho biểu thức vector 4 lane.

**Nhưng hiện chưa chạy được.** Đường duy nhất bị chặn ở runtime:

```
Unhandled Exception: System.IO.FileNotFoundException:
Could not load file or assembly 'System.Numerics.Vectors, Version=4.1.4.0' ...
   at TkvSimd.MulI32(Int32[] a, Int32[] b, Int32[] o)
```

`System.Numerics.Vectors.dll` không có trong repo compiler (không có DLL nào trong
`Release/`) và không được copy cạnh `.exe` sinh ra. ⇒ **compile được, chạy là nổ.**

## 2. API dùng thế nào (đúng như docstring `sample_simd.tkv`)

```tokenvector
import numpy as np

def main() -> "f32":          # <- kiểu trả về quyết định kiểu phần tử của np.zeros
    a = np.zeros(37, dtype=np.float32)
    b = np.zeros(37, dtype=np.float32)
    o = np.zeros(37, dtype=np.float32)
    simd_mul(a, b, o)         # ghi kết quả vào o
    return simd_sum(o)
```

Hai điều dễ sai:

1. **`np.zeros` lấy kiểu phần tử từ kiểu trả về của hàm.** Probe của tôi ban đầu
   khai `main() -> "i32"` nên ra mảng `int32` và `simd_mul` gọi đúng `MulI32` — không
   phải bug dispatch. Nếu `main() -> "f32"` thì ra `MulF32`.
2. **`simd_*` chỉ nhận 3 mảng cùng kiểu**; lệch là báo lỗi ngay
   (`ba mang phai CUNG kieu phan tu`).

Mẫu đầy đủ (7 nhóm dtype f32/f64/i32/i64, 7 phép): `Release/tests_suite/de_vao/sample_simd.tkv`.

## 3. Vì sao repo `TokenVector.Inference` chưa dùng được

Không phải vì thiếu SIMD, mà vì **tensor của repo không phải mảng liên tục**:

| Yếu tố | Trạng thái ở repo inference |
|---|---|
| Arena | `ExecutionMemoryArena.fmap: dict[i64, list[f32]]` — **boxed**, mỗi phần tử là một tham chiếu heap (`tv/runtime/session.tkv`) |
| Kiểu mảng cho `simd_*` | Cần mảng số liên tục kiểu numpy; repo không có |
| `native_arena_alloc` | stub `return 0` ⇒ chưa có bộ nhớ unmanaged nào |
| Khối 8-lane | `tv/optimization/kernels.tkv:8` có `VEC_W = 8` và vòng khối — nhưng thân khối vẫn **scalar** từng phần tử |
| `tv.numerics.*` | stub rỗng do `tkvc` sinh ra; **không có source canonical trong repo** |
| Compiler còn dùng được không | ⛔ **Không** — xem mục 4 |

Còn lại đúng một việc: **đổi arena sang mảng numpy liên tục, rồi thay thân khối
`VEC_W = 8` bằng `simd_*`.** Không cần thêm gì vào compiler.

## 4. Chặn cứng: runtime chưa biên dịch được với compiler mới

`tv/runtime/session.tkv` làm compiler nổ:

```
[tkvc] loi: ...: The given key was not present in the dictionary.
```

Đây là lỗi phía compiler (Python-style `KeyError`), **không phải** lỗi repo. Đã loại
trừ bằng thực nghiệm: stub cả 105 method, cả 8 `__init__`, cả 33 hàm cấp module,
bỏ từng import — tất cả vẫn crash. Chi tiết đầy đủ + cách tái hiện trong file
handoff ở mục trên.

Ngoài ra repo phải sửa 4 quy tắc ngôn ngữ mới trước khi build được (tất cả đã
migrate, chưa commit): `.shape[N]` không index được, không index trên kết quả lời
gọi hàm, `//`/`%` cần nhớ tạm, và trùng tên hàm giữa 2 module là lỗi cứng.

## 5. CLI đã đổi

```
tkvc <nguon.tkv> <ra.exe> [ten_ham_vao]
```

Không còn subcommand `build`, không còn `--out`. `argv()` là stub trả `[]` ⇒ **exe đã
biên dịch không nhận được tham số dòng lệnh** (`sample_simd.tkv` với entry `run_hw`
nổ `IndexOutOfRangeException` vì đọc `argv()[0]`).

## 6. Bẫy đã biết (rút từ các phiên trước)

1. **Đừng build lẻ từng module.** `tkvc build <module>` (compiler cũ) tự ghi lại bản
   sao import `tv.*.tkv` và từng làm 9 file thành byte NUL. Với compiler mới chưa
   quan sát lại, nhưng `build_inference.ps1` vẫn chỉ build entry point + có guard.
2. **Sửa canonical ⇒ phải sync copy.** Compiler resolve import phẳng `tv.*.tkv` cạnh
   entry. Sửa `tv/runtime/session.tkv` mà quên copy thì test vẫn xanh nhưng **đang
   test code cũ**:
   ```powershell
   Get-ChildItem -Recurse -Filter 'tv.runtime.session.tkv' | ForEach-Object { Copy-Item tv\runtime\session.tkv $_.FullName -Force }
   ```
3. **Trùng tên hàm giữa 2 module là lỗi cứng** (`trung ten ham [...]`). Đặt tên helper
   duy nhất theo module: `_shp_at_sd`, `_shp_at_tr`, …
4. **Chuỗi chỉ nhận dấu nháy kép.** `print(..., flush=1)` không hỗ trợ. `len()` trả
   `TkvInt` — dùng làm biên `while` hoặc so sánh với số nguyên là nguồn lỗi nhiều lần.
5. **`__init__` của record bị bỏ qua lúc biên dịch**; khởi tạo phải ở factory +
   gán tay (`new_embedded_server` làm mẫu).
6. **Hằng số cấp module không import được** giữa module — khai báo lại ở file dùng.
7. **Đừng tin số liệu benchmark của repo.** `now_ms`/`now_us`/`gc_allocated_bytes`
   trong `compare_rivals.tkv` là stub trả `0`, nên suite in `0.0 ms` và `req/s = 5e14`.

## 7. Lộ trình (rút gọn — không còn phần "sửa compiler")

1. **Đóng gói `System.Numerics.Vectors`.** Đây là việc **duy nhất chặn SIMD**, và nó
   thuộc phía compiler repo. Sau khi có, `simd_*` chạy được ngay.
2. **Đo trước khi sửa.** Nối `now_ms`/`now_us` thật trong `compare_rivals.tkv`, và
   một hàm đo riêng cho `simd_*` so với vòng scalar. Không có số liệu thật thì không
   chứng minh được cải tiến.
3. **Đổi arena sang mảng numpy liên tục.** Giữ `dict[i64, list[f32]]` cho đường
   scalar và thêm nhánh vector, để rủi ro thấp. Lưu ý: khởi tạo phải qua
   `arena_create()` (vì `__init__` bị bỏ qua), và phải giữ ngữ nghĩa zero-copy của
   `copy_from`/`copy_to`.
4. **Thay thân khối `VEC_W = 8`.** Mỗi khối 8 phần tử → `simd_mul(a, w, o)` rồi cộng
   dồn bằng `simd_sum`. Đuôi dư (`while j < d`) giữ nguyên scalar. Chuẩn hoá biên về
   `int` trước.
5. **Test hồi quy:** 26 case `inference_tests` + 16 case `gap_tests` phải xanh **và
   kết quả số học phải giống hệt** (không chỉ "chạy được"). Đây là điều kiện bắt
   buộc — lỗi âm thầm kiểu `int32` thay vì `f32` là loại bug chỉ lộ ra khi assert giá trị.

## 8. Câu hỏi còn mở

- `System.Numerics.Vectors` nên được **copy cạnh `.exe`** hay **nhúng vào IL**? Nếu
  nhúng, `.exe` vẫn tự chứa như quảng cáo.
- Bề rộng vector thực tế là bao nhiêu? Mã tham chiếu `System.Numerics.Vector4`
  (128-bit) và có `simd_hw`/`simd_isa` để chọn theo phần cứng — nhưng không thấy tham
  chiếu `Vector128`/`Vector256`. Cần đo `simd_lanes_f32` trên máy thật.
- `simd_*` có bản AVX2 (`Vector256`, FMA) không, hay chỉ 4 lane?
- Có `simd_*` cho kiểu `i8`/`f16` không? (INT8 định lượng của repo inference sẽ cần.)
- Bao giờ `argv()` được nối? CLI của repo inference phụ thuộc vào nó.

## 9. Lịch sử: điều chỉnh so với bản đầu

| Mục | Bản đầu (sai) | Bản này |
|---|---|---|
| Kết luận | "không có vector type/intrinsic, phải sửa compiler" | **Có `simd_*` sẵn**; không cần thêm dtype/instruction nào |
| Bề rộng | — | `Vector4` 128-bit + `simd_hw`/`simd_isa`; chưa rõ có AVX2 không |
| Đường đi | thêm `f32x8`, `v_load`/`v_store`, `read_f32`… vào compiler | đổi arena sang mảng numpy liên tục rồi dùng `simd_*` |
| `alloc_bytes` / FFI | đã có, có con trỏ thật (đúng) | vẫn có (8 chỗ tham chiếu trong compiler mới) |
| Rào cản lớn nhất | thiếu intrinsic ở tầng ngôn ngữ | **thiếu assembly lúc chạy** + **runtime repo chưa biên dịch được** |