# YASU COMPILER TEAM #13 — `ZeroListF32` cấp phát 2 lần + vòng scalar thừa

**Ngày:** 2026-11-10
**Nhị:** HANDOFF §0.137 + `BENCHMARKS.md` §9.3/§9.8/§9.9
**Đối tượng:** Compiler team (`compiler_bootstrap/compiler/il_features/slicing.tkv` + `build/selfhost/tv_gop.tkv`)
**Ưu tiên:** cao — đây là **~220 ms/bước** trên seq1024, và là hạng mục lớn nhất còn lại sau khi đã bỏ hướng trans-A.
**Khác #10/#11/#12:** lần này **không phải** thêm native op mới. Chỉ sửa **IL thân của một method có sẵn** (`TkvSlice::ZeroListF32`). Không đổi ABI, không đổi builtin signature, không đổi ngữ nghĩa `.tkv`.

---

## 1. Bối cảnh — số đo thực tế

### 1.1 Chỗ tiền nằm

Trong `_f124_bwd` (production path, seq1024), có chuỗi 4 lần đụng mảng 38.6M phần tử:

| # | chỗ | ms |
|---|---|---:|
| 1 | `dwte = t_zeros([VocabSize, h])` | 125 |
| 2 | `gwte_e = t_zeros([VocabSize * h])` | 125 |
| 3 | `f32_list_add(dwte, gwte_e, V*h)` | 63 |
| 4 | `f32_list_add(gflat, dwte, V*h)` | 63 |

Mục #1 và #2 đều đi qua `t_zeros` → `zeros_list(n)` → `TkvSlice::ZeroListF32(n)`.

### 1.2 Đo trực tiếp trên .NET 10 (cùng máy, 147.2 MB, best-of-5)

Harness: `out/zltest/Program.cs` (chạy được, `dotnet run -c Release`).

| cách | ms | GB/s |
|---|---:|---:|
| **A. IL hiện tại** (`newarr` + vòng `stelem.r4` + `List(IEnumerable)`) | **102.2** | **1.41** |
| B. bỏ vòng scalar, giữ `List(IEnumerable)` | 60.0 | 2.40 |
| C. `newarr` + `List(n)` + `AddRange` | 64.2 | 2.24 |
| **D. `List(n)` + `SetCount` (reflection)** | **15.8** | **9.12** |
| **F. `List(n)` + `CollectionsMarshal.SetCount`** | **15.7** | **9.16** |

Cả 5 cách đều cho `count = 38597376`, toàn `0.0`, ghi/đọc `z[0]`/`z[n-1]` bình thường.

**A → F = 6.5×.** Trên `.tkv` đo được 125–172 ms (chậm hơn harness vì thêm chi phí thông dịch), nhưng tỷ lệ giữa các cách là như nhau.

### 1.3 Nguyên nhân: IL hiện tại làm hai việc thừa

Đọc IL thật trong `slicing.tkv` L171–188 (và bản copy `tv_gop.tkv` L33813–33830):

```
.method public static List`1<float32> ZeroListF32(int32 n) cil managed
{
  .locals init (int32 n0, float32[] arr, List`1<float32> z, int32 i)
  ldarg.0 ; stloc.0
  ldloc.0 ; ldc.i4.0 ; ble TKVZM_RZ
TKVZM_OK:
  ldloc.0 ; newarr float32 ; stloc.1        <-- (1) CLR ĐÃ zero sẵn mảng này
  ldc.i4.0 ; stloc.3
TKVZM_CL:
  ldloc.3 ; ldloc.0 ; bge TKVZM_CD
  ldloc.1 ; ldloc.3 ; ldc.r4 0.0 ; stelem.r4  <-- (2) ghi 0.0 vào từng phần tử: THỪA
  ldloc.3 ; ldc.i4.1 ; add ; stloc.3 ; br TKVZM_CL
TKVZM_CD:
  ldloc.1
  newobj List`1<float32>::.ctor(IEnumerable`1<!0>)  <-- (3) CẤP MẢNG THỨ HAI + copy 147 MB
  stloc.2 ; ldloc.2 ; ret
TKVZM_RZ:
  newobj List`1<float32>::.ctor() ; ret
}
```

Ba vấn đề, xếp theo mức tốn:

1. **(3) là chính.** `new List<T>(IEnumerable<T>)` không nhận mảng làm backing store. Nó `new T[n]` mới rồi `Array.Copy` 147 MB vào. Vậy tổng là: 1 mảng 147 MB (zero bởi CLR) + 1 mảng 147 MB nữa (copy) = **294 MB cấp phát + 147 MB copy** cho một buffer cần 147 MB.
2. **(2) là thừa hoàn toàn.** `newarr` đã bảo đảm mọi phần tử = 0. Vòng 38.6M lần `stelem.r4 0.0` chỉ tốn thời gian. Đo được: A − B = 42 ms.
3. **Comment trong source sai.** Docstring ở `tv_gop.tkv` L33704 và `slicing.tkv` L62 ghi *"alloc List<float32>(n), SetCount, Span.Fill(0)"*. IL thật **không** làm vậy — không có `SetCount`, không có `Span.Fill`. Comment này có từ một phiên bản trước và đã lệch khỏi code. Nên sửa luôn để người sau không tin nhầm.

---

## 2. Ask — sửa thân `ZeroListF32`, không đổi chữ ký

**Không đổi:** tên method, chữ ký `List<float32> ZeroListF32(int32 n)`, hành vi `n <= 0` → list rỗng, ngữ nghĩa `zeros_list(n)` ở tầng `.tkv`, ABI builtin.

**Chỉ đổi:** thân IL.

### Phương án ưu tiên — `CollectionsMarshal.SetCount`

```il
.method public static class [mscorlib]System.Collections.Generic.List`1<float32>
        ZeroListF32(int32 n) cil managed
{
  .maxstack 8
  .locals init (class [mscorlib]System.Collections.Generic.List`1<float32> z)
  ldarg.0
  ldc.i4.0
  ble TKVZM_RZ
  ldarg.0
  newobj instance void class [mscorlib]System.Collections.Generic.List`1<float32>::.ctor(int32)
  stloc.0
  ldloc.0
  ldarg.0
  call void [System.Runtime]System.Runtime.InteropServices.CollectionsMarshal::SetCount<float32>(
        class [mscorlib]System.Collections.Generic.List`1<!!0>, int32)
  ldloc.0
  ret
TKVZM_RZ:
  newobj instance void class [mscorlib]System.Collections.Generic.List`1<float32>::.ctor()
  ret
}
```

Lý do chọn cách này:
- `new List<T>(n)` cấp backing array **một lần**; CLR zero-fill sẵn (đúng ngữ nghĩa `zeros_list`).
- `CollectionsMarshal.SetCount` chỉ set `_size = n`, **không** copy, **không** cấp thêm.
- Là API công khai, có từ .NET 8, không reflection, không `Unsafe`, không phụ thuộc layout nội bộ của `List<T>`.
- Đo được **15.7 ms** — ngang phương án reflection (15.8) nhưng không có rủi ro vỡ khi Microsoft đổi layout.

### Phương án dự phòng (nếu IL toolchain không resolve được `CollectionsMarshal`)

Dùng `TkvSlice::SetCountF32` tự viết trong cùng class, dùng `Unsafe.As` trên offset `_size`:

```il
.method public static void SetCountF32(
        class [mscorlib]System.Collections.Generic.List`1<float32> z, int32 n) cil managed
{
  .maxstack 8
  ldarg.0
  ldc.i4.s 16            // x64: obj header 8 + _items 8 = 16
  conv.i
  add
  ldarg.1
  stind.i4
  ret
}
```

**Cảnh báo:** cách này phụ thuộc layout `List<T>` của runtime. Chỉ dùng nếu Ask chính bị chặn, và phải có test chặn (xem §4).

### Điều cần kiểm tra trước khi chọn

IL sinh ra hiện tại chỉ khai `.assembly extern mscorlib`, `System.Core`, `System`, `System.Numerics`, `System.Numerics.Vectors`, `System.Runtime.Intrinsics` (kiểm trên `out/zeroscale.il`). **Chưa có `System.Runtime`.** `CollectionsMarshal` nằm trong `System.Runtime`, nên có thể phải thêm `.assembly extern System.Runtime` vào header IL. Đây là việc của compiler team, không phải của `.tkv`.

Nếu thêm được `System.Runtime` thì Ask chính là cách sạch nhất. Nếu không, dùng dự phòng.

---

## 3. Kỳ vọng

| hạng mục | hiện tại | sau khi sửa | tiết kiệm |
|---|---:|---:|---:|
| `t_zeros([V,h])` 147 MB | 125–172 ms | ~16–25 ms | ~110–145 ms |
| `gwte_e = t_zeros([V*h])` 147 MB | 125 ms | ~16–25 ms | ~110 ms |
| **tổng mỗi bước seq1024** | | | **~220–250 ms** |

Mục tiêu ≤1.0× so với torch cần cắt **494 ms** ở seq1024 (TKV 4578 vs torch 4040 ms, full step). Ask này chiếm **~45%** khoảng cách đó.

**Lợi ích lan rộng:** `t_zeros` được dùng ở khắp codebase, không riêng gpt2. Mọi chỗ cấp buffer lớn đều ăn theo. Đây là lý do nên sửa ở compiler thay vì vá từng chỗ trong `.tkv`.

---

## 4. Điều kiện bắt buộc

- **Ngữ nghĩa giữ nguyên:** `zeros_list(n)` trả `List<float32>` có đúng `n` phần tử, mọi phần tử `== 0.0f`, `Count == n`, ghi/đọc `z[i]` với `0 <= i < n` hoạt động, `z.Add(...)` sau đó hoạt động (capacity đã đủ nên không grow).
- **`n <= 0`** → `new List<float32>()` rỗng, như hiện tại.
- **Loss EXACT** trên cả 3 seq: `10.572291 / 10.756927 / 10.760399`.
- **Gate 135/135** (9 suite nhỏ = 102 assert + `gpt2_full` 33, fails=0).
- **Trainer 15/15.**
- **Self-host `tools/vong.exe` → `IL_IDENTICAL`.**
- **Test chặn mới** (đề nghị thêm vào suite): `zeros_list` với `n` = 0, 1, 7, 1000, 38597376 → kiểm `Count`, kiểm mẫu `z[0]`, `z[n/2]`, `z[n-1]` đều `0.0`, ghi rồi đọc lại đúng. Test này phải **fail** nếu ai đó quay lại dùng `List(IEnumerable)` sai cách hoặc offset `_size` lệch.
- **Không regress seq8 / seq256.** Đo cả 3 seq, best-of-3, xen kẽ torch cùng phiên.

---

## 5. Cách đo lại sau khi làm

```powershell
# 1) so 4 cach cap phat tren .NET that (harness da co san)
cd out\zltest
dotnet run -c Release

# 2) do tren .tkv that: zeros_list o 5 kich thuoc
D:\TokenVector\Release\dist\tkvc-tv.exe tv/tests/_zeros_scale.tkv out/zeroscale.exe
.\out\zeroscale.exe

# 3) do tren duong production
#    tv/tests/_seq_cfg.txt = "1024 / 1 / 1 / 1"
D:\TokenVector\Release\dist\tkvc-tv.exe tv/tests/simd_bench_124m_seq.tkv out/bseq.exe
.\out\bseq.exe

# 4) parity
D:\TokenVector\Release\dist\tkvc-tv.exe tv/tests/gpt2_full_tests.tkv out/gft.exe
```

**Tiêu chí đạt:**
- `_zeros_scale`: `zeros_list(38597376)` ≤ **30 ms** (hiện 125–172 ms).
- `simd_bench_124m_seq` mode 1 @seq1024: full step ≤ **4350 ms** (hiện 4578 ms).
- Loss EXACT cả 3 seq, gate 135/135, `IL_IDENTICAL`.

---

## 6. Ghi chú phương pháp (để không ai đo sai lại)

- **Probe phải mirror đúng số thread và đúng BLAS của production.** Một probe trước đó đo `lmhead_bwd_fused` ra 924 ms trong khi production là 542 ms, chỉ vì probe chạy softmax 1 thread và không nạp `libscipy_openblas64_.dll`. Chênh gần 2×.
- **`TKV_GEMM_THREADS` là biến ẩn.** `tkv_nthreads` (L284) tự nhân đôi lên 28 logical khi `n >= 512`, dù bench khai 14. Đo A/B: 28 nhanh hơn 14 đúng 1.28×. Mọi so sánh "14 vs 14" đang cho TKV bất lợi.
- **Đừng tin comment trong source hơn IL.** Docstring của `ZeroListF32` ghi `SetCount + Span.Fill` nhưng IL thật là `newarr + vòng stelem + List(IEnumerable)`. Luôn đọc IL sinh ra (`out/*.il`) trước khi kết luận.
- **`f32buf_wrap` không thay được `t_zeros` ở tầng `.tkv`.** Nó là statement, trả con trỏ thô `f32buf`, không phải `List<float32>`; chỉ dùng được trong `main` và không truyền được vào `lmhead_bwd_fused_list` (builtin đó đòi đúng `list[f32]`). Đã thử và revert. Đó là lý do việc này phải làm ở compiler.
