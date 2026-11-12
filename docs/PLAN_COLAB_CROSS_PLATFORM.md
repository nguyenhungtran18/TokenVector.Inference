# PLAN — Cross-platform runtime for .tkv (colab / linux / macOS)

**Ngày:** 2026-11-12
**Báo cáo nguồn:** `docs/HANDOFF.md` §0.164; `BENCHMARKS.md` §9.18–§9.22; phân tích
bởi `docs/COMPILER_TEAM_REQUEST1–14` (yêu cầu cũ đều đi theo hướng tăng tốc native,
chưa có yêu cầu nào sửa vì lý do portable).
**Mục tiêu:** `.tkv` binaries chạy trực tiếp trên Windows / Linux / macOS, không cần
shim Windows-only nào; GPT-2 124M full step trên Colab CPU.
**Không phải phạm vi:** viết backend GPU; tối ưu SIMD cho Linux (đó là bước sau).

---

## 0. TRẠNG THÁI — PHASE A XONG (2026-11-12)

Đã làm, khác đề xuất ban đầu ở 3 điểm (đơn giản hơn, ít rủi ro hơn):

| Đề xuất §3 | Thực tế đã làm | Lý do |
|---|---|---|
| Xóa hẳn 2 P/Invoke kernel32 | **Giữ**, nhưng chuyển toàn bộ thân cũ sang `InitWin()`; `Init()` chỉ `call InitWin()` khi `OSVersion.Platform == Win32NT` | Giữ nguyên đường AVX2 native trên Windows (không hồi quy hiệu năng). Trên Linux `InitWin()` **không được JIT** nên P/Invoke không bao giờ được resolve |
| Dùng `Avx2.IsSupported` để detect | Không dùng | Compiler target .NET Framework 4.6 (không có `System.Runtime.Intrinsics`), và không cần: Linux luôn fallback |
| Chỉ sửa `Init` | Sửa thêm `_elem_block()`: bản fallback `...V` chuyển từ `Vector<T>` sang **vô hướng** | .NET 10 không resolve `Vector<T>.op_Addition` từ IL tham chiếu `[System.Numerics.Vectors]` → `MissingMethodException` (xem §9.24) |

**Kết quả:** `tools/vong.exe` → `IL_IDENTICAL`; `testkit_gate` **PASS 136/136**,
negative 44/44; `trainer_tests` **15/15 PASS**; `dotnet demo/bseq.dll` (124M) chạy
được trên WSL/Linux lần đầu tiên. Windows vẫn `isa=1` (native AVX2).

**Còn lại (Phase B):** port kernel AVX2 sang `System.Runtime.Intrinsics` để Linux
có SIMD thật (hiện chỉ có C `.so` lo GEMM/attention/AdamW, op phần tử là vô hướng).

---

## 1. Vấn đề chính xác

`TkvSimd.Init()` / `TkvSimd.Kern()` (compiler tự phát IL) gọi hai P/Invoke
**chỉ tồn tại trên Windows**:

| P/Invoke | Mục đích | Dùng ở đâu |
|---|---|---|
| `kernel32.dll!IsProcessorFeaturePresent(40)` | Phát hiện CPU hỗ trợ AVX2 (PF_X64_AVX2_INSTRUCTIONS_AVAILABLE) | gate chọn AVX2 path |
| `kernel32.dll!VirtualAlloc(RWX)` | Cấp vùng nhớ RWX để ghi machine code SIMD vào đó (~12 KB) và trỏ `tab[14]` vào đó | nạp kernel JIT |
| (không có) | `VirtualProtect` chưa được dùng | chưa cần |

Trên Linux / macOS cả hai đều không tồn tại → **DllNotFoundException** ngay lần
đầu hàm SIMD được gọi (điểm gọi: `TkvSimd.Kern` → `Init` → `IsProcessorFeaturePresent`).
Khi đó chỉ đường scalar / managed (`TkvSimd...V()`) còn chạy được, nhưng
**chúng không được chọn vì `isa` được đặt 0 bằng default** — `IsProcessorFeaturePresent`
ném lỗi trước khi quyết định.

**Hậu quả:** GPT-2 124M demo (và bất kỳ .tkv binary nào) không chạy được trên
WSL/Linux/Colab; chỉ scalar code chạy được nếu tránh SIMD hoàn toàn.

---

## 2. Hiện trạng code (đọc)

### 2.1 Khai báo IL của `_pt_native_ho_tro()`
`Release/build/selfhost/tv_gop.tkv` (bản compiler tu-host đang chạy, dòng 45807–45899):

- `.field` public static: `tab: IntPtr[]`, `st: int32`, `isa: int32`
- Hai P/Invoke `IsProcessorFeaturePresent` và `VirtualAlloc`
- `Init()`: set `isa=0`, `st=2` → đọc env `TKV_SIMD_ISA`; nếu `"off"` return; nếu
  `"sse"` → `isa=2`, chọn SSE blob; nếu không → `call IsProcessorFeaturePresent(40)` →
  `isa=1`, chọn AVX2 blob; nạp `fromBase64String`, `VirtualAlloc`, `Marshal.Copy`,
  điền `tab[14]` từ các offset đã biết; set `st=1`.
- `Kern(id)`: gọi `Init()` nếu `st != 1`; trả `tab[id]` nếu `isa != 0`, ngược lại trả 0.
- `Isa()`: trả `isa`.
- Các hàm `add`/`sub`/`dot`/`sum`/`axpy`/`scale`/`max`/`min`/`sgemm`… gọi `Kern(id)`
  để lấy raw function pointer và **gọi qua delegate không tên** (Interop V2 style).

### 2.2 Đường fallback đã có sẵn
Khi `isa == 0`, `Kern` trả `IntPtr.Zero`, và mọi call site (trong Interpreter? trong
`_simd_*_native_*`) đều có nhánh `if IntPtr.Zero == rsp -> scalar`. Đo thực tế: mọi
đường scalar/managed IL chạy đúng ở Windows kể cả khi tắt JIT.

**Tức là:** chỉ cần `Init()` không crash trên Linux/Mac → mọi binary chạy được,
chỉ là không nhanh bằng đường AVX2 JIT.

---

## 3. Kế hoạch chỉnh (2 giai đoạn)

### Giai đoạn A — runtime hoạt động trên mọi platform (giảm thiểu; khuyến ngh)

**Nơi sửa:** `Release/build/selfhost/tv_gop.tkv` — hàm `_pt_native_ho_tro()` và các
phụ trợ liên quan (`_KERN_AVX2_B64`, `_KERN_SSE_B64`, `_KERN_*_OFFS` — giữ nguyên).

**Các thay đổi:**

1. **Xóa P/Invoke kernel32.** `Init()` dùng `System.Runtime.Intrinsics.X86.Avx2.IsSupported`
   để phát hiện (một dòng IL thuần, portable, đã có trong .NET reference assemblies).
   Cách detection không cần quyền, không cần getenv, không cần platform-specific code.
   Tuy nhiên bước này *chỉ để xác định đường chạy*; chọn đường nào ở bước 3.

2. **`VirtualAlloc` thay bằng gì?** Hai phương án:
   - **Phương án A1 (đơn giản nhất, khuyến nghị):** không nạp JIT kernel;
     đặt `isa=0, st=1` khi `Avx2.IsSupported == true` (hoặc `Environment.Version`
     trên platform bất kỳ) — mọi call site rơi xuống managed / scalar path.
     Ưu điểm: **không phải viết lại kernel AVX2**; runtime chạy ngay
     trên Linux/Mac; an toàn tuyệt đối về tính đúng.
   - **Phương án A2:** giữ JIT kernel nhưng nạp lại qua `mprotect` (POSIX) —
     `libc!mprotect` ở Linux, `VirtualProtect` ở Windows — phát hiện platform
     bằng `RuntimeInformation.IsOSPlatform`. Phức tạp hơn (pointer layout của RWX
     page khác nhau), rủi ro hỏng memory protection; không khuyến nghị cho giai đoạn A.

3. **Call site giữ nguyên:** không đụng `Kern(id)`, `Isa()`, `vec_*` call sites.
   `st` chuyển mốc `2 → 1` sau khi init xong để `Kern` không gọi lại; `Isa()` vẫn
   trả 0 → thông báo "không có AVX2 kernel", caller fallback scalar.

4. **ENV overrides một phần hoạt động:** `TKV_SIMD_ISA=off` → `isa=0`;
   `TKV_SIMD_ISA=avx2` → `isa=1` (chỉ khi `Avx2.IsSupported`; nếu không, `isa=1`
   là vô hại, mọi đường gọi đều check `Avx2.IsSupported` trước khi dùng pointer).

**Chấp nhận được:** hiệu năng (bằng scalar khỏi phải lo); loss EXACT như gate hiện tại
(vì fallback đường managed/scalar đã verify bit-exact ở Windows).

**Ước lượng công sức:** nhỏ — 1 file, ~20 dòng IL, build lại self-host, verify gate.
**Phụ thuộc:** quyền build compiler tu-host; không cần Windows SDK, không cần tests mới.

### Giai đoạn B — phục hồi hiệu năng AVX2 kiểu .NET x86 intrinsics

**Nơi sửa:** compiler emission — viết lại các kernel (sgemm, axpy, dot, fmax,…)
dưới dạng **IL gọi `System.Runtime.Intrinsics.X86.Avx2`/`Vector256<float>`** thay vì
nạp machine code qua `VirtualAlloc`. .NET JIT của VM sẽ tự dịch IL intrinsics
thành máy AVX2 khi chạy trên CPU hỗ trợ — **không còn phụ thuộc OS**.

Các kernel chủ yếu đã có ở dạng dòng cha: `simd_fma` (FMA), `simd_axpy`, kernels
GEMM blocked (TransA/Rect), `simd_fused_*` (attention_b2c, dw_fused,…) — chỉ cần
thêm một tầng emission `_emit_avx2(op, …)` và thay `.Kern = nạp machine code`
bằng `.Kern = reference Avx2 intrinsic`.

**Chấp nhận được:** phải chọn intrinsic đúng cho từng kiến trúc: `Sse2` cho CPU
x86 cũ không có AVX2, `Avx`/`Avx2` cho x86 máy mới, `Arm64.AdvSimd` (đề xuất sau)
cho ARM (Colab T4/A100, Apple Silicon).

**Ước lượng:** lớn — 1–2 tuần cốt lõi (port các kernel AVX2 sang bản C#/IL `Avx2.*`),
phụ thuộc chuyên gia SIMD + reviewer compiler.

---

## 4. Các files/hàm cần sửa (đối chiếu lại)

**Compiler (nguồn đang chạy, `tv_gop.tkv`):**
- `_pt_native_ho_tro()` — `Init`, `Kern`, `Isa` — xóa P/Invoke ×2, sửa `Init` body
- Xóa `_KERN_AVX2_OFFS`, `_KERN_SSE_OFFS`, `_KERN_AVX2_B64`, `_KERN_SSE_B64`
  (nếu chọn A1; giữ nếu chọn A2)
- `_kmap[]` — `call conv.i` / `delegate Invoke` — giữ nguyên; return zero ok
- `TkvSimdInit` / `TkvSimdInit_v2` … — có thể xóa; call sites không đổi

**Call sites (các hàm `simd_*` ở vùng phụ trợ interpreter):** giữ nguyên.

**Tests (gate_f32 / pack_tests / gap_tests):** không thay đổi — gate phải XANH
trên Linux/WSL trước khi merge; chạy bộ gate bằng `bash tools/gate_f32.sh` (rồi cập
nhật `tools/run_cases.ps1` cho Linux/WSL).

**Docs:** `docs/HANDOFF.md` §0.16x — ghi chú
"cross-platform (Phase A)"; `docs/linux-portability-spike.md` — cập nhật kết luận
(2026-10-17) và §9.17 (binary không kèm runtimeconfig.json); README quickstart —
`sh build_inference.ps1 → dotnet-gate; mono/dpnet dotnet out/x.exe`.

---

## 5. Rủi ro & điều kiện dừng

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| Compiler tu-host không chạy được ở Linux/WSL vì self-host cần .NET SDK gate | Cao | CI gate trên Linux (dotnet-sdk-10.0, tools/gate_f32.sh) |
| `TkvSimd.Init` gọi ở **mọi** hàm SIMD, kể cả scalar path, làm mọi binary treo ở Linux | Trung bình | Giữ `TkvSimd.Init` gọi được từ mọi đường; fallback sẵn |
| `Avx2.IsSupported` khi build IL → `unverifiable code detected` | Thấp | Verify bởi ilasm; nếu lỗi → dùng `Type.GetType("System.Runtime.Intrinsics.X86.Avx2")` check IsSupported bằng reflection |
| Phase A giữ scalar path → hiệu năng Colab quá chậm để "thử nghiệm hợp lệ" | Trung bình | Phase A là nền tảng; Phase B mới có hiệu năng cạnh tranh |

**Tiêu chí hoàn thành (DoD)**
- `pack_tests`, `gap_tests`, `gate_f32.sh`, `trainer_tests`, `gpt2_full_tests` chạy trên
  **Linux/WSL** không lỗi truy cập `kernel32.dll`.
- `dotnet demo/trainer_tests.dll` pass trên Linux.
- `demo/colab_bench_ai.ipynb` chạy được, in thời gian nhưng không bắt buộc cạnh tranh
  tốc độ (Phase A).

---

## 6. Thứ tự thực hiện

1. Branch `port-cross-platform` (nếu đã có branch port; otherwise fork) trên
   `TokenVector` compiler repo.
2. Giai đoạn A: `Init()` mới — PR "portable init without kernel32 P/Invoke".
3. CI gate: `bash tools/gate_f32.sh && pwsh -File tools/run_cases.ps1` trên Linux.
4. Cập nhật `docs/HANDOFF.md` + `BENCHMARKS.md` §9.23 (chạy Colab bench).
5. Giai đoạn B (sau khi A ổn định): port AVX2 kernels sang `System.Runtime.Intrinsics` IL.

---

## 7. Ghi chú bảo trì

- `docs/linux-portability-spike.md` (2026-10-17) đã kết luận compiler không chạy Linux —
  ghi chú đúng cho *compiler source*, không phải cho *binary biên dịch*. Binary biên dịch
  chạy trên Linux nếu runtime là Linux (Phase A làm được).
- Ghi chú `docs/HANDOFF.md` §0.151: `pack_tests` crash khi `Avx.IsSupported` (Linux) sau khi
  sửa run — thực ra `pack_tests` không gọi SIMD; lỗi thực do `string` P/Invoke `kernel32.dll`
  `ReadFile/WriteFile` — nằm ngoài phạm vi.

---

*Kế hoạch này là bước 2 của user request (benchmark on Colab) — bước 1 (notebooks cho MLP
bench) đã hoàn thành và commit. Bước 2 là thay đổi compiler, phụ thuộc quyền build self-host.*
