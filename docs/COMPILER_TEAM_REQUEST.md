# YÊU CẦU COMPILER TEAM — ĐƠN VỊ TÍNH NĂNG F32 NATIVE

## Ngày: 2026-10-25
## Đối tượng: Compiler Team (tkvc native/ABI)
## Trạng thái: M4–M6 đã wire xong ở `.tkv` level, M7 + f32 locals chờ compiler

---

## 1. TỔNG QUAN

Sau khi hoàn thành wire M4 (`simd_gemm_packed`), M5 (fused AdamW f32), M6 (lm-head backward fused), full step GPT-2 124M hiện tại vẫn **6.36× – 8.33× slower** so với PyTorch 1-thread (seq8/256/1024). **Chok-point duy nhất** là **f32 locals unavailable** — `.tkv` bắt buộc dùng f64 intermediates thông qua container `list[f64]` boxing, mỗi call wrapper 76–89% time là BUILD+ACC+ALLOC interpreted (không compute).

Mục tiêu: Yêu cầu compiler team cung cấp builtin/ABI f32 mới trong `tkvc` để `.tkv` dùng f32 thay f64, giảm wrapper overhead và đạt ≤1.0× vs PyTorch 1-thread.

---

## 2. VẤN ĐỀ HIỆN TẠI

| Mức performance | Giá trị |
|----------------|---------|
| Full step seq8 | **8.422 s** (vs PyTorch 1.323 s → **6.36×**) |
| Full step seq256 | **11.297 s** (vs PyTorch 1.404 s → **8.05×**) |
| Full step seq1024 | **29.484 s** (vs PyTorch 3.542 s → **8.33×**) |
| Wrapper overhead (mỗi call) | **76–89%** time là BUILD+ACC+ALLOC interpreted |
| Nguyên nhân | `.tkv` f64 intermediates + list boxing, không có builtin f32 |

---

## 3. YÊU CẦU CHO COMPILER TEAM

### Yêu cầu 1: Builtin f32 locals (Cấp bách)

| # | Yêu cầu | Chi tiết | KPI mong đợi |
|---|---------|---------|-------------|
| 1 | **Tạo builtin f32** | Hàm dựng sẵn (0 tham số) trả `f32` literals, hoặc `f32` register keyword trong `tkvc` IL | `.tkv` có thể ghi `f32 x = 1.0` trực tiếp (không phải `list[f64]`) |
| 2 | **ABI convert f64→f32 bulk** | `simd_gemm_packed_create` nhận `List[Double]` nhưng convert f64→f32 xảy ra **trong DLL**, không qua interpreted | M4 giữ gain 1.6–2× (côCY) hoặc thậm chí hơn khi f32 tràn thẳng |
| 3 | **f32 buffer primitives** | `f32_zero`, `f32_copy_to_buf`, `f32_copy_from_buf`, `f32_addcmul` — đã có sẵn spec (section 2.13 REFERENCE.md) nhưng chưa hoạt động vì param type | M5 `adamw_step` f32-native, mất 11% overhead bây giờ do boxing |

### Yêu cầu 2: Dispatch kernel f32 theo shape class (M7)

| # | Yêu cầu | Chi tiết | KPI mong đợi |
|---|---------|---------|-------------|
| 1 | **Giữ TKV_GEMM_DISPATCH=shape** | Mặc định đã có sẵn tại native level — giữ nguyên | Numerics bit-identical giữa chế độ `shape` và `legacy` (đã test 4 test) |
| 2 | **Không đổi export DLL** | Các env đo vẫn có hiệu lực: `TKV_GEMM_MIN_N`/`TKV_GEMM_THREADS`/`TKV_GEMM_COLS` | Không breakpoint현有 |
| 3 | **M7 entry vào REFERENCE.md** | Đã có ở native level — chỉ cần update docs compiler team nếu cần | Không ảnh hưởng runtime |

---

## 4. KẾ QUẢ ĐỰC TIÊN (sau compiler team cung cấp)

| seq | Hiện tại (sau M4+M5+M6) | **Dự kiến sau f32 locals** | **Δ vs PyTorch 1T** |
|-----|------------------------|--------------------------|-------------------|
| 8 | **8.422 s** | **~1.5–2.0 s** | **~1.5× (thắng PyTorch)** |
| 256 | **11.297 s** | **~1.8–2.2 s** | **~1.8× (thắng PyTorch)** |
| 1024 | **29.484 s** | **~3.8–4.5 s** | **~1.1–1.3× (thắng PyTorch)** |

**Tóm lại:** Giảm khoảng **60–70%** so với baseline sau M4+M5+M6, đạt hoặc vượt mặt PyTorch 1-thread.

---

## 5. EVIDENCE GHI NHẬN (tại repo này)

| Chu trình | Kết quả | Hữu ích cho yêu cầu |
|-----------|---------|---------------------|
| **CH (M3a song parallel ACC)** | ❌ 0.67–0.99× (BIT=0.0) — acc worker cướp tài nguyên simd_gemm | Chứng minh: *"container + merge đơn luồng chết"* → cần f32 native, không cần thêm thread |
| **CX (M3b dự phòng T=1024 wrapper thật)** | ❌ **1.01–1.12×** « ngưỡng wire 1.3 — SUM thủ công không phản ánh wrapper thật** | Dispatch kernel đã hoạt động, nhưng wrapper vẫn chiếm 76-89% |
| **M5 wire (fused AdamW)** | seq8 −11.0%, seq256 −10.4%, seq1024 +7.1% (do paging RAM) | Đã wire xong, nhưng vẫn qua f64 intermediates — f32 locals sẽ rút ngắn thêm |
| **P6 multithreading (Embedding backward)** | seq8: 281→16ms (~17.5×); full step: 8,687→8,646ms (−0.5%) | Chỉ giúp phần embedding, không full step — xác nhận bottleneck là GEMM + wrapper |

---

## 5. CÔNG THỨC TRIỂN KHAI (theo spec §13.2, §14.4)

```powershell
# 1. Compiler team thêm builtin f32 vào tkvc IL
#    Ví dụ: cú pháp `f32 x = 1.0` biên dịch thành register f32 (không qua list[f64])

# 2. Đổi ABI simd_gemm_packed_create
#    - Hiện tại: nhận List[Double], convert f64→f32 qua DLL (slow)
#    - Cần: nhận List[Double] nhưng convert f64→f32 **trong DLL** (fast path)
#    - Hoặc: overload `simd_gemm_packed_create(src, rows, cols, stride, T, trans, dtype=f32)`

# 3. Thêm f32 buffer primitives (đã có spec, cần enable)
#    - f32_zero(p, n) — memset 0 buffer native
#    - f32_addcmul(dst, x, y, scale, n) — fused AVX2+FMA
#    - adamw_step(p, m, v, g, n, lr, b1, b2, eps, wd, bc1s, bc2s) — fp32 version

# 4. Giữ TKV_GEMM_DISPATCH=shape (mặc định)
#    - Không đổi export DLL
#    - Numerics bit-identical 2 chế độ (test đã pass 4 test)
```

---

## 6. KPI SUCCESS CRITERIA (cùng test)

| KPI | Mức success |
|-----|------------|
| **Full step seq8** | ≤ 2.0 s (vs 8.422 s hiện tại) |
| **Full step seq256** | ≤ 2.5 s (vs 11.297 s hiện tại) |
| **Full step seq1024** | ≤ 5.0 s (vs 29.484 s hiện tại) |
| **Gate 135/135** | Pass (subset M3c 84/84, trainer 15/15) |
| **Loss exact** | Khớp 7 chữ số PyTorch (toleranced 1e-5) |
| **Numerics contract** | fp32, GELU erf, thứ tự tích lũy cố định |

---

## 7. TÀI LIỆU THAM KHẢO (để compiler team inspect)

- `docs/spec-m4-m7-compiler-team.md` — spec đầy đủ 4 mục M4–M7
- `HANDOFF.md §0.45–§0.47` — kết quả đo round 2 (M4 f32 wire, M6 scale 1/n)
- `tools/simd_native/tkv_simd_native_m1.c` — M1 OpenBLAS shim (đã bust, lý do: wrapper 76-89%)
- `REFERENCE.md` section 2.13 — hàm builtin f32/adamw/lmhead_bwd_fused (đã register nhưng chưa enable f32 path)
- `out/gate_m5.log` — gate results sau M5 wire
- `out/cj*_run.log` — kết quả A/B recent

---

## 8. LỘ TRÌNH SAU KHI COMPILER TEAM REPLY

| Step | Mô tả |
|------|-------|
| **Week 1** | Compiler team cung cấp builtin f32 + ABI update |
| **Week 2** | `.tkv` test builtin (smoke test `_cy0`, `_cj2` với f32 path) |
| **Week 3** | Run gate 135/135 + trainer 15/15 + loss exact |
| **Week 4** | A/B so sánh vs PyTorch 1-thread/14-thread (`tools/bench_torch.py d 8/256/1024`) |
| **Week 5** | Nếu pass: commit + tag version; nếu fail: iterate spec |

---

**Tóm lại:** Yêu cầu này là **chok-point duy nhất** để đạt ≤1.0× vs PyTorch. M4–M6 đã wire xong ở `.tkv` level, nhưng **f32 locals** cần compiler team cung cấp builtin/ABI mới trong `tkvc`. Khi có f32 path, toàn bộ wrapper overhead 76–89% có thể giảm xuống ~10–20%, và full step sẽ đạt hoặc vượt mặt PyTorch 1-thread.

**Mong nhận được phản hồi từ compiler team trong vòng 1 tuần.**
---

## 10. KẾT QUẢ ĐO SAU KHI COMPILER GIAO f32() (2026-10-26) — YÊU CẦU 1.1 BỊ ĐO BÁC BỎ

Compiler đã giao `f32(x)` builtin (handoff `2026-10-26-f32-builtin-compiler-reply.md`).
Repo này smoke 6/6 PASS, gate 135/135 + trainer 15/15 xanh, rồi **đo A/B thật**
(trước khi wire — probe `tv/tests/_f32p.tkv`, best-of-3 × 4 lần chạy):

| Case | So sánh | Speedup | BIT_MAXDIFF |
|---|---|---|---|
| [1] BUILD tile (`_tile_a_w` copy + zero-fill) | `z: f64` vs `z: "f32"` | **~1.0×** (0.92–1.20×, nhiễu) | 0.0 |
| [2] ACC (`_acc_f` production) | không local vs `v: "f32"` | **0.97–1.0×** | 0.0 |
| [3] ARITH thuần 12M phep | f64 local vs f32 local | **~1.0×** | 0.0 |

**CHỐT: BUST — không wire (ngưỡng 1.3× theo §13.2).** Giả định "f32 locals giảm
wrapper overhead 76–89% xuống 10–20%" **SAI trên số đo**: BUILD/ACC đã chạy trên
`list[f32]` (List[Single]) từ đầu — element ops là f32; chi phí BUILD+ACC+ALLOC
thật = **list indexing + append/bounds-check interpreted**, không phải widen f32↔f64.

**Yêu cầu 1.1 khép lại.** Khoảng cách còn lại vs PyTorch (~6.3×) cần yêu cầu mới
hướng vào **list access interpreted** — ví dụ ABI native build-tile/acc nhận
`list[f32]` và trả buffer, hoặc bulk copy builtin (`f32_list_copy(dst, src, n)`
native) thay vòng `while` interpreted — **đã viết thành `docs/COMPILER_TEAM_REQUEST2.md` (2026-10-26): Ưu tiên A `simd_gemm_full` (BUILD+GEMM+ACC trong DLL, 1 ABI) / Ưu tiên B 3 bulk builtin, bằng chứng bảng CG + `_f32p`.**
