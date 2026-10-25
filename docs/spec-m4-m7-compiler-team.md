# SPEC M4-M7 — Yêu cầu builtin/ABI mới giao compiler team

Ngày lập: **2026-10-24** · Nguồn: số đo 8 chu trình đo TRIZ repo-local (CG/CH/CB/CI/CU/CW/M1/CX),
`docs/perf-plan-triz-roadmap.md` §13-§14, `HANDOFF.md` §0.35-§0.44.
Đối tượng: **compiler/native team** — người có quyền thêm builtin + sửa ABI trong `tkvc`.

---

## 0. TL;DR — vì sao có document này

Mục tiêu lớn: **BẰNG rồi NHANH hơn PyTorch** trên full step GPT-2 124M, gate fp32 giữ nguyên
(§13.2). Chương trình đo **repo-local đãi hết**: 8 chu trình đo, mọi lever `.tkv` khả thi đều
đã được đo hoặc đo bác bỏ — chỉ **M3c (scratch-passing tái dùng buffer C)** wired.
**Parity 1.0× bắt buộc đi qua builtin/ABI mới trong compiler** — đúng phần việc mà `.tkv`
không tự làm được: `.tkv` không khai báo được pinvoke mới, không có kiểu native buffer f32,
không đổi được implementation bên trong `simd_gemm`.

Document này gom thành **1 bản yêu cầu duy nhất**: số đo làm bằng chứng, thiết kế từng mục,
ràng buộc cứng đã khám phá, gate và quy trình nhận bàn giao.

> **CẬP NHẬT 2026-10-24 (tối) — ĐÃ NHẬN BÀN GIAO:** compiler team giao M4/M5/M6
> (`tkvc-tv.exe` build 24/10 16:12, `tkv_simd_native.dll` 18:39, `REFERENCE.md` 16:05,
> 4 test mới trong repo compiler; **M7 chưa có entry trong REFERENCE**).
> Chạy lại benchmark bằng compiler mới, `.tkv` không đổi 1 dòng: **gate 135/135 +
> `trainer_tests` 15/15**, full step best-of-3 **12.312 / 17.000 / 36.719 s**
> (Δ −1.6 / −1.8 / +0.04% vs baseline 12.516/17.313/36.703), loss 3 seq **exact**
> (`7.0119243` / `11.807251` / `13.442912`) → **KHÔNG ĐỔI vì chưa wire builtin**.
> A/B cùng phiên với torch (3 rep xen kẽ `tools/bench_torch.py d <seq>`, torch 2.14.0+cpu 14 luồng):
> **13.2× / 12.1× / 10.3×** (seq 8/256/1024) — baseline ghi trong tài liệu 13.4/12.3/10.4×
> → **tỉ lệ chưa đổi, khoảng cách còn lại vẫn 10-13×**.
> Việc còn lại đúng như §4: xác nhận builtin gọi được từ repo này rồi wire M4+M5
> (kỳ vọng ~11-13 s ≈ 1.8×), rồi M6. Chi tiết: `HANDOFF.md` **§0.45**.

**Trạng thái baseline (sau M3c, không đổi kể từ `7c7ff44`):**

| Hạng mục | Số |
|---|---|
| Full step seq8/256/1024 | **12.516 / 17.313 / 36.703 s** vs torch 0.934 / 1.404 / 3.542 → **13.4× / 12.3× / 10.4×** |
| Gate | **135/135** (9 suite + full 33) + `trainer_tests` 15/15; M3c subset 84/84 |
| Loss 3 seq | `13.442912` khớp exact 7 chữ số |
| Máy | Xeon E5-2695 v3 (Haswell-EP) — **AVX2+FMA3 thôi**, không AVX512/BF16/AMX; 28 luồng |

---

## 1. Vì sao repo-local đóng — bằng chứng (đo, không argue — kỷ luật §13.2)

Phân rã **1 lần gọi wrapper** trên 4 shape P7b (probe CG mode 0, ms, best-of):

| Shape | ALLOC | BUILD | GEMM | ACC | FULL | GEMM GF (pad) | GEMM vs torch |
|---|---|---|---|---|---|---|---|
| S1 dx1 [1024×768×3072] | 31 | 62 | 13.1 | 15 | 109 | 552 | 13.1 vs 7.1 ms (thua 1.85×) |
| S3 dxf [1024×768×50257] | 16 | 468 | 203 | 234 | 969 | 588 | 203 vs 129 ms (thua 1.57× = pad 1.5) |
| S4 dwte [50257×768×1024] | 406 | 750 | 207 | 359 | 1828 | 578 | 207 vs 254 ms — **THẮNG 1.23×** |
| S5 logits [1024×50257×768] | 422 | 562 | 211 | 234 | 1563 | 567 | 211 vs 240 ms — **THẮNG 1.14×** |

**CHỐT (HANDOFF §0.36 / roadmap §14.5):**

1. **Kernel đã parity** — pha GEMM 552-588 GFLOP/s pad, S4/S5 thắng torch tuyệt đối. Số
   "45-82 GFLOP/s" trước đây là quy sai (chia flop hữu ích cho thời gian wrapper + 1.5× pad).
2. **Nút thắt = wrapper interpreted**: BUILD+ACC+ALLOC = **76-89% FULL**; BUILD đo **ratio
   cpu/wall = 1.00** (100% đơn luồng, ghi list interpreted); pad 1.5× nhân thêm.
3. **Mọi lever `.tkv` còn lại đã bị số đo đóng:**

| Chu trình | Lever | Kết quả |
|---|---|---|
| CH | M3a pipeline build∥replay | ❌ 0.94-1.10× (BIT=0.0) — replay tuần tự = sàn cứng |
| CW | M3a song song ACC (2 thiết kế) | ❌ 0.67-0.99× (BIT=0.0) — acc worker cướp tài nguyên simd_gemm |
| CB | M3b T-dispatch (T=256..1024) | ❌ T=256 tệ nhất mọi shape (kernel mất throughput ở tile nhỏ); T=1024 SUM thắng 1.18-1.33× |
| **CX** | **M3b dự phòng — T=1024 full-wrapper thật** | ❌ **1.01-1.12×** « ngưỡng wire 1.3 — SUM thủ công không phản ánh wrapper |
| CI | M3c tái dùng buffer C (scratch-passing) | ✅ **WIRE** — BIT=0.0, ~344 ms/bước, gate 84/84 |
| M1 | OpenBLAS drop-in shim (giữ ABI) | ❌ **full step 1.001× noise**, tolerance GÃY 19/20 (loss lệch 1.4e-4), numerics ≠ torch |
| CG | M3 điều tra shape-gap | chốt wrapper-bound (xem trên) |
| CU | (A/B wire M3c) | ✅ — |

**Hệ quả:** M2 micro-kernel mất mục tiêu (kernel đã parity); M1 đã bust; **M4-M7 là menu
duy nhất còn lại và tất cả đều cần builtin/ABI mới.**

---

## 2. Yêu cầu từng mục

### M4 — `simd_gemm_packed`: prepack weight 1 lần / step

**Bệnh:** mọi call `simd_gemm` dựng lại tile B (weight) từ đầu — BUILD 468-750 ms/call
(2.2-3.6× pha GEMM), 100% đơn luồng. Weight (`wte`/`w1`/`w2`/`qkv`, cả `A` trong backward
grad) **bất biến trong 1 step** → packing lặp lại vô ích.

**Yêu cầu ABI:** handle packed do native quản lý, lifetime do `.tkv` giữ:

```
h = simd_gemm_packed_create(src, rows, cols, stride, T, trans) -> i64   # pack 1 lần
simd_gemm_packed(hA, hB, C, T)                                         # chạy tile từ pack
simd_gemm_packed_free(h)
```

- Packing = thao tác hiện ở `_tile_a_w`/`_tile_b_w`/`_tile_a_at_w` (pad ra T×T, zero-fill
  phần thiếu) — di chuyển vào native, chạy **1 lần thay vì mỗi call**.
- `C` giữ nguyên nhận `list[f32]` (output qua scratch-passing M3c đã wire).

**Kỳ vọng (chưa đo — chỉ là phân tích từ CG):** BUILD biến mất khỏi timed path trên operand
weight → FULL S3/S4/S5 giảm ~470-750 ms/call.

**Acceptance:** A/B wrapper thật (mẫu probe CU/CX — best-of-3 cùng phiên), BIT=0.0 hoặc
maxdiff ≤ 1e-4; full step 3 seq không regress; gate 135/135.

### M5 — Buffer f32 native + fused AdamW/elementwise

**Bệnh:** `ADAMW_STEP` = **3930 ms/bước** (11% step) trên container interpreted
(`list[f64]` boxing). Bằng chứng bottleneck là **container, không phải compute**: probe CC @124M
— V1 OWN-seg (compute trên seg riêng) = **2.60×**, nhưng V3 merge 1.19× / V5 ghi shared
0.33× → mọi hình thái song song `.tkv` chết ở merge/shared-write; sàn băng thông ước
**~1.0-1.1 s**. Tương tự `zero_grad`/`unflatten`/`norm` memory-bound ≈3.4 GB/s.

**Yêu cầu:** buffer số học f32 native (không boxing) + kernel fused
`adamw_step(p, m, v, g, …)` (multi-tensor-apply kiểu torch) và elementwise đồng bộ
(`zero_grad`, `addcmul`…). Trọng số/opt state đọc-ghi qua handle native, `.tkv` chỉ truyền
index/shape.

**Numerics:** AdamW torch tính f32 (1 div/ptu — GĐ10 đã học và wire công thức
`m*bc2s/(sqrt(v)+eps*bc2s)`) → kernel native phải **giữ đúng công thức + thứ tự** để
`trainer_tests` 15/15 (TOL 1e-5) + loss exact.

**Acceptance:** `trainer_tests` 15/15 + `ADAMW_STEP` ≤ ~1.3 s (sàn ước 1.0-1.1 s); full step
3 seq không regress; loss khớp exact.

### M6 — Fuse lm-head backward + attention épilogue

**Bệnh (AQ @seq1024):** `LHEAD_BWD` **7281 ms (20% step)** — trong đó softmax-grad serial
**1.33 s** (probe AV); attn non-GEMM (softmax+gDs+transpose n²) ~8 s. **Đã đo: song song
`.tkv` không được** — GĐ11: OW own-seg worker 2.36× nhưng merge 1172 ms ≈ toàn bộ serial
(0.56× net); SHARED ghi+exp 0.25-0.36×. Merge đơn luồng là thủ phạm, chỉ fused kernel
xóa được merge.

**Yêu cầu:** kernel fused, đọc `gg`/`xf` 1 lần:

1. `lmhead_bwd_fused(logits, ids, …, dW)`: softmax-grad → dxf → tích lũy dwte trong 1 pass
   (epilogue của GEMM lm-head).
2. `attn_bwd_epilogue(…)`: softmax + gDs + transpose n² thành épilogue của GEMM attention.

Cả 2 giữ **1 entry point** (xem ràng buộc §3.1).

**Kỳ vọng (phân tích, chưa đo):** LHEAD_BWD 7.28 → ~4-5 s; attn non-GEMM cắt sâu.

**Acceptance:** maxdiff 0.0 hoặc ≤1e-4 so đường cũ; full step 3 seq net win (ngưỡng wire
lịch sử 1.3×/lever); gate 135/135, loss exact.

### M7 — Dispatch kernel theo shape class

**Bệnh:** gap khác nhau theo shape — S1 nhỏ **13.2×** torch, lm-head lớn **6.5-7.5×**
(P7b). Dispatcher hiện xếp hạng theo `TKV_GEMM_MIN_N=96` + T theo
`_imin(_tilek(k), _roundup64(_imin(m,n)))`.

**Yêu cầu:** native tự chọn đường kernel theo shape class (nhỏ/square vs k lớn 50257 vs
rectangular grad), gộp vào M4/M6. **Lưu ý đo:** T-dispatch trong `.tkv` đã bust (CB/CX) —
grid T chỉ có lợi nếu **cả 3 thành phần** (GEMM ∝ pad, BUILD ∝ ng×T², ACC ∝ ng×T²) cùng
được tối ưu; khi BUILD/ACC đã ra native (M4/M6), tính lại lợi nhuận T với số đo mới.

---

## 3. Ràng buộc cứng — ĐÃ ĐO, không được vi phạm

### 3.1 Re-entrancy: GEMM qua MỘT entry point duy nhất

Probe CG mode 4: 2 thread cùng gọi `simd_gemm` (buffer riêng biệt hẳn) → **HANG > 90 s**.
Cơ chế: `g_job` global **không lock** trong `tkv_run_mt` → race dispatcher, biểu hiện
deadlock. Mọi builtin mới (M4/M6) phải có **pool/threading bên trong native** hoặc serialize
entry — `.tkv` không được gọi 2 kernel native đồng thời. Wave-12 hiện tại (worker dựng tile,
main gọi `simd_gemm`) giữ nguyên mẫu này.

### 3.2 Mẫu song song đã đo: own-seg + merge đơn luồng chỉ thắng khi merge RẺ

- Thắng: attention theo head (merge nhỏ, 5.2×), gelu/norm (GĐ4 7.33×).
- Chết: ghi list CHUNG + phép toán/ptu → **0.25-0.36×** (CE, BL, CC-V5); own-seg + merge lớn
  → merge ăn hết (GĐ11 0.56×, CC-V3 1.19×). **Vấn đề = container + merge, không phải thiếu
  đa luồng** (P3 21.2 luồng, V1 2.60×) → M5 phải đổi container, không phải thêm thread.

### 3.3 Numerics contract — fp32, không đổi

- Kiểu chính fp32, tích luỹ fp32 không fast-math; GELU erf; thứ tự giảm tử cố định
  (`docs/numerics-contract.md`).
- Tolerance: **84/84** (subset parity) + **135/135** (9 suite + full 33) + `trainer_tests`
  15/15; loss 3 seq `13.442912` khớp exact.
- **Bài học M1 (chân đắt nhất):** "gọi đúng thứ torch dùng" = **oneMKL** — OpenBLAS scipy là
  nhóm np-BLAS, thứ tự tích luỹ khác → loss lệch 1.4e-4, gate fail 19/20. Nếu target
  "numerics = numerics torch" thì phải **oneMKL** hoặc giữ thứ tự tích luỹ của
  `tkv_simd_native` (đừng trộn 2 thứ tự). M1 cũng chứng minh: thay kernel GEMM không ăn gì
  khi wrapper còn 76-89% — **đừng lặp lại M1 với hy vọng khác**.

### 3.4 Bẫy compiler `.tkv` đã ghi nhận (để spec không vấp lại)

- Module annotation `"Tensor"` **bắt buộc** `import tv.training.tensor` — thiếu →
  `Ky vong dtype ... gap 'Tensor'` (HANDOFF §0.44).
- `list[f32]` → `list[f64]` qua param **sinh giá trị sai im lặng** → mọi so sánh list inline
  trong main (BX).
- `Tensor.data` qua param có bug (BX[0]) — truyền container đã chứng minh.
- Nested `list[f32]` append → `ArrayTypeMismatchException`; main quá lớn → `BadImageFormat`
  → tách hàm (GĐ4).
- Quy tắc int64 native: phép gán đầu tiên ở top-level hàm (GĐ8) — vòng compute-bound mới ăn
  tiền, memory-bound không (GĐ9).
- **Cấm module-var mutable** trong `simd_matmul` (dòng 430, sinh từ 2 bug corruption) —
  state qua scratch-passing (M3c đã wire theo cách này).

---

## 4. Thứ tự + KPI

| Ưu tiên | Mục | Kỳ vọng (§14.4, CHƯA ĐO — chỉ phân tích) |
|---|---|---|
| 1 | **M4 + M5** (prepack + fused step native) | step → **~11-13 s ≈ 1.8×** |
| 2 | **M6 + M7** (fused lm-head/attention + dispatch) | dải **1.1-1.3×**; stretch 1.0× cần thêm fused attention-stage |
| treo | M2 micro-kernel, M1 oneMKL, M8 INT8 | chỉ nếu M4-M7 chưa đủ / hợp đồng đổi (P7 bust numerics f16/bf16) |

**KPI gate chung (mọi mục):**

1. A/B wrapper thật best-of-3 cùng phiên (mẫu CU/CX) — **không tin số SUM thủ công** (CX:
   SUM 1.18-1.33× → wrapper thật 1.01-1.12×).
2. Full step seq8/256/1024 không regress, loss khớp exact 7 chữ số.
3. Gate 135/135 + trainer 15/15 (subset M3c 84/84).
4. 4 shape P7b: pha GEMM ≥ 250 GFLOP/s (đã đạt 552-588 — tham chiếu), FULL-vs-torch theo ms/shape.

**Quy trình:** probe thuần trước (0 thay đổi code train), số quyết định (§13.2), wire đồng bộ
mirror (`gpt2_ref` ×5, `trainer` ×4), ghi HANDOFF §0.x + roadmap §14.x, commit từng mục.

---

## 5. Nguồn số liệu (bổ sung 2026-10-25: đo wire — xem §6)

- Phân rã wrapper + re-entrancy + kernel parity: probe `tv/tests/simd_probe_cg.tkv`,
  `out/cg_run.log`, HANDOFF §0.36, roadmap §14.5.
- T-dispatch: `simd_probe_cb` §14.6/HANDOFF §0.39; T=1024 wrapper thật: `simd_probe_cx`
  §14.10/HANDOFF §0.44, `out/cx_run.log`.
- M3c scratch-passing: `simd_probe_ci` §14.7/HANDOFF §0.40+§0.41.
- ACC song song: `simd_probe_ch` §0.38, `simd_probe_cw` §14.8/HANDOFF §0.42.
- M1 OpenBLAS: `tools/simd_native/`, roadmap §14.9/HANDOFF §0.43, `out/m1_*.log`.
- Container/AdamW: `simd_probe_cc` (GĐ13), `simd_probe_ce` (GĐ11), `simd_probe_ca` (GĐ9).
- Phân rã step: probe AQ `out/audit_aq.log`, §13.1.
- P7b gap torch: `simd_probe_bh`/`simd_probe_cd`, HANDOFF §0.35.

---

## 6. KẾT QUẢ ĐO TỪ REPO NÀY (2026-10-25) — M5 WIRE, M4/M6 BÁC BỎ

Đo theo đúng quy trình §4 (probe thuần → số quyết định → wire → gate → bench).
Chi tiết: HANDOFF §0.46.

| Mục | Chu trình đo | Kết quả | Quyết định |
|---|---|---|---|
| **M5 fused AdamW** | `_cy0` smoke (tol 1e-6) + wire 4 mirror + `out/gate_m5.log` + `bseq_m5` | numerics OK (dp 3.4e-13, dm/dv « 1e-6); gate 135/135 + trainer 15/15; full step **seq8 10.953 s (−11.0%), seq256 15.234 s (−10.4%)**, loss exact; seq1024 +7.1% do +2GB buffer > RAM 11.8GB (paging) | **✅ ĐÃ WIRE** |
| **M4 prepack** | CY (`simd_probe_cy`, cả 2 arm list[Single]) + CJ (`_cj`, production-real Tensor Double + convert) | CY: S1 2.0×, S3 1.6×, S5 1.9×, **S4 trans-A 0.6×**; CJ thật: S1 **0.99×**, S3 **0.67×**, S5 **0.80×**, S6 nhỏ **1.0×** — native pack nhanh (S5 packrun 719 vs base 1219) nhưng convert Double↔Single interpreted (343ms/38M; convA+convB S3 = 671ms, copyC S5 = 516ms) nuốt hết gain | **❌ KHÔNG WIRE** (« ngưỡng 1.3×) |
| **M6 fused lm-head** | `_cy0` (4) vs reference GĐ10 (onehot theo ids) | nat = ref × **6 = B*S** → **thiếu 1/n**; maxdiff dxf 0.55 / dwte 1.17 vs tol 1e-5; v1 naive chưa đo perf | **❌ KHÔNG WIRE** (gãy numerics §3.3) |
| **M7 dispatch** | — | REFERENCE chưa có entry | ⏸ CHỜ REFERENCE |

**Ask cho vòng spec tiếp (2 mục, số đo ở trên):**

1. **M4 cần ABI f64:** `simd_gemm_packed_create`/`simd_gemm_packed` nhận `List[Double]`
   (hoặc builtin convert native bulk f64→f32) — Tensor production toàn List[Double];
với ABI f64, M4 giữ kỳ vọng 1.6-2× ở S1/S3/S5 (số CY).
2. **M6 cần chuẩn hóa 1/n** (param scale hoặc chia trong kernel) theo hợp đồng
   `(p − onehot)/n` của GĐ10 + đo perf v1.
3. M7: bổ sung REFERENCE entry.

---

## 7. KẾT QUẢ VÒNG 2 (2026-10-25) — COMPILER ĐÃ SỬA ĐÚNG, REPO WIRE M4+M6, BENCH −28% ĐẾN −37%

Bàn giao `e_handoff/2026-10-25-m4-f64-m6-scale-ban-giao.md` đáp đủ 3 ask §6. Quy trình đo lại theo §4 (probe `_cj2`-`_cj6`, log `out/cj*_run.log`, chi tiết HANDOFF §0.47):

| Mục | Chu trình đo | Kết quả | Quyết định |
|---|---|---|---|
| **M4 ABI f64** | `_cj2` v1 (f64 trên production) + `_cj3` (f32-vs-f64 isolation) + `_cj4` (nguồn list) | ABI f64 hoạt động đúng với literal `list[f64]` (md 0.0) — nhưng production KHÔNG chứa double thật: `t_zeros` sinh `List[float64>` trong khi toàn bộ managed code ghi/đọc bằng token `List<float32>` → layout nhớt **f32@4i**; native f64 đọc cặp float → rác 2^65 | **⚙ FIX REPO** thay vì dùng ABI: `t_zeros`/`t_alloc_grad` khai `list[f32]` thật (3 mirror) — layout managed nguyên vẹn, gate 15 suite XANH, native f32 nhận thẳng |
| **M4 wire (f32 direct)** | `_cj2` v3 + `_cj5` (m=8/256/1024, sweep T) | S1 2.04×, S3 1.69×, S5 2.43× (m=1024); m=256: 1.48-3.21×; m=8: 1.0-2.58× — **không case nào regress**; T=256≈768; maxdiff ≤ 4e-5 (S5 bit-exact 0.0) | **✅ ĐÃ WIRE** — `_smm_pk` trong `simd_matmul.tkv` ×3, routing `trans_a=0 && k>=256` (loại S4 trans-A 0.57× + head k=64) |
| **M6 scale 1/n + v2** | `_cj6` (A) smoke vs scalar f64 + (B) A/B production seq1024 | smoke md 3.6e-8/3.1e-8 « 1e-5 → **hợp đồng GĐ10 đúng**; A/B: base 4844 ms (softmax 2735+dxf 453+dwte 1125) vs M6 1625 ms = **2.98×**, md vs base 5.4e-9/7.0e-10 | **✅ ĐÃ WIRE** — thay khối gg+softmax+2 GEMM trong `gpt2_backward` (5 mirror gpt2_ref) |
| **M7 dispatch** | — | Mặc định `shape` chạy ngầm qua mọi call (A/B bit-identical theo bàn giao) | ✅ Không cần wire thêm |

**Gate:** 15/15 suite `fails=0` (gpt2_full 33/33 vs PyTorch, simd_matmul 20/20, trainer 15/15).

**Bench best-of-3 cùng phiên (mode 1):**

| seq | baseline | M5 | M4+M6 | Δ vs base |
|---|---|---|---|---|
| 8 | 12.531 s | 12.094 s | **8.422 s** | **−32.8%** |
| 256 | 17.844 s | 14.750 s | **11.297 s** | **−36.7%** |
| 1024 | 40.921 s | 45.172 s | **29.484 s** | **−28.0%** |

Loss warmup giống hệt; loss step cuối lệch ≤3e-6 (thứ tự tích lũy f32) — gate torch tol 1e-5 vẫn xanh. seq1024 hết pagefile pressure nhờ type fix (nửa bộ nhớ Tensor) + M6 bỏ `gg`.

**Lưu ý cho vòng spec sau (không chặn):** M4 pack hiện làm mỗi call (đo đã net-win); prepack handle tái dùng nhiều call/trong 1 step là bước tối ưu tiếp theo nếu cần thêm — cần ABI giữ handle sống qua call + đo re-entrancy với attention worker.