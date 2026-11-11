# YÊU CẦU COMPILER TEAM #2 — MATMUL TRỌN TRONG DLL (BUILD + GEMM + ACC)

## Ngày: 2026-10-26
## Đối tượng: Compiler Team (tkvc native/ABI)
## Trạng thái: Yêu cầu #1 (`f32()` builtin) ĐÃ GIAO + ĐO BÁC BỔ (xem §4) — yêu cầu mới nhắm root cause thật
## Tiền đề: M4/M5/M6/P6 đã wire ở repo `.tkv`, gate 135/135 + trainer 15/15, gap vs PyTorch 1T còn 6.4–8.3×

---

## 1. TÓM TẮT — asks chính

**Ưu tiên A (khuyến nghị): MỘT ABI gọi trọn matmul**

```
i32 simd_gemm_full(a, b, c, m, n, k, ndim, trans_b, T, mode)
```

Nhận **3 `list[f32]`** (Tensor.data thật, không handle), làm **toàn bộ BUILD tile + GEMM +
ACC trong DLL**, ghi `c` xong trả về. `.tkv` không còn vòng `while` nào quanh GEMM.

**Ưu tiên B (fallback nhỏ): bulk builtins** di chuyển từng phần ra native:

```
i32 f32_list_fill(dst, off, n, v)                       # zero-fill (ALLOC + pad)
i32 f32_block_copy(dst, d0, src, s0, stride, rows, cols, T)  # BUILD tile (copy + zero-pad T×T)
i32 f32_list_acc(cd, c0, src, s0, n)                    # cd += src (ACC)
```

A thắng B: B vẫn để nhiều pinvoke round-trip + vòng điều phối `.tkv`; A đóng 1 lần.

---

## 2. VÌ SAO — bằng chứng đã ĐO (không argue)

### 2.1 Wrapper phân rã 4 shape P7b (probe CG, HANDOFF §0.36 / spec M4-M7 §1)

| Shape | ALLOC | BUILD | GEMM | ACC | FULL | GEMM GF (pad) | GEMM vs torch |
|---|---|---|---|---|---|---|---|
| S1 dx1 [1024×768×3072] | 31 | 62 | 13.1 | 15 | 109 | 552 | 7.1 ms (thua 1.85×) |
| S3 dxf [1024×768×50257] | 16 | 468 | 203 | 234 | 969 | 588 | 129 ms (thua 1.57× = pad) |
| S4 dwte [50257×768×1024] | 406 | 750 | 207 | 359 | 1828 | 578 | 254 ms — **THẮNG 1.23×** |
| S5 logits [1024×50257×768] | 422 | 562 | 211 | 234 | 1563 | 567 | 240 ms — **THẮNG 1.14×** |

- **BUILD+ACC+ALLOC = 76–89% FULL**; BUILD đo cpu/wall = **1.00** (100% đơn luồng — ghi list interpreted).
- **GEMM kernel ĐÃ parity** (552–588 GFLOP/s pad; S4/S5 thắng torch tuyệt đối) → kernel không phải mục tiêu.
- Re-entrancy: 2 thread gọi `simd_gemm` đồng thời → **HANG >90 s** (`g_job` global không lock) — entry mới phải giữ ràng buộc MỘT entry point (spec §3.1).

### 2.2 f32 locals = YÊU CẦU #1 ĐÃ ĐO BÁC BỎ (probe `_f32p`, 2026-10-26)

Compiler đã giao `f32()` builtin; repo đo A/B thật (best-of-3 × 4 lần chạy, batch >300 ms):

| Case | So sánh | Speedup | BIT_MAXDIFF |
|---|---|---|---|
| BUILD tile (`_tile_a_w` copy + zero-fill) | `z: f64` vs `z: "f32"` | **~1.0×** (0.92–1.20× nhiễu) | 0.0 |
| ACC (`_acc_f` form production) | không local vs `v: "f32"` | **0.97–1.0×** | 0.0 |
| ARITH thuần 12M phep | f64 vs f32 local | **~1.0×** | 0.0 |

**CHỐT:** đổi kiểu local **không giảm được gì** — BUILD/ACC đã chạy trên `list[f32]`
(List[Single]) từ đầu; chi phí thật = **list indexing + bounds-check interpreted
mỗi phep trong vòng `while`**, không phải widen f32↔f64. → `.tkv` hết lever;
chỉ dời vòng ra native mới attack được root cause.

### 2.3 Mọi lever `.tkv` còn lại cũng đã bust (spec §1 bảng chu trình)

CH pipeline 0.94–1.10× · CW song-song ACC 0.67–0.99× · CB T-dispatch 0.94–1.33×
· CX T=1024 wrapper thật 1.01–1.12× · M1 OpenBLAS shim full step 1.001× (và gãy
tolerance) — **tất cả « ngưỡng wire 1.3×**.

---

## 3. CHI TIẾT YÊU CẦU

### 3.1 Ưu tiên A — `simd_gemm_full` (một ABI, toàn bộ trong DLL)

```
i32 simd_gemm_full(a, b, c, m, n, k, ndim, trans_b, T, mode)
```

| Tham số | Ý nghĩa |
|---|---|
| `a`, `b`, `c` | `list[f32]` (List[Single]) — layout **giống wrapper hiện tại**: `a` [m][k]; `b` [k][ndim] nếu `trans_b=0`, [n][k] nếu `trans_b=1`; `c` [m][n] |
| `m, n, k, ndim` | shape như `_smm_core` |
| `T` | tile size (`.tkv` truyền 256 như hiện tại, hoặc native tự chọn nếu证明 an toàn) |
| `mode` | `0` = ghi đè `c` (C = A×B), `1` = tích lũy `c += A×B` (thay `_acc_f` trong p-loop) |

**Native chịu trách nhiệm:** build tile A/B (copy + zero-pad T², kể cả nhánh
`trans_b`, `trans_a` nếu mở rộng), gọi GEMM pool hiện có, acc Ct→C — **toàn bộ
không quay lại `.tkv`**. Thread pool nội bộ giữ nguyên (đã đo 22 luồng thật).

**Ràng buộc cứng:**
1. **Re-entrancy:** pool phải an toàn khi 2 thread cùng gọi (bài học CG mode 4 HANG) —
   hoặc giữ MỘT entry point như `simd_gemm` cũ (đúng cảnh báo `simd_matmul:419`).
2. **Numerics (contract §3.3):** thứ tự tích lũy `i0 → p0 → j0`, tile-walk giống
   wrapper hiện tại để nhắm **BIT=0.0**; nếu native được phép reorder → maxdiff ≤ 1e-4
   (f32 ulp) và phải khai báo trong reply.
3. **Không đổi export cũ** — `simd_gemm`, `simd_gemm_packed_*`, env
   `TKV_GEMM_MIN_N`/`TKV_GEMM_THREADS`/`TKV_GEMM_COLS` giữ nguyên (các probe cũ còn chạy).

**Tiện ích đi kèm (không bắt buộc):** overload nhận `list[f64]` cho `a`/`b` như bài
học M4 (convert trong DLL) — Tensor production đã fix `t_zeros` sang List[Single]
nên **không chặn** yêu cầu này.

### 3.2 Ưu tiên B — 3 bulk builtin (nếu A quá lớn cho vòng này)

| Builtin | Thay cái gì | Từ số CG |
|---|---|---|
| `f32_list_fill(dst, off, n, v)` | zero-fill C (ALLOC) + pad `z` trong `_tile_*_w` | ALLOC 16–422 ms/call |
| `f32_block_copy(dst, d0, src, s0, stride, rows, cols, T)` | `_tile_a_w`/`_tile_b_w`/`_tile_a_at_w` (copy block row-stride, zero-pad về T²) | BUILD 62–750 ms/call |
| `f32_list_acc(cd, c0, src, s0, n)` | `_acc_f` (cd += Ct, tích lũy theo thứ tự tile-walk) | ACC 15–359 ms/call |

Acceptance từng builtin: A/B wrapper thật (mẫu CU/CX), **BIT=0.0**, không regress gate.

---

## 4. YÊU CẦU #1 — TRẠNG THÁI ĐÓNG (tham khảo)

`f32()` builtin đã giao 2026-10-26 (handoff `2026-10-26-f32-builtin-compiler-reply.md`),
repo smoke 6/6 PASS + negative đúng, gate 135/135 + trainer 15/15 xanh — **nhưng A/B
`_f32p` cho ~1.0× cả 3 case → BUST, không wire** (HANDOFF §0.49). Giả định "f32 locals
giảm wrapper 76–89%" sai trên số đo. **Không cần làm thêm gì cho #1** — yêu cầu mới
này thay nó.

---

## 5. KPI / ACCEPTANCE (đo ở repo này)

| Mục | Chỉ tiêu |
|---|---|
| A/B wrapper 4 shape P7b (best-of-3, cùng phiên) | FULL giảm về gần GEMM + ε: kỳ vọng S3 ~969→≤350 ms, S4 ~1828→≤400, S5 ~1563→≤400, S1 109→≤40 — **nếu < 1.3× thì BUST** (ngưỡng §13.2) |
| Numerics | BIT=0.0 (ưu tiên) hoặc maxdiff ≤ 1e-4 có khai báo |
| Gate | 135/135 + trainer 15/15, loss 3 seq khớp exact 7 chữ số |
| Full step 3 seq (best-of-3 vs PyTorch 1T) | baseline hiện tại 6.36× / 8.05× / 8.33× (seq8/256/1024) → **kỳ vọng ≤ 3×** sau khi BUILD+ACC+ALLOC về native |
| RAM | không tăng (seq1024 đã gần trần — đo peak, đề phòng paging) |

---

## 6. NGUỒN SỐ LIỆU

- `docs/archive/spec-m4-m7-compiler-team.md` §1 (bảng CG) + §3 (ràng buộc cứng)
- `docs/HANDOFF.md` §0.36 (CG), §0.49 (f32 BUST), §0.47 (M4+M6 wire), §0.48 (M5-opt + PyTorch)
- Probe: `tv/tests/simd_probe_cg.tkv` (phân rã wrapper), `tv/tests/_f32p.tkv` (f32 locals BUST)
- `docs/COMPILER_TEAM_REQUEST.md` §10 (kết quả đo yêu cầu #1)
- Handoff compiler: `D:\TokenVector\Release\docs\e_handoff\2026-10-26-f32-builtin-compiler-reply.md`

**Mong phản hồi trong vòng 1 tuần.** Nếu A quá lớn, trả lời B (3 bulk builtin) —
repo wire ngay và đo theo §5.

---

## §6. PHẢN HỒI ĐO TỪ REPO (2026-10-26) — WIRE THÀNH CÔNG m≤256, CẦN v2 CHO m=1024

Đã nhận `simd_gemm_full` (reply `2026-10-26-simd-gemm-full-compiler-reply.md`), smoke +
gate + A/B đo tại repo này (probe `tv/tests/_gf0.tkv`, `_gf1.tkv`, HANDOFF §0.50):

**Đạt:**
- Smoke 5/5 PASS (mode 0/1, tb 0/1, edge) — **BIT=0.0 vs `_smm_pk`** ở shape đơn lẻ.
- Gate 135/135 + trainer 15/15 xanh cả trước và sau wire.
- **WIRE routing:** `m ≤ 256 → simd_gemm_full`, `m > 256 → _smm_pk` (2 site
  `_smm_core`/`_smm_core_pd`). Full-step A/B xen kẽ: **seq8 −8.0%** (7813→7187 ms),
  **seq256 −12.1%** (10937→9609 ms), seq1024 giữ nguyên (loss EXACT 13.442915 — routing
  không đổi).

**Chưa đạt KPI §5 (đúng cảnh báo trung thực §3.2 của reply)** — đo best-of-5 T=256 với
probe thật của repo, so `simd_gemm_full` vs `_smm_pk` M4 đang wire:

| shape | m=8 | m=256 | m=1024 |
|---|---|---|---|
| S1 [m×768×3072] tb1 | thắng | 2.07× | 1.0× |
| S3 [m×768×50257] tb0 | ~hòa | 1.26× | 0.83× (453→547 ms) |
| S5 [m×50257×768] tb1 | 2.01× | 1.63× | **0.57×** (422→735 ms) |

Threshold ≤350-400 ms (§5) chưa chạm ở m=1024: full = 547/735 ms. **Nguyên nhân đã đo:**
B-tile build lại mỗi `(i0,p0,j0)` — số lần gọi tile-B nhân `ceil(m/T)` (m=1024 → 4× so
m=256), trong khi `_smm_pk` pack nguyên ma trận 1 lần/call. m≤256 full thắng vì tránh
được pack; m=1024 thua vì lặp build.

**Yêu cầu v2 (theo hướng tối ưu B §3.2 reply):** **B-tile cache theo i0** (giữ tile B đã
build qua các i0, hoặc pack B 1 lần bên trong `SgemmFull` rồi chạy tile loop) — mục tiêu
S3/S5 m=1024 ≤ 400 ms và BIT giữ nguyên 0.0 vs wrapper. Khi có v2, repo sẽ nới routing
(`m ≤ 1024` hoặc bỏ nhánh m) và đo lại gate + bench 3 seq.

Ghi chú: BIT `simd_gemm_full` vs `_smm_pk` (packed) ≤ 3.6e-5 tại repo (khác thứ tự tile,
trong tol gate 1e-4; loss lệch ≤9.5e-6) — với `_smm_core` cổ điển vẫn 0.0 như test của
các bạn. Overload `list[f64]` chưa cần.
