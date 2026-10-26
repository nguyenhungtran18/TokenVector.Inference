# YÊU CẦU COMPILER TEAM #3 — NATIVE DATA PLANE: BULK LIST OPS + GRAD_NORM + ADAMW G-LIST + 2 BUG

## Ngày: 2026-10-26
## Đối tượng: Compiler Team (tkvc native/ABI)
## Trạng thái: Phase A (đối chuẩn 2 phía) XONG — HANDOFF §0.52, log `out/bench_torch_e.log`. Yêu cầu #2 (`simd_gemm_full`) đã giao + wire m≤256 (§0.50–0.51).
## Bối cảnh: full step seq8 = 7187 ms vs torch 14T = 1190 ms (6.1–7.7×). Mục tiêu ≤1× theo `PLAN_TO_PARITY.md` — Phase B = native data plane, ước tiết kiệm ~2.3–2.4 s/step.

---

## 1. TÓM TẮT — asks chính

**Ưu tiên A (net-win, không đổi ABI Tensor):**

```
i32 f32_list_copy(dst, src, n)     # dst[i] = src[i]  — bulk list[f32] → list[f32]
i32 f32_list_add(dst, src, n)      # dst[i] += src[i] — tích lũy 1 pass native
f64 f32_grad_norm(g, n)            # sqrt(sum g^2), 1-pass native, tích lũy fp64
i32 adamw_step_glm(p, m, v, g, ...)  # g nhận list[f32] đọc thẳng _items (xem §2.4)
```

**Ưu tiên B (cần đề xuất thiết kế TRƯỚC khi implement):**

```
# f32buf data plane: Tensor.data thành handle vào native buffer
# (mẫu f32buf_wrap(p, n) borrow semantics đã có ở REFERENCE §509)
# → repo cần đề xuất ABI + ảnh hưởng ckpt/resume trước khi wire
```

**Bug cần fix kèm (§4) — cả 2 đều làm JIT fail `InvalidProgramException` mà compile KHÔNG báo lỗi.**

---

## 2. VÌ SAO — bằng chứng đã ĐO (Phase A, HANDOFF §0.52)

### 2.1 Phân rã BWD 4922 ms (seq8) — probe `_bd2.tkv`

| thành phần | ms | ask liên quan |
|---|---|---|
| 12 × block_bwd | 2244 | (đã native, Phase E) |
| [1] `dwte += gwte_e` 38.6M interpreted | **531** | `f32_list_add` |
| [2] `gflat += dwte` 38.6M interpreted (use_flat=1) | **671** | `f32_list_add` |
| [3] wpe loop 0.8M | 15 | (không đáng kể) |
| [4] m6 lmhead section (alloc+copies+zero+fused+free) | 359 | fused đã OK |
| [5] op_embed_bwd ×2 | 219 | (đã P6) |
| glue không giải thích được | ~880 | Phase C |

→ **2 vòng dense `for k in range(V*h)` interpreted = 1202 ms = 24% BWD** — mỗi vòng
là `a[k] = a[k] + b[k]` 38.6M phần tử, chính là contract của `f32_list_add`.

### 2.2 Copy-back interpreted (production, mỗi step)

`gpt2_unflatten(ps, opt.p)` trong `gpt2_train` = **844 ms/step** @seq8 — vòng
interpreted ghi `list[f32]` → Tensor.data. Thay bằng `f32_list_copy` (hoặc sống
trên native ngay từ đầu — ask §2.5).

### 2.3 `grad_norm` = 390 ms/step (CJ8)

`opt.grad_norm()` hiện interpreted đọc toàn bộ 124M phần tử. `f32_grad_norm`
1-pass native đọc thẳng `_items` → dự kiến ~20–50 ms (memcpy-bound: 496MB @ 6.4 GB/s
đo được trong Phase A ≈ 78 ms cho copy; norm có thêm 1 phép nhân/ cộng — ~100–150 ms
an toàn, vẫn **−250–300 ms**).

**Ràng buộc parity:** tích lũy fp64 **tuần tự cùng thứ tự** như phiên interpreted
hiện tại → `grad_norm` → `coef clip` → chuỗi loss 3 seq phải giữ EXACT (gate 15/15
+ loss chain hiện tại). Nếu bắt buộc đổi thứ tự cộng, báo lại repo trước để đổi tolerance.

### 2.4 `adamw_step` nhận `g` là `list[f32]` — bỏ r1

Hiện `step()` phải `f32_copy_to_buf(g_list → pgl)` (r1) rồi mới `adamw_step(..., pgl, ...)`.

**Số đúng sau fact-check A3:** `f32_copy_to_buf` thật chỉ **78–94 ms cho 496 MB =
6.4 GB/s** (số 1516 ms trong CJ8 trước đây là artifact `sync_mv()` — KHÔNG phải production).
Vậy lợi ích ask này không phải 1516 ms, mà là:

- bỏ 1 vòng round-trip ở biên (78 ms + điều phối), và
- **là bước đệm cho ask §2.5** — khi g đã sống native, optimizer không còn biên list↔buf.

Gợi ý ABI: đọc thẳng `List<Single>._items` như `f32_copy_to_buf` đã làm
(không ToArray trung gian), giữ nguyên công thức/thứ tự GĐ10 (bit-exact).

### 2.5 f32buf data plane (ưu tiên B — cần đề xuất trước)

Mở rộng `f32buf_wrap` (mẫu đã có, borrow, REFERENCE §509): `Tensor.data` là handle
vào native buffer thay vì `List[Single>`; `list↔buf bulk` xảy ra **ở biên** (ask §2.1
đủ cho các bước chuyển đó). Đụng tới: định dạng ckpt, resume bit-exact, mọi
`Tensor.data[k]` interpreted còn lại.

→ **Nhờ compiler team đưa đề xuất ABI + danh sách điểm đụng** (ckpt/resume là rủi ro
số 1 theo PLAN B) trước khi implement. Repo sẽ review rồi wire theo mẫu M5-opt
(`pdirty`/`push_p`/`sync_mv`).

### 2.6 ĐÓNG mục «điều tra f32_copy_to_buf» (mục 3 của kế hoạch cũ)

Phase A đã sốc lại: **không có bug** — 78–94 ms (6.4 GB/s) ở mọi yếu tố context
(probe `_cp1`: +live-set, worker spawn, churn, interpreted read đều không đổi).
Không cần fix phía DLL. Số 1516 ms = `sync_mv()` artifact của probe CJ8.

---

## 3. NGÂN SÁCH — ước tiết kiệm @seq8 (mọi con số chốt lại bằng đo khi wire)

| bucket | hiện tại | sau native | tiết kiệm |
|---|---|---|---|
| 2 vòng dense BWD (`f32_list_add`) | 1202 | ~40 | ~1160 |
| copy-back unflatten (`f32_list_copy`) | 844 | ~30 | ~810 |
| `grad_norm` fused | 390 | ~100–150 | ~250–290 |
| r1 g→buf (bỏ với ask §2.4) | 78 | 0 | ~78 |
| **Tổng** | | | **~2.3 s** |

→ seq8 dự kiến 7187 → **~4800 ms** (kế hoạch PLAN B ghi 4000–4500 — phần chênh thuộc
Phase C/D, không thuộc ask này). Target torch 14T: 1190 ms.

---

## 4. 2 BUG COMPILER — JIT `InvalidProgramException`, compile KHÔNG báo lỗi

### Bug C1 — local variable shadowing module global

```text
import tv.training.gpt2_ref

def main() -> 'i32':
    MaxPos = 1024        # gpt2_ref có global `MaxPos = 1024` — trùng tên local
    print(str(MaxPos))
    return 0
```

- tkvc 26/10: `[tkvc] da dich` (thành công) → exe chạy ném
  `System.InvalidProgramException: Common Language Runtime detected an invalid program. at TKVApp.main()`.
- Đổi tên local (`MaxPosN`) → chạy ngay (đã xác minh A/B).
- Mức độ nguy hiểm: **không báo lỗi ở compile** → ai khai local trùng tên global của
  module import (danh sách global: `VocabSize/HiddenSize/NHeads/HeadDim/NLayers/MaxPos/EPS`
  của gpt2_ref) sẽ build được exe hỏng.
- Đề nghị fix: emit local slot thật đúng scope, hoặc **báo lỗi compile** khi local át tên global.

### Bug C2 — `f32_zero(p, n)` gọi trong `while` ở `main`

```text
def main() -> 'i32':
    p = alloc_bytes(4096)
    q = 0
    while q < 3:
        z = f32_zero(p, 1024)     # giữ dòng này -> InvalidProgramException
        q = q + 1
    return 0
```

- Bisect A/B cùng mọi dòng khác: **giữ `f32_zero` trong loop = lỗi; xóa = chạy.**
- `f32_copy_to_buf`, `alloc_bytes`, `free_bytes`, `lmhead_bwd_fused` trong **cùng loop** thì OK.
- `f32_zero` ở top-level (ngoài loop) thì OK — probe `_cp1` chạy tốt.
- Đo bằng tkvc 26/10 với entry `main`, biến thể đặt trong `out/` (import không resolve).

---

## 5. KẾ NGHIỆM KHI GIAO (repo sẽ làm, đúng pattern cũ)

1. **Smoke** probe mới gọi từng builtin mới từ `.tkv` — TOL 1e-6 so với phiên interpreted
   + negative test (n vượt len → lỗi rõ ràng).
2. `grad_norm`/`adamw_step_glm`: đối chiếu bit-exact trên fixture 8 token.
3. Gate **135/135** + trainer **15/15** (kèm resume case) + loss chain 3 seq EXACT.
4. B2 wire dần theo mẫu M5-opt, sync 4 mirror `*.tkv`.
5. Bench 3 seq best-of xen kẽ torch fresh → HANDOFF §0.53.

## 6. PHẢN HỒI MONG ĐỢT

- File reply trong `D:\TokenVector\Release\docs\e_handoff\` (mẫu
  `2026-10-26-*-compiler-reply.md`) + cập nhật `REFERENCE.md` cho từng builtin mới.
- Với ưu tiên B (§2.5): chỉ cần **đề xuất thiết kế** trước, chưa implement.
- Ghi rõ ABI đầy đủ (tham số, kiểu, hợp đồng lỗi, hợp đồng số liệu fp32/fp64) như các
  đơn trước (`simd_gemm_full`, `lmhead_bwd_fused`).