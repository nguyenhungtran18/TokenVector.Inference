# YÊU CẦU COMPILER TEAM #5 — PHASE C: CODEGEN VÒNG `list[f32]` NHANH (tkvc)

**Ngày:** 2026-10-27 · **Nguồn:** `docs/PLAN_TO_PARITY.md` Phase C (C1)
**Đối tượng:** Compiler team (tkvc codegen) · **Ưu tiên:** cao — đòn bẩy ước 40–50% theo plan
**Bằng chứng chính:** HANDOFF §0.49 (probe `_f32p`) + §0.52 (phân rã BWD) + §0.53 (Phase B mới wire)

---

## 1. Bối cảnh & khoảng cách hiện tại

Sau Phase B (commit `829a679` — `f32_list_add` thay 2 vòng dense BWD, loss EXACT):

| seq | tkv sau Phase B | torch 14T | torch 1T | tỉ lệ |
|---|---|---|---|---|
| 8 | **6015 ms** | 934.2 ms | 1322.7 ms | **6.4× / 4.5×** |
| 256 | **8422 ms** | 1394.1 ms | ~1.5 s | **6.0×** |
| 1024 | **20406 ms** | 3546.7 ms | — | **5.8×** |

Còn lại **không có lever `.tkv`-only nào** (§0.49 chốt): gap nằm ở **vòng interpreted
trên `list[f32]`** — list indexing + bounds-check + append chạy interpreted trong khi
phép toán vốn đã f32. Đúng lever này thuộc codegen tkvc (Phase C); fallback (Phase C2)
là builtin từng op — chỉ làm khi C1 không đạt §acceptance.

## 2. Bằng chứng (đo thật, không suy luận)

### 2.1 Root cause đã loại bỏ giả thiết sai — probe `_f32p` (HANDOFF §0.49)

A/B 3 case, **cùng logic chỉ khác kiểu local** (`z: f64` hiện tại vs `z: "f32"` mới),
best-of-3 × 4 lần chạy, batch >300 ms/việc:

| Case | Đường code | Speedup | BIT_MAXDIFF |
|---|---|---|---|
| [1] BUILD tile 512/768 (`_tile_a_w` copy + zero-fill) | `z: f64` vs `z: "f32"` | **~1.0×** (0.92–1.20×) | 0.0 |
| [2] ACC T=768 (`_acc_f` production) | không local vs f32 local | **0.97–1.0×** | 0.0 |
| [3] ARITH thuần 12M phép (`x*x+0.5`) | f64 vs f32 local | **~1.0×** | 0.0 |

→ **f32 locals BUST.** Chi phí BUILD+ACC+ALLOC (đo §0.35/CG là **76–89% full step**)
không phải widen kiểu mà là: **mỗi `a[i]` = interpreted bounds-check + Count/indexer
+ ghi từng phần tử**, không có hoisting, không có `_items` trực tiếp, không SIMD.
Yêu cầu #1 §2 dự đoán «f32 locals giảm wrapper 76–89% xuống 10–20%» — số đo cho thấy
cần đúng lever **indexing**, không phải typing.

### 2.2 Mức độ của cùng lớp code — Phase B chứng minh biên độ

Vòng dense `for k in range(38.6M): a[k] = a[k] + b[k]` interpreted = **1202 ms/bước**
(24% BWD, §0.52) → thay bằng bulk native `f32_list_add` = **62 ms (35–67×, BIT=0)**
và full step giảm −20.1%/−16.6%/−12.8% ở 3 seq (§0.53).
**Chỉ 2 vòng** đã ăn 1.2 s — còn hàng chục vòng cùng hình dạng trong repo chưa có
builtin tương ứng. Codegen tốt sẽ generalized lợi ích này cho **mọi** vòng, không
cần viết builtin từng cái.

### 2.3 Bucket còn tồn đọng (phân rã A2 — sau khi Phase B gỡ 1202 ms)

| bucket | ms/bước (seq8) | chứa vòng `list[f32]` interpreted |
|---|---|---|
| 12 × block_bwd | 2244 | wrapper BUILD/ACC/ALLOC (76–89% chi phí thuộc indexing) |
| glue/attention “ít giải thích được” | ≈880 | chủ yếu interpreted |
| `gpt2_unflatten` copy-back | ≈875 (CJ8) | `d[j] = flat[off+j]` 124M ptu |
| wpe loop, `zero_grad` 124M, `_norm_groups`, `gpt2_flatten`, … | vài trăm | tất cả cùng mô hình |

## 3. Yêu cầu kỹ thuật (C1)

**Mục tiêu:** vòng `while`/`for` duyệt `list[f32]` (mở rộng `list[i64]` nếu miễn phí)
được codegen ra IL chạy nhanh trên chính ngữ cảnh hiện tại — **không đổi syntax `.tkv`**,
không thêm builtin bắt buộc.

1. **Hoist bounds-check:** kiểm tra phạm vi kiểu `0 ≤ lo && hi ≤ Count` (biết trước
   số phần tử của vòng) **một lần ngoài vòng**; truy cập `a[i]`/`a[i*k + c]` bên trong
   đã được chứng minh trong phạm vi → bỏ check từng phần tử.
   *Hợp đồng:* truy cập **thật sự** OOB (ngoài phạm vi đã chứng minh) vẫn phải ném
   đúng exception như hiện tại — semantics không đổi, gate negative giữ nguyên.
2. **Truy cập `_items` trực tiếp:** load backing array (`List<Single>._items`) một
   lần ngoài vòng, index trực tiếp; **không `ToArray`, không alloc, không copy**
   (giữ kỷ luật `assert_il` hiện có: `ToArray = 0` trong wrapper; `get_Count` trong
   vòng phải giảm về 0 sau hoisting — đây là chỉ số kiểm tra dễ nhất).
3. **Auto-vectorize AVX2** cho pattern elementwise trong hồi đã hoist: copy, zero-fill,
   scale, `a[i] ± b[i]`, `a[i] = x*b[i] + c[i]`, fused `a[i] += k*b[i]` và
   copy-write `a[i] = b[idx(i)]` (đúng hình BUILD tile `out[dst] = d[src+j]`).
   *Hợp đồng numerics:* elementwise **BIT-Exact** (1 phép/ptu, thứ tự không đổi) —
   loss chain 3 seq phải giữ EXACT. **Reduction (sum, sum-of-squares) KHÔNG vectorize
   re-order** — giữ tích lũy tuần tự f64 như hiện tại (`_norm_groups`, grad_norm) trừ
   khi có annotation riêng; nếu thêm opt-in reduction nhanh thì phải expose kiểu
   contract rõ để repo chọn tolerance.
4. Mẫu ngữ cảnh production (dùng để test tự nhiên, không cần viết lại):
   - `_b64`/`_b32` + `_a64`/`_a32` trong [tv/tests/_f32p.tkv](../tv/tests/_f32p.tkv)
     (BUILD copy + zero-fill + ACC — đã có sẵn harness A/B 3 case);
   - `gpt2_unflatten` ([tv/training/gpt2_train.tkv](../tv/training/gpt2_train.tkv)),
     `_zr_one`/`zero_grad`, `_norm_groups` ([tv/training/trainer.tkv](../tv/training/trainer.tkv)),
     vòng `gflat[o_wpe + k]` ([tv/training/gpt2_ref.tkv](../tv/training/gpt2_ref.tkv)).

## 4. Acceptance (theo plan — chốt số, không nhận chung chung)

| mục | ngưỡng |
|---|---|
| BUILD + ACC + ALLOC wrapper, 4 shape (A/B `_f32p` style) | **≥ 2×** vs codegen hiện tại |
| `get_Count` / bounds-check lặp lại bên trong vòng IL | **0** (đã hoist) · `ToArray = 0` |
| Full-step bench 3 seq (repo chạy, xen kẽ best-of-3) | giảm theo, không seq nào regress |
| Gate repo: 135/135 + trainer 15/15 + loss chain 3 seq | **EXACT giữ nguyên** (elementwise bit-exact) |
| Negative OOB | vẫn ném đúng exception lúc chạy (test riêng) |
| Compiler-side | self-host IL_IDENTICAL + testkit + perf_gate baseline không regress |

**Quy trình hai bên (giống #2/#3):** compiler giao → repo smoke probe (`_f32p` chạy
được ngay, TOL/BIT sẵn) → repo wire-từng phần nếu probe thắng ngưỡng §13.2 →
gate → bench → HANDOFF.

## 5. Fallback (C2) — nếu C1 < 2×

Theo plan, repo sẽ chuyển sang builtin từng op cho bucket >5% step theo thứ tự A2:
`softmax_bwd` → `layernorm_bwd` (gộp dγ/dβ) → `gelu_bwd` → `bias_grad`
(request #6 trở đi). Ghi rõ ở đây để hai bên ưu tiên đúng hướng: **C1 làm được thì C2 gần như bỏ phần lớn**.

## 6. Ghi chú liên quan (không thuộc scope lần này)

- `adamw_step_glm` ở n=124M chậm bất thường (+3.2 s/buoc so với `adamw_step` cùng
  input — số đo pb2/pb3, HANDOFF §0.53) và số «copy-back 844 ms» trong đơn #3 §2.2
  không khớp production (A/B cho 0%) — **đã tách thành yêu cầu #4** (2026-10-27).
- Machine: Xeon E5-2695 v3 (14C/28T, AVX2+FMA, DDR3) — threading 14 luồng là Phase D,
  **không** thuộc yêu cầu này; C1 chỉ tối ưu đơn luồng.