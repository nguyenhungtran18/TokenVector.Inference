# YÊU CẦU COMPILER TEAM #8 — MỞ MATCHER P6 CHO THÂN NÊN NHIỀU CÂU (block_bwd)

**Ngày:** 2026-10-28 · **Nguồn:** HANDOFF §0.52 (phân rã BWD) + §0.55 (IL sau #6)
**Đối tượng:** Compiler team (tkvc codegen P6) · **Ưu tiên:** cao — bucket lớn nhất
còn lại của BWD (≈2244 ms/bước seq8, §0.52)
**Bằng chứng chính:** phân rã BWD 4922 ms (§0.52) · IL `bseq_c6` (§0.55) · shape
thật trong [gpt2_ref.tkv](../tv/training/gpt2_ref.tkv) (`gpt2_layernorm_bwd`,
`gpt2_linear_bwd`)

---

## 1. Bối cảnh — matcher #5/#6 vẫn gói thân trong «ĐÚNG 2 CÂU»

Hợp đồng C1 từ design phase (comment trong `control_flow.tkv`): matcher chỉ nhận
vòng «thân **ĐÚNG 2 câu** (idx_assign + `iv = iv + 1`)», #6 mở thêm offset/
const-fill/counter nhưng **vẫn giữ giới hạn thân 2 câu**. Kết quả (§0.55): 134 site
`__p6a*` trong `bseq_c6`, nổ tốt các loop 2 câu (`gpt2_block_bwd` 15 Vector4 hit,
`gpt2_mlp_bwd` 13, `_tile_a_w` 7…) — nhưng **những vòng thân N câu vẫn đi scalar**,
và đó chính là bucket 2244 ms.

## 2. Bằng chứng

### 2.1 Bucket — phân rã BWD 4922 ms (§0.52, probe `_bd2`/AM, best-of-3, seq8)

| thành phần | ms/bước | % BWD | trạng thái matcher #6 |
|---|---|---|---|
| **12 × `gpt2_block_bwd` (tổng)** | **2244** | **45.6%** | một phần (các loop 2 câu đã nổ — §0.55), **các loop thân N câu vẫn scalar** |
| `dwte += gwte_e` + `gflat += dwte` (2 vòng dense) | 1202 | 24% | ✅ đã gỡ bằng builtin Phase B (`f32_list_add`) |
| tail lm-head/alloc/copy | ≈795 | 16% | partly (`lmhead_bwd_fused` M6) |
| glue/attention chưa giải thích | ≈880 | 18% | trộn cả 2 loại |

Sau Phase B + #6, full-step seq8 còn **5547 ms** (§0.55) → bucket block_bwd **vẫn
≈ 35–40% full-step** — lớn hơn mọi lever còn lại gộp lại. Đúng lever này nếu codegen
không làm được thì fallback C2 là viết **4 builtin nữa** (`softmax_bwd →
layernorm_bwd → gelu_bwd → bias_grad` — PLAN_TO_PARITY Phase C2), nặng hơn nhiều
cho cả hai phía.

### 2.2 Shape thật bị từ chối — trích trực tiếp từ production

**(A) `gpt2_layernorm_bwd` — vòng ghi cuối (mỗi `i`, `h` phần tử, 12 block × 2):**
```text
for j in range(h):
    xh = (x.data[base + j] - mu) * rstd            # câu 1 — 2 load + arith
    gg = gy.data[base + j] * g.data[j]             # câu 2 — 3 load + mul
    dx.data[base + j] = rstd * (gg - m1 - xh * m2) # câu 3 — store (attr base)
    if o_dg >= 0:                                  # câu 4 — branch loop-INVARIANT
        gflat[o_dg + j] = gflat[o_dg + j] + gy.data[base + j] * xh
    else:
        dg.data[j] = dg.data[j] + …
    if o_db >= 0:                                  # câu 5 — branch invariant
        gflat[o_db + j] = gflat[o_db + j] + gy.data[base + j]
    else:
        db.data[j] = db.data[j] + …
```
→ **6–7 câu, có 2 branch loop-invariant, index dạng `base + j` và offset `o_dg + j`,
base là attr `x.data`/`gy.data`.** Toàn bộ nằm ngoài matcher.

**(B) `gpt2_linear_bwd` vòng `db` (gọi trong mọi linear bwd):**
```text
for i in range(n):
    base = i * out_d
    for o in range(out_d):
        if dboff >= 0:
            gflat[dboff + o] = gflat[dboff + o] + gy.data[base + o]
        else:
            db.data[o] = db.data[o] + gy.data[base + o]
```
→ thân 3 câu + branch invariant + offset index.

**(C) Các vòng ACC/BUILD `list[f32]` nhiều câu** (giống `_f32p` `_a32`: thêm local
trung gian → 3 câu → reject; số liệu §0.55: arm `_a32` chạy 563 ms vs `_a64`
vector 140 ms) và hàng chục vòng shape tương tự trong attention/mlp glue.

### 2.3 IL — #6 đã vector hóa đúng cái gì, bỏ sót cái gì (§0.55)

`bseq_c6`: `gpt2_block_bwd` 15 hit Vector4, `gpt2_mlp_bwd` 13, `gpt2_attention` 11
— các loop 2 câu **đã nổ**; nhưng song song đó **46 lệnh `TkvInt::` còn trong
`_step124`**, và các method trên **vẫn còn loop scalar** (thân N câu). Nghĩa là:
cơ chế vector + hoist + preheader đã hoạt động; thứ thiếu **chỉ là khả năng nuốt
thân nhiều câu**.

## 3. Yêu cầu kỹ thuật

**Mục tiêu:** mở matcher P6 cho thân **N câu** (N ≥ 3) trên ngữ cảnh đã quen —
**không đổi syntax, không đổi numerics contract** (đã chốt từ #5 §3.3: elementwise
BIT-Exact, **reduction không re-order**).

### Y1 — Thân N câu không branch (giai đoạn lõi)
Thân là dãy các câu thuộc tập sau, independent (không có read-after-write tạo
dependency giữa 2 câu cùng target):
1. gán local thuần (`xh = …`, `gg = …`, tích lũy local trung gian **không phải
   reduction tích lũy tuần tự toàn vòng**);
2. idx_assign `base[expr] = vector-expr` với base ∈ {biến `list[f32]`, attr
   `obj.truong`}, `expr` là counter/offset đã công nhận (dùng lại bộ range-proof
   `_p6_range_proof` của #6);
3. counter increment (`iv = iv + 1`, `dst = dst + 1`, `bc = bc + 1`…).

→ sinh **một loop vector duy nhất** cho cả thân (tất cả access đã hoist
`_items`, preheader OOB check gộp **union** các offset: `0 ≤ off_k ∧ off_k + n − 1
≤ Count − 1` cho mọi k).

### Y2 — Loop unswitching cho branch loop-invariant (giai đoạn 2 — cái này mới nuốt được LN_BWD)
`if o_dg >= 0` / `if dboff >= 0` **không phụ thuộc `iv`** → **unswitch ra ngoài vòng**:
sinh 2 phiên bản loop specialized (then / else), điều kiện test 1 lần ở preheader.
Branch **có phụ thuộc counter** → không unswitch được → reject về scalar (giữ
nguyên tắc «hỏi nghi = từ chối»).

### Y3 — Form `for … in range(N)` và `for … in range(len(x))`
Production viết toàn bộ bằng `for` (LN_BWD, linear_bwd, `gpt2_unflatten`) — matcher
phải phủ cả 2 form (hiện design gốc viết quanh `while iv < n`; nếu `for` đã được
desugar thì chỉ cần xác nhận, probe sẽ kiểm).

### Giữ nguyên hợp đồng (không thương lượng)
- **Reduction tích lũy tuần tự**: `mu`, `v`, `s1`, `s2` (LN), `_norm_groups`,
  grad_norm… **không vectorize re-order**, không đổi thứ tự cộng — loss chain
  3 seq phải EXACT;
- OOB thật vẫn ném đúng exception (mở rộng `_p6_guard_neg`/`_oob_probe` cho
  multi-offset union);
- thân không phân loại được → scalar như hiện tại (không «cố đoán»).

### Tương tác với yêu cầu #7 (đọc trước khi làm)
Thân N câu vector mà vẫn chứa TkvInt ceremony sẽ tự giết gain (§0.55: 3.2× chỉ do
ceremony). **#7 (native index trong thân) là điều kiện để #8 có giá trị** — nên làm
#7 trước hoặc làm cùng nhau; test acceptance #8 đo trên binary đã có #7.

## 4. Acceptance (chốt số)

| mục | ngưỡng | baseline hôm nay |
|---|---|---|
| Probe `_p6ms` (mới — 3 shape LN_BWD/linear_db/BUILD-ACC, arm N-câu vector vs arm N-câu reject) | ≥ **4×** mỗi shape, BIT=0 | hôm nay 2 arm cùng reject ≈ 1.0× |
| Bucket `12 × block_bwd` (probe AM/`_bd2` shape thật) | giảm ≥ **25%** (2244 → ≤1680 ms) | 2244 ms (§0.52) |
| Full-step 3 seq xen kẽ best-of-3 (vs `bseq_c6` 5547/7313/21578) | không seq nào regress; seq8 cải thêm | §0.55 |
| IL snipe | loop vector (`stobj Vector4`/`__p6a*`) xuất hiện trong `gpt2_layernorm_bwd`/`gpt2_linear_bwd`; `get_Count` trong thân = 0; **0 `TkvInt::` trong thân** (kèm #7) | LN_BWD: 0 loop nổ |
| Gate repo | 135/135 + trainer 15/15 + negative OOB (kèm case multi-offset) | ✅ đang xanh |
| Loss chain 3 seq | **EXACT** — reduction không đổi thứ tự | ✅ |
| Compiler-side | self-host chạy được + diff IL chỉ đúng codegen dự kiến + testkit/P6 perf_gate | — |

**Quy trình hai bên (giống #2/#3/#5/#6/#7):** compiler giao → repo viết +
smoke probe `_p6ms` (harness bit-check + best-of xen kẽ theo mẫu `_p6off`) →
bắn tỉa IL → gate → bench 3 seq → HANDOFF §0.56.

## 5. Ghi chú scope

- Máy Xeon E5-2695 v3 — **đơn luồng**; threading 14 luồng = Phase D, không trộn.
- **Không mở lại** `adamw_step_glm`/copy-back (đã chốt reply #4); unflatten buffer-slice
  chờ repo sửa bench gọi production rồi A/B riêng (§0.53 ĐÍNH CHỈNH) — không thuộc #8.
- Fallback C2 (4 builtin `softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad`) vẫn có hiệu
  lực: nếu #8 không đạt acceptance §4 ở shape LN_BWD, hai bên chuyển C2 theo plan —
  nên #8 chính là cơ hội gộp C2 vào codegen.
- Refs: budget §0.52 · IL §0.55 · ceremony #7 · matcher design `2026-10-27-phase-c1-baseline-and-design.md`.