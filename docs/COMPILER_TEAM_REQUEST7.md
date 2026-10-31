# YÊU CẦU COMPILER TEAM #7 — BỎ TkvInt CEREMONY TRONG THÂN VÒNG VECTOR (TYPED-LOCAL/LICM)

**Ngày:** 2026-10-28 · **Nguồn:** HANDOFF §0.55 (verify request #6) + probe `_p6kv`/`_p6off`
**Đối tượng:** Compiler team (tkvc codegen P6) · **Ưu tiên:** cao — đây là nút thắt
còn lại của toàn bộ Phase C
**Bằng chứng chính:** `_p6kv` 3.2× · `_p6off` 3 arm · IL `_step124` (bseq_c6) ·
`_p6cf` 2.90× < 5× · full-step chỉ −3.8/−11.2/+0.4% dù 134 site vector

---

## 1. Bối cảnh — C1-b đã nổ nhưng overhead mỗi phần tử nuốt phần lớn gain

Request #6 đã giao và verify xong (HANDOFF §0.55): matcher mở cho offset/const-fill/
counter, gate 135/135 + trainer 15/15 xanh, **134 site `__p6a*`** trong `bseq_c6`
(vs 1 loop ở §0.54), nổ đúng hot loops (`gpt2_block_bwd` 15, `gpt2_mlp_bwd` 13,
`gpt2_attention` 11, `_tile_a_w/_tile_b_w` 7, `_acc_f` 5).

Nhưng **acceptance speed chỉ 1/4 đạt**: `_f32p` 1.2×/0.25×, const-fill 2.90×<5×,
offset micro-probe mới đo được 3.0×, full-step seq8 −3.8% / seq256 −11.2% /
seq1024 +0.4% (noise). **Lý do đã cô lập được: thân vòng vector đang trả TkvInt
ceremony cho MỖI phép tính chỉ số** — phần codegen #6 làm đúng (vector loop 4-wide,
hoist `_items`, preheader OOB), nhưng mỗi `off + j`, `j = j + 1` với local **không
có type annotation** vẫn đi qua đường bignum-safe của `TkvInt`.

## 2. Bằng chứng

### 2.1 A/B thuần ceremony — probe `_p6kv` (cùng source, chỉ khác annotation)

Hình 148-slice copy 63,007,488 ptu, 2 arm **cùng logic 100%**:

| arm | local | ms (best-of-3) | tỉ lệ |
|---|---|---|---|
| untyped (`j`, `off` không annotation) | TkvInt | **250** | 3.205× |
| typed (`jt: "i64"`, `offt: "i64"`) | i64 native | **78** | 1× |

→ **3.2× chỉ do ceremony** — chưa đổi một phép tính nào (bit-exact, `fails=0`).

### 2.2 Offset A/B 3/4 arm — probe `_p6off` (mới, hôm nay — basline cho acceptance)

Cùng 63M ptu, best-of-3 xen kẽ U-V-T:

| arm | form thân | P6 | ms | tỉ lệ |
|---|---|---|---|---|
| **U_old** (proxy codegen cũ): 3 câu + toàn local khieu | interpreted + TkvInt | ❌ reject | **140** | 1× |
| **V_prod** (hình production `_step124`): 2 câu, `off` typed / `j` khieu | vector 4-wide | ✅ | **47** | 2.98× |
| **T_typed**: 2 câu, cả `off`/`j` typed | vector 4-wide | ✅ | **31–47** | 2.98–4.52× |

`bit_bad=0` (3 arm cùng kết quả). Quan sát: **U chậm 3× V** dù V cũng chỉ có `j`
khieu — cost tập trung ở phép `off + j` TkvInt trong thân; T/V ≈ 1.0–1.5× (noise
giữa 2 lần chạy: 31 vs 47 ms) → dư địa sau khi bỏ ceremony nằm ở chính arm U =
hình production thật.

### 2.3 IL thật trong production — `_step124` copy-back (bseq_c6, §0.55)

`.locals` của `_step124`: **`valuetype TkvInt 'off'`** — và thân vector loop mỗi
vòng lặp 4 phần tử vẫn gọi:

```
TkvInt::FromI64 → ldfld big ×2 → ldfld lo ×4 → stfld lo/big ×2
  → TkvInt::AddSlow → TkvInt::ToI32 → ldelema → stobj Vector4
```

≈ **15–19 lệnh IL ceremony cho mỗi bước index** bên trong đúng loop đã vector hóa
(chưa kể preheader). Toàn binary: `bseq_c6` = **2044 lệnh `TkvInt::`** (pb5 = 1897 —
C1 không đụng tới nó); `_step124` một mình 46 lệnh. Trong khi `_p6kv` đã chứng minh
con đường typed nhanh hơn 3.2×.

### 2.4 Vì sao acceptance #6 chưa đạt — cùng gốc ceremony

- `_p6cf` const-fill 2.90× < 5×: counter `i = 0` untyped → `i = i + 1`, `i < n`
  đều TkvInt **trong loop 124M phần tử**;
- full-step seq1024 gần như phẳng (+0.4%): GEMM dominate, các loop list[f32] gain
  bị ceremony mỗi ptu bù lại.

**Tính toán trần (để chốt số acceptance, tránh phóng đại):** 63M ptu = 252 MB;
  đường typed đã chạy 31–47 ms ≈ 5–8 GB/s (gần trần ghi RAM DDR3). Nên với
  **micro-probe**, 10× (140 → 14 ms) là **vật lý không thể** — trần thật ≈ 5×.
  Ngược lại, **shape production thật** (`gpt2_unflatten` 148-slice đo được
  **4453 ms interpreted** — reply #4 §B; copy-back CJ8 ≈ 875 ms) thì 10× hoàn toàn
  trong tầm (4453 → ≤445 ms). Do đó acceptance §4 tách 2 mức: micro ≥4×, production ≥10×.

## 3. Yêu cầu kỹ thuật

**Mục tiêu:** trong thân vòng đã match P6, mọi phép tính **chỉ số/int induction**
chạy bằng i64/i32 native — không `TkvInt` — khi (và chỉ khi) phạm vi đã được
preheader chứng minh. Không đổi syntax `.tkv`, không đổi semantics.

### Y1 — Native index arithmetic trong thân vòng đã match
Khi matcher đã phát range-proof (`0 ≤ off ∧ off + n − 1 ≤ Count − 1`):
- `off + j`, `dst + 1`, `j = j + 1`, `src + kdim`… sinh `add`/`conv` i64/i4 thuần,
  **bỏ hẳn** `FromI64/AddSlow/ToI32/ldfld-stfld` trong thân.
- *Lý do an toàn (không phải approximation):* after preheader check, mọi index nằm
  trong `[0, Count−1]`, `Count ≤ i32.MaxValue` (List.Count) → sum nằm trong i32,
  native add **không overflow được**, semantics = hệt TkvInt ở miền này.
- Chưa prove được (loop ngoài matcher, index tràn ra ngoài) → giữ nguyên `TkvInt`
  như hiện tại (nguyên tắc «hỏi nghi = từ chối» của C1 không đổi).

### Y2 — LICM cho index invariant
Trong preheader (chung chỗ với OOB check hiện tại):
- hoist `&_items[off]` (base address đã offset) và mọi hàm đúng hạn của offset ra
  ngoài vòng — loop chỉ cần `base + j`;
- loop counter typed native (đã có: `iv` i64 trong match) — áp dụng cho cả counter
  phụ (`dst`, `bc`, `bt`) vốn đã được #6 công nhận.

### Y3 — (Khuyến nghị, không bắt buộc) suy luận typed-local
Local không annotation nhưng **chỉ xuất hiện** trong ngữ cảnh induction/index/so sánh
trong vòng đã match → tự coi là native index. Nếu làm được Y3 thì repo không phải
annotate lại hàng chục loop × 5 mirror file; **nếu chưa làm được, repo sẽ annotate
từng chỗ** (`off: "i64"`) như stopgap — nhưng đó là việc phía repo, không chặn compiler.

### Hợp đồng giữ nguyên
- OOB exception đúng message/paramName (test `_p6_guard_neg`, `_oob_probe`);
- elementwise BIT-Exact — loss chain 3 seq EXACT;
- negative test: loop không match vẫn đi đường TkvInt cũ.

### Mẫu ngữ cảnh để test (đo trên chính cái này)
- `_step124` copy-back (`d[j] = opt.p[off+j]` — 124M ptu/bước, `off` hiện là TkvInt);
- `gpt2_unflatten` 148-slice ([tv/training/gpt2_train.tkv](../tv/training/gpt2_train.tkv));
- `_p6kv` 2 arm + `_p6off` 3 arm ([tv/tests/_p6kv.tkv](../tv/tests/_p6kv.tkv),
  [tv/tests/_p6off.tkv](../tv/tests/_p6off.tkv)) — đã có sẵn harness best-of-3 + bit check.

## 4. Acceptance (chốt số, 2 mức micro/production)

| mục | ngưỡng | baseline hôm nay |
|---|---|---|
| `_p6off` arm **U_old** (hình production, khieu) | ≥ **4×** vs hôm nay (140 → ≤35 ms) | 140 ms · trần vật lý ≈ 5× |
| `_p6kv` untyped/typed ratio | **≤ 1.15×** (ceremony đã bỏ — 2 arm phải bằng nhau) | 3.205× |
| `_p6cf` const-fill @124M (kế thừa #6) | ≥ **5×**, bit-exact | 2.90× |
| **Production shape** — copy-back/unflatten trong full-step + probe shape thật (`_cj8`) | ≥ **10×** so số cũ (4453 ms unflatten interpreted / 875 ms copy-back — reply #4 §B) | repo tự A/B, số cũ đã đo |
| IL trong thân vòng vector | **0** lệnh `TkvInt::` (chỉ preheader được phép có, nếu có) · `get_Count`=0 · `conv.ovf.i4`=0 · `ToArray`=0 | `_step124`: 46 lệnh |
| Full-step 3 seq (xen kẽ best-of-3, vs `bseq_c6`) | cải thêm, **không seq nào regress**; loss chain EXACT | c6: 5547 / 7313 / 21578 ms |
| Gate repo | 135/135 + trainer 15/15 + negative OOB xanh | ✅ đang xanh |
| Compiler-side | self-host chạy được + diff IL chỉ đúng codegen dự kiến + testkit/P6 perf_gate không regress | — |

**Quy trình hai bên (giống #2/#3/#5/#6):** compiler giao → repo smoke `_p6kv`/
`_p6off` (harness sẵn, bit-check sẵn) → bắn tỉa IL đếm `TkvInt::` trong thân →
gate → bench 3 seq → HANDOFF §0.56.

## 5. Ghi chú scope

- Máy Xeon E5-2695 v3 (14C/28T, AVX2+FMA, DDR3) — **đơn luồng**; threading 14 luồng
  là Phase D, không trộn vào đây.
- **Thân nhiều câu (block_bwd 2244 ms)** vẫn ngoài scope — chuyển thành yêu cầu
  **#8** (trước đây ghi nhầm là #7 trong HANDOFF §0.55).
- `adamw_step_glm`/copy-back số cũ đã chốt ở reply #4 — không mở lại; số 4453/875 ms
  chỉ dùng làm thước đo lợi thế nhánh (a).
- Fallback C2 (builtin từng op `softmax_bwd → layernorm_bwd → gelu_bwd → bias_grad`)
  vẫn còn hiệu lực nếu sau #7 `gpt2_block_bwd` (thân nhiều câu — số #8) không kịp.
