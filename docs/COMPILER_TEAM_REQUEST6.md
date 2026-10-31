# YÊU CẦU COMPILER TEAM #6 — MỞ RỘNG C1: OFFSET / CONST-FILL / COUNTER RIÊNG

**Ngày:** 2026-10-27 · **Nguồn:** `docs/COMPILER_TEAM_REQUEST5.md` (Phase C1) + HANDOFF §0.54
**Đối tượng:** Compiler team (tkvc codegen) · **Ưu tiên:** cao — tiếp nối C1 đã giao, chỉ mở rộng matcher
**Bằng chứng chính:** HANDOFF §0.54 (verify C1 trên repo) + probe `_f32p` + reply #4 §B

---

## 1. Bối cảnh — C1 đã giao, đúng lever, nhưng chưa loop hot nào khớp

C1 (`control_flow.tkv` + `vector4_type.tkv` + `file_io.tkv`, self-host `IL_IDENTICAL`,
`tkvc-tv.exe` 17:09) đã được repo kiểm chứng độc lập (§0.54):

| kiểm | kết quả |
|---|---|
| Gate compiler mới | **gpt2_full 33/33 + trainer 15/15 (kèm resume) + simd_matmul 20/20 — xanh** |
| Hợp đồng OOB | giữ nguyên (đúng type/message/paramName, không ném khi vòng không chạy) |
| `_f32p` acceptance ≥2× của #5 | ❌ vẫn **1.09× / 1.00× / 1.00×** |
| Bắn tỉa IL bench `bseq_c1` vs `bseq_pb5` | chỉ **1 loop nổ** (`mlp_loss_grad` — hàm demo, **0 loop GPT-2**); `gpt2_full` = 0; `trainer_tests` = 4 (3× `_copy_into` + `mlp_loss_grad`) |
| A/B xen kẽ seq8 best-of-3 | pb5 **5797 ms** vs c1 **5781 ms** = **−0.3% (noise)**, loss EXACT `10.572291→7.0119185` |

→ **C1 đúng lever, sạch, không regress — nhưng pattern hiện tại (`a[iv] = vecexpr`,
thân đúng 2 câu, `loads==0` bị từ chối) quá hẹp.** Cả plan và acceptance #5 đều đạt
được chỉ khi matcher nhận thêm 3 dạng hình dạng sau. Đây là **phần còn lại của cùng
C1**, không phải yêu cầu mới.

## 2. Ba dạng chưa khớp — kèm số đo thật

### (a) C1-b — offset loop-invariant: `a[off + iv]`

`off` không phụ thuộc vòng (biến invariant hoặc hằng). Matcher hiện tại đòi chi so
**chính là `iv`** (`_p6_chi_so_la_iv`) nên mọi vòng có offset đều tuột:

```tkv
# tv/training/gpt2_train.tkv — gpt2_unflatten (copy-back)
for j in range(len(d)):      d[j] = flat[off + j]        # ← src có offset
# tv/training/gpt2_ref.tkv — gộp grad embedding
for k in range(VocabSize*h): gflat[o_wte + k] = ... + dwte.data[k]   # ← dst có offset
# tv/tests/simd_bench_124m_seq.tkv — _step124 copy-back
for j in range(len(d)):      d[j] = opt.p[off + j]       # ← src có offset
```

**Số đo (reply #4 §B — đã đo thật shape production):** hình 148-slice/124,439,808 ptu:
**interpreted 4937 ms → native 63 ms (78×, sample_bad=0)**; khoảng đúng theo JIT-shape
**875–4937 ms interpreted / 63–94 ms native**. «A/B 0%» cũ là đo sai chỗ (bench inline
không gọi `gpt2_unflatten`) — đã đính chính trong §0.53/§0.54. Thêm `gflat[o_wte+k]` +
`gflat[o_wpe+k]` + `gpt2_flatten` cùng hình dạng, ước tổng **≈1–2 s/bước @seq1024**.

### (b) Const-fill — `a[iv] = c` (c loop-invariant)

Hiện bị `len(loads) == 0` **từ chối thẳng** (`_p6_viet_lai_rhs` trả `None` khi RHS không
có `f[i]`):

```tkv
# tv/training/trainer.tkv — zero_grad / _zr_one (124M ptu/bước, §0.53 bucket)
for i in range(self.n): self.g[i] = 0.0
# tv/tests/_f32p.tkv — _b64 zero-fill đuôi tile (nửa thời gian của shape [1] BUILD)
while j < T: out[dst] = z; dst = dst + 1
```

**Số đo:** bucket `zero_grad 124M + wpe + _norm_groups + flatten ≈ vài trăm ms/bước`
(phân rã §0.53); riêng zero-fill nằm trong bucket BUILD+ACC+ALLOC **76–89% full step**
(§0.35/CG). Vector fill (`v_fill4a`/store-lane 4 ptu/iterations) đơn giản hơn load-mixed
vì không cần đọc nguồn.

### (c) C1-c — counter riêng: `out[dst]` với `dst++` đồng bộ

Đây **chính là shape của acceptance `_f32p`** (Phase-C1 design §4.2 đã xác nhận):

| case `_f32p` | thân vòng | phân loại |
|---|---|---|
| [1] BUILD | `out[dst] = d[src + j]` | `dst` counter riêng (+ `src+j` offset thưa) — **C1-c** |
| [2] ACC | `cd[bc] = cd[bc] + Ct[bt]` | `bc`, `bt` counter riêng — **C1-c** |
| [3] ARITH | không index list | đã ~1.0× (loại khỏi scope) |

`cd[bc] = cd[bc] + Ct[bt]` đồng thời là read+write qua 2 counter — viết lại RHS đã
có (`_p6_viet_lai_rhs`), chỉ thiếu **chứng minh range của counter** qua mọi lần lặp.

## 3. Yêu cầu kỹ thuật

Giữ nguyên toàn bộ hợp đồng của #5 (OOB type/message/paramName, `OverflowException`
i64→i32, không ném khi vòng không chạy, elementwise BIT-Exact, không đổi syntax `.tkv`).
Mở thêm 3 nhánh trong cùng bộ matcher:

**(a) Offset loop-invariant — `a[off + iv]` (ưu tiên 1 — ăn production ngay)**
- Preheader chứng minh `0 ≤ off` **và** `off + n − 1 ≤ Count − 1` (n = số lần lặp
  của vòng, `off` load một lần trước vòng — đã invariant nên LICM/local sẵn);
  không chứng minh được → **fallback đường cũ** (từ chối = tự nguyện, như C1).
- Hợp đồng: vòng thật sự OOB (ví dụ `off < 0` hoặc `off + n > Count`) vẫn ném
  `ArgumentOutOfRangeException` đúng message/paramName — emit nhánh check như
  `_p6_preheader_list` đang làm cho `n ≤ Count`, chỉ thêm điều kiện involving `off`.
- RHS/chi so thứ hai cùng dạng (`flat[off + iv]`, `gflat[o_wte + iv]`) đều đi qua
  preheader validate tương ứng; load `_items` vẫn 1 lần/vòng.
- Bắt buộc chạy được 3 ngữ cảnh nêu §2(a) — đây là mẫu acceptance phía repo.

**(b) Const-fill — `a[iv] = c` (ưu tiên 2)**
- Cho phép RHS là scalar invariant (biến local không ghi trong vòng / hằng) khi
  `loads` rỗng — **bỏ điều kiện `len(loads) > 0` cho trường hợp này** (hiện đang
  `return 0` trước cả khi gọi `_p6_vec_phat`).
- Vector-fill 4 ptu/bước + đuôi scalar đi `get_Item/set_Item` cũ (giữ OOB contract);
  `c: f32` lưu lane 16 byte 1 lần ngoài vòng. NUMERICS: ghi đúng bit của `c`
  (bit-exact với `a[i] = c` interpreted — không phép toán thừa).

**(c) C1-c — counter riêng (ưu tiên 3 — chính acceptance `_f32p`)**
- Thân vòng đúng 2 câu gồm `out[dst] = rhs` + `iv = iv + 1`, **`dst` là biến duy
  nhất ngoài `iv` được ghi trong thân** và được tăng đồng độ (`dst = dst + 1`,
  step ±1) → chứng minh định dạng: `dst = dst₀ + (iv − iv₀)` với `dst₀` load trước
  vòng → preheader check `dst₀ ≥ 0 ∧ dst₀ + n − 1 ≤ Count − 1`.
- Ngay khi chứng minh được, cả read (`cd[bc]`, `Ct[bt]`) lẫn write đều đi
  `v_load4a/v_store4a` với index đã hoist — không cần thay đổi đầu ra IL nhiều.
- Nếu range analysis chung làm được thì **dùng chung cho (a)** (offset = bài toán
  range cùng họ) — hai bên nên gộp thành một `_p6_range_proof` dùng cho cả 3 nhánh.

**Thân nhiều câu (block_bwd 2244 ms, §0.53)** — *ngoài scope #6*: giữ nguyên yêu
cầu tách riêng (sẽ là #7 nếu cần) để lần này chỉ mở matcher, giảm rủi ro self-host.

## 4. Acceptance

| mục | ngưỡng |
|---|---|
| `_f32p` [1] BUILD + [2] ACC (C1-c) | **≥ 2×** — kế thừa acceptance #5, vẫn đang mở |
| Probe offset (hình 148-slice unflatten + `gflat[o_wte+k]`) | ≥ **10×** vs interpreted (native data-plane đo 78×; codegen vectorize kỳ vọng cùng bậc, repo tự A/B) |
| Probe const-fill (zero 124M + zero-fill tile) | ≥ **5×** vs interpreted |
| Bắn tỉa IL | 3 nhóm loop production nêu §2 xuất hiện `__p6a*`; `get_Count`/`conv.ovf.i4` trong thân = **0**; `ToArray` = 0 |
| Full-step bench 3 seq (repo, xen kẽ best-of-3) | không seq nào regress; **loss chain EXACT giữ nguyên** |
| Gate repo | 135/135 + trainer 15/15 + negative OOB (`_p6_guard_neg` + `_oob_probe`) xanh |
| Compiler-side | self-host `IL_IDENTICAL` + testkit 3 pattern mới + perf_gate không regress |

**Quy trình hai bên (giống #2/#3/#5):** compiler giao → repo smoke probe mới (viết
`_p6x` A/B 3 pattern, TOL/BIT sẵn) → bắn tỉa IL đếm loop nổ → gate → bench 3 seq →
HANDOFF §0.55. Repo không cần đổi code production — mọi thứ đo trên cấu hình hiện tại.

## 5. Ghi chú scope

- Máy Xeon E5-2695 v3 (14C/28T, AVX2+FMA) — **C1 chỉ tối ưu đơn luồng**; threading 14
  luồng thuộc Phase D, không trộn vào đây (giống #5 §6).
- `adamw_step_glm`/copy-back đã chốt trong reply #4 — **không mở lại**; số 875–4937 ms
  chỉ dùng làm thước đo lợi thế cho nhánh (a).
- Fallback của #5 (C2 builtin từng op `softmax_bwd → layernorm_bwd → gelu_bwd →
  bias_grad`) **vẫn còn hiệu lực nếu (c) không đạt ≥2×** trên `_f32p` — nhưng lúc này
  (a)+(b) đã ăn production nên C2 chỉ còn phụ thuộc bucket block_bwd.
