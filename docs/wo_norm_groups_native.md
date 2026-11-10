# WORK ORDER — `_norm_groups` dùng builtin `f32_grad_norm` (1 group)

**Ngày:** 2026-11-10
**Nguồn:** §0.136 (điều tra `opt.step`), đính chính sau khi đo lại
**Trạng thái:** spec sẵn sàng, chưa thực thi
**Ước tính:** 390 ms/lần gọi `grad_norm`

---

## 0. ĐÍNH CHÍNH §0.136 — không cần sửa compiler

§0.136 kết luận "`F32GradNorm` là dead code, phải đăng ký builtin trong compiler".
**Kết luận đó sai.** Đo lại thì:

| Bằng chứng | Kết quả |
|---|---|
| `f32_grad_norm(g, n)` gọi từ `.tkv` | **biên dịch được, chạy được** |
| IL sinh ra (`out/gnb.il` L2079) | `pinvokeimpl("tkv_simd_native.dll" as "F32GradNorm")` |
| IL sinh ra (L4757) | `call TkvSimd::F32GradNormNative` |
| Compiler source | `simd_feature.tkv` L4009 `compile_f32_grad_norm`, L4132 `register_expr_builtin(reg, 'f32_grad_norm', ...)` |

Builtin **đã có sẵn và đã hoạt động**. Không có việc gì cho compiler team.

Cái thiếu chỉ là: **`_norm_groups` không gọi nó.** Grep `f32_grad_norm` trong toàn
bộ cây `.tkv` trước bản này: 0 kết quả. Builtin tồn tại, không ai dùng.

Bài học ghi vào §0.136 §4 vẫn đúng, nhưng lý do khác: không phải "thiếu builtin",
mà là "builtin có mà không ai nối vào".

---

## 1. Hợp đồng số liệu — đã verify bit-identical

`F32GradNorm` (`tkv_simd_native.c` L2014, comment hợp đồng L2005–2013):

- bình phương trên **f32**: `float p = g[i]*g[i]`
- tích lũy **f64 tuần tự** i = 0..n-1, không gom, không song song
- `sqrt` trên f64, trả f64

`_norm_groups` với **một** group hiện tại:

```
s = 0.0
for i in range(cnt): s = s + g[off+i] * g[off+i]
nrm = sqrt(s)
tot = tot + nrm * nrm
return sqrt(tot)          # kiểu trả: f32
```

Với một group, đường này có **một vòng `sqrt(nrm*nrm)` thừa** — về toán là đồng
nhất, nhưng về dấu phẩy động thì `sqrt(x*x)` không luôn bằng `x`.

**Đo thực tế (n = 124,439,808, `g[i] = 0.001`):**

| Cách gọi | ms | giá trị |
|---|---|---|
| `_norm_groups(g, [n])` | **437** | `11.15526008605957` |
| `f32_grad_norm(g, n)` **không ép kiểu** | 47 | `11.155259828125383` |
| `f32(f32_grad_norm(g, n))` **có ép kiểu** | **47** | `11.15526008605957` |

- Không ép kiểu: lệch **2.58e-07** → KHÔNG bit-identical
- Có ép `f32(...)`: lệch **0.0** → **BIT-IDENTICAL**

Nguyên nhân: `_norm_groups` khai báo `-> 'f32'` nên bị cắt về f32; native trả f64.
Ép `f32()` tái lập đúng phép cắt đó. Vòng `sqrt(nrm*nrm)` thừa **không** tạo khác
biệt sau khi cắt f32 — đã kiểm chứng bằng 0.0.

**Kết luận: `f32(f32_grad_norm(g, n))` bit-identical với đường hiện tại, nhanh 9.3x.**

---

## 2. Thay đổi cần làm

### 2.1 File

Bốn mirror đều có `_norm_groups`:

- `tv/training/trainer.tkv`
- `tv/training/tv.training.trainer.tkv`
- `tv/tests/tv.training.trainer.tkv`
- `tv/tools/tv.training.trainer.tkv`

Sửa **cả bốn**, cùng một nội dung.

### 2.2 Patch

Chèn ngay đầu thân `_norm_groups`, **trước** khối `tot_elem >= 4000000`:

```
def _norm_groups(g: 'list[f32]', groups: 'list[i64]') -> 'f32':
    ng = len(groups)
    # §0.137: 1 group -> builtin native (bit-identical, 9.3x).
    # f32(...) bat buoc: _norm_groups tra f32, native tra f64. Thieu ep ->
    # lech 2.58e-07. Co ep -> 0.0 (da verify n=124M).
    if ng == 1:
        return f32(f32_grad_norm(g, groups[0]))
    ... phan con lai giu nguyen ...
```

Điều kiện `ng == 1` là **bắt buộc**. Xem §3.

### 2.3 Không đụng gì khác

- Không sửa compiler
- Không sửa `tkv_simd_native.c` (kernel đã đúng)
- Không sửa `step()`, `grad_norm()`, `AdamW`
- Không đổi chữ ký hàm

---

## 3. Vì sao CHỈ `ng == 1`

Công thức PyTorch `clip_grad_norm_`:

```
norm = sqrt( sum_k norm_k^2 ),  norm_k = sqrt( sum_{i in group k} g_i^2 )
```

Với nhiều group, phải tính **từng** `norm_k` rồi gộp **đúng thứ tự** group — đó
chính là lý do `_norm_groups` hiện có 12 worker tính `s` từng nhóm rồi main gộp
tuần tự.

`F32GradNorm` nhận `(g, g_len, n)` và tính trên **một dải liên tục** `[0, n)`.
Không có tham số `offset`, không có bản per-group. Nên:

- **1 group** → dải chính là toàn mảng → dùng được
- **nhiều group** → phải gọi nhiều lần với offset khác nhau, mà builtin không nhận
  offset → **không dùng được**

Layout GPT-2 124M hiện tại dùng **đúng 1 group** (`groups = [n]` trong
`_mk_opt` và `_opt_split.tkv`), nên nhánh nhanh phủ được trường hợp thật.

Nếu sau này cần nhiều group nhanh: phải thêm builtin mới `f32_grad_norm_range(g,
off, cnt)` — **work order riêng**, không nằm trong spec này.

---

## 4. GIÁ TRỊ THỰC TẾ: 0 ms/step ở cấu hình bench hiện tại

**Đọc kỹ trước khi làm.** `_mk_opt` (`simd_bench_124m_seq.tkv` L250–251) truyền
`clip = 1.0`. `step()` chỉ gọi `grad_norm()` khi `self.clip != 1.0`:

```
gn = 0.0
if self.clip != 1.0:
    gn = self.grad_norm()
```

Nên trong bench, `grad_norm` **không chạy**. Tối ưu nó trả về **0 ms/step** cho
con số bench.

Giá trị thật nằm ở chỗ khác: **training thật có bật clip** (chuẩn là `clip=1.0`
theo giá trị norm, không phải "tắt"). Khi bật, mỗi step trả 390 ms. Với 1000 step
là **390 giây**.

Đo A/B trên bench để chứng minh tối ưu này cũng **không** cho kết quả sạch: hai
optimizer chỉ khác `clip`, chênh lệch nằm trong nhiễu ±140 ms dù bản có clip phải
làm thêm 438 ms `grad_norm`. Đừng dùng bench để verify work order này.

**Verify bằng probe trực tiếp** (§5), không bằng bench.

---

## 5. Kế hoạch verify

### 5.1 Bit-exact (bắt buộc)

Probe `tv/tests/_gn_final.tkv` (đã có, chạy được):

```
A) _norm_groups      ms=437  r=11.15526008605957
B) f32(native)       ms=47   r=11.15526008605957
max_abs_diff = 0.0   BIT-IDENTICAL: True
speedup = 9.3x   tiet kiem = 390 ms
```

Sau khi patch, probe này phải cho **A ≈ B về thời gian** (cả hai đều ~47 ms) và
**vẫn 0.0**. Nếu A vẫn 437 ms → patch không vào đúng file mirror đang được resolve.

### 5.2 Gate

- `gpt2_full_tests` **33/33 PASS** (bắt buộc)
- `trainer_tests` **15/15 PASS** — quan trọng nhất, đây là test của `AdamW`
- `gap_tests` exit=82 — **pre-existing, không sửa, không tính là fail**

### 5.3 Loss EXACT

Ba seq, phải khớp tuyệt đối:

```
10.572291 / 10.756927 / 10.760399
```

Vì `clip = 1.0` trong bench nên `grad_norm` không chạy → loss **không thể đổi**.
Nếu loss đổi, patch đã chạm vào chỗ khác — dừng và kiểm tra lại.

Muốn verify đường có clip thật: dựng optimizer `clip = 0.5`, chạy 3 step, so
`opt.p` từng phần tử với bản trước patch. Phải **0.0**.

### 5.4 Self-host

`vong.exe` → **IL_IDENTICAL**.

---

## 6. Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| Quên ép `f32()` | **cao** | lệch 2.58e-07, gate trainer_tests sẽ bắt (TOL 1e-5 → có thể LỌT) — phải tự verify 0.0, đừng tin gate |
| Patch 1 mirror, quên 3 mirror còn lại | **cao** | test/bench resolve bản khác → đo không thấy đổi. Sửa đủ 4 |
| `ng == 1` nhưng `groups[0] != len(g)` | trung bình | `F32GradNorm` trả `-1.0` khi `g_len < n`. Thêm assert hoặc tin caller |
| Tưởng ăn được 390 ms/step trên bench | **cao** | §4: bench tắt clip, giá trị = 0. Đừng báo cáo sai |

Rủi ro thứ nhất đáng lo nhất: `trainer_tests` dùng TOL 1e-5, mà lệch không ép
kiểu chỉ 2.58e-07 → **gate sẽ PASS dù sai bit**. Phải verify bằng `max_abs_diff == 0.0`
chứ không bằng gate.

---

## 7. Tóm tắt một dòng

`_norm_groups` thêm 2 dòng: `if ng == 1: return f32(f32_grad_norm(g, groups[0]))`.
Không sửa compiler. Bit-identical 0.0. 437 → 47 ms. Nhưng bench tắt clip nên
số đo bench sẽ không đổi — verify bằng probe, không bằng bench.
