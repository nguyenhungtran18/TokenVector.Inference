**TRẠNG THÁI 2026-11-11: ĐÃ DONE** — `adamw_step_glm` nhận `coef` (13 tham số), loss EXACT, gate 135/135, trainer 15/15; opt 781 → 438–469 ms; seq1024 0.989× torch (`benchmark/BENCHMARKS.md` §9.21–§9.22).

# YÊU CẦU COMPILER TEAM #14 — `adamw_step_glm` nhận thêm `coef`: bỏ 3 bước phụ khi clip hoạt động

**Ngày:** 2026-11-11
**Nguồn:** HANDOFF §0.152/§0.153 + `benchmark/BENCHMARKS.md` §9.19
**Đối tượng:** Compiler team (`build/selfhost/tv_gop.tkv` → `compile_adamw_step_glm`, + native `tkv_simd_native.c`)
**Ưu tiên:** cao — **~220 ms/bước** trên seq1024, là hạng mục lớn nhất còn lại sau #13.
**Khác #13:** #13 sửa thân IL một method có sẵn. Lần này **thêm 1 tham số** vào một builtin có sẵn. Không đổi ngữ nghĩa `.tkv`, không đổi ABI cũ nếu làm đúng cách dưới đây.

---

## 1. Bối cảnh — chỗ tiền nằm

`AdamW.step()` (`tv/training/trainer.tkv` L192–276) có **hai đường**, chọn theo `coef`:

```
gn = self.grad_norm()          # ~11.155 với gradient thật
coef = 1.0
if gn > 0.0:
    coef = self.clip / (gn + 1e-6)
    if coef > 1.0: coef = 1.0
```

- **`coef == 1.0`** (không clip, hoặc clip rất lớn) → đường **NHANH**:
  `adamw_step_glm(ppl, pml, pvl, gl, ...)` đọc `gl` list trực tiếp.
- **`coef != 1.0`** (clip hoạt động) → đường **CHẬM**, 3 bước phụ:

```
r1  = f32_copy_to_buf(gl, pgl, self.n)          # chép 474 MB list -> buffer
rzp = f32_zero(ptl, self.n)                     # zero 474 MB
rax = f32_addcmul(ptl, pgl, ponl, coef, self.n) # 474 MB đọc + 474 MB ghi
stw = adamw_step(ppl, pml, pvl, ptl, ...)       # kernel buffer
```

`ponl` là buffer toàn `1.0` (khởi tạo trong `__init__`), nên `f32_addcmul(ptl, pgl, ponl, coef, n)`
thực chất chỉ là **`ptl = coef * gl`** — một phép nhân vô hướng, nhưng đi qua 3 lượt
chạm 474 MB.

**Production LUÔN đi đường chậm**, vì gradient thật cho `gn ≈ 11.155` nên
`coef = 1.0/11.155 ≈ 0.0896 ≠ 1.0`. Đường nhanh chỉ chạy khi gradient bằng 0.

---

## 2. Số đo (n = 124,439,808 = 474.7 MB mỗi buffer, Xeon E5-2695 v3)

Đo từng mảnh bằng probe riêng (`tv/tests/_clip_m1..m5.tkv`):

| mảnh | ms |
|---|---:|
| `r1` = `f32_copy_to_buf` | **156** |
| `rzp` = `f32_zero` | **125** |
| `rax` = `f32_addcmul` | **172** |
| `adamw_step` (buffer) | 343 |
| **`adamw_step_glm` (list)** | **343** |

Đường chậm tổng: 156 + 125 + 172 + 343 = **796 ms**.
Đường nhanh tổng: 343 ms (+ `grad_norm` 203 ms cho cả hai đường).

**Đo end-to-end `opt.step()` với gradient thật** (`tv/tests/_clip_real.tkv`,
`g[i] = 0.001` → `gn = 11.155`):

| lần chạy | `clip=1.0` (chậm) | `clip=1e9` (nhanh) | chênh |
|---|---:|---:|---:|
| 1 | 781 | 563 | **−218** |
| 2 | 875 | 547 | **−328** |
| 3 | 781 | 657 | **−124** |

Chênh dao động 124–328 ms, trung bình **~220 ms**. Dao động vì cả hai đường đều
memory-bound trên 474 MB × nhiều buffer, nhưng **xu hướng nhất quán: đường chậm
luôn chậm hơn**.

---

## 3. Ask — thêm tham số `coef` vào `adamw_step_glm`

Chữ ký hiện tại (12 tham số):

```
adamw_step_glm(p, m, v, g, n, lr, b1, b2, eps, wd, bc1s, bc2s) -> i32
```

Đề nghị thêm `coef` thành tham số thứ 5 (sau `g`, trước `n`) hoặc cuối cùng —
compiler team chọn vị trí nào thuận, miễn nhất quán:

```
adamw_step_glm(p, m, v, g, n, lr, b1, b2, eps, wd, bc1s, bc2s, coef) -> i32
```

**Ngữ nghĩa:** `g_eff = coef * g` được tính **bên trong kernel**, một lượt, không
cấp buffer trung gian. Khi `coef == 1.0` phải cho kết quả **bit-identical** với
đường hiện tại (nhân với 1.0 là phép đồng nhất trong IEEE-754, không đổi bit).

Trong kernel native, chỗ đọc `g` đổi từ:

```c
float gv = g[i];
```

thành:

```c
float gv = coef * g[i];
```

Không đổi thứ tự tích luỹ, không đổi công thức AdamW, không đổi layout `p/m/v`.

### Vì sao cách này đúng

`f32_addcmul(ptl, pgl, ponl, coef, n)` với `ponl` toàn `1.0` cho `ptl[i] = coef * pgl[i] * 1.0`.
Nhân `coef` ngay trong kernel cho **cùng giá trị bit** (cùng một phép nhân, cùng
một toán hạng), nhưng bỏ được:
- `r1` (156 ms, chép list → buffer)
- `rzp` (125 ms, zero buffer)
- `rax` (172 ms, đọc + ghi buffer)
- và buffer `pt` (474 MB working set)

**Tiết kiệm: 453 ms lý thuyết, ~220 ms thực đo end-to-end** (phần chênh do
`adamw_step` buffer và `adamw_step_glm` list đều 343 ms — kernel như nhau).

### Phương án dự phòng (nếu không muốn đổi chữ ký)

Thêm builtin mới `adamw_step_glm_c(p, m, v, g, n, lr, b1, b2, eps, wd, bc1s, bc2s, coef)`
bên cạnh cái cũ. Không phá vỡ caller nào. Nhưng hai entry gần trùng nhau thì khó
bảo trì hơn — tôi nghiêng về thêm tham số.

---

## 4. Kỳ vọng

| hạng mục | hiện tại | sau khi sửa |
|---|---:|---:|
| `opt` trong BPHASE @seq1024 | 781 | **~560** |
| full step @seq1024 | 4219 | **~4000** |
| working set | 7.4 GB | **−474 MB** (bỏ `pt`) |

Mục tiêu ≤1.0× so với torch (`4040.2` ms @seq1024) cần cắt **179 ms**. Ask này
một mình **đủ vượt**, và còn dư cho seq8/seq256 (nơi `opt` cũng cố định 781 ms).

**Lợi ích lan rộng:** mọi training có bật `clip_grad_norm_` đều ăn theo. Đó là
mặc định của hầu hết recipe GPT.

---

## 5. Điều kiện bắt buộc

- **`coef == 1.0` phải bit-identical** với `adamw_step_glm` hiện tại. Verify bằng
  `max_abs_diff == 0.0`, không bằng gate TOL.
- **`coef != 1.0` phải bit-identical** với đường cũ (`f32_addcmul` + `adamw_step`).
  Cùng một phép nhân `coef * g[i]`, cùng thứ tự tích luỹ → phải 0.0.
- **Loss EXACT** cả 3 seq: `10.572291 / 10.756927 / 10.760399`.
- **Gate 135/135** (9 suite nhỏ 102 + `gpt2_full` 33, fails=0).
- **Trainer 15/15.**
- **Self-host `tools/vong.exe` → `IL_IDENTICAL`.**
- **Không regress seq8/seq256.** Đo cả 3 seq, best-of-3.

---

## 6. Cách đo lại sau khi làm

```powershell
# 1) bit-identical cua coef=1.0 (phai 0.0)
D:\TokenVector\Release\dist\tkvc-tv.exe tv/tests/_clip_real.tkv benchmark/clipreal.exe
Set-Content benchmark/clipreal.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/clipreal.exe

# 2) full step 3 seq
#    tv/tests/_seq_cfg.txt = "<seq> / 1 / 1 / 1"
dotnet benchmark/bseq_new.exe

# 3) parity
dotnet out/gft2.exe
```

**Tiêu chí đạt:** `opt.step()` với `clip=1.0` ≤ **600 ms** (hiện 781–875);
full step @seq1024 ≤ **4050 ms** (hiện 4219); loss EXACT; gate 135/135.

---

## 7. Ghi chú phương pháp — ba lần tôi đo sai trước khi ra số này

Ghi lại để không ai lặp lại:

1. **Probe đầu dùng `g` toàn 0** → `grad_norm = 0` → `if gn > 0.0` không bao giờ
   đúng → `coef = 1.0` → **cả hai nhánh đều đi đường nhanh**. Kết luận "clip không
   tốn gì" là SAI. Phải có gradient thật (`g[i] = 0.001`) mới vào được đường clip.
2. **`adamw_step_glm(b1, b1, b1, ...)`** — cùng một buffer cho `pp/pm/pv` → kernel
   chỉ chạm 474 MB thay vì 1.42 GB → ra 125 ms thay vì 343. Phải cấp buffer RIÊNG.
3. **`zeros_list` trả dtype `i32`**, không phải `list[f32]`. Mọi thao tác trên
   `list[f32]` phải đi qua **tham số hàm**, không index trực tiếp trong `main`.

Ngoài ra: quá nhiều local trong một hàm → `InvalidProgramException`. Tách mỗi phép
đo ra một file riêng.
