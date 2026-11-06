# YÊU CẦU COMPILER TEAM #10 — cắt 117 ms/lớp trong `gpt2_attention_bwd` + `gpt2_mlp_bwd` (native true-k + fused dw)

**Ngày:** 2026-11-01 · **Nguồn:** HANDOFF §0.60 + probe `tv/tests/_am4.tkv` (mới) + PLAN_TO_LE_1X §H2/H5
**Đối tượng:** Compiler team (native runtime + builtin/ABI) · **Ưu tiên:** cao — bucket lớn nhất còn lại
**KHÁC #7/#8/#9:** lần này **KHÔNG phải việc của matcher P6**. Vector hoá đã xong (0 `TkvInt::` trong
mọi thân vòng của `block_bwd`/`attention_bwd`/`mlp_bwd`). Tiền còn lại nằm ở **kernel native + ABI**.

---

> **TRẠNG THÁI (2026-11-02): HẾT — Ask A + C.1 + B + C.2 ĐÃ LÀM.**
>
> **Ask A — `linear_dw_fused` (native, `/fp:precise`, TU riêng `tkv_linear_dw.c`):**
> `BLOCK_BWD_FLAT` 130.875 → **70.375 ms**; bucket 1570.5 → **844.5 ms** (−47%);
> seq8 3656 → **2922 ms** (−20.1% so #9, −29.4% so H1). Loss **EXACT**, gate 135/135,
> trainer 15/15, `_p6ms` A 4.21× · B 5.40× · C 5.06× `bit_bad=0`, self-host `IL_IDENTICAL`.
>
> **Ask C.1 — `dw_proj`: KHÔNG cần mở `trans_a=0`.** `ctxT` đã lưu `[n hang][h cot]` =
> layout `trans_a=1` mà builtin hiện có đọc thẳng ⇒ chỉ cần **bỏ `smm_transpose` trung gian**
> (đúng mẫu P5.4). Không đổi native, không đổi compiler. Bucket 960 → **844.5 ms**.
> **Cảnh báo:** fallback khi native từ chối (`n > 64`) **phải giữ đường cũ nguyên vẹn**
> (`smm_transpose` + `smm_fwd`), **không được** thay bằng `smm_fwd_at` — ở `k ≥ 256`
> `_smm_core` đi packed native (khác thứ tự cộng), probe đo `bad=274384` ⇒ mất loss EXACT.
>
> **Ask B — `linear_dx_fused` (2026-11-02) ĐÃ LÀM** — tách riêng như đề nghị (hợp đồng
> `trans_b=1` khác `dw`). Native `LinearDxFused` đặt trong **cùng TU** `tkv_simd_native.c`,
> gọi thẳng static helper với `i1 = m` → bỏ 248/256 hàng thừa của `SgemmFull` (32× phép nhân
> ở seq8). Builtin: pinvoke + wrapper IL + codegen + đăng ký (`simd_feature.tkv` + `file_io.tkv`).
> Probe `_am6.tkv` **`bit_bad=0`**; `test_dx.c` **`ALL BIT-EXACT`**; kernel **x1.58–1.65×**
> trên shape `m=8`. Fallback `-9 k<256 / -10 m>256 / -11 m%4 / -12 k%256`.
>
> **Bẫy bit-exact tìm ra bằng đo:** `k=300` (`m=200, n=96`) lech **1264 ptu**. Tách confound
> bằng 3 arm → chỉ khi **p-tile le + nhiều block `ic`** cùng xuất hiện mới lệch ⇒ thêm guard
> **`-12` khi `k % 256 != 0`**. Sản phẩm không ảnh hưởng (`out_d` ∈ {768, 2304, 3072}).
>
> **Ask C.2 — cache 5 scratch/12 head (repo-side, thuần `.tkv`) ĐÃ LÀM** — hoist
> `gQ/gK/gV/gDc/gDs` ra trước vòng `for head` (nhánh `n < 128`); mỗi head ghi đầy đủ
> trước khi đọc → không đổi 1 phép tính ⇒ bit-exact. Áp cả 5 mirror. Nhánh worker mỗi
> worker đã nhận đúng 1 head → hoisting ở đó là no-op, không làm.
>
> **Số (cùng phiên 02/11, best-of-3 xen kẽ, .NET 10):** seq8 3687 → **2735 ms (−25.8%)**,
> seq256 5547 → 5531, seq1024 15688 → 16500 (cả hai bất biến đúng như thiết kế:
> `m=1024>256` → native từ chối; `m=256=T` → không bỏ được hàng).
> Loss **EXACT 3/3**, gate **135/135** (×2), trainer **15/15** (×2), self-host **`IL_IDENTICAL`**.
>
> Chi tiết: [Release/docs/e_handoff/2026-11-01-request10-linear-dw-fused-true-k.md](../Release/docs/e_handoff/2026-11-01-request10-linear-dw-fused-true-k.md) §7 +
> [2026-11-02-request10-linear-dx-fused.md](../Release/docs/e_handoff/2026-11-02-request10-linear-dx-fused.md) + HANDOFF §0.61/§0.62/§0.63.
>
> **Lưu ý khi đo lại:** `simd_probe_am` có 2 biến thể — `BLOCK_BWD` (không `gflat`, `goff=-1`,
> native KHÔNG chạy) và **`BLOCK_BWD_FLAT`** (production). Chỉ số liệu latter là số thật.

## 1. Bối cảnh — bucket không còn ceremony, chỉ còn 2 hàm con

Sau #9 (`§0.60`): `12 × block_bwd` = **130.875 ms/gọi = 1570.5 ms** (batch 8, `simd_probe_am`).
Trong đó probe `_am4` (mới, cùng compiler 01/11) đo **cùng phiên**:

| vùng | ms/gọi | % bucket |
|---|---|---|
| `gpt2_attention_bwd` | **43.94** | 34% |
| `gpt2_mlp_bwd` | **73.25** | 56% |
| glue + `layernorm_bwd` + 4 vòng `n*h` + `t_zeros` của `block_bwd` | ~13.7 | 10% |

⇒ **90% bucket là 2 hàm con**; ~13.7 ms còn lại là `layernorm_bwd` + 2 vòng `n*h` + `t_zeros`
của chính `block_bwd` + nhiễu context (AM §0.60 đo `SUM_cac_vung` = 125 ms ⇒ glue ~6 ms khi
gọi lồng trong cùng hàm — chênh lệch là do đo đứng riêng vs đo trong `block_bwd`).

---

## 2. Bằng chứng — phân rã pha (seq8: n=8, h=768, hd=64, ih=3072, 12 head)

`tv/tests/_am4.tkv` — mỗi pha đo riêng với **shape thật**, best-of-3, `now_us` + batching;
cột "khớp" = tổng các pha so với hàm tổng (chênh = phần chưa phân bổ: softmax loop, packing,
writeback trong head loop, overhead gọi hàm).

### 2.1 `gpt2_mlp_bwd` = **73.25 ms**

| pha | ms | % | ghi chú |
|---|---|---|---|
| **`gpt2_linear_bwd(x, w1, d_inner, …, gflat, o_w1, o_b1)`** | **38.06** | **52%** | xem 2.3 |
| `smm_fwd_at(a, gout, ih, h, n)` → dw2 | **20.5** | **28%** | GEMM dw2 2.36M ra |
| `smm_dot(gout, w2, n, ih, h)` → dx2 | 7.81 | 11% | GEMM dx2 |
| `gpt2_linear(x, w1, b1, n, h, ih)` (fwd recompute) | 5.88 | 8% | GEMM + bias |
| `dw2 → gflat` copy (2.36M) | 2.94 | 4% | vòng offset |
| `gelu_erf_bwd` (threaded) | 0.94 | 1.3% | |
| `_gelu_fwd_par` | ~0.0 | — | dưới sàn đo |
| vòng `db2` (n·h) | ~0.0 | — | đã vector hoá |
| **tổng pha** | **74.4** | | khớp total (sai số ≤1.2 ms) |

### 2.2 `gpt2_attention_bwd` = **43.94 ms**

| pha | ms | % |
|---|---|---|
| **`gpt2_linear_bwd(x, w_qkv, dqkv, …, gflat, o_qkv, o_bqkv)`** | **22.5** | **51%** |
| `dw_proj` (transpose + `smm_fwd(768,768,8)` + db loop + dw writeback 0.59M) | 5.75 | 13% |
| 12 head block (2×dot(8,8,64) + 4×fwd + 2 transpose + 5 `t_zeros`/head) | 4.39 | 10% |
| `gpt2_linear(x, w_qkv, b_qkv, n, h, 3h)` (qkv fwd) | 4.88 | 11% |
| 5 × `t_zeros` của hàm ([8,768]×2, [8,2304], [768,768], [768]) | 2.11 | 5% |
| `smm_dot(gout, w_proj, n, h, h)` (dctx) | 1.0 | 2% |
| **tổng pha** | **38.6** | khớp total (chênh 5.3 = softmax/packing/writeback trong head loop) |

### 2.3 Bên trong `gpt2_linear_bwd` (= 38.06 ms với w1, 22.5 ms với w_qkv)

| pha | w1 (768→3072) | qkv (768→2304) |
|---|---|---|
| `dx = smm_dot(gy, w, n, in_d, out_d)` | 12.69 ms | 5.81 ms |
| `dw = smm_fwd_at(x, gy, in_d, out_d, n)` | **20.5 ms** | **13.69 ms** |
| `dw → gflat[goff + k]` (copy 2.36M / 1.77M) | 1.94 ms | ~2 ms |
| vòng `db` + ghi gflat `dboff` | ~0 | ~0 |

---

## 3. Chẩn đoán — tiền nằm ở **tile build O(T²) cho k thật = n = 8**

`dw = smm_fwd_at(x, gy, in_d, out_d, n)` là GEMM **contraction k = n = 8** nhưng đường
`_smm_core` build tile **T × T = 64 × 64** rồi mới `simd_gemm(AtB, BtB, Ct, T)`:

- T=64, k thật = 8 ⇒ **8× thừa** ở pha BUILD (và ở ACC);
- `smm_fwd_at` ra 2.36M phần tử ⇒ **36 864 tile**, mỗi tile build 2 × 4096 phần tử
  ⇒ **~302 M phần tử tile** cho một GEMM mà chỉ có 2.36M × 8 = **18.9M MAC hữu dụng**;
- đo được 20.5 ms ⇒ **~14 GFLOP/s hữu dụng**, trong khi GEMM lớn của repo đã parity
  (~550 GFLOP/s, §0.41/§0.48) ⇒ đây **không phải giới hạn phần cứng**, là **thừa padding + 2 pass**.

P9/Task-2 đã vá `tkv_k_eff` cho **rows của B** (đường `SgemmF32`), nhưng chi phí còn lại nằm ở
**BUILD tile** phía `.tkv` (`_tile_a_at_w`/`_tile_b_w`) và ở **ACC** — hai pha mà `k_eff` không chạm tới.

Thêm 2 pass vô ích nữa:

- `dw` được sinh ra trong một `Tensor` mới rồi **copy sang `gflat`** (1.9–2.9 ms/lần) — cùng
  2.36M phần tử, đọc-rồi-ghi lần nữa;
- `dw2` (20.5 ms) rồi cũng **copy sang gflat** (2.94 ms) ở `mlp_bwd`.

---

## 4. Ask A (bắt buộc, đòn lớn nhất) — native dw-GEMM **true-k**, ghi thẳng vào `gflat`

**Hợp đồng đề nghị** (1 builtin, cùng khuôn `f32_list_add`: IL wrapper lấy `_items` + pin, native
nhận con trỏ):

```tkv
linear_dw_fused(x, gy, gdw, gdw_off, gdb, gdb_off, n, in_d, out_d) -> i32
```
- `x`: `list[f32]` [n][in_d], `gy`: `list[f32]` [n][out_d] — đọc trực tiếp (zero-copy);
- `gdw`: `list[f32]`, ghi **`gdw[gdw_off + k*out_d + o] += dw[k][o]`** (accumulate, KHÔNG tạo
  tensor trung gian, KHÔNG copy pass thứ 2);
- `gdb`: nếu `gdb_off >= 0` thì `gdb[gdb_off + o] += Σ_i gy[i][o]` (thứ tự i TĂNG DẦN);
- **contraction = n thật** (n = 8), tile build theo `ceil(n/8)` — không dùng T=64 cho k;
- trả `rc` (0 = ok, khác 0 = shape/lệch biên) thay vì ghi tràn.

**Hợp đồng số học (KHÔNG thương lượng):** với mỗi `(k,o)` cộng **tuần tự p = 0..n-1** — đúng thứ tự
`p` tăng dần của `_p58_core` hiện tại ⇒ **BIT-EXACT** với `smm_fwd_at` + copy. Không FMA-đổi-thứ-tự,
không chia tile theo hướng k làm đổi thứ tự cộng.

**Kỳ vọng (ước lượng — phải đo):** dw2 20.5 → ~5, dw1 20.5 → ~5, dwqkv 13.7 → ~4; bỏ luôn 3 pass
copy (1.9 + 2.9 + 2 ≈ 6.8) ⇒ **−45 ms/lớp**.

**Fallback nếu không làm được fused:** chỉ cần `smm_fwd_at_truek(a, b, m, n, k, K)` — native nhận
thêm `k` thật và tự chọn `T = ceil(k/8)*8`, không đụng layout — vẫn ăn ~70% phần trên.

---

## 5. Ask B (tuỳ chọn, +10–12 ms/lớp) — fused toàn bộ `linear_bwd`

Mở rộng Ask A thành **một lời gọi** cho cả `gpt2_linear_bwd`:

```tkv
linear_bwd_fused(x, w, gy, dx, gdw, gdw_off, gdb, gdb_off, n, in_d, out_d) -> i32
```
- `dx[i][k] = Σ_o gy[i][o]*w[k][o]` (thứ tự o TĂNG DẦN — khớp `smm_dot`),
- dw/db như Ask A,
- 1 pass đọc `gy`, 1 pass ghi `dx`, ghi thẳng `gflat` — bỏ 3 lần cấp phát `Tensor` (`dx`, `dw`,
  `db`) và 2 pass copy.

**Kỳ vọng:** `gpt2_linear_bwd` 38.06 → **~15** (w1), 22.5 → **~9** (qkv) ⇒ thêm **−12 ms/lớp** so
với Ask A. Nếu phải chọn 1, chọn **Ask A** (rủi ro thấp hơn, cùng đường bit-exact).

---

## 6. Ask C (thấp) — 2 chỗ còn lại nếu còn ngân sách

1. `dw_proj` attention (5.75 ms): transpose + `smm_fwd(768,768,8)` + db loop + writeback 0.59M
   — cùng bệnh tile-build T=64 cho k=8; dùng lại Ask A/B là gần đủ.
2. 12 head block (4.39 ms): 2×GEMM(8,8,64) + 4×GEMM + 2 transpose + **5 `t_zeros`/head** = 60 cấp
   phát/lần gọi. Có thể xin **scratch theo head trong native** hoặc để repo tự cache (repo-side).

---

## 7. Acceptance (chốt số — đo bằng `_am4`, cùng phiên)

| mục | ngưỡng | baseline 01/11 |
|---|---|---|
| `gpt2_mlp_bwd` (seq8) | ≤ **40 ms** | 73.25 |
| `gpt2_attention_bwd` (seq8) | ≤ **30 ms** | 43.94 |
| `BLOCK_BWD` (batch 8) | ≤ **80 ms/gọi** | 130.875 |
| Bucket `12 × block_bwd` | ≤ **960 ms** | 1570.5 |
| Full-step 3 seq (best-of-3 xen kẽ) | không seq nào regress; seq8 ≤ **3400 ms** | 3735 / 5531 / 15891 |
| Loss chain 3 seq | **EXACT** `10.572291→7.0119185` · `10.756927→11.807267` · `10.760399→13.442915` | ✅ |
| Gate | **135/135 + trainer 15/15** | ✅ |
| Self-host | `tools/vong.exe` → `IL_IDENTICAL` | ✅ |
| Probe `_p6ms` | ≥4× mỗi shape, `bit_bad=0` | A 4.12–4.44 · B 5.98–6.09 · C 5.06–5.47 |

**Kỳ vọng tổng (Ask A):** BLOCK 130.9 → **~87** ⇒ bucket ~1044 ms (−33%); thêm Ask B ⇒ **~75**
⇒ bucket ~900 ms (−43%). Con số này **phải đo lại**, không dùng để tuyên bố.

---

## 8. Điều kiện (giữ nguyên từ #7/#8/#9)

- **Loss EXACT** — cấm re-order reduction; mọi kernel mới phải cộng theo thứ tự p/o/i tăng dần
  như hàm `.tkv` đang thay thế, và có probe bit-check (`bit_bad=0`) trước khi wire.
- Gate 135/135 + trainer 15/15 sau mỗi thay đổi; self-host `IL_IDENTICAL`.
- **Đo trước, sửa sau** — báo số đo xen kẽ torch cùng phiên trước khi ghi HANDOFF.
- Không trộn H3 (threading 14 luồng) vào request này: mọi số ở trên là **đơn luồng**.

---

## 9. Cách lặp lại phép đo

```bash
cd D:/TokenVector.Inference
/D/TokenVector/Release/dist/tkvc-tv.exe tv/tests/_am4.tkv out/r9_am4.exe main
cp out/r9_am4.exe out/r9_am4_n10.dll
cp out/_am.runtimeconfig.json out/r9_am4_n10.runtimeconfig.json
dotnet out/r9_am4_n10.dll          # in ra từng pha của 2 hàm + tổng 2 hàm
```

Probe **không** đụng code train; mọi pha gọi đúng hàm production với shape thật, nên số của
`ATT_TOTAL`/`MLP_TOTAL` so trực tiếp được với `AM BLOCK_BWD`.

---

## 10. Ghi chú scope

- **Không** mở lại matcher P6/`TkvInt`: đã 0 trong thân vòng ở cả 3 hàm (§0.60).
- **Không** mở lại `mu`/LICM: đã đo là sàn `vaddsd`/JIT (§0.60 §4).
- `t_zeros` 5 alloc/lần của attention (2.11 ms) là **repo-side** (scratch cache) — không tính
  vào acceptance của request này, nhưng repo sẽ làm song song nếu compiler team cần bớt biến.
