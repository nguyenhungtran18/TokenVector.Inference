**TRẠNG THÁI 2026-11-11:** Gap mà doc này nhắm tới **đã đóng**: backward @seq1024 = 2328 ms (TKV mode 2) vs torch 2357.0 (e, T=14) — gần ngang torch; tổng 0.989× (`BENCHMARKS.md` §9.22). Giữ file làm hồ sơ yêu cầu; không thực thi.

# YÊU CẦU COMPILER TEAM #12 — BWD non-GEMM native + opt.step copy-back (sau #10/#11)

**Ngày:** 2026-11-05
**Nguồn:** HANDOFF §0.72 + PLAN_BEAT_PYTORCH §2c/§2d
**Đối tượng:** Compiler team (native runtime `tkv_simd_native.c` + builtin/ABI)
**Ưu tiên:** cao — sau #10/#11, gap còn lại nằm ở non-GEMM interpreted + opt.step.
**KHAC #10/#11:** #10 là `linear_*_fused`, #11 là thread-safety dispatcher. Lần này là **các op elementwise/reduction còn lại trong backward** và **copy `opt.step()`**.

---

## 1. Bối cảnh (số đo thực tế, seq1024, 14 luồng)

| pha | TKV | torch | tỷ lệ |
|---|---:|---:|---:|
| backward | 10593 ms | 2347.9 ms | **4.51x** |
| `opt.step()` | 1188 ms | 660.7 ms | **1.80x** |
| copy-back / zero_grad | 141 ms | 52.2 ms | — |
| **Tổng** | **15266 ms** | **4134 ms** | **3.7x** |

Non-GEMM: **~8.2 s** trong 11.2 s chênh lệch. `gpt2_block_bwd` 1 luồng: `mlp_bwd` 50.5%, `attention_bwd` 36.6%, `layernorm_bwd` 3.0%. GEMM đã native, `attention_bwd`/`mlp_bwd` còn mắc ở các vòng `.tkv` phần mềm (gelu/softmax/LN/bias-grad) và scale luồng (2.3x vs torch 3.7x).

---

## 2. Ask A — native fused cho các op non-GEMM trong backward

Không đổi thuật toán, không đổi thứ tự tích luỹ. Mỗi op: 1 pass native, đọc/ghi trực tiếp `.data` (list[f32]) hoặc handle zero-copy nếu đã có API.

1. **`layernorm_bwd_fused(x, gy, g, n, h, gflat?, o_dg?, o_db?)`** → `dx, dg, db`
   - Thay `gpt2_layernorm_bwd`; giữ loss EXACT.
   - dg/db ghi thẳng vào `gflat` nếu có offset; không alloc `t_zeros([h])` mỗi block.

2. **`gelu_erf_bwd(inner, dy, n)`** → `dx` (erf-formulation, không approximate)
   - Thay `op_gelu_erf_bwd` hiện tại; 1 pass AVX2; `n = 3.14M` @ seq1024.

3. **`softmax_causal_bwd(dP, P, n)`** → cộng dscore (in-place)
   - Dùng trong `attention_bwd`; causal mask giữ nguyên.

4. **`bias_grad(gy, n, cols)`** → `db[cols]`
   - Thay vòng tích luỹ bias trong `linear_bwd`.

**Ky vong:** `mlp_bwd` 73.25 → ~45-50 ms/block; `layernorm_bwd` ~3.0% → ~1%; giảm ~10-15% bwd seq1024.

---

## 3. Ask B — cạnh tranh song song op-level trong DLL (optional, nếu Ask A không đủ)

- Mở API op-level song song cho `layernorm_bwd_fused`, `gelu_erf_bwd`, `bias_grad`
  (`n` lớn: nhiều GB phần tử/lượt); thread trong DLL, không qua `.tkv` scheduler.
- Nhóm A/B phải giữ nguyên thứ tự tích luỹ (chunk by rows, không split reduction) → bit-exact.
- Ky vong: scale 2.3x → gần 3.0-3.3x (torch 3.64x là trần).

---

## 4. Ask C — `opt.step()`: đo lại `adamw_step_glm` / zero-copy `r1`/`r3`

- Hiện tại `r1 = f32_copy_to_buf(gl, pgl, n)` (498 MB) + `r3 = f32_copy_from_buf(ppl, pl, n)`.
- Đã thử `adamw_step_glm` trước đây và bị từ chối (`+3.2 s` do alloc/marshal).
  **Yêu cầu lần này:** nếu native có thể đọc trực tiếp `list[f32]` mà không marshal thêm byte, đo lại A/B xen kẽ trên cùng máy:
  - `adamw_step_list_native(ppl, pml, pvl, gl, n, …)` đọc `gl` trực tiếp, bỏ `r1`;
  - `p_copy_from_buf` có thể giữ nguyên `r3` nếu chứng minh là bắt buộc.
- Ky vong: `opt.step()` 1188 → ~700-800 ms (giảm copy + memory traffic).

---

## 5. Điều kiện bắt buộc

- Giữ mô hình `row-major` hiện tại; không đổi layout `p/m/v`.
- Loss EXACT trên cả 3 seq (`10.572291/10.756927/10.760399`).
- Gate 135/135 + trainer 15/15 + `simd_probe_am` `BLOCK_BWD_FLAT` bit_bad=0.
- Self-host `tools/vong.exe` → `IL_IDENTICAL`.
- Đo xen kẽ torch cùng phiên; ghi số vào HANDOFF trước khi commit.

---

## 6. Cách đo lại sau khi làm

```powershell
dotnet out/bseq_p12_n10.dll   # seq1024 mode 0/1, best-of-3 xen kẽ
TKV_GEMM_THREADS=14 dotnet out/bseq_p12_n10.dll
```

Tiêu chí đạt: `backward <= 9000 ms` @seq1024, `opt.step() <= 900 ms`, không regress seq8/seq256, loss EXACT.
