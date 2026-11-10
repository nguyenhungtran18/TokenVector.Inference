# SPEC FUSED SDPA BACKWARD — C kernel AVX2 cho attention bwd (production fallback + future bake-off)

**Ngày lập:** 2026-11-09. **Nguồn:** PLAN_BEAT_PYTORCH §6 + HANDOFF §0.115-§0.128 + profile `tv/tests/_glue.tkv` & `tv/tests/_glue_seq.tkv` (2026-11-09).
**Đối tượng:** native/SIMD kernel team (hoặc chính code Release/tools/simd_native/tkv_flash_attn.c tác giả).

---

## 0. TL;DR — vì sao có document này

Mục tiêu: **đạt parity rồi nhanh hơn PyTorch** (1.0× → ≤ 0.8×) trên full step GPT-2 124M, gate fp32 giữ nguyên (AGENTS s0.96 waiver cho attention path). Trạng thái hiện tại:

- **Attention backward** chiếm lớn nhất trong bwd; dùng `.tkv` production route qua ATen `at::scaled_dot_product_attention(is_causal=true)` = **~94 ms/layer** (đo `tv/tests/_glue.tkv` mode 0 seq1024); **native flash path** (tkv_flash_attn.c) = **~177-215 ms/layer** sau Phase 1 + Phase 2 — vẫn ~2× chậm hơn.
- **PyTorch** có một kernel flash-attention AVX2/AVX512 fused C++ được JIT-built cho từng CPU — đó là lý do gap.
- **Đã xác nhận**: 12.6M expf/layer @ seq1024; expf chiếm ~50-60% cấu trúc giá thành của backward; wrapper ATen chiếm ~0.43 ms dispatch + memcpy.
- **Phase 3 waiver** (s0.124/s0.126) đã relax bit-exact cho attention bwd — cộng dồn cho dự án này.

> **Hướng đã chứng minh:** Phase 1 vector transcendental (m256_exp_ps, exp_row_fast, exp_sum_row_fast) + Phase 2 (row_max_fast, ds_row_fast). Loss EXACT được giữ ở mode 0 và mode 1 bite-ident (10.764943 → 12.684581) sau cả 2 phase (no regressions).

**Nội dung file:**

1. Context + dữ liệu nền (đ o thực tế)
2. Yêu cầu chức năng + non-functional
3. Thiết kế C kernel (data layouts, function signatures, intrinsics, algorithm)
4. Chi tiết plan (Phase 2 → Phase 5 + acceptance criteria)
5. Rủi ro & mitigation
6. Gate & verification checklist

---

## 1. Context & dữ liệu nền

### 1.1 Hiện trạng bwd (seq1024, best-of-3, pool=1)

| Đoán | Mode 0 (f+b) | `att bwd` ước | `MLP bwd` ước | LN bwd | GELU bwd |
|---|---|---|---|---|---|
| Trạng thái hiện | **4063 ms** (s0.126) | ~94 ms/layer (ATen) = ~1130 | ATen fused ~50 | native ~0.6 | ATen ~2.8 |

### 1.2 Đo thật Makefile callable

```
Sequential per layer (seq1024, _glue.tkv):
- attention (fwd pass 12 layers)                     ~411-423 ms
- attention BACKWARD (gpt2_attention_bwd)           ~94-104 ms/layer (ATen) / ~177-215 ms (native)
- MLP backward                                        ~92-100 ms/layer
- LN fwd (obuf)                                     ~0.86 ms/call (ATen bridge)
- LN bwd (native)                                   ~0.6 ms/call
- gelu_bwd (ATen)                                   ~2.8 ms/call
```

### 1.3 Cái gì gây chậm

1. **12.6M expf/layer @seq1024** (online-softmax + recompute pass trong backward) — ~10-20 cycle, dominant transcendental.
2. **ATen dispatcher overhead** ~0.3-0.4 ms/call × #calls.
3. **dot64** đã AVX2 (tốt); `cblas_sgemm` route cho dW/dx (s0.124/s0.125) đã kích hoạt cross-layer mọi fc.
4. **Wrapper ATen cho LN fwd** vẫn trả về qua `TkvTorchLayerNormFwdNative` (P/Invoke) cho mỗi call.

---

## 2. Yêu cầu

### 2.1 Functional

Khi hoàn thiện: `LinearDxFused` và `LinearDwAtFused` có thể route BLAS; production có thể mở `TKV_FLASH_OFF` để nhảy ATen → native flash path.

**Function cần có (đã đạt):**
- `static inline __m256 m256_exp_ps(__m256 x)` - fast exp AVX2 polynomial, ULP ≤ 2 (Phase 1).
- `static inline void exp_row_fast(float* qk, float newmax, int kvb)` (Phase 1).
- `static inline float exp_sum_row_fast(float* qk, float newmax, int kvb)` (Phase 1).
- `static inline float row_max_fast(const float* v, int kvb)` - AVX2 max_ps + horizontal reduce (Phase 2).
- `static inline void ds_row_fast(const float* attn, const float* gA, float* ds, float dsum_i, int kvb)` (Phase 2).

**Function cần thêm (Phase 2):**
- `static inline void causal_mask_blend_row(float* v, int kvb, int row, int col)` - AVX2 mask blend blending `u > t` → weight = 0.0f * µ before the exp. **Chưa có — cần thực hiện Phase 2b.**
- `static inline void gA_row_fast(const float* drow, const float* v_base, ptrdiff_t stride3, int coff_v, float* gA, int kvb)` - precomputed context row INFO.
- `static inline void softmax_lse_row_fwd(const float* qk, float* lse_out, float* mx_out, float newmax, int kvb)` - AVX2 single-pass LSE accumulator (pass 2).

### 2.2 Non-functional

- **Loss EXACT** bị break trong waiver path — không được phép mất thêm giá trị bit-exact vì native path đã drift tự nhiên.
- **RAM**: không hơn pool hiện có (sequential-arena, reuse buffers qua caller).
- **Băng thông**: Ưu tiên AVX2 vPCLMULQDQ + memory-bound awareness; không spawn threads mới.
- **Compiler safety**: mọi thay đổi phải đi cùng rebuild `build.ps1` + `vong.exe` (selfhost IL_IDENTICAL) (s0.122 pattern).
- **Determinism**: AVX2 mode chỉ; không VADDPS run-time switch (RunTime CPU dispatch unsupported).

### 2.3 Ràng buộc

- C++ standard: C99 + intrinsics immintrin.h (AVX2 + FMA), no OpenMP. Không dùng OpenMP; dùng native pool riêng (`tkv_run_aux`).
- Subroutine `static inline` cho mọi helper; geometric `FL_QS=64` `FL_KS=64` các block-size.
- C-API exported qua `__declspec(dllexport)` để IL builtin gọi (`.tkv` → P/Invoke).

---

## 3. Thiết kế C kernel

### 3.1 File: `Release/tools/simd_native/tkv_flash_attn.c`

Cây hàm hiện tại:

```
flash_fwd_head()       // pass 1, writes ctx_out + lse
  - FL_QS=64, FL_KS=64 blocks, online softmax
  - dot64() AVX2, mul_row(), fma_row() AVX2
  - lse[] = log(sum(exp(qk - max))) via m256_exp_ps/exp_sum_row_fast (Phase 1)

flash_bwd_head()       // pass 2 + pass 3
  - pass 2: recompute out+lse from qkv (inline pass-1 again)
  - pass 3: per-block backward contributions
       - qk[0..kvb) * expf(qk*finv - lse) (attention recomputation) — vectorized exp
       - gA = dot64(dctx, V) (gMA vectorized)
       - ds = attn ⊙ (gA - dsum)           — vectorized (Phase 2 ds_row_fast)
       - dv += attn^T * dctx               — fused accumulation
       - dq += ds * K                      — fused
       - dk += ds^T * Q
```

### 3.2 Signature layout ("spec của C")

```c
// [opaque to .tkv; only visible to P/Invoke]
static inline float dot64(const float* a, const float* b) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 64; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    ... horizontal reduce ...
}

static inline __m256 m256_exp_ps(__m256 x) {          // fast expf, waiver-sULP ≤ 2
    const __m256 vexp_magic   = _mm256_set1_ps(12582912.0f);
    const __m256 vexp_invln2  = _mm256_set1_ps(1.4426950408889634f);
    const __m256 vexp_ln2     = _mm256_set1_ps(0.6931471805599453f);
    x = _mm256_min_ps(x, _mm256_set1_ps(88.7f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    __m256 t  = _mm256_mul_ps(x, vexp_invln2);
    __m256 n  = _mm256_sub_ps(_mm256_add_ps(t, vexp_magic), vexp_magic);
    __m256 f  = _mm256_sub_ps(t, n);
    f  = _mm256_mul_ps(f, vexp_ln2);
    __m256 r = _mm256_set1_ps(1.388888654e-04f);
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(8.333333e-03f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(4.1666665e-02f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.6666666e-01f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(5.0000000e-01f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.0000000e+00f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.0000000e+00f));
    __m256i ni = _mm256_cvttps_epi32(n);
    __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(r, _mm256_castsi256_ps(bits));
}

static inline void exp_sum_row_fast(float* qk, float newmax, int kvb) {   // vectorized exp+sum
    __m256 vmax, vsum;
    // ... (existing helper)
}

static inline float row_max_fast(const float* v, int kvb) {                // vectorized max
    __m256 rmax = _mm256_set1_ps(-INFINITY);
    for (int j = 0; j < kvb; j += 8)
        rmax = _mm256_max_ps(rmax, _mm256_loadu_ps(v + j));
    // ... horizontal reduce ...
}

static inline void ds_row_fast(const float* attn, const float* gA,
                                float* ds, float dsum_i, int kvb) {         // ds = attn*(gA-dsum)
    __m256 vdsum = _mm256_set1_ps(dsum_i);
    for (int j = 0; j < kvb; j += 8) {
        __m256 va = _mm256_loadu_ps(attn + j);
        __m256 vg = _mm256_loadu_ps(gA + j);
        _mm256_storeu_ps(ds + j, _mm256_mul_ps(va, _mm256_sub_ps(vg, vdsum)));
    }
}
```

### 3.3 Data layout

- `qkv`: [n, 3*h] fp32 row-major — Q = +0, K = +h, V = +2h (per token h floats, h=768)
- `dctx`: [n, h] fp32 — output row-major
- `ctx_out`/`dqo`: [n, hd] and [n, 3*hd] (hd = HeadDim = 64) row-major
- `lse[]`: [n] fp32
- `FL_QS = FL_KS = 64` (block size)

### 3.4 Error contract (rc)

- 0: OK
- -1: NULL pointer
- -2: bad arg (n<0, h<=0, hd != FL_HD)
- -5: buffer too short
- -13: shape not supported (n < 64 hoặc hd ≠ 64)
- -6: malloc failed
- Caller (.tkv) fallback khi rc≠0.

---

## 4. Kế hoạch chi tiết (plan)

### 4.1 Phase 1 (ĐÃ XONG - 2026-11-09)

**Task:** Vectorize transcendental functions in flash_bwd_head and flash_fwd_head.

**Done:**
- [x] `m256_exp_ps` - fast polynomial exp AVX2 (deg-6 Fq, ULP ≤ 2, waiver admitted)
- [x] `exp_row_fast` + `exp_sum_row_fast` - replace 2 scalar `bsum += e; qk[j] = e` loops (pass 1 + pass 2)
- [x] `zero row mask` deferred đến Phase 2b (explicitly documented)
- [x] Loss EXACT giữ ceil ở seq1024 / seq256 / seq512 + mode 1 bite-ident `10.764943 -> 12.684581`
- [x] Gate 9/10, trainer 15/15
- [x] HANDOFF §0.127 (release) + docs written

---

### 4.2 Phase 2a (ĐÃ XONG - 2026-11-09)

**Task:** Vectorize ds and row-max.

**Done:**
- [x] `row_max_fast` (AVX2 `_mm256_max_ps` + hsum) - replace scalar `for j: qk[...] > rmax`
- [x] `ds_row_fast` (AVX2 `_mm256_sub_ps` + `_mm256_mul_ps`) - replace scalar `ds[i] = attn*(gA-di)`
- [x] `ds_row_fast` wired in flash_bwd_head pass 2 (line ~388 → 415)
- [x] Loss EXACT
- [x] HANDOFF §0.128

---

### 4.3 Phase 2b (ĐÃ XONG - 2026-11-09)

**Task:** Vectorize causal mask + gA row + compute the ds row fully inline.

**Done:**
- [x] **FIX BUG NGHIEM TRONG**: `flash_fwd_head` gọi `row_max_fast`/`exp_sum_row_fast`
      TRƯỚC khi định nghĩa → MSVC C4013 `extern int` → giá trị trả về bị cắt thành int
      → **forward sai hoàn toàn** (probe: max_abs 0.98). Đã dời toàn bộ helper lên đầu file.
      Verify: `cl /W3` sạch C4013, fwd max_abs 5.88e-07.
- [x] `causal_mask_zero_row` / `causal_mask_neginf_row` — `_mm256_cmpgt_epi32` + `_mm256_blendv_ps`
- [x] `gemm_nt_block` — S = Q·Kᵀ và gA = dctx·Vᵀ qua K-transposed `BT[d][j]`: 8 output / 8 FMA,
      không horizontal-reduce mỗi phần tử (thay `dot64` per-element)
- [x] `acc_block` / `acc_block_t` — dq/dk/dv/PV tích lũy trong 8 thanh ghi ymm, flush 1 lần/block
- [x] **PACK** Q/K/V/dctx → buffer `[n][64]` contiguous (trước đó mỗi hàng 256 B cách nhau 9216 B)
- [x] `FlashAttentionBwdFusedLse` (export mới) — caller có ctx+lse từ fwd → **bỏ hẳn pass-1 recompute**
      (verify khớp tuyệt đối: dqo max_diff = 0.0, ctx max_diff = 0.0)
- [x] Benchmark probe_flash2 (1 head, n=1024, best-of-3)

**Actuals (1 head, n=1024):**

| đường | fwd | bwd |
|---|---|---|
| Phase 2a (87bbce6) | 12.0 ms (SAI) | 42.5 ms |
| Phase 2b recompute | 4.8 ms | 14.1 ms |
| Phase 2b + Lse | — | 9.0–10.4 ms |

Edge case sạch: n = 64/96/128/192/256/320/512/1024 (kể cả n lẻ).

**⚠️ PHÁT HIỆN: production KHÔNG đi flash.** `gpt2_attention_bwd` đi `torch_attention_bwd`
(ATen) trước vì `opt.g` (gflat) luôn có data; flash chỉ là nhánh fallback khi `gflat == None`.
Đo `_glue.tkv` attn_bwd NATIVE: flash ON **139 ms/layer** vs `TKV_FLASH_OFF=1` **219 ms/layer**
(-36%), nhưng ATen-compat path vẫn **101 ms/layer**. Flash chưa thắng ATen ở 1 head.

**Ràng buộc:**
- Không có mem alloc ngoài caller cung cấp (pool)
- Bit drift acceptable, mode 1 bite-ident sẽ lệch (expected — waiver)
- Loss EXACT giữ nguyên ở mode 0 (forward không đổi)

---

### 4.4 Phase 3 (TODO) — Switch production

**Task:** Wire `attention_bwd` IL builtin → native `flash_bwd_head` thay vì ATen autograd.

**Trạng thái thực tế (2026-11-09):** dispatch `AttentionBwdFused` trong `tkv_simd_native.c`
dòng 2184 ĐÃ có sẵn từ §0.97 — flash chạy trước, `rc != 0` → đường cũ, `TKV_FLASH_OFF=1` tắt.
Nhưng `.tkv` production KHÔNG đi vào nhánh đó: `gpt2_attention_bwd` gọi `torch_attention_bwd`
(ATen) trước, chỉ rơi xuống native khi `gflat == None`. Vì `opt.g` luôn có data trong bench,
production đi ATen 100%.

**Việc còn lại của Phase 3 (khác plan gốc):**
1. Bỏ điều kiện `gflat != None` cho attention bwd, hoặc wire `FlashAttentionBwdFusedLse`
   vào đường có lse (fwd đã tính sẵn ctx+lse).
2. **Điều kiện tiên quyết:** flash phải ≤ 80 ms/layer mới có lợi so với ATen 101 ms/layer.
   Hiện 139 ms/layer → chưa đủ. Xem §4.7.
3. `/gate run_gates_n10.ps1` — verify PASS 9/10 (hoặc re-baseline nếu drift)
4. `trainer_tests.tkv` verify 15/15
5. mode 1 seq512: re-record loss EXACT constant (có thể drift do ATen removed)
6. **QUAN TRỌNG**: keep ATen path như fallback: `if (rc == 0) return; rc != 0 → torch_attention_bwd`

**Exit criteria:**
- attention bwd ≤ 50 ms/layer @seq1024 (sau A/B) — **chưa đạt, hiện 139**
- Loss EXACT (drift acceptable theo waiver s0.125)
- Gate giữ 9/10 + trainer 15/15 green

---

### 4.5 Phase 4 (TODO) — Finalize + ship

**Task:** Re-baseline gate + consolidating + release.

- [ ] Re-record bench baselines 2026-11-09
- [ ] mode 1 bite-ident new constant documented
- [ ] `HANDOFF.md` updated với Phase 3/4 kết quả
- [ ] `PLAN_BEAT_PYTORCH.md` §6 updated với actuals
- [ ] Git release tag 1.0.13+ (optional)

---

### 4.6 Phase 5 (Tùy chọn - sau khi phase 4 xanh)

**Task:** Attention fwd + LN fwd AVX2 replacements.

- [ ] `FlashAttentionFwdFused` already vectorized (Phase 1 w/ transcendentals)
- [ ] LN fwd could become AVX2 `LayernormFwdAvx2` — replace ATen `layernorm_fwd_fused`
- [ ] ~20-30 ms/step gain expected

---

### 4.7 Phân tích nút thắt & hướng thắng PyTorch (2026-11-09, sau Phase 2b)

**Số đo hiện tại (seq1024, cùng máy Xeon E5-2695 v3, AVX2 only):**

| đường | ms/layer | ×12 layer |
|---|---|---|
| ATen SDPA bwd (production hiện tại) | 101 | 1212 |
| Native flash bwd, Phase 2b, flash ON | 139 | 1668 |
| Native flash bwd, `TKV_FLASH_OFF=1` | 219 | 2628 |
| PyTorch full step (bench_torch.py, T=14) | — | **4193 total** |
| Ta hiện tại (mode0 fwd+bwd) | — | 4312 total |

→ Còn chậm hơn PyTorch **1.24×**. Attention bwd không phải chỗ duy nhất, nhưng là chỗ lớn nhất.

**Vì sao flash 1-head vẫn chậm hơn ATen:** ATen `scaled_dot_product_attention` CPU dùng
kernel C++ JIT-built với blocking đa cấp và **vector hóa trên nhiều head cùng lúc**
(head-dim 64 × nhiều head trong một lần duyệt), trong khi kernel này xử lý **1 head/lần gọi**
và transpose K/V lại cho **mỗi** block `m0`.

**Ba hướng có tiềm năng thật, xếp theo tỷ lệ lợi/công:**

1. **Transpose K/V một lần cho cả head** (không phải mỗi block `m0`). Hiện
   `transpose_block64` chạy lại trong vòng `m0` — với n=1024 có 16 block `m0` × 16 block `n0`.
   Transpose toàn bộ K/V một lần ra buffer `[n][64]` transposed → bỏ ~½ chi phí transpose.
   Ước tính: −15…25 ms/layer. **Nên làm trước, rẻ nhất.**

   > **SAI — đã đính chính ở §4.8.** `transpose_block64` nằm ở mức vòng `n0`
   > (16 lần cho n=1024), KHÔNG phải mỗi block `m0`. Không có gì để hoist.

2. **Xử lý 2 head cùng lúc** (register blocking theo head). Với hd=64, hai head = 128 float
   mỗi hàng; 8 thanh ghi ymm đang dùng cho 1 head → cần chia lại. Lợi: chia sẻ hoàn toàn
   việc đọc K/V (cùng token, khác head) → giảm một nửa lưu lượng đọc K/V.
   Ước tính: −30…40 ms/layer. Công: viết lại `acc_block`/`gemm_nt_block` cho cặp head.

3. **AVX-512**: CPU này (Haswell) **không có** AVX-512. Bỏ qua trên máy hiện tại, nhưng
   nên có nhánh `#ifdef __AVX512F__` cho máy khác — không giúp được mục tiêu trên máy này.

**Điều cần nói thẳng:** ngay cả khi (1)+(2) thành công và đưa flash xuống ~80 ms/layer,
tổng step chỉ giảm ~250 ms (4312 → ~4060), tức **~1.03× so với PyTorch**. Để đạt ≤0.8×
(≤3350 ms) cần cắt thêm ~700 ms nữa ở chỗ khác — MLP bwd (~92-100 ms/layer) và LN fwd
(Phase 5) là các ứng viên tiếp theo, không phải attention.

**Khuyến nghị:** làm (1) trước vì rẻ và chắc chắn có lợi; đo lại; rồi quyết định (2) dựa
trên số thật. Đừng mở production sang flash cho tới khi flash ≤ 80 ms/layer.

---

## 5. Rủi ro & mitigation

| # | Rủi ro | Mức | Mitigation |
|---|---|---|---|
| 1 | Loss EXACT drift | Cao | waiver s0.125 đã trao; re-record baseline; gates 15-tolerance |
| 2 | Segfault AVX2 misalign | TB | mọi load `loadu_ps` (unaligned OK); `posix_memalign` cho buffers caller |
| 3 | Compiler warning wave | TB | keep /fp:fast; AVX2 unchanged |
| 4 | Bit-exactness loss | Cao | chấp nhận; waiver đã xác nhận |
| 5 | ISA availability | Thấp | CPU e5-2695 v3 = Haswell = AVX2+FMA |
| 6 | thread safety (pool AM) | Thấp | native pool đã có; flash gọi single-threaded |
| 7 | Buffer overrun | Trung bình | mọi call `n >= 64` guard; buffer size check |

---

## 6. Gate & verification checklist

```text
[ ] dot64 correctness: `_simd_matmul_tests` PASS (existing)
[ ] m256_exp_ps: within 2 ULP of expf for x in [-87, 88]
[ ] flash_fwd_head: logits/ctx match bf16-vs-fp32 within tolerance
[ ] flash_bwd_head: loss EXACT mode 0 seq8/256/1024 (=10.572291 / 10.756927 / 10.760399)
[ ] mode 1 seq512 full step: loss bite-identical baseline (=10.764943 -> 12.684581) OR re-record
[ ] gates 9/10 PASS
[ ] trainer_tests 15/15 PASS
[ ] FLASH_OFF=1 fallback path works (ATen path)
[ ] No `_seq_cfg.txt` side effects / no committed build artifacts
[ ] HANDOFF updated for each phase
[ ] PLAN_BEAT_PYTORCH §6 updated with actuals
```

---

## 7. Terminology + references

- **FL_HD = 64** (HeadDim) - không đổi
- **FL_QS = FL_KS = 64** (query/key block, stride 8)
- **AVX2+FMA3**, không AVX-512
- **PyTorch** = torch 2.14.0+cpu, MKL 2026.1/oneDNN — dùng làm reference A/B đối chiếu
- **waiver bit-exact** = s0.125/§0.126 - user chấp nhận drift do waiver
- **dogfood** = selfhost: `dist/tkvc-tv.exe` rebuild tự thân qua `tools/vong.exe` (IL_IDENTICAL)

---

## 8. Cross-references

- HANDOFF.md §0.115-§0.128 (historical context)
- PLAN_BEAT_PYTORCH.md §0-6 (overall plan)
- PLAN_BEAT_PYTORCH.md §6 (fused SDPA project specific plan)
- Release/tools/simd_native/tkv_flash_attn.c (current C source)
- tv/tests/_glue.tkv + _glue_seq.tkv (probes)

---

### 4.8 ĐÍNH CHÍNH §4.7 sau khi đo thật (2026-11-09, Phase 2c)

§4.7 ở trên đặt cược vào 3 hướng micro-optimization. Đo thật cho thấy **cả 3 đều
không phải nút thắt**. Giữ nguyên §4.7 làm lịch sử suy luận, nhưng đọc §4.8 mới.

**Bằng chứng đo được (Xeon E5-2695 v3, 28 logical / 14 physical core):**

| phép đo | kết quả |
|---|---|
| Peak FMA của máy (16 chain độc lập) | 2.58 G-FMA-instr/s = 41.3 GFLOP/s |
| Ngân sách FMA của bwd_lse @n=1024, 1 head | 136 block-pair × 5 GEMM × 64×64×8 = 22.28 M FMA-instr |
| Ở peak thì phải mất | 8.64 ms |
| **Đo thật** | **10.05 ms → 86% của peak** |

Kernel đã ở 86% trần phần cứng. Không còn chỗ cho micro-opt.

**Đối chiếu từng hướng của §4.7:**

1. **Transpose K/V một lần** — *đã làm rồi*, từ Phase 2b. `transpose_block64`
   nằm NGOÀI vòng `m0` (mỗi `n0` transpose 1 lần, không phải mỗi block `m0`).
   Ước tính "−15…25 ms/layer" trong §4.7 không còn áp dụng được.
2. **2 head cùng lúc** — trần lý thuyết chỉ ~1.15× vì đã ở 86% peak.
   Không đáng một đợt viết lại `acc_block`/`gemm_nt_block`.
3. **AVX-512** — CPU này (Haswell) không có. Bỏ.

**Nút thắt thật: 12 head chạy TUẦN TỰ trên 1 core.**

| đường | fwd/layer | bwd/layer |
|---|---|---|
| serial 1 core | 57.9 ms | 119.3 ms |
| ATen SDPA (production hiện tại) | — | 101 ms |

ATen không thắng vì kernel nó giỏi hơn. Nó thắng vì **chia head ra nhiều thread**.
Ta có 28 logical core và chỉ dùng 1. Đó là toàn bộ khoảng cách.

**Cách đã sửa (Phase 2c):** thêm `FlashAttentionBwdFusedLseMulti` /
`FlashAttentionFwdLseMulti` — work-queue atomic, mỗi head ghi vào slice riêng,
không lock, không data race.

| threads | fwd/layer | bwd/layer | bwd speedup |
|---|---|---|---|
| 1 (serial) | 57.87 | 119.33 | 1.00× |
| 4 | 18.60 | 37.07 | 3.22× |
| 7 | 13.09 | 24.53 | 4.86× |
| **12** | **8.87** | **18.38** | **6.49×** |
| 14 | 8.35 | 19.66 | 6.07× |

**bwd 18.4 ms/layer so với ATen 101 ms/layer = nhanh hơn 5.5×.**
12 layer: 220 ms thay vì 1212 ms → tiết kiệm ~990 ms/step.

12 thread là điểm tối ưu (14 chậm lại do SMT tranh chấp). Đặt `nthreads` = số head,
chặn trên = số core vật lý.

**Correctness:** bit-identical với đường tuần tự, `max_abs = 0.000e+00` cho cả
`dqo` và `ctx` ở n = 64/96/128/192/256/320/512/1024. Không thêm drift nào ngoài
waiver s0.125 — vì mỗi head ghi vùng nhớ riêng và thứ tự tính trong một head
không đổi.

**Điều chỉnh mục tiêu:** với bwd ở 18.4 ms/layer, "≤ 80 ms/layer" của §4.4 không
còn là ngưỡng đúng — đã vượt xa. Điều kiện tiên quyết của Phase 3 giờ là
**wire vào production + chạy gate**, không phải tối ưu thêm.

---

### 4.9 Phase 2c-2 ĐO THẬT (2026-11-10) — số thật khác §4.8

§4.8 dự đoán bwd 18.4 ms/layer (5.5x so với ATen 101). **Đo end-to-end trong production
thì không được như vậy.** Ghi lại số thật để không ai đọc §4.8 rồi tưởng đã xong.

**Bẫy đã dính: hai bản module trùng tên.** Compiler resolve `import tv.training.gpt2_ref124`
theo thư mục của file đang dịch. `tv/tests/simd_bench_124m_seq.tkv` -> nó nạp
`tv/tests/tv.training.gpt2_ref124.tkv`, KHÔNG phải `tv/training/...`. Sửa bản
`training` rồi build bench = build bản cũ, im lặng, không báo lỗi. Đã đồng bộ hai bản.

**Số đo end-to-end (Xeon E5-2695 v3, 28 logical, AVX2, seq1024, glue probe #8):**

| đường | ms/layer | ×12 layer |
|---|---|---|
| ATen SDPA bwd (production, `gflat` có data) | 101.7 | 1220 |
| Native cũ (12 thread .tkv + 12x merge) | 140.7 | 1688 |
| Native mới (`attention_bwd_merge_multi`, work-queue C) | **135.4** | **1625** |
| Native, `TKV_FLASH_OFF=1` (scalar) | 220.0 | 2640 |

**Kết luận: 140.7 -> 135.4 ms/layer, tức 1.04x. Không phải 5.5x.**

Vì sao lệch với §4.8: con số 18.4 ms/layer ở §4.8 là **chỉ phần flash kernel**, đo
trên probe gọi thẳng `FlashAttentionBwdFusedLseMulti`. Trong production, `gpt2_attention_bwd`
còn làm `gpt2_linear` (qkv), `smm_dot` (dctx), `gpt2_linear_bwd`, `dw_proj` GEMM,
`bias_grad`, và merge — cộng lại ~117 ms/layer không nằm trong kernel. Kernel nhanh
6.5x chỉ rút được ~5 ms trên tổng, vì phần còn lại đã chiếm 87%.

**Điều này khớp với cảnh báo cuối §4.7:** attention không phải nút thắt duy nhất, và
tối ưu kernel đơn lẻ không đưa được step xuống ≤0.8x.

**Trạng thái Phase 3:** đã wire xong, gate 9/10 (chỉ `gap_tests` pre-existing),
trainer 15/15, self-host `IL_IDENTICAL`, loss EXACT cả 3 mốc
(`10.572291` / `10.756927` / `10.760399`). Đường mới **chạy đúng và bit-identical**,
chỉ là lợi ích nhỏ hơn dự đoán.

**Việc còn lại để thắng PyTorch:** cắt phần 117 ms/layer ngoài kernel (qkv linear,
dctx smm_dot, linear_bwd, dw_proj) — đó mới là chỗ có tiền. Không phải attention kernel.
---

### 4.10 PROFILE LAI TU DAU (2026-11-10) — so that, va mot phat hien lam doi uu tien

Truoc khi sua bat cu thu gi, do lai toan bo bang probe moi
(`tv/tests/_attn_split.tkv`, `_block_split.tkv`, `_bwd_outer.tkv`).
Ket qua **khong khop** voi cac so dang ghi trong HANDOFF §0.130 va §4.9.

#### 4.10.1 So do lai (seq1024, Xeon E5-2695 v3, .NET 10, best-of-N)

| duong | ms/layer | ghi chu |
|---|---|---|
| `gpt2_attention_bwd` ATen, **co abuf** | **65.0** | production that (block_bwd truyen abuf) |
| `gpt2_attention_bwd` ATen, khong abuf | 99.0 | so cu trong HANDOFF — thieu pool |
| `gpt2_attention_bwd` NATIVE, **co abuf** | **119.1** | |
| `gpt2_attention_bwd` NATIVE, khong abuf | 138.8 | so cu trong HANDOFF — thieu pool |
| `gpt2_block_bwd` (gflat -> ATen attn) | **187.0** | production that |
| `gpt2_block_bwd` (gflat=None -> native attn) | 234.0 | |

**Ket luan 1: ca hai con so trong HANDOFF deu bi thoi phong vi probe cu khong
truyen `abuf`.** Moi call `gpt2_attention_bwd` khong co pool cap phat 9.4M float
(qkv 2.36M + dqkv 2.36M + ctxT/dctxT 0.79M x2). Production LUON co pool. Do lai
dung: ATen 65, native 119 — khong phai 101.7 va 135.4.

#### 4.10.2 Phan ra `gpt2_attention_bwd` (co abuf, ms/layer)

| manh | ms |
|---|---|
| qkv linear (onednn_linear_bias) | 7.8 |
| dctx `smm_dot` | 7.8–14.1 |
| **flash kernel multi-head (12 head + merge)** | **34.4–35.9** |
| dw_proj (chuoi that: `linear_dw_fused` rc=-8 -> `linear_dw_at_fused` rc=0) | 1.5–3.1 |
| db_proj `bias_grad` | 0.0–1.5 |
| `gpt2_linear_bwd` (dx + dw_qkv + db_qkv) | 25.0–26.5 |
| **tong manh** | **84.3** |
| do duoc (full native) | 119.1 |
| chenh (glue + `_ab_pool` + `t_zeros([h,h])`, `t_zeros([h])`) | ~35 |

**Phat hien phu:** `linear_dw_fused` tra **rc = -8** voi shape that cua dw_proj
(k=1024 > gioi han 64 cua kernel). Nghia la nhanh native do **chua bao gio chay**
trong production; moi buoc roi xuong `linear_dw_at_fused`. Khong phai loi, nhung
dang ton 1 lan goi ham vo ich moi layer.

#### 4.10.3 Phan ra `gpt2_backward` (best-of-3)

| manh | ms |
|---|---|
| `gpt2_backward` FULL | **2766** |
| 12 x `gpt2_block_bwd` (ATen attn) | 2244 |
| **phan NGOAI 12 layer** (lm-head dwte 38.6M + embed bwd + lnf) | **522** |
| 12 x `gpt2_block_bwd` (native attn) | 2808 |

**Ket luan 2: attention khong con la nut that.** Trong 2766 ms bwd:
- 522 ms (19%) la lm-head + embed, ngoai 12 layer
- 2244 ms (81%) la 12 layer, trong do attention chi 65/187 = **35%** cua 1 layer
- 1 layer = 187 ms: attention 65 + mlp_bwd 82 + LN bwd 1.6 + simd_add 0.6 + ~37 glue

#### 4.10.4 Do lai PyTorch CUNG PHIEN (bat buoc theo PLAN §5.5)

| | ms |
|---|---|
| PyTorch fwd+bwd seq1024 | **3324.6** (median, n=2) |
| PyTorch full step seq1024 | **4041.4** |
| Ta fwd+bwd (mode 0) | ~4300 |
| Ta full step (mode 2, BPHASE) | **5312** (fwd 1000 / loss 125 / bwd 3156 / opt 891 / copyback 140) |

Gap that: **1.31x** full step (5312 vs 4041). Khong phai 1.26x nhu §0.130.

#### 4.10.5 Doi uu tien — va ly do

§4.9 ket luan "cat 117 ms/layer ngoai kernel". **Sai don vi.** 117 ms la chenh
giua hai duong attention, khong phai khoi luong con lai. Khoi luong that:

1. **mlp_bwd 82 ms/layer = 984 ms/step (19% bwd)** — lon hon attention (65 ms/layer).
   Da co ATen `onednn_linear_bwd_off` cho ca 2 tang, nhung van 82 ms. Chua ai profile
   ben trong no.
2. **lm-head + embed 522 ms/step (19% bwd)** — ngoai 12 layer, chua ai dung toi.
   `dwte` la 38.6M phan tu, GEMM lon nhat trong bwd.
3. **opt.step 891 ms/step (17% full step)** — §5.1 P1a da thu NT stores (bi loai),
   prefetch giu lai. Chua thu gi khac.
4. **copyback 140 ms/step** — vong managed copy `opt.p -> ps[i].data` moi buoc.
5. **attention 65 ms/layer = 780 ms/step** — chi 15% full step. Native dang CHAM HON
   ATen (119 vs 65), nen wire native la **regression**, khong phai toi uu.

**Khuyen nghi:** dung toi uu attention them. Thu tu dung: (a) profile mlp_bwd
ben trong, (b) lm-head/embed 522 ms, (c) opt.step 891 ms. Ba cho nay cong lai
~2400 ms — du de dong gap 1270 ms.

**Canh bao ve native flash:** §4.8/§4.9 trinh bay native la "5.5x nhanh hon ATen"
roi "1.04x". Do lai co pool cho thay native **cham hon ATen 1.83x** (119 vs 65).
Duong native trong production chi la fallback khi `gflat == None`, va dieu do la
dung — dung bat no lam mac dinh.
