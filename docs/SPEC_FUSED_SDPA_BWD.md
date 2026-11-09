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

### 4.3 Phase 2b (TODO – next)

**Task:** Vectorize causal mask + gA row + compute the ds row fully inline.

**Plan:**
- [ ] `causal_mask_row` - blend 0/keep based on row/col index, dùng `_mm256_greater_epi32` mask + `_mm256_blendv_ps`
- [ ] `gA_row_fast` - hlookup + load/store qk/dv in 8-wide chunks with `stride3 = 3*h`
- [ ] `ds` fully fused with gA and attn (remove intermediate `ds` alloc 7MB per row)
- [ ] `lse` recompute pass 1 & pass 2 fused cùng nhau khi lse đã cho trước (skip)
- [ ] Benchmark seq1024 mode 0 và mode 1: target attention bwd bwd ~50 ms/layer (vs ~94 ms ATen) **sau Phase 3**
- [ ] Loss EXACT (drift acceptable theo waiver s0.125)

**Ràng buộc:**
- Không có mem alloc ngoài caller cung cấp (pool)
- Bit drift acceptable, mode 1 bite-ident sẽ lệch (expected — waiver)
- Loss EXACT giữ nguyên ở mode 0 (forward không đổi)

---

### 4.4 Phase 3 (TODO) — Switch production

**Task:** Wire `attention_bwd` IL builtin → native `flash_bwd_head` thay vì ATen autograd.

**Sub-phases:**
1. `attention_bwd_fused` IL builtin emit lệnh gọi `FlashAttentionBwdFused` (đã có sẵn)
2. `.tkv` production line update: remove dispatcher check `if gflat != None` → chỉ native
3. `/gate run_gates_n10.ps1` — verify PASS 9/10 (hoặc re-baseline nếu drift)
4. `trainer_tests.tkv` verify 15/15
5. mode 1 seq512: re-record loss EXACT constant (có thể drift do ATen removed)
6. **QUAN TRỌNG**: keep ATen path như fallback: `if (rc == 0) return; rc != 0 → torch_attention_bwd` (soft-fail graceful)

**Exit criteria:**
- attention bwd ≤ 50 ms/layer @seq1024 (sau A/B)
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
