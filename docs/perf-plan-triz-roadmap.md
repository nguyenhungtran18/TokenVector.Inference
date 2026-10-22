# TRIZ OPTIMIZATION ROADMAP — từ "thua PyTorch" sang "khác được"

**Ngày:** 2026-10-20 · **Baseline:** P5.3 consolidated (commit `f85ea3d`) — tkv `simd=1` (P5.1+P5.2+P5.3) vs PyTorch 2.14.0+cpu (14 threads)
**Máy:** Xeon E5-2695 v3 14C · **Cùng ngày, cùng máy, cùng token/config** cho mọi phép đo.
**Mục tiêu (CẬP NHẬT 2026-10-20, chủ khoa):** seq1024 full step phải **BẰNG rồi NHANH HƠN PyTorch** (≤3.542 s = 1.0×, rồi ≤0.8×) trên **cả 3 seq**, vẫn giữ 84/84 parity assertions. Mốc dọc đường (đã đạt): 154.234 s (43.5×) → 141.219 (P5.4) → **118.641 s (33.5×, P5.5)**. Ladder + phân tích sàn đơn/đa luồng: **§12** (mục ~20× bên dưới bị thay thế).

Chi tiết số đo gốc: `docs/perf-plan-simd-vs-pytorch.md` (mục P5.2 + P5.3). Bàn giao: `HANDOFF.md` mục 0.13 + mục 7.

---

## 1. BASELINE ĐO ĐẠC (2026-10-20)

tkv `simd_bench_124m_seq` (warmup 1 + 1 bước, `out/bseq.exe` build lại từ nguồn hiện tại) vs `tools/bench_torch.py d <seq>` (warmup 1 + median n=2, chạy **2 lần/seq**, spread ≤3%). Cùng token `ids[i]=(i*37+11)%50257`, `pos[i]=i`; cùng shapes (vocab 50257 / 12 layer / 768 / 12 head); cùng AdamW lr 0.0025, wd 0.

| seq | tkv fwd | tkv bwd | tkv fwd+bwd | torch fwd+bwd | **tỉ lệ fwd+bwd** | tkv full step | torch full step | **tỉ lệ full step** |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.922 s | 13.890 s | **15.812 s** | 0.302 s | **52.4×** | **22.219 s** | 0.934 s | **23.8×** |
| 256 | 5.719 s | 29.453 s | **35.172 s** | 0.782 s | **45.0×** | **41.218 s** | 1.404 s | **29.4×** |
| 1024 | 25.281 s | 125.344 s | **150.625 s** | 2.987 s | **50.4×** | **154.234 s** | 3.542 s | **43.5×** |

**Đọc số:**

1. **Backward = 83-88%** fwd+bwd của tkv → sau P5.3, **attention không còn là nút thắt**; nút thắt là backward **không**-attention (LN / gelu / embed-LM-head).
2. **Plumbing/optimizer = full step − fwd+bwd** = **6.4 / 6.0 / 3.6 s** (tkv) vs **0.63 / 0.62 / 0.55 s** (torch AdamW 124M) → tkv nặng hơn **~10×**, gần như không phụ thuộc seq; ở seq8 nó chiếm **29%** cả bước.
3. Tỉ lệ fwd+bwd gần phẳng **45-52×** theo seq; tỉ lệ full step tăng dần 23.8 → 43.5× vì optimizer cố định của torch che phần compute ở seq nhỏ.
4. **seq256: 109.4 s (P5.1-only) → 41.218 s = 2.65×** nhờ P5.2+P5.3; tỉ lệ vs torch 76.7× → 29.4× → hiệu năng P5.x **không chỉ có ở seq1024**.
5. Loss `simd=1` khớp **từng chữ số** (`10.760399 → 13.442912`) ⇒ cùng thuật toán, chỉ khác FMA/thứ tự tích lũy.

**ĐÃ ĐO (2026-10-20, probe `tv/tests/simd_probe_aq.tkv`, seq 8/256/1024, self-check SUM vs TOTAL ≤3%)** — backward 125.3 s @seq1024 tách ra:

| Pha (gọi thẳng library, simd=1) | seq8 | seq256 | seq1024 | % @1024 |
|---|---|---|---|---|
| `attention_bwd` ×12 | 2.20 s | 6.06 s | **48.86 s** | **39%** |
| fwd recompute (`gpt2_forward` trong bwd) — **P5.5 ĐÃ BỎ** (xem §11) | 1.03 s | 3.75 s | ~~19.11 s~~ | ~~15%~~ → 0 |
| `mlp_bwd` ×12 (gelu fwd+bwd = 3.3 s) | 4.70 s | 7.58 s | 16.34 s | 13% |
| attn-fwd tính lại trong `block_bwd` | ~0.3 s | 1.80 s | ~15.4 s | 12% |
| ghi flatten `gflat` (bench − probe) | ~2.6 s | ~4.8 s | ~12.2 s | 10% |
| lm-head bwd (softmax grad + 3 GEMM) | 2.20 s | 4.39 s | 10.84 s | 9% |
| **`layernorm_bwd` ×25** | ~0 | **0.17 s** | **0.73 s** | **0.6%** |
| `embed_bwd` | 0.80 s | 0.80 s | 0.98 s | 0.8% |

**Đọc số — sửa tiền đề của chính roadmap này:**

1. "Sau P5.3 attention không còn là nút thắt" → **SAI**: attention (bwd 48.9 s + fwd recompute 15.4 s) vẫn **51%** backward.
2. "Nút thắt là LN / gelu / embed-LM-head" → LN **0.6%**, embed **0.8%**, gelu **2.6%** — gộp <5%; item thật trong nhóm này là **lm-head 9%** + **flatten 10%**.
3. Mỗi full step forward chạy **2 lần** (một lần cho loss + một lần trong `gpt2_backward`) → ~19 s "tính lại" chưa attack (ứng viên kế tiếp).

**Phân rã từng phép trong `attention_bwd` (probe `tv/tests/simd_probe_ar.tkv`, 1 layer @seq1024, lib 4265 ms/layer, self-check +1.1%):**

| Vùng | ms/layer | % |
|---|---|---|
| 6 GEMM/head (score/ctx/dp/dq/dk/dv) | 2469 | 58% |
| **2 transpose n²/head (gDts+gPts)** | **672** | **16%** |
| qkv linear_bwd | 454 | 11% |
| softmax + gDs scalar | 375 | 9% |
| qkv fwd + extract + dctx + proj + writeback | 345 | 8% |

---

## 2. TRIZ PRINCIPLE → HƯỚNG SỬA

| Principle | Hướng đi trong tkv | Vì sao đúng (theo số đo) |
|---|---|---|
| **1 — Segmentation** | Tách backward thành stage riêng: `ln_bwd` / `gelu_bwd` / `proj_bwd` / `attn_bwd` (đã GEMM) | Mỗi stage tối ưu độc lập, không ảnh hưởng stage khác; backward là 83-88% chi phí |
| **15 — Asymmetry** | Khác phương pháp theo stage: attention = GEMM (xong); LN bwd = FMA tiled; gelu bwd = erf path trong numerics contract; optimizer = fused | Không thể dùng một phương pháp cho mọi stage — GEMM cho attention đã ăn 3.75×, phần còn lại còn scalar |
| **25 — Merging** | Lazy materialize (copy `ps → opt.p` **1 lần** trước `opt.step()`, bỏ copy sau mỗi backward) + fused AdamW (gộp momentum + WD vào 1 vòng lặp) | Plumbing đang ~10× torch dù không phụ thuộc seq ⇒ merge là cách rẻ nhất để cắt |
| **40 — Nature of invention** | Policy adaptiv theo stage: FMA khi có hardware (fallback scalar), FP16 intermediate + rescale FP32 trước softmax khi còn giữ được loss tolerance | Đây là "thận giữa" giữa tốc độ và precision: chọn path nhanh nhất **vẫn trong** numerical contract |
| **33 — Numerical analysis** | Mọi tối ưu phải nằm trong bound: logits ≤ 2.4e-7, 124M backward grad ≤ 8.3e-7, `gpt2_train_tests` loss bit-exact | Đã chứng minh 84/84 — đây là hợp đồng, không phải đề xuất |
| **26 — Nesting doll** | Tuần tự: Phase 1 (backward) → Phase 2 (plumbing) → Phase 3 (adaptiv); mỗi phase đo lại cả 3 seq trước khi đi tiếp | P0→P5 đã chứng minh mô hình này: mỗi lần đo lại tìm ra nút thắt kế tiếp |

---

## 3. PHASE 1 — BACKWARD KHÔNG-ATTENTION (6 tuần)

**Mục tiêu:** seq1024 fwd+bwd 150.625 s → **~70 s**; full step → **~85 s**; tỉ lệ **43.5× → ~24×**.

> **CẬP NHẬT 2026-10-20 (theo số đo, không theo giả thuyết):** thứ tự 3.1 → 3.2 → 3.3 bị số đo lật ngược — `layernorm_bwd` chỉ **0.6%** backward (0.73 s / 125 s @seq1024, probe AQ, xem §1) → **hạ ưu tiên task 3.1**; attack attention bwd trước (51%). Đã làm **P5.4** (§10): −8.8% full step seq1024, 84/84 gate.

### 3.1 `layernorm_bwd` FMA tiled (2 tuần)
**File:** `tv/training/layernorm_bwd.tkv` (mới, theo pattern `tensor.tkv`)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Tách `layernorm_bwd` thành 3 pha: `γ²`, `∂γ`, `∂β` bằng **AVX2/FMA tiled** | `∂h = γ·(∂γ·(x−μ)/σ + ∂β)`, `∂γ = Σδᵢ·xᵢ`, `∂β = Σδᵢ` | **≤ 25 ms/block** ở seq1024 (torch ~1.2 ms/block ở seq8) |
| **In-place** thay vì materialize intermediates | Ghi đè `hx` (pattern `simd_gemm`) | **Memory traffic ↓ 40%** |
| Probe `simd_bench_124m_profile` trước/sau | Đo từng pha riêng | **Latency ↓ ≥ 40%** |

**Deliverable:** `layernorm_bwd_simd(hx, grad, gamma, beta, simd)` — `simd` optional, scalar fallback giữ parity.

### 3.2 `gelu_bwd` erf path (2 tuần)
**File:** `tv/training/gelu_bwd.tkv` (mới, theo pattern `autograd.tkv`)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Triển khai `erf'(x) = 2/√π·e^(−x²)` bằng FMA (không gọi `math.erf` mỗi phần tử) | `t = x²`, `erf'(x) ≈ 1.1283791670955126·e^(−t)` | **Latency ↓ 30-40%** |
| Giữ trong numerics contract | So sánh `torch.nn.functional.gelu(approximate='erf')` | **Maxdiff ≤ 1.5e-7** mỗi bước |
| `approximate='tanh'` chỉ làm đường debug | Chỉ dùng khi được yêu cầu, không phải default | Default = erf |

**Deliverable:** `gelu_bwd_simd(dx, x, simd)`.

### 3.3 Fusion `proj_bwd` + `dw_proj` (1 tuần)
**File:** `tv/training/proj_bwd.tkv` (mới, theo pattern `gpt2_ref.tkv`)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Gộp `dv = Pᵀ·dctx` và `dw = Σctxᵀ·dctx` vào **1 vòng lặp** | Dùng `smm_fwd` có sẵn + transpose `Pᵀ` | **Memory write ↓ 50%** |
| Đo phase proj bwd | `simd_bench_124m_profile` | **Latency ↓ 15-20%** (seq1024 ~19 s → ~16 s) |

**Deliverable:** `proj_bwd_simd(P, dctx, simd)`.

---

## 4. PHASE 2 — PLUMBING / OPTIMIZER (5 tuần)

**Mục tiêu:** plumbing 6.4/6.0/3.6 s → **~2/1.9/1.1 s**; full step → **~80 s**; tỉ lệ **~23×**.

### 4.1 Lazy materialize (2 tuần)
**File:** `tv/training/optimizer.tkv` (mới, theo pattern `trainer.tkv` AdamW)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Bỏ copy `opt.p[off+j] = d[j]` **sau mỗi backward** | Chỉ copy **1 lần** ngay trước `opt.step()` | **Overhead ↓ 60-70%** (6.4 s → ~2 s ở seq1024) |
| `opt.zero_grad(set_to_none=True)` pattern | Giống torch: không giữ grad cũ | **Memory write ↓ 30%** |
| Probe 10 bước train có/không lazy | Chạy `gpt2_train_tests` 5/5 + đo thời gian | **Loss bit-exact** (maxdiff 0.0), **time ↓ ≥ 50%** |

### 4.2 Fused AdamW (2 tuần)
**File:** `tv/training/adamw.tkv` (mới)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Gộp momentum + weight decay vào **1 vòng lặp** | `m ← β₁m + (1−β₁)g`; `dw ← m − lr·β₂w` | **Loop traversal ↓ 1/3** (tương đương torch `fused=True`) |
| AVX2 FMA cho cả 3 phép | Pattern `simd_gemm`: `y = a·x + b·y` | **Throughput ↑ 2-3×** |
| Numerical test | Overfit 20 mẫu, so loss curve | **Loss khớp** (≤1e-5 sau 20 bước), **time ↓ 30%** |

### 4.3 Cache-blocking cho linear bwd (1 tuần)
**File:** `tv/training/linear_bwd.tkv` (mới, theo pattern T×T của `simd_matmul.tkv`)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Tiling theo L1 cache | seq8: T=64, seq256: T=128, seq1024: T=256 | **Memory traffic ↓ 4-5×** (đã có tiền lệ 3390→63 ms = 53×) |
| Register tiling (accumulator ở `__m256`) | Không write/read L1/L2 giữa các tile | **Latency ↓ 20%** phase linear bwd |

---

## 5. PHASE 3 — POLICY ADAPTIV + MIXED PRECISION (2 tuần)

**Mục tiêu:** thêm 5-10%; tỉ lệ **~20-22×**.

### 5.1 Runtime feature detection
**File:** `tv/training/dispatch.tkv` (mới, theo pattern `backend.tkv` kind dispatch)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| Detect `AVX2` / `FMA` / `AVX-512_FP16` lúc init | `cpuid` / `mkldnn.is_available()` | Flags `has_avx2`, `has_fma`, `has_fp16` |
| Chọn path theo flag | `simd=1` khi có hardware, `simd=0` fallback | **100% parity**; path nhanh gấp 2-3× |
| Auto-fallback khi vượt numerical bound | Probe maxdiff mỗi bước, >1e-6 → scalar | **Graceful degradation** |

### 5.2 Mixed precision cho attention intermediate
**File:** `tv/training/precision.tkv` (mới)

| Sub-task | Chi tiết | Success Criteria |
|---|---|---|
| FP32 accumulate, FP16 intermediate, rescale FP32 trước softmax | `ctx.half()` → `smm_fwd` → softmax FP32 | **Maxdiff ≤ 2.4e-7** logits (`gpt2_tests` 17/17) |
| Loss-scale adaptive khi loss diverge | Giống `amp.autocast` nhưng cho CPU | **Stable 100+ bước**, không NaN |
| Benchmark seq1024 mode 0 | So FP32 vs FP16 intermediate | **Throughput ↑ 15-20%** |

---

## 6. LỘ TRÌNH (3 tháng, CHƯA COMMIT)

| Tuần | Việc | Deliverable |
|---|---|---|
| 1-2 | `layernorm_bwd` FMA tiled | `tv/training/layernorm_bwd.tkv` + latency report |
| 3-4 | `gelu_bwd` erf path | `tv/training/gelu_bwd.tkv` + maxdiff ≤1.5e-7 |
| 5-6 | Fusion `proj_bwd` | `tv/training/proj_bwd.tkv` + latency ↓15-20% |
| 7-8 | Lazy materialize | `tv/training/optimizer.tkv` + time ↓50% |
| 9-10 | Fused AdamW | `tv/training/adamw.tkv` + throughput ↑30% |
| 11-12 | Cache-blocking linear bwd | `tv/training/linear_bwd.tkv` + throughput ↑15-20% |
| 13-14 | Policy adaptiv + mixed precision | `tv/training/dispatch.tkv`, `tv/training/precision.tkv` |
| 15 | Báo cáo KPI + quyết định P6 | Cập nhật file này + `HANDOFF.md` |

**Quy tắc bắt buộc:** mỗi tuần đo lại **cả 3 seq** (8/256/1024) và chạy lại **5 gate**. Chỉ giữ thay đổi nếu **net win trên cả 3 seq** và **84/84 vẫn xanh**. Mặc định vẫn `simd=0` cho tới khi chứng minh được.

---

## 7. KPI CUỐI

| KPI | Baseline (P5.3) | Mục tiêu | Cải thiện |
|---|---|---|---|
| seq1024 fwd+bwd | 150.625 s | **~70 s** | ≈ 53% giảm |
| seq1024 full step | **154.234 s** | **~85 s** | ≈ 45% giảm |
| **Tỉ lệ tkv/torch (full step)** | **43.5×** | ~~20×~~ **1.0× rồi <1.0× (chủ khoa 2026-10-20, xem §12)** | "thua nặng" → **bằng/nhanh hơn torch** |
| seq256 vs P5.1-only | 2.65× | **≥ 3.5×** | |
| 84/84 assertions | Pass | **Pass** | Không hồi quy |
| Loss vs PyTorch fixture | ≤2.4e-7 logits / ≤8.3e-7 124M grad | **Không đổi** | Parity duy trì |

**Cập nhật 2026-10-20 sau P5.4 (§10):** full step seq1024 **141.219 s → 39.9×** (từ 154.234 s / 43.5×); seq8 20.7×, seq256 26.9×.

**Cập nhật 2026-10-20 sau P5.5 (§11):** full step seq1024 **118.641 s → 33.5×** (gộp forward, −16.0%); seq8 18.7×, seq256 23.2×.

**Cập nhật 2026-10-20 sau P5.6 (§12, dòng P5.6):** full step seq1024 **99.719 s → 28.2×** (cache attn-fwd trong `block_bwd`, −16.0%; bit-exact 0.0); seq8 18.1×, seq256 21.2×. Bwd còn 68.2 s: ATTN_BWD 41.5 / MLP_BWD 16.0 / LHEAD 11.5 — mục tiêu kế P5.7.

**Mục tiêu mới 2026-10-20 (chủ khoa) — THAY "~20×":** **BẰNG PyTorch (1.0×) rồi NHANH HƠN (≤0.8×)** trên cả 3 seq, giữ 84/84. Ladder đầy đủ + phân tích sàn: **§12**.

## 8. RỦI RO + CONTINGENCY

| Rủi ro | Xác suất | Contingency |
|---|---|---|
| GELU erf approx trôi khỏi 2.4e-7 | Thấp (numerics contract đã test) | Default erf; tanh-approx chỉ là debug path |
| Lazy materialize phá parity | Trung bình | Gate từng stage; revert op về eager materialize |
| Thắng seq256 không transfer sang seq8 | Trung bình | Benchmark đủ 3 seq mỗi lần; giữ `simd=0` mặc định |
| Trượt tiến độ 3 tháng | Cao (codebase phức tạp) | Phase 1 đã mang ~50% giảm → dừng vẫn có giá trị |
| ~~**tkvc không có primitive thread → P6 (đa luồng) bị chặn**~~ **ĐÃ HẾT 2026-10-21** | — | ⓐ đã trả lời: CÓ primitive (`thread_spawn/join`) → P6 GĐ1 đã song song attention_bwd, xem §12 row P6 |
| Drift nhiệt/thermal khi chạy đa luồng làm số đo bay | Cao | A/B cùng phiên + best-of + control; luôn ghi drift trong bench |
| Mixed precision phá parity fp32 (nếu làm P7) | **ĐÃ ĐÓNG (2026-10-22): P7 bị bác bỏ bằng số đo** — xem dòng P7 | Amendment numerics-contract + gate AMP riêng; fp32 master weights |
| Policy adaptiv thêm latency | Thấp | Bake policy vào compile-time constant nếu cần |

## 9. FILE MỚI (dự kiến tạo)

| File | Vai trò |
|---|---|
| `docs/perf-plan-triz-roadmap.md` | File này — roadmap + KPI |
| `tv/training/layernorm_bwd.tkv` | LN backward FMA tiled |
| `tv/training/gelu_bwd.tkv` | GELU erf backward |
| `tv/training/proj_bwd.tkv` | Fusion proj_bwd + dw_proj |
| `tv/training/optimizer.tkv` | Lazy materialize + optimizer wiring |
| `tv/training/adamw.tkv` | Fused AdamW |
| `tv/training/linear_bwd.tkv` | Cache-blocked linear backward |
| `tv/training/dispatch.tkv` | Runtime feature detection + path policy |
| `tv/training/precision.tkv` | Mixed precision policy |
| `tv/tests/simd_probe_aq.tkv` | Phân rach backward theo pha, seq qua `_seq_cfg.txt` (8/256/1024) |
| `tv/tests/simd_probe_ar.tkv` | Phân rach từng phép trong `attention_bwd` + P13 build-vs-gemm |
| `tv/tests/simd_probe_at.tkv` | Phân giải `now_ms`/`now_us` (~15.6 ms — bài học đo) |
| `tv/tests/simd_probe_au.tkv` | Test ngôn ngữ `<=` + if/else lớn |
| `tv/tests/simd_probe_as.tkv` | Bit-exact saves-vs-recompute + đo fwd_saved overhead (P5.5) |
| `tv/tests/simd_probe_av.tkv` | Tách LHEAD @seq: softmax-grad / transpose / 2 GEMM (P5.7) |
| `tv/tests/simd_probe_aw.tkv` | Micro-bench transpose blocked-vs-plain (P5.7 — blocked chậm 2.7×) |
| `tv/tests/simd_probe_bh.tkv` | P6 GĐ3: prototype wave-parallel tile-build (bit-exact 0.0) + phân rã MLP_BWD — chốt Stage A (builder fix) + Stage B (ngưỡng route 40M) |
| `tv/tests/simd_probe_bk.tkv` | P6 GĐ4: phân rã full-step @seq (fwd/loss/zero/bwd/step/unflatten) — chốt "khoang thua" plumbing ~6.6s seq-independent |
| `tv/tests/simd_probe_bl.tkv` | P6 GĐ4: test parallel AdamW @124M — SHARED-12 = 0.36× (chết), OWN+MERGE = 0.29× (merge 372M 11.7s chết), norm PAR12 = 2.24× BIT 0.0 → bỏ AdamW parallel |
| `tv/tests/simd_probe_bm.tkv` | P6 GĐ4: prototype norm/loss/gelu-parallel + copy-pathology — ALL BIT-EXACT: norm 2.79×, loss 5.10×, gelu_fwd 7.33×, gelu_bwd 5.03×, copy SHARED 3.0× (không container-lock); 2 bug compiler mới (nested list[f32] append → ArrayTypeMismatch; main quá lớn → BadImageFormat) |
| `tv/tests/simd_probe_bn.tkv` | P6 GĐ5: phân rã ATTN_BWD @n=1024 (copy worker/merge nguyên văn) — whole 1453ms = alloc 359 (25%!) + worker wall 516 (serial ref 4468 → eff **8.66×**) + GEMM serial 469 → chốt Stage A packed buffers (bỏ zero-fill 11/12) |
| `tv/tests/simd_probe_bo.tkv` | P6 GĐ6: phân rã MLP_BWD @1024 — r2 391 (dx2 140 + xT2 94 + dw2 156) + r1 313 + geluB 62; GEMM 640ms near-peak → chốt smm_fwd_at + geluB overlap |
| `tv/tests/simd_probe_bp.tkv` | P6 GĐ6: xác thực smm_fwd_at (maxdiff 0.0, 2 shape) + geluB worker ‖ dw2 main (266→187ms, maxdiff 0.0 — nested spawn an toàn) |
| `tv/tests/simd_probe_bq.tkv` | P6 GĐ7: LHEAD còn lại + EMBED_BWD — dwte-add PAR12 seg+merge 719ms vs serial 328ms **đo bác bỏ** (merge 50257×768 đắt) |
| `tv/tests/simd_probe_br.tkv` | P6 GĐ7: phân rã plumbing forward-side @124M — LN-fwd recompute 24 gói = 359ms, zero_grad = 250ms, AdamW step 4234ms + unflatten 766ms → chốt 2 thay đổi GĐ7 |
| `tv/tests/simd_probe_bw.tkv` | P6 GĐ7: main-thread + 11 worker (eff main 5.59× < worker 6.48×) **đo bác bỏ** — giữ nguyên spawn 12 head |
| `tv/tests/simd_probe_bz.tkv` | P6 GĐ8: dò quy tắc **boxing biến cục bộ** (int64 native vs `TkvInt` 16 byte) — dùng `time_perf()` (Stopwatch) vì `now_ms()` có lượng tử 15.6 ms làm probe BY v1 vô hiệu |
| `tv/tests/simd_probe_narrow.tkv` | P6 GĐ8: probe quy tắc narrowing (biến nào khai native int64) — đọc `.locals` trong `out/*.il` |
| `tv/tests/simd_probe_narrow2.tkv` | P6 GĐ8: **bằng chứng quy tắc** — khai ở TOP-LEVEL + chỉ gán "chứng minh được" (`x = x + 1`, `x = y`) ⇒ native int64; gán trong vòng lặp / biểu thức có `*` ⇒ `TkvInt` |
| `tv/tests/simd_probe_by.tkv` | P6 GĐ8: A/B tile-build v1 — **vô hiệu** (đồng hồ `now_ms()` lượng tử 15.6 ms + biến P bị box) |
| `tv/tests/simd_probe_by2.tkv` | P6 GĐ8: **probe quyết định** (đồng hồ `time_perf`) — A-tile 1.49×, B-tile 1.43×, acc 1.08×, core 1.54× (Smat) / 2.28× (gCtx), maxdiff 0.0 |
| `tv/tests/simd_probe_ca.tkv` | P6 GĐ9: nhân rộng quy tắc int64 ra vòng ngoài GEMM — A/B unflatten + norm @124M, **4 lần chạy**: unflatten 1.02/0.998/0.991/0.990, norm 1.04/0.996/1.002/0.996 (**≈ 1.00, bit-exact 0.0**) ⇒ **đo bác bỏ** (vòng memory-bound ≈3.4 GB/s, arithmetic bị ẩn) |
| `tv/tests/simd_probe_cb.tkv` | P6 GĐ10: so công thức **AdamW vs PyTorch** + A/B 4 biến thể @124M (3 lần chạy): V1 hoist `(1-b1)/(1-b2)` **0.93×** ❌, V2 gộp `p` **0.92×** ❌ (CSE/JIT đã tự làm), V3 **gộp 2 div → 1 div** = `m*bc2s/(sqrt(v)+eps*bc2s)` **1.068×** ✅ maxdiff **1.19e-07** → wire 4 file trainer |
| `tv/tests/simd_probe_cc.tkv` | **DỰ KIẾN P6 GĐ13**: probe chẩn đoán AdamW parallel lần 3 @124M — V0 serial / V1 compute-only PAR12 (OWN segs) / V2 ghi SHARED phân đoạn / V3 OWN+MERGE / V4 copy control (BM 3.0×) → chốt nguyên nhân 2 lần bust, hoặc ĐÓNG item (§13.3) |
| `tv/tests/simd_probe_cd.tkv` | **DỰ KIẾN P7b**: đo GEMM native vs `torch.matmul` cùng shape cùng phiên (3 shape đại diện, GFLOPS + số thread THẬT của native pool) → điều hướng parity (§13.3) |

---

---

## 10. TIẾN ĐỘ 2026-10-20 — P5.4 ATTACK ATTENTION BWD (XONG)

**Giữ lại (đo net win trên cả 3 seq, 5 mirror `gpt2_ref` đồng bộ):**

1. **Bỏ 2 transpose n²/head** trong `gpt2_attention_bwd` — dk/dv tính qua transpose **nhỏ** của Q/dctx (n×hd = 65K thay vì n×n = 1M) + writeback đọc strided `gDkT[d*n+t2]`. Cùng tích lũy p-tile tăng dần → **bit-exact**.
2. **`dw = xᵀ·gy` trực tiếp** trong `gpt2_linear_bwd` — bỏ gyT + dwT transpose (seq1024 qkv: 4.13M → 0.79M copy/layer). Bit-exact.

**Đã REVERT (đo chứng minh là regression):** thứ tự vòng `_smm_core` p0→j0→i0 cho m>n (lý thuyết P13: "ít build tile hơn") — chạy thật **chậm hơn**: FWD_BODY +101%, mọi pha GEMM +40-100% @seq256 (tách biệt bằng `git stash` từng nhóm). **Bài học: số lượt build tile không phải chi phí thật — chỉ có đo A/B mới chốt được.**

| seq | full step TRUOC | SAU | Δ |
|---|---|---|---|
| 8 | 22.219 s | **19.297 s** | −13.1% |
| 256 | 41.218 s | **37.813 s** | −8.3% |
| 1024 | 154.891 s (ctrl cùng phiên) | **141.219 s** | **−8.8%** |

- fwd+bwd @1024: 150.750 (ctrl) → **136.469 s** (−9.5%); **bwd 125.969 → 111.625 s (−11.4%)**.
- Pha AQ @1024 trước/sau: `attention_bwd`×12 48.86 → **38.20 s (−21.8%)**, `mlp_bwd` −8.1%; LN/GELU/EMBED không đổi ✓ (đúng code không sửa).
- Loss khớp từng chữ số (10.760399 → 13.442912); **84/84 gate xanh**; tỉ lệ vs torch 43.5× → **39.9×**.
- ~~Chưa làm (ứng viên kế, theo số đo §1): gộp forward loss+backward (−19 s)~~ → **đã làm: P5.5 (§11)**. Còn: cache attn-fwd trong `block_bwd`, Phase 2 flatten, lm-head.

---

## 11. TIẾN ĐỘ 2026-10-20 — P5.5 GOP FORWARD: BACKWARD DÙNG SAVES (XONG)

**Cơ chế:** `gpt2_forward_saved` trả `[logits, xf, xf_in, xins…]`; `gpt2_backward(…, saves=None)` — param mới **cuối cùng**: `saves != None` → dùng lại forward của loss pass + `saves[0]` thay GEMM logits lm-head; `saves == None` → đường recompute cũ nguyên vẹn (mọi gate/fixture không đổi). Bit-exact xác minh bằng probe `simd_probe_as.tkv` (logits/gflat/out/reuse = 0.0 @n=8) + loss khớp từng chữ số @seq1024. Đánh đổi: +~250 MB memory @seq1024 để giữ activations (chuẩn LSTM-style).

| seq | full step TRƯỚC (P5.4) | SAU (P5.5) | Δ |
|---|---|---|---|
| 8 | 19.297 s | **17.469 s** | −9.5% |
| 256 | 37.813 s | **32.578 s** | −13.8% |
| 1024 | 141.219 s | **118.641 s** | **−16.0%** |

- bwd @1024: 111.6 → **86.2 s (−22.8%)**; fwd không đổi (24.6 s, overhead saves ≈ 0).
- AQ @1024: TOTAL_bwd 103.2 → **84.3 s** — đúng bằng phần recompute bị bỏ; phân bổ mới: ATTN_BWD×12 42.4 s, MLP_BWD 16.2 s, LHEAD 11.7 s, LN_BWD 0.77 s, EMBED 1.2 s.
- **84/84 gate**; tỉ lệ vs torch 39.9× → **33.5×** (seq8 18.7×, seq256 23.2×).
- Lưu ý đo: máy drift ~5-8% trong ngày → % A/B bench có thể lệch vài điểm; ground truth cùng tiến trình: probe AS bwd @n8 −16.8%, AQ TOTAL_bwd −18.9 s @1024.
- Ứng viên kế (theo AQ mới): cache attn-fwd + MLP fwd trong `block_bwd` (P5.6), lm-head, Phase 2 flatten (thực chất chỉ ~2 s).

---

## 12. KẾ HOẠCH TỚI PARITY — "BẰNG RỒI NHANH HƠN PYTORCH" (2026-10-20, chủ khoa)

**Mục tiêu mới (thay mục ~20×):** full step ≤ torch (1.0×) rồi **nhanh hơn (≤0.8×)** trên **cả 3 seq**, giữ 84/84 parity assertions (fp32 contract).

**Phân tích sàn trung thực (số đo 2026-10-20):** ~560 GFLOP/fwd+bwd @seq1024 (ước tính kiến trúc: 12 layer × 12h² × 3 (fwd+bwd) + lm-head + attention n²); torch 3.542 s với **14 thread** ≈ 190 GFLOP/s tổng. tkv đơn luồng 118.6 s ≈ 4.7 GFLOP/s. Sàn lý tưởng **đơn luồng** với kernel f32 AVX2 hoàn hảo (~15 GFLOP/s thực dụng) ≈ **37 s — VẪN > 3.542 s** → **đa luồng là ĐIỀU KIỆN CẦN**, không phải tùy chọn; kernel đơn luồng tốt nhất chỉ đưa ~33.5× xuống ~5-7×.

| Bước | Nội dung | Ước @seq1024 | Tỉ lệ |
|---|---|---|---|
| **P5.6** ✅ **XONG 2026-10-20** | Cache attn-fwd trong `block_bwd` (rs trong saves) — **đo: −18.9 s @1024** (ước ban đầu −8-10 s; full step 118.641 → **99.719 s**), bit-exact 0.0, 84/84 | −18.9 s | **28.2×** |
| **P5.7** ❌ **THỬ 2026-10-20, ĐO BÁC BỎ** | Lm-head bwd: tách phần (probe AV/AW): softmax-grad 1.33 s, transpose ggT 1.56 s, dxf 2.44 s, dwte 3.06 s. Thử (a) dwteT tránh transpose → **regression seq8 +22%/seq256 +17.6%**; (b) transpose blocked → **chậm 2.7×** → revert. **Cần kernel API** (A-transposed + beta) — chuyển yêu cầu sang P5.8 | 0 (chờ kernel) | 28.2× giữ nguyên |
| **P5.8** ✅ **XONG 2026-10-21** | Kernel GEMM v2: (a) **A-transposed ✅** — `smm_fwd_at` bỏ transpose ggT, wire vào lm-head `dwte` (cả 5 file `gpt2_ref`), **bit-exact 0.0**; (b) **tích lũy native `simd_add` ❌ ĐO BÁC BỎ** — chậm hơn (dxf @1024: 3000 vs 2703 ms; `simd_add` allocate ~16 B/element); (c) **nguyên nhân v2 chậm 2.5× đã CHỐT** bằng probe BC — call site `simd_add`/`simd_sub` trong cùng function làm codegen vòng scalar chậm 2.5-4× **dù không chạy** → tách `_p58_core_simple` không call site. **Revert hunk P5.7 bug** (`use_flat=1` bỏ qua `dxf` → grad sai). Gate **84/84**, loss seq1024 khớp exact P5.4 (10.760399→13.442912). Bench A/B full step: seq8 16.94→**17.16** (+1.3% drift nhiệt, loss deterministic), seq256 29.74→**29.64**, seq1024 99.72→**98.20 s (−1.5%)** | LHEAD −0.9 s @1024 | **27.7×** |
| **P6** ✅ **GĐ1 XONG 2026-10-21** | **Trả lời ⓐ: tkvc CÓ primitive** (`thread_spawn/join` — Task<T> không GIL; native simd_gemm đã có thread pool riêng, `TKV_GEMM_MIN_N=96`). **Đã làm: song song `gpt2_attention_bwd` theo 12 head** — quy tắc đo được (probe BF/BG/BG2): worker **list RIENG + main merge = 5.2×**, ghi list **CHUNG chỉ 1.24×** → own-list pattern; worker gemm đều T=64<96 → `gemm_rows` trực tiếp (bp `__declspec(thread)`, KHÔNG đụng `g_job` global không lock) → an toàn concurrent; fallback `n<128` giữ đường cũ. Bit-exact: loss khớp exact từng chữ số, gate **84/84**, digits gate n=192 giống hệt trước. **Kết quả: seq8/256/1024 = 17.015/28.671/75.297 s (−0.8/−3.3/−23.3%), ATTN_BWD 40.7→17.8 s, TOTAL_bwd 68.8→44.3 s.** Bác bỏ: `TKV_GEMM_MIN_N=64/48` (chậm hơn). Chưa làm: attention FWD per-head, MLP_BWD 15.8 s, LHEAD 11.9 s, AdamW/plumbing ~7 s | bwd −23.3 s @1024 | **21.3×** |
| **P6** ✅ **GĐ2 XONG 2026-10-21** | **Song song `gpt2_attention` FWD theo 12 head** — cùng pattern GĐ1 (class `_AtnFJob` + worker `_attn_f_one` + main `_attn_f_merge`, own-list; fallback `n<128` giữ đường cũ nguyên vẹn). An toàn native: 2 GEMM trong worker đều **T=64<96** (`Smat = smm_dot(n,n,hd)` → T=min(64,roundup64(n)); `gCtx = smm_fwd(n,hd,n)` → T=min(_tilek(n),64)) → `gemm_rows` thread-local, không đụng `g_job` không lock; proj (T≥96) chạy main thread sau join. **Thêm guard `NHeads` vào worker/merge cả bwd+fwd** — sửa latent OOB (mirror nhỏ NHeads=4 vẫn spawn 12 nhóm [k,k+1); gate nhỏ không trigger nhưng chạy n≥128 sẽ ghi ngoài hàng). Bit-exact: loss seq1024 khớp exact (10.760399→13.442912), gate **84/84**, gate n=192 fwd = 1.1e-09 vs scalar. **Kết quả: seq8/256/1024 = 17.078/26.141/63.781 s (+0.4/−8.8/−15.3% vs GĐ1), AQ FWD_BODY 20.8→11.06 s (−47%), ATTN_BWD 17.8 giữ nguyên, TOTAL_bwd 43.9.** | fwd −9.7 s @1024 | **18.0×** |
| **P6** ✅ **GĐ3 XONG 2026-10-21** | **Builder fix + wave-parallel tile-build** (probe BH `tv/tests/simd_probe_bh.tkv`, ALL BIT-EXACT 0.0): (a) **Stage A** — builder `append` → indexed write vào buffer tái sử dụng trong `_smm_core`/`_p58_core_simple` (append List capacity-doubling ~23 ns/ptu): dxf 2468→1203, dwte 3813→1984, logits 2469→1781 ms — **1.4-2× mọi GEMM, không cần thread**; (b) **Stage B** — `smm_*_pd`: 12 worker build tile RIENG (indexed write, KHÔNG `simd_gemm` → không đụng `g_job` không lock), main replay gemm đúng thứ tự i0→p0→j0 → bit-exact; route `vol<40M` → serial (24 buffer 78 ms + spawn làm chậm shape nhỏ — BH S1/S2), `≥40M` → wave 12 (chỉ shape vocab 50257); wire 4 site lm-head (logits fwd/bwd, dxf, dwte) ×5 file `gpt2_ref`. MLP phân rã (BH[3], sum-vs-whole 1.2%): inner 156 · gelu_fwd 125 · r2 469 · gelu_bwd 187 · r1 391 ms/layer → **gelu scalar 24% MLP_BWD (ứng viên GĐ4)**. **Kết quả: seq8/256/1024 = 16.109/25.922/55.469 s (−5.7/−0.8/−13.0%), TOTAL_bwd 44.9→36.5 s (−18.8%), FWD_BODY 11.2→9.4 s; 84/84, loss 3 seq khớp exact.** seq256 lãi ít: vocab GEMM n=256 vol 38.6M < 40M → serial (PAR tại T=256 chưa đo) | bwd −8.4 s @1024 | **15.7×** |
| **P6** ✅ **GĐ4 XONG 2026-10-21** | **Plumbing + gelu parallel** (probe BK phân rã full-step → BL test AdamW @124M → BM prototype, ALL BIT-EXACT): (a) **adamw_parallel BỎ theo số đo** — BL: SHARED-12 ghi disjoint = 0.36× (11.9s vs 4.2s serial), OWN+MERGE = 14.6s (merge 372M = 11.7s) → cả 2 biến thể chết (worker compute vẫn 3.9→0.5s — lỗi ở ghi/merge quy 124M); (b) **BM thắng: `_norm_groups` 2.79× (1047→375ms), `gpt2_loss` 5.10× (719→141ms, n=1024), gelu_fwd 7.33× (110→15ms), gelu_bwd 5.03× (156→31ms)** — mẫu chung: worker own-seg + main gop DUNG thu tu → bit-exact; **copy SHARED 3.0× → KHÔNG có container-lock** (BL[2] 0.36× do vòng f64-boxing 124M); guard theo đo: gelu 786K break-even → 2M, loss n*V ≥ 4M, norm ≥ 4M; 2 bug compiler mới: nested list[f32] append top-level → ArrayTypeMismatchException (f64 OK) → merge unrolled helper phang, main ~900 dòng → BadImageFormat → tách hàm. Wire 10 file (trainer×4, gpt2_ref×3, ops×3). **Kết quả: seq8/256/1024 = 15.062/25.406/50.875 s (−6.5/−2.0/−8.3%), TOTAL_bwd 36.5→33.5 s (−8.0%), MLP_BWD 14.3→12.3; 135/135 gate, loss 3 seq khớp exact** | bwd −3.0 s @1024 (toàn bộ bước −4.6 s) | **14.4×** |
| **P6** ✅ **GĐ5 XONG 2026-10-21** | **Attention packed buffers** (probe BN phân rã ATTN_BWD @1024, copy worker/merge nguyên văn): whole 1453ms/layer = **alloc 12 ctxo+dqo 359ms (25%)** + worker wall 516 (serial ref 4468 → **eff 8.66× — worker KHÔNG yếu**) + GEMM serial 469. Nguyên nhân: mỗi worker chỉ ghi 1 head (n·hd + n·3hd) nhưng driver alloc buffer FULL n·h / n·3h → 11/12 zero-fill + merge copy cả phần zero. **Stage A:** `_attn_hb_one`/`_attn_hb_merge` ghi-đọc packed `[t2·hd]`/`[t2·3·hd]`, driver `n·hd`/`n·3·hd` (37.7M → 3.15M zero-fill); **FWD cùng bệnh** (`co0..11 = t_zeros([n,h])`) → `_attn_f_one`/`_attn_f_merge`/driver cũng packed; serial n<128 nguyên vẹn. Bit-exact (giá trị + thứ tự tích lũy không đổi). Lưu ý: lần chạy đầu gpt2_full crash transient `ArgumentOutOfRangeException` lúc load fixture 230MB → chạy lại cùng binary 33/33. **Kết quả: seq8/256/1024 = 15.187/23.625/45.890 s (drift +0.8/−7.0/−9.8%), ATTN_BWD 15.9→11.4 (−28.7%), FWD_BODY 9.13→7.78, TOTAL_bwd 33.5→29.7 (−11.5%); 135/135, loss 3 seq khớp exact** | ATTN_BWD −4.5 s + FWD −1.3 s @1024 | **13.0×** |
| **P6** ✅ **GĐ6 XONG 2026-10-21** | **MLP_BWD: smm_fwd_at + geluB overlap** (probe BO phân rã mlp_bwd @1024 — whole 1000ms = r2 391 [dx2 140 + **xT2 94** + dw2 156] + r1 313 + geluB 62 + inner 94; GEMM 640ms near-peak AVX2 → chu địch là PHI + SERI; probe BP xác thực: smm_fwd_at maxdiff **0.0** 2 shape, geluB worker + dw2 main 266→187ms maxdiff 0.0 — **nested spawn an toàn**): (a) `gpt2_linear_bwd` simd==1: `xT + smm_fwd` → `smm_fwd_at` đọc x trực tiếp (lợi mọi caller mlp+attn); (b) attn worker gDkT/gDvT cùng cơ chế (T=64<96); (c) `gpt2_mlp_bwd` simd==1 tách r2: dx → **spawn geluB worker ‖ dw2 main** → join → db → pack → r1; thứ tự gflat nguyên vẹn → bit-exact; simd==0 giữ nguyên. Lưu ý phiên đo nóng (FWD_BODY +2.3% drift): seq8 +0.8%. **Kết quả: seq8/256/1024 = 15.313/23.516/45.000 s (+0.8/−0.5/−1.9%), MLP_BWD 11828→10890 (−8.0%), TOTAL_bwd 29.7→28.8 s (−3.0%); 135/135, loss 3 seq khớp exact** | MLP_BWD −0.94 s @1024 (phi transpose 109 + geluB 62 ms/layer, drift nhiệt nuốt bớt) | **12.7×** |
| **P6** ✅ **GĐ7 XONG 2026-10-22** | **Plumbing: zero_grad overlap + LN-fwd saves** (probe BR phân rã plumbing @1024: whole-step − forward_saved = 6.9-9.8s; trong đó **LN-fwd recompute 24 gói = 359ms**, **zero_grad = 250ms** @124M; probe BQ: LHEAD dwte-add song song theo 12 seg **đo bác bỏ** — serial 328ms vs PAR12+merge 719ms; probe BW: main-thread + 11 worker **đo bác bỏ** — spawn11+main = 453ms eff 5.59× vs spawn12 = 391ms eff 6.48×, eff main thread < 1 worker): (a) `gpt2_train_step` + bench `_step124`: **spawn zero_grad worker TRƯỚC forward, join TRƯỚC backward** (forward không đụng `opt.g` → bit-exact, ăn 250ms/buộcb @124M); (b) `gpt2_block_saved` lưu thêm `a_ln`/`m_ln` (đầu vào LN), `gpt2_block_bwd` nhận `save_a`/`save_m` — bỏ 24 gói `gpt2_layernorm` recompute (359ms @1024); cùng input cùng phép → **bit-exact**, saves format cũ (append CUỐI) vẫn đọc được. **Kết quả: seq8/256/1024 = 15.172/22.844/43.890 s (−0.9/−2.9/−2.5%), 135/135, loss 3 seq khớp exact** | plumbing −0.6 s @1024 (đo thực −1.1s gồm nhiễu) | **12.4×** |
| **P6** ✅ **GĐ8 XONG 2026-10-22** | **Giảm overhead interpreter trong GEMM** (probe BY2 đồng hồ `time_perf` — `now_ms()` lượng tử 15.6 ms làm probe BY v1 vô hiệu): đọc IL phát hiện biến cục bộ gán **trong vòng lặp** bị compiler **box thành `TkvInt`** (struct 16 B) ⇒ mỗi phép cộng chỉ số là phép cộng trên struct; quy tắc (probe `narrow2`): **phép gán đầu tiên ở top-level hàm** + các phép gán sau đều "chứng minh được" ⇒ **native int64** (một phép gán không chứng minh được ở top-level ⇒ box lại `TkvInt`). Đo: A-tile **1.49×**, B-tile **1.43×**, acc **1.08×**, core **1.54×** (Smat) / **2.28×** (gCtx), maxdiff **0.0** (bản "copy p-tile đầu" **chậm hơn** cộng đơn → bỏ, giữ nguyên phép cộng ⇒ không đổi dấu ±0.0). Wire `_tile_a_w`/`_tile_b_w`/`_tile_a_at_w` (index chạy + đọc trực tiếp) + helper mới `_acc_f` cho `_smm_core` + `_p58_core_simple` + `_pd_do` (đường wave-12 lm-head); IL xác nhận cả 4 hàm khai `int64`. **KHÔNG** đổi thứ tự vòng (bản p0→j0→i0 đã đo REGRESSION ở §10). **Kết quả: seq8/256/1024 = 12.969/18.125/37.828 s (−15.0/−21.1/−15.1%), loss 3 seq khớp EXACT từng chữ số, 135/135**; tỉ lệ vs torch 16.2/16.3/12.4× → **13.9×/12.9×/10.7×** | GEMM interpreter −6.7 s @1024 | **10.7×** |
| **P6** ❌ **GĐ9 ĐO BÁC BỎ 2026-10-22** | **Nhân rộng quy tắc native int64 ra vòng ngoài GEMM** — scan IL của `out/bseq_new.il` (đường train đầy đủ) tìm mọi biến người dùng bị box `TkvInt`: `_step124.off` (unflatten), `_tn_one.off`/`_norm_groups.off` (norm), `_ls_one.k`, vòng ngoài `_smm_core`; **`step` (AdamW 4.2 s), `zero_grad`, `gpt2_loss`, LN, gelu ĐÃ dùng `int32 'i'` → không thuộc phạm vi sửa**. Probe `simd_probe_ca.tkv` (N=124M, `time_perf`, **4 lần chạy**): unflatten 2.27-2.34→2.29-2.30 ns/ptu (**1.02/0.998/0.991/0.990 — median ≈ 0.99**), norm 2.23-2.33→2.24-2.25 (**1.04/0.996/1.002/0.996 ≈ 1.00**), bit-exact 0.0. **Bác bỏ:** cả hai vòng memory-bound (≈3.4 GB/s đọc+ghi) nên arithmetic bị ẩn hết — bản "sửa" không nhanh hơn; quy tắc chỉ ăn tiền ở vòng compute-bound (GĐ8: tile-build GEMM 1.5-2.3×); tiết kiệm tối đa ~18 ms @1024 = **0.05%**, nhỏ hơn nhiễu đo, mà chi phí wire = 7 file mirror. Hướng thật: `AdamW.step` (4.2 s) phải giảm số phép/phần tử, không phải sửa khai báo biến | **0 (không làm)** | giữ **10.7×** |
| **P6** ✅ **GĐ10 XONG 2026-10-22** | **Học PyTorch cho AdamW: gộp 2 div → 1 div** (so công thức trước với `torch/optim/adamw.py` rồi mới đo — probe `simd_probe_cb`, `time_perf`, N=124M, 3 lần chạy): PyTorch `denom = sqrt(v)/sqrt(bc2) + eps` rồi `addcdiv_` = **1 div/ptu**; ta `denom = sqrt(v)/bc2s + eps` **rồi** `m/denom` = **2 div/ptu**. Định lý gộp (đại số chính xác): `m/(sqrt(v)/bc2s+eps) == (m*bc2s)/(sqrt(v)+eps*bc2s)`. Đo 4 biến thể: V1 hoist `(1-b1)/(1-b2)` **0.93×** ❌, V2 gộp `p` **0.92×** ❌ (JIT/CSE đã tự làm — **phải đo từng cái riêng**), V3 gộp div **1.068×** ✅ maxdiff **1.19e-07**. Wire 4 file `trainer`. **Kết quả: `ADAMW_STEP` 4370→3930 ms (−10.1%); seq8/256/1024 = 12.516/17.313/36.703 s (−2.6/−3.3/−3.1%); loss in ra khớp exact 7 chữ số cả 3 seq; gate 135/135 + `trainer_tests` 15/15 PASS với fixture PyTorch thật (TOL=1e-5, chênh lớn nhất 2.5e-06)**; tỉ lệ vs torch 13.9/12.9/10.7× → **13.4×/12.3×/10.4×** | AdamW −440 ms/bước @124M | **10.4×** |
| **P6** ❌ **GĐ11 ĐO BÁC BỎ 2026-10-22** | **Song song softmax-grad lm-head bwd** (kế hoạch §13.3 — mẫu `_ls_one` GĐ4) — wire thử 12 worker row-parallel ghi thẳng `gg.data` phân đoạn (bit-exact, chk AQ `0.000105296414` giữ nguyên) rồi **đo bác bỏ**: AQ LHEAD_BWD **7407 → 10719 ms (+3.3 s)**. Probe `simd_probe_ce.tkv` tách nguyên nhân @51.5M ptu (2 lần chạy, maxdiff 0.0 cả SH/OW): serial **1110-1140**; **SHARED ghi+exp = 3110-4453 (0.25-0.36×)** — CÙNG bệnh AdamW BL/SHARED; **CP copy shared KHÔNG exp = 484-547 (2.0-2.3×)** → shared write thuần KHÔNG bệnh, bệnh là **phep toan + ghi list tuong minh**; **OW own-seg: worker 484 (2.36×) + merge 1172 + alloc 359-375 = TOTAL 2031 (0.56×)** — merge phiên bản đơn luồng (~22.7 ns/ptu) ăn hết lợi thế, ceiling kể cả merge miễn phí cũng chỉ ≈ −0.28 s < bar −0.9 s. **KHÔNG WIRE** — `gpt2_backward` giữ vòng serial; 5 mirror gpt2_ref + AQ revert nguyên vẹn | **0 (không làm)** | giữ **10.4×** |
| **P6** ❌ **GĐ12 ĐO BÁC BỎ 2026-10-22** | **LN_BWD dùng saves mu/rstd từ forward** (kế hoạch §13.3 — helper `gpt2_layernorm_saved`, saves append CUỐI format 3+8N+2, len guard, wire thử 5 mirror) — **đo bác bỏ, revert toàn bộ**. (1) **Tiền đề "bitwise bằng" của plan SAI**: saves `Tensor` chỉ chứa `list[f32]` nhưng mu/rstd tính bằng **f64** → lưu vào saves bị làm tròn f32 (probe CF: mu lệch 6.79e-10, rstd 2.52e-08) → **MAXDIFF toàn bộ grad = 3.17e-08 (không phải 0.0)** → loss in ra **đổi chữ số cuối 13.442912 → 13.442913** (bench A/B 2 binary cùng phiên, deterministic) → vi phạm cả 2 điều kiện §13.2 (maxdiff 0.0 HOẶC loss khớp exact). (2) **Lợi ích nhỏ hơn nhiều**: LN_BWD ×25 chỉ **656-703 → 578-594 ms (1.10-1.22×, −62..−125 ms)** thay vì −215 ms như ước — 2 pass bị bỏ là 2 pass **RẺ nhất** (chỉ đọc x, cache-hot); 2 pass còn lại (s1/s2 + dx) nặng hơn nhiều. Muốn bit-exact phải lưu f64 (Tensor không chứa được) → điều kiện sống lại: compiler hỗ trợ f64 container / typed f32 locals (LN tính f32 toàn bộ như torch) — gắn hướng P7b/native | **0 (không làm)** | giữ **10.4×** |
| **P7** ❌ **ĐO BÁC BỎ 2026-10-22** | **Mixed precision (bf16/tf32/f16): ĐO TRƯỚC rồi BỎ — probe `tv/tests/simd_probe_bx.tkv`.** Lý do cứng: (1) máy = **Xeon E5-2695 v3 (Haswell-EP)** → **chỉ AVX2+FMA3**, không AVX512/VNNI/**BF16**/AMX; (2) native lib `tkv_simd_native.c` **chỉ kernel F32/F64** (`SgemmF32`/`SgemmF64`) — grep `f16\|bf16\|half` = **0 hit**; (3) compiler (`compiler_src`) **không có kiểu/intrinsic f16/bf16** — grep = **0 hit** ⇒ muốn f16 chỉ còn 1 đường: **làm tròn f16 bằng PHẦN MỀM** (scalar .tkv `native_f16_encode/decode`). **Số học:** f16-round vào GEMM 256×768×768 → **maxdiff 0.0497** (max\|C32\|=46.206 vs max\|C16\|=46.202 → relerr **1.08e-3**; check chéo scalar C[0][0] 4.0e-8) ⇒ **vượt cửa sổ parity 2.4e-7 gấp 2.1e5×** ⇒ không thể giữ 135/135 gate fp32. **Tốc độ:** chi phí làm tròn f16 phần mềm = **378-458 ns/ptu** (3 lần chạy; baseline đọc list 19-20 ns/ptu) ⇒ **ngay cả khi GEMM f16 MIỄN PHÍ** thì AMP best-case = **0.056-0.065× (768³) / 0.047-0.056× (1024×768×768) / 0.096-0.108× (lm-head 1024×768×50257)** = **chậm 9-21×**. ⇒ **P7 chỉ sống lại khi có kernel f16/bf16 NATIVE** (viết C/Rust thêm vào `tkv_simd_native`) **+ CPU hỗ trợ** (AVX512-BF16/AMX) hoặc GPU — không phải việc làm được bằng .tkv. | **0 (không làm)** | giữ **12.4×** |

**Song song — seq8 riêng:** hiện 18.7×; threads ít giúp (GEMM quá nhỏ, overhead interpreter chiếm chủ yếu) → cần specialise batch nhỏ: unroll, fuse loop, giảm vật liệu hoá trung gian. Theo dõi KPI seq8 riêng, không hòa lẫn seq1024.

**Quyết định cần chủ khoa / compiler team:**

- ⓐ **tkvc có primitive thread không?** → **ĐÃ TRẢ LỜI 2026-10-21: CÓ** — `thread_spawn(fn[, 1 arg])`/`thread_join` (Task<T>, không GIL); record field chỉ nhận scalar/container/class-cùng-file (phải truyền `list[f32]`, KHÔNG truyền được `Tensor` qua record); native `simd_gemm` có thread pool riêng nhưng `tkv_run_mt` **không có lock** → chỉ an toàn khi 1 thread gọi hoặc n < 96 (`gemm_rows` trực tiếp, bp thread-local). P6 GĐ1/GĐ2 đã song song `attention_bwd` + `attention` FWD theo head (row P6): hợp đồng số giữ nguyên bằng **own-list + merge** (tích lũy từng phần tử giữ nguyên thứ tự).
- ⓑ **Parity so ở precision nào?** fp32-vs-fp32 (fair fight) hay tkv-bf16 vs torch-fp32 (torch CPU fp32 không dùng AMX — tkv bf16+VNNI/AVX512 có thể thắng, nhưng phải sửa numerics-contract + thêm gate AMP riêng). → **ĐÃ TRẢ LỜI 2026-10-22: fp32-vs-fp32 — GIỮ NGUYÊN contract 2.4e-7.** Lý do (đo, không phải lý thuyết): máy này **không có phần cứng f16/bf16** ⇒ mixed precision chỉ còn đường làm tròn f16 **bằng phần mềm** ⇒ lỗi số học **1.08e-3** (×2.1e5 cửa sổ parity) + chi phí làm tròn **378-458 ns/ptu** ⇒ AMP best-case **0.047-0.108×**. Chi tiết: dòng P7 + `HANDOFF.md` §0.27.
- ⓒ **"Nhanh hơn" cụ thể bao nhiêu, giữ bao nhiêu gate?** Đề xuất: ≤0.8× full step cả 3 seq, 84/84 fp32 giữ nguyên + gate AMP mới (nếu ⓑ = mixed).

**Rủi ro riêng P6/P7:** xem §8 (blocked threads, thermal drift, contract precision).

---

## 13. KẾ HOẠCH CÁC CƠ HỘI CÒN LẠI SAU GĐ10 — RÀ SOÁT TOÀN BỘ ĐƯỜNG TRAIN (2026-10-22)

**Mục tiêu tài liệu này:** liệt kê MỌI cơ hội còn lại theo số đo, xếp hạng theo lợi nhuận × độ tin cậy, chốt trình tự + tiêu chí cho các GĐ kế tiếp. Số nền (sau GĐ10): **full step seq8/256/1024 = 12.516/17.313/36.703 s** vs torch 0.934/1.404/3.542 = **13.4×/12.3×/10.4×**, gate **135/135 + `trainer_tests` 15/15**, loss khớp exact 7 chữ số cả 3 seq.

### 13.1 Phân rã bước train @seq1024 — nền của mọi ưu tiên

Probe AQ chạy lại **2026-10-22** (sau GĐ8/GĐ10, `out/audit_aq.log`) + ADAMW/UNFLATTEN theo bench:

| Pha | ms | % | So PyTorch | Cơ hội còn lại trong `.tkv` |
|---|---|---|---|---|
| `ATTN_BWD` ×12 | 8844 | 24% | GEMM AVX2 native | **thấp** — chỉ còn P7b (kernel C) |
| `MLP_BWD` ×12 | 7953 | 22% | đã bỏ transpose, overlap gelu | **thấp** |
| `LHEAD_BWD` | 7281 | 20% | vocab GEMM wave-parallel | **softmax-grad 1.33 s còn serial → GĐ11** |
| `FWD_BODY` (embed+12blk+lnf) | 6656 | 18% | đã parallel gelu/norm/loss + attn/heads | thấp (GEMM chiếm chủ đạo) |
| `ADAMW_STEP` | 3930 | 11% | torch: f32 + AVX2 + multi-tensor-apply; ta: f64-ish interpreter | **trung bình — GĐ13 (lớn nhất còn lại)** |
| `LN_BWD` ×25 | 765 | 2% | torch lưu rstd từ fwd; ta tính lại mu/v | **có — GĐ12 (nhỏ, chắc chắn)** |
| `EMBED_BWD` | 829 | 2% | scatter chỉ số trùng | thấp (BQ đã bust PAR) |
| `UNFLATTEN` | 734 | 2% | memory-bound ≈3.4 GB/s (GĐ9 đo) | xong |
| `GRAD_NORM` (đã PAR 2.79×) | ~375 | 1% | — | xong |

**Đọc số:**
1. `GRAD_ACC` ~1232 ms trong profile bench là **artifact** — đường thật `gpt2_train_step` truyền `gflat=opt.g` vào backward → grad ghi tại chỗ, không có vòng acc riêng.
2. `RESIDUAL` = SUM(32719) − TOTAL_bwd(22281) = **−10438 ms là artifact đo** (probe AQ gọi library tuần tự không gflat; bench ghi thẳng + overlap worker) — không phải pha âm. `LN_FWD 391` + `GELU 1125/797` nằm bên trong các mục trên.
3. **3 bucket GEMM (ATTN+MLP+LHEAD) = 24.1 s = 66% bước** — native AVX2, `.tkv` không thêm được gì → chỉ còn P7b.

### 13.2 Nguyên tắc chung (bắt buộc cho mọi GĐ dưới đây)

- **Đo trước, quyết theo số** — mọi thứ mới phải có probe; đồng hồ `time_perf` (`now_ms` lượng tử 15.6 ms đã làm vô hiệu probe BY).
- **Bit-exact**: maxdiff 0.0 với đường cũ, hoặc loss khớp exact 7 chữ số; saves **append CUỐI + len guard** (saves cũ vẫn đọc được — tiền lệ GĐ7).
- **Gate mỗi GĐ**: 135/135 + `trainer_tests` 15/15 + bench A/B 3 seq best-of-3 cùng phiên (ghi drift nhiệt).
- **Wire đồng bộ mirror** (gpt2_ref ×5, trainer ×4) — bench import `tv.training.gpt2_ref124` + `tv.training.trainer`.
- Ghi kết quả: dòng §12 ladder + `HANDOFF.md` §0.x + **commit từng GĐ**.

### 13.3 Các hạng mục — xếp theo lợi nhuận × độ tin cậy

#### GĐ11 — Song song softmax-grad trong lm-head bwd (~ −1.0 s @1024) — LÀM TRƯỚC

- **Hiện trạng:** vòng serial trong `gpt2_backward` simd==1 (`gg = t_zeros([n,V])` rồi `for i in range(n)`: max + exp + chuẩn hoá + onehot): probe AV **1.33 s** @seq1024 (51.5M ptu; code không đổi từ P5.7) + zero-fill `gg` 206 MB ≈ 60 ms. Softmax-grad ≈ 18% `LHEAD_BWD` hiện tại.
- **Thiết kế:** row-parallel 12 worker theo **mẫu `_ls_one` GĐ4 — đã thắng 5.10× cho CÙNG math ở forward**: worker nhận (logits, gg, ids, n, rlo, rhi), tự tính rows `[rlo,rhi)` **ghi thẳng vào `gg.data` phân đoạn (row-disjoint → không merge → bit-exact từng bit)**; guard `n*V ≥ 4M` (seq8 = 402K → serial nguyên vẹn). Tuỳ đo: worker tự zero seg, bỏ t_zeros serial.
- **Wire:** 5 mirror `gpt2_ref` (bench: `gpt2_ref124`).
- **KẾT QUẢ ĐO — ĐO BÁC BỎ, KHÔNG WIRE (2026-10-22):** wire thử 12 worker
  row-parallel ghi thẳng `gg.data` (bit-exact 0.0 — chk AQ giữ nguyên
  `0.000105296414`) → AQ LHEAD_BWD **7407 → 10719 ms (+3.3 s)**, bench seq1024
  38391 ms (drift) — ngược chiều kỳ vọng. Probe `simd_probe_ce.tkv` (2 lần
  chạy, @51.5M ptu, maxdiff 0.0): serial 1110-1140 · SHARED ghi+exp 3110-4453
  (**0.25-0.36×**) · CP copy shared **không** exp 484-547 (**2.0-2.3×**) · OW
  own-seg worker 484 (**2.36×**) + merge 1172 + alloc 359-375 = **2031
  (0.56×)**. Kết luận: (1) shared-write thuần không bệnh (CP 2.0-2.3×) — bệnh
  xuất hiện khi có **phep toan per-ptu + ghi** (CÙNG pattern AdamW BL
  0.36×/0.29×); (2) own-seg + merge (mẫu GĐ4) cũng chết ở đây vì **merge
  đơn luồng 51.5M ≈ 1172 ms ≈ toàn bộ serial** — ceiling kể cả merge miễn phí
  ≈ −0.28 s < bar −0.9 s. → ĐÓNG GĐ11; `gpt2_backward` giữ vòng serial.
- **Thắng nếu:** `LHEAD_BWD` trong AQ giảm ≥0.9 s + net win 3 seq + gate xanh.
- **Rủi ro:** thấp (mẫu đã thắng); đề phòng bài học GĐ4: main quá lớn → BadImageFormat → tách hàm; nested `list[f32]` append → ArrayTypeMismatch → ghi index.

#### GĐ12 — LN_BWD dùng saves mu/rstd từ forward (~ −0.25 s @1024) — CHẮC CHẮN NHẤT

- **Hiện trạng:** `gpt2_layernorm_bwd` tính lại `mu` (pass 1) + `v`/`rstd` (pass 2) từ `x` — **2/4 pass quét**; 25 gói = **765 ms** @1024. GĐ7 đã lưu `a_ln`/`m_ln` (bỏ 24 gói LN-fwd recompute) nhưng **chưa mu/rstd**; torch lưu `rstd` từ forward.
- **Thiết kế:**
  1. Helper `gpt2_layernorm_saved(x,g,b) → [out, mu, rstd]` (mu/rstd = list f32 `n` ptu ≈ 8 KB/gói @1024) — `gpt2_layernorm` cũ giữ nguyên cho parity.
  2. `gpt2_block_saved` trả thêm mu/rstd của ln1+ln2, `gpt2_forward_saved` thêm của lnf — **append CUỐI saves (format 3+6N), len guard** → saves cũ (3+4N…) vẫn chạy.
  3. `gpt2_layernorm_bwd` nhận `save_mu/save_rstd` optional (default `None`): có → bỏ 2 pass đầu; `gpt2_block_bwd`/`gpt2_backward` truyền tiếp; `saves == None` → đường recompute cũ nguyên vẹn.
- **Bit-exact:** mu/rstd do **cùng hàm + cùng input + cùng thứ tự tích lũy** với bản recompute → bitwise bằng → downstream không đổi (xác minh maxdiff 0.0 + loss exact).
- **Wire:** 5 mirror `gpt2_ref`. **Thắng nếu:** `LN_BWD` 765 → ≤550 ms (−≥30%), gate xanh.
- **KẾT QUẢ ĐO — ĐO BÁC BỎ, KHÔNG WIRE (2026-10-22):** wire thử đầy đủ (5
  mirror, saves format 3+8N+2 append cuối, len guard) rồi đo bằng probe
  `simd_probe_cf.tkv` @124M: LN_BWD ×25 **656-703 → 578-594 ms (chỉ 1.10-1.22×,
  −62..−125 ms)** — thấp hơn nhiều mức thắng −215 ms; **so học fail điều kiện
  §13.2**: saves `Tensor` = `list[f32]` trong khi mu/rstd tính f64 → làm tròn
  f32 (mu 6.79e-10, rstd 2.52e-08) → maxdiff grad toàn bộ **3.17e-08 ≠ 0.0**,
  loss in ra **13.442912 → 13.442913** (A/B 2 binary cùng phiên). → revert 5
  mirror; probe CF giữ làm bằng chứng. Điều kiện sống lại: compiler f64
  container / typed f32 locals.
- **Không làm mặc định:** row-parallel LN_BWD — mỗi gói 786K < ngưỡng break-even 2M (GĐ4 đo) → nhiều khả năng bust; chỉ làm thêm nếu probe cho net win.

#### GĐ13 — AdamW parallel LẦN 3: probe chẩn đoán TRƯỚC, wire CHỈ nếu thắng (−2.4..2.7 s nếu thành)

- **Hiện trạng:** `ADAMW_STEP` **3930 ms (11%)** — 124M ptu, ~25 phép/ptu interpreter + ~3.5 GB traffic (đọc g,m,v + đọc/ghi p,m,v; 4 mảng × 496 MB) → **sàn băng thông ≈ 1.0-1.1 s** (mức 3.4 GB/s đo ở GĐ9). Công thức ĐÃ bắt kịp torch (GĐ10: 1 div/ptu). Đã bust 2 lần: **BL SHARED-12 = 0.36× (11.9 vs 4.2 s), OWN+MERGE = 0.29×** (merge 372M = 11.7 s); GĐ9 + scan IL mới nhất **loại giả thuyết boxing** (`step` đã `int32 'i'`, `i/k` worker native int64) → **nguyên nhân chưa rõ**. Đây là mục `.tkv` lớn nhất còn lại.
- **Probe chẩn đoán (`simd_probe_cc`, N=124M, `time_perf`, ≥3 lần/variant):**

  | V | Variant | Câu trả lời |
  |---|---|---|
  | V0 | serial baseline (công thức GĐ10) | mốc |
  | V1 | compute-only PAR12, ghi OWN segs | compute có scale? (BL: 3.9→0.5 s — xác nhận lại) |
  | V2 | SHARED: đọc g,m,v shared, ghi p,m,v shared phân đoạn | đường GHI shared có phải thủ phạm? |
  | V3 | OWN+MERGE (như BL) | merge còn 11.7 s không |
  | V4 | control copy PAR12 @124M (BM: 3.0×) | shared-write@124M có bình thường không — chạy CÙNG phiên |

- **Quy tắc ra quyết định:**
  - V1 ≥2× mà V2 ≈1× hoặc chậm hơn → bệnh ở ghi SHARED → probe V5 tách dần (p-only shared, m/v own…) → tìm pattern thắng → wire `trainer` ×4.
  - V1 ≈1× → compute không scale trong worker → **ĐÓNG vĩnh viễn**, ghi §12 dòng bác bỏ.
  - V4 <2× (lệch BM) → vấn đề hệ thống (NUMA/hyperthread/drift) → điều chỉnh rồi đo lại trước khi kết luận.
- **Bit-exact:** từng phần tử độc lập (`p[i],m[i],v[i]` chỉ phụ thuộc `i`) → thứ tự tùy ý đều exact; OWN+MERGE gộp theo thứ tự segment.
- **Kỳ vọng nếu thắng:** 3930 → 1.2-1.5 s (sàn băng thông) = **−2.4..2.7 s/bước (−6.5-7.5%)**.
- **KHÔNG làm lại:** fused step+unflatten (P1.2/AH bust), giảm phép mới (GĐ10 V1/V2 đã đo âm).

#### P7b — Đo khoảng cách GEMM vs torch rồi chốt đường parity (probe thuần, 0 thay đổi code)

- **Bối cảnh:** ATTN+MLP+LHEAD = **24.1 s = 66%** bước, được ghi là “near-peak AVX2” — nhưng torch chạy **cả bước** trong 3.542 s → mâu thuẫn: nếu GEMM thật near torch thì full step phải ≪ 36.7 s. Khẳng định “near-peak” phải kiểm lại ở mức **hệ thống**.
- **Probe (`simd_probe_cd`, cùng phiên):**
  1. `simd_gemm` native trên 3 shape đại diện (attn T=64, proj/mlp 768→3072, lm-head 1024×768×50257) → GFLOPS thật + **in số thread THẬT của native pool** (câu hỏi AO còn treo: `TKV_GEMM_MIN_N=96` → tile 64 có đang chạy 1 thread?).
  2. `torch.matmul` f32 cùng shape, 14 thread, cùng máy cùng phiên.
  3. Roofline `2·m·n·k / time`, đối chiếu sàn ~190 GFLOP/s của torch.
- **Kịch bản ra quyết định:**
  - Ta ≥70% torch/shape → khoảng cách nằm NGOÀI kernel (dispatch interpreter giữa các GEMM, alloc buffer, merge copy) → attack driver/alloc trong `.tkv`.
  - Ta 10-30% torch → **kernel/threading là nút thắt** → đề xuất compiler/native team: C source có sẵn tại `D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c` + `build.ps1` (thuộc repo compiler, cần phối hợp). Phương án: thread cho tile nhỏ, f32 everywhere, prefetch/register-block; **kernel fused AdamW/elementwise native** — step native ≈ sàn băng thông ~1.0-1.1 s mà KHÔNG cần thread (giải pháp thay thế GĐ13 nếu GĐ13 bust).
- **Ý nghĩa:** GĐ11-13 chỉ đưa ~36.7 → **~32.7-35.5 s ≈ 9-10×** — mục tiêu 1.0× SỐNG hay CHẾT ở item này. P7b là probe quyết định, không phải tối ưu.

### 13.4 Bảng tổng hợp kỳ vọng @seq1024

| Item | Tiết kiệm | Loại | Độ tin cậy |
|---|---|---|---|
| ~~GĐ11 softmax-grad PAR12~~ **ĐÃ ĐO BÁC BỎ 2026-10-22** (probe CE: SHARED 0.25-0.36×, OWN+MERGE 0.56× — xem §12 dòng GĐ11) | 0 | — | đã đóng |
| ~~GĐ12 LN saves mu/rstd~~ **ĐÃ ĐO BÁC BỎ 2026-10-22** (probe CF: LN_BWD chỉ −62..−125 ms; mu/rstd lưu f32 → maxdiff grad 3.17e-08, loss đổi chữ số cuối — xem §12 dòng GĐ12) | 0 | — | đã đóng |
| GĐ13 AdamW parallel | −2.4..2.7 s (hoặc 0) | `.tkv` | **thấp** — 2 lần bust, cần probe chẩn đoán |
| **Tổng — GĐ13 thắng** | −2.4..2.7 s → **~34.0-34.3 s = 9.6×** | | |
| **Tổng — GĐ13 bust** | 0 → **~36.7 s = 10.4×** (giữ nguyên) | | |
| P7b | 0 (probe) | điều hướng parity | — |

**Kết luận trung thực:** trần của mọi việc làm được bằng `.tkv` sau GĐ10 ≈ **9-10×**; còn ~9× nằm ở 3 bucket GEMM 24.1 s (+ sàn băng thông các vòng elementwise) → **parity 1.0× bắt buộc đi qua native/C hoặc compiler** (f32 locals, fused kernel) — P7b điều hướng.

### 13.5 Trình tự thực hiện + gates

1. **GĐ11** ✅ **ĐÃ ĐO — BÁC BỎ, KHÔNG WIRE** (2026-10-22): wire thử → AQ
   LHEAD +3.3 s → probe `simd_probe_ce` tách nguyên nhân → revert 5 mirror +
   AQ, ghi §12 + HANDOFF §0.32. Tiến tiếp GĐ12.
2. **GĐ12** ✅ **ĐÃ ĐO — BÁC BỎ, KHÔNG WIRE** (2026-10-22): wire thử → probe
   `simd_probe_cf`: LN_BWD chỉ 1.10-1.22× + vi phạm bit-exact/loss-exact →
   revert 5 mirror, ghi §12 + HANDOFF §0.33. Tiến tiếp GĐ13.
3. **GĐ13 probe `simd_probe_cc`** → ra quyết định (wire `trainer` ×4 **hoặc** đóng + ghi dòng bác bỏ §12) → gates/docs/commit.
4. **P7b probe `simd_probe_cd`** → báo cáo số + đề xuất (quyết định cần chủ khoa/compiler team).

Mỗi bước giữ nguyên toàn bộ gate hiện có; **không GĐ nào được wire khi chưa net win trên cả 3 seq**.

### 13.6 ĐÃ BÁC BỎ — không làm lại (tra nhanh)

| Hướng | Lý do (probe) |
|---|---|
| f32 typed locals (`x: 'f32' = 0.0`) | compiler từ chối — `simd_probe_aa` |
| AdamW f32 toàn bộ (giống torch) | chặn ở compiler, cùng nguyên nhân trên |
| Fused step+unflatten | P1.2/AH: mọi hình thức chậm hơn 2 vòng riêng |
| Mixed precision f16/bf16 | P7/BX: relerr 1.08e-3 + 378-460 ns/ptu → 0.047-0.108× |
| int64-narrow ngoài GEMM | GĐ9/CA: memory-bound ≈3.4 GB/s, arithmetic bị ẩn (0.05%) |
| dwte-add / embed PAR12 | BQ: PAR12+merge 719 > serial 328 ms |
| main-thread + 11 worker | BW: eff 5.59× < spawn12 6.48× |
| UNFLATTEN parallel/fused | memory-bound + AH/BL bust |
| Lưu softmax (exp) từ forward làm saves | hơn GĐ11 ~0.2 s nhưng +206 MB + sửa forward → không đáng |
| transpose ggT / blocked / `simd_add` tích lũy | P5.7 / AW / P5.8b đã đo bust |
| Song song softmax-grad lm-head (GĐ11) | CE: SHARED ghi+exp 0.25-0.36× (cùng bệnh AdamW BL), OWN+MERGE 0.56× (merge 1172 ms ≈ serial) |
| LN_BWD saves mu/rstd (GĐ12) — kiểu container hiện tại | CF: chỉ 1.10-1.22×; Tensor=list[f32] không chứa nổi mu/rstd f64 → maxdiff grad 3.17e-08 + loss đổi chữ số cuối. Sống lại nếu compiler có f64 container / typed f32 locals |
