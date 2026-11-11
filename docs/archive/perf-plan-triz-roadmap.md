# TRIZ OPTIMIZATION ROADMAP — từ "thua PyTorch" sang "khác được"

**Ngày:** 2026-10-20 · **Baseline:** P5.3 consolidated (commit `f85ea3d`) — tkv `simd=1` (P5.1+P5.2+P5.3) vs PyTorch 2.14.0+cpu (14 threads)
**Máy:** Xeon E5-2695 v3 14C · **Cùng ngày, cùng máy, cùng token/config** cho mọi phép đo.
**Mục tiêu (CẬP NHẬT 2026-10-20, chủ khoa):** seq1024 full step phải **BẰNG rồi NHANH HƠN PyTorch** (≤3.542 s = 1.0×, rồi ≤0.8×) trên **cả 3 seq**, vẫn giữ 84/84 parity assertions. Mốc dọc đường (đã đạt): 154.234 s (43.5×) → 141.219 (P5.4) → **118.641 s (33.5×, P5.5)**. Ladder + phân tích sàn đơn/đa luồng: **§12** (mục ~20× bên dưới bị thay thế).

Chi tiết số đo gốc: `docs/archive/perf-plan-simd-vs-pytorch.md` (mục P5.2 + P5.3). Bàn giao: `docs/HANDOFF.md` mục 0.13 + mục 7.

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
| 15 | Báo cáo KPI + quyết định P6 | Cập nhật file này + `docs/HANDOFF.md` |

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
| `docs/archive/perf-plan-triz-roadmap.md` | File này — roadmap + KPI |
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
| `tv/tests/simd_probe_cc.tkv` | P6 GĐ13: probe chẩn đoán AdamW parallel lần 3 @124M (V0 serial / V1 OWN-seg / V2 SHARED+readback / V3 OWN+merge / V4 copy / **V5 REGISTERS**) — **kết quả: V5 (toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back) vẫn BUST 0.33× ≈ V2 0.30× → ĐÓNG GĐ13** (bệnh = phep toan + ghi list CHUNG, không phải read-back; xem §12 dòng GĐ13) |
| `tv/tests/simd_probe_cd.tkv` | **P7b ĐÃ ĐO**: tkv `simd_gemm` vs `torch.matmul` cùng shape cùng phiên — kernel tkv **chậm 6-14×** torch (standalone `simd_probe_cd` tái xác nhận: 93/968/1734/1562 ms vs 6.5/134/234/248; wrapper PAR12 chỉ ~1.25× → khoảng cách nằm trong KERNEL, không phải driver → chốt parity phải đi qua native/C (§13.3 P7b) — **~~chốt này ĐẢO NGƯỢC bởi probe CG (§14.5): wrapper-bound, kernel đã parity~~** |
| `tv/tests/simd_probe_cg.tkv` | **M3 ĐÃ ĐO 2026-10-23** — phân rã wrapper 4 shape P7b thành ALLOC/BUILD/GEMM/ACC (mode 0) + 3 phép cpu/wall qua `tools/probe_threads.py` (GEMM ratio **22.64**, BUILD ratio **1.00**, FULL ratio **3.97**) + re-entrancy (mode 4): 2 thread gọi `simd_gemm` T=192 đồng thời → **HANG > 90 s** (`tkv_run_mt` dùng `g_job` global không lock — đúng cảnh báo simd_matmul:419/BH:11). **CHỐT: GEMM phase 567-589 GFLOP/s pad ≈ torch; BUILD+ACC+ALLOC = 76-89% wrapper (đơn luồng) + pad 1.5× → wrapper-bound, KHÔNG phải kernel (§14.5)** |
| `tv/tests/simd_probe_ch.tkv` | **M3a ĐÃ ĐO 2026-10-23 — BÁC BỎ** — pipeline double-buffer build(w+1)∥replay(w) cho `_smm_core_pd` (worker tự derive i0/p0/j0, 2 set 48 buffer, handle reassign theo wave le/chan): **BIT-EXACT 0.0 cả 4 shape** nhưng speedup S3 1.02× / S4 0.94× / S5 1.10× + alloc 48 buffer 187 ms → dưới ngưỡng wire 1.3×. PD đã song song build → pipeline chỉ ẩn build(w+1); bottleneck thật = replay tuần tự (simd_gemm re-entrancy) → **kỳ vọng 1.6-1.7× đảo ngược** (§14.5, HANDOFF §0.38) |
| `tv/tests/simd_probe_cb.tkv` | **M3b ĐÃ ĐO 2026-10-23 — BÁC BỎ** — T-dispatch grid T=256/384/512/768/1024 trên 4 shape P7b (ALLOC/BUILD/GEMM/ACC/SUM + FULL_prod tham chiếu): **T=256 (pad 1.0×) tệ nhất mọi shape** (kernel simd_gemm mất throughput ở tile nhỏ — 282 GFLOP/s @256 vs 546-616 @768); T=1024 (pad 1.36×) thắng SUM cả 4 nhưng chỉ 1.18-1.33× (S3 750 vs 953) — quá mỏng cho ngưỡng wire 1.3×, phải sửa 3 đường code + wave-12 96 MB buffer, wrapper thật chưa đo → **KHÔNG wire** (§14.6, HANDOFF §0.39) |

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
| **P6** ❌ **GĐ13 ĐO BÁC BỎ 2026-10-22** | **AdamW parallel LẦN 3 — probe chẩn đoán `simd_probe_cc` TRƯỚC rồi ĐÓNG** (kế hoạch §13.3; 2 lần bust trước: BL 0.36×/0.29×, hypothesis từ CE/CF). @N=124M, `time_perf`, best-of, **bit-check subset 12M**: **V0 serial 3297 ms (0.0266 ns/ptu)** · **V1 OWN-seg (ceiling compute) 1266 ms = 2.60×** · **V2 SHARED+readback 10984 ms = 0.30× ❌** · **V4 copy shared (control) 1297 ms = 2.54×** · **V5 REGISTERS (toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back) 9937 ms = 0.33× ❌** · **V3 OWN+serial merge 2766 ms = 1.19×** (worker 1891 + merge 875). **CHỐT NGUYÊN NHÂN:** V5 ≈ V2 → **read-back KHÔNG phải thủ phạm**; bệnh thật là **phep toan per-ptu + ghi list CHUNG** (cùng pattern CE SHARED, BL SHARED). Worker chỉ scale khi ghi **seg RIENG** (V1 2.60×); đưa kết quả về list chung bằng **merge đơn luồng** (V3 1.19×) hay **ghi thẳng shared** (V5 0.33×) đều **ăn hết lợi thế** — không có hình thái AdamW song song nào **net win** → **không wire**, `trainer.step` giữ serial. Bit: V2-vs-V0 dp=dm=dv **0.0** (bit-exact); V5-vs-V0 dm=dv 0.0, dp **9.5e-07** (V5 không round f32 trước khi tính p → lệch 1 ULP, vô hại vì bust). **Ghi thêm:** ước −2.4..2.7 s (nếu thắng) → **0**; giữ **10.4×**; AdamW 3930 ms giờ là **native-kernel candidate duy nhất còn lại** (sàn băng thông ~1.0-1.1 s, không cần thread — xem §13.3 GĐ13 + hướng P7b/native) |
| **P6** ❌ **GĐ11 ĐO BÁC BỎ 2026-10-22** | **Song song softmax-grad lm-head bwd** (kế hoạch §13.3 — mẫu `_ls_one` GĐ4) — wire thử 12 worker row-parallel ghi thẳng `gg.data` phân đoạn (bit-exact, chk AQ `0.000105296414` giữ nguyên) rồi **đo bác bỏ**: AQ LHEAD_BWD **7407 → 10719 ms (+3.3 s)**. Probe `simd_probe_ce.tkv` tách nguyên nhân @51.5M ptu (2 lần chạy, maxdiff 0.0 cả SH/OW): serial **1110-1140**; **SHARED ghi+exp = 3110-4453 (0.25-0.36×)** — CÙNG bệnh AdamW BL/SHARED; **CP copy shared KHÔNG exp = 484-547 (2.0-2.3×)** → shared write thuần KHÔNG bệnh, bệnh là **phep toan + ghi list tuong minh**; **OW own-seg: worker 484 (2.36×) + merge 1172 + alloc 359-375 = TOTAL 2031 (0.56×)** — merge phiên bản đơn luồng (~22.7 ns/ptu) ăn hết lợi thế, ceiling kể cả merge miễn phí cũng chỉ ≈ −0.28 s < bar −0.9 s. **KHÔNG WIRE** — `gpt2_backward` giữ vòng serial; 5 mirror gpt2_ref + AQ revert nguyên vẹn | **0 (không làm)** | giữ **10.4×** |
| **P6** ❌ **GĐ12 ĐO BÁC BỎ 2026-10-22** | **LN_BWD dùng saves mu/rstd từ forward** (kế hoạch §13.3 — helper `gpt2_layernorm_saved`, saves append CUỐI format 3+8N+2, len guard, wire thử 5 mirror) — **đo bác bỏ, revert toàn bộ**. (1) **Tiền đề "bitwise bằng" của plan SAI**: saves `Tensor` chỉ chứa `list[f32]` nhưng mu/rstd tính bằng **f64** → lưu vào saves bị làm tròn f32 (probe CF: mu lệch 6.79e-10, rstd 2.52e-08) → **MAXDIFF toàn bộ grad = 3.17e-08 (không phải 0.0)** → loss in ra **đổi chữ số cuối 13.442912 → 13.442913** (bench A/B 2 binary cùng phiên, deterministic) → vi phạm cả 2 điều kiện §13.2 (maxdiff 0.0 HOẶC loss khớp exact). (2) **Lợi ích nhỏ hơn nhiều**: LN_BWD ×25 chỉ **656-703 → 578-594 ms (1.10-1.22×, −62..−125 ms)** thay vì −215 ms như ước — 2 pass bị bỏ là 2 pass **RẺ nhất** (chỉ đọc x, cache-hot); 2 pass còn lại (s1/s2 + dx) nặng hơn nhiều. Muốn bit-exact phải lưu f64 (Tensor không chứa được) → điều kiện sống lại: compiler hỗ trợ f64 container / typed f32 locals (LN tính f32 toàn bộ như torch) — gắn hướng P7b/native | **0 (không làm)** | giữ **10.4×** |
| **P7** ❓ **P7b ĐÃ ĐO — CHỐT ĐƯỜNG PARITY 2026-10-22** | **Đo khoảng cách GEMM: tkv native vs torch** (probe thuần, 0 thay đổi code train — kế hoạch §13.3 P7b). rebuild `simd_probe_bh` + `tools/p7b_gemm_torch.py` **cùng phiên**, 3 shape đại diện, f32, best-of: S1 dx1 [1024,768,3072] tkv **94 ms** vs torch **7.1 ms** = **13.2×** · S3 dxf [1024,768,50257] **969** vs **129** = **7.5×** · S4 dwte [50257,768,1024] **1781** vs **254** = **7.0×** · S5 logits [1024,50257,768] **1547** vs **240** = **6.5×**. **CHỐT — rơi vào kịch bản "Ta 10-30% torch" (thực tế 8-15%): khoảng cách nằm ở KERNEL/threading, không phải driver/wrapper/alloc.** Bằng chứng: builder đã tối ưu GĐ3 (append→buffer 1.4-3.2×) + wave-12 lm-head PAR chỉ **1.22-1.29×** serial → overhead ngoài kernel ~20-25%, còn **~7-13×** là chính `simd_gemm`. Claim "near-peak AVX2" **SAI**: `simd_gemm` ~50-70 GFLOP/s vs torch MKL 300-700 GFLOP/s cùng shape. **KHÔNG wire được bằng `.tkv`** — việc còn lại là native/C (verify số thread pool, micro-kernel AVX2 register-block + prefetch, fused AdamW/elementwise ≈ sàn băng thông 1.0-1.1 s) → **parity 1.0× bắt buộc đi qua native/C, chốt hẳn** | **0 (probe thuần)** | giữ **10.4×** (trần `.tkv`) |
| **P7** ❌ **ĐO BÁC BỎ 2026-10-22** | **Mixed precision (bf16/tf32/f16): ĐO TRƯỚC rồi BỎ — probe `tv/tests/simd_probe_bx.tkv`.** Lý do cứng: (1) máy = **Xeon E5-2695 v3 (Haswell-EP)** → **chỉ AVX2+FMA3**, không AVX512/VNNI/**BF16**/AMX; (2) native lib `tkv_simd_native.c` **chỉ kernel F32/F64** (`SgemmF32`/`SgemmF64`) — grep `f16\|bf16\|half` = **0 hit**; (3) compiler (`compiler_src`) **không có kiểu/intrinsic f16/bf16** — grep = **0 hit** ⇒ muốn f16 chỉ còn 1 đường: **làm tròn f16 bằng PHẦN MỀM** (scalar .tkv `native_f16_encode/decode`). **Số học:** f16-round vào GEMM 256×768×768 → **maxdiff 0.0497** (max\|C32\|=46.206 vs max\|C16\|=46.202 → relerr **1.08e-3**; check chéo scalar C[0][0] 4.0e-8) ⇒ **vượt cửa sổ parity 2.4e-7 gấp 2.1e5×** ⇒ không thể giữ 135/135 gate fp32. **Tốc độ:** chi phí làm tròn f16 phần mềm = **378-458 ns/ptu** (3 lần chạy; baseline đọc list 19-20 ns/ptu) ⇒ **ngay cả khi GEMM f16 MIỄN PHÍ** thì AMP best-case = **0.056-0.065× (768³) / 0.047-0.056× (1024×768×768) / 0.096-0.108× (lm-head 1024×768×50257)** = **chậm 9-21×**. ⇒ **P7 chỉ sống lại khi có kernel f16/bf16 NATIVE** (viết C/Rust thêm vào `tkv_simd_native`) **+ CPU hỗ trợ** (AVX512-BF16/AMX) hoặc GPU — không phải việc làm được bằng .tkv. | **0 (không làm)** | giữ **12.4×** |

**Song song — seq8 riêng:** hiện 18.7×; threads ít giúp (GEMM quá nhỏ, overhead interpreter chiếm chủ yếu) → cần specialise batch nhỏ: unroll, fuse loop, giảm vật liệu hoá trung gian. Theo dõi KPI seq8 riêng, không hòa lẫn seq1024.

**Quyết định cần chủ khoa / compiler team:**

- ⓐ **tkvc có primitive thread không?** → **ĐÃ TRẢ LỜI 2026-10-21: CÓ** — `thread_spawn(fn[, 1 arg])`/`thread_join` (Task<T>, không GIL); record field chỉ nhận scalar/container/class-cùng-file (phải truyền `list[f32]`, KHÔNG truyền được `Tensor` qua record); native `simd_gemm` có thread pool riêng nhưng `tkv_run_mt` **không có lock** → chỉ an toàn khi 1 thread gọi hoặc n < 96 (`gemm_rows` trực tiếp, bp thread-local). P6 GĐ1/GĐ2 đã song song `attention_bwd` + `attention` FWD theo head (row P6): hợp đồng số giữ nguyên bằng **own-list + merge** (tích lũy từng phần tử giữ nguyên thứ tự).
- ⓑ **Parity so ở precision nào?** fp32-vs-fp32 (fair fight) hay tkv-bf16 vs torch-fp32 (torch CPU fp32 không dùng AMX — tkv bf16+VNNI/AVX512 có thể thắng, nhưng phải sửa numerics-contract + thêm gate AMP riêng). → **ĐÃ TRẢ LỜI 2026-10-22: fp32-vs-fp32 — GIỮ NGUYÊN contract 2.4e-7.** Lý do (đo, không phải lý thuyết): máy này **không có phần cứng f16/bf16** ⇒ mixed precision chỉ còn đường làm tròn f16 **bằng phần mềm** ⇒ lỗi số học **1.08e-3** (×2.1e5 cửa sổ parity) + chi phí làm tròn **378-458 ns/ptu** ⇒ AMP best-case **0.047-0.108×**. Chi tiết: dòng P7 + `docs/HANDOFF.md` §0.27.
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
- Ghi kết quả: dòng §12 ladder + `docs/HANDOFF.md` §0.x + **commit từng GĐ**.

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

- **KẾT QUẢ ĐO — ĐO BÁC BỎ, ĐÓNG GĐ13 (2026-10-22):** probe `simd_probe_cc.tkv` chạy đủ 6
  biến thể @N=124M, `time_perf`, best-of, bit-check subset 12M:

  | V | Variant | ms | so serial | Bit vs V0 |
  |---|---|---|---|---|
  | V0 | serial (công thức GĐ10) | 3297 | 1.00× | — |
  | V1 | OWN-seg compute-only (ceiling) | 1266 | **2.60×** | n/a (không merge) |
  | V2 | SHARED + read-back | 10984 | **0.30×** ❌ | dp=dm=dv **0.0** |
  | V4 | copy shared (control) | 1297 | **2.54×** | — |
  | V5 | **REGISTERS** (toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back) | 9937 | **0.33×** ❌ | dm=dv 0.0, dp 9.5e-07 |
  | V3 | OWN + serial merge | 2766 | **1.19×** | exact theo segment |

  **Chốt nguyên nhân (lật giả thuyết CE/CF):** V5 ≈ V2 → **read-back KHÔNG phải thủ phạm**;
  bệnh thật là **phep toan per-ptu + ghi list CHUNG** (cùng pattern CE SHARED, BL SHARED).
  Worker chỉ scale khi ghi **seg RIENG** (V1 2.60×); đưa kết quả về list chung bằng **merge
  đơn luồng** (V3 1.19× — merge 875 ms ăn mất phần lớn) hay **ghi thẳng shared** (V5 0.33×)
  đều **ăn hết lợi thế**. **V4 copy shared 2.54× → không phải lỗi hệ thống/NUMA/drift** (V4 ≥2×
  theo đúng quy tắc ra quyết định) → kết luận vững. **KHÔNG wire** — `trainer.step` giữ serial.
  Ước −2.4..2.7 s (nếu thắng) → **0**; giữ **10.4×**. Hướng còn lại cho AdamW 3930 ms: **kernel
  fused native** (sàn băng thông ~1.0-1.1 s, không cần thread — mục con của P7b).

#### P7b — Đo khoảng cách GEMM vs torch rồi chốt đường parity (probe thuần, 0 thay đổi code)

- **Bối cảnh:** ATTN+MLP+LHEAD = **24.1 s = 66%** bước, được ghi là “near-peak AVX2” — nhưng torch chạy **cả bước** trong 3.542 s → mâu thuẫn: nếu GEMM thật near torch thì full step phải ≪ 36.7 s. Khẳng định “near-peak” phải kiểm lại ở mức **hệ thống**.
- **Probe (`simd_probe_cd`, cùng phiên):**
  1. `simd_gemm` native trên 3 shape đại diện (attn T=64, proj/mlp 768→3072, lm-head 1024×768×50257) → GFLOPS thật + **in số thread THẬT của native pool** (câu hỏi AO còn treo: `TKV_GEMM_MIN_N=96` → tile 64 có đang chạy 1 thread?).
  2. `torch.matmul` f32 cùng shape, 14 thread, cùng máy cùng phiên.
  3. Roofline `2·m·n·k / time`, đối chiếu sàn ~190 GFLOP/s của torch.
- **Kịch bản ra quyết định:**
  - Ta ≥70% torch/shape → khoảng cách nằm NGOÀI kernel (dispatch interpreter giữa các GEMM, alloc buffer, merge copy) → attack driver/alloc trong `.tkv`.
  - Ta 10-30% torch → **kernel/threading là nút thắt** → đề xuất compiler/native team: C source có sẵn tại `D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c` + `build.ps1` (thuộc repo compiler, cần phối hợp). Phương án: thread cho tile nhỏ, f32 everywhere, prefetch/register-block; **kernel fused AdamW/elementwise native** — step native ≈ sàn băng thông ~1.0-1.1 s mà KHÔNG cần thread (giải pháp thay thế GĐ13 nếu GĐ13 bust).
- **KẾT QUẢ ĐO (2026-10-22, cùng phiên — tkv `simd_probe_bh` rebuild + `tools/p7b_gemm_torch.py`; tái xác nhận standalone `simd_probe_cd` + cùng script torch):** 3 shape đại diện (cùng `m n k`, f32, best-of, `simd_gemm` native qua wrapper thật):
  | Shape (m×n×k) | tkv native (best) | torch.matmul (14T) | tỉ lệ |
  |---|---|---|---|
  | S1 dx1 [1024,768,3072] | 94 ms | 7.1 ms | **13.2×** |
  | S3 dxf [1024,768,50257] | 969 ms | 129 ms | **7.5×** |
  | S4 dwte [50257,768,1024] | 1781 ms | 254 ms | **7.0×** |
  | S5 logits [1024,50257,768] | 1547 ms | 240 ms | **6.5×** |

  **CHỐT — rơi vào kịch bản "Ta 10-30% torch" (thực tế 8-15%):** khoảng cách nằm ở **KERNEL/threading**, không phải driver/alloc/wrapper. Bằng chứng: builder đã được GĐ3 tối ưu (append→buffer, 1.4-3.2×), wave-12 lm-head PAR chỉ **1.22-1.29×** so serial (probe BH S3/S4/S5 LIB vs PAR) — tức overhead ngoài kernel chỉ ~20-25%, còn **~7-13×** là chính `simd_gemm`. Claim "near-peak AVX2" **SAI** — `simd_gemm` đạt ~45-82 GFLOP/s trong khi torch MKL đạt 320-740 GFLOP/s cùng shape (probe CD standalone: 52.0/81.7/45.6/50.6 vs 741/588/337/319 GFLOPS). Lý do có thể: native pool 1 luồng cho tile nhỏ (`TKV_GEMM_MIN_N=96`), không packing/AVX-register-block như MKL. ~~Hoặc thread pool không thực sự parallel trên Haswell~~ **ĐÃ BÁC BỎ 2026-10-22 (xác nhận theo yêu cầu): P3 đo cpu/wall = 21.20× @768³ (539 GFLOP/s, ~21 luồng) — pool PARALLEL THẬT và đạt đỉnh ngang MKL ở shape vuông lớn; vấn đề là shape/tiling/khai thác ở shape thật, xem §14 mục M3.** **CHỈNH LẠI 2026-10-23 (probe CG, §14.5): phần "khoảng cách nằm ở KERNEL/threading" BỊ ĐẢO NGƯỢC — phân rã 4 pha cho thấy BUILD+ACC+ALLOC = 76-89% thời gian wrapper (BUILD đo ratio cpu/wall 1.00 = đơn luồng) + pad 1.5×; bản thân pha GEMM chạy 567-589 GFLOP/s (pad) — S4/S5 GEMM 207/211 ms còn THẮNG torch 254/240 ms → kernel ĐÃ parity, thủ phạm chính là wrapper.**
- **Tái xác nhận standalone (2026-10-22, `simd_probe_cd` + `tools/p7b_gemm_torch.py` cùng phiên, best-of-3):** S1 93 vs 6.5 ms = **14.3×** · S3 968 vs 134.4 = **7.2×** · S4 1734 vs 234.3 = **7.4×** · S5 1562 vs 247.5 = **6.3×** → chốt khoảng cách **6-14×**, cùng kết luận (bảng trên là probe BH best-of LIB/SER/PAR).
- **Đề xuất (cần compiler/native team — không làm được bằng `.tkv`):** C source tại `tools/simd_native/tkv_simd_native.c` — (1) verify số thread THẬT của pool, (2) micro-kernel f32 AVX2 register-block + prefetch, (3) parallel tile nhỏ; **kernel fused AdamW/elementwise native** ≈ sàn băng thông ~1.0-1.1 s (thay GĐ13 bust). Đây là đường DUY NHẤT tới parity 1.0×.
- **Ý nghĩa:** GĐ11-13 chỉ đưa ~36.7 → **~32.7-35.5 s ≈ 9-10×** — mục tiêu 1.0× SỐNG hay CHẾT ở item này. P7b là probe quyết định, không phải tối ưu.

### 13.4 Bảng tổng hợp kỳ vọng @seq1024

| Item | Tiết kiệm | Loại | Độ tin cậy |
|---|---|---|---|
| ~~GĐ11 softmax-grad PAR12~~ **ĐÃ ĐO BÁC BỎ 2026-10-22** (probe CE: SHARED 0.25-0.36×, OWN+MERGE 0.56× — xem §12 dòng GĐ11) | 0 | — | đã đóng |
| ~~GĐ12 LN saves mu/rstd~~ **ĐÃ ĐO BÁC BỎ 2026-10-22** (probe CF: LN_BWD chỉ −62..−125 ms; mu/rstd lưu f32 → maxdiff grad 3.17e-08, loss đổi chữ số cuối — xem §12 dòng GĐ12) | 0 | — | đã đóng |
| ~~GĐ13 AdamW parallel~~ **ĐÃ ĐO BÁC BỎ 2026-10-22** (probe CC: V5 REGISTERS 0.33× ≈ V2 0.30× → bệnh = phep toan + ghi list CHUNG, không phải read-back; xem §12 dòng GĐ13) | 0 | `.tkv` | **đã đóng** |
| **Tổng — GĐ13 thắng** | −2.4..2.7 s → **~34.0-34.3 s = 9.6×** | | |
| **Tổng — GĐ13 bust** | 0 → **~36.7 s = 10.4×** (giữ nguyên) | | |
| **P7b ĐÃ ĐO — parity PHẢI qua native/C (2026-10-22)** | 0 (probe) | kernel tkv chậm **6-14×** torch → `.tkv` trần ~10.4×, không chạm 1.0× được | **đã chốt: native/C bắt buộc** |

**Kết luận trung thực:** trần của mọi việc làm được bằng `.tkv` sau GĐ10 ≈ **9-10×**; còn ~9× nằm ở 3 bucket GEMM 24.1 s (+ sàn băng thông các vòng elementwise) → **parity 1.0× bắt buộc đi qua native/C hoặc compiler** (f32 locals, fused kernel) — P7b điều hướng.

### 13.5 Trình tự thực hiện + gates

1. **GĐ11** ✅ **ĐÃ ĐO — BÁC BỎ, KHÔNG WIRE** (2026-10-22): wire thử → AQ
   LHEAD +3.3 s → probe `simd_probe_ce` tách nguyên nhân → revert 5 mirror +
   AQ, ghi §12 + HANDOFF §0.32. Tiến tiếp GĐ12.
2. **GĐ12** ✅ **ĐÃ ĐO — BÁC BỎ, KHÔNG WIRE** (2026-10-22): wire thử → probe
   `simd_probe_cf`: LN_BWD chỉ 1.10-1.22× + vi phạm bit-exact/loss-exact →
   revert 5 mirror, ghi §12 + HANDOFF §0.33. Tiến tiếp GĐ13.
3. **GĐ13 probe `simd_probe_cc`** ✅ **ĐÃ ĐO — BÁC BỎ, ĐÓNG** (2026-10-22): V5 REGISTERS (không read-back) vẫn 0.33× ≈ V2 0.30× → nguyên nhân = phep toan + ghi list CHUNG; không wire, ghi §12 dòng GĐ13 + §13.3 + §13.6.
4. **P7b** ✅ **ĐÃ ĐO — CHỐT ĐƯỜNG PARITY** (2026-10-22): rebuild `simd_probe_bh` + `tools/p7b_gemm_torch.py` cùng phiên → tkv native **chậm 6-14×** torch (bảng §13.3; standalone `simd_probe_cd` tái xác nhận 6.3-14.3×) → khoảng cách ở KERNEL, không phải driver → ghi §12 dòng P7b + §13.3 KẾT QUẢ + HANDOFF §0.35. Việc còn lại (kernel GEMM + fused AdamW native/C) là của compiler/native team — quyết định cần chủ khoa/compiler team, KHÔNG làm được bằng `.tkv`.

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
| "GEMM near-peak AVX2" → attack driver/alloc ngoài kernel | P7b: kernel `simd_gemm` chậm 6-14× torch (wrapper PAR12 chỉ ~1.25×) → nút thắt nằm TRONG kernel/threading, không phải dispatch/alloc — `.tkv` không sửa được |
| AdamW parallel (GĐ13) — mọi hình thái bằng `.tkv` | CC: V5 REGISTERS (toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back) **0.33×** ≈ V2 SHARED+readback 0.30× → read-back không phải thủ phạm; bệnh = phep toan + ghi list CHUNG. V1 OWN-seg 2.60× nhưng V3 merge serial 1.19× / V5 shared 0.33× đều ăn hết lợi thế. Sống lại chỉ bằng **kernel fused native** (~1.0-1.1 s sàn băng thông, không cần thread) |

---

---

## 14. TRIZ — MENU GIẢI PHÁM PARITY (2026-10-22, phân tích — CHƯA COMMIT WIRE)

Sau P7b (§13.3) mâu thuẫn đã **đổi cấp**: không còn "tkv thua vì plumbing/driver" mà là **kernel tkv không đạt được mức tối ưu mà chính nó + resource sẵn có trong máy đã đạt ở nơi khác**. Phân tích lại bằng TRIZ (§2 là khung cũ cho P5.x; khung này cho level native).

> **XÁC NHẬN ĐA LUỒNG (2026-10-22, theo yêu cầu — tra lại số đo đã có):** TokenVector **CÓ đa luồng**, bằng chứng 3 lớp: (1) primitive `.tkv` `thread_spawn/join` (Task<T>, không GIL; overhead spawn+join ×100 = **0.0 ms**, probe BH[0]) — đang chạy thật trong trainer `_tn_one` (12 worker), `simd_matmul` wave-12 `_pd_one`, attention fwd/bwd per-head; (2) native `simd_gemm` có **thread pool riêng ~21 luồng** — P3 (`simd_probe_ao` + `tools/probe_threads.py`, đo cpu/wall ratio): 768³ × 800 = wall 1.397 s / cpu 29.63 s → **21.20×, 539 GFLOP/s**; nhưng 64³ → ratio **0.99 = 1 luồng** (`TKV_GEMM_MIN_N=96`, tile < 96 đi `gemm_rows` serial); (3) các bài song song đo được speedup thật: attention_bwd own-list **5.2×** (GĐ1), worker eff **8.66×** (GĐ5/BN), gelu_fwd **7.33×** / gelu_bwd **5.03×** (GĐ4/BM), wave-12 lm-head **3.2×** (BH S3/S4). **Hệ quả:** giả thuyết "pool không parallel thật" (§13.3) **BỊ BÁC BỎ**; nút thắt là (a) các *hình thái* song song `.tkv` ghi list CHUNG/merge (BL/CE/CC bust — nguyên nhân đã biết), và (b) **shape thật chỉ ăn 6-33% khả năng đã đo của chính pool** — đó là mục M3 bên dưới.

### 14.1 Phát biểu mâu thuẫn

**Mâu thuẫn kỹ thuật:**

| # | Muốn tăng | Bị hy sinh (ràng buộc) |
|---|---|---|
| TC1 | FLOPS `simd_gemm` ở các shape thật (45-82 GFLOP/s) | tính bit-exact/deterministic audit được bằng `.tkv`; không thêm dependency ngoài |
| TC2 | Khai thác nốt 12C/28T (pool đã chứng minh 21 luồng @768³) | các hình thái song song `.tkv` đã đo bust (BL/CE/CC) — ghi CHUNG + merge nuốt lợi thế |
| TC3 | f32 everywhere như torch (băng thông ×2) | hợp đồng numerics hiện tại: tích lũy f64 + 84/84 parity |

**Mâu thuẫn vật lý (giải bằng separation principles):**

- **PC1:** kernel phải **đơn giản/deterministic** VÀ phải **tối ưu triệt để** → tách theo *hệ thống*: phần tối ưu nằm ở native/C (compiler repo), phần điều phối giữ nguyên `.tkv` (giữ ABI `simd_gemm`).
- **PC2:** thread pool phải **phẳng/ổn định trên mọi shape** (như MKL) VÀ **chỉ chạy hết công suất ở tile vuông lớn** (21.2× @768³ vs 1.0× @64³, 1.23-3.2× ở shape thật) → tách theo *điều kiện*: shape nào được phép đi multi-thread/tile lớn (M3 sửa scheduling), shape nào đi serial — thay vì viết lại toàn bộ.
- **PC3:** số liệu phải **f32 như torch** VÀ **f64 như hợp đồng cũ** → tách theo *thời gian*: hợp đồng numerics là ràng buộc **của phiên này**; mục tiêu chủ khoa ("BẰNG torch") cho phép đổi sang numerics-torch nếu 84/84 theo tolerance vẫn pass.

### 14.2 IFR + tài nguyên (resources) đang bỏ không

**IFR (Ideal Final Result):** `simd_gemm` đạt ngang MKL **mà không phải viết một kernel mới nào** — vì năng lực đó đã được chứng minh ngay trong chính hệ thống.

9-windows: mỗi lần một level "hết" (P0 → P5.3 simd → GĐ3 builder) đều phải nhảy level để ăn tiếp. Level `.tkv` đã cạn ở **10.4×**; level kế tiếp là native — nhưng **bằng chứng P3 cho thấy không nhất thiết phải viết lại từ đầu**.

Resource chưa dùng (xếp theo đòn bẩy):
1. **Đỉnh đã đo của chính `simd_gemm`: 539 GFLOP/s @768³ (21.2 luồng, P3)** — lớn hơn torch trên shape tương đương (320-740). Khoảng cách 6-14× ở shape thật là **khai thác sai/đánh rơi**, không phải thiếu năng lực.
2. **oneMKL đã nằm trong máy** — torch 2.14.0+cpu gọi MKL. Gọi đúng thứ torch dùng = "bằng torch" cả numerics lẫn tốc độ (M1, phương án B).
3. **Weights bất biến trong 1 step** (wte/w1/w2/qkv) → prepack 1 lần được (prior action #10).
4. **gg/xf dùng lại cho 3 GEMM lm-head** → tài nguyên đọc-trong-kernel (merge #5).

### 14.3 MENU (xếp theo ideality — lợi ích / chi phí / rủi ro)

| # | TRIZ | Giải pháp | Lợi ích kỳ vọng | Chi phí / rủi ro |
|---|---|---|---|---|
| **M3** ✅ **ĐÃ ĐO 2026-10-23 → §14.5** | #2 Extraction + resources | **Điều tra shape-gap: vì sao shape thật chỉ 45-82 GFLOP/s trong khi chính pool đạt 539 @768³ (P3)?** Lặp lại phép đo P3 (cpu/wall `tools/probe_threads.py`) trên 4 shape P7b; giả thuyết cần kiểm: T/k-tiling với k=50257/3072, luồng acc per k-tile, wave 12 chỉ dựng tile mà worker không gọi `simd_gemm` (BH: worker KHÔNG simd_gemm), `TKV_GEMM_MIN_N=96` chặn S1 | Nếu đúng (đánh rơi scheduling/tiling): GEMM 24.1 → **~4-6 s** ở **đỉnh đã đo sẵn**, **không cần kernel mới, không cần BLAS** — step ~36.7 → **~16-18 s ≈ 2.0-2.3×** | **1-2 ngày probe — LÀM ĐẦU TIÊN, đòn bẩy cao nhất/chi phí thấp nhất** |
| **M1** ❌ **ĐÃ ĐO BÁC BỎ 2026-10-23 → §14.9** | **#28 Mechanics substitution** | **Gọi BLAS sẵn có** (OpenBLAS `cblas_sgemm` BSD, hoặc oneMKL) từ runtime native qua `dlopen` — **thay implementation trong `simd_gemm`, giữ nguyên ABI**, `.tkv` không đổi 1 dòng | ~~GEMM 24.1 s → ~3.5-4 s (ngang torch); numerics = numerics torch~~ **SỐ ĐO ĐẢO NGƯỢC CẢ 2 CHÂN:** GEMM phase 0.91-0.98× (14T) / 1.05-1.13× (28T) — **full step 1.001× noise** (34,734 vs 34,687 ms), hợp đồng tolerance GÃY (19/20, loss lệch 1.4e-4), numerics ≠ numerics torch (OpenBLAS scipy ≠ oneMKL torch) | C shim trong repo compiler; license OK (OpenBLAS); phải set số thread để không oversubscribe pool tkv; gate 84/84 theo tolerance. **Phương án B khi M3 không đóng được gap** |
| **M2** | #24 Asymmetry + #1 Segmentation | **Micro-kernel AVX2 BLIS-style** (8×8 register-block, macro MC/KC/NC, prefetch) thay phần cốt lõi `simd_gemm` | chuẩn hóa 45-82 → 150-300 GFLOP/s mọi shape | compiler team, ~2-4 tuần; **phương án C cuối cùng** (P3 chứng minh kernel không yếu về gốc) |
| **M4** | #10 Prior action + #1 Segmentation | **Prepack weights 1 lần** trước train loop (`simd_gemm_packed`): wte/w1/w2/qkv bất biến trong step | bỏ packing khỏi timed path; ~20-25% wrapper còn lại biến mất | ABI mới, kèm M1/M3 |
| **M5** | #35 Parameter change | **Buffer f32 native + fused AdamW/elementwise kernel** (thay list[f64]) | Plumbing 3.93 s → sàn băng thông **~1.0-1.1 s** (GĐ13 V1 2.60× chứng minh compute đủ nhanh, bệnh chỉ là container) | Gate: AdamW f32 = numerics torch → loss-exact; kèm M1/M3 |
| **M6** | #5 Merging + #26 Nesting doll | **Fuse softmax-grad + dxf + dwte** lm-head trong 1 kernel (gg đọc 1 lần); fuse attention: softmax+gDs+transpose n² (16% attn bwd, §1) thành kernel épilogue | lm-head bwd 10.84 s → ~4-5 s; attn non-GEMM ~8 s → cắt sâu | native, sau M1/M3 |
| **M7** | #15 Asymmetry + #3 Local quality | **Dispatch kernel theo shape class**: nhỏ/square (S1 13.2×) vs lớn k=50257 (lm-head 6.5-7.5×) | khoảng cách ~13× ở shape nhỏ biến mất | trong M3/M2 |
| **M8** | #40 Nature (treo lại) | INT8 GEMM + fp32 acc | ~2× nữa (nếu M1-M7 chưa đủ) | **numerics đã đo bust** với f16/bf16 (P7) — chỉ mở lại nếu hợp đồng đổi |

**Đã bác bỏ từ TRIZ (không làm lại):** #21 Skipping/sparsity (fp32 dense — §13.6); mọi song song `.tkv` kiểu ghi CHUNG/merge (§13.6 — nguyên nhân đã biết, **KHÔNG phải thiếu đa luồng**: P3 21.2×, GĐ1 5.2×, GĐ4 7.33×); "attack driver/alloc ngoài kernel" (P7b); giả thuyết "pool không parallel thật" (**P3 bác bỏ**).

### 14.4 Thứ tự đề xuất + KPI gate

0. **M3 — điều tra shape-gap** ✅ **ĐÃ ĐO 2026-10-23 (probe CG, §14.5): nguyên nhân = BUILD/ACC/ALLOC ĐƠN LUỒNG (BUILD ratio 1.00) + pad 1.5×, KHÔNG phải kernel — GEMM phase đã 567-589 GFLOP/s pad ≈ torch, S4/S5 thắng torch tuyệt đối.** Chuyển thành **M3a song song ACC → M3b T-dispatch giết pad → M3c tái dùng buffer** (đều `.tkv`, gate §13.2). **M3a biến thể pipeline ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CH, §14.5): BIT=0.0 nhưng 0.94-1.10× — replay tuần tự là sàn cứng; **M3b T-dispatch ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CB, §14.6): T=256 (pad 1.0×) tệ nhất mọi shape, T=1024 chỉ 1.18-1.33× SUM thủ công — quá mỏng. Cửa sổ còn lại: **M3c (tái dùng buffer)** → ~~M3c~~ **✅ ĐÃ WIRE + 84/84 2026-10-23** (HANDOFF §0.41).
1. ~~**M1 dlopen OpenBLAS** giữ ABI `simd_gemm`~~ **❌ M1 ĐÃ ĐO BÁC BỎ 2026-10-23 (§14.9):** dựng được shim drop-in (MSVC + OpenBLAS scipy, `tools/simd_native/`), GEMM phase 0.91-1.13×, **full step 1.001× noise**, tolerance GÃY 19/20 (loss lệch 1.4e-4), numerics ≠ torch (OpenBLAS ≠ oneMKL). M2 cũng đã mất mục tiêu (CG chứng minh kernel đã parity — tiền đề M2 "45-82 GFLOP/s" bị đảo ngược).
2. **M4 + M5** (prepack + fused step native) → step tiếp về **~11-13 s ≈ 1.8×**.
3. **M6 + M7** (fused lm-head + attention épilogue + dispatch theo shape) → mục tiêu dải **1.1-1.3×**; stretch goal 1.0× đòi thêm fused attention-stage kernel.

**KPI gate chung cho mọi mục:** 4 shape P7b ≥ **250 GFLOP/s mỗi shape** (đỉnh đã đo 539 @768³ là tham chiếu); full step 3 seq không regress; 84/84 parity theo tolerance; numerics target = **numerics torch** (vì mục tiêu là "BẰNG torch" — cùng thư viện = cùng số).

### 14.5 M3 — ĐÃ ĐO (2026-10-23, probe CG): CHỐT WRAPPER-BOUND, KERNEL ĐÃ PARITY

*Mục tiêu (§14.3 M3): tìm nguyên nhân shape thật chỉ ăn 6-33% đỉnh đã đo 539 GFLOP/s. Probe thuần `tv/tests/simd_probe_cg.tkv`, **0 thay đổi code train**, 5 chế độ: (0) phân rã 4 pha trên đúng 4 shape P7b; (1) GEMM thuần tile dựng sẵn; (2) BUILD thuần; (3) FULL wrapper; (4) re-entrancy — modes 1-3 đo cpu/wall qua `tools/probe_threads.py`. Cùng phiên, cùng máy; SUM khớp FULL trong ±5-12% → phép đo tự nhất quán. Mọi số dưới đây = **1 lần gọi wrapper** (best-of).*

**Mode 0 — phân rã theo pha (ms):**

| Shape | ALLOC | BUILD | GEMM | ACC | SUM | FULL | GEMM GF (pad) | FULL GF (hữu ích) | GEMM vs torch |
|---|---|---|---|---|---|---|---|---|---|
| S1 dx1 [1024x768x3072] | 31 | 62 | 13.1 | 15 | 121.1 | 109 | 552 | 44.3 | 13.1 vs 7.1 ms (1.85×) |
| S3 dxf [1024x768x50257] | 16 | 468 | 203.3 | 234 | 921.3 | 969 | 588 | 81.6 | 203 vs 129 ms (1.57× ≈ pad 1.5) |
| S4 dwte [50257x768x1024] | 406 | 750 | 207 | 359 | 1722 | 1828 | 578 | 43.2 | 207 vs 254 ms — **THẮNG 1.23×** |
| S5 logits [1024x50257x768] | 422 | 562 | 211 | 234 | 1429 | 1563 | 567 | 50.6 | 211 vs 240 ms — **THẮNG 1.14×** |

**Modes 1-3 — tỉ lệ luồng (cpu/wall, `probe_threads.py`):**

| Pha đo | ratio (best) | Ý nghĩa |
|---|---|---|
| GEMM thuần, tile dựng sẵn (mode 1) | **22.64** (21.9-22.9) | pool chạy hết công suất ở T=768; per-pass 203-214 ms khớp mode 0 → tự nhất quán |
| BUILD thuần (mode 2) | **1.00** | tile-build **100% đơn luồng** (interpreted list write); tổng 4 shape = 1813 ms |
| FULL wrapper (mode 3) | **3.68-3.97** | BUILD+ACC+ALLOC đơn luồng làm loãng 22 luồng của pha GEMM |

**Mode 4 — re-entrancy (kiểm tra tài nguyên đang bỏ không của M2/M6):** serial 24 call T=192 = **187 ms**; 2 thread `thread_spawn`, mỗi thread gọi `simd_gemm` trên A/B/C **riêng rọ** (không chung phần tử) → **HANG > 90 s** (timeout giết, không ra thêm dòng nào sau stage=serial). Đúng cơ chế đã cảnh báo `simd_matmul.tkv:419` + BH:11-13: T≥96 đi `tkv_run_mt` dùng `g_job` **toàn cục không lock** → race dispatcher, biểu hiện là **deadlock** chứ không chỉ sai số. **Ràng buộc thiết kế cho mọi mục sau: GEMM song song phải đi qua MỘT entry point duy nhất (pool trong native); wave-12 giữ nguyên worker dựng tile + main gọi `simd_gemm` (đúng BH/PD hiện tại); M6 fused kernel cũng giữ 1 entry.**

**CHỐT — nguyên nhân shape-gap (trả lời M3):**

1. **Kernel ĐÃ parity với torch khi quy đổi cùng công suất:** pha GEMM chạy **552-588 GFLOP/s (pad)**; torch tính trên flop hữu ích: S1 683, S3 612, S4 311, S5 329 GFLOP/s. Số "45-82 GFLOP/s" trước đây (§13.3 P7b) = chia flop HỮU ÍCH cho thời gian WRAPPER gồm BUILD/ACC/ALLOC + 1.5× flop pad → **quy sai, quy hết cho kernel**.
2. **Pad 1.5×:** T=768 trên m=1024 → 2 tile pad lên 1536 (GFP/GF = 1.51 đúng hệ số pad). Chừa phần kernel S3: 79×1.51/0.588 = 202 ms ≈ 203 ms đo được — S3 thua torch 1.57× **chính là pad**, không phải micro-kernel.
3. **BUILD+ACC+ALLOC = 76-89% FULL** (S1 88%, S3 79%, S4 89%, S5 86%): BUILD 468-750 ms (2.2-3.6× pha GEMM), ACC 234-359 ms (1.1-1.7×), ALLOC 16-422 ms (S4/S5 ~400 ms) — BUILD đã đo ratio **1.00**.
4. **Tóm lại:** khoảng cách FULL 6.5-14× torch = **pad 1.5× × overhead wrapper 2.2-4.4×**, KHÔNG phải kernel yếu. Kết luận "khoảng cách nằm ở KERNEL/threading" (§13.3 CHỐT, HANDOFF §0.35) **bị đảo ngược một phần** — đã ghi chú chỉnh ở cả 2 nơi.

**Xếp lại menu (§14.3/§14.4 theo số đo này):**

- **M3 ✅ đóng** ở đây.
- **M1 (dlopen BLAS) + M2 (micro-kernel) hạ ưu tiên:** chỉ nhắm phần kernel-side còn ~1.2-1.6× (pad + arch) — không phải 7×. M1 giữ vai trò "cùng thư viện = cùng numerics torch" cho bước 1.0× cuối.
- **Mục mới (đều `.tkv`, KHÔNG cần native, gate §13.2):**
  - **M3a — song song ACC theo wave 12**: ❌ **biến thể pipeline double-buffer (§0.37) ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CH): BIT=0.0 nhưng speedup 0.94-1.10× (S3 1.02/S4 0.94/S5 1.10), alloc 48 buffer 187 ms — dưới ngưỡng wire 1.3×. PD đã song song build bằng 12 worker nên pipeline chỉ ẩn được build(w+1) (nhỏ); bottleneck thật = replay tuần tự 12×(simd_gemm+acc) trên main, simd_gemm T≥96 re-entrancy cấm song song → **kỳ vọng 1.6-1.7× bị số đo đảo ngược** (HANDOFF §0.38). Biến thể gốc "song song ACC tile rời rạc" **ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CW, §14.8): 2 thiết kế (E: acc∥gemm; A: 12 gemm trước → ACC wave-12) đều BIT=0.0 nhưng **0.67-0.99×** « ngưỡng 1.3 — simd_gemm đã dùng ~22 thread nên acc worker cướp tài nguyên của chính nó; S3 vẫn p-reduction chặn. — HANDOFF §0.42.
  - **M3b — T-dispatch giết pad 1.5×**: ❌ **ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CB, §14.6) — grid T=256/384/512/768/1024 trên 4 shape: T=256 (pad 1.0×) **tệ nhất mọi shape** (SUM 156/1781/2375/2375 ms — kernel mất throughput ở tile nhỏ: 282 GFLOP/s @256 vs 546-616 @768); T=1024 (pad 1.36×) thắng SUM cả 4 shape nhưng chỉ 1.18-1.33× — quá mỏng cho ngưỡng wire 1.3×, phải sửa 3 đường code + wave-12 96 MB buffer, wrapper thật chưa đo. **Bài học: "giết pad" chỉ đổi GEMM lấy BUILD+ACC; pad 1.5× giá rẻ hơn pad 1.0× ở wrapper này.** Cửa sổ còn lại: M3c tái dùng buffer.
  - **M3c — tái dùng buffer tile/C**: ✅ **ĐÃ WIRE + GATE 84/84 (2026-10-23)** (probe CI, §14.7 + HANDOFF §0.41) — t_zeros C S4/S5 tốn 359-421 ms/call, ZERO_FILL chỉ 63-94 ms (ratio 0.18-0.22 « 0.3). Wire bằng **scratch-passing** (KHÔNG module-var — tôn trọng lệnh cấm `simd_matmul.tkv:430`): caller truyền `list[f32]` optional (`sc5` logits fwd / `sc4` dwte bwd), `_smm_cd` chỉ mượn khi `len(sc) >= m*n` → caller cũ bit-identical. A/B wrapper thật (CU): BIT=0.0, ~172 ms/shape × 2 ≈ **344 ms/bước** (sửa ước lượng 623 của §14.7 — wave-parallel nhanh hơn probe serial). Đã nối 3 tầng `gpt2_ref`/`gpt2_train`/bench; gate 5+17+9+33+20 = **84/84**, full-step loss `13.442912` bit-identical.
- **Trần ước lượng (CHƯA ĐO — chỉ là phân tích):** ~~S3 FULL 938 → ~250-300 ms ≈ 3.1-3.7×~~ **đã bị số đo M3a (probe CH) hạ thấp** — M3a-pipeline bust cho thấy replay tuần tự là sàn cứng; ~~M3b (giết pad)~~ **cũng ĐÃ ĐO BÁC BỎ** (probe CB, §14.6). Cửa sổ thật còn lại: ~~M3c tái dùng buffer~~ **✅ ĐÃ WIRE 2026-10-23** (A/B wrapper BIT=0.0 ~344 ms/bước, gate 84/84 — HANDOFF §0.41) + ~~parallel-ACC biến thể gốc~~ **❌ ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CW, §14.8: 2 thiết kế 0.67-0.99× BIT=0.0 — HANDOFF §0.42). **Cửa sổ `.tkv` ĐÓNG — còn M1 OpenBLAS (native/C).** **Không hứa số bước tổng trước khi đo.**
- **KPI gate diễn giải lại:** "≥250 GFLOP/s mỗi shape" tính cho pha GEMM (pad) → **4/4 ĐÃ VƯỢT** (552-588); gate thật còn lại là FULL-vs-torch theo ms/shape + full step 3 seq + 84/84.

### 14.6 M3b — ĐÃ ĐO (2026-10-23, probe CB): T-DISPATCH BÁC BỎ — GIẢM PAD LUÔN LÀM CHẬM HƠN

*Mục tiêu (§14.5 M3b): chọn T theo shape class để tile chia hết, giết pad 1.5×. Probe thuần `tv/tests/simd_probe_cb.tkv` (0 thay đổi code train): grid T = 256/384/512/768/1024 trên 4 shape P7b, mỗi (shape, T) đo ALLOC/BUILD/GEMM(per-pass)/ACC → SUM; tham chiếu = full-wrapper production T=768 (best-of-2). Cùng phiên, cùng máy; CB1 FULL_prod khớp CG mode 3 trong ±3% → tự nhất quán.*

**Số đo (ms, 1 lần gọi wrapper):**

| Shape | T | ng | pad | ALLOC | BUILD | GEMM | ACC | SUM | GEMM GF (pad) |
|---|---|---|---|---|---|---|---|---|---|
| S1 dx1 | 256 | 144 | 1.00× | 15 | 78 | 16.0 | 47 | 156 | 302 |
| S1 dx1 | 512 | 24 | 1.33× | 15 | 47 | 15.8 | 15 | 92.8 | 409 |
| S1 dx1 | **768 (prod)** | 8 | 1.50× | 0 | 63 | 11.8 | 15 | **89.8** | 617 |
| S1 dx1 | 1024 | 3 | 1.33× | 0 | 47 | 7.8 | 16 | **70.8** | 831 |
| S3 dxf | 256 | 2364 | 1.00× | 15 | 782 | 281 | 703 | 1781 | 282 |
| S3 dxf | 384 | 786 | 1.13× | 0 | 625 | 234 | 454 | 1313 | 380 |
| S3 dxf | 512 | 396 | 1.34× | 15 | 547 | 219 | 344 | 1125 | 485 |
| S3 dxf | **768 (prod)** | 132 | 1.51× | 0 | 531 | 204 | 218 | **953** | 586 |
| S3 dxf | 1024 | 50 | 1.36× | 16 | 359 | 203 | 172 | **750** | 529 |
| S4 dwte | 256 | 2364 | 1.00× | 375 | 1031 | 282 | 687 | 2375 | 281 |
| S4 dwte | **768 (prod)** | 132 | 1.51× | 360 | 750 | 218 | 329 | **1657** | 549 |
| S4 dwte | 1024 | 50 | 1.36× | 375 | 641 | 203 | 188 | **1407** | 529 |
| S5 lgts | 256 | 2364 | 1.00× | 562 | 828 | 281 | 704 | 2375 | 282 |
| S5 lgts | **768 (prod)** | 132 | 1.51× | 422 | 562 | 219 | 235 | **1438** | 546 |
| S5 lgts | 1024 | 50 | 1.36× | 406 | 344 | 195 | 235 | **1180** | 551 |

(FULL_prod đo thật: S1 94, S3 968, S4 1703, S5 1500 ms — khớp SUM T=768 thủ công trong ±1-5%.)

**CHỐT — M3b BÁC BỎ:**

1. **Pad 1.0× (T=256) là TỆ nhất trên cả 4 shape** — SUM 156/1781/2375/2375 ms, chậm hơn production T=768 (89.8/953/1657/1438) từ 1.3× đến 2.0×. Giả định gốc "T=256 pad 1.0× nhưng calls ×18 — phải gate đo" đã được đo: **calls ×18 không phải vấn đề, mà là kernel mất throughput ở tile nhỏ**.
2. **Kernel throughput suy giảm mạnh khi T nhỏ** (GEMM GFLOP/s pad): 282 @256 → 380 @384 → 485 @512 → 546-616 @768 → 529-831 @1024. T=256 cho GEMM phase 281 ms cho S3 dù flop hữu ích chỉ 66% của T=768 (204 ms) — tức **hiệu suất kernel ở T=256 ≈ 0.47× của T=768**, bù không nổi phần BUILD+ACC tăng thêm.
3. **T=1024 (pad 1.36×) là T tốt nhất đo được** — SUM 70.8/750/1407/1180 ms vs T=768 thủ công 89.8/953/1657/1438 → **T=1024 thắng SUM trên cả 4 shape** (1.18-1.33×), NHUNG không đủ để wire: (a) biên lợi nhuận quá mỏng — lever chính S3 chỉ 1.26× (750 vs 953), cả 4 shape 1.18-1.33× đều quanh ngưỡng wire 1.3×, và đây mới là SUM thủ công chứ không phải wrapper thật; (b) **chưa tính chi phí wire**: `_smm_core`/`_p58_core_simple`/`_smm_core_pd` 3 đường đều hard-code `T = _imin(_tilek(k), _roundup64(_imin(m,n)))` — dispatch T=1024 phải sửa 3 chỗ, mỗi chỗ có wave-12 (pd) với 24 buffer T² (T=1024 → 4 MB/buffer, 24 buffer = 96 MB, alloc ~2× hiện tại); (c) full-wrapper T=1024 chưa đo thật (probe CB chỉ đo SUM thủ công tách pha — wrapper thật có thể lệch do in-context effect, như CG đã thấy RESID). **Kết luận trung thực: T=1024 là ứng viên tốt nhất nhưng là lever MỎNG — 1.18-1.33× ở SUM thủ công, sát ngưỡng wire 1.3×, đòi sửa 3 đường code + wave-12 96 MB buffer, mà wrapper thật chưa đo.** Theo §13.2 ("mọi lever phải đo wrapper thật trước khi wire"), KHÔNG wire trên số SUM thủ công — nhưng đây là lever DUY NHẤT còn mở của M3b, ghi nhận để đo lại đầy đủ nếu quay lại.
4. **Bài học (đảo ngược giả định §14.5):** "giết pad" bằng T nhỏ đổi lấy GEMM nhanh hơn nhưng BUILD+ACC tăng theo số tile (ng × T²) — và kernel simd_gemm không duy trì throughput ở T nhỏ. Pad 1.5× ở T=768 **rẻ hơn** pad 1.0× ở T=256 trong wrapper này. Kết luận: **không có lever T-dispatch an toàn trên grid đã đo**; nếu quay lại, chỉ số đo T=1024 full-wrapper thật (không phải SUM thủ công) mới quyết định. **[ĐÃ ĐO 2026-10-23 — probe CX: wrapper thật 1.01-1.12× « 1.3 → BUST — §14.10]**

**Quyết định:** KHÔNG wire. `_smm_core`/`_p58_core_simple`/`_smm_core_pd` giữ nguyên `T = _imin(_tilek(k), _roundup64(_imin(m,n)))`. Menu còn lại: ~~M3c tái dùng buffer tile/C~~ **✅ ĐÃ WIRE 2026-10-23** (HANDOFF §0.41) → ~~parallel-ACC biến thể gốc~~ **❌ ĐÃ ĐO BÁC BỎ 2026-10-23** (probe CW, §14.8) → M1 OpenBLAS.

**Bài học chung:** (1) T-dispatch không free — pad chỉ là 1 trong 3 thành phần (GEMM ∝ pad, BUILD ∝ ng×T², ACC ∝ ng×T²), và kernel throughput là hàm của T; (2) SUM thủ công ≠ full-wrapper — mọi wire phải đo wrapper thật; (3) probe CB xác nhận pattern "grid T + so với production" dùng lại được cho mục sau.

### 14.7 M3c — ĐÃ ĐO (2026-10-23, probe CI): RE-USE BUFFER CÓ LỜI — ✅ ĐÃ WIRE 2026-10-23 (HANDOFF §0.41, gate 84/84)

*Mục tiêu (§14.5 M3c): S4/S5 ALLOC ~400 ms/call biến mất nếu re-use C đã cấp phát. Probe thuần `tv/tests/simd_probe_ci.tkv` (0 thay đổi code train): đo (1) chi phí `t_zeros([m,n])` thật cho 2 shape lớn nhất và (2) chi phí zero-fill C đó (điều kiện bắt buộc của re-use — phải xóa C trước mỗi lần tích lũy). Best-of-3, cùng phiên.*

**Số đo (ms, best-of-3):**

| Shape (C) | ptu (m×n) | ALLOC (`t_zeros`) | ZERO_FILL | ratio zero/alloc |
|---|---|---|---|---|
| S5 logits [1024×50257] | 51,463,168 | **421** | **94** | **0.22** |
| S4 dwte [50257×768] | 38,597,376 | **359** | **63** | **0.18** |

**CHỐT — M3c CÓ LỜI (đảo ngược kết luận "bust-by-construction" chưa đo):**

1. **Zero-fill rẻ hơn alloc 4-5×** (ratio 0.18-0.22, « ngưỡng 0.3 "không lời" giả định). Nguyên nhân: `t_zeros` dựng C bằng vòng interpreted `append(0.0)` (capacity-doubling, ~8 ns/ptu), còn zero-fill ghi thẳng `cd[i] = z` trên List[Single] đã có capacity (~2 ns/ptu). Cùng m×n, cùng thứ tự thao tác trên list — chỉ khác append vs indexed write.
2. **Lợi nhuận tái tạo:** re-use C + zero-fill mỗi call cắt **(421−94) + (359−63) ≈ 623 ms/bước** trên đường PD lm-head (S5 forward `smm_dot_pd` + S4 backward `smm_fwd_at_pd`) — so với full-step ~36.7 s thì nhỏ (~1.7%), nhưng đây là lever `.tkv` **duy nhất còn đo được lời** sau khi M3a/M3b bust.
3. **Ràng buộc còn lại (chưa giải quyết — cần quyết định wire):** re-use đòi C persistent qua các call → **module-variable mutable state** — đúng thứ `simd_matmul.tkv:430` cấm ("không gán lại biến module", sinh từ 2 bug corruption). Đường an toàn hơn: caller truyền scratch-Tensor vào wrapper (nhưng 3 call site ở `gpt2_ref.tkv:497/1322/1334` đều gán C vào biến sống trên tape — không phải transient). Bất kỳ đường nào cũng phải qua **BIT=0.0 + gate 84/84 + full step 3 seq**.
4. **Kết luận lúc đo: CHƯA wire ngay** — lever đã đo CÓ LỜI nhưng vướng ràng buộc state; chờ quyết định cách giữ state (module-var có kiểm chứng vs scratch-passing) trước khi viết code. Nếu wire, ước tính trần ~−623 ms/bước (chưa trừ overhead dispatch).

**Cập nhật 2026-10-23 (sau khi đo): ĐÃ WIRE — HANDOFF §0.41.** Cách chọn: **scratch-passing** (`list[f32]` optional `sc5`/`sc4` do caller cấp phát persistent, không module-var → không đụng lệnh cấm `simd_matmul.tkv:430`); `_smm_cd` chỉ mượn khi `len(sc) >= m*n`, thiếu vẫn `t_zeros` → caller cũ bit-identical. Số đo CU trên wrapper thật: **BIT=0.0, ~172 ms/shape × 2 ≈ 344 ms/bước** (thanh hơn ước tính 623 của mục 2 — wave-parallel nhanh hơn probe serial). Gate 5+17+9+33+20 = **84/84**; full-step seq1024 loss `13.442912` bit-identical. Đã nối: `gpt2_ref` (124M + small V=256 + `gpt2_ref124`) 3 site, `gpt2_train` (`steps` cấp phát 2 buffer → mọi gate đi qua scratch), `simd_bench_124m_seq`.

**Bài học:** (1) lại một lần "argument vs measurement" — kết luận "bust-by-construction" tôi suy ra từ `t_zeros` + aliasing **bị số đo CI đảo ngược** (zero-fill 0.18-0.22 alloc); đúng kỷ luật §13.2: mọi lever phải ĐO, không được argue.; (2) append-0.0 interpreted là gốc mọi chi phí alloc lớn trong TKV — đã thấy ở BH (builder 1203 ms), nay ở C-tensor; (3) probe CI cực nhỏ (1 hàm zero-fill + t_zeros) — pattern "mổ xẻ 1 thành phần nghi vấn" rẻ, dùng lại được.

> Lưu ý trung thực: M1-M8 là **phân tích + ước lượng từ số đo đã có** (P3, P7b, BH, GĐ13), CHƯA có số đo mới — mọi bước đều phải qua probe/Gate trước khi wire, đúng kỷ luật §13.2. **Cập nhật 2026-10-23: M3a (CH), M3b (CB), M3c (CI) đã có số đo thật — xem §14.5/§14.6/§14.7.**

### 14.8 M3a BIEN THE GOC — SONG SONG ACC (probe CW): ĐÃ ĐO, BỨC CA 2 THIET KE (2026-10-23)

*Mục tiêu (§14.5 còn lại cuối cùng): song song hoá pha ACC (`_acc_f` interpreted 234-359 ms/call, §0.36) trên wrapper `_smm_core_pd` thật, cho tile rời rạc. Probe thuần `tv/tests/simd_probe_cw.tkv` (0 thay đổi code train), baseline = `smm_*_pd` production, best-of-3 cùng phiên, 3 shape wave (S3/S4/S5). Đo 2 thiết kế:*
- **E (hoist=0):** main `simd_gemm` vẫn TUYÊN TU (tôn re-entrancy — CG mode 4), spawn acc worker ngay sau mỗi slot; chống p-reduction bằng join-trước-spawn khi cùng (i0,j0); 12 Ct buffer riêng/slot; join toàn bộ cuối wave.
- **A (hoist=1, §0.36 «ACC theo wave 12»):** 12 gemm xong trước, rồi mới spawn 12 acc — acc không cùng tải gemm.

**Số đo (ms, best-of-3, run 2 — 2 thiết kế cùng phiên; run 1 E standalone 0.98/0.92/0.82× cùng kết luận):**

| Shape | base PD | E (acc∥gemm) | sp E | A-hoist | sp A | BIT |
|---|---|---|---|---|---|---|
| S3 dxf | 734 | 750 | 0.98× | 766 | 0.96× | **0.0** |
| S4 dwte | 1437 | 1453 | 0.99× | 1735 | 0.83× | **0.0** |
| S5 logits | 1219 | 1703 | 0.72× | 1829 | 0.67× | **0.0** |

alloc 12 Ct (T=768) = 46-47 ms/call (trong số variant).

**CHỐT:**
1. **BIT=0.0 cả 2 thiết kế lần đầu chạy** — chain-join theo khối + Ct riêng/slot đúng; pattern conditional-join của CH tái dùng (lần thứ 3 BIT=0.0 lần đầu).
2. **BUST 0.67-0.99× vs ≥1.3× — KHÔNG WIRE.** E: S5 (nhiều acc song song nhất, 12 khối rời rạc) lại thua nhiều nhất 0.72× → nguyên nhân = `simd_gemm` main đã dùng ~22 thread (CG mode 1 cpu/wall 22.64), acc worker cướp core/bandwidth của gemm — đúng bệnh oversubscription CH đã thấy (S4 0.94×). A: mất overlap, acc wave-12 không scale (interpreted loop memory-bound + 132 spawn/join ≈ 21 ms + 47 ms Ct) → 0.83/0.67×. S3 p-reduction (12 p-tile/cùng khối) không song song được — đúng dự đoán.
3. **ACC 234-359 ms là thật nhưng KHÔNG thể song song hoá có lãi trong wrapper này** — gemm đã bão hoà máy, acc không scale out. Sàn cấu trúc: kể cả acc miễn phí, trần S4 = 1828/(406+750+207) ≈ 1.34× và phải trả spawn+Ct — số đo cho thấy thật tế ÂM.
4. **Hệ quả menu: cửa sổ `.tkv` (M3a/M3b/M3c) ĐÓNG tại đây.** Còn lại: M1 OpenBLAS (native/C — ngoài `.tkv`) → M2 micro-kernel → M4-M7. Chi tiết: HANDOFF §0.42, log `out/cw_run.log`.

### 14.9 M1 — GOI OPENBLAS QUA DROP-IN SHIM (2026-10-23): ĐÃ ĐO BÁC BỎ — 1.001× FULL STEP + MẤT TOLERANCE

*Mục tiêu (§14.3/§14.4 M1): thay implementation bên trong `simd_gemm` bằng BLAS sẵn có, giữ nguyên ABI, `.tkv` không đổi 1 dòng — đóng "cùng thư viện = cùng numerics torch" + residual kernel-side. Đo trong phiên thật, không argue (§13.2).*

**Cơ chế dựng được (repo-local, không cần compiler team):** exe `.tkv` chỉ P/Invoke đúng 2 export của `tkv_simd_native.dll` (`SgemmF32`/`SgemmF64`, vuông n×n row-major beta=0 — xác minh qua IL sinh ra `out/*.il`; 13 `simd_*` elementwise là managed IL, không đụng DLL). ⇒ shim drop-in cùng tên DLL là đủ, 0 dòng `.tkv`. Đã có MSVC cl thật (VS 18) + OpenBLAS LP64 trong scipy (`scipy.libs`, export `scipy_cblas_sgemm`).

**Shim:** `tools/simd_native/tkv_simd_native_m1.c` + `build_m1.ps1` → `out/m1/tkv_simd_native.dll` (re-export 2 symbol, `LoadLibrary` `m1blas.dll` = OpenBLAS scipy, forward `cblas_sgemm/dgemm`, thread = env `TKV_M1_THREADS` mặc định 14, máy 28 luồng). Load OK, chạy được cả 5 gate.

**Số đo (cùng phiên, best-of; chi tiết HANDOFF §0.43):**

| Phep do | base (DLL goc) | shim 14T | shim 28T |
|---|---|---|---|
| GEMM phase thuần (CG mode 1, 4 shape) | 547-567 GF pad | **0.91-0.98×** | **1.05-1.13×** (594-612 GF) |
| FULL wrapper (CG mode 0, 4 shape) | S1 93 / S3 938 / S4 1797 / S5 1578 | — | S1 94 / S3 968 / S4 1797 / S5 1562 (±3-6%, nhiễu) |
| **Full step 3 seq (best-of-3 xen kẽ)** | **34,734 ms**, loss 13.442912 | **34,687 ms = 1.001×**, loss 13.442902 | 37,593 ms, loss 13.442896 |
| Gate tolerance | 84/84 | **19/20** — "train 3 buoc loss buoc 3" 1.407e-4 > 1e-4 (bất biến 1/14/28 thread) | same |

**CHỐT — BUST theo gate của chính nó (§14.4):**
1. **"GEMM 24.1 s → 3.5-4 s" không xảy ra** — "24.1 s" đã bị CG re-attributed thành wrapper (build+acc+alloc 76-89% + pad 1.5×); thay kernel chỉ đổi pha GEMM ~200 ms/shape-call → **full step biến mất trong nhiễu (1.001×)**. Không oversubscribe (28T > 14T thắng đều ở phase thuần; 22T tệ nhất — quan sát chưa giải thích, ghi nhận).
2. **Tolerance GÃY:** OpenBLAS accumulation ≠ `tkv_simd_native` → loss lệch 1.4e-4, 1 case gate fail → vi phạm "84/84 theo tolerance".
3. **"Numerics = numerics torch" THẤT BẠI cả nguyên lý:** shim nạp OpenBLAS scipy = cùng nhóm np-BLAS (bit-diff 1609/2048), torch = oneMKL (1618) — 2 thứ tự tich luy KHÁC NHAU, đã ghi ở `docs/numerics-contract.md`; không có đường "gọi đúng thứ torch dùng" mà không kèm oneMKL proprietary.
4. **KHÔNG wire** — production giữ `tkv_simd_native.dll` gốc.

**Hệ quả menu — chương trình đo ĐÃI HẾT ở mức repo-local:** M2 mất mục tiêu (CG chứng minh kernel đã parity — tiền đề "45-82 GFLOP/s" bị đảo ngược); M4/M5/M6/M7 đều đòi **builtin/ABI mới trong compiler** (`simd_gemm_packed`, fused AdamW/elementwise/lm-head kernels) — `.tkv` không khai báo được pinvoke mới → chặn trên compiler team. Còn đúng 1 mục `.tkv` treo có điều kiện: **T=1024 full-wrapper thật** (§14.6 footnote — SUM thủ công 1.18-1.33×, wrapper thật chưa đo) **→ ĐÃ ĐO BÁC BỎ 2026-10-23 (probe CX, §14.10): wrapper thật 1.01-1.12×.** Kết luận lớn: nút thắt full-step = wrapper interpreted, đã thu hẹp qua **8 chu trình đo (CG/CH/CB/CI/CU/CW/M1/CX)**; **parity 1.0× không đạt được từ repo-local.** Logs: `out/m1_*.log`, `out/cx_run.log`.

### 14.10 M3b DU PHONG — T=1024 FULL-WRAPPER THAT (probe CX, 2026-10-23): ĐÃ ĐO BÁC BỎ — MỤC TREO CUỐI ĐÓNG

*Mục tiêu (footnote §14.6): số đo duy nhất còn treo có điều kiện — T=1024 wrapper THẬT (không phải SUM thủ công như CB). Gate: wrapper thật ≥1.3× trên S3/S4/S5 mới mở đường wire (sửa công thức T ở 3 đường + bench full-step + gate 84/84).*

**Cách đo:** `tv/tests/simd_probe_cx.tkv` (0 thay đổi code train, log `out/cx_run.log`) — A/B cùng exe, best-of-3: wrapper production (công thức T = 768 cho 4 shape) vs bản sao nguyên vẹn `_smm_core` (serial) + `_smm_core_pd` (wave-12) với T=1024 tham số; mini tol 256³ so scalar. (Compile lần đầu fail vì thiếu `import tv.training.tensor` → annotation `"Tensor"` không resolve — thêm import là đủ.)

**Số đo (ms, best-of-3, cùng phiên):**

| Shape | route | base prod | T=1024 | sp | maxdiff |
|---|---|---|---|---|---|
| S1 dx1 | serial | 94 | 78 | **1.205×** | 1.9e-06 |
| S3 dxf | wave-12 | 735 | 656 | **1.120×** | 1.3e-05 |
| S4 dwte | wave-12 | 1250 | 1141 | **1.096×** | 3.0e-08 |
| S5 lgts | wave-12 | 1187 | 1171 | **1.014×** | 0.0 |

mini 256³ T=1024 vs scalar: 3.73e-08 ≤ 1e-4 ✓ (builder đúng khi T > mọi dim).

**CHỐT:**
1. **BUST — 1.01-1.12× « ngưỡng wire 1.3×.** KHÔNG wire, 3 đường công thức T giữ nguyên. Footnote §14.6 đóng: SUM thủ công 1.18-1.33× **không** phản ánh wrapper — production in-context đã nhanh hơn SUM tách pha (S3: 735 vs 953 — build/acc chồng lấn với replay trong wave thật) nên headroom SUM bị tính trùng; phần cố định (spawn/join, alloc C, replay simd_gemm-reentrancy) không giảm theo T; S5 k=768 < T=1024 pad k vô ích → 1.014×.
2. **Menu repo-local khép lại hoàn toàn** — mục treo cuối đã có số đo. 8 chu trình đo (CG/CH/CB/CI/CU/CW/M1/CX): M3a ❌×2, M3b ❌×2, M3c ✅, M1 ❌. Từ đây parity 1.0× chỉ qua compiler team (M4-M7 builtin/ABI — §14.9).

Chi tiết: HANDOFF §0.44.
