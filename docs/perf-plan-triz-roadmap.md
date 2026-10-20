# TRIZ OPTIMIZATION ROADMAP — từ "thua PyTorch" sang "khác được"

**Ngày:** 2026-10-20 · **Baseline:** P5.3 consolidated (commit `f85ea3d`) — tkv `simd=1` (P5.1+P5.2+P5.3) vs PyTorch 2.14.0+cpu (14 threads)
**Máy:** Xeon E5-2695 v3 14C · **Cùng ngày, cùng máy, cùng token/config** cho mọi phép đo.
**Mục tiêu:** seq1024 full step **154.234 s (43.5× vs torch)** → **~85 s (~20×)** mà vẫn giữ 84/84 parity assertions.

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
| **Tỉ lệ tkv/torch (full step)** | **43.5×** | **~20×** | "thua nặng" → "khác được" |
| seq256 vs P5.1-only | 2.65× | **≥ 3.5×** | |
| 84/84 assertions | Pass | **Pass** | Không hồi quy |
| Loss vs PyTorch fixture | ≤2.4e-7 logits / ≤8.3e-7 124M grad | **Không đổi** | Parity duy trì |

**Cập nhật 2026-10-20 sau P5.4 (§10):** full step seq1024 **141.219 s → 39.9×** (từ 154.234 s / 43.5×); seq8 20.7×, seq256 26.9×.

**Cập nhật 2026-10-20 sau P5.5 (§11):** full step seq1024 **118.641 s → 33.5×** (gộp forward, −16.0%); seq8 18.7×, seq256 23.2×. Còn lại để tới ~20× (~85 s): cache attn-fwd + MLP fwd trong block_bwd (~50 s → cắt dần), Phase 2 flatten (~2 s thật, nhỏ hơn dự kiến), lm-head.

## 8. RỦI RO + CONTINGENCY

| Rủi ro | Xác suất | Contingency |
|---|---|---|
| GELU erf approx trôi khỏi 2.4e-7 | Thấp (numerics contract đã test) | Default erf; tanh-approx chỉ là debug path |
| Lazy materialize phá parity | Trung bình | Gate từng stage; revert op về eager materialize |
| Thắng seq256 không transfer sang seq8 | Trung bình | Benchmark đủ 3 seq mỗi lần; giữ `simd=0` mặc định |
| Trượt tiến độ 3 tháng | Cao (codebase phức tạp) | Phase 1 đã mang ~50% giảm → dừng vẫn có giá trị |
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
