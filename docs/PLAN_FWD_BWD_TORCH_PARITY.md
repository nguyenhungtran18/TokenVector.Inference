# PLAN — .tkv fwd+bwd tiến gần tốc độ torch (2026-11-15)

**Mục tiêu:** đo lại và đóng gap .tkv vs torch CPU trên cùng máy/thread; verify rằng
".tkv chậm hơn torch" trên Colab là do thread mismatch, không phải do code.

**Đối tượng:** GPT-2 124M, seq=1024, `tv/tests/simd_bench_124m_seq.tkv`, loss EXACT mode 0.

**Không làm:** rewrite runtime; đổi compiler / native layer trừ khi phép trừ ROI lớn hơn 5%.

---

## 0. Sự thật về ".tkv chậm torch"

| Phiên | Máy | .tkv full step | torch | .tkv/torch |
|---|---|---:|---:|---:|
| 2026-11-11 (§9.22) | Windows, Xeon E5-2695 v3 | 4016 | 4061 | **0.99×** |
| 2026-11-11 (§9.22) | Windows, Xeon E5-2695 v3, mode 0 | 3375 | 3395 | 0.99× |
| 2026-11-15 (§9.44) | Colab T4, torch 2 thr, .tkv 12 thr (oversubscribe 6×) | 15 548 | 10 906 (§9.43) | **1.43×** |
| 2026-11-15 (§9.44) | Colab T4, same | 15 783 (§9.45 GREEN) | 10 906 | **1.45×** |

Conclude: **trên cùng máy, cùng protocol, .tkv ~ parity (0.99×)**. Colab chậm hơn vì `TKV_GEMM_THREADS` mặc định 12 trên 2 vCPU — oversubscribe 6×.

**0a-fix:** `TKV_GEMM_THREADS=nproc` (hoặc trực tiếp =2 trên Colab 2-vCPU) → kỳ vọng quay lại ~1.0×.

### 0b (2026-11-15, cell `[10/9 torch-profile]`) — same-machine parity, gap nhỏ

Đo torch CPU backward trên chính VM (T4, L4) như sau:

```
aten::mm                             52.12%  4.843s   99 calls
aten::addmm                          17.12%  1.591s   48 calls
aten::_scaled_dot_product_flash_attention  13.6%  1.260s  24 calls (12 fwd + 12 bwd)
aten::mul / add / log_softmax / fill / pow  ~10%    ~900ms
Self CPU total (backward): 9.291s
```

**So sánh cùng máy (chỉ backward):**
- torch: **9.29 s** (aten ops, CPU) | .tkv: **~10.6 s** (`TKV_PROF` bwd §9.44/§9.45)
- Gap ~1.1× — phù hợp với `BENCHMARKS.md §9.22` (Xeon 2695 v3, 0.99×).

**Conclude (cập nhật):** gap ``.tkv full step chậm torch`` ~1.45× Colab là tổng của
(a) fwd+CE (~1.1×), (b) bwd (~1.1×), (c) opt (.tkv ~469 vs torch ~594)
— các yếu tố đều nhỏ; KHÔNG phải variance lớn như phiên trước tưởng.

Ghi chú: Colab luôn chậm hơn Windows Xeon đáng kể; .tkv/torch ngang trên Windows.

---

## 0.5 GPU bridge hiện tại đã GREEN (2026-11-15 — xem `BENCHMARKS.md §9.44/9.45`)

**Câu hỏi user đặt:** làm sao để .tkv không chỉ
- chạy được trên GPU (đó là tiêu chuẩn tối thiểu rồi),
- mà **còn cạnh tranh sòng phẳng** với torch CUDA (~270–300 ms full step),
- tức là **không chấp nhận tình trạng hiện tại: 8 252 ms trên GPU, chậm 25–30× torch**.

Hiện trạng (~25× chậm torch CUDA, 1.9× nhanh hơn .tkv CPU):

| Thành phần | Hiện tại (bridge) | Torch CUDA |
|---|---|---|
| ops | ATen bridge, from_blob CPU→H2D→CPU, D2H mỗi op | direct CUDA kernels (cuBLAS/cuDNN/FlashAttention) |
| attention bwd | **fallback native AVX2** (vì `TkvTorchAttentionBwd` nổ exception c10, bị swallow) | FlashAttention bwd (cuDNN/ampere) |
| GEMMs | ATen addmm có H2D/D2H wrap | direct cublasSgemm/GemmEx |
| device-resident | **không** — mỗi op chở data qua PCIe/CUDA | weights + activations trên GPU mọi bước |
| Loss parity | loss_val 17.111387 (≠ torch 9.8851; chấp nhận được vì init khác) | loss_val 9.8851 |

**Plan cụ thể để tiến đến cạnh tranh KHẢ NĂNG (không phải "chạy được"):**
1. Đọc `PLAN_GPU_BACKEND.md` §2 Phương án A/B (cuBLAS + dropout host → flash attention).
2. Ưu tiên tìm nơi ROI lớn nhất:
   - **Device-resident alloc**: weights + activations trên GPU suốt step (sinh ra từ bridge, device_ptr cached).
   - **Attention fwd/bwd**: gọi cuDNN SDPA (is_causal) qua torch ATen C++ `at::scaled_dot_product_attention` — chẻ đo cùng build khác.
   - **GEMMs**: direct `cublasSgemm`/`cublasGemmEx` thay vì ATen wrapper from_blob.
   - **AdamW**: CUDA kernel đơn giản (có thể port từ AdamwStepF32 AVX2 sang CUDA nếu muốn chắc digest).
3. Không cần rewrite runtime —— tất cả là native C code gọi qua FFI như pattern `libsimd_native.dll`.
4. Benchmark trên Colab T4 15 GB: GPT-2 124M full step, mục tiêu **≤ 500 ms** (tăng tốc >30× vs bridge hiện tại).

---

## 1. Breakdown gây chậm (Colab 2026-11-15)

Xem `BENCHMARKS.md §9.46` cho torch-profile tạo cùng máy.

| Op (fwd bwd) | .tkv ms | Ghi chú |
|---|---:|---|
| `ATTN_FWD` | 4 674–6 400 (khác phiên) | vs torch: 1.25s (aten flash fwd) |
| `ATTN_BWD_MERGE` | 5 663–6 507 | vs torch: ~1.25s (aten flash bwd). **Gap ~4–5x** — I must optimize this |
| `LMHEAD_BWD` | 6 507–8 121 | vs torch: gộp trong `aten::mm` (≈3.2s) |
| `LIN_DW` + `LIN_DX` | 10 548–10 689 | vs torch: gộp trong mm/addmm |
| LN (fwd+bwd) | 1 063–1 132 | gần sát torch |
| GELU fwd+bwd | ~1 175 | |
| `CE fwd+bwd` | ~900 | |

Hot ops trên .tkv (bwd): `ATTN_BWD_MERGE` (~6 s) + `LMHEAD_BWD` (~6.8 s) = 12.8 s
= **~75% bwd**. Tối ưu attention backward sẽ mang lại 6+ s.

**Attention backward gap cụ thể:**
- .tkv `att_bwd_merge_multi` (AVX2 flash bwd, 12 threads) ≈ 6 s
- .tkv fallback scalar (nếu không dùng `attention_bwd_merge_multi`) ≈ 23 s
- torch CPU flash bwd ≈ 1.26 s

Cơ hội lớn nhất: **giảm `ATTN_BWD_MERGE` ≈ 6 → ~2.5 s** → bwd từ 10.6 → ~7 s,
full step từ 15.7 → ~11.7 s ≈ **torch parity (10.9 s)**.

---

## 2. Kế hoạch theo ROI

### Bước 1 — Fix thread oversubscription (nhanh nhất, code KHÔNG đổi)

```
TKV_GEMM_THREADS=$(nproc)  # Colab 2-vCPU -> 2
```

Verify: chạy lại `[6/9]` xem TONG ≈ 8-9k (khớp torch full step 10.9k) hay không.
Nếu đúng, ghi lại phiên TRACKED_PLAN A5/§9.45.

### Bước 2 — Kiểm tra loss EXACT không regressed

- `TKV_GEMM_BLAS=0` (tắt BLAS default) → `loss_val ≠ 25.407877`?
- `TKV_GEMM_THREADS=2` (hoặc 2) → `loss_val ≠ 25.407877`?

Nếu lệch, BLAS path đang không bit-exact; ghi chú ngay và fallback lệnh đó không dùng trên x86.

### Bước 3 — Breakdown fwd/bwd op-by-op trên **cùng máy** (không Colab noise)

- Chạy `TKV_PROF=1` trên máy Windows local hoặc Colab vCPU=14.
- So sánh `LMHEAD_BWD` native fused vs ATen `addmm_out` (đo cả hai trước/sau).
- Chỉ giữ native fused nếu nhanh hơn; ngược lại — bật ATen path (hoặc ngược lại, gọi `onednn_linear_bwd(addmm_out)` trực tiếp với kiểu khác).

### Bước 4 — Đóng hoàn toàn copyback — REJECTED

**Đã thử** ở `gpt2_unflatten` (2026-10-27, commit trước session này):
est 844 ms; thực tế **0% gain** trên cả seq8/seq1024 + phình ~0.9 GB RAM
so với 844 ms ước tính → **DROP, giữ đường interpreted cũ**.
Kết luận: không làm zero-copy phẳng. Không ROI đủ lớn, chỉ negative.
Không làm metric "copyback = 87 ms → 0" cho bước này nữa — chấp nhận con số hiện tại
(~0.6% tổng step).

### Bước 5 — Tối ưu `LMHEAD_BWD` + `ATTN_BWD_MERGE` (nếu hot-op vẫn chênh lệch)

- `LMHEAD_BWD` (6.6 s): thử ATen `addmm_out` vs native fused; chọn nhanh hơn.
- `ATTN_BWD_MERGE` (6.1–6.5 s): đo ROI của `attention_bwd_merge_multi` vs TBH SDPA
  (bit-exact khác nhau; cần EXACT mode gate).

### Bước 6 — Phân tích `fwd` gap (torch nhanh hơn ~2,000 ms)

- `ATTN_FWD` hiện đang fast-path torch_sdpa_fwd; thử ép về native flash fwd
  (`attention_fwd_fused`) để xem .tkv nhanh hay chậm (cái nào nhanh giữ).
- Cân nhắc fusion teacher: .tkv bọc `onednn_linear_bias` + `gelu` cho MLP fwd
  (đang làm rồi) — kiểm tra xem `.tkv` có thể bỏ được một lần đọc/ghi activation trung gian.

---

## 3. Chiến lược benchmark công bằng

1. **Chạy cùng máy** (Windows Xeon E5-2695 v3): `bseq.exe` native + `bench_torch.py e 1024`
   cùng phiên → đối chiếu `BENCHMARKS.md §9.22`.
2. **Colab**: luôn set `TKV_GEMM_THREADS=2` (2 vCPU) trước `[5/9]` `[6/9]` `[9/9]`.
3. **Ghi lại phiên** theo luật: TRACKED_PLAN A5 + BENCHMARKS mới (timestamp).
4. Loss EXACT (mode 0) phải luôn = `25.407877`; lệch ⇒ rollback ngay.

---

## 4. Ràng buộc

- Giữ nguyên bit-exact: bất kỳ change nào đụng bwd/fwd làm loss lệch = revert.
- Không thêm threading mới trừ khi `TKV_GEMM_THREADS` cho lợi rõ ràng (≥5%).
- **CUDA là lợi thế cạnh tranh chính** — không thể né. Nếu .tkv không có CUDA, repo
  này không cạnh tranh được với torch (~25× chậm hơn). Không được coi GPU
  là "phải tránh"; phải coi nó là mục tiêu bắt buộc.
- GPU bridge hiện tại (TKV_TORCH_GPU=1) **chỉ là "CPU trên GPU"** — mỗi op đều:
  H2D copy → ATen CPU kernel → D2H copy, hoàn toàn không có device-resident
  weights/activations; attention bwd lại fallback native (43% op-time). Nó **GREEN**
  (chạy được, loss_val 17.1) nhưng **~25× chậm hơn torch CUDA**.
- Mục tiêu CUDA thực sự = `PLAN_GPU_BACKEND.md` §2 (Phương án A/B):
  device-resident tensors + cuBLAS + custom flash fwd/bwd kernel hoặc
  gọi thẳng CUDASDPA, + CUDA AdamW fused — không qua from_blob+wrapper.
- Bắt buộc chạy đúng T4 (hoặc cao hơn) và đối chiếu loss_val vs
  `BENCHMARKS.md §9.45` (loss_val GPU = 17.111387) để xác nhận không mất parity.

---

## 5. Ưu tiên thực thi

Thứ tự cập nhật, sau khi đo thực tế trên cùng VM (Cell `[10/9]`):

1. **Đo gap same-machine mode=2** — session Colab mới chạy notebooks `[6/9]` `[8/9]`
   và `[10/9]` → so sánh bwd self-CPU của torch (9.29 s) vs .tkv bwd wall
   (`TKV_PROF` ≈ 10–11 s). Gap thực sự ~1.1–1.2×; KHÔNG phải 25% nhiều như suy luận ban đầu.
2. **Nếu muốn nhanh hơn nữa:** tập trung vào `ATTN_BWD_MERGE` — flash AVX2
   `attention_bwd_merge_multi` (12 luồng) có thể tối ưu thêm 2–4× ở nhánh fallback;
   có thể thử `attention_bwd_fused` per-head hoặc tái cấu trúc backward "--recompute"
   theo flash-attn-2 algorithm (save q,k,v và recompute on the fly).
3. **LayerNorm .tkv (~1.1 s) vs torch (~50 ms)** — đi tìm vì sao AVX2 LN mất 46 ms
   /call: hội thoại seq1024 h=768; 75 calls = 1.2M elements; tính thôi 0.015ms mỗi element,
   tức là LN_back += 1.2s so với torch. Bước này có khả năng cắt giảm 1+ giây dễ dàng.
4. **GELU + CE** — ít tác động (~1 s combined).
5. **Output ATEN vs native GEMM comparison** — .tkv line DW/DX 10+ s; aten mm+addmm 6.4 s —
   có thể coi `SgemmPacked` là đường mở đầu; đáng benchmark `TKV_GEMM_BLAS=0
   thử cùng máy để xem gap lớn vào đâu.
6. **Nếu muốn đạt parity torch CPU:** fix LayerNorm + attention bwd ≈ 11.7 s total
   full step: **CLOSEST candidates**.
