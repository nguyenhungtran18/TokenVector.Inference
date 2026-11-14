# TRACKED PLAN — TokenVector.Inference vs PyTorch (2026-11-14)

> **Nguyên tắc đo:** mọi con số đều từ protocol công bằng — cùng VM, cùng input, seed 7,
> 2 thread CPU, xen kẽ x3, median. Loss ≠ torch (init khác; FLOPs giống). Không so sánh
> single-shot.

## 0. Trạng thái hiện tại (đo cùng phiên Colab, 2026-11-14)

| | `.tkv` native | torch CPU 2T | Tỷ lệ |
|---|---:|---:|---:|
| fwd+CE+bwd | **14 088** | **8 605** | **1.64×** |
| TỔNG (full step) | **14 593** | **10 402** | **1.40×** |
| Loss | **25.407877** EXACT | 9.8851 (init khác) | so TIME riêng |

Peak RAM: `.tkv` **4.09 GB** < torch **4.29 GB**.

**Gates đang xanh:** testkit 136/136, NEG 44/44; trainer 15/15; loss EXACT (mode 0).
**Notebook:** `demo/colab_bench_ai_gpu.ipynb` (cell `[5/6]` GPU bench .tkv vs torch CUDA, xen kẽ x3 median; `[6/6]` op-breakdown `TKV_PROF=1`; cell attention-check `[5/5]` cũ đã bỏ vì trùng op-breakdown).

## 1. Mô hình bottleneck hiện tại

Đo bang `TKV_PROF=1` (op-time, native, seq1024):

| Op | n | ms | % |
|---|---:|---:|---:|
| LMHEAD_BWD | 3 | 6 497 | **23%** |
| ATTN_FWD | 432 | 5 106 | **18%** |
| ATTN_BWD_MERGE | 36 | 5 646 | **20%** |
| LIN_DX | 108 | 4 528 | 16% |
| LIN_DW | 108 | 4 684 | 16% |
| **GELU (sau fix AVX2)** | **863** | **980** | **(đã từ 11 624 xuống)** |

Giải nghĩa: LMHead + attention backward (= reshape + 3-4 GEMM) chiếm >60% bwd.

## 2. Mục tiêu theo thứ tự ưu tiên

| # | Mục tiêu | Win-condition | Triển vọng | Ghi chú |
|---|---|---|---|---|
| 1 | `.tkv` ≤ torch CPU (full step) | **< 10 400 ms** | **KHÓ** | cần bwd nhanh hơn ~40%, sát mức torch, không bitexact; chạm 1.0× khó |
| 2 | **seq1024 mode 1 bit-ident loss** ↔ torch | gate gradient | **KHÓ** | đưa GELU AVX2 kỹ; quality duy trì |
| 3 | **seq1024 full step < 4000 ms** | **THẮNG torch** | **KHÓ nhưng khả thi** | cần: attention native native để bwd, GELU đã AOT, LMhead đã AOT, LN |
| 4 | **GPU full-Tck** | chạm ~3-5 s/full step | **RẤT KHÓ** | dựa P1 device-resident (PLAN_GPU_BACKEND.md); ngắt hot path |

**Win condition thực tế nhất:** mục #3. Mốc 4 000 ms seq1024 = lặp lại con số 4016 ms
(đúng param + cùng máy Colab). Và đó là cái mà chúng ta đang gần nhất.

## 3. Việc cần làm (theo thứ tự)

### A. Cụ thể, trước mắt
- [x] **[A1]** mode 1 seq1024 chạy được (8.2 s); loss=10.760399 -> 13.686271 (warmup loss **trùng** mode 0 -> score EXACT). Không crash/nan. **Chưa verify độc lập** (loss sau step phụ thuộc optimizer; không so sánh được giữa các phiên).
- [ ] **[A2]** Rebuild `demo/bseq.dll` từ source mới; chạy cell `[5/5]` lại; ghi
      `BENCHMARKS.md §9.43`.
- [ ] **[A3]** Đo lại op-breakdown với `TKV_PROF=1` trên cell `[7/5]`.
- [ ] **[A4]** Tìm trong `bwd` vùng có thể `#ADD` thêm: ATTN_FWD/ATTN_BWD tận dụng
      `attention_fwd_fused` (AVX2) có thể nhanh hơn whisper. cell `[3/5]` sạch hoàn toàn.

### B. Việc tác động + flags
- [ ] **[B1]** Giống A2/A3: kiểm tra `TKV_GEMM_BLAS=0` loss native, TKV_GELU_AVX2=0
      loss native — nếu loss EXACT fail, có điểm cầu vượt dễ phát hiện.
- [ ] **[B2]** Xác nhận loss EXACT từng mode (0/1/2/1) từ Colab (2 phiên).
- [ ] **[B3]** Gap `LMHEAD_BWD` (3 calls = 6 500 ms) — có thể refactor bằng `lmhead_bwd_fused`
      trực tiếp `addmm_out` (oneDNN) → nhưng `gflat` phải ở đó; thiết kế cẩn thận.

### C. Việc "sâu" (dự án kế)
- [ ] **[C1]** `ATTN_BWD_MERGE` (6 GEMM + softmax bwd native, 5 646 ms) — dùng `ATen SDPA bwd`
      sẽ nhanh, nhưng không tương thích với packed layout IL hiện tại.
- [ ] **[C2]** `LMHEAD_BWD`: tách ra cần (do sink, portable).
- [ ] **[C3]** SCALABILITY: GELU này chưa bit-exact. Nếu waiver thì re-record.
- [ ] **[C4]** **GPU parity**: hiện không có parity con pipeline CPU → GPU (loss khác
      25.4 → 17.1). Không đặt GPU làm mục tiêu parity; GPU chỉ là bench phụ trợ.

### E. Nợ tài liệu
- [ ] **[E1]** Merge `ROADMAP_TRAIN.md` vào `BENCHMARKS` hoặc làm rõ mối quan hệ.
- [ ] **[E2]** Reference `docs/HANDOFF.md` cập nhật từ mọi commit (PLAN) → `HANDOFF.md`.

## 4. Luật cập nhật
1. Mọi con số số liệu đều phải có **timestamp và phiên**, kèm link notebook/benchmark.
2. Không sửa Cell `[5/5]`, `[6/5]`, `[7/5]`, `[8/5]` nếu không có lý do cụ thể; nếu
   phải sửa, phải ghi chú (cite phiên đo).
3. Nếu loss EXACT sụt: dừng, rollback, kiểm tra ngay, không push.
4. Gate phải trong sạch trước khi cập nhật plan — không push plan nháp.

## 5. Tham chiếu liên quan

| File | Mục đích |
|---|---|
| `docs/PLAN_GPU_BACKEND.md` | GPU path (đã đóng 2026-11-14, §0.184) |
| `docs/PLAN_BEAT_PYTORCH.md` | Plan gốc 2026-11-04; 2/3 phase đã hoàn thành |
| `docs/ROADMAP_TRAIN.md` | Khởi thủy từ G0-G1 của training |
| `docs/SPEC_FUSED_SDPA_BWD.md` | Thiết kế fused SDPA backward |
| `docs/HANDOFF.md` | Nhật ký mọi bước; không xóa |
| `Notebook` = `demo/colab_bench_ai_gpu.ipynb` | Single source of truth cho bench |
| `Notebook` = validation script (Windows dev) | Win-only benchmark, `_seq_cfg.txt` |
