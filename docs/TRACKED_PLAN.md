# TRACKED PLAN — TokenVector.Inference vs PyTorch (2026-11-14)

> **Nguyên tắc đo:** mọi con số đều từ protocol công bằng — cùng VM, cùng input, seed 7,
> 2 thread CPU, xen kẽ x3, median. Loss ≠ torch (init khác; FLOPs giống). Không so sánh
> single-shot.

## 0. Trạng thái hiện tại (nguồn: `BENCHMARKS.md §9.43` #22 2026-11-14; mới nhất: `§9.44`/`§9.45` Colab 2026-11-17)

| | `.tkv` native | torch CPU 2T | Tỷ lệ |
|---|---:|---:|---:|
| fwd+CE+bwd | **14 745** | **9 167** | **1.61×** |
| TỔNG (full step) | **15 259** | **10 906** | **1.40×** |
| Loss | **25.407877** EXACT | 9.8851 (init khác) | so TIME riêng |

GPU bridge (cùng phiên notebook #22, `TKV_TORCH_GPU=1`): fwd+CE+bwd **6 902** vs torch CUDA 268.1 (**25.7×**); TỔNG **7 407** vs 299.2 (**24.8×**); loss_val **17.106686** (ATen bridge, không parity với AVX2).
**GPU fix (2026-11-15, commit Release `8929830` — `host()` `std::memcpy` CUDA ptr → `copy_`): bridge giờ GREEN; phiên 2026-11-15 đo `TONG 8 252 ms`, loss_val **17.111387** (§9.45).**
**GPU manual backward chay that (2026-11-17, notebook (20), T4): TONG **5 256 ms**, fwd 803 / bwd **3 069 ms** / opt 417 / copyback 87; loss_val **17.111390** (17.111387 - lech 3e-6, ATen non-determinism). ATTN_BWD_MERGE **BIEN MAT** (manual attention backward ATen CUDA chay that thay cho fallback AVX2). Nguyên nhân trước đó: TkvTorchAttentionBwd truyền tensor 4-D vao at::bmm (ATen docs: bmm bắt buộc 3-D) -> c10::Error -> fallback mỗi lần; fix = at::matmul + softmax-bwd P*(dP-sum(dP*P)) + host() cho dx (Release `a6dfc30`) + notebook clone in TokenVector HEAD & parse loss_val thật (Inference `cc70fe4`). GPU nhanh native **~3.0x** (§9.47/§9.48).**

**Milestone GPU bridge 2026-11-17 (notebooks (19)->(30), T4):** ATen CUDA du cho attention bwd/fwd,
LM head, LN; manual backward khong qua autograd (autograd CUDA trong engine nay "im lang":
`.grad()` leaves undefined -> throw UNKNOWN_SCALAR). So do:
- TONG 7 892 (19) -> **3 849 ms (28)** | bwd 5 676 -> **1 582 ms** | LMHEAD_BWD 5 748 -> **1 456 ms**
- loss_val GPU ~17.1113 (waiver numerics ATen); native EXACT 25.453880/25.407877 khong doi.
- Phiên (30): build/numerics GREEN nhung GPU bi throttle (dev H2D 3MB: 0.95->8-21ms) — perf
  khong dung de so sanh (§9.54); bench cell da in nvidia-smi clocks tu chan doan.
- Con lai ~3.4-3.6s la interpreter .tkv + GEMM/SDPA quality; torch CUDA ~0.27s — can
  custom CUDA kernels moi bum, ngoai pham vi bridge.

Peak RAM: `.tkv` **4.09 GB** < torch **4.29 GB** (§9.40, notebook #14).

**Gates đang xanh:** testkit 136/136, NEG 44/44; trainer 15/15; loss EXACT (mode 0) (§9.37).
**Notebook:** `demo/colab_bench_ai_gpu.ipynb` — 10 cells: `[6/9]` Bench native (.tkv CPU, x3, median), `[7/9]` Bench GPU bridge (x3, median; giờ GREEN sau fix `8929830`), `[8/9]` op-breakdown `TKV_PROF=1`, `[9/9]` GDB backtrace GPU segfault (tuỳ chọn, chỉ chạy khi [7/9] fail).

## 1. Mô hình bottleneck hiện tại

Đo bang `TKV_PROF=1` (op-time, native, seq1024). Mỗi bảng ghi kèm phiên đo (luật #1).

Phiên mới nhất — notebook #22 (2026-11-14), `BENCHMARKS.md §9.43`:

| Op | ms |
|---|---:|
| LMHEAD_BWD | 6 755 |
| ATTN_BWD_MERGE | 6 507 |
| ATTN_FWD | 5 090 |

Tổng op-time native 32.7 s vs bridge-GPU 14.0 s.

Bản đầy đủ 7 op (kèm %) — notebook #14 (2026-11-13), `BENCHMARKS.md §9.40`:

| Op | ms | % |
|---|---:|---:|
| LMHEAD_BWD | 7 085 | 21.6% |
| ATTN_FWD | 7 143 | 21.8% |
| ATTN_BWD_MERGE | 6 037 | 18.4% |
| LIN_DW | 5 094 | 15.6% |
| LIN_DX | 4 943 | 15.1% |
| LN | ~1 122 | — |
| **GELU (sau fix AVX2)** | **1 116** | **(đã từ 11 624 xuống)** |

Giải nghĩa: LMHead + attention backward (= reshape + 3-4 GEMM) chiếm >60% bwd.

## 2. Mục tiêu theo thứ tự ưu tiên

| # | Mục tiêu | Win-condition | Triển vọng | Ghi chú |
|---|---|---|---|---|
| 1 | `.tkv` ≤ torch CPU (full step) | **< 10 906 ms** (torch T=2, §9.43 #22) | **KHÓ** | cần bwd nhanh hơn ~30%, sát mức torch, không bitexact; chạm 1.0× khó |
| 2 | **seq1024 mode 1 bit-ident loss** ↔ torch | gate gradient | **KHÓ** | đưa GELU AVX2 kỹ; quality duy trì |
| 3 | **seq1024 full step < 4000 ms** | **THẮNG torch** | **KHÓ nhưng khả thi** | cần: attention native native để bwd, GELU đã AOT, LMhead đã AOT, LN |
| 4 | **GPU full-Tck** | chạm ~3-5 s/full step | **RẤT KHÓ** | dựa P1 device-resident (PLAN_GPU_BACKEND.md); ngắt hot path |

**Win condition thực tế nhất:** mục #3. Mốc 4 000 ms seq1024 = lặp lại con số 4016 ms
(đúng param + cùng máy Colab). Và đó là cái mà chúng ta đang gần nhất.

## 3. Việc cần làm (theo thứ tự)

### A. Cụ thể, trước mắt
- [x] **[A1]** mode 1 seq1024 chạy được (8.2 s); loss=10.760399 -> 13.686271 (warmup loss **trùng** mode 0 -> score EXACT). Không crash/nan. **Chưa verify độc lập** (loss sau step phụ thuộc optimizer; không so sánh được giữa các phiên).
- [x] **[A2]** Rebuild `demo/bseq.dll` từ source mới; chạy cell bench lại; ghi
      `BENCHMARKS.md §9.43`. → Verify 2026-11-14 (local): rebuild bằng
      `D:\TokenVector\Release\dist\tkvc-tv.exe` cho IL SHA256 **trùng khớp**
      `out/bseq.il` hiện tại (compiler+source deterministic); `.exe` chỉ khác
      19 byte (MVID + PE timestamp); `demo/bseq.dll` == `out/bseq.exe`. Nghĩa là
      binary đã track trên repo là bản mới nhất từ source+compiler hiện tại.
      Số đo đã ghi §9.43 + `HANDOFF.md §0.187` (notebook #22, 2026-11-14,
      xen kẽ x3, median). Cell CPU `[5/5]` đã bỏ khỏi notebook (commit
      ee5db29); cell bench GPU hiện tại là `[5/6]`.
- [x] **[A3]** Đo lại op-breakdown với `TKV_PROF=1` (cell `[6/6]`).
      → Đã có trong §9.43 (notebook #22): native ATTN_FWD 5 090 /
      ATTN_BWD_MERGE 6 507 / LMHEAD_BWD 6 755; bridge-GPU 5 966 / 6 089;
      tổng op-time native 32.7 s vs bridge-GPU 14.0 s.
- [x] **[A4]** Tìm trong `bwd` vùng có thể `#ADD` thêm: ATTN_FWD/ATTN_BWD tận dụng
      `attention_fwd_fused` (AVX2) — **ĐÃ PHÂN TÍCH 2026-11-14, kết luận: KHÔNG #ADD,
      production thực tế không đi qua đường native bwd.** Bằng chứng (cell `[6/6]`
      notebook #22):
      1. `TKV_PROF` chỉ đếm timer tại entry native (`tkv_prof.h`); op đi qua
         ATen/bridge → counter = 0.
      2. Chạy bridge-GPU (`TKV_TORCH_GPU=1`) vẫn thấy **`ATTN_BWD_MERGE` 5 966 ms**
         ≈ native 6 507 ⇒ production (simd==1 + `gflat=opt.g`) đã rơi rớt về
         native `FlashAttentionBwdMergeMulti` — ATen attention-bwd
         (`torch_attention_bwd` → HF autograd) đã trả rc≠0 ngay trong phiên
         (Release repo 2026-11-13..14: 60cbb54 → 02e81b2 → 9cfdc14 "de Colab
         xem vi sao fallback" → 986e4e0 → a6b9749 → e043b3f, toàn đăng debug
         + fix CUDA-device; chưa có bằng chứng bwd chạy qua ATen).
      3. `ATTN_FWD` không thấy ở bridge-GPU (qua `torch_sdpa_fwd` → SDPA CUDA),
         thấy 5 090 ms ở native; native fwd vốn đã gọi `attention_fwd_fused`
         (kèm fallback) — không có chỗ #ADD mới.
      4. `LMHEAD_BWD` 6 755 native / 6 089 bridge (3 call, `lmhead_bwd_fused`;
         chỉ phần dwte đi qua bridge `TkvTorchAddmmTaOut` bên trong) — target
         lớn nhất còn ở native; xem **B3**.
      5. Hệ quả: đường fallback chính là đường production thật. Fix A4a (tính
         bù qkv khi ATen fail) là bắt buộc — trước fix #22 đo bằng gradient sai.
         Còn ~125 ms/step phí extra (GEMM 3.6 GFLOP mỗi layer nếu fallback).
      6. Bước tiếp theo cần Colab: xem stderr `[TkvTorchAttentionBwd]` ở cell
         `[6/6]` để biết lỗi ATen-bwd; nếu bwd ATen chạy được → xoá hẳn
         `ATTN_BWD_MERGE` khỏi op-time; nếu không → fallback native là đường
         chính thức, bench + gate phải xác nhận lại.
- [x] **[A4a]** Sửa bug correctness (phát hiện trong A4): `_qkv_ready` bỏ
      `gpt2_linear(qkv)` kể cả khi ATen fail → fallback native đọc pool slot 6
      chưa ghi → gradient sai im lặng (dw_qkv lệch 6.3e-3, dw_proj 4.6e-2;
      tái hiện local 2026-11-14). Fix: fallback tính bù qkv ngay ở nhánh fail
      (chỉ khi ATen rc≠0, đường ATen thành công không tốn phí). Test hồi quy
      `tv/tests/gpt2_attn_bwd_gflat_fallback.tkv` — 5 case, FAIL 3 case trên
      code cũ, PASS sau fix; C vs A lệch 6.3e-3 → 5.96e-07. **Chưa gate Colab.**
- [x] **[A4b]** Rebuild `demo/bseq.dll` từ source đã fix (2026-11-14):
      `out/bseq_new.exe` rc=0 (IL có thêm `gpt2_linear` bù qkv ở nhánh ATen-fail,
      line 25666); binary tracked thay mới (SHA256 `C7E16C6D…`; binary cũ
      `DC772599…` là stale). Notebook: `[2/6]` in HEAD + sha256 bseq + fix marker;
      `[6/6]` lọc thêm `TkvTorchAttentionBwd`/`AttentionBwd` (lộ stderr ATen
      fail; luật #2 — lý do: A4 phân tích 2026-11-14, Release repo 9cfdc14);
      `[5/6]` **không sửa**. Push xong 2026-11-14: `cf61c42..befa5ae main -> main`;
      Colab clone được fix.
- [x] **[A5]** Phiên Colab 2026-11-15 (T4, HEAD `dbe4f97`, notebook 9 cells — GREEN sau fix).
      Kết quả (`BENCHMARKS.md §9.44` native, `§9.45` GPU+fix): native median
      **TONG 15 783 ms** (fwd 3 745 / bwd 10 628 / opt 436 / copyback 87).
      **GPU bridge GREEN: TONG 8 252 ms** (fwd 1 019 / bwd 5 859), loss_val **17.111387**
      (ATen bridge; gần #22 = 17.106686). **Root cause GPU segfault**:
      `torch_fused.cpp::host()` — `std::memcpy` GPU→CPU trên CUDA pointer ⇒ SIGSEGV;
      fix commit Release **`8929830`** (`tgt.copy_(t)`). **GPU bench lần đầu GREEN.**
      GPU sai nhanh nhưng attn-bwd vẫn fallback AVX2 native (~40% op-time GPU);
      đó là design hiện tại, không phải bug. **§9.43 GPU 7 407 ms vẫn tham chiếu gốc;
      phiên này thấp 8 252 ms (máy chậm).**

- [x] **[A5-1]** Phien Colab 2026-11-17 (notebook (20), T4, Inference cc70fe4 / TokenVector a6dfc30).
      **GPU bridge: TONG 5 256 ms** (fwd 803 / bwd 3 069 / opt 417); loss_val **17.111390**; ATTN_BWD_MERGE bien mat;
      native cung phien **TONG 15 811 ms** (loss_val 25.453880) - **GPU nhanh native ~3.0x**.
      LMHEAD_BWD con 5 748 ms (77.5% op-time) - dxf = dl @ wte van SgemmF32 native,
      de xuat route qua ATen CUDA (doi y user xac nhan waiver numerics). LN GPU cham hon native. (`BENCHMARKS.md` 9.47/9.48)
- [x] **[A5-2]** Phien Colab 2026-11-17 (notebook (21), T4, Inference daf3197 / TokenVector 42832c2).
      dxf (pha 2 lmhead) route ATen CUDA cuBLAS (waiver numerics user duyet 2026-11-17).
      **GPU bridge: TONG 4 533 ms** (fwd 808 / bwd 2 361 / opt 413); loss_val **17.111393**;
      LMHEAD_BWD 5 748 -> **3 638 ms**; native cung phien **TONG 15 208 ms** - GPU nhanh native **~3.36x**.
      Native EXACT 25.407877 giu nguyen (native mode khong doi code path). (`BENCHMARKS.md` 9.49)

- [x] **[A5-3]** Phien Colab 2026-11-17/18 (notebook (22), T4, Inference d24b044 / TokenVector ef01ab8).
      pha 1 LMHEAD qua ATen CUDA (waiver tiep noi): **LMHEAD_BWD 3 638 -> 3 210 ms**; TONG 4 605 ms (x1, noise);
      loss_val 17.111393; native EXACT khong doi. Cell 11/9 ln-bench BUG do code (khong requires_grad +
      autograd.grad(None)) - fix Inference **9f7d2f8** (khong raise). (`BENCHMARKS.md` 9.50)

- [x] **[A5-4]** Phien Colab 2026-11-17/18 (notebook (23), T4, Inference e5c9891 / TokenVector eae6c97,
      bench x3 median): **GPU bridge TONG 4 268 ms**; bwd **2 099 ms**; loss_val **17.111298** (x3);
      native median **TONG 15 417 ms**, loss EXACT. LMHEAD_BWD 2 520 ms (pha 1+2 ATen chay).
      ln-bench cell OK (torch LN fwd 0.450 ms bwd 1.284 ms). Notebook bi invalid JSON vi edit
      cua minh (fixed `08ce021`/`e5c9891`). LN bridge debug (tkv-ln prints) cho (24). (`BENCHMARKS.md` 9.51)

- [x] **[A5-5]** Phien Colab 2026-11-17/18 (notebook (27), T4, Inference 99f5183 / TokenVector ca77396).
      LN bwd manual qua ATen (autograd CUDA im lang - leaves .grad undefined): **LN_BWD 862 -> 505 ms**.
      **GPU bridge TONG 4 145 ms** (fwd 778 / bwd **1 925**); loss_val **17.111305** (x3); native EXACT,
      GPU ~3.97x native. op-time 3 803 ms; LMHEAD_BWD 2 530 (66.5%). Native mode khong doi. (`BENCHMARKS.md` 9.52)

- [x] **[A5-6]** Phien Colab 2026-11-17/18 (notebook (28), T4, Inference f3443d5 / TokenVector 993ad6e/+1f4efad).
      **LMHEAD_BWD 2 530 -> 1 456 ms** (A0.151 tron goi 1 CUDA call, dl khong ve CPU).
      **GPU bridge TONG 3 849 ms** | fwd 796 / bwd **1 582** | loss_val **17.111270**; native 17 310, GPU ~4.5x native.
      LN_FWD: call dau ~465 ms (CUDA lazy module load) - warm-up trong constructor Release `1f4efad`;
      steady-state ~1.9 ms/call. Ghi `BENCHMARKS.md` 9.53.

- [x] **[A5-7]** Phien Colab 2026-11-17/18 (notebook (33), T4, Inference ff27e83 / TokenVector 313894d).
      pined staging + warm-up chay ththat: GPU bridge **TONG 3 322 ms** | bwd **1 280** | loss_val 17.111270;
      native 15 440, **GPU ~4.65x native**, GPU khoe (1260MHz/41W/78C). op-time 2 346ms: LMHEAD 1 206, LN_FWD 792, LN_BWD 334.
      Ghi `BENCHMARKS.md` 9.55.

### B. Việc tác động + flags
- [ ] **[B1]** Giống A2/A3: kiểm tra `TKV_GEMM_BLAS=0` loss native, TKV_GELU_AVX2=0
      loss native — nếu loss EXACT fail, có điểm cầu vượt dễ phát hiện.
- [ ] **[B2]** Xác nhận loss EXACT từng mode (0/1/2/1) từ Colab (2 phiên).
- [ ] **[B3]** Gap `LMHEAD_BWD` (6 755 ms — §9.43 notebook #22) — có thể refactor bằng `lmhead_bwd_fused`
      trực tiếp `addmm_out` (oneDNN) → nhưng `gflat` phải ở đó; thiết kế cẩn thận.

### C. Việc "sâu" (dự án kế)
- [ ] **[C1]** `ATTN_BWD_MERGE` (6 GEMM + softmax bwd native, 6 507 ms — §9.43
      notebook #22) — dùng `ATen SDPA bwd` sẽ nhanh, nhưng không tương thích
      với packed layout IL hiện tại.
- [ ] **[C2]** `LMHEAD_BWD`: tách ra cần (do sink, portable).
- [ ] **[C3]** SCALABILITY: GELU này chưa bit-exact. Nếu waiver thì re-record.
- [ ] **[C4]** **GPU parity**: hiện không có parity con pipeline CPU → GPU (loss khác
      25.4 → 17.1). Không đặt GPU làm mục tiêu parity; GPU chỉ là bench phụ trợ.

### E. Nợ tài liệu
- [ ] **[E1]** Merge `ROADMAP_TRAIN.md` vào `BENCHMARKS` hoặc làm rõ mối quan hệ.
- [ ] **[E2]** Reference `docs/HANDOFF.md` cập nhật từ mọi commit (PLAN) → `HANDOFF.md`.

## 4. Luật cập nhật
1. Mọi con số số liệu đều phải có **timestamp và phiên**, kèm link notebook/benchmark.
2. Không sửa Cell `[5/6]`, `[6/6]` nếu không có lý do cụ thể; nếu
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
| `Notebook` = `demo/colab_bench_ai_gpu.ipynb` | Single source of truth cho bench (Colab) |
| `docs/PLAN_FWD_BWD_TORCH_PARITY.md` | Giải thích gap vs torch (threads), kế hoạch đóng gap |
| `tv/tests/simd_bench_124m_seq.tkv` + `demo/bseq.dll` | Bench native/Windows dev; đọc `_seq_cfg.txt` (seq, threads) |
