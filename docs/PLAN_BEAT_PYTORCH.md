# PLAN BEAT PYTORCH — TRẠNG THÁI 2026-11-11: WIN CONDITION ĐÃ ĐẠT

> Banner (2026-11-11): mục tiêu "full step seq1024 ≤ torch" **đã đạt**:
> **4016 ms** vs torch 4061.1 (cùng phiên, `BENCHMARKS.md` §9.22); loss EXACT;
> gate 135/135. Bảng "torch re-measure" bên dưới (3986.1/4985…) đã lỗi thời —
> đừng dùng. Phần lớn kế hoạch này đã thực thi; các mảnh còn giá trị nay nằm
> ở `docs/COMPILER_TEAM_REQUEST11-14.md` và `BENCHMARKS.md`. Giữ file làm hồ
> sơ quyết định; không thực thi.

# PLAN: TokenVector.Inference beat PyTorch (GPT-2 124M full train step)

Ngay: 2026-11-04 (cap nhat 2026-11-09; them muc 5 - lo trinh toi uu tiep)
Muc tieu: full step <= PyTorch 14T (1042.7 / 1572.7 / 4148.4 ms) o ca 3 seq, gate 135/135 + trainer 15/15 + loss EXACT.

## 0. Baseline hien tai (do moi 2026-11-09, .NET 10; HANDOFF §0.123)
- TV SIMD (2026-11-09, loss EXACT ca 3 seq):
  - seq1024 fwd+bwd ~4063 (fwd 1047 / bwd ~3016), full step mode 1 ~3438 (s0.126)
  - seq512 fwd+bwd ~2250 (fwd ~516 / bwd ~1734), full step mode 1 ~3438
  - seq256 fwd+bwd ~1422 (fwd 281 / bwd ~1141), opt.step ~860-890 (s0.126)
  - seq8 chua do lai (giu so 2026-11-06: fwd+bwd 1829, full step 2813)
  - (so 2026-11-06: seq1024 fwd+bwd 11875, full step 12828 -> nay -61%; §0.115-§0.122)
- PyTorch (re-do 2026-11-09, threads=14, torch 2.14.0+cpu):
  - seq8 fwd+bwd 317.8, full step 1012.1  (TV 1078; TV 3.4x)
  - seq256 fwd+bwd 848.6, full step 1517.5  (TV 1672; TV 2.0x)
  - seq1024 fwd+bwd 3451.3, full step 3986.1  (TV ~4063; TV 1.18x)
- Gap 2026-11-09 (s0.126): seq1024 fwd+bwd ~4063 vs PyTorch 3451.3 (~1.18x). Win condition f+b < 4148.4 da dat; full step 4985 vs 3986 - chua.
  Gap con lai la GEMM efficiency + ATen dispatch overhead; bf16 bi loai (s0.124); Pha 2/3 xong (s0.123/s0.125).
## 1. Nguyen nhan (do duoc)
1. Backward nang; sau #15/#16 bwd seq1024 ~8750 (best 8734), opt.step ~920-1060, fwd ~3400.
2. Copy round-trip con r1/r3 trong opt.step.
3. list[f32] thao tac managed (da giam bang seed*n trong t_zeros).
4. Threading scale ~2.3x vs torch 3.7x.
5. GEMM da native moi k in {64,768,3072}; kernel GEMM khong phai nut that.

## 2. Phases pipelines & state

### P0 - Do va ghi so (hoan thanh)
- Da do TV va PyTorch, ghi bang so.

### P1/P2/P3/P4 - #10/#14/#15/#16 (hoan thanh boi compiler team)
- native linear_dw_at_fused, attention_bwd_fused, embed_bwd_fused, softmax_causal_fwd/bwd.
- bwd best 8734 (< 9000), full step 13328 (seq1024).

### P5 - Peel cuoi
- [x] Revert 2 dong sai cua ConstantFoldingPass.
- [x] gpt2_block_bwd: mutate gout/drl1 in-place, bo 2 t_zeros + 2 loop.
- [x] t_zeros: `seed: list[f32] = []; seed.append(0.0); big = seed * n` — nhanh hon memset; `[0.0] * n` bi loai vi List[Double], native castclass fail.
- [x] Do lai 2026-11-06: numbers o muc 0.

## 2e. Trang thai 2026-11-06 (sau #12)
(Xem ben duoi muc 0 — duoc hop nhat vao bang baseline.)

## 2f. Trang thai 2026-11-06 (sau t_zeros list-multiply)
- Loss EXACT: seq8 7.0119185 / seq256 11.807271 / seq1024 13.442915.
- So voi truoc: seq8 1860->1829 / 3140->2813; seq256 3900->3750 / 4859->4782; seq1024 12350->11875 / 13453->12828.
- Gate .NET 10: 7 PASS (simd_matmul/gpt2/gpt2_scope/gpt2_train/grad/parity/inplace); pack/gap pre-existing (.NET 10 invalid program; pack PASS direct); gpt2_full chua chay xong.

## 2g. Trang thai 2026-11-06 (sau bias_grad wiring)
- `gpt2_attention_bwd`: `db_proj` qua native `bias_grad` (gflat offset hoac db_proj tensor), fallback interpreted nguyen van.
- A/B xen ke: seq8 khong doi, seq1024 kho do do may dao dong; native bias_grad giam vong managed.
- Gate .NET 10: 7 PASS (giong truoc); pack/gap pre-existing; gpt2_full chua chay.

## 3. Win condition
- seq8 < 1042.7 ms
- seq256 < 1572.7 ms
- seq1024 < 4148.4 ms
- Loss EXACT, gate xanh, RAM seq1024 khong tang.

## 4. Quy tac
- Do truoc, sua sau.
- Gate 135/135 sau moi thay doi.
- Ghi so vao HANDOFF/report sau moi phase.
- Xem COMPILER_TEAM_REQUEST12.md cho ask chi tiet (native non-GEMM + opt.step).
## 5. Lo trinh toi uu tiep (2026-11-09, sau HANDOFF §0.115-§0.121)
Nguon so: HANDOFF §0.115-§0.121 + bench transcripts 2026-11-09 (seq1024 mode 0).
Gap con lai vs PyTorch (cung may, torch 2.14.0+cpu MKL 2026.1): ~1.27x.
Con thieu ~415 ms de dat win condition seq1024 (muc 3) ma KHONG dung bit-exact.

### 5.0 Baseline hien tai (so do gan nhat, 2026-11-09 sau Pha 1)
- TV SIMD seq1024 mode 0: fwd ~1078, bwd ~3547, fwd+bwd ~4625 ms (HANDOFF §0.122)
- TV seq256 mode 0: fwd ~281, bwd ~1406, total ~1797 ms
- TV seq512 mode 0: fwd ~500, bwd ~2046, total ~2546 ms; full step mode 1: ~3782-4047 ms
- TV opt.step (mode 2): ~859-891 ms (AdamW native 8-thread + prefetch; NT stores DA LOAI, xem §0.122)
- Full step tuong duong (mode 2, seq1024): TONG=5375 ms
- PyTorch (re-do 2026-11-09, threads=14, torch 2.14.0+cpu):
  seq8 317.8 / 1012.1; seq256 848.6 / 1517.5; seq1024 3451.3 / 3986.1.
- Loss van EXACT ca 3 seq (10.572291 / 10.756927 / 10.760399), mode 1 bite-ident 10.764943 -> 12.684581.
- Pha 1 da THU: prefetch giu lai (~2% standalone), NT stores bi loai vi cham hon;
  Pha 1 KHONG dong gap (xem s0.122). Win condition can Pha 2 hoac Pha 3 (waiver).

### 5.1 Pha 1 - Bit-exact, rui ro thap (lam truoc, dung thu tu)

**P1a. AdamW non-temporal stores + prefetch - DA LAM (HANDOFF s0.122): NT bi LOAI (cham hon 8-13%), prefetch giu.** (`Release/tools/simd_native/tkv_simd_native.c`,
`adamw_f32_range` va nhanh mode-8 `AdamwStepF32`):
- Hien: 7 luong (doc p,m,v,g; ghi p,m,v) dung store thuong -> doc-for-ownership oan.
- Sua: `_mm256_stream_ps` cho p,m,v ghi (sau khi rong 32B alignment - AllocHGlobal + offset,
  can peel/fallback store thuong neu khong 32B aligned); prefetch p,g scope [i+64 floats];
  `_mm_sfence()` truoc khi tra ve.
- Du kien: opt ~860 -> ~600 ms (save ~30% traffic). Bit-exact (CHI doi cach Ghi, KHONG doi
  gia tri, khong doi thu tu doc/ghi co dependency trong kernel; pool wait giu order giua worker
  va reader ben ngoai).
- Verify: test_adamw.c van PASS; adamw_mt_check.c maxdiff ~ULP; loss EXACT seq8/256/512/1024;
  mode 1 seq512 bite-ident ca 2 pool.

**P1b. `_save_a` — DA KIEM TRA: da pool qua sa_bufs (production); non-saved da co obuf (§0.121). Khong co viec.** (nhanh NON-saved `gpt2_mlp` chung mach §0.117/§0.120):
- Hien moi goi t_zeros [n,4h] cho _save_a; production (saved path) khong anh huong nhieu,
  nhung nhanh non-saved + 2 mirror file can don. Du kien: vai chuc ms. Bit-exact (doc/ghi khong doi).

**P1c. Cache transposition - DA KIEM TRA: production hot path khong transpose (chi n<128 fallback); khong co viec.** (cung 1 step):
- dqkv/dctx re-pack moi lan; luu transpose 1 lan dung lai. Du kien: vai chuc ms. Kiem tra
  hien co pack nao da cache (pack pool §0.119) truoc de khong doi lap. Bit-exact (pure copy).

**P1d. Do lai head-to-head cung may + ghi HANDOFF §0.122 — DA LAM (xem Muc 0 + HANDOFF §0.122)** §0.122**:
- Chay: mode 0 x3 seq + mode 1 seq512 + `tools/bench_torch.py` (seq8/256/1024).
- Cap nhat Muc 0 baseline + bang gap; gate 9/10 + trainer 15/15 bat buoc.

**Du kien Pha 1:** seq1024 full step ~4860-5100 ms (van chua dat 4148.4), opt ~600.
Neu dua seq1024 ve < 4600 ma PyTorch re-measure van ~4190 thi con thieu ~400 ms -> Pha 2.

### 5.2 Pha 2 - Managed glue: simd_add list wrapper - DA LAM (HANDOFF s0.123)
- Probe tv/tests/_glue.tkv: glue-floor ~0.75 us/call; LN obuf ~0.86 ms/call;
  simd_add 5.2 ms -> 0.328 ms/call (~16x): wrapper AddLF32/SubLF32/MulLF32 doi sang
  pin _items + n=min(get_Count()) + calli unmanaged (kem managed AddF32 fallback).
- Compiler selfhost (vong) IL_IDENTICAL; trainer 15/15; gate 9/10 (pack/gap pre-existing);
  mode 1 seq512 bit-ident. Loss EXACT ca 3 seq. fwd: 1024 1203->1078, 512 593->500, 256 329->281.
- Muon dong gap hoac gate lai tot hon: Pha 3 (flash) van can waiver; Pha 2 chi lay mot
  tang cuc bo (glue) chu khong dong duoc backward.
### 5.3 Pha 3 - trang thai thuc (HANDOFF s0.124)
- Flash-attention-class: DA CO SAN trong production (ATen SDPA fwd + ATen autograd
  SDPA bwd, ~94 ms/layer vs native 190); loss EXACT giu. __5.3 cu da loi thoi.__
- BLAS default: da ON tu s0.93; s0.124 them sgemm Trans beta=1 cho dW GEMM
  (LinearDwAtFused) - bwd ~3547 -> ~3297. Loss in-ra khong doi.
- bf16/TF32: KHONG kha thi (ha tang f16 + hong toan bo gates EXACT).
- LN (obuf 0.86 ms): khong can sua (ATen 0.43 + wrapper 0.43; ~0.5% step).
- Win seq1024 < 4148.4: chua dat (sau s0.124+s0.125 - f+b ~4340; mode1 seq512 ~3469).
- Gap MLP bwd vs ATen hoan toan: chi con ~150-250 ms/buoc (dx/dW da BLAS, s0.124/s0.125) -
  duoi nguong hieu qua cho mot dot compiler selfhost + risk gate re-base.
- Seq8/256 benefit tuot dang ke do fixed-cost cua bwd (seq8 bwd 1032 vs fwd 46;
  seq256 bwd 1391 vs fwd 281). Nguyen nhan nghi do thread_spawn/dispatcher x millions
  of calls - candidiate cho dot tiep theo nhung CHUA do chinh xac (can profile).

- Con lai: seq8 bwd anomaly (1032 ms, thread-spawn overhead) + thread scaling 2.3x vs 3.7x.

### 5.4 Win condition & gate (khong doi)
- seq8 < 1042.7 ms; seq256 < 1572.7 ms; seq1024 < 4148.4 ms (full step, cung may voi PyTorch).
- Loss EXACT ca 3 seq; mode 1 bite-ident; gate 9/10 (pack/gap pre-existing); trainer 15/15.
- RAM seq1024 khong tang (pool da gom lai; giam allocation moi buoc, khong them).

### 5.5 Rui ro nhat ky
- Luu y: "mode 2" loss_val khac mode 0 (duong bench rieng); khong ket luan win tu mode 2 mot minh.
- A/B phai xen ke 0/1 trong CUNG binary, best-of-3, may dao dong (observed ±5%).
- mode 1 la gate gradient: mode 0 co the xanh loss ma gradient van sai; luon chay mode 1 (seq512).
- PyTorch ref 2026-11-06 cu; moi pha PHAI re-run bench_torch de co cap so cung phien.
- `_seq_cfg.txt` line 4 = pool bit; pool=0 cho ra duong t_zeros cu y nhu truoc (A/B trong cung binary).


## 6. Dự án: Fused SDPA backward AVX2 (2026-11-09, để thay PyTorch/ATen)
Mục tiêu: viết/cải thiện native SDPA backward trên AVX2+FMA đủ nhanh để (waiver) §0.125, có thể thay thế ATen `scaled_dot_product_attention(is_causal)`, đóng trần hiệu năng backward (attention_bwd hiện ~1.1 s/step

### 6.0 Baseline hiện tại (đo thực tế)
- **ATen path (production)**: 6 iters = 579-685 ms → ~94-114 ms/layer @seq1024. Loss EXACT, mode-1 bite-ident 10.764943 -> 12.684581.
- **Native flash (tkv_flash_attn.c, hiện đã AVX2 hóa)**: 12 iters = 2125-2578 ms → ~177-215 ms/layer. Loss EXACT, mode-1 bite-ident (giá trị tương đương — cả hai đều vẫn EXACT).
- **Chênh lệch**: ATen khoảng 2x nhanh hơn — đây là mục tiêu phải thu hẹp.

### 6.1 Đã làm (Phase 1 — vectorize transcendental)
- `static inline __m256 m256_exp_ps(__m256)` — fast expf AVX2 polynomial (deg-6), clamp [-87.3, 88.7], scalef qua `slli_epi32(+127, 23)`. KHÔNG cần bit-exact (waiver §0.125).
- `static inline void exp_row_fast(float* qk, float newmax, int kvb)` — vectorize `qk[j] = expf(qk[j] - newmax)`.
- `static inline float exp_sum_row_fast(float* qk, float newmax, int kvb)` — vectorize `bsum += e` reduction + store.
- `dot64`, `fma_row`, `mul_row` đã AVX2 (có sẵn từ trước).
- Thay thế 2 bsum-loop (pass 1 recompute + pass 2 recompute attn) qua helper.
- **Số liệu sơ bộ**: native ~177-215 ms/layer (dao động ±10% (noise); ATen ~94-114 ms/layer — ATen vẫn thắng).
- **Production vẫn đi ATen** (`torch_attention_bwd`); native chỉ là fallback (`attention_bwd_fused` IL builtin).

### 6.2 Cần làm tiếp
- **Phase 2**: profile & loại bỏ branch/throw/mask overhead trong hot loop:
  - Vectorize causal mask (không branch `u > t`) bằng `_mm256_cmp_ps` + blend.
  - Vectorize `gA = dotBN(d_row, V)` và `ds = attn * (gA - dsum)`.
  - Pass 1 dùng `_mm256_max_ps` theo chunk 8 để tìm `rmax`.
  - Vectorize `_mm256_log_ps` nếu cần cho `lse = mx + logf(sm)` (ln-1 lần/jalanc/row).
- **Phase 3**: wire production — `TKV_FLASH_OFF=0` → `attention_bwd_fused` native thay ATen trong `gpt2_attention_bwd` (đã có dispatcher `_pt_flash_slots`).
  - Quyết định dựa trên benchmark cuối: giữ ATen nếu native vẫn chậm hơn; switch production nếu native ≤ ATen.
- **Phase 4**: Re-baseline gate:
  - loss EXACT (có thể drift ở ULP level — waiver s0.125);
  - mode 1 bite-ident — phải đạt giá trị trước §0.126 (10.764943 → 12.684581) hoặc re-record;
  - gate 9/10, trainer 15/15 phải xanh.
- **Phase 5** (tùy chọn): áp dụng pattern cho các shape nhỏ (seq8/256), nơi fixed overhead ATen dominates — native có thể thắng ở đó.

### 6.3 Rủi ro & ràng buộc
- Loss kommer trôi vĩnh viễn nếu chuyển production sang native — re-baseline bắt buộc.
- Mã C AVX2 dễ bị dính mmx/mxcsr; mỗi thay đổi phải test loss EXACT + gate.
- `m256_exp_ps` hiện ULP ~2 — đủ khi waiver; nếu return bit-exact cần Sleef-style table lookup (thêm công).

### 6.4 Win condition Phase 2-3
- Native ≤ ATen (bwd-attention portion): đóng ≥ 1.1s/step.
- Loss EXACT (drift ≤ ULP tolerance) + mode-1 giữ 12.684581 hoặc re-record.
- Gate 9/10, trainer 15/15 giữ nguyên.

### 6.5 Phạm vi công việc (giấy tờ)
Các file sẽ chạm:
- `Release/tools/simd_native/tkv_flash_attn.c` — sẽ fill exp_sum_row_fast selection + mask blending; có thể tách pass-2 thành 2-pass fused khi nhảy sàn.
- `tv/training/tv.training.gpt2_ref124.tkv` + `tv/tests/tv.training.gpt2_ref124.tkv` (mirror) — chỉ Phase 3.
- `Release/runtime/tkv_simd_native.dll` — rebuild theo mỗi commit.
- Commit series theo phase cho phép revert clean nếu Phase 2 fails quality gate.

