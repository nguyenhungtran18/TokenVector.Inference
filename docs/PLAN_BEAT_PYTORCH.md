# PLAN: TokenVector.Inference beat PyTorch (GPT-2 124M full train step)

Ngay: 2026-11-04 (cap nhat 2026-11-09; them muc 5 - lo trinh toi uu tiep)
Muc tieu: full step <= PyTorch 14T (1042.7 / 1572.7 / 4148.4 ms) o ca 3 seq, gate 135/135 + trainer 15/15 + loss EXACT.

## 0. Baseline hien tai (do moi 2026-11-06, .NET 10)
- TV SIMD:
  - seq8 fwd+bwd 1829, full step 2813
  - seq256 fwd+bwd 3750, full step 4782
  - seq1024 fwd+bwd 11875, full step 12828
- PyTorch:
  - seq8 full step 1060.6, seq8 fwd+bwd (T=1) 530.4, (T=14) 384.7
  - seq1024 fwd+bwd 3599.5, full step 4190.6
- Gap: backward van la phan lon; non-GEMM da qua native (#15/#16); thread scale con chua du 3.7x.

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
## 5. Lo trinh toi uu tiep (2026-11-09, sau HANDOFF �0.115-�0.121)
Nguon so: HANDOFF �0.115-�0.121 + bench transcripts 2026-11-09 (seq1024 mode 0).
Gap con lai vs PyTorch (cung may, torch 2.14.0+cpu MKL 2026.1): ~1.27x.
Con thieu ~415 ms de dat win condition seq1024 (muc 3) ma KHONG dung bit-exact.

### 5.0 Baseline hien tai (so do gan nhat, 2026-11-09)
- TV SIMD seq1024 mode 0: fwd ~1210, bwd ~3344-3375, fwd+bwd ~4563 ms (truoc do 5313 @�0.115, 4671 @�0.119)
- TV seq256 mode 0: fwd ~281-312, bwd ~1313-1454, total ~1766 ms
- TV seq512 mode 0: fwd ~550-750, bwd ~2125-2265 ms; full step mode 1: ~3766-3954 ms
- TV opt.step (mode 2): ~859-875 ms (AdamW native 8-thread, �0.121; standalone 0.317 s - memory-bound)
- Full step tuong duong (mode 2, seq1024): TONG=5328 ms (fwd 1078 + loss 109 + bwd 3141 + opt 875 + copyback 125)
- PyTorch (do 2026-11-06, CHUA chay lai sau �0.115-0.121): seq1024 fwd+bwd 3599.5, full step 4190.6;
  seq8 full step 1060.6. CHU Y: bench_torch d can chay lai sau moi pha de co cap so cung phien.
- Loss van EXACT ca 3 seq (10.572291 / 10.756927 / 10.760399), mode 1 bite-ident 10.764943 -> 12.684581.

### 5.1 Pha 1 - Bit-exact, rui ro thap (lam truoc, dung thu tu)

**P1a. AdamW non-temporal stores + prefetch** (`Release/tools/simd_native/tkv_simd_native.c`,
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

**P1b. `_save_a` cap [n,4h] vao hpool** (nhanh NON-saved `gpt2_mlp` chung mach �0.117/�0.120):
- Hien moi goi t_zeros [n,4h] cho _save_a; production (saved path) khong anh huong nhieu,
  nhung nhanh non-saved + 2 mirror file can don. Du kien: vai chuc ms. Bit-exact (doc/ghi khong doi).

**P1c. Cache transposition A/B cua QKV giua fwd va bwd** (cung 1 step):
- dqkv/dctx re-pack moi lan; luu transpose 1 lan dung lai. Du kien: vai chuc ms. Kiem tra
  hien co pack nao da cache (pack pool �0.119) truoc de khong doi lap. Bit-exact (pure copy).

**P1d. Do lai head-to-head cung may, ghi HANDOFF �0.122**:
- Chay: mode 0 x3 seq + mode 1 seq512 + `tools/bench_torch.py` (seq8/256/1024).
- Cap nhat Muc 0 baseline + bang gap; gate 9/10 + trainer 15/15 bat buoc.

**Du kien Pha 1:** seq1024 full step ~4860-5100 ms (van chua dat 4148.4), opt ~600.
Neu dua seq1024 ve < 4600 ma PyTorch re-measure van ~4190 thi con thieu ~400 ms -> Pha 2.

### 5.2 Pha 2 - Managed glue: block_saved overhead
- Do wall that trong vong `gpt2_forward_saved` loop (timer quan quanh tung call), khong doan.
- Neu qua cao: gop dispatch cua 1 block fwd/bwd thanh it ATen calls hon, hoac native glue
  cho chuoi op lap lai; giu loss EXACT (moi op van dung kernel cu, chi giam chi phi goi/marshal).
- Du kien: 100-300 ms/step. Rui ro: trung binh; can probe + A/B truoc khi merge.

### 5.3 Pha 3 - CAN USER WAIVER (KHONG tu y trien khai)
- **Flash-attention-class attention fwd+bwd**: PyTorch th?ng y boi lam duoc fused attention;
  attention_bwd van la phan lon bwd (~1-1.5 s/step). Doi accumulation order -> loss KHONG EXACT nua.
- **BLAS dispatch lam default** (hien opt-in TKV_GEMM_BLAS=1, �0.92/0.94): output lech ->
  can re-cast + re-gate toan bo.
- **bf16/TF32**: lech numerics, nghiem tuyet doi cua hop dong.
- Ghi ro: neu user waiver cho attention, do day la don danh manh nhat vao gap ~1.27x
  (HANDOFF �0.94: flash co the cat ~1-1.5 s), truoc do moi pha chi la bo chin.

### 5.4 Win condition & gate (khong doi)
- seq8 < 1042.7 ms; seq256 < 1572.7 ms; seq1024 < 4148.4 ms (full step, cung may voi PyTorch).
- Loss EXACT ca 3 seq; mode 1 bite-ident; gate 9/10 (pack/gap pre-existing); trainer 15/15.
- RAM seq1024 khong tang (pool da gom lai; giam allocation moi buoc, khong them).

### 5.5 Rui ro nhat ky
- Luu y: "mode 2" loss_val khac mode 0 (duong bench rieng); khong ket luan win tu mode 2 mot minh.
- A/B phai xen ke 0/1 trong CUNG binary, best-of-3, may dao dong (observed �5%).
- mode 1 la gate gradient: mode 0 co the xanh loss ma gradient van sai; luon chay mode 1 (seq512).
- PyTorch ref 2026-11-06 cu; moi pha PHAI re-run bench_torch de co cap so cung phien.
- `_seq_cfg.txt` line 4 = pool bit; pool=0 cho ra duong t_zeros cu y nhu truoc (A/B trong cung binary).
