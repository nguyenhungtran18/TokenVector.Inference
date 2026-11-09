# PLAN: TokenVector.Inference beat PyTorch (GPT-2 124M full train step)

Ngay: 2026-11-04 (cap nhat 2026-11-06)
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
