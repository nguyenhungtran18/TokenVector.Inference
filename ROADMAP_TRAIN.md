# ROADMAP — Huấn luyện LLM trong TokenVector (chứng minh thuật toán)

Ngày lập: **2026-10-17** · cập nhật gần nhất: **2026-10-22**
Trạng thái: **ĐÃ CODE — G0→G5 XANH, G6 chưa bắt đầu** (bảng TRẠNG THÁI dưới đây).
Kế hoạch các bước còn lại tới parity (2026-10-22, sau GĐ10): `docs/perf-plan-triz-roadmap.md` **§13** — ~~GĐ11 song song softmax-grad~~ **❌ ĐO BÁC BỎ 2026-10-22** (probe `simd_probe_ce`: SHARED ghi+exp 0.25-0.36× — cùng bệnh AdamW BL; OWN+MERGE 0.56× — merge 1172 ms ≈ serial; HANDOFF §0.32) → ~~GĐ12 LN saves mu/rstd~~ **❌ ĐO BÁC BỎ 2026-10-22** (probe `simd_probe_cf`: LN_BWD chỉ 1.10-1.22×; mu/rstd lưu f32 trong saves → maxdiff grad 3.17e-08, loss đổi chữ số cuối 13.442912→13.442913 — HANDOFF §0.33) → ~~GĐ13 AdamW parallel~~ **❌ ĐO BÁC BỎ 2026-10-22** (probe `simd_probe_cc`: V5 REGISTERS — toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back — vẫn **0.33×** ≈ V2 SHARED+readback 0.30× → read-back không phải thủ phạm, bệnh = phep toan + ghi list CHUNG; V1 OWN-seg 2.60× nhưng V3 merge 1.19× / V5 shared 0.33× đều ăn hết lợi thế; HANDOFF §0.34) → ~~P7b đo GEMM gap vs torch~~ **✅ ĐÃ ĐO — CHỐT 2026-10-22** (tkv native `simd_gemm` **chậm 6-14×** torch.matmul cùng shape cùng phiên (standalone `simd_probe_cd` tái xác nhận 6.3-14.3×): S1 94 vs 7.1 ms, S3 969 vs 129, S4 1781 vs 254, S5 1547 vs 240; wrapper PAR12 chỉ ~1.25× → khoảng cách ở KERNEL, không phải driver; claim "near-peak AVX2" SAI — HANDOFF §0.35). **Trần `.tkv`-only = 10.4× chốt hẳn (GĐ11/12/13 đều bust, P7 mixed-precision bust); parity 1.0× BẮT BUỘC đi qua native/C** — việc còn lại là việc của compiler/native team (kernel GEMM + fused AdamW/elementwise), không phải `.tkv`. **Menu TRIZ: `docs/perf-plan-triz-roadmap.md` §14 — ~~M3 điều tra shape-gap~~ ✅ **M3 ĐÃ ĐO 2026-10-23** (probe `simd_probe_cg`, §14.5): **CHỐT wrapper-bound, kernel đã parity** — pha GEMM 552-588 GFLOP/s pad ≈ torch (S4/S5 GEMM 207/211 ms THẮNG torch 254/240 ms), BUILD ratio cpu/wall **1.00** đơn luồng, BUILD+ACC+ALLOC = **76-89% wrapper** + pad **1.5×**; re-entrancy: 2 thread gọi `simd_gemm` đồng thời → **HANG** (`g_job` global không lock) → GEMM phải giữ 1 entry point. Thứ tự mới: ~~M3a song song ACC~~ **❌ M3A-PIPELINE ĐO BÁC BỎ 2026-10-23** (probe `simd_probe_ch`: pipeline double-buffer build(w+1)∥replay(w) cho `_smm_core_pd` — BIT=0.0 cả 4 shape nhưng speedup chỉ S3 1.02×/S4 0.94×/S5 1.10×, alloc 48 buffer 187 ms ăn hết lãi → dưới ngưỡng wire 1.3×; PD đã song song build nên pipeline chỉ ẩn build(w+1), bottleneck thật = replay tuần tự simd_gemm-reentrancy — HANDOFF §0.38) → ~~M3b T-dispatch giết pad~~ **❌ M3B ĐO BÁC BỎ 2026-10-23** (probe `simd_probe_cb`, §14.6): grid T=256/384/512/768/1024 trên 4 shape P7b — T=256 (pad 1.0×) **tệ nhất mọi shape** (kernel simd_gemm mất throughput ở tile nhỏ: 282 GFLOP/s @256 vs 546-616 @768); T=1024 (pad 1.36×) thắng SUM cả 4 nhưng chỉ 1.18-1.33× — quá mỏng cho ngưỡng wire 1.3×, phải sửa 3 đường code + wave-12 96 MB buffer, wrapper thật chưa đo → **KHÔNG wire**; HANDOFF §0.39) → **M3c tái dùng buffer C** ✅ **ĐÃ WIRE + 84/84 2026-10-23** (probe `simd_probe_ci`, §14.7/HANDOFF §0.40+§0.41): ALLOC t_zeros C S4/S5 = 359-421 ms/call nhưng ZERO_FILL chỉ 63-94 ms (ratio 0.18-0.22); wire bằng **scratch-passing** (`list[f32]` optional `sc5`/`sc4`, không module-var — tôn trọng lệnh cấm `simd_matmul:430`), A/B wrapper thật **BIT=0.0 ~344 ms/bước**; gate 5+17+9+33+20 = **84/84**, loss `13.442912` bit-identical → ~~song song-ACC biến thể gốc~~ **❌ M3A-ACC ĐO BÁC BỎ 2026-10-23** (probe `simd_probe_cw`, §14.8/HANDOFF §0.42: 2 thiết kế — E acc∥gemm, A 12-gemm-trước-rồi-ACC-wave-12 — đều BIT=0.0 nhưng 0.67-0.99× « ngưỡng 1.3; simd_gemm main đã dùng ~22 thread nên acc worker cướp tài nguyên, acc wave-12 không scale; S3 p-reduction chặn) → M1 OpenBLAS (hạ ưu tiên, residual 1.2-1.6× + numerics-torch) → M2 micro-kernel → M4-M7** — M3c đã wire; M3a-variant đã đo bác bỏ; **M1 ĐÃ ĐO BÁC BỎ 2026-10-23** (shim drop-in `tools/simd_native/` = MSVC + OpenBLAS scipy, ABI `SgemmF32/F64` giữ nguyên, 0 dòng `.tkv`: GEMM phase 0.91-1.13× nhưng **full step 1.001× noise** 34,687 vs 34,734 ms, tolerance GÃY 19/20 loss lệch 1.4e-4, numerics ≠ torch vì OpenBLAS scipy ≠ oneMKL — HANDOFF §0.43 + roadmap §14.9). **M2 cũng mất mục tiêu** (CG chứng minh kernel đã parity 552-612 GF); M4-M7 đòi builtin/ABI mới trong compiler → chặn compiler team. Menu repo-local đãi hết ở mức đo; ~~còn 1 mục treo có điều kiện: T=1024 full-wrapper thật (§14.6 footnote)~~ **→ ĐÃ ĐO BÁC BỎ 2026-10-23** (probe `simd_probe_cx`: wrapper thật T=1024 chỉ **1.01-1.12×** trên 4 shape « ngưỡng 1.3 — SUM thủ công 1.18-1.33× không phản ánh wrapper — HANDOFF §0.44 + roadmap §14.10): **menu repo-local khép lại hoàn toàn — 8 chu trình đo (CG/CH/CB/CI/CU/CW/M1/CX), M3c là lever DUY NHẤT wired.** 3 phiên trước đã commit `e7a3eb4`; phiên CX đã commit `7c7ff44`. Spec M4-M7 giao compiler team: `docs/spec-m4-m7-compiler-team.md` (2026-10-24). **→ 2026-10-24 tối: compiler team GIAO M4/M5/M6** (`tkvc-tv.exe` 24/10 16:12 + `tkv_simd_native.dll` 18:39 + REFERENCE 16:05 + 4 test trong repo compiler; M7 chưa có entry REFERENCE) → **chạy lại benchmark, `.tkv` không đổi 1 dòng: gate 135/135 + `trainer_tests` 15/15, full step best-of-3 12.312/17.000/36.719 s (Δ −1.6/−1.8/+0.04% vs 12.516/17.313/36.703 = KHÔNG ĐỔI vì chưa wire), loss 3 seq exact `7.0119243`/`11.807251`/`13.442912`** → việc còn lại = xác nhận builtin gọi được từ repo này rồi wire M4+M5 rồi M6 (HANDOFF §0.45) → **2026-10-25: smoke builtin từ `.tkv` OK (M4 maxdiff 4.5e-08, M5 tol 1e-6 PASS); M5 ĐÃ WIRE** (4 mirror trainer: `adamw_step` native + buffer f32, gate 135/135 + trainer 15/15, full step **seq8 10.953 s −11.0% / seq256 15.234 s −10.4%** loss exact; seq1024 +7.1% do +2GB buffer vượt RAM 11.8GB — bottleneck paging, không phải kernel); **M4 ĐO BÁC BỎ** (probe CJ production-real: S1 0.99× / S3 0.67× / S5 0.80× — native pack nhanh nhưng convert Double↔Single interpreted nuốt hết gain; Tensor production = List[Double], ABI M4 chỉ nhận List[Single] → ask compiler ABI f64); **M6 ĐO BÁC BỎ** (kernel = ref×6, thiếu chuẩn hóa 1/n của GĐ10 — gãy numerics §3.3); M7 chờ REFERENCE entry — chi tiết + ask: spec §6 + HANDOFF §0.46.
Phạm vi: chứng minh đúng thuật toán huấn luyện LLM; việc kiểm thử hiệu năng thực
sẽ diễn ra sau trên GPU Google. Ưu tiên số tuyệt đối với PyTorch.

## TRẠNG THÁI

| Giai đoạn | Trạng thái |
|---|---|
| **G0** | ✅ **XONG** (17-10-2026) — xem bên dưới |
| **G1** tensor core + autodiff thật | ✅ **XONG** (2026-10-17, bù 2026-10-19: `swiglu` fwd/bwd + guard in-place — `swiglu_tests` 3/3, `inplace_tests` 5/5): `tensor.tkv`; `ops.tkv` (12 op fwd); fixture 47; `grad_tests.tkv` 20/20; `tape.tkv` 8/8; `parity_tests.tkv` 4/4; `kernels_reg.tkv` co mau |
| **G2** GPT-2 thu nho (256/64/4/2) + **124M that: parity 20/20 ngoai repo** (`verify_gpt2_124m.ps1`, fixture ~2.3 GB shard tai tv/tests/fixtures-124m — gitignore, xem HANDOFF 0.9) | ✅ **XONG** (2026-10-19: parity 124M THẬT **20/20** — `verify_gpt2_124m.ps1`, logits 5.56e-06, grad wte 4.11e-06; `gpt2_init` normal(0,0.02) tat dinh + duong padding mask khop HF, `gpt2_scope_tests` **9/9**; 2026-10-18): `gpt2_ref.tkv` day du embed/LN row/MHA causal/MLP/tie head + `attention_bwd`/`block_bwd`/`gpt2_backward`; `gpt2_tests` **10/10 PASS** (hidden h0-h2, logits 2.68e-07, loss 0.0; grad wte 1.79e-07, ln1_g0 3.3e-09, attn_w0 3.4e-08, mlp_w1_0 2.5e-08, lnf_g 9.3e-09) vs `torch.autograd`; `run_cases` [8/8] + `--verify-all` xanh. Chốt dung sai 1e-5/1e-4 — bit-exact với HF không thể đạt được vì GELU erf xấp xỉ lệch ~1.5e-7 mỗi bước (numerics-contract) |
| **G3** data pipeline | ✅ **XONG** (2026-10-19): tokenizer GPT-2 BPE parity `8ac487d` (9/9 case + roundtrip); packing + mask + shard + stream `d8a203a`/`pack_tests.tkv` **5/5 PASS** (pack 8x32, mask block 32x32 + causal4, shard roundtrip 16 shard/256 token, stream seed42 checksum 1184745045, determinism seed 42/43) vs reference `tools/gen_pack_fixture.py` |
| **G4** trainer | ✅ **XONG** (2026-10-19, bù 19-10: đã nối vào GPT-2 nhỏ — `gpt2_train_tests` **5/5**, overfit 1 chuỗi loss 5.083 → 0.500, resume bit-exact trên 118.528 tham số): `tv/training/trainer.tkv` — AdamW (decoupled WD) + warmup/cosine + clip norm + grad accumulation + checkpoint bit-exact (trọng số/m/v/RNG), vòng train/eval log loss-ppl-throughput, lệnh `train` trong `cli.tkv`; `trainer_tests` **15/15 PASS** (1 bước khớp PyTorch loss 1.2e-7 / grad norm 0.0 / trọng số 3e-8; overfit 20 mẫu eval_loss 0.01495 acc 1.000; resume 3+save/load+3 **bit-for-bit**). Lệch còn lại vs PyTorch ~2.4e-6 (thứ tự làm tròn f32), không phải 0 tuyệt đối |
| **G5** harness + seam GPU | ✅ **XONG GATE** (2026-10-19): `tools\verify_train_parity.ps1` **một lệnh** → `5/5 khớp (100%)` (gelu_erf 1.2e-7, gelu_tanh 0.0, softmax 1.9e-9, rmsnorm 1.2e-7, CE 0.0) + báo cáo `reports/`; backend giả (`tv/training/backend.tkv`, dispatch `kind`) bị bắt **đúng case** (tanh-thay-erf 1/5, rmsnorm /(n-1) 1/5, +1e-3 5/5) → harness không luôn xanh. Còn lại: notebook Colab |
| G6 phân tán (có thể bỏ) | chưa bắt đầu |
| **SIMD train** (ngoài G — mục 8) | ✅ **XONG** (2026-10-19): `tv/training/simd_matmul.tkv` — `smm_fwd`/`smm_dot`/`smm_transpose` tiled, bọc native `simd_gemm` AVX2+FMA (o T×T, bù 0, tích luỹ từng p-tile); cả GPT-2 ref nhận tham số `simd = 0` (**mặc định scalar — mọi gate cũ nguyên vẹn**), SIMD opt-in. Gate `simd_matmul_tests` **20/20** (linear/forward/loss/28 grad backward/train 3 buoc simd vs scalar ≤1e-4) + 5 gate hồi quy xanh (train 5/5, **gpt2 17/17**, scope 9/9, **full 124M 33/33**, tổng **84/84**). Tốc độ: linear 256x768x2304 **3390→63 ms (~53×)**, head-like 64x768x4096 641→94 ms (~6.7×). 2 bug bắt được: (a) layout B của `smm_dot` ([n][k]) vs B chuẩn ([k][n]) → 3 call site phải dùng `smm_fwd`; (b) biến `gg` trùng tên 2 kiểu (Tensor/f64) trong 2 nhánh simd của `gpt2_backward` → compiler gộp 1 slot, hỏng bộ nhớ 1e15 (đổi tên `gsc`). **P5.2 + P5.3 ✅ (2026-10-20)**: score/ctx và `dp/ds/dqkv` của attention chuyển sang `smm` (vòng **head ngoài** vòng token, mask causal + scale `1/sqrt(hd)`; `dq=inv·ds·K`, `dk=inv·dsᵀ·Q`, `dv=Pᵀ·dctx`). Đo A/B/C cùng process: attention 1 layer seq1024 **20 390 → 5 437 ms = 3.75×** (fwd 4 937→1 281, bwd 15 453→4 156); seq8 **−2%** → giao điểm ≈ seq 16-32, lỗ cả 12 layer ≈ +0.25% một bước seq8. **Đo thẳng CẢ BUỚC seq1024** (2 lần chạy cùng binary, chỉ khác 2 hàm attention): **420.6 s → 157.1 s = 2.68× (−263.5 s)**, loss khớp tới chữ số cuối; torch cùng máy cùng ngày 3.672 s → tỉ lệ 114.5× → **42.8×**. **Bench lại toàn dải seq (2026-10-20, lần 2)**: full step seq8/256/1024 = **22.2 / 41.2 / 154.2 s** vs torch **0.934 / 1.404 / 3.542 s** → **23.8× / 29.4× / 43.5×** (fwd+bwd 15.8/35.2/150.6 s vs 0.302/0.782/2.987 s = 45-52×; backward chiếm 83-88% fwd+bwd tkv; plumbing/optimizer 6.4/6.0/3.6 s vs torch 0.63/0.62/0.55 s). Trước P5.2/P5.3 **không gate nào chạy attention GEMM với `simd=1`** → nay `gpt2_tests` thêm 7 assert `simd=1` vs fixture PyTorch (≤2.4e-7), `gpt2_full_tests` thêm `SIMD logits/loss` 124M (2.6e-6/0.0) + attention n=192 (đa ô 3×3) fwd/bwd simd-vs-scalar ≤3.6e-5 + **backward toàn bộ 124M `simd=1` vs fixture PyTorch** (5 grad ≤8.3e-7). Số ở `docs/perf-plan-simd-vs-pytorch.md` (mục P5.2 + P5.3) |

| **PARITY vs PyTorch** (2026-10-20 — **MỤC TIÊU MỚI của chủ khoa: BẰNG rồi NHANH HƠN PyTorch**, thay mục ~20×) | 🔄 ĐANG LÀM: P5.4 bỏ 2 transpose n²/head + dw=xᵀ·gy (−8.8%), P5.5 backward dùng saves, bỏ forward recompute (−16.0%, bit-exact 0.0 — probe `simd_probe_as`) → full step seq8/256/1024 = **17.5/32.6/118.6 s** vs torch 0.934/1.404/3.542 = **18.7×/23.2×/33.5×**, 84/84 gate. **Sàn đơn luồng ~37 s > 3.542 s → đa luồng là điều kiện cần.** Ladder: **P5.6 ✅** (cache block-fwd, seq8/256/1024 = 16.94/29.74/99.72 s, 28.2×, 84/84) → **P5.7 ❌** (2 phương án lm-head đo bác bỏ; hunk P5.7 bug `dxf` bị bỏ qua `use_flat=1` → đã revert) → **P5.8 ✅ (2026-10-21)**: `smm_fwd_at` trans-A wire lm-head `dwte` (bỏ transpose ggT, bit-exact 0.0); nguyên nhân v2 chậm 2.5× **chốt** bằng probe BC (call site `simd_add`/`simd_sub` trong cùng function làm codegen vòng scalar chậm 2.5-4× dù không chạy → tách `_p58_core_simple`); tích lũy native `simd_add` đo bác bỏ (dxf @1024 3000 vs 2703 ms) → full step **17.16/29.64/98.20 s = 18.4×/21.1×/27.7×**, 84/84 gate, loss seq1024 khớp exact P5.4 → **P6 GĐ1 ✅ (2026-10-21)**: tkv CÓ primitive (`thread_spawn/join`); song song `gpt2_attention_bwd` theo 12 head — quy tắc đo: worker **list RIENG + main merge = 5.2×**, ghi list CHUNG chỉ **1.24×**; worker gemm T=64<96 đi `gemm_rows` trực tiếp (không đụng `g_job` không lock); ATTN_BWD **40.7→17.8 s**, TOTAL_bwd 68.8→44.3 s → full step **17.02/28.67/75.30 s = 18.2×/20.4×/21.3×**, 84/84, loss bit-exact → **P6 GĐ2 ✅ (2026-10-21)**: song song `gpt2_attention` FWD theo 12 head (own-list + main merge, fallback n<128; thêm guard `NHeads` sửa latent OOB mirror nhỏ) → full step **17.08/26.14/63.78 s = 18.3×/18.6×/18.0×**, AQ FWD_BODY **20.8→11.06 s**, 84/84, loss seq1024 khớp exact; còn: MLP/LHEAD/AdamW → **P6 GĐ3 ✅ (2026-10-21)**: builder fix (append→buffer tái dùng, 1.4-2× mọi GEMM, bit-exact 0.0 — probe BH) + `smm_*_pd` wave-parallel lm-head (route <40M serial, ≥40M wave 12; worker KHÔNG simd_gemm) → full step **16.11/25.92/55.47 s = 17.2×/18.5×/15.7×**, TOTAL_bwd 44.9→36.5 s (−18.8%), 84/84, loss 3 seq khớp exact; còn: ATTN_BWD 15.8/MLP_BWD 14.3 → **P6 GĐ4 ✅ (2026-10-21)**: plumbing + gelu parallel THEO SỐ ĐO (probe BK phân rác full-step → BL test AdamW @124M: SHARED-12 = 0.36× / OWN+MERGE = 0.29× **đều chết** → bỏ AdamW parallel, chỉ giữ norm; BM prototype ALL BIT-EXACT: norm 2.79×, loss 5.10×, gelu_fwd 7.33×, gelu_bwd 5.03× — mẫu own-seg + main gop đúng thứ tự; copy SHARED 3.0× → không container-lock, BL[2] chết do f64-boxing 124M; guard theo đo: gelu ≥2M (786K break-even), loss n*V ≥4M, norm ≥4M; 2 bug compiler mới: nested list[f32] append → ArrayTypeMismatchException, main quá lớn → BadImageFormat → tách hàm; wire 10 file trainer/gpt2_ref/ops) → full step **15.06/25.41/50.88 s = 16.1×/18.1×/14.4×**, TOTAL_bwd 36.5→33.5 s (−8.0%), MLP_BWD 14.3→12.3, **135/135** (9 suite + full 33), loss 3 seq khớp exact; → **P6 GĐ5 ✅ (2026-10-21)**: attention **packed buffers** THEO SỐ ĐO (probe BN phân rã ATTN_BWD @1024: whole 1453ms/layer = alloc 12 ctxo+dqo **359ms (25%)** + worker wall 516 (serial ref 4468 → eff **8.66× — worker KHÔNG yếu**) + GEMM serial 469; nguyên nhân: mỗi worker ghi 1 head nhưng driver alloc FULL n·h/n·3h → 11/12 zero-fill + merge copy phần zero; Stage A `_attn_hb_one/_attn_hb_merge` + `_attn_f_one/_attn_f_merge` ghi-đọc packed, serial n<128 nguyên vẹn, bit-exact) → full step **15.19/23.63/45.89 s = 16.3×/16.8×/13.0×**, ATTN_BWD 15.9→11.4 (−28.7%), FWD_BODY 9.13→7.78, TOTAL_bwd 33.5→29.7 (−11.5%), **135/135** (lần đầu gpt2_full crash transient lúc load fixture 230MB → chạy lại 33/33), loss 3 seq khớp exact → **P6 GĐ6 ✅ (2026-10-21)**: MLP_BWD **smm_fwd_at + geluB overlap** THEO SỐ ĐO (probe BO phân rã mlp_bwd @1024: whole 1000ms/layer = r2 391 [dx2 140 + **xT2 94** + dw2 156] + r1 313 + geluB 62 + inner 94 — GEMM 640ms near-peak AVX2 nên chu địch là PHI transpose + SERI geluB; probe BP: smm_fwd_at **maxdiff 0.0** 2 shape, geluB worker ‖ dw2 main 266→187ms maxdiff 0.0 — nested spawn an toàn; wire: gpt2_linear_bwd `xT+smm_fwd`→`smm_fwd_at` (lợi mọi caller), attn worker gDkT/gDvT cùng cơ chế, gpt2_mlp_bwd simd==1 tách r2 dx→[geluB worker ‖ dw2]→db→pack→r1, simd==0 nguyên vẹn; phiên đo nóng drift FWD_BODY +2.3%) → full step **15.31/23.52/45.00 s = 16.4×/16.7×/12.7×**, MLP_BWD 11828→10890 (−8.0%), TOTAL_bwd 29.7→28.8 s (−3.0%), **135/135**, loss 3 seq khớp exact → **P6 GĐ7 ✅ (2026-10-22)**: plumbing **zero_grad overlap + LN-fwd saves** THEO SỐ ĐO (probe BR phân rã plumbing: LN-fwd recompute 24 gói = 359ms @1024, zero_grad 250ms @124M; probe BQ: LHEAD dwte-add PAR12 719ms vs serial 328ms **bác bỏ**; probe BW: main-thread+11 worker eff 5.59× < spawn12 6.48× **bác bỏ**; wire: `gpt2_train_step` spawn zero_grad worker trước forward join trước backward — forward không đụng `opt.g` → bit-exact; `gpt2_block_saved` lưu `a_ln`/`m_ln` → `gpt2_block_bwd` bỏ 24 gói `gpt2_layernorm` recompute, saves format cũ vẫn đọc được) → full step **15.17/22.84/43.89 s = 16.2×/16.3×/12.4×**, **135/135**, loss 3 seq khớp exact; còn: MLP GEMM đã near-peak, ATTN_BWD 11.3 (serial chain + worker 516ms), LHEAD 9.6, AdamW ~4.9s (unflatten+zero chưa chạm) → **P7 ❌ ĐO BÁC BỎ (2026-10-22)**: probe `simd_probe_bx` đo trước rồi mới quyết → **KHÔNG LÀM mixed precision**: máy **Xeon E5-2695 v3 (Haswell-EP)** chỉ AVX2 (không AVX512/VNNI/**BF16**/AMX), native lib **chỉ kernel F32/F64** (`SgemmF32`/`SgemmF64`, grep f16/bf16 = 0 hit), compiler **không có kiểu/intrinsic f16/bf16** ⇒ f16 chỉ còn đường **làm tròn bằng phần mềm** → **lỗi số học relerr 1.08e-3** (vượt cửa sổ parity 2.4e-7 gấp **2.1e5×**) + **chi phí làm tròn ~380-460 ns/ptu** ⇒ AMP best-case **0.047-0.108×** (chậm 9-21×) → **quyết định ⓑ = fp32-vs-fp32, GIỮ contract 2.4e-7**; P7 chỉ sống lại khi có **kernel f16/bf16 native** + CPU hỗ trợ (AVX512-BF16/AMX) hoặc GPU. Kèm **bug compiler mới**: phép chuyển `list[f32]`→`list[f64]` sinh **giá trị sai, im lặng** ⇒ mọi pép so sánh list phải **inline trong main** (bản probe draft báo `maxdiff=813056` là do lỗi này, không phải do GEMM). Chi tiết: `docs/perf-plan-triz-roadmap.md` §12 + `HANDOFF.md` §0.27 → **P6 GĐ8 ✅ (2026-10-22)**: **giảm overhead interpreter trong GEMM** — đọc IL thấy biến cục bộ gán trong vòng lặp bị compiler **box thành `TkvInt`** (struct 16 B); quy tắc (probe `simd_probe_narrow2`): **phép gán đầu tiên ở top-level hàm** + các phép gán sau đều "chứng minh được" ⇒ **native int64**; probe `simd_probe_by2` (đồng hồ `time_perf`) đo A-tile **1.49×**, B-tile **1.43×**, acc **1.08×**, core **1.54×/2.28×** (maxdiff 0.0) → wire `_tile_a_w`/`_tile_b_w`/`_tile_a_at_w` (index chạy) + `_acc_f` cho cả đường serial lẫn wave-12 lm-head; **KHÔNG** đổi thứ tự vòng (đã đo REGRESSION) → full step **12.97/18.13/37.83 s = 13.9×/12.9×/10.7×** (−15.0/−21.1/−15.1%), **135/135**, loss 3 seq khớp exact; còn: interpreter ngoài GEMM (AdamW/unflatten) + mở rộng native kernel (F64) → **P6 GĐ9 ❌ ĐO BÁC BỎ (2026-10-22)**: nhân rộng quy tắc int64 ra vòng ngoài GEMM — scan IL của đường train tìm biến bị box (`_step124.off`, `_tn_one.off`, `_norm_groups.off`, `_ls_one.k`, vòng ngoài `_smm_core`; `AdamW.step` 4.2s + `zero_grad` + gelu + LN **đã dùng `int32 'i'`** → không thuộc phạm vi), probe `simd_probe_ca` @124M (4 lần chạy): unflatten **1.02/0.998/0.991/0.990** (median ≈ 0.99), norm **1.04/0.996/1.002/0.996 ≈ 1.00**, bit-exact 0.0 → **bác bỏ**: cả hai vòng memory-bound (≈3.4 GB/s) nên arithmetic bị ẩn hết — bản "sửa" không nhanh hơn; quy tắc chỉ ăn tiền ở vòng compute-bound; tiết kiệm tối đa ~18 ms @1024 = **0.05%** < nhiễu đo, chi phí wire 7 file mirror → **P6 GĐ10 ✅ (2026-10-22)**: **học PyTorch cho AdamW — gộp 2 div thành 1** (so `torch/optim/adamw.py`: PyTorch chỉ **1 div/ptu**, ta 2; định lý `m/(sqrt(v)/bc2s+eps) == (m*bc2s)/(sqrt(v)+eps*bc2s)`; probe `simd_probe_cb` @124M: V1 hoist `(1-b1)/(1-b2)` **0.93×** ❌, V2 gộp `p` **0.92×** ❌ (JIT/CSE đã tự làm), V3 gộp div **1.068×** maxdiff 1.19e-07 ✅ → wire 4 file `trainer`) → `ADAMW_STEP` 4370→3930 ms (−10.1%), full step **12.52/17.31/36.70 s = 13.4×/12.3×/10.4×** (−2.6/−3.3/−3.1%), loss in ra khớp exact 7 chữ số 3 seq, **135/135 + trainer_tests 15/15 parity PyTorch** — Chi tiết: `docs/perf-plan-triz-roadmap.md` §12 + `HANDOFF.md` §0.28 + §0.29 + §0.30 |

### Kết quả G0

| Việc | Kết quả |
|---|---|
| Spike Linux/Colab | `.tkv` **chạy được** trên Ubuntu/mono. Compiler **không biên dịch được** trên Linux (IL của chính compiler bị mono từ chối + thiếu `ilasm`). Chi tiết: `docs/linux-portability-spike.md` |
| Fixture từ PyTorch | `tools\gen_fixtures.py` (torch 2.14.0+cpu, fp32, seed 0) sinh 9 fixture vào `tv\tests\fixtures\` |
| Đối chiếu phía `.tkv` | `tv\tests\parity_tests.tkv` — **4/4 PASS**, đã nối vào `tools\run_cases.ps1` thành cổng `[4/4]` |
| Hợp đồng số học | `docs\numerics-contract.md` (chờ duyệt) |

Số đo đạt được: `gelu(tanh)` **0.0**, `cross_entropy` **0.0** (bit-exact);
`softmax` 1.86e-09; `rmsnorm` 1.19e-07.
Đồng thời xác nhận bẫy đã dự đoán: GELU tanh-approx lệch PyTorch mặc định **4.12e-04**
→ bắt buộc phải viết GELU erf ở G1.

Cổng hồi quy sau G0: `run_cases` **26/26 + gap 16/16 + parity 4/4**,
`--verify-all` **49/49 module**.

---

## 1. Quyết định đã chốt

| # | Quyết định |
|---|---|
| 1 | Mục tiêu: **full LLM** (pretrain + fine-tune), chỉ cần **chứng minh thuật toán** |
| 2 | Kiến trúc chuẩn đối chiếu: **GPT-2 124M** |
| 3 | Hệ quả so sánh: **golden fixture sinh từ PyTorch**, check-in sẵn vào repo |
| 4 | Cả CPU và GPU đều cần; phần GPU để sau, chỉ phải có **seam đúng** |
| 5 | Việc test trên GPU Google: kế hoạch riêng, không nằm trong tài liệu này |

## 2. Mục tiêu thật (viết không mời)

Sau kế hoạch này, repo **chứng minh được** rằng:

1. Thuật toán backward của từng op là **đúng**, có gradcheck tự động.
2. Một kiến trúc Transformer **đầy đủ** (GPT-2 124M) chạy forward/backward/train
   khớp **PyTorch từng chữ số** trên một cấu hình thực thi tất định (CPU).
3. Có thể **overfit** một tập dữ liệu cực nhỏ tới hội tụ, và **resume** checkpoint
   cho kết quả giống hệt lần chạy không ngắt.
4. Tách được phần thuật toán khỏi phần thực thi, để sau này gắn backend GPU mà
   **không sửa thuật toán**.

Không nằm trong mục tiêu: huấn luyện thật ở quy mô lớn, tốc độ, GPU, phân tán.

## 3. Định nghĩa "khớp 100% PyTorch"

Đây là điểm dễ hiểu sai nhất, nên đặt rõ trước.

| Đường thực thi | Cam kết |
|---|---|
| **CPU xác định** (fp32, seed cố định, thứ tự giảm tử tài liệu hoá) | **bit-exact 100%** với PyTorch |
| **GPU** | bật deterministic algorithm, tắt TF32 → khớp trong dung sai tuyên bố (mục tiêu rel ≤ 1e-5). **Không hứa bit-exact** |

Lý do: GPU giảm theo thứ tự khác CPU, TF32 mặc định trên Ampere+, và atomic
scatter-add không xác định thứ tự. Bit-exact giữa CPU và GPU là điều không tồn tại
ở PyTorch, nên không thể là mục tiêu.

**Hệ quả bắt buộc:** mọi phép giảm tử (sum/mean/matmul) phải có thứ tự tích lũy
**cố định và ghi trong tài liệu**. Không được để thứ tự phụ thuộc vào parallelization.

## 4. Hợp đồng số học (đóng băng — sửa sau này rất đắt)

| Mục | Chốt | Bẫy parity đã biết |
|---|---|---|
| Kiểu trọng số chính | `fp32` | |
| Kiểu tích luỹ | `fp32`, không fast-math | TF32 làm lệch ngay |
| GELU | **erf chính xác** | tanh-approx lệch ~1e-3 so với PyTorch mặc định |
| LayerNorm / RMSNorm | trừ mean/square-mean rồi chia `+eps`; tính bằng fp32 | chỗ đặt `eps` khác nhau là khác số |
| Softmax | trừ max → exp → tổng → chia, theo đúng trục, thứ tự cố định | |
| CrossEntropy | log-sum-exp + NLL, `mean` trên token **không** mask | nhiều token mask vẫn phải nằm trong mẫu số đúng |
| Dropout | **tự implement RNG có seed** | không thể dùng RNG của hệ đối với đường parity |
| Init trọng số | normal(0, 0.02) với đúng thứ tự phát sinh | thứ tự sinh khác → khác số |
| Adam/AdamW | betas 0.9/0.999, eps 1e-8, bias-correction theo step, weight decay tách khỏi gradient | dễ nhầm Adam với AdamW |
| LR schedule | đúng công thức `get_linear_schedule_with_warmup` | |
| Embedding backward | sắp xếp chỉ số trước khi cộng | thứ tự cộng ảnh hưởng số |
| Gradient accumulation | cộng theo thứ tự micro-batch cố định | |

### 4.1 Bẫy riêng của GPT-2 trên HuggingFace — phải biết trước

| Vấn đề | Chi tiết |
|---|---|
| `Conv1D` không phải `Linear` | HF GPT-2 lưu trọng số cỡ `[in, out]`, ngược chiều với `Linear`. Sai chiều là sai số. |
| Không scale attention theo tầng | `scale_attn_by_inverse_layer_idx = False` cho `gpt2` |
| Softmax attention tính ở fp32 | |
| Dropout mặc định | `embd_pdrop=0.1`, `attn_pdrop=resid_pdrop=0.1` → ở chế độ train có **dropout thật**. Xem 4.2. |
| Tokenizer | phải trùng GPT-2 BPE (tiktoken / HF `gpt2`), kể cả cách tách từ bằng regex |

### 4.2 Về dropout — phải tách khỏi mục tiêu parity

Khớp bit-exact cả phần dropout đòi hỏi phải phát lại đúng PRNG của PyTorch
(Philox) và đúng thứ tự tiêu thụ số ngẫu nhiên. Đó là việc riêng, rủi ro cao.

Đề xuất tách:
- **Parity giai đoạn đầu**: chạy với dropout = 0 (tương đương chế độ eval của HF).
  Khi đó parity là khả thi và là bằng chứng thuật toán đúng.
- **Parity có dropout**: mục tiêu riêng, sau. Nếu làm thì phải phát lại Philox.
- Hạt giống tự viết vẫn bắt buộc: để kết quả **tái lập được**, không phải để khớp PyTorch.

## 5. Cơ chế chứng minh — golden fixture từ PyTorch

### 5.1 Nguyên tắc

Không so sánh "kết quả cuối cùng". So sánh **từng tầng**, để khi sai biết sai ở đâu.

### 5.2 Một fixture gồm gì

Với mỗi ca kiểm thử, lưu:

| Thành phần | Kích thướt | Mục đích |
|---|---|---|
| `input_ids` | `[B, T]` int64 | đầu vào |
| `loss` | 1 số fp32 | so sánh tổng |
| `hidden` theo tầng | **thống kê** mỗi tầng (mean, std, L2) + vài phần tử chọn sẵn | khoanh vùng lỗi mà không giữ tensor khổng lồ |
| `logits` | **mẫu thưa**: vài vị trí × top-k xác suất | kiểm tra đầu ra mà không giữ 50257×T |
| `grad` | một số tham số đã chọn, toàn bộ | kiểm tra gradient |
| `config` + `seed` + `phiên bản torch` | JSON | tái lập |

Lý do lấy mẫu: logits GPT-2 124M ở T=1024 là 50257×1024×4B ≈ 200 MB — không thể
check-in. Thống kê theo tầng + mẫu thưa vừa đủ định vị lỗi, vừa nhỏ.

### 5.3 Quy trình

1. Viết script Python sinh fixture từ `transformers` + `torch` (chạy 1 lần trên máy có GPU hoặc CPU).
2. Chốt môi trường: `torch` phiên bản ghi trong fixture; `attn_implementation="eager"`; `fp32`; `device="cpu"`.
3. Check-in fixture vào repo cạnh test.
4. Test của TokenVector đọc fixture và so khớp.

### 5.4 Fixture tối thiểu cho G1 (chỉ op)

Mỗi op một fixture: input → output → gradient. Gradcheck **và** so fixture.
Hai tầng kiểm chứng độc lập: gradcheck bắt lỗi công thức, fixture bắt lỗi khớp với
cách PyTorch **thực sự** làm (thứ tự giảm tử, cách reshape, cách mask).

## 6. Kiến trúc đích

```
                 ┌──────────────────────────────────────────┐
                 │  Toán thuật (không biết backend)          │
                 │  autograd tape · GPT-2 · trainer         │
                 └───────────────┬──────────────────────────┘
                                 │ gọi qua interface thuần
                 ┌───────────────▼──────────────────────────┐
                 │  Kernel registry (tên op → hiện thực)    │
                 ├───────────────┬──────────────────────────┤
                 │  CPU backend  │  GPU backend (để sau)    │
                 │  scalar/SIMD  │  seam đã có: smm_* (opt-in)   │
                 └───────────────┴──────────────────────────┘
```

Yêu cầu cứng: **đổi backend không được sửa một dòng thuật toán nào.** Đó là tiêu chí
để "sau này test trên GPU Google" thực sự khả thi thay vì phải viết lại.

## 7. Các giai đoạn

Quy ước cỡ: **S** ≈ vài ngày, **M** ≈ 1–2 tuần, **L** ≈ 1 tháng trở lên.

### G0 — Spike khả thi + chốt hợp đồng số học  *(S)* — ✅ XONG (2026-10-17)

**Nội dung**
- Chạy thử 1 forward + 1 backward tối giản trên CPU.
- Trả lời câu hỏi sống còn: `tkvc` có biên dịch/chạy được trên Linux/Colab không?
  (nếu không: mono? dotnet? WSL? hay chỉ sinh PE Windows?)
- Ghi tài liệu `docs/numerics-contract.md` chốt mục 4.
- Chốt định dạng fixture và **sinh thử 1 fixture op** từ PyTorch để xác nhận đường
  làm việc trước khi xây gì.

**Gate**
- Forward/backward CPU chạy đúng.
- Câu hỏi Colab có câu trả lời dứt khoát (có/không/khác).
- Đường sinh fixture PyTorch → đọc kiểm tra trong `.tkv` chạy một vòng.

---

### G1 — Tensor core backend-agnostic + autodiff thật  *(L)* — ✅ XONG (2026-10-17, bù 2026-10-19)

**Nội dung**
- Abstraction tensor: shape, dtype, offset, đọc/ghi phần tử.
- **Kernel registry**: đăng ký hiện thực theo tên op. CPU backend hiện tại là
  hiện thực duy nhất.
- **Autodiff thật**: tape op, gradient tích luỹ về node, backward theo topo.
- Backward cho: matmul, add/mul, layernorm, rmsnorm, embedding (index-add có sort),
  softmax, cross-entropy (theo chuỗi + mask), gelu, silu, swiglu.
- GELU chính xác (erf) và bản tanh-approx để đối chiếu.
- Phát hiện in-place nguy hiểm; activation checkpointing (tái tính thay vì lưu).
- Gradient accumulator theo micro-batch có thứ tự cố định.

**Gate**
- Gradcheck pass cho **mọi** op.
- **Mọi op khớp 100% fixture PyTorch** (bit-exact trên CPU).
- `tools\run_cases.ps1` và `--verify-all` vẫn xanh (không hồi quy).

**Kết quả (2026-10-17, bù 2026-10-19): ✅** — gradcheck mọi op (`grad_tests` 20/20 + `swiglu_tests` 3/3 + `inplace_tests` 5/5); khớp fixture ở tol ~1e-7, riêng gelu_tanh + cross_entropy bit-exact 0.0; run_cases + `--verify-all` xanh.

---

### G2 — GPT-2 124M  *(L)* — ✅ XONG (2026-10-19)

**Nội dung**
- Embedding (wte/wpe) + 12 block + head LM + liên kết trọng số (weight tying).
- Đúng công thức HF: `Conv1D` (chiều `[in, out]`), scale `1/sqrt(d_head)`, không
  scale theo tầng, softmax attention ở fp32.
- `n_positions = 1024`, vocab 50257.
- Weight init đúng normal(0, 0.02), thứ tự phát sinh cố định.
- Attention: causal mask + padding mask, và kiểm tra đường mask đúng.

**Gate**
- Forward: loss khớp **bit-exact** với HF GPT-2 124M trên 1 batch cố định.
- Hidden-state thống kê theo tầng khớp → phát hiện sai ở tầng nào.
- Backward: gradient các tham số chọn mẫu khớp bit-exact.

**Kết quả (2026-10-19): ✅** — bản thu nhỏ (256/64/4/2): `gpt2_tests` **10/10** + `gpt2_scope_tests` **9/9**; **124M thật: 20/20** (`tools\verify_gpt2_124m.ps1`, gate 2 phút 49 giây) — h0=0.0, h1..h12 ≤ 2.1e-06, logits 5.56e-06, loss 9.5e-07, grad wte 4.11e-06. Đạt ở **dung sai 1e-5/1e-4** thay vì bit-exact: GELU erf xấp xỉ lệch HF ~1.5e-7 mỗi bước (numerics-contract) — ngưỡng đã chốt, không phải lỗ hổng.

---

### G3 — Data pipeline  *(M)* — ✅ XONG (2026-10-19)

**Nội dung**
- Tokenizer: khớp GPT-2 BPE (kể cả regex tách từ) — kiểm bằng fixture token id.
- Packing nhiều chuỗi thành chuỗi dài `T`, có attention mask đúng.
- Chia shard, prefetch, seed tất định, số thứ tự mẫu tái lập.

**Gate**
- Token id của một đoạn văn bất kỳ khớp HF ** tuyệt đối** (so từng id).
- Cùng seed → cùng thứ tự mẫu, cùng checksum.

**Kết quả (2026-10-19): ✅** — tokenizer GPT-2 BPE parity **9/9** (so từng id + roundtrip); packing/mask/shard/stream `pack_tests` **5/5** (determinism seed 42/43).

---

### G4 — Trainer  *(M)* — ✅ XONG (2026-10-19)

**Nội dung**
- AdamW + warmup/cosine đúng công thức, grad accumulation, clip norm.
- Checkpoint save/load: trọng số + trạng thái optimizer + step + RNG + lịch LR.
- Vòng lặp train/eval, log loss/ppl/throughput.
- Lệnh `train` trong `cli.tkv` (hiện mới có `serve|gen|export|calibrate|latency`).

**Gate**
- **Overfit** tập ~20 mẫu: loss giảm về gần 0.
- **Resume bit-exact**: dừng giữa chừng rồi chạy tiếp cho kết quả y hệt chạy liền.
- Khớp 1 bước train với PyTorch cùng hyper-parameter (bit-exact).

**Kết quả (2026-10-19): ✅** — overfit 20 mẫu: eval_loss 0.015, acc 1.000; resume **bit-exact** tuyệt đối trong .tkv (checkpoint IEEE-754 bit); khớp 1 bước PyTorch ở 1.2e-7 (loss), grad norm 0.0, trọng số 3e-8 — lệch tích lũy ~2e-6 do thứ tự làm tròn f32. Đã nối trainer vào GPT-2 nhỏ: `gpt2_train_tests` **5/5** (overfit 5.08 → 0.50, resume bit-exact 118.528 tham số).

---

### G5 — Harness + seam GPU  *(M)* — ✅ XONG GATE (2026-10-19, còn notebook Colab)

**Nội dung**
- Script sinh fixture (PyTorch) — chạy 1 lần, check-in kết quả.
- Bộ test đối chiếu: 1 lệnh cho báo cáo tỉ lệ op/khối khớp.
- Notebooks Colab (để dùng sau): chạy CPU reference vs PyTorch GPU, rồi bật backend GPU.
- Viết interface backend để khi thêm GPU chỉ cần một module mới.

**Gate**
- Một lệnh → báo cáo `N/N khớp 100%`.
- Thêm backend giả (trả về kết quả sai có chủ ý) → bộ test **bắt được**, chứng minh
  harness thật sự kiểm tra chứ không phải luôn xanh.

**Kết quả (2026-10-19): ✅ GATE** — `tools\verify_train_parity.ps1` báo cáo **5/5 khớp (100%)**; backend giả bị bắt đúng case (tanh-thay-erf 1/5, rmsnorm /(n-1) 1/5, +1e-3 5/5). **Còn ngoài gate:** notebook Colab.

---

### G6 — Phân tán  *(L — có thể bỏ)*

Data parallel (allreduce), ZeRO-1/2, checkpoint phân mảnh.
Với mục tiêu "chứng minh thuật toán", **G6 không cần**: gradient accumulation trên
một thiết bị đã chứng minh được thuật toán. Chỉ làm khi thật sự cần scale.

## 8. Ngoài phạm vi

- Backend GPU, CUDA, OpenCL (chỉ làm seam).
- Phân tán nhiều node.
- ~~Tối ưu tốc độ, đa luồng hoá GEMM.~~ **ĐÃ CHO PHÉP LẠI (2026-10-20, chủ khoa):**
  mục tiêu mới là BẰNG rồi NHANH HƠN PyTorch → đa luồng GEMM là ĐIỀU KIỆN CẦN
  (sàn đơn luồng ~37 s > torch 3.542 s @seq1024 — xem
  `docs/perf-plan-triz-roadmap.md` §12). SIMD GEMM đã xong 2026-10-19
  (mục TRẠNG THÁI "SIMD train", gate 20/20).
- Quantization trong huấn luyện (QLoRA).
- Sinh văn bản sau huấn luyện chất lượng.

## 9. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| `tkvc` không chạy được trên Linux/Colab | **cao** | G0 spike phải trả lời trước mọi thứ |
| Bẫy parity kiến trúc HF (Conv1D, dropout, init) | cao | liệt kê sẵn ở mục 4.1/4.2 |
| Bĩu căn số tích luỹ khi song song hóa | cao | cấm song song hóa trong đường parity |
| Fixture quá lớn để check-in | trung bình | đã xử lý ở 5.2 (lấy mẫu + thống kê) |
| Autodiff không đủ sức cho kiến trúc mới | trung bình | registry op mở, ghi rõ op nào có backward |
| Tokenizer lệch HF | trung bình | fixture token id bắt buộc ở G3 |
| Phạm vi nở ra (muốn làm cả GPU) | trung bình | G6/GPU nằm ngoài phạm vi, ghi rõ |

## 10. Ba việc đầu tiên

1. **G0-spike-1**: kiểm tra `tkvc` trên Linux/Colab — build, chạy, kết luận.
2. **G0-spike-2**: sinh fixture PyTorch cho một op đơn giản (`gelu` hoặc `softmax`),
   nạp trong `.tkv`, so khớp. Xác nhận mọi mắt xích đều thông trước khi xây G1.
3. **G0-doc**: viết `docs/numerics-contract.md` chốt mục 4, đưa để duyệt.

## 11. Quy ước kiểm thử dùng lại

| Cổng | Lệnh |
|---|---|
| Hồi quy toàn repo | `powershell -File tools\run_cases.ps1` |
| Toàn bộ thân hàm | `tkvc <module> <out.exe> --verify-all` |
| Pipeline đầy đủ | `powershell -File build_inference.ps1` |
| Parity GPT-2 thu nhỏ (G2) | `powershell -File tools\verify_parity.ps1` |
| Parity GPT-2 124M (G2) | `powershell -File tools\verify_gpt2_124m.ps1` (SKIP exit 0 nếu thiếu python/transformers) |
| Train parity harness (G5) | `powershell -File tools\verify_train_parity.ps1` |

Mọi gate trong tài liệu này đều phải giữ ba cổng trên ở trạng thái xanh.
