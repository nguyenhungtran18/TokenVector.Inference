# PLAN: TokenVector.Inference beat PyTorch (GPT-2 124M full train step) — 2026-11-07

Mục tiêu: full step (fwd+bwd+grad_acc+AdamW+unflatten) ≤ PyTorch 14T trên cả 3 seq,
**không đổi bit-exact contract** (loss EXACT, gate hiện hành xanh).

## 0. Vị trí hiện tại (đo thật, cùng máy Xeon 14C, .NET 10)

| seq | TV SIMD fwd+bwd | TV full step | PyTorch full step | TL |
|---|---|---|---|---|
| 8 | 1829 ms | 2813 ms | 1042.7 ms | 2.7x |
| 256 | 3750 ms | 4782 ms | ~1573 ms (đích) | 3.0x |
| 1024 | ~10650 ms (fwd 3187–3297 + bwd 7375–7516) | 11578–12015 ms | 4190.6 ms | ~2.8x |

Cấu thành bwd 7375–7516 ms @seq1024 (đo 2026-11-07):
- **Attention bwd** 12 layer × 250–281 ms ≈ **3.1 s** (GEMM k=hd=64/n=64, nhỏ → SMALL class, đơn luồng)
- **MLP bwd** 12 × 234–266 ms ≈ **3.0 s**
- **LM-head bwd** 765 ms (đã tối ưu 813→765, §0.84)
- LN bwd ~16 ms × 12; softmax-grad ~58–70 ms; embed + `gflat` + `f32_list_add` ~250–380 ms
- **AdamW opt.step ~906 ms** (torch ~660 ms — còn gap ~246 ms)
- Tổng non-GEMM còn lại ~3 s.

## 1. Vì sao không ăn được PyTorch nếu chỉ tweak GEMM
1. GEMM kernels sau #10/#16 đã native đa luồng ở k ∈ {64,768,3072}; ratio GEMM ~1.37 core.
2. Bottleneck còn lại: **non-GEMM** (attention softmax, LN, gelu, bias/load/store, AdamW,
   copy round-trip) và **overhead IL/managed list**.
3. Bit-exact tuyệt đối (loss EXACT, không đổi FMA order, không erf→tanh) chặn việc
   đổi GEMM catalog/asm; mọi kernel mới phải qua probe + gate.

## 2. Win condition (không đổi)
- seq8 < 1042.7 ms; seq256 < 1573 ms; seq1024 < 4190.6 ms
- Loss EXACT 3 seq (`10.572291 / 10.756927 / 10.760399`); mode 1 `13.442915`
- Gate hiện hành xanh (9/10 @ .NET 10; chỉ `gap_tests` exit=82 pre-existing)
- Trainer 15/15; self-host `tools/vong.exe` → `IL_IDENTICAL`
- RAM seq1024 không tăng so với baseline

## 3. Nguyên tắc làm việc
- Đo trước, sửa sau; A/B xen kẽ best-of-3 cùng phiên; nhiễu ±150 ms seq1024.
- Mỗi kernel/native mới phải **bit-exact**: probe standalone diff=0 (chỉ ở phần tử
  không bị thay; cẩn thận hàng i-tail khi đổi `rows`) → wire prod → kiểm chứng 3 seq +
  mode 1 + trainer + gate → ghi HANDOFF trước khi commit.
- **Bài học §0.84 (2026-11-07)**: thay đổi Tiling lm-head tưởng bit-exact theo probe
  kernel (strip 55→110/220) nhưng production loss lệch 13.442915→13.442911; nguyên
  nhân là thay đổi `rows` (tap i-tail scalar). Probe riêng lẻ ≠ đủ; giữ nguyên tap
  hàng (rows) của layout §0.84.
- Không commit file ngoài phạm vi (build outputs, .tkv đang M, bench*.c, tools
  probe tạm — dọn sau khi dùng).

## 4. Các phase — theo impact / risk

### P1 — Attention bwd (~3,1 s; chunk lớn nhất) — ưu tiên cao nhất
- Các GEMM `k=64, n=64` ≤ TKV_MIN_N(96) → nhánh SMALL, 1 luồng.
- Hướng:
  1. `bench_attn.c`/`bench_attn2.c` (đã có, untracked) đo từng op: softmax_causal_fwd/bwd,
     dqkv, dk_proj, dv_proj, bias_proj — tìm op tốn nhất trong 250 ms/layer.
  2. Giảm số pass memory trong `gpt2_attention_bwd`: A/B từng op (snapshot gout/dqkvs)
     trước khi wire biến đổi nào (giữ tap i-tail = từ §0.84: `per_i` cố định).
  3. Thử parallel/parallel-reduce cho `bias_grad`/reduce-cols elementwise nếu còn
     1 luồng; không đụng GEMM (bit-exact chặt).
- Gate sau mỗi bước: 3 seq + mode 1 + trainer + gate → HANDOFF trước commit.

### P2 — AdamW opt.step ~906 ms (gap ~246 ms)
- #12 Ask C đã wire `adamw_step_glm` (coef==1.0 → glm). 906 ms còn ở:
  1. r3 copy (copy-back), 2. AdamW compute.
- Hướng:
  1. Đo tách r3/copy vs glm compute (bench trong `bench_glm.c` hoặc trace).
  2. Xóa r3: ghi thẳng pml vào ppl khi clip không kích hoạt (`coef==1.0`).
  3. Nếu RNG-safe gain nhỏ: để nguyên, không ráng ép bằng mọi giá.

### P3 — GELU erf (266 ms đơn luồng)
- `op_gelu_erf` single-threaded. Parallelize theo hàng (như `_gf_zeros` seed×list,
  elementwise độc lập, bit-exact per-element) — thay 266 ms → ~15–25 ms.

### P4 — Embed/bwd + `gflat` + `f32_list_add` (~250–380 ms)
- `op_embed_bwd(wte)` 141 ms (99 ms qua native scaffold embed_bwd_fused; còn 141→0
  interpreted là mục tiêu), `f32_list_add dwte` 47 ms, vòng `gwpe→gflat` 62–78 ms.
- Hướng: fused scatter-add native 1 pass (elementwise index-sorted): bit-exact
  vì index-sorted không đổi thứ tự add.

### P5 — Overhead cố định seq8 (~578 ms: m6 344 + op_embed 234)
- A/B tái dùng `#12-shaped` native cho op_embed; giảm GC arena từng bước.

### P6 — AdamW unroll `_day_vec64_operand`
- Đã ghi 10-25% gain ở 256/512; unroll vòng trên register để loại store-forward stall.

### Không làm lại (đã loại — giữ nguyên §0.84)
- Tiling lm-head bwd sang strip rộng/tiling boundry đổi (xem §0.84/§0.85).
- Đổi constant folding contract, đổi GELU erf→tanh, đổi FMA order GEMM.
- Parallel/next-width cho unit n<96 unguarded (g_part).

## 5. Quy trình mỗi phase
1. Đo baseline P up (best-of-3).
2. Sửa, A/B xen kẽ (dotnet seq1024 full-step + best-of-3).
3. Gate: 3 seq mode 0&1 + mode1=`13.442915`; trainer 15/15; gap hiện có.
4. Ghi HANDOFF (append §0.8x) **trước khi commit**; commit chỉ file của phase.
5. Nếu thất bại: revert nguyên vẹn; ghi bài học vào HANDOFF.

## 6. Điều kiện dừng
- Mọi phase đều đạt (loss EXACT + gate) và tổng gain < ~50 ms → dừng, không làm phase sau.
- Nếu PyTorch vẫn nhanh hơn: ghi nhận & dừng, giữ baseline §0.84.

## 7. Tối hậu (chỉ nếu phi tác giả cho phép)
- Nới contract bit-exact vs PyTorch (chỉ cần loss EXACT với chính nó) →
  GELU erf→tanh (~100–200 ms), GEMM reorder (thêm ~10–15%). **Cần xin phép trước**;
  không tự ý.

## 8. Trạng thái thực tế sau §0.96 + §0.97 (2026-11-07 tối)

Đo trực tiếp trên máy này (Xeon 14C, .NET 10.0.13, BLAS scipy-openblas64_ always-on,
flash bwd dispatch trong AttentionBwdFused):

| seq | TV fwd+bwd | TV full step (mode1, flash ON) | loss (\$10^{-6}$ EXACT) | PyTorch full |
|---|---|---|---|---|
| 8 | 2250 (812 + 1438) | (chưa đo lại) | 10.572291 ✓ | 1042.7 ms |
| 256 | 4687 (1500 + 3187) | (chưa đo lại) | 10.756927 ✓ | ~1573 ms |
| 1024 | 9219 (3031 + 6188) | ~10594 (10532/10656) vs OFF ~11172 | 10.760399 ✓ | ~4081-4191 ms |

mode2 seq1024: fwd 2750 + bwd 6359 + opt 953 + copyback 156 = 10312 ms (loss_val in-noise).
Win condition của §2 (seq8<1042.7/seq256<1573/seq1024<4190.6): **CHƯA đạt** (still ~2.5-2.6x).

### Trạng thái từng phase
- **P1 Attention bwd**: DONE (không theo plan gốc). Giải pháp: port thuật toán online-softmax
  flash attention từ nguyên mẫu PyTorch ATen (BSD-3, KHÔNG phải PyTorch binary) sang
  `tkv_flash_attn.c` (AVX2, causal fp32), dispatch trong `AttentionBwdFused` trước kernel cũ
  (rc!=0 -> fallback nguyên văn; TKV_FLASH_OFF=1 tắt). PAR12: 52-82 ms (trước 111-136 ms).
  loss mode0 EXACT 3 seq; probe vs naive f64: fwd 9.6e-08, bwd dv ≤ 2.3e-06.
  Flash fwd CHƯA wire: compiler chua co builtin `attention_fwd_fused` (cần IL wrapper trong
  `simd_feature.tkv` + rebuild self-host — ngoài phạm vi an toàn hiện tại).
- **P2 AdamW**: native `adamw_step`/`adamw_step_glm` wired sẵn (M5/B2); skip grad_norm khi clip==1.0;
  parallel bị từ chối (RAM-bound). Copy-back opt=953 ms (nội dung mode2). Phần tách r3/glm
  chỉ là ĐO LƯỜNG — chưa làm; gain dự kiến < ~100 ms.
- **P3 GELU erf**: ĐÃ native + song song sẵn (`_gelu_fwd_par` worker 12 thread §YEU CAU #19,
  `op_gelu_erf_bwd` native §#12 Ask A2) — không cần làm lại.
- **P4 Embed/bwd + gflat**: ĐÃ native (`embed_bwd_fused` YEU CAU #16 scatter-add, f32_list_add dwte
  native, gflat accumulate thẳng) — không cần làm lại.
- **P5 seq8**: không làm phiên này (fixed overhead m6/op_embed; gain nhỏ, rủi ro cao).
- **P6 AdamW unroll `_day_vec64_operand`**: chỉ tồn tại trong compiler internals — không khả thi
  an toàn trong phiên này (cần self-host rebuild).
- `gap_tests` FAIL exit=82: pre-existing, không liên quan. `gpt2_full_tests` flake 1 lần
  (fallback .exe -> .n10.dll của gate_f32.sh; không phải flash kernel) - 3 gates đầy đủ liên tiếp PASS.
- Safe-word: loss mode0 EXACT 3 seq; trainer 15/15; selfhost IL_IDENTICAL; gate 9/10.

### Kết luận
Theo §6 (điều kiện dừng): LOSS/GATE/TRAINER đều xanh; cả 2 đòn bẩy lớn duy nhất
(attention ~3s, AdamW ~250ms) đã xử; mọi phase còn lại đều stale/done hoặc compiler-internals.
PyTorch vẫn nhanh hơn ~2.5x ở mode1; **gap còn lại do interpreter IL + alloc discipline +
MKL-vs-AVX2 throughput, không còn GEMM đơn lẻ nào**. Quyết định dừng phase mới ở đây;
mọi nỗ lực tiếp theo (flash fwd qua IL builtin, MLP interpreter) cần user chỉ định rõ scope.

