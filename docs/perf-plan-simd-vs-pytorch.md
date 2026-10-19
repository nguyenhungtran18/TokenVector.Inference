# PLAN hiệu năng — tkv train vs PyTorch (2026-10-19)

Số liệu nền: `reports/bench-simd-vs-pytorch-20261019.md`.
Mọi con số dưới đây **đo được** (probe `tv/tests/simd_probe_x/y/z.tkv`,
`tv/tests/simd_bench_124m_profile.tkv`), không suy đoán. Máy: Xeon E5-2695 v3
14C, torch 2.14+cpu (14T). Benchmark chuẩn: 1 buoc train GPT-2 124M seq8
(**PyTorch 999.8 ms | tkv-SIMD 31 333 ms | tkv-scalar 52 578 ms**).

## 1. Nguyên nhân (đo được)

### 1.1 Phân rã 1 buoc 124M (simd=1, tổng 31 079 ms — `simd_bench_124m_profile`)

| Phase | ms | % |
|---|---|---|
| FWD+LOSS | 4 125 | 13% |
| **BWD** | **19 719** | **63%** |
| ZERO_GRAD | 266 | 1% |
| GRAD_ACC (grad -> opt.g) | 1 265 | 4% |
| ADAMW_STEP | 4 954 | 16% |
| UNFLATTEN (opt.p -> ps) | 750 | 2% |

### 1.2 Tách BWD (`simd_probe_z`)

| Thành phần | ms |
|---|---|
| head 3 GEMM: logits 1 328 + dxf 1 235 + dwte 1 718 | **4 281** |
| block_bwd ×12 (simd) | **11 719** |
| block_bwd ×12 (scalar, cung input) | 23 578 |
| embed bwd + accumulate dwte (vòng 38.6M) | 625 |
| t_zeros 50257×768 (38.6M) alloc | 390/vòng |
| phần còn lại (gg softmax scalar, unpack, t_zeros grads) | ~3 000 |

Tổng khớp 19 719 ✓. Đối chiếu forward: chain 12 block simd = 2 578 ms
(block bwd nặng gấp **4.5×** block fwd), logits head = 1 328 ms.

### 1.3 Phân rã GEMM tier A — `simd_probe_x` (fwd 256×768→2304, T=768)

| Thành phần | ms / fwd | % |
|---|---|---|
| **build tile** (A+B+Ct, 3 combo, `append` từng phần tử) | **43.8** | 64% |
| simd_gemm thuan (3×768³ ≈ 2.73 GFLOP → **~350 GFLOPS**) | **7.8** | 11% |
| full fwd | 68.8 | — |
| residual (bias add, tích luỹ Ct→C, t_zeros C) | ~17 | 25% |

**simd_gemm KHÔNG phải nút thắt — dựng tile mới là.**

### 1.4 Số luồng (`simd_probe_y` + GetProcessTimes)

cpu/wall = 2.86 — pha `append` single-thread chiếm phần lớn wall; simd_gemm có
đa luồng nhưng CẦN đo lại riêng pha gemm-thuần (xem P3).

### 1.5 Từ các đo trên — nguyên nhân cụ thể

- **R1 — Dựng tile lặp + đệm vô ích (nút thắt GEMM, ~85% pha build+residual):**
  - **R1a:** `Ct = _zeros_list(T*T)` dựng MỚI mỗi combo `(i,j,p)` trong khi
    `simd_gemm` ghi đè toàn bộ (GAN) → lãng T² append/combo (~33%).
  - **R1b:** A-tile dựng lại cho mọi `j`, B-tile dựng lại cho mọi `i` — không
    cache (mọi `append` = f64 math + conv f32 + boxing, ~121M append/s đo được).
  - **R1c:** `T = _tilek(k)` bỏ qua m,n: dxf m=8 → T=1024 (A-tile đệm 8→1024,
    ~171× thể tích hữu dụng); dx m256/n768 → T=1024. Thể tích build ≈
    m'·n'·k/T với m',n' đã đệm.
  - **R1d:** ô đệm tính lại `(float(0)-0.0)/1.0` mỗi phần tử (nhỏ, sửa free).
- **R2 — Backward nặng gấp ~5× forward:** mỗi `gpt2_linear_bwd` = 2 GEMM
  (dx, dw) + **2 smm_transpose** + `t_zeros(dw)` (≈5 vòng elementwise cho 1 GEMM
  của forward). Trong 11.7s block_bwd: GEMM build chiếm phần lớn ở m=8, song
  **scalar residue lớn** (chỉ 2× khi chuyển scalar: 23.6s) — dqkv/attention bwd,
  layernorm bwd, gelu bwd, bias-add loops, t_zeros — SIMD không đả động.
- **R3 — Plumbing vector phẳng ≈ 8.5s/bước (27%):** t_zeros grads ~1.3s
  (124M phần tử/lần, dựng lại MỖI bước) + zero_grad 0.27 + grad_acc 1.27 +
  unflatten 0.75 + AdamW 4.95 — tất cả scalar elementwise trên ~500M phần tử.
- **R4 — Head vocab GEMM 4.3s** (14% bước): cả 3 đều build-bound, m=8 gầy.
- **R5 — Floor scalar ở block fwd/bwd:** chain 12 block fwd đã 2.6s ở seq8 với
  flop rất nhỏ — chi phí vòng lặp IL per-element (LN, softmax, gelu); tăng seq
  thì floor này ×seq.
- **R6 — Chưa đo:** GC churn (GB alloc/bước: grads 498MB + trung gian) và
  count luồng thật của simd_gemm trong pha gemm-thuần.

## 2. Plan khắc phục (ưu tiên theo tác động/rủi ro)

Mỗi mục hoàn thành = **gate `simd_matmul_tests` 20/20 + 5 gate hồi quy xanh +
chạy lại bench A/B/C, ghi số vào report**. Không làm song song các mục đổi
cùng `simd_matmul.tkv`.

### P0 — Cắt dựng tile — **HOÀN THÀNH 2026-10-19** (đo thật: tier A 321.9→**262.5 ms**, B 31.3→**21.8 ms** — SIMD giờ vượt scalar 26.6!, C 31.3s→**~24.2-24.5 s**; gate simd_matmul 20/20 + hồi quy 4 suite xanh)

| # | Việc | Kỳ vọng |
|---|---|---|
| P0.1 | **Ct tái sử dụng**: 1 `_zeros_list` mỗi lần gọi smm (gemm ghi đè) | −33% build |
| P0.2 | **Cache A-tile**: đổi thứ tự vòng `i→p→j`, dựng `At` chỉ khi `(i,p)` đổi; B-tile tag `(p,j)` nếu đo được lợi | ~−50% build |
| P0.3 | Ô đệm append biến zero đã tính sẵn 1 lần | nhỏ, free |
| P0.4 | Thử variant T theo m,n (T = min(_tilek(k), max(roundup64(m), roundup64(n)))) — **chỉ chấp nhận nếu probe X đo build giảm** (giảm T tăng số combo → volume m'·n'k/T, mọi thứ khác bằng) | trung thực từng case |

Kỳ vọng tổng: tier A 323→**~150-200 ms**; bước 124M 31s→**~24-26 s**.

### P1 — Cấu trúc 1 buoc 124M (`gpt2_ref`, `gpt2_train`, `trainer`)

| # | Việc | Kỳ vọng |
|---|---|---|
| P1.1 | **Bwd ghi thẳng vào flat `opt.g`** | **HOÀN THÀNH 2026-10-19 — −1.4 s thật** (bench C 24.17→22.78 s/bước). Refactor: `gpt2_backward`/`block_bwd`/`attention_bwd`/`mlp_bwd`/`linear_bwd`/`layernorm_bwd` nhận optional `gflat`/offsets (default off = đường cũ parity nguyên ven); bỏ GRAD_ACC + materialization; probe AL xác nhận **maxdiff=0.0 cả 28 nhóm** + loss bit-exact. Ít hơn ước tính 2.5 s vì: dwte/gg/smm intermediates vẫn materialize, copy-add simd ~0.3-0.5 s. **BÀI HỌC**: `tv/tests/tv.training.gpt2_ref.tkv` + `tv/training/tv.training.gpt2_ref.tkv` là **bản NHỎ 256/64/2** — sync mirror phải giữ hằng số riêng; builder offset `acc0` sai hệ số 4*h*h + indent làm loss khác ULP→to (gate parity bắt được, probe AL tìm ra trong 1 lần chạy) |
| P1.2 | ~~Fuse unflatten vào AdamW step~~ | **REVERT — đã probe (`simd_probe_ah`, best-of-5, 124M): fused mọi hình thức ĐỀU CHẬM HƠN 2 vòng riêng (4516 ms): inline-while 6468 / inline-for 5453 / bỏ ghi-d 4704 / function-level 7547. Vào lúc probe đã bắt được 2 bẫy: (1) `for jj in range(sz)` sót `jj = jj + 1` từ while cũ → compiler skip nửa phần tử, loss SAI (gate parity bắt được); (2) biến đếm `while` local = TkvInt boxed slow-path (~+1 s, đúng lỗi perf_f32buf ghi nhận). Bài học: `opt.step()` vòng phẳng `range(self.n)` ĐÃ LÀ FORM TỐT NHẤT trên runtime này — đừng gộp.** |
| P1.3 | **Pool buffer grads/activation** + GC — **đã probe (`simd_probe_aj`, builtins thật: `gc_collect()`/`gc_allocated_bytes()`/`gc_counts()` — REFERENCE chưa liệt kê)**: 1 bước 124M = 23.2s, BWD +1577 MB live (grads 148 tensor materialized), GRAD_ACC free 1577 MB (1.42s), UNFLAT free 2837 MB (gen2+1 — LOH garbage từ boxed f64 AdamW ~496M object/bước). **743 gen0 + 8 gen1 + 7 gen2 collections/bước** — churn ~8 GB/bước là THẬT (giải thích wall time cao hơn cpu time). Pool grads tiết kiệm 1.58 GB sống + churn; AdamW boxing cần native/f32 buffer (phụ thuộc compiler team) | −1.3 s + giảm churn |
| P1.4 | AdamW f32 — **đã probe (`simd_probe_ae`)**: vòng m/v vectorize P6 ~2.9× (47→16 ms/16M phần tử), nhưng p-loop sqrt bị P6 từ chối (buf 109 > list 78 ms) → chỉ lời ~1-1.5 s nếu redesign nặng | **lùi — không làm vội** |

Kỳ vọng tổng sau P1.1 (nếu −2.5 s đo được): **~21.5-22 s**/bước (P1.2/P1.4 không đóng góp).

### P2 — Floor scalar — **ĐÃ PROBE (simd_probe_am): KHÔNG CÒN scalar residue đáng sửa**

Phân rã block_bwd 124M (best-of-3, layer 6, n=8):

| Vùng | ms/block | Ghi chú |
|---|---|---|
| MLP bwd (2 linear_bwd + gelu + fwd-linear) | **391 (57%)** | GEMM-bound ~380 GFLOPS |
| Attention bwd | **219 (32%)** | qkv fwd + rq linear_bwd simd; scalar dctx+dw_proj ≈ 10M MAC ≈ ~1 ms |
| Fwd recompute trong block (LN×2 + attn fwd) | ~77 (11%) | |
| LN bwd ×2 | **0.0** | 37K iteration — không đáng kể |
| **Block tổng** | **687** | ×12 layer ≈ 8.2s trong BWD 13.25s; head 4.3s + embed ≈ phần còn

**Kết luận**: “bwd nặng gấp 4.5× fwd” = do ~3× GEMM volume (mỗi linear_bwd = dx + dw + fwd-linear), **KHÔNG phải scalar residue**. Sửa scalar attention (dctx/dw_proj → smm) chỉ lời ~1ms/block — không làm. Nút thắt thật còn lại:
1. **Head skinny GEMM** (m=8: logits 1.33 + dxf 1.24 + dwte 1.72 = 4.3s chỉ ~232 GFLOPS) → **P4 seq1024** sửa tự nhiên (M=1024).
2. **GC/boxing ~8GB/bước** (probe AJ) — phụ thuộc native/f32 buffer từ compiler team.
3. GEMM volume itself → native kernels (compiler team).

### P3 — Số luồng simd_gemm

- Probe: lặp `simd_gemm` trên CÙNG tile đã dựng (không build xen vào), đo
  cpu/wall như probe Y → biết thật sự dùng bao nhiêu luồng.
- Nếu <14: ghi issue cho compiler team (knob luồng trong dll) — **phụ thuộc
  ngoài**; sau P0 pha gemm chiếm tỉ trọng lớn hơn nên đây là nút thật.

### P4 — Mức seq thực (1024)

- Sau P0/P1: chạy lại bench C với seq1024 (1024× flop, tile m'=1024 → build
 /flop giảm mạnh, tỉ trọng gemm tăng, floor scalar ×seq). Kỳ vọng khoảng cách
  vs torch **thu hẹp rõ**; scalar ở seq1024 không khả thi (ước tính >1 giờ/bước)
  — simd là đường duy nhất.

### Kỳ vọng trung thực

| Mốc | Bước 124M |
|---|---|
| trước P0 | 31.3 s |
| **sau P0 (đo thật 2026-10-19)** | **~24.2-24.5 s** |
| **sau P1.1 (đo thật: −1.4 s)** | **22.78 s** |
| sau P2+P3 (nếu compiler team cộng tác) | ~12-15 s |
| PyTorch | 1.0 s |

Khoảng cách còn lại đến PyTorch là **cấu trúc**: MKL đọc thẳng strided (không
copy), 14 luồng, kernel native không boxing per-element. Đóng nốt cần native
kernels từ compiler team — ghi rõ trong HANDOFF khi tiếp.

## 3. Thứ tự thực thi

1. ✅ P0.1-P0.4 — hoàn thành 2026-10-19 (số ở mục P0; gate + hồi quy + bench).
2. ✅ P1.2 + P1.4 — đã probe → kết luận REVERT / lùi (mục P1, có số đo).
3. ✅ P1.1 — hoàn thành 2026-10-19 (−1.4 s thật; số ở mục P1; probe AL
   = grads parity old-vs-flat, probe AJ = GC breakdown).
4. ✅ P2 — probe AM phân rã xong: scalar residue <5% (LN 0ms, attn scalar
   ~1ms/block) → bỏ ý định sửa scalar; nút thật = head skinny GEMM → P4.
5. P3 probe luồng; P4 bench seq1024 (ưu tiên cao — sửa luôn head m=8).

Probe còn dùng được: `simd_probe_x` (build vs gemm), `simd_bench_124m_profile`
(phase), `simd_probe_z` (bwd tách), `simd_probe_y` (luồng).
