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
| P1.1 | **Bwd ghi thẳng vào flat `opt.g`** (tính offset theo thứ tự ps) — bỏ t_zeros grads 28 tensor + vòng GRAD_ACC | −2.5 s — **CÒN LẠI, việc nên làm tiếp** |
| P1.2 | ~~Fuse unflatten vào AdamW step~~ | **REVERT — đã probe (`simd_probe_ah`, best-of-5, 124M): fused mọi hình thức ĐỀU CHẬM HƠN 2 vòng riêng (4516 ms): inline-while 6468 / inline-for 5453 / bỏ ghi-d 4704 / function-level 7547. Vào lúc probe đã bắt được 2 bẫy: (1) `for jj in range(sz)` sót `jj = jj + 1` từ while cũ → compiler skip nửa phần tử, loss SAI (gate parity bắt được); (2) biến đếm `while` local = TkvInt boxed slow-path (~+1 s, đúng lỗi perf_f32buf ghi nhận). Bài học: `opt.step()` vòng phẳng `range(self.n)` ĐÃ LÀ FORM TỐT NHẤT trên runtime này — đừng gộp.** |
| P1.3 | **Pool buffer grads/activation** + đo GC — `gc_collect()` builtins **có thật** (`tv/benchmarks/compare_rivals.tkv` gọi; REFERENCE chưa liệt kê) nhưng chưa probe đo | −1.3 s + GC churn |
| P1.4 | AdamW f32 — **đã probe (`simd_probe_ae`)**: vòng m/v vectorize P6 ~2.9× (47→16 ms/16M phần tử), nhưng p-loop sqrt bị P6 từ chối (buf 109 > list 78 ms) → chỉ lời ~1-1.5 s nếu redesign nặng | **lùi — không làm vội** |

Kỳ vọng tổng sau P1.1 (nếu −2.5 s đo được): **~21.5-22 s**/bước (P1.2/P1.4 không đóng góp).

### P2 — Floor scalar (cần đo thêm trước khi làm)

- Tách `gpt2_block_bwd` (simd) thành từng vùng: attention dqkv loops, LN bwd,
  gelu bwd, bias-add — probe Z-style, chọn vùng chiếm >1s trước.
- Vòng elementwise hot: thử f32 trước (P1.4 ra kết luận) — nếu chưa đủ,
  **phụ thuộc compiler team**: `simd_*` 3-đối-tượng hiện allocate 16B/phần tử
  (HANDOFF) → không dùng được; cần biến in-place / kernel `adamw`, `layernorm_bwd`.

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
| sau P1.1 (nếu −2.5 s đo được) | ~21.5-22 s |
| sau P2+P3 (nếu compiler team cộng tác) | ~12-15 s |
| PyTorch | 1.0 s |

Khoảng cách còn lại đến PyTorch là **cấu trúc**: MKL đọc thẳng strided (không
copy), 14 luồng, kernel native không boxing per-element. Đóng nốt cần native
kernels từ compiler team — ghi rõ trong HANDOFF khi tiếp.

## 3. Thứ tự thực thi

1. ✅ P0.1-P0.4 — hoàn thành 2026-10-19 (số ở mục P0; gate + hồi quy + bench).
2. ✅ P1.2 + P1.4 — đã probe → kết luận REVERT / lùi (mục P1, có số đo).
3. **TIẾP THEO: P1.1** (bwd ghi thẳng opt.g — probe materialization grads
   trước, rủi ro: parity `gpt2_backward` core) + P1.3 (gc_collect đo GC).
4. P2 probe phân rã scalar → chọn vùng; P3 probe luồng; P4 bench seq1024.

Probe còn dùng được: `simd_probe_x` (build vs gemm), `simd_bench_124m_profile`
(phase), `simd_probe_z` (bwd tách), `simd_probe_y` (luồng).
