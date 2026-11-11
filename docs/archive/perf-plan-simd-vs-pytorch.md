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

### P3 — Số luồng simd_gemm — **ĐÃ PROBE (simd_probe_ao): GIẢ THUYẾT BỊ BÁC BỎ**

Đo bằng `tools/probe_threads.py` (poll `psutil.Process.cpu_times()` của tiến
trình con → cpu/wall = số core đang chạy THẬT; control scalar chứng minh
phương pháp đúng):

| Pha | wall | cpu | ratio | Ý nghĩa |
|---|---|---|---|---|
| **pure `simd_gemm` 768³ × 800** (tile dựng sẵn) | **1.397 s** | **29.63 s** | **21.20** | gemm **ĐÃ đa luồng** (~21 luồng trên 14C/28T) — 539 GFLOPS (flop của tile) |
| **pure `simd_gemm` 64³ × 200 000** (tile đúng shape seq8) | **2.178 s** | **2.156 s** | **0.99** | tile NHỎ → gemm **1 luồng**, chỉ 49.3 GFLOPS (11× chậm hơn 768³) |
| pure scalar 20e6 (control) | 0.369 s | 0.359 s | 0.97 | 1 luồng ✓ (control) |
| **1 bước 124M seq8 (fwd+bwd)** | **28.77 s** | **28.77 s** | **1.00** | **CẢ BƯỚC hiệu 1 luồng** (fwd 2.09 s + bwd 14.83 s) |

(Đo lại 2026-10-20, best-of-3; lần đo đầu 1.395 s / 29.14 s / 20.9 — khớp
trong ±3% ⇒ phương pháp ổn định.)

Kết luận (đảo ngược giả thuyết ban đầu):

- **KHÔNG cần issue “gemm chỉ 1 luồng”** cho compiler team — native gemm đã
  dùng ~21 luồng khi tile đủ lớn (768³).
- **Nhưng ở seq8, P0.4 chọn T=64 (min(m,n) m=8) ⇒ tile 64³ ⇒ gemm rơi về 1
  luồng, 49.3 GFLOPS.** Đây là điểm giao giữa “tối ưu build” và “đa luồng”:
  thể tích build gần như không đổi khi đổi T (V(T) = (⌈m/T⌉+⌈n/T⌉)⌈k/T⌉T²),
  nên có thể nâng T lên (vd 768) để gemm chạy đa luồng mà không tăng build
  nhiều — **cần probe riêng** (P5).
- Nút thắt ở seq8 là **mọi phần NGOÀI gemm đều 1 luồng**: dựng tile
  (`append`), attention vòng scalar, AdamW/grad-copy. Ratio 1.00 của cả bước
  ⇒ phần gemm chỉ ~4% wall time (45 GFLOP tile ÷ 49.3 GFLOPS ≈ 0.9 s).
- Hệ quả cho các mục sau: **mọi tối ưu tiếp phải nhắm phần 1 luồng** (giảm
  số phần tử phải append/boxing, hoặc đẩy chúng xuống native) — không phải
  tăng luồng cho gemm (đã hết đường).

### P4 — Mức seq thực (1024) — **ĐÃ ĐO (bench seq256/seq1024 + torch tier D)**

Bench: `tv/tests/simd_bench_124m_seq.tkv` (cau hinh seq qua
`tv/tests/_seq_cfg.txt` — 1 file .tkv cho moi seq) + `tools/bench_torch.py`
tier `d` (cung data: `ids[i] = (i*37+11) % 50257`, `pos[i] = i`).

| seq | tkv SIMD (full step) | torch (full step) | **tỉ lệ** | tkv fwd+bwd | torch fwd+bwd | tỉ lệ fwd+bwd |
|---|---|---|---|---|---|---|
| 8 | 22.78 s | 1.0216 s | 22.3× | — | — | — |
| 256 | **109.4 s** | 1.4262 s | **76.7×** | — | 0.8228 s | — |
| 1024 | **674.6 s** | 3.8401 s | **175.7×** | **666.2 s** (fwd 120.0 + bwd 546.2) | 3.2387 s | **205.7×** |

(Số torch: median 2-3 lần chạy, tier `d`/`c`; lần đo đầu 1.000 / 1.529 /
3.846 s — khớp trong ±2% ⇒ tỉ lệ ổn định. Ghi chú: plumbing optimizer của tkv
= 674.6 − 666.2 = **8.4 s** — khớp với ~5.5-7 s đo ở seq8 ⇒ phần này **không
phụ thuộc seq**.)

**Kết luận P4 (đảo ngược giả thuyết “khoảng cách thu hẹp rõ”):**

- **Khoảng cách KHÔNG thu hẹp mà NỞ RỘNG theo seq**: 22.3× → 76.7× → 175.7×.
  Torch ở seq8 còn overhead-bound (1.02 s cho 8 token) nên seq1024 chỉ 3.84 s
  (3.76×); tkv nhân 30× (22.78 → 674.6 s).
- **Xác nhận giả thuyết P2 về head GEMM** (probe AN, 124M thật, best-of-2):

  | n | block fwd | block bwd | head logits GEMM | head GFLOPS (flop hữu ích) |
  |---|---|---|---|---|
  | 8 | 125 ms | 657 ms | 531 ms | 1.16 |
  | 64 | 406 ms | 1578 ms | 1031 ms | 4.79 |
  | 256 | 1375 ms | 5062 ms | 1187 ms | 16.65 |

  Head GEMM **build-bound** (dựng tile), không phải flop-bound: seq ×32 thì
  thời gian chỉ ×2.2, hiệu suất hữu ích ×14. Ở seq1024 head **không còn là
  nút thắt**. (Đo lại 2026-10-20: lần đầu 125/719/563, 453/1844/1171,
  1610/5984/1266 — cùng xu hướng, ±10% do nhiễu máy.)
- **Nút thắt thật ở seq1024 là phần scalar O(n²)**: attention (vòng per-head
  scalar) + proj (n·h² scalar) + build tile + plumbing — tất cả **1 luồng**
  (P3). Torch làm chúng bằng kernel vectorized/đa luồng ⇒ gap nở rộng.
- Hệ quả: **ưu tiên tối ưu tiếp theo = đưa attention/proj về GEMM (smm) hoặc
  native**, không phải chỉnh gemm (đã đa luồng, P3).

### P5 — Đưa attention + proj về `smm` — **ĐÃ PROBE (simd_probe_ap): XÁC NHẬN, CÓ LỢI**

Probe `tv/tests/simd_probe_ap.tkv` (124M thật, layer 6, **2 bản copy giống hệt**
code library + biến `mode` để tắt từng vùng, tự kiểm bằng self-check
copy-vs-library). Cấu hình: `tv/tests/_seq_cfg.txt` (dong 1 = n, dong 4 = số
lần lặp để trung bình).

**Kết quả (ms / layer / 1 lần attention):**

| Vùng | n=8 fwd | n=8 bwd | n=1024 fwd | n=1024 bwd | **Prototype GEMM (n=1024)** |
|---|---|---|---|---|---|
| qkv linear (SIMD) | 26 | 26 | 187 | 187 | — |
| **core O(n²) scalar** | ~0 | ~1 | **5 858** | **15 999** | 756 (12 head: score 16 + ctx 47) |
| **proj scalar (n·h²)** | **31** | — | **3 970** | — | **78** |
| **dctx scalar (n·h²)** | — | **19** | — | **2 204** | **47** |
| **dw_proj scalar (n·h²)** | — | **51** | — | **7 032** | **78** |
| attention total (library) | 54 | 225 | 9 188 | 24 703 | — |

**Tính đúng (self-check):** copy mode 0 = library (n=8: 54.0 vs 54.1; n=1024:
10 015 vs 9 188 — chênh 9% do nhánh `mode` trong vòng hot). Tổng các vùng
= tổng library ✓.

**Cross-check scale (xác nhận đo đúng):** các vùng n·h² scale **tuyến tính**
theo n → proj 3 970/128 = 31.0 ms ở n=8, đo được **31.1 ms** ✓; dctx
17.2 vs 19.0 ✓; dw_proj 54.9 vs 51.4 ✓.

**Tiết kiệm P5 (thành thật):**

| seq | fwd/layer | bwd/layer | **tổng 12 layer** | % của bước |
|---|---|---|---|---|
| 8 | 22.7 ms | 39.8 ms | **0.75 s** | 3.3% (22.78 → ~22.0 s) |
| 1024 | 8 994 ms | 14 213 ms | **279 s** | **41%** (674.6 → ~395 s) |

- fwd: (5 858−756) + (3 970−78) = 8 994 ms/layer.
- bwd: (5 858−756) + (2 204−47) + (7 032−78) = 14 213 ms/layer.
- **CẢNH BÁO**: phần `dp/ds/dqkv` của core bwd (15 999 − 5 858 = **10 141
  ms/layer**) **VẪN SCALAR** nếu chỉ làm P5.1 — con số 41% là trần của
  P5.1, không phải của P5 đầy đủ.
- Ở seq1024, attention scalar = 12 × (9 188 + 24 703) = **406 s / 666 s
  fwd+bwd = 61%** ✓ đúng gia thuyết P4.

**SỬA ƯỚC LƯỢNG SAI CỦA P2:** P2 kết luận “sửa scalar attention (dctx/dw_proj
→ smm) chỉ lời ~1 ms/block — không làm”. **SAI ~6×**: probe AP đo được 70
ms/layer ở n=8 (dctx 19 + dw_proj 51) ⇒ ~5.8 ms/block, tức ~0.5 s trong bwd
124M. Nguyên nhân: ước lượng cũ dùng “10M MAC ≈ 1 ms” (≈10 GMAC/s) trong
khi scalar thật chỉ **~134 MMAC/s** (đóng hộp f64, 1 luồng — đúng P3).

**Kết luận P5: ĐÚNG — làm.** Đây là nút thắt lớn nhất còn lại trong repo
(attention scalar 1 luồng), và **không cần compiler team** (chỉ cần `smm`
đã có). Chia 2 bước:
- **P5.1** (ít rủi ro): proj / dctx / dw_proj → `smm` (n·h²).
- **P5.2** (rủi ro cao hơn): score/ctx theo từng head → `smm` + softmax
  nhân mask causal (core O(n²)); phần `dp/ds/dqkv` vẫn scalar.

### Bài học về đo (từ probe AP — áp dụng cho mọi probe sau)

1. **`now_ms()` có độ phân giải ~15.6 ms** (đồng hồ Windows) → ở seq8, một
   lần đo là vô nghĩa (mọi số đều là bội số 15.6). Phải **trung bình N
   lần** (probe AP đọc `reps` ở dong 4 config) — 24 lần ở n=8 cho số sạch.
2. **Vòng hot của library dùng LIST cục bộ** (`ctx[r]`, `qv[d]`), KHÔNG dùng
   `tensor.data[...]` → bản copy dùng tensor indexing **chậm ~1.8×** → phép
   “total − qkv − proj” ra số **ÂM**. Copy phải giống hệt cách truy cập.
3. Không đo từng token (3 now_ms/token × 1024 token → lỗi lũy cộng hàng
   trăm ms): đo **cả hàm** rồi tách bằng biến `mode`.

### P5 — Đưa attention + proj về `smm` — **ĐÃ PROBE (simd_probe_ap): XÁC NHẬN, CÓ LỢI**

Probe `tv/tests/simd_probe_ap.tkv` (124M thật, layer 6, **2 bản copy giống hệt**
code library + biến `mode` để tắt từng vùng, tự kiểm bằng self-check
copy-vs-library). Cấu hình: `tv/tests/_seq_cfg.txt` (dong 1 = n, dong 4 = số
lần lặp để trung bình).

**Kết quả (ms / layer / 1 lần attention):**

| Vùng | n=8 fwd | n=8 bwd | n=1024 fwd | n=1024 bwd | **Prototype GEMM (n=1024)** |
|---|---|---|---|---|---|
| qkv linear (SIMD) | 26 | 26 | 187 | 187 | — |
| **core O(n²) scalar** | ~0 | ~1 | **5 858** | **15 999** | 756 (12 head: score 16 + ctx 47) |
| **proj scalar (n·h²)** | **31** | — | **3 970** | — | **78** |
| **dctx scalar (n·h²)** | — | **19** | — | **2 204** | **47** |
| **dw_proj scalar (n·h²)** | — | **51** | — | **7 032** | **78** |
| attention total (library) | 54 | 225 | 9 188 | 24 703 | — |

**Tính đúng (self-check):** copy mode 0 = library (n=8: 54.0 vs 54.1; n=1024:
10 015 vs 9 188 — chênh 9% do nhánh `mode` trong vòng hot). Tổng các vùng
= tổng library ✓.

**Cross-check scale (xác nhận đo đúng):** các vùng n·h² scale **tuyến tính**
theo n → proj 3 970/128 = 31.0 ms ở n=8, đo được **31.1 ms** ✓; dctx
17.2 vs 19.0 ✓; dw_proj 54.9 vs 51.4 ✓.

**Tiết kiệm P5 (thành thật):**

| seq | fwd/layer | bwd/layer | **tổng 12 layer** | % của bước |
|---|---|---|---|---|
| 8 | 22.7 ms | 39.8 ms | **0.75 s** | 3.3% (22.78 → ~22.0 s) |
| 1024 | 8 994 ms | 14 213 ms | **279 s** | **41%** (674.6 → ~395 s) |

- fwd: (5 858−756) + (3 970−78) = 8 994 ms/layer.
- bwd: (5 858−756) + (2 204−47) + (7 032−78) = 14 213 ms/layer.
- **CẢNH BÁO**: phần `dp/ds/dqkv` của core bwd (15 999 − 5 858 = **10 141
  ms/layer**) **VẪN SCALAR** nếu chỉ làm P5.1 — con số 41% là trần của
  P5.1, không phải của P5 đầy đủ.
- Ở seq1024, attention scalar = 12 × (9 188 + 24 703) = **406 s / 666 s
  fwd+bwd = 61%** ✓ đúng gia thuyết P4.

**SỬA ƯỚC LƯỢNG SAI CỦA P2:** P2 kết luận “sửa scalar attention (dctx/dw_proj
→ smm) chỉ lời ~1 ms/block — không làm”. **SAI ~6×**: probe AP đo được 70
ms/layer ở n=8 (dctx 19 + dw_proj 51) ⇒ ~5.8 ms/block, tức ~0.5 s trong bwd
124M. Nguyên nhân: ước lượng cũ dùng “10M MAC ≈ 1 ms” (≈10 GMAC/s) trong
khi scalar thật chỉ **~134 MMAC/s** (đóng hộp f64, 1 luồng — đúng P3).

**Kết luận P5: ĐÚNG — làm.** Đây là nút thắt lớn nhất còn lại trong repo
(attention scalar 1 luồng), và **không cần compiler team** (chỉ cần `smm`
đã có). Chia 2 bước:
- **P5.1** (ít rủi ro): proj / dctx / dw_proj → `smm` (n·h²).
- **P5.2** (rủi ro cao hơn): score/ctx theo từng head → `smm` + softmax
  nhân mask causal (core O(n²)); phần `dp/ds/dqkv` vẫn scalar.

### Bài học về đo (từ probe AP — áp dụng cho mọi probe sau)

1. **`now_ms()` có độ phân giải ~15.6 ms** (đồng hồ Windows) → ở seq8, một
   lần đo là vô nghĩa (mọi số đều là bội số 15.6). Phải **trung bình N
   lần** (probe AP đọc `reps` ở dong 4 config) — 24 lần ở n=8 cho số sạch.
2. **Vòng hot của library dùng LIST cục bộ** (`ctx[r]`, `qv[d]`), KHÔNG dùng
   `tensor.data[...]` → bản copy dùng tensor indexing **chậm ~1.8×** → phép
   “total − qkv − proj” ra số **ÂM**. Copy phải giống hệt cách truy cập.
3. Không đo từng token (3 now_ms/token × 1024 token → lỗi lũy cộng hàng
   trăm ms): đo **cả hàm** rồi tách bằng biến `mode`.

### P5.2 + P5.3 — **HOÀN THÀNH 2026-10-20** (đo thật + gate xanh)

**P5.2** — score/ctx theo **từng head** → `smm_dot(Q,K)` + scale `1/sqrt(hd)`
+ mask causal (`u > t` → 0) + softmax theo hàng + `smm_fwd(S,V)`.
**Vòng head phải NGOÀI vòng token.** Bản draft 2026-10-20 không dùng được:
chèn nó *trong* vòng token ⇒ mỗi `(t, head)` tính cả ma trận n×n → **O(n³)**,
thiếu scale `inv`, thiếu mask causal, và `import math` lồng trong vòng lặp.

**P5.3** (`dp/ds/dqkv` → `smm`) — cùng cấu trúc vòng head ngoài:
`P = softmax_causal(inv·Q·Kᵀ)`, `dp = dctx·Vᵀ`,
`dot[t] = Σ_d dctx[t,d]·ctx[t,d]` (đồng nhất với `Σ_u P·dp` nhưng chỉ O(n·hd),
và `ctx` đã cần cho `dw_proj`), `ds = P∘(dp − dot)`, rồi
`dq = inv·ds·K`, `dk = inv·dsᵀ·Q`, `dv = Pᵀ·dctx`.
Bản draft cũ sai 4 chỗ: đọc `sc.data` trên **list** cục bộ (list không có
`.data`), truyền `dctxT [n,h]` vào `smm_dot` với `k = hd` (sai stride), đảo
chu/dk, và bỏ luôn nhánh `dw_proj/db_proj` GEMM của P5.1 → `simd=1` cho
gradient `w_proj` = 0.

**Đo A/B/C — cùng process, cùng trọng số, cùng đầu vào**
(`tv/tests/_bench_p53.tkv`; A = 2 hàm `gpt2_attention(_bwd)` chép **nguyên văn**
từ `git HEAD`, chỉ đổi tên `def`; B = mã hiện tại `simd=1`; C = `simd=0`):

| seq | A: P5.1-only | B: P5.1+P5.2+P5.3 | **B/A** | C: scalar | **B/C** |
|---|---|---|---|---|---|
| 8 | 227.7 ms | 232.4 ms | **0.98×** | 725.8 ms | 3.12× |
| 32 | 285.3 ms | 279.3 ms | 1.02× | 2 908 ms | 10.4× |
| 64 | 398.5 ms | 347.8 ms | 1.15× | 5 902 ms | 17.0× |
| 128 | 679.5 ms | 492.0 ms | 1.38× | 11 328 ms | 23.0× |
| **1024** | **20 390 ms** | **5 437 ms** | **3.75×** | 110 016 ms | 20.2× |

Số = attention fwd+bwd của **1 layer 124M**. Tách đôi ở seq1024:
**fwd (P5.2) 4 937 → 1 281 ms = 3.85×**; **bwd (P5.3) 15 453 → 4 156 ms =
3.72×**. seq8/32/64 trung bình 40/8/4 lần (bài học 1 ở trên), seq128/1024
1-2 lần.

- **Giao điểm ≈ seq 16-32.** Dưới đó GEMM lỗ ~2% (tile T=64 bù 0 cho n=8 →
  ~256× dư thừa việc); cái lỗ 4.7 ms/layer = **+56 ms/12 layer ≈ +0.25%** một
  bước train seq8 → không đáng kể. Mặc định vẫn `simd=0`, gate cũ nguyên vẹn.
- **Self-check A≈B≈C**: `|A−B| ≤ 1.9e-9` (fwd), `1.6e-8` (dx), `7.1e-9`
  (dw_qkv); `|B−C| ≤ 6.1e-9` (fwd), `3.3e-8` (dx) → cả ba tính cùng một phép,
  chỉ khác FMA/thu tu tích lũy.

**Gate** — trước P5.2/P5.3 **không gate nào chạy attention GEMM với `simd=1`**
(nên bản draft sai vẫn “xanh” vì không ai đo nó):

- `gpt2_tests` **10/10 → 17/17**: thêm 7 assert `simd=1` đối chiếu **ngay fixture
  PyTorch** (logits/loss + 5 grad) ≤ 2.4e-7.
- `gpt2_full_tests` **20/20 → 33/33**: thêm `SIMD logits/loss vs PyTorch` 124M
  = 2.6e-6 / 0.0; `P5.2 attn fwd n=192` = 1.1e-9; 6 assert `P5.3 attn bwd n=192
  [0..4]` ≤ 3.6e-5. **n=192 > 64 → GEMM chạy đa ô 3×3** (khác hẳn n=8 fixture),
  gọi trực tiếp 2 hàm đã sửa; và **backward TOAN BO 124M `simd=1`** đối chiếu
  ngay 5 mẫu grad vào fixture PyTorch ≤ 8.3e-7 (chạy sau phần scalar để giảm
  đỉnh bộ nhớ — ghi chú OOM ở mục 0.9).
- Không hồi quy: `simd_matmul_tests` 20/20, `gpt2_scope_tests` 9/9,
  `gpt2_train_tests` 5/5.

**Đo cả buớc seq1024** — `simd_bench_124m_seq` mode 0 (fwd+bwd, không
optimizer), 2026-10-20, cùng máy với bảng A/B/C:

- **fwd 25 328 ms + bwd 128 453 ms = 153 781 ms (153.8 s)** với P5.1+P5.2+P5.3.
- Chỉ sửa 2 hàm `gpt2_attention(_bwd)` ⇒ phần **không** attention không đổi —
  đây là nền cho phép A/B cả buớc bên dưới (một biến duy nhất).
- **Không so trực tiếp với số 666.2 s / 674.6 s ở mục P4**: chính mục P4 mô tả
  nút thắt là “proj (n·h² **scalar**)” ⇒ thời điểm đo (commit `0d81a02`)
  **chưa có P5.1**, khác chuẩn mã. Đã đo lại **cùng ngày, cùng máy** bên dưới.

**Bench lại TOÀN DẢI seq — cùng ngày / cùng máy (2026-10-20, lần 2)** — tkv
`simd_bench_124m_seq` (warmup 1 + 1 buớc, `out/bseq.exe` build lại từ nguồn
hiện tại), torch `tools/bench_torch.py d <seq>` (warmup 1 + median n=2, chạy
**2 lần/seq**, spread ≤3%). Cùng token (`ids[i]=(i*37+11)%50257`, `pos[i]=i`),
cùng shapes (vocab 50257 / 12 layer / 768 / 12 head), cùng lr 0.0025 wd 0:

| seq | tkv fwd | tkv bwd | tkv fwd+bwd | torch fwd+bwd | fwd+bwd tỉ lệ | tkv full step | torch full step | **full step tỉ lệ** |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.922 s | 13.890 s | **15.812 s** | 0.302 s | **52.4×** | **22.219 s** | 0.934 s | **23.8×** |
| 256 | 5.719 s | 29.453 s | **35.172 s** | 0.782 s | **45.0×** | **41.218 s** | 1.404 s | **29.4×** |
| 1024 | 25.281 s | 125.344 s | **150.625 s** | 2.987 s | **50.4×** | **154.234 s** | 3.542 s | **43.5×** |

Loss đầu `simd=1` khớp từng chữ số với lần đo trước trong ngày (`10.572291` /
`10.756927` / `10.760399`), buớc 2 seq1024 `13.442912` — cùng chuẩn mã, chỉ khác
trạng thái máy. Số seq1024 lần đo sáng (153.8 / 157.1 s) cao hơn ~2% — nằm
trong dao động máy; **bảng trên là số chốt**.

Đọc thêm từ bảng:

1. **Backward chiếm 83-88%** fwd+bwd của tkv (13.9 / 29.5 / 125.3 s trên
   15.8 / 35.2 / 150.6); torch fwd:bwd đo cùng ngày = 1.05 / 2.03 s (66%).
   Nút thắt tiếp theo nằm ở backward KHÔNG phải attention (LN/gelu/embed-LM-head
   bwd) — đã ghi ở mục P5.3.
2. **Plumbing/optimizer** = full step − fwd+bwd = **6.4 / 6.0 / 3.6 s** (tkv)
   so với **0.63 / 0.62 / 0.55 s** (torch AdamW 124M) — gần như không phụ thuộc
   seq ở cả hai phía, tkv nặng hơn ~10×; ở seq8 nó chiếm **29%** cả buớc tkv.
3. Tỉ lệ fwd+bwd gần phẳng **45-52×** theo seq, còn tỉ lệ full step tăng dần
   23.8 → 43.5× — vì optimizer cố định của torch che bớt phần compute ở seq nhỏ.
4. **seq256: 109.4 s (P5.1-only, đầu ngày) → 41.2 s = 2.65×** nhờ P5.2+P5.3,
   tỉ lệ vs torch 76.7× → **29.4×** — hiệu năng P5.x không chỉ có ở seq1024.

**Đo thẳng A/B cả buớc (2026-10-20)** — hai lần chạy **cùng binary
`simd_bench_124m_seq`**, cùng config `1024/1/1`, chỉ khác 2 hàm attention
(tạm `git show 4a14acc:` về 2 mirror 124M rồi tự khôi phục; `diff` xác nhận
khôi phục nguyên vẹn, backup đã xoá):

| Chuẩn mã | full step seq1024 | tỉ lệ |
|---|---|---|
| P5.1-only (`4a14acc`) | **420.6 s** | — |
| P5.1 + P5.2 + P5.3 | **157.1 s** | **2.68×** (**−263.5 s**) |

Loss hai lần chạy khớp tới chữ số cuối (`10.760399 → 13.44291` /
`13.442912`) → hai chuẩn **tính cùng một phép**, chỉ khác FMA/thu tu tích lũy.

**Phương pháp (đọc số cho đúng):** bảng A/B/C cô lập ở trên (3.75×,
−179.5 s) cho số **thấp hơn** đo cả buớc (2.68×, −263.5 s) — đo một hàm rời
khỏi allocator/cache của pipeline làm ước lượng thấp. **Số 420.6 → 157.1 là
số quyết định**; bảng A/B/C chỉ để tách đóng góp từng nửa (fwd/bwd) và tìm
giao điểm theo seq.

Còn lại **157.1 s**: chưa tách được attention / không-attention bằng đo trực
tiếp. Ước theo tỉ lệ A/B của bảng cô lập → attention ≈ 96 s (≈ 61% buớc),
phần còn lại ≈ 61 s — **con số này CHƯA chốt**, muốn chọn đúng mục tiêu tiếp
hãy chạy `simd_bench_124m_profile` (đo từng pha) trước khi kết luận.

### Kỳ vọng trung thực

| Mốc | Bước 124M |
|---|---|
| trước P0 | 31.3 s |
| **sau P0 (đo thật 2026-10-19)** | **~24.2-24.5 s** |
| **sau P1.1 (đo thật: −1.4 s)** | **22.78 s** (seq8) |
| **đo thật seq256 (2026-10-20)** | **109.4 s** (torch 1.4262 s → 76.7×) |
| **đo thật seq1024 (2026-10-20, commit `0d81a02` — TRƯỚC P5.1, proj còn scalar)** | **674.6 s** (torch 3.8401 s → 175.7×) |
| **sau P5.1 — seq1024 (đo thật 2026-10-20)** | xem mục P5.2 + P5.3 (đo lại A/B/C cùng process) |
| **sau P5.2+P5.3 — seq8 / 256 / 1024 (đo lại toàn dải 2026-10-20, cùng máy với torch)** | fwd+bwd **15.8 / 35.2 / 150.6 s** · full step **22.2 / 41.2 / 154.2 s** |
| PyTorch (seq8 / 256 / 1024, đo lại 2026-10-20) | fwd+bwd **0.302 / 0.782 / 2.987 s** · full step **0.934 / 1.404 / 3.542 s** → tỉ lệ full step **23.8× / 29.4× / 43.5×** (số cũ 1.0216 / 1.4262 / 3.8401 s) |

Khoảng cách còn lại đến PyTorch là **cấu trúc**: MKL đọc thẳng strided (không
copy), 14 luồng, kernel native không boxing per-element. P3/P4 đã **đo** được
cơ chế cụ thể: ở seq1024 phần lớn thời gian của tkv là **scalar 1 luồng**
(attention O(n²) + proj + dựng tile + plumbing), trong khi gemm native đã chạy
đa luồng. Đóng nốt cần **native kernels** (compiler team) hoặc chuyển
attention/proj sang GEMM (P5) — ghi rõ trong HANDOFF khi tiếp.

## 3. Thứ tự thực thi

1. ✅ P0.1-P0.4 — hoàn thành 2026-10-19 (số ở mục P0; gate + hồi quy + bench).
2. ✅ P1.2 + P1.4 — đã probe → kết luận REVERT / lùi (mục P1, có số đo).
3. ✅ P1.1 — hoàn thành 2026-10-19 (−1.4 s thật; số ở mục P1; probe AL
   = grads parity old-vs-flat, probe AJ = GC breakdown).
4. ✅ P2 — probe AM phân rã xong: scalar residue <5% (LN 0ms, attn scalar
   ~1ms/block) → bỏ ý định sửa scalar; nút thật = head skinny GEMM → P4.
5. ✅ P3 — probe AO (`tools/probe_threads.py`): gemm **ĐÃ đa luồng ~21 luồng**
   (768³, 546 GFLOPS); cả bước 124M seq8 **ratio cpu/wall = 1.00 (1 luồng)**
   ⇒ **không** có issue "gemm 1 luồng"; mọi tối ưu sau nhắm phần 1 luồng.
6. ✅ P4 — probe AN + bench seq256/seq1024 + torch tier D: head GEMM **hết là
   nút thắt** (build-bound, hiệu suất hữu ích ×14 khi seq ×32), nhưng gap vs
   torch **nở rộng** theo seq (22.8× → 71.6× → 175.4×) vì attention/proj
   scalar O(n²) 1 luồng.
7. ⏭ P5 — **ĐÃ PROBE (simd_probe_ap): xác nhận có lợi** (mục P5). Bước kế
   tiếp: **P5.1** (proj/dctx/dw_proj → `smm`) rồi **P5.2** (score/ctx theo
   head → `smm` + mask). Mỗi bước = gate `simd_matmul_tests` 20/20 + 4 suite
   hồi quy + bench lại.

Probe còn dùng được: `simd_probe_x` (build vs gemm), `simd_bench_124m_profile`
(phase), `simd_probe_z` (bwd tách), `simd_probe_y` (luồng), `simd_probe_am`
(block_bwd theo vung), `simd_probe_an` (chi phi theo seq), `simd_probe_ao`
(so luong gemm vs ca buoc), `simd_probe_ap` (attention theo vung + prototype
GEMM).
