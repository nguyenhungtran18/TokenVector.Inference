# TokenVector.Inference Benchmark Journal

> **Trang thai 2026-11-11 (§9.22, cung phien):** full train step GPT-2 124M
> @seq1024 = **4016 ms** vs torch 2.14.0+cpu **4061.1 ms** -> **0.989x**,
> loss EXACT, gate 135/135.  Bang do moi nhat: **§9.21-§9.22**.
> Suite doi thu `compare_rivals.tkv` (§2) la **smoke suite legacy** - cac
> hang so ORT/Torch trong no la literal co dinh, khong phai phep do tren may
> nay; dung dung trich dan. ``docs/HANDOFF.md` la nhat ky ky thuat dung sau moi con so.
Huong dan chi tiet tung buoc: **`benchmark/BENCHMARKING.md` / `benchmark/BENCHMARKING_VI.md`**.

## 1. Cach do (recipe hien hanh)

TKV side (GPT-2 124M; `tv/tests/_seq_cfg.txt` chon `<seq> / <mode> / <steps> / <warmup>`):

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc tv/tests/simd_bench_124m_seq.tkv benchmark/bseq.exe main
Set-Content tv/tests/_seq_cfg.txt "1024`n1`n1`n1"   # seq=1024, mode=1, steps=1
dotnet benchmark/bseq.exe                                # can .runtimeconfig.json canh exe (xem §9.17)
```

`mode=0` = fwd+bwd (doi chieu torch phan d) | `mode=1` = full step (best-of-3, dao dong manh) | `mode=2` = phan ra 5 pha (min-of-2, tin cay cho quyet dinh theo pha).

Torch side (cung phien - LUON do lai torch, khong muon so lieu phien truoc):

```powershell
python benchmark/bench_torch.py d 1024   # tier d: fwd+bwd & full step
python benchmark/bench_torch.py e 1024   # tier e: phan ra pha, T=1 va T=14
```

Log goc cua lan do moi nhat: `benchmark/_bench_matrix_post14_20261111_145719.log`
(script `benchmark/_bench_matrix.ps1`, thu muc `benchmark/` khong commit tru 2 file cua phien do). Log danh so
cu: `out/_bench_matrix_*.log`.

Hardware note: production chay kernel native AVX2/FMA (`tkv_simd_native.dll`,
build tu repo compiler `D:\TokenVector`, P/Invoke tu IL). Duong scalar `.tkv`
thuan van ton tai lam tham chieu ngu nghia. Khong bench voi compiler build
thieu native export; `opt` (AdamW) khong con nhanh noi xet theo clip, xem
§9.19-§9.21.

Quy tac trinh bay: moi bang phai chi ro **cung phien hay khac phien**
(§9.21 muon torch 4130.6 cua §9.18; §9.22 do lai cung phien -> 4061.1).
Mode 1 chi de so voi lich su; mode 2 la so tin cay.

## 2. Legacy suite (compare_rivals.tkv) - chi de smoke

Sau 6 kich ban; 4 dau la benchmark, 2 cuoi la stress:

| Scenario | What it measures |
| :--- | :--- |
| Cold-start & arena alloc | Graph build + arena setup, via `session_from_graph` |
| In-process single-sample latency | Fused linear / elementwise kernels |
| Heap per N inferences | Allocation pressure across repeated `run_single` |
| FP32 vs INT8 throughput | Quantized GEMM vs float reference |
| Hot-path endurance | Continuous inference loop, reports GC counters |
| Concurrent burst | `DynamicBatcher.enqueue` in a loop, reports ok/err |

Iteration counts are deliberately small (latency 500, heap 500, throughput 200,
endurance 2000, burst 500) so the suite finishes in CI. These are smoke/regression
gates, not a performance comparison against rivals.

`stress_concurrent` reports `ok=` and `err=`: an `err` count above zero means requests
failed rather than merely ran slowly.

**Caution:** the `now_ms`/`now_us` helpers compile and run today, but the `gc_*`
arena counters in that suite are unverified stubs - do NOT quote its arena/heap
numbers. ORT_*/TORCH_* constants are fixed literals, not measurements. The suite
prints a `QUICK-RUN SCALE` banner saying the same.

## 3. Recording results

When you run the suite, append a dated table here with:

- Host CPU model and flags
- `tkvc` build id / commit hash
- Scenario, metric, median ~ spread
- Rival baseline command lines for the same machine (ONNX Runtime, llama.cpp, PyTorch) if compared
- Whether the comparison is same-session or cross-session

Until a fresh run is recorded, leave result cells empty rather than copying historical
figures. Same rule for the hardcoded ORT/Torch constants in the source: they are
literals in the file, not measurements from this machine.

---


## 4. Kết quả H4 (2026-11-04) — TokenVector vs PyTorch (best-of-3)

Máy: Windows, .NET 10, PyTorch 2.14.0+cpu (14 thread). TokenVector: `out/bseq_h4c.exe` (H4a+H4c), loss exact.

| seq | TKV full step (ms) | Torch T=1 (ms) | Torch T=14 (ms) | TKV/Torch T=1 | TKV/Torch T=14 |
|---|---:|---:|---:|---:|---:|
| 8 | 3750 | 1405.4 | 1069.9 | 2.67× | 3.50× |
| 256 | 5734 | 4373.5 | 1604.4 | 1.31× | 3.57× |
| 1024 | 16203 | 15164.3 | 4140.0 | 1.07× | 3.91× |

Ghi chú:
- H4a: hoist 6 buffer m6 lên cấp step (bỏ alloc/free mỗi bước).
- H4c: op_embed_bwd dùng counting sort thay bubble sort.
- Gate 135/135 + grad_tests PASS.
- H3 threading: 8T nhanh nhất (15625 ms seq1024) nhưng loss lệch 13.442913–13.442917 (rủi ro exactness).
- H4b/H2: chờ compiler/native team (`docs/COMPILER_TEAM_REQUEST9.md`).


## 5. Cách đo (TokenVector vs PyTorch)

### 5.1 TokenVector (bseq)

Compiler: `D:\TokenVector\Release\dist\tkvc-tv.exe`
Build:
```powershell
& "D:\TokenVector\Release\dist\tkvc-tv.exe" tv/tests/simd_bench_124m_seq.tkv out/bseq_h4c.exe main
```
Runtime config: tạo `out/bseq_h4c.runtimeconfig.json` (net10.0) nếu thiếu.

Chạy 1 cấu hình:
```powershell
Set-Content -Path tv/tests/_seq_cfg.txt -Value "1024`n1`n1"   # seq / mode(0=fwd+bwd,1=full step) / steps
dotnet out/bseq_h4c.exe
```
Best-of-3: lặp 3 lần, lấy min. Log pattern: `BSEQ full-step simd seq=... ms=...`.

### 5.1b Kiểm nền tảng máy trước khi đo (2026-11-05)

Trước mọi kết luận "torch nhanh hơn vì ISA", chạy 2 probe này:

```powershell
# (các probe này đã bị XÓA 2026-11-11 cùng tools/simd_native/ — số đo lịch sử giữ nguyên)
# CPU feature AVX2/FMA + AVX-512: kiểm bằng repo compiler D:\TokenVector (out/m1/, or xem HANDOFF §0.156)
```

Máy hiện tại: **AVX2 + FMA, KHÔNG AVX-512** (`AVX512F=0`, `/arch:AVX512` → illegal instruction),
14 logical = **7 core vật lý + SMT**. Đỉnh đo được: **75.5 GFLOP/s 1 luồng**, **~515 GFLOP/s 14 luồng**
(torch 76.2/476.6; numpy OpenBLAS 75.7). tkv GEMM đạt 34.5 GFLOP/s (46% đỉnh) ở 1 luồng và scale 4.52×.
Micro-kernel: `tools/simd_native/bench_micro.c` (4 biến thể, bit-exact bằng `memcmp`) — **file đã xóa 2026-11-11** (probe scratch; số đo 34.5 GFLOP/s / scale 4.52× là lịch sử).

### 5.1c Phân rã 1 bước theo pha, seq1024 (2026-11-05)

`mode 2` của `tv/tests/simd_bench_124m_seq.tkv` — cùng đường production với `mode 1`
(`sc4/sc5/m6/gflat`), đo riêng 5 pha, best-of-2 mỗi pha:

| pha | 1 luồng | 14 luồng | scale |
|---|---:|---:|---:|
| forward_saved | 8844 | 3000 | 2.95× |
| gpt2_loss | 109 | 78 | 1.40× |
| backward | 24282 | 10156 | 2.39× |
| `opt.step()` | 984 | **937** | **1.05×** |
| copy-back 124.4M | 125 | 110 | 1.14× |

**`opt.step()` không scale** — 937 ms cố định = 6.6% bước 14 luồng. Cả cụm opt+copy+loss
= 1.12 s = 7.9% dù đã 14 luồng.

`gpt2_block_bwd` (1 luồng, seq1024) = 1578 ms: **mlp_bwd 797 (50.5%) · attention_bwd 578
(36.6%) · layernorm_bwd ×2 47 (3.0%) · còn lại 156 (9.9%)**. Σ12 block = 18936 ms; lm-head +
`op_embed_bwd` ngoài block = 5346 ms.

Tỷ lệ GEMM/non-GEMM (suy ra từ 410 GF): 1 luồng ≈ 35/65, **14 luồng ≈ 18/82** — ở cấu hình
đích non-GEMM là nút thắt. Probe: `out/bd1024.exe`, `out/tail1024.exe`, `out/bseq_p2.exe`.

### 5.2 PyTorch

```powershell
python benchmark/bench_torch.py e <seq>      # phase breakdown T=1 và T=14
python benchmark/bench_torch.py d <seq>      # fwd+bwd và full step
```
So sánh apple-to-apple:
- TKV mode 0 (fwd+bwd) ↔ Torch `fwd(+CE)+bwd`.
- TKV mode 1 (full step) ↔ Torch `TOTAL`.


## 6. Kết quả P10 (2026-11-05) — hạ ngưỡng native `k >= 64`, TKV vs PyTorch

Thay đổi: `_smm_core` rẽ native ở `k >= 64` thay vì `k >= 256` (`tv/training/simd_matmul.tkv`
+ 2 mirror). Mọi GEMM `k = 64` (attention bwd) rời đường tiled 256 tile sang native packed.

| seq | TKV trước (ms) | **TKV sau (ms)** | Torch T=14 (ms) | TKV/Torch | Δ TKV |
|---:|---:|---:|---:|---:|---:|
| 8 | 3829 | **2875** | 1009.4 | 2.85× | **−25%** |
| 256 | 5672 | **5218** | 1538.4 | 3.39× | **−8%** |
| 1024 | 16375 | **14516** | 4077.1 | 3.56× | **−11%** |

(best-of-3, 14 luồng; `loss` = 7.0119185 / 11.807271 / 13.442915 — seq8 + seq1024 EXACT,
seq256 lệch 4e-6 do native vs tiled khác ULP, xem HANDOFF §0.72.)

Phân rã seq1024 (mode 2) sau thay đổi, so với torch cùng pha:

| pha | TKV | Torch T=14 | tỉ lệ |
|---|---:|---:|---:|
| fwd + loss | 3344 | 1084.7 | 3.08× |
| backward | 10593 | 2347.9 | **4.51×** |
| opt.step | 1188 | 660.7 | 1.80× |
| copy-back / zero_grad | 141 | 52.2 | 2.70× |
| tổng | 15266 | 4134.0 | 3.69× |

⇒ Nút thắt còn lại là **backward** (4.5×) — chiếm 8.2 s trong 11.2 s chênh lệch; muốn
xuống ≤ torch phải cắt được vùng non-GEMM interpreted của `gpt2_block_bwd` (xem HANDOFF §5.1c
cũ: mlp_bwd 50.5% · attention_bwd 36.6%) hoặc nâng scale luồng 2.3× → 3.7×.

Gate: 9/10 suite PASS; `gap_tests` hỏng `InvalidProgramException` trên fallback .NET 10 —
đã xác nhận có từ trước (stash thay đổi rồi build lại từ HEAD vẫn lỗi). Trainer 15/15 PASS.

Build đo: `dotnet out/bseq_p10_n10.dll` với `tv/tests/_seq_cfg.txt` = `<seq> / 1 / 1`.


## 7. Kết quả đo lại toàn bộ (2026-11-10 15:48) — TKV vs PyTorch, cùng phiên

Máy: **Intel Xeon E5-2695 v3 @ 2.30 GHz**, 14 core vật lý / 28 logical, Windows 10 19045,
.NET 10.0.402, PyTorch **2.14.0+cpu threads=14**. TKV: `out/bseq.exe` build **15:26:27**,
git HEAD `38bfb04`. Script: `benchmark/_bench_matrix.ps1`, log `out/_bench_matrix_20261110_154851.log`.
TKV = best-of-3 (min). Torch = median of 2 (harness `tier_d` chạy `timeit(fb,1,2)`).

### 7.1 Bảng chính

| seq | TKV fwd+bwd (mode 0) | Torch fwd+bwd | tỉ lệ | TKV full step (mode 1) | Torch full step | tỉ lệ |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 766 | 314.1 | 2.44× | 1437 | 999.6 | 1.44× |
| 256 | 1360 | 852.5 | 1.60× | 2141 | 1490.1 | 1.44× |
| 1024 | **4016** | 3453.7 | **1.16×** | **4578** | 4040.2 | **1.13×** |

seq1024 full step **1.13×** — tốt hơn 1.22× ghi ở §0.133. `loss` TKV 3 seq đều EXACT
(10.572291 / 10.756927 / 10.760399), khớp giá trị đã ghi.

### 7.2 Phân rã pha seq1024 (TKV mode 2 vs Torch `tier_e` T=14)

| pha | TKV | Torch T=14 | tỉ lệ |
|---|---:|---:|---:|
| fwd + loss | 1172 (1062+110) | 1086.8 | 1.08× |
| **backward** | **2844** | **2352.0** | **1.21×** |
| `opt.step` | **484** | 655.2 | **0.74×** (TKV nhanh hơn) |
| copyback / zero_grad | 125 | 62.0 | 2.02× |
| **TỔNG** | **4625** | **4130.6** | **1.12×** |

### 7.3 Đính chính §0.133: `opt.step` KHÔNG còn là nút thắt

§0.133 ghi `opt` = 875 ms và gọi `grad_norm` (437 ms) là nút thắt. Đo lại trên build
15:26:27: **`opt` = 484 ms**, và TKV **nhanh hơn** torch (655.2) ở pha này.

Lý do: `_mk_opt` truyền `clip = 1.0`, `AdamW.step()` chỉ gọi `grad_norm()` khi
`clip != 1.0` (`tv/training/trainer.tkv` L202). Bench **không chạy** `grad_norm`.
Con số 875 của §0.133 không tái lập được trên build này — cần coi 484 là số đúng
cho đường bench, và 875 là số của một build/đường khác.

### 7.4 Toàn bộ gap còn lại nằm ở backward

4625 − 4130.6 = **494 ms**. Trong đó backward chiếm **492 ms** (2844 − 2352).
fwd+loss chỉ 85 ms, copyback 63 ms, còn `opt` TKV đang **thắng** 171 ms.

⇒ Muốn xuống ≤1.0× chỉ cần cắt backward ~1.21× → 1.0×. Không cần đụng `opt`, không
cần đụng lm-head. Đây là mục tiêu hẹp hơn nhiều so với "ba ưu tiên" của §0.131.

### 7.5 Cảnh báo về cách đọc bảng này

- **Đây là so sánh THỜI GIAN, không phải parity số học.** TKV init bằng `_gi_noise`
  (LCG), torch init bằng `torch.manual_seed(7)` + `GPT2LMHeadModel` mặc định → loss
  khác nhau (10.76 vs 9.89). Số tham số khớp (124,439,808 cả hai), ids/pos giống hệt,
  nên khối lượng tính toán như nhau. Parity được kiểm riêng: `gpt2_full_tests` 33/33.
- **seq8 có tỉ lệ xấu nhất (2.44×)** vì chi phí cố định theo tham số (lm-head `t_zeros`
  154 MB, copyback 496 MB, opt) chiếm gần hết, mà đây đúng là chỗ glue interpreted
  của TKV đắt nhất. Không nên trích seq8 làm đại diện.
- Torch `tier_d` chỉ n=2 → median = trung bình 2 lần, độ tin thấp hơn best-of-3 của TKV.
  Chênh lệch ±2% giữa hai lần chạy torch đã thấy trong log (3975.7 vs 4104.7).

## 8. Phân rã backward trên ĐƯỜNG PRODUCTION (2026-11-10 16:xx) — §0.133 đã sai chỗ nào

Build `out/bseq.exe` 15:26:27, HEAD `38bfb04`, cùng máy (Xeon E5-2695 v3, 14T).
Probe mới: `tv/tests/_bwd_insitu.tkv`, `_bwd_total.tkv`, `_lmhead_bwd.tkv`,
`_gemm_peak.tkv`, `_bwd_tail_a.tkv`, `_dwte_zero3.tkv`.

### 8.1 Số đo (seq1024)

| mảnh | ms | ghi chú |
|---|---:|---|
| `gpt2_backward` FULL | **2860** | khớp bench BPHASE 2844 |
| 12 × `gpt2_block_bwd` (in-situ) | **1680** | 140 ms/layer |
| **phần NGOÀI 12 block** | **~1180** | **41% của backward** |

Phân rã 12 block (in-situ, đúng thứ tự, glue = **0.0**):

| mảnh | ms/layer | ×12 |
|---|---:|---:|
| `mlp_bwd` | 62 | 744 |
| `attention_bwd` | 78 | 936 |
| `layernorm_bwd` ×2 | ~0 | ~0 |
| `simd_add` ×2 | ~0 | ~0 |
| **tổng** | **140** | **1680** |

Phân rã phần NGOÀI block (~1180 ms):

| mảnh | ms | ghi chú |
|---|---:|---|
| `lmhead_bwd_fused_list` | **515** | 2 GEMM 39.5 GFLOP + softmax-grad |
| `t_zeros` dwte [V,h] 154 MB | **125** | phải giữ: kernel CỘNG DỒN |
| `t_zeros` gwte_e [V*h] 154 MB | **~130** | phải giữ: `_emb_bwd_worker` cộng dồn |
| `f32_list_add` dwte += gwte_e | ~80 | 38.6M phần tử |
| `f32_list_add` gflat += dwte | ~80 | 38.6M phần tử |
| `op_embed_bwd` gwpe + lnf + misc | ~250 | |
| **tổng** | **~1180** | |

### 8.2 §0.133 SAI: "lm-head/embed sát trần, không còn gì"

§0.133 tính 158 GFLOPS rồi kết luận sát trần. Đo lại:

| phép | ms | GFLOPS |
|---|---:|---:|
| `smm_dot_pd` [1024,50257]×[50257,768] | 234 | **338** |
| `smm_fwd_at_pd` (trans-A) [50257,1024]×[1024,768] | 641 | **123** |
| `smm_dot_pd` vuông 2048³ | 47 | **366** |
| `lmhead_bwd_fused_list` (cả 2 GEMM + softmax) | 515 | 153 (nếu tính 79 GF) |

Đỉnh thật của tkv GEMM trên máy này là **~366 GFLOPS**, không phải 158. §5.1b ghi
515 GFLOP/s là đỉnh *torch/numpy*, tkv chưa bao giờ đạt.

**Trans-A GEMM chỉ đạt 123 GFLOPS = 34% đỉnh** — chậm hơn bản không trans 1.36×.
Nhưng nó **KHÔNG nằm trên đường production**: production đi `lmhead_bwd_fused_list`,
mà fused (515 ms) đã nhanh hơn 2 GEMM riêng (234+641=875 ms) **1.70×**.

⇒ Kết luận đúng: lm-head **không sát trần**, nhưng đường fused đã tốt hơn đường
GEMM rời. Muốn cắt 515 ms phải sửa **bên trong kernel fused** (trans-A bên trong nó),
không phải quay lại 2 GEMM rời.

### 8.3 Đã kiểm chứng: 2 zero-fill 154 MB KHÔNG bỏ được

`_dwte_zero3.tkv`: nạp `dwte` = 1.0, chạy kernel với grad khác 0 → `dwte[0]` =
**1.0000199** (giữ nguyên 1.0 + grad nhỏ). Kernel **CỘNG DỒN**, không ghi đè.
`_emb_bwd_worker` cũng cộng dồn (`out_data[r*d+c] = ... + gy_data[...]`).
⇒ 255 ms zero-fill là chi phí thật, không phải lãng phí.

### 8.4 Hướng cải tiến còn lại, xếp theo ms/độ rủi ro

| # | việc | ms tiết kiệm | rủi ro |
|---|---|---:|---|
| 1 | trans-A bên trong `lmhead_bwd_fused_list`: 123 → 338 GFLOPS | **~330** | trung bình, phải giữ bit-exact |
| 2 | gộp 2 `f32_list_add` 38.6M (dwte+=gwte_e, gflat+=dwte) | **~80** | thấp |
| 3 | `mlp_bwd` 62 ms/layer (744 ms tổng) — chưa profile bên trong | ? | ? |
| 4 | `attention_bwd` 78 ms/layer (936 ms) — ATen SDPA bwd là 65 | ~156 | cao (native đã chậm hơn) |

Mục tiêu ≤1.0× cần cắt **494 ms**. Việc #1 + #2 = ~410 ms, gần đủ, và cả hai đều
nằm ngoài 12 block — chỗ mà §0.131 nói "không còn gì để lấy".

### 8.5 Cảnh báo phương pháp

- Đo từng mảnh trong **vòng lặp riêng** cho số lạc quan: `_bwd_split2` ra 138 ms/layer,
  in-situ ra **140** — nhưng `_bwd_split2` báo block 148.5 còn in-situ báo 140, tức
  nhiễu ±6%. Chỉ tin số in-situ.
- `_block_split.tkv` (09:07) và `_mlp_split.tkv` (09:19) truyền `gflat=None` → đo
  native fallback, **không phải production**. Đừng dùng lại.
- Probe quá nhiều local trong `main` → `InvalidProgramException` (giới hạn codegen,
  cùng loại với `gap_tests`). Phải chia nhỏ probe.

---

## 9. Sửa lại §8: đo đúng đường production (2026-11-10)

§8 ở trên **sai ở hai chỗ**, và cả hai đều do đo sai đường. Ghi lại đầy đủ để không
ai lặp lại.

### 9.1 §8 đo nhầm đường: thiếu BLAS và thiếu song song

`probe_lmhead_phases2.c` gọi `lmhead_softmax_rows` **1 thread** và để GEMM chạy
kernel native, vì probe không tìm thấy `libscipy_openblas64_.dll`. Production thì
khác: `lmhead_bwd_fused` gọi `tkv_run_aux(nt, 3, ...)` cho softmax và
`tkv_blas_load()` nạp OpenBLAS cho cả hai GEMM.

Số đúng (`probe_lmhead_phases3.c`, mirror đúng 5 pha, BLAS=1, 28 thread):

| pha | ms | % của 542 |
|---|---:|---:|
| malloc dl+tmp (343 MB) | 0.03 | 0.0% |
| pha1 softmax-grad (28 thr) | 71.7 | 13.2% |
| pha2 `dxf = dl @ wte` | 141.9 | 26.2% |
| pha3 `tmp = dlT @ xf` (trans-A) | **255.5** | **47.1%** |
| pha4 `dwte += tmp` (28 thr) | 36.0 | 6.6% |
| **SUM** | **505.1** | |
| **FULL** | **542.5** | |

Chênh 37 ms (6.9%) là phần `tkv_run_aux` khởi động pool lần đầu.

**`malloc` 343 MB mỗi lần gọi chỉ 0.03 ms.** Nó không phải vấn đề. Ghi chú cũ
trong source ("do 2026-11-07 @seq1024: 110 ms" cho transpose) cũng đã lỗi thời.

### 9.2 Trans-A KHÔNG phải chỗ cắt được — giả thuyết §8.4 #1 sai

`probe_ta_fix.c` đo 4 phương án trên cùng shape `[50257,1024]×[1024,768]`, 39.5 GFLOP:

| phương án | ms | GFLOPS |
|---|---:|---:|
| P1 transA trực tiếp (hiện tại) | **264.8** | 298.5 |
| P2 transpose `dl`→`dlT` (blocked 32×32) | 212.7 | — |
| P3 NoTrans trên `dlT` | 258.2 | 306.1 |
| P4 = P2 + P3 | 482.3 | 163.9 |

NoTrans (258) so với transA (265) chỉ chênh **3%**. OpenBLAS xử lý transA gần như
miễn phí ở shape này. Còn transpose rồi NoTrans thì **chậm hơn 1.82×** vì riêng
bước transpose đã 213 ms cho 196 MB đọc + 196 MB ghi.

⇒ **Bỏ hẳn hướng "sửa trans-A".** 255 ms đó là giá thật của 39.5 GFLOP ở 309 GFLOPS,
và 309 GFLOPS là 84% của đỉnh đo được 366 GFLOPS. Không còn gì để lấy ở đây.

### 9.3 Phát hiện thật: `t_zeros` chậm hơn `f32_zero` **8×**

Đây là chỗ tiền thật nằm. Cùng 147 MB:

| đường | ms | GB/s |
|---|---:|---:|
| `t_zeros([V,h])` | 125–172 | 1.2 |
| `zeros_list(n)` (thân của `t_zeros`) | 125–172 | 1.2 |
| `alloc_bytes` + `f32_zero` + `f32buf_wrap` | **15** | **10.0** |
| `f32_zero` riêng (buffer native) | 16–32 | 5–10 |
| `f32buf_wrap` riêng | **0.0** | — |

`t_zeros` → `zeros_list(n)` → `List<f32>`. Đường đó đi qua cấp phát managed, và
CLR zero-fill object lớn trên LOH trước khi trả về — cộng thêm chi phí thông dịch.
`alloc_bytes` + `f32_zero` là `memset` thẳng trên buffer native, không qua GC.

Kiểm chứng alias hai chiều (`_zero_wire.tkv`): ghi `bl[0]=42.5`, `bl[7]=-3.25` qua
`list[f32]` → đọc lại qua `f32buf_wrap` thứ hai thấy đúng `42.5 / -3.25`. Gọi
`f32_zero` xong thì cả hai đọc ra `0.0`. **Không copy, không lệch bit.**

### 9.4 Chuỗi 154 MB trong backward: 4 lần đụng, ~415 ms

Đường production (`_f124_bwd` trong `tv.training.gpt2_ref124.tkv`):

| # | chỗ | ms |
|---|---|---:|
| 1 | `dwte = t_zeros([VocabSize, h])` (L2150) | 125 |
| 2 | `gwte_e = t_zeros([VocabSize * h])` (L2269) | 125 |
| 3 | `f32_list_add(dwte, gwte_e, V*h)` (L2281) | 63 |
| 4 | `f32_list_add(gflat, dwte, V*h)` (L2290) | 63 |

Cả bốn đều là thao tác trên đúng một mảng 38.6M phần tử. Thay #1 và #2 bằng
`alloc_bytes`+`f32_zero`+`f32buf_wrap` tiết kiệm **~220 ms** mà không đổi một bit nào.

### 9.5 Một biến chưa ai kiểm soát: kernel tự bật hyperthread

`tkv_nthreads` (L284) tự nhân đôi lên **28** logical khi `n >= 512`, dù bench khai
14 thread. Đo A/B trên chính kernel fused:

| `TKV_GEMM_THREADS` | FULL | pha2 | pha3 |
|---:|---:|---:|---:|
| 7 | 1655 | 689 | 777 |
| 14 | 1179 | 510 | 473 |
| **28 (auto)** | **924** | **371** | **418** |

28 nhanh hơn 14 đúng **1.28×**. Nghĩa là mọi so sánh "TKV 14 thread vs torch 14
thread" đều đang cho TKV **bất lợi**, không phải ưu thế. Đây là biến cần khai báo
rõ trong mọi bench về sau.

### 9.6 Bảng ưu tiên sửa lại

| # | việc | ms | rủi ro |
|---|---|---:|---|
| 1 | `t_zeros` → `alloc_bytes`+`f32_zero`+`f32buf_wrap` cho 2 buffer 154 MB | **~220** | thấp, đã chứng minh bit-identical |
| 2 | gộp `f32_list_add` #3 vào #4 | ~63 | thấp |
| 3 | `mlp_bwd` 62 ms/layer (744 ms) — chưa mở bên trong | ? | ? |
| 4 | `attention_bwd` 78 ms/layer (936 ms) | ~156 | cao |
| ~~5~~ | ~~sửa trans-A~~ | ~~0~~ | **bỏ, xem §9.2** |

Mục tiêu ≤1.0× cần 494 ms. Việc #1 + #2 = **~283 ms**, và cả hai đều bit-identical.

### 9.7 Cảnh báo phương pháp (bổ sung §8.5)

- **Probe phải mirror đúng số thread và đúng BLAS của production.** §8 sai vì
  probe chạy 1 thread softmax + GEMM native, production chạy 28 thread + OpenBLAS.
  Chênh 924 vs 542 ms — gần 2×.
- Copy `libscipy_openblas64_.dll` vào cạnh exe probe, hoặc set
  `TKV_GEMM_BLAS_PATH`. Không có nó thì mọi số GEMM đều sai.
- `#include` nguyên `tkv_simd_native.c` vào probe là cách gọi được hàm `static`
  (`run_rect_shaped`, `run_rect_ta_shaped`) mà không sửa source production. Phải
  link thêm `tkv_bwd_ops.c` + `tkv_flash_attn.c` cho các symbol attention.

### 9.8 Vì sao không sửa được ngay: `f32buf_wrap` trả con trỏ thô, không phải `list[f32]`

Đã thử thay trực tiếp trong production và **revert**. Lý do kỹ thuật, ghi lại để
không ai thử lại:

`f32buf_wrap` không phải builtin expression. Nó là **statement** riêng
(`assign_f32buf_wrap`, `tv_gop.tkv` L23378) và trả về một **con trỏ `f32buf` thô**
(`TypeAnn('f32','f32buf')`), không phải `List<float32>`. Hệ quả:

- Chỉ dùng được dạng `a = f32buf_wrap(p, n)` trong **`main`**, không trong hàm module.
- Không truyền được vào `lmhead_bwd_fused_list`, vì builtin đó kiểm tra
  `_la_list(args[idx], scope)` và đòi đúng `list[f32]` (L48589).
- `t_new(...)` cần `list[f32]`, không nhận con trỏ.

Muốn dùng đường native thì phải thêm một builtin mới (ví dụ
`zeros_list_buf(n) -> list[f32]` gọi thẳng `ZeroListF32` trên buffer native), hoặc
mở rộng `f32buf_wrap` để trả `List<float32>`. Cả hai đều là sửa **compiler**, không
phải sửa `.tkv`. Đó là lý do việc này chưa làm trong phiên này.

### 9.9 Nguyên nhân thật của 1.2 GB/s: cấp phát managed, không phải băng thông

`_zeros_scale.tkv` đo `zeros_list(n)` ở 5 kích thước:

| n | MB | ms | GB/s |
|---:|---:|---:|---:|
| 1,000,000 | 3.8 | 0.0 | — |
| 4,000,000 | 15.3 | 16 | 1.00 |
| 8,000,000 | 30.5 | 31 | 1.03 |
| 16,000,000 | 61.0 | 78 | 0.82 |
| 38,597,376 | 147.2 | 141 | 1.09 |
| 38,597,376 (lần 2) | 147.2 | 125 | 1.24 |

**GB/s gần như hằng số ~1.0 ở mọi kích thước, kể cả 15 MB.** Nếu nghẽn là LOH/GC
thì chỉ ngưỡng lớn mới chậm. Nếu nghẽn là băng thông thì phải ~10 GB/s như
`f32_zero`. Con số 1.0 GB/s phẳng lì ở mọi cỡ cho thấy chi phí nằm ở **đường cấp
phát managed** (`TkvSlice::ZeroListF32`: `List<float32>(n)` + `SetCount` +
`Span.Fill`), không phải ở `Span.Fill` và không phải ở GC ngưỡng lớn.

`f32_zero` là `memset` thẳng trên buffer native đã cấp → 10 GB/s. Chênh đúng 8×.

### 9.10 Trạng thái: chưa sửa được, cần sửa compiler

Việc #1 trong §9.6 (~220 ms) **không làm được ở tầng `.tkv`**. Nó cần một trong:

1. Builtin mới `zeros_list_buf(n) -> list[f32]` cấp buffer native rồi wrap thành
   `List<float32>` (nhanh nhất, nhưng phải giữ được ngữ nghĩa sở hữu/giải phóng).
2. Mở `f32buf_wrap` trả `List<float32>` thay vì con trỏ thô.
3. Sửa `ZeroListF32` để dùng `GC.AllocateUninitializedArray` + `Span.Clear`
   (bỏ bước zero-fill kép mà CLR làm khi cấp `List<float32>(n)`).

Cách 3 là gọn nhất và có lợi cho **mọi** `t_zeros` trong toàn bộ codebase, không
riêng gpt2. Nên làm trước.

Việc #2 trong §9.6 (gộp 2 `f32_list_add`) vẫn làm được ở tầng `.tkv` — chưa thử.

### 9.11 Nguyên nhân chính xác của `ZeroListF32` — đọc IL, không đoán (2026-11-10)

§9.9 ở trên nói "chi phí nằm ở đường cấp phát managed". Đúng nhưng chưa đủ. Đọc IL
thật trong `compiler_bootstrap/compiler/il_features/slicing.tkv` L171–188:

```
ldloc.0 ; newarr float32 ; stloc.1              <-- (1) CLR ĐÃ zero sẵn
TKVZM_CL:
  ldloc.1 ; ldloc.3 ; ldc.r4 0.0 ; stelem.r4    <-- (2) 38.6M lần ghi scalar: THỪA
  ...
TKVZM_CD:
  ldloc.1
  newobj List`1<float32>::.ctor(IEnumerable`1<!0>)  <-- (3) CẤP MẢNG THỨ HAI + copy
```

Ba lỗi, xếp theo mức tốn:

1. **(3) là chính.** `new List<T>(IEnumerable<T>)` không nhận mảng làm backing store.
   Nó `new T[n]` mới rồi `Array.Copy`. Tổng: 1 mảng 147 MB (CLR zero) + 1 mảng
   147 MB nữa (copy) = **294 MB cấp phát + 147 MB copy** cho một buffer 147 MB.
2. **(2) thừa hoàn toàn.** `newarr` đã bảo đảm mọi phần tử = 0. Đo được: bỏ vòng này
   tiết kiệm 42 ms (A 102.2 → B 60.0).
3. **Comment trong source sai.** Docstring ghi *"alloc List(n), SetCount, Span.Fill(0)"*
   nhưng IL **không** có `SetCount` và **không** có `Span.Fill`. Comment lệch khỏi code
   từ phiên bản trước. Đừng tin comment hơn IL.

### 9.12 Đo 5 cách cấp phát trên .NET 10 thật (`out/zltest/Program.cs`)

Cùng máy, 147.2 MB, best-of-5, `.NET 10.0.13`, 28 core:

| cách | ms | GB/s |
|---|---:|---:|
| **A. IL hiện tại** (`newarr` + vòng `stelem` + `List(IEnumerable)`) | **102.2** | **1.41** |
| B. bỏ vòng scalar, giữ `List(IEnumerable)` | 60.0 | 2.40 |
| C. `newarr` + `List(n)` + `AddRange` | 64.2 | 2.24 |
| D. `List(n)` + `SetCount` (reflection) | 15.8 | 9.12 |
| **F. `List(n)` + `CollectionsMarshal.SetCount`** | **15.7** | **9.16** |

Cả 5 cách đều cho `count = 38597376`, toàn `0.0`, ghi/đọc `z[0]`/`z[n-1]` bình thường.

**A → F = 6.5×.** Đây là con số phải đưa vào request, không phải con số 8× ước lượng
ở §9.3 (8× là so `t_zeros` với `f32_zero`, tức so hai đường khác nhau; 6.5× là so
đúng hai phiên bản của cùng một hàm).

### 9.13 Request đã gửi compiler team

`docs/COMPILER_TEAM_REQUEST13.md` (2026-11-10). Nội dung: sửa **thân IL** của
`TkvSlice::ZeroListF32`, không đổi chữ ký, không đổi ABI, không đổi ngữ nghĩa `.tkv`.

Phương án chính: `new List<float32>(n)` + `CollectionsMarshal.SetCount(z, n)`.
Không copy, không cấp thêm, không reflection, không phụ thuộc layout nội bộ.

Phương án dự phòng (nếu IL toolchain không resolve được `CollectionsMarshal`): tự
viết `TkvSlice::SetCountF32` dùng `stind.i4` ở offset 16. Có cảnh báo phụ thuộc
layout runtime.

**Điều kiện tiên quyết đã phát hiện:** IL sinh ra hiện chỉ khai `.assembly extern`
cho `mscorlib`, `System.Core`, `System`, `System.Numerics`,
`System.Numerics.Vectors`, `System.Runtime.Intrinsics` (kiểm trên `out/zeroscale.il`).
**Chưa có `System.Runtime`**, mà `CollectionsMarshal` nằm trong đó. Compiler team có
thể phải thêm extern này. Nếu không thêm được thì dùng phương án dự phòng.

Kỳ vọng: `zeros_list(38597376)` từ 125–172 ms xuống ≤30 ms; full step @seq1024 từ
4578 ms xuống ≤4350 ms; tiết kiệm ~220–250 ms/bước = ~45% của 494 ms cần cắt để
đạt ≤1.0×.

### 9.14 COMPILER DA SUA — `ZeroListF32` xong, do lai THAY (2026-11-10)

Compiler team sua bang mot duong khac va **tot hon** de xuat cua toi trong
`COMPILER_TEAM_REQUEST13.md`:

```
ZeroListF32(n):
  n <= 0  -> new List<float32>()
  else    -> new List<float32>(new TkvZeroSrc(n))
```

`TkvZeroSrc` la mot `ICollection<float32>` gia: `Count = n`, `CopyTo` **RONG**,
`GetEnumerator` nem `NotSupportedException`. Ctor `List<T>(IEnumerable<T>)` co
duong nhanh khi tham so la `ICollection<T>`: no doc `Count` de cap backing array
**dung mot lan** roi goi `CopyTo`. `CopyTo` rong nen khong co lan chep 147 MB nao,
va backing array da duoc CLR zero-fill san.

**Tai sao tot hon `CollectionsMarshal.SetCount`:** chay duoc ca .NET Framework 4.8
lan .NET 10, khong reflection, khong phu thuoc layout noi bo cua `List<T>`
(`_size` o offset 24 tren Framework, 16 tren .NET 10 — dung offset tho la sai).

**Test chan:** `tests/zeros_list_scale_test.tkv`, 29 assert, n = 0/1/7/1000/38597376.
Da chay: **29/29 PASS**.

---

### 9.15 CANH BAO PHUONG PHAP — con so "0.0 ms" la SAI, chi phi bi HOAN

Sau khi sua, `_zeros_scale.tkv` bao `zeros_list(38597376)` = **0.0 ms** cho 147 MB.
Khong the tin. Toi viet `_zeros_verify.tkv` de kiem va tim ra su that:

| buoc | ms |
|---|---:|
| A. tao, KHONG cham du lieu | **0.0** |
| B. tao + doc toan bo | 78 |
| C. tao + ghi toan bo | 110 |
| D. tao + doc lan 2 | 109 |
| E. 1M (3.8 MB) tao + doc | 16 |

Chi phi **khong mat di, no bi hoan**: `new T[38597376]` tren LOH chi dat con tro;
CLR dua trang zero cua OS vao khi CPU cham lan dau (page fault). `count` tra ve
dung 38597376 nen khong phai DCE — chi la chi phi nam o buoc cham dau tien.

**He qua do luong:** moi phep do `t_zeros`/`zeros_list` phai **cham vao du lieu**
roi moi ket luan. Do moi buoc "tao" khong cho so co nghia. Day la cung loai loi
voi §8 (probe do sai duong), va la lan thu ba trong phien nay.

**Nhung tren duong production thi no van thang that**, vi `f32_list_add` cham toan
bo buffer ngay sau do — chi phi bi hoan do duoc tra o buoc do, va tong van giam.

---

### 9.16 KET QUA THAT — full step seq1024 giam 359 ms

Do tren `out/bseq_new.exe` (compile bang compiler moi), best-of-3, cung phien:

| seq | truoc | sau | giam |
|---|---:|---:|---:|
| 8 (mode 1) | 1437 | **1187** | −250 |
| 256 (mode 1) | 2141 | **1782** | −359 |
| **1024 (mode 1)** | **4578** | **4219** | **−359** |

**Loss EXACT ca 3 seq:** `10.572291 / 10.756927 / 10.760399` — khong doi mot bit.

**Parity:** `gpt2_full_tests` **33/33 PASS, fails=0**. (84/84 = 9 suite nho 51 +
gpt2_full 33; suite nho chua chay lai phien nay.)

Vuot ky vong 220–250 ms trong request. Muc tieu <=1.0x can cat 494 ms; da cat 359,
con **135 ms**.

---

### 9.17 BAY MOI PHAT HIEN — apphost khong chay duoc binary do tkvc sinh ra

`out/*.exe` do `tkvc-tv.exe` sinh ra **KHONG kem `.runtimeconfig.json`**. Chay
truc tiep `.exe` se bao:

```
System.IO.FileNotFoundException: Could not load file or assembly
'System.Runtime.Intrinsics, Version=10.0.0.0, ...'
```

Loi nay **danh vao `gpt2_layernorm`** (no that su goi
`Avx::ConvertToVector256Double`), nen rat de tuong la `Avx` hong. Khong phai.

**Nguyen nhan that:** may nay khong co `C:\Program Files\dotnet`; .NET nam o
`C:\Users\Admin\.dotnet` va khong co bien `DOTNET_ROOT`. Apphost khong tim duoc
runtime nen roi ve che do sai.

**Cach chay dung:**
```powershell
dotnet out/bseq_new.exe
```
nhung **phai co** `out/bseq_new.runtimeconfig.json` truoc:
```json
{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}
```

Khong co file do thi `dotnet` cung bao `hostpolicy.dll not found`. Day chinh la
blocker da gap o phien truoc khien parity khong verify duoc — **khong phai loi
compiler hay loi agent**.

**De nghi:** `tkvc-tv.exe` nen sinh `.runtimeconfig.json` canh `.exe`, hoac
`testkit_gate.ps1` nen tu tao no. Hien tai moi test deu phai copy tay.

### 9.18 PHÂN RÃ 5 PHA SAU KHI SỬA (2026-11-11)

Cùng phiên với torch (`out/_bench_matrix_20261110_154851.log`, torch T=14):

| pha | TKV | torch | chênh |
|---|---:|---:|---:|
| fwd | 1032 | 1086.8 | **−54.8** (TKV THẮNG) |
| loss | 109 | (gộp trong fwd) | — |
| bwd | 2437 | 2352.0 | +85.0 |
| opt | 781 | 655.2 | +125.8 |
| copyback | 125 | 62.0 | +63.0 |
| **TỔNG** | **4484** | **4130.6** | **+353.4** |

**Forward của TKV đã nhanh hơn torch.** Chỗ thua nằm ở `opt` + `copyback` (189 ms)
và `bwd` (85 ms). `opt`=781 và `copyback`=125 **giống hệt ở cả seq8/256/1024** →
chi phí cố định, không co giãn theo seq.

### 9.19 `opt` 781 ms = ĐƯỜNG CLIP — tìm ra sau 5 lần đo sai (2026-11-11)

`AdamW.step()` có hai đường theo `coef = clip/(grad_norm+1e-6)`:

- `coef == 1.0` → `adamw_step_glm` đọc `gl` list trực tiếp (NHANH)
- `coef != 1.0` → `r1 = f32_copy_to_buf` + `f32_zero` + `f32_addcmul` + `adamw_step` (CHẬM)

Gradient thật cho `gn = 11.155` → `coef ≈ 0.0896 ≠ 1.0` → **production LUÔN đi
đường chậm**. Đường nhanh chỉ chạy khi gradient bằng 0.

Đo từng mảnh (`tv/tests/_clip_m1..m5.tkv`), n = 124,439,808 (474.7 MB/buffer):

| mảnh | ms |
|---|---:|
| `r1` = `f32_copy_to_buf` | **156** |
| `rzp` = `f32_zero` | **125** |
| `rax` = `f32_addcmul` | **172** |
| `adamw_step` (buffer) | 343 |
| `adamw_step_glm` (list) | 343 |

Đường chậm = 156+125+172+343 = **796 ms**; đường nhanh = 343 ms.

End-to-end `opt.step()` với gradient thật (`_clip_real.tkv`):

| lần | clip=1.0 (chậm) | clip=1e9 (nhanh) | chênh |
|---|---:|---:|---:|
| 1 | 781 | 563 | −218 |
| 2 | 875 | 547 | −328 |
| 3 | 781 | 657 | −124 |

Chênh 124–328 ms, trung bình **~220 ms**.

`ponl` là buffer toàn `1.0`, nên `f32_addcmul(ptl, pgl, ponl, coef, n)` thực chất
chỉ là `ptl = coef * gl` — một phép nhân vô hướng đi qua 3 lượt chạm 474 MB.

**Request #14** (`docs/COMPILER_TEAM_REQUEST14.md`): thêm tham số `coef` vào
`adamw_step_glm`, nhân ngay trong kernel. Bỏ được `r1`+`rzp`+`rax` và buffer `pt`
(474 MB working set). Kỳ vọng `opt` 781 → ~560, full step 4219 → ~4000.

### 9.20 NĂM LẦN ĐO SAI TRONG HAI PHIÊN — ghi để không lặp

| # | lỗi | hậu quả |
|---|---|---|
| 1 | probe `lmhead_bwd_fused` chạy 1 thread softmax + không nạp BLAS | 924 ms thay vì 542 (§8 sai, §9 sửa) |
| 2 | đo `t_zeros` chỉ ở bước "tạo", không chạm dữ liệu | 0.0 ms giả tạo (§9.15) |
| 3 | `adamw_step_glm(b1, b1, b1, ...)` cùng buffer cho pp/pm/pv | 125 ms thay vì 343 |
| 4 | probe clip dùng `g` toàn 0 → `gn = 0` → cả hai nhánh đi đường nhanh | "clip không tốn gì" — SAI |
| 5 | `zeros_list` trả dtype `i32`, index trực tiếp trong `main` | `AccessViolation` / `InvalidCastException` |

Bài học chung: **mỗi phép đo phải chạm đúng đường production, đúng số buffer, đúng
dữ liệu thật.** Con số đẹp thường là dấu hiệu đo sai, không phải tối ưu tốt.

---

### 9.21 KẾT QUẢ SAU REQUEST #14 (2026-11-11) — `opt` 781 → 438-469 ms

Compiler: repo `D:\TokenVector` commit `82e5426`; `tkv_simd_native.dll` hash
`1E49BCE2…` (thêm export `AdamwStepF32C`); 4 bản `trainer.tkv` đổi sang
**một** call `adamw_step_glm(..., coef)`, bỏ `r1` + `rzp` + `rax` + `gsrc`.

**Bit-exact (điều kiện bắt buộc của request):**

| phép kiểm | kết quả |
|---|---|
| `coef == 1.0` vs `adamw_step` (buffer) | **0/0/0** phần tử lệch p/m/v (`f32_list_ops_test`, n=64/37) |
| `coef != 1.0` vs `f32_zero`+`f32_addcmul`+`adamw_step` | **0/0/0** (`f32_list_ops_test` + probe n=131072 → đường thread-pool mode 8) |
| loss 3 seq | **EXACT**: `10.572291 / 10.756927 / 10.760399` |
| `gpt2_full_tests` | **33/33, fails=0, exit 0** |

**Phân rã 5 pha (mode 2), cùng phiên, `_seq_cfg = <seq>/2/1/1`:**

| seq | fwd | loss | bwd | **opt** | copyback | TỔNG | opt trước |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 94 | 0 | 375 | **438** | 141 | **1048** | 781 |
| 256 | 297 | 31 | 641 | **453** | 141 | **1563** | 781 |
| 1024 | 1016 | 125 | 2296 | **469** | 109 | **4015** | 781 |

`opt` giảm **−312 … −343 ms** và **không còn phụ thuộc clip** — `clip=1.0`
(production, `coef≈0.0896`) giờ bằng đúng `clip=1e9`:

```
_clip_real.tkv:  A clip=1.0 = 484 ms   B clip=1e9 = 484 ms   chênh = 0.0
```

**Full step mode 1, best-of-3:**

| seq | trước | sau (3 lần) | best |
|---|---:|---|---:|
| 8 | 1187 | 1547 / 1157 / 1141 | **1141** |
| 256 | 1782 | 1656 / 1562 / 1656 | **1562** |
| 1024 | 4219 | 4125 / 4109 / 4281 | **4109** |

Mode 1 dao động lớn trên phiên này (seq8: 1141–1547; seq1024: 4109–4344 ở lần
đo trước), nên **mode 2 là số tin được** cho phần `opt`; mode 1 chỉ dùng để
so tổng với baseline cũ. TỔNG mode 2 @seq1024 = **4015** so với torch cùng
phiên §9.18 (`4130.6`) = **0.972×** — đạt ≤1.0×.

**Gate/regression:**

| gate | kết quả |
|---|---|
| Trainer | **15/15, fails=0** (clip `max_abs_diff=0.0`) |
| `tools/gate_f32.sh` | **10/10 suite PASS, TOTAL fails=0 = 135/135** |
| `tools/run_cases.ps1` | **pass=22 fail=4**, gap_tests FAIL — **giống hệt baseline** (`System.Runtime.Intrinsics` thiếu trên .NET Framework; `gap_tests` PASS khi chạy .NET 10 trong gate) |

**Working set:** `pt`/`pone` **vẫn còn cấp phát** (nhiều probe `_cj7/_cj8/_cy5/
_det/_field_read/_opt_inline/_opt_split` đọc `opt.pt`/`opt.pone`) → chưa lấy
lại 948 MB. Muốn bỏ phải sửa cả các probe đó — việc riêng, không thuộc #14.

### 9.22 TAI DO VOI PYTORCH SAU #14 - Cung phien, cung may (2026-11-11 14:57)

Benchmark lai toan bo matrix sau khi trainer dung `adamw_step_glm(..., coef)`
(request #14, commit ae2dc7a) va compiler build moi (12:45:58, sha256=712704151A996917).
TKV binary: `benchmark/bseq_post14.exe` (build 14:56:12 tu HEAD `6b9d56f`, `tv/tests/simd_bench_124m_seq.tkv`).
Log goc: `benchmark/_bench_matrix_post14_20261111_145719.log`. Script: `benchmark/_bench_matrix.ps1`
(bien the cua `benchmark/_bench_matrix.ps1`, them mode 2).

May: Xeon E5-2695 v3 @2.30GHz (14C/28T), Win10 19045, dotnet 10.0.402,
torch 2.14.0+cpu threads=14.

**Loss EXACT ca 3 seq o moi run:** `10.572291 / 10.756927 / 10.760399`.

**Mode 0 (fwd+bwd), best-of-3:**

| seq | runs (ms) | best | loss |
|---|---|---:|---|
| 8 | 453.0 / 547.0 / 469.0 | **453.0** | EXACT |
| 256 | 985.0 / 1000.0 / 1000.0 | **985.0** | EXACT |
| 1024 | 3437.0 / 3516.0 / 3375.0 | **3375.0** | EXACT |

**Mode 1 (full step), best-of-3:**

| seq | runs (ms) | best | loss |
|---|---|---:|---|
| 8 | 1125.0 / 1047.0 / 1172.0 | **1047.0** | 10.572291 -> 7.011912 EXACT |
| 256 | 1625.0 / 1610.0 / 1594.0 | **1594.0** | 10.756927 -> 11.807263 EXACT |
| 1024 | 4016.0 / 4078.0 / 4719.0 | **4016.0** | 10.760399 -> 13.442921 EXACT |

**Mode 2 (5 pha, min-of-2):**

| seq | fwd | loss | bwd | opt | copyback | TONG |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 94 | 0 | 515 | **468** | 141 | **1218** |
| 256 | 266 | 32 | 671 | **469** | 140 | **1578** |
| 1024 | 1016 | 109 | 2328 | **453** | 141 | **4047** |

`opt` 468/469/453 - khong phu thuoc seq, khong phu thuoc clip (giong H4
ch o 9.21: 438/453/469). `copyback` van ~140.

**PyTorch cung phien (tier d, T=14):**

| seq | fwd+bwd | full step |
|---|---:|---:|
| 8 | 320.1 | 981.4 |
| 256 | 873.4 | 1551.4 |
| 1024 | 3395.3 | 4061.1 |

**PyTorch phase (tier e, T=14; T=1 trong log):**

| pha | seq8 | seq256 | seq1024 |
|---|---:|---:|---:|
| zero_grad | 21.7 | 12.1 | 52.5 |
| fwd(+CE) | 82.9 | 282.9 | 1024.7 |
| bwd | 289.2 | 585.3 | 2357.0 |
| opt.step | 651.2 | 668.9 | 646.4 |
| TOTAL | 1049.7 | 1636.7 | 4097.6 |

(T=1 TOTAL: 1348.7 / 4244.7 / 14741.6.)

**TKV vs torch, cung phien (ti le TKV/torch, nho hon = TKV nhanh hon):**

| phep do | seq8 | seq256 | seq1024 |
|---|---:|---:|---:|
| mode 1 best vs torch d full step | 1.067x | 1.027x | **0.989x** |
| mode 1 best vs torch e T=14 TOTAL | **0.998x** | **0.974x** | **0.980x** |
| mode 0 best vs torch d fwd+bwd | 1.416x | 1.128x | **0.994x** |
| mode 2 TONG vs torch e T=14 TOTAL | 1.160x | **0.964x** | **0.988x** |

Phase @seq1024 (mode 2 vs e T=14): fwd 1016 vs 1024.7 (-8.7); bwd 2328 vs
2357.0 (-29); **opt 453 vs 646.4 (-193.4)**; copyback 141 (torch khong co
khai niem tuong duong). @seq256: opt 469 vs 668.9 (-200). @seq8: opt 468
vs 651.2 (-183).

**So voi 9.21 (torch khac phien 4130.6):** torch cung phien hom nay =
4061.1 (d) / 4097.6 (e T=14 TOTAL). TKV mode 2 TONG 4047 -> **0.988x**
so voi torch cung phien - giu nguyen ket luan "<=1.0x o seq1024", nhung
nay la cung phien nen chac chan hon.

**Luu y do:**
- Mode 1 seq1024 dao dong manh (4016/4078/4719) - mode 2 (min-of-2) la
  so tin duoc cho phase; mode 1 chi de so tong voi lich su.
- Mode 0 best: 453/985/3375 (log raw; parser best-of-3 cua script in
  `best=0` do bug `-match` tren array - da fix trong `benchmark/_bench_matrix.ps1`).
- Chua chay lai gate trong phien nay; gate 135/135 + trainer 15/15 + loss
  EXACT da xac nhan o 9.21 (khong doi code tu do).

## 9.23 BENCHMARK COLAB-CPU: .tkv vs PyTorch tren ung Colab VM (2026-11-12)

Benchmark cung workload tren may Colab (CPU instance, khong phai may Windows).
Trainable end-to-end .tkv MLP 4-8-3 (67 params) vs PyTorch cung init, cung
sampler LCG Fisher-Yates (seed=42), cung lich lr (warmup 5 + cosine), cung clip.
Fixture: `tv/tests/fixtures/train_cases.txt` (20 mau, PyTorch reference).

Workload: **MLP 4->8->3, 400 buoc, accum=2, clip=1.0, warmup=5, total=400,
lr_max=0.1, lr_min=0.005, wd=0.01, b1=0.9, b2=0.999, eps=1e-08**.

| Benchmark | wall-clock (median of 5) |
|---|---:|
| .tkv trainer (.NET 10) | **0.241 s** |
| PyTorch (cung workload, chi training loop) | 0.495 s |
| .tkv / PyTorch | **~0.49x** (tkv ~2x nhanh hon) |

**Loss cuoi (verify):** `eval_loss = 0.01494988` (.tkv PASS 15/15 voi fixture;
PyTorch reimplemented cung workload cho `0.0149499`, acc = 1.0 — khop
**0.01494989** trong fixture).

**Note quan trong:**
- .tkv 0.241s bao gom: 400 buoc training **va** parse fixture + cross-check
  PyTorch 1e-07 + eval.
- PyTorch 0.495s chi la vong training + eval.
- PyTorch sau khi them quy trash `popen`/subprocess se cham hon nua.
- Colab VM specs khong duoc in ra (notebook chua capture CPU info); phai do
  lai tren may Windows de so sanh input locale:
    `python benchmark/_bench_matrix.ps1` -> xem `benchmark/BENCHMARKS.md §9.21/§9.22`.

**Cach chay:** `demo/colab_bench_ai.ipynb` (file local hoac tu GitHub:
https://github.com/nguyenhungtran18/TokenVector.Inference/blob/main/demo/colab_bench_ai.ipynb).

## 9.24 PORTABLE RUNTIME: `bseq` 124M CHAY TREN LINUX (2026-11-12)

Đây **không phải** phép so sánh với PyTorch — đây là kết quả gỡ blocker
cross-platform (Phase A, `docs/PLAN_COLAB_CROSS_PLATFORM.md`). Mọi số dưới đây đo
trên **cùng một máy** (Windows native và WSL2 Ubuntu), GPT-2 124M random init
(`gpt2_init(7)`), `_seq_cfg` = `seq / mode / steps`.

### 9.24.1 Hai lỗi đã sửa (nguyên nhân `.tkv` không chạy được trên Colab/Linux)

1. **`DllNotFoundException: kernel32.dll`** — `TkvSimd.Init()` P/Invoke
   `IsProcessorFeaturePresent` / `VirtualAlloc`. Sửa: tách `InitWin()`, chỉ gọi khi
   `Environment.OSVersion.Platform == Win32NT`.
2. **`MissingMethodException: Vector`1.op_Addition`** — bản fallback (`...V`) dùng
   `System.Numerics.Vector<T>`; trên .NET 10 method này không resolve được từ IL
   tham chiếu `[System.Numerics.Vectors]`. Sửa: fallback dùng vòng vô hướng
   (`_pt_dot_scalar`, `_pt_sum_scalar`, `_pt_vong_2_scalar`).
   *(Lỗi này có sẵn từ trước, chỉ lộ ra khi chạy không có kernel native — tái hiện
   được cả trên Windows với `TKV_SIMD_ISA=off`.)*

### 9.24.2 Kết quả chạy

| Môi trường | seq | mode | cách đo | kết quả |
|---|---:|---:|---|---|
| WSL2 Ubuntu, .NET 10.0.15, `isa=0` | 8 | 1 (full step) | 1 warmup + 1 đo | **2 985 ms**, loss `10.572291 → 6.925554` |
| Windows native, `isa=1` (AVX2) | 8 | 1 | 1 warmup + 1 đo | **2 187 ms**, loss `10.572291 → 6.925548` |
| Windows native, `isa=1` | 1024 | 1 | 1 warmup + 1 đo | **7 563 ms** (compiler cũ) / **7 781 ms** (compiler mới) |

- Chênh loss `6.925554` vs `6.925548` (~6e-6) do thứ tự cộng khác nhau (vô hướng
  vs AVX2), không phải sai số logic; loss đầu giống hệt `10.572291`.
- **Không hồi quy trên Windows:** A/B cùng phiên, cùng máy, cùng cfg — compiler cũ
  7 563 ms vs compiler mới 7 781 ms (chênh 2.9%, trong dao động của máy).
  `simd_isa()` trả **1** khi mặc định và **0** khi `TKV_SIMD_ISA=off` → đường native
  vẫn hoạt động.
- **Cảnh báo số tuyệt đối:** phiên này máy chậm hơn phiên §9.22 (~1.9×: 7 563 ms so
  với 4 016 ms cho cùng workload). Vì vậy **không được** lấy 7 563 ms so với số torch
  của §9.22; phải đo lại cả hai bên trong cùng phiên (đúng quy tắc §1).

### 9.24.3 Gate sau khi sửa compiler

| Gate | Kết quả |
|---|---|
| `tools/vong.exe` (self-host) | `IL_IDENTICAL` |
| `tools/testkit_gate.ps1` | **PASS 136/136**, negative 44/44, BUILD/TEST/IL/OTHER = 0 |
| `trainer_tests` (Windows, `isa=1`) | **15/15 PASS**, `eval_loss=0.014949894`, resume bit-exact |
| `TKV_SIMD_ISA=off` + .NET 10 | `simd_add` / `simd_dot` đúng (`11.0`, `99.0`, `2850.0`) |
| `dotnet demo/bseq.dll` (WSL) | chạy hết 1 full step, không còn P/Invoke kernel32 |

### 9.24.4 Còn thiếu

- Chưa có số **GPU** (chưa chạy lại `demo/colab_bench_ai_gpu.ipynb` trên T4 sau khi
  sửa) → chưa chứng minh được cuBLAS dispatch thực sự được dùng.
- Linux vẫn **không có SIMD cho op phần tử** (chỉ GEMM/attention/AdamW nằm trong C
  `.so`); đây là nội dung Phase B.

## 9.25 COLAB T4: `.tkv` 124M CHẠY ĐƯỢC NHƯNG **CHẠM TRẦN CPU** (2026-11-12)

Notebook `demo/colab_bench_ai_gpu.ipynb` chạy trên Colab GPU (Tesla T4 15 360 MiB,
CUDA 13.0, driver 580.82.07), .NET SDK 10.0.404, Ubuntu 24.04, **2 vCPU**.
seq1024, mode 0 (fwd+bwd, không optimizer), 1 warmup + 1 bước đo.

| Bên | Thời gian | Ghi chú |
|---|---:|---|
| `.tkv` fwd | **9 674 ms** | CPU (C `.so`, 2 thread) |
| `.tkv` bwd | **17 094 ms** | CPU |
| `.tkv` fwd+bwd | **26 768 ms** | loss 10.760399 |
| PyTorch **CPU** (cùng VM) | **10 540.5 ms** | `TKV_BENCH_DEVICE=cpu`, loss 9.8851 |
| PyTorch "CUDA" (log cũ) | 10 540.5 ms | **thực chất là CPU** — xem bên dưới |

**Hai sai lệch phải nói rõ (không tự sửa số):**

1. **"PyTorch CUDA" trong log cũ KHÔNG phải CUDA.** `benchmark/bench_torch.py` thời
   điểm đó không có `.cuda()`/`device` nào — model và tensor mặc định ở CPU. Vì vậy
   con số 10 540 ms là **PyTorch CPU trên 2 vCPU**, không phải T4.
   Đã thêm `TKV_BENCH_DEVICE=cpu|cuda` + `torch.cuda.synchronize()` (mặc định vẫn
   `cpu` để không đổi số liệu cũ) để lần chạy sau mới có số CUDA thật.
2. **cuBLAS nạp được nhưng GPU không được dùng.** `TkvCUDA_Available() = 1`, nhưng
   hook nằm ở `SgemmF32(a,b,o,n)` — hàm này chỉ được gọi từ **kernel tile** bên trong
   `SgemmRect`/`SgemmFull` với `n = T` (kích thước tile, `T < 512`), nên điều kiện
   `n >= 512` **luôn sai** → `TkvCUDA_SgemmF32` trả 0 ngay, không `cudaMalloc`.
   Ngoài ra ngay cả khi `n >= 512` thì mỗi lần gọi vẫn `cudaMalloc` 3 buffer + 2 H2D
   + 1 D2H, nằm trong vòng lặp tile → không thể nhanh hơn CPU.

**Kết luận trung thực:** sau Phase A, `.tkv` 124M **chạy được** trên Colab (hết crash
kernel32) nhưng **chậm hơn PyTorch CPU ~2.54×** (26 768 / 10 540). Nguyên nhân:
2 vCPU, không có kernel ASM AVX2 trên Linux (op phần tử vô hướng), GEMM C nội bộ so
với MKL/oneDNN của torch. **Chưa có bằng chứng nào cho việc `.tkv` dùng GPU.**

**Việc còn lại để có số GPU thật:** (a) chạy lại notebook với `TKV_BENCH_DEVICE=cuda`
để có baseline T4; (b) muốn `.tkv` dùng GPU thì phải hook ở tầng GEMM đầy đủ
(`SgemmRect`/`SgemmFull`/`SgemmPacked`) + giữ weight trên device (xem
`docs/PLAN_GPU_BACKEND.md` §P1/P2).

## 9.26 COLAB T4: BASELINE 3 CHIỀU — `.tkv` 26.2 s / torch CPU 10.2 s / torch CUDA 0.27 s (2026-11-12)

Chạy lại `demo/colab_bench_ai_gpu.ipynb` (commit `db5ad47`) trên **cùng một phiên**
Colab: Tesla T4 15 360 MiB, driver 580.82.07, CUDA 13.0, Ubuntu 24.04, .NET SDK
10.0.404, **2 vCPU**. GPT-2 124M (vocab 50 257, n_embd 768, 12 layer, 12 head),
seq 1024, fp32, dropout 0, warmup 1 + 1 bước đo.

| Bên | fwd+bwd | full step (fwd+bwd+AdamW) |
|---|---:|---:|
| `.tkv` (CPU, C `.so`, 2 thread) | **26 230 ms** (fwd 8 727 + bwd 17 503) | — |
| PyTorch **CPU** cùng VM | **10 243.7 ms** | **11 078.9 ms** |
| PyTorch **CUDA** (T4) | **271.7 ms** | **307.1 ms** |

Tỷ lệ: `.tkv` / torch CPU = **2.56×**; `.tkv` / torch CUDA = **96.5×**;
torch CPU / torch CUDA = **37.7×**.
Độ lặp lại: `.tkv` 26 230 ms (phiên này) vs 26 768 ms (§9.25) → lệch 2.1%.
Loss `.tkv` 10.760399 vs torch 9.8851 — **khác init, không so sánh được**.

**Log GPU trong pha `.tkv` (mỗi dòng = 1 mẫu 2 s):**

```
117 MiB, 0 %      117 MiB, 4 %      231 MiB, 1 %
231 MiB, 4 %      231 MiB, 6 %      243 MiB, 6 %
```

Đây **không phải** GEMM trên GPU. `TkvCUDA_SgemmF32` gọi `cuda_init()` ở dòng đầu
(trước khi kiểm tra `n < 512`) → `cublasCreate_v2` tạo **CUDA context** ≈ 126 MiB
(117 → 243 MiB) và vài kernel khởi tạo (util 4–6%). Sau đó `n = T` (tile `T = 256`
hoặc `T = 64`, xem `SgemmFull`/`SgemmPacked`/`smm_fwd_at`) luôn `< 512` → trả 0
ngay, **không `cudaMalloc`, không `cublasSgemm`**.

**Bằng chứng chiều ngược lại (torch CUDA thật):** trong pha `TKV_BENCH_DEVICE=cuda`
log ghi `2715 MiB, 100 %` → GPU thật sự chạy, nên 271.7 ms là số T4 thật.

**Kết luận trung thực:** engine `.tkv` hiện **chỉ dùng CPU**. Trên 2 vCPU của Colab
nó chậm hơn PyTorch CPU **2.56×** (nguyên nhân: không có kernel ASM AVX2 trên Linux
→ op phần tử chạy vô hướng; GEMM C nội bộ so với MKL/oneDNN của torch). Muốn có số
GPU cho `.tkv` phải làm P1/P2 trong `docs/PLAN_GPU_BACKEND.md`, không phải chỉnh
ngưỡng `n >= 512`.

## 9.27 PHASE BREAKDOWN: 65% CHÊNH LỆCH NẰM Ở BACKWARD — VÀ BLAS CHƯA BAO GIỜ NẠP TRÊN LINUX (2026-11-12)

Cùng phiên/baseline như §9.26, thêm cell `[5/5]`: `.tkv` mode 2 (5 pha, warmup + 2
vòng, lấy min) và `bench_torch.py e 1024` với `TKV_BENCH_THREADS=2` (đúng 2 thread
GEMM pool của `.tkv` trên Colab 2 vCPU).

| Pha | `.tkv` (2 thread) | torch CPU (2 thread) | Tỷ lệ |
|---|---:|---:|---:|
| forward | 8 936 ms | 4 041 ms (fwd+CE) | 2.44× |
| CE loss | 908 ms | (nằm trong fwd của torch) | — |
| **backward** | **16 751 ms** | **6 076 ms** | **2.76×** |
| opt.step (AdamW) | **466 ms** | 634 ms | **0.73× — `.tkv` nhanh hơn** |
| zero_grad / copyback | 89 ms (copyback) | 37 ms (zero_grad) | không đáng kể |
| **TỔNG** | **27 150 ms** | **11 191 ms** | **2.43×** |

Cùng phiên, mode 0: `.tkv` 26 067 ms vs torch CPU 10 923.8 ms (**2.39×**), torch CUDA
279.5 ms (mode 0) / 311.1 ms (full step). `.tkv` lặp lại 26 067 / 26 230 / 26 768 ms
(3 phiên, lệch ≤ 2.7%).

**Phân rã khoảng chênh (~16 s):** backward ≈ **65%**, forward ≈ **36%**, opt ≈ **−1%**
(`.tkv` nhường lại), copyback ≈ 0.4%. Copyback (89 ms) rẻ → **không đáng tối ưu**;
AdamW C của `.tkv` đã nhanh hơn torch. Toàn bộ vấn đề là GEMM + op phần tử trong
fwd/bwd.

**Nguyên nhân gốc tìm được khi đọc code (chưa sửa):** nhánh BLAS trong
`tkv_simd_native.c` (`tkv_blas_load`, dòng ~154) chỉ thử **`libscipy_openblas64_.dll`**
qua `LoadLibraryA`, và tìm file cạnh module bằng `strrchr(path_buf, '\\')`. Trên Linux
shim map `LoadLibraryA` → `dlopen` nhưng tên `.so` thật là
`libscipy_openblas64_<hash>.so`, và đường dẫn không có `'\'` → **BLAS không bao giờ
nạp** → mọi GEMM dùng kernel AVX2 C nội bộ. Trên máy dev Windows cũng **không có**
`libscipy_openblas64_.dll` trong `demo/`, nên số Windows (7.5 s) cũng là kernel nội bộ.
Nói cách khác: **chưa từng có lần đo nào của `.tkv` dùng BLAS**, dù code có nhánh đó.

**Lưu ý về log GPU phiên này:** `tail -12 /content/gpu.log` của pha torch CUDA chỉ còn
`231 MiB, 6%` (phiên trước thấy `2715 MiB, 100%`). Sampler `--loop=2` ghi ra file nên
phần cuối có thể chưa flush; bản thân 279.5 ms cho 124M seq1024 là bất khả thi trên
2 vCPU nên vẫn kết luận torch CUDA thật, nhưng **log phiên này không tự chứng minh**.

## 9.28 BLAS TRÊN LINUX: ĐÃ NẠP ĐƯỢC NHƯNG **CHẬM HƠN** KERNEL NỘI BỘ (2026-11-12)

Sửa `tkv_blas_load()` (commit `7e653be`, repo TokenVector) để Linux nạp được BLAS:
thử soname `libscipy_openblas64_.so` / `libopenblas64_.so.0` / `libopenblas.so.0` /
`libmkl_rt.so.2`, rồi `glob()` các bản có hash trong `numpy.libs`. Linux **mặc định
TẮT** (opt-in bằng `TKV_GEMM_BLAS=1` hoặc `TKV_GEMM_BLAS_PATH=...`) để giữ loss
bit-identical; thêm ngưỡng `TKV_GEMM_BLAS_MIN_N` (Linux mặc định 512, Windows 0 =
hành vi cũ) vì gọi BLAS cho tile `n = T` (64/256) là lỗ.

**Đo trên WSL2** (28 logical core, OpenBLAS 0.3.20 `libopenblas.so.0`, seq1024, mode 2,
3 vòng xen kẽ native/BLAS, cùng một `.so`, `TKV_GEMM_BLAS_MIN_N=512`):

| Pha | native (3 vòng) | BLAS (3 vòng) | BLAS/native |
|---|---:|---:|---:|
| fwd | 1 962 / 1 980 / 2 011 (TB 1 984) | 2 682 / 2 956 / 2 993 (TB 2 877) | **1.45× chậm hơn** |
| bwd | 5 396 / 5 473 / 5 526 (TB 5 465) | 5 798 / 5 799 / 5 751 (TB 5 783) | **1.06× chậm hơn** |
| **TỔNG** | 8 649 / 8 777 / 8 873 (**TB 8 766**) | 9 802 / 10 091 / 10 111 (**TB 10 001**) | **1.14× chậm hơn** |

Loss `25.407898` (native) vs `25.40787` (BLAS) — khác ở chữ số thứ 6 do thứ tự cộng,
không phải sai số logic.

**Kết luận (phủ định, phải ghi lại):** bật BLAS **không** thu hẹp khoảng cách với
torch; ở 14 core nó còn chậm hơn kernel AVX2 nội bộ (overhead gọi BLAS + thread pool
riêng của OpenBLAS cạnh pool pthread của engine, và kernel nội bộ đã block cache theo
đúng shape). Vì vậy **GEMM không phải chỗ đáng tối ưu tiếp**; giả thuyết còn lại cho
khoảng cách 2.4–2.8× trên Colab là **op phần tử chạy vô hướng trên Linux** (kernel ASM
AVX2 chỉ có cho ABI Windows x64) — cần đo/kiểm chứng ở Phase B.

> **ĐÍNH CHÍNH (thêm 2026-11-12, xem §9.31):** kết luận trên **sai vì cách đo**. Phép đo
> §9.28 trao cho OpenBLAS **toàn bộ 28 thread**, mà OpenBLAS bão hoà ở 4–8 thread rồi chậm
> đi. Ở 1 thread, BLAS nhanh hơn kernel nội bộ **1.61×** và ở 2 thread là **1.40×**; trên
> Colab 2 vCPU BLAS nhanh hơn **1.54×** (§9.30). Gốc rễ thật là **microkernel**, và câu
> "GEMM không phải chỗ đáng tối ưu" **không còn đúng**. Giả thuyết "op phần tử vô hướng"
> sau đó cũng bị bác bỏ ở §9.29 (−2.2%).

**Không đo được trên Windows:** `tkv_simd_native.c` hiện **không build được bằng MSVC**
(`build.ps1`) — bản `HEAD` cũng lỗi y hệt (`t_nested` undeclared, `C2143`), tức lỗi có
sẵn từ trước, không liên quan thay đổi này; và `demo/tkv_simd_native.dll` trong repo là
ELF nên bench Windows cần dựng lại DLL PE (chưa làm).

## 9.29 OP PHẦN TỬ QUA P/INVOKE TRÊN LINUX: ĐÚNG NHƯNG CHỈ **+2.2%** (2026-11-12)

Đổi dispatcher IL (`simd_feature.tkv`, `_mang_khoi`) thành 3 tầng:

1. kernel RWX trong bộ nhớ thực thi (Windows AVX2/SSE) — như cũ;
2. **mới:** `plat == 2` (Linux/macOS) → P/Invoke thẳng `AddF32/SubF32/MulF32/ScaleF32/
   AxpyF32/DotF32/SumF32` (+F64) trong `tkv_simd_native.dll` (trên Linux là `.so`, đã
   được nạp sẵn cho GEMM);
3. vô hướng managed — chỉ còn cho Windows khi `TKV_SIMD_ISA=off`.

Kèm field `TkvSimd::plat` (1 = Win32NT, 2 = Unix/macOS) đặt trong `Init()`.
Rebuild self-host: **IL_IDENTICAL**; `testkit_gate`: **PASS 136/136, NEG 44/44**.

**Đo WSL2 (28 core, mode 2, seq1024, 3 vòng, cùng `.so`):**

| Pha | codegen cũ (scalar) | codegen mới (P/Invoke) | Thay đổi |
|---|---:|---:|---:|
| fwd | 1 984 | 1 979 | ~0% |
| bwd | 5 465 | **5 288** | −3.2% |
| **TỔNG** | 8 766 | **8 573** | **−2.2%** |
| loss | 25.407898 | 25.407898 | giống hệt |

**Kết luận (phủ định thêm một giả thuyết):** op phần tử **không** phải nút cổ chai
(< 3% tổng). Loại bỏ được cả BLAS (chậm hơn 14%) và elementwise (2%).
Giả thuyết còn lại cho gap 2.4–2.8× ở Colab: **GEMM của engine ở 2 thread chậm hơn
MKL/oneDNN ở 2 thread** — khác với phép đo 14 core ở WSL (nơi kernel nội bộ thắng
OpenBLAS). Cần đo trên Colab: `.tkv` mode 2 với `TKV_GEMM_BLAS=1` (cell `[5/5]` đã thêm).

**Windows:** không chạy được `bseq` tại máy dev vì `demo/tkv_step124.dll` và
`tkv_simd_native.dll` trong repo là ELF, không có bản PE `tkv_step124.dll` để đối chiếu
→ xác nhận Windows chỉ dựa vào `testkit_gate` (PASS) và nhánh `calli` giữ nguyên byte.

## 9.30 COLAB 2 vCPU: BLAS **NHANH HƠN 1.54×** — GIẢ THUYẾT GEMM ĐÚNG, NHƯNG CÒN CÁCH TORCH 1.55×

Phiên Colab mới (T4, 2 vCPU, Ubuntu 24.04, seq1024, GPT-2 124M, cùng phiên, cell `[5/5]`
chạy `.tkv` native rồi `TKV_GEMM_BLAS=1` rồi torch `tier_e` 2 thread):

| Pha | `.tkv` native | `.tkv` + OpenBLAS (`min_n=512`) | torch CPU (T=2) |
|---|---:|---:|---:|
| forward | 8 904 ms | 5 379 ms | 3 783 ms (fwd+CE) |
| CE loss | 1 045 ms | 1 045 ms | (gộp trong fwd) |
| backward | 20 954 ms | 13 453 ms | 8 115 ms |
| opt.step | **454 ms** | 502 ms | 1 254 ms |
| copyback | 97 ms | 98 ms | 41 ms (zero_grad) |
| **TỔNG** | **31 454 ms** | **20 477 ms** | **13 247 ms** |

- **BLAS nhanh hơn native 1.54×** (−35%): fwd −40%, bwd −36%. Loss `25.4079` (native) vs
  `25.407888` (BLAS) — chỉ khác thứ tự cộng.
- So với torch vẫn **1.55× chậm** (20 477 vs 13 247 ms); riêng AdamW `.tkv` nhanh hơn
  2.5× (502 vs 1 254 ms) nên toàn bộ khoảng chênh nằm ở fwd (1.70×) + bwd (1.66×).
- Cùng ngưỡng `min_n=512` này, trên WSL 28 core BLAS lại **chậm hơn** native 1.14×
  (§9.28) ⇒ hành vi phụ thuộc số core, không phải "BLAS luôn tốt/xấu".

**Chênh lệch tuyệt đối giữa các phiên (ghi lại để không so sai):** phiên này `.tkv` native
mode 0 = 32 640 ms (fwd 11 827 + bwd 20 813), torch CPU = 11 880.9 ms (fwd+bwd) /
13 039.6 ms (full step), torch CUDA 260.7 / 296.8 ms — VM phiên này chậm hơn phiên 1
khoảng 16–25% (native 26 067, torch 10 243.7). Vì vậy **mọi so sánh ở trên chỉ dùng tỷ lệ
trong cùng phiên**; không lấy số tuyệt đối giữa các phiên.

**Còn lại cần giải thích:** tại sao GEMM nội bộ thua OpenBLAS ở 2 thread nhưng thắng ở
28 thread. Số đo crossover trên WSL (T = 1/2/4/8) ở §9.31.

## 9.31 GỐC RỄ LÀ **MICROKERNEL 1 THREAD**, KHÔNG PHẢI THREAD POOL — VÀ §9.28 SAI VÌ ÉP BLAS 28 THREAD (2026-11-12)

Sweep `TKV_GEMM_THREADS` (ép cả pool nội bộ lẫn `openblas_set_num_threads` cùng số) trên
WSL2 28 core, mode 2, seq1024, mỗi mức 1 lần, cùng một `.so`:

| T | native TONG (fwd / bwd) | BLAS TONG (fwd / bwd) | Ai nhanh hơn |
|---:|---:|---:|---|
| 1 | 26 592 (7 257 / 18 003) | **16 529** (4 545 / 10 690) | BLAS **1.61×** |
| 2 | 16 402 (4 116 / 10 973) | **11 703** (3 307 / 7 081) | BLAS **1.40×** |
| 4 | 11 660 (2 786 / 7 503) | **8 786** (2 489 / 5 000) | BLAS **1.33×** |
| 8 | 8 838 (1 949 / 5 575) | **8 153** (2 191 / 4 657) | BLAS **1.08×** |
| 28 (mặc định) | **8 766** | 10 001 | native **1.14×** |

Hiệu suất scale (TỔNG):

| | 1→2 | 2→4 | 4→8 | 8→28 |
|---|---|---|---|---|
| native | 1.62× | 1.41× | 1.32× | 1.01× |
| BLAS | 1.41× | 1.33× | 1.08× | 0.82× (chậm đi) |

**Kết luận 1 — gốc rễ là microkernel:** ở **1 thread** (không có thread pool, không
tranh chấp) native 26 592 ms vs BLAS 16 529 ms ⇒ **kernel AVX2 nội bộ chậm hơn kernel
Haswell của OpenBLAS ~1.6×/core**. Đây là khoảng cách chất lượng codegen
(packing/prefetch/unroll/micro-tile), không phải overhead đồng bộ. Nó cũng giải thích
luôn số Colab 2 vCPU (BLAS 1.54×, khớp WSL T=2 là 1.40×).

**Kết luận 2 — §9.28 sai vì cách đo:** kết luận "BLAS là ngõ cụt" có được khi trao cho
OpenBLAS **toàn bộ 28 thread**; OpenBLAS bão hoà ở 4–8 thread rồi **chậm đi** khi thêm
thread (0.82× từ 8→28), trong khi kernel nội bộ vẫn scale. Nghĩa là **không phải BLAS
dở — mà là BLAS bị ép quá nhiều thread**. Cần đo lại với `TKV_GEMM_BLAS_THREADS` (biến
mới, 0 = không chặn) để tìm cấu hình tốt nhất trên máy nhiều core (§9.32).

**Số loss:** native `25.407898` ở mọi T; BLAS `25.40786`/`25.407883`/`25.407887` — lệch
chữ số thứ 6 do thứ tự cộng, đúng như dự kiến.

## 9.32 CHẶN THREAD CHO BLAS (`TKV_GEMM_BLAS_THREADS`): BLAS TRỞ THÀNH LỰA CHỌN TỐT Ở **MỌI** SỐ CORE (2026-11-12)

Thêm `TKV_GEMM_BLAS_THREADS` (0 = không chặn) cho phép OpenBLAS dùng ít thread hơn pool
của engine. WSL2 28 core, mode 2, seq1024, mỗi cấu hình 1 lần:

| Cấu hình | fwd | bwd | TỔNG |
|---|---:|---:|---:|
| native (pool 28) | **2 053** | 5 504 | 8 886 |
| BLAS cap 4 | 2 465 | 5 180 | 8 942 |
| **BLAS cap 8** | 2 261 | **4 647** | **8 226** |
| BLAS cap 16 | 2 559 | 4 920 | 8 777 |
| BLAS cap 0 (=28, cũ) | 2 940 | 5 783 | 10 034 |
| T=2 native | 4 182 | 10 847 | 16 297 |
| **T=2 BLAS `min_n=0`** | 3 187 | 6 912 | **11 352** |

- **BLAS cap 8 thắng cả hai đầu:** 28 core (8 226 vs native 8 886, −7.4%) và 2 vCPU trên
  Colab (§9.30, −35%). Lúc đó fwd thua native ở 28 core (2 261 vs 2 053) nhưng bwd thắng
  gấp (4 647 vs 5 504) nên tổng vẫn thắng.
- Ngưỡng `min_n=512` là **không** tối ưu ở 2 thread: `min_n=0` cho 11 352 vs 11 703 (−3%).
  Trên máy nhiều core thì `min_n=0` chậm hơn (§9.28 lúc đầu) — nhưng ở thời điểm đó BLAS
  bị ép 28 thread nên kết luận đó vẫn chưa được kiểm chứng lại.

**Bài học thực nghiệm:** một thư viện BLAS có thể **tốt hơn ở mọi số thread** nếu để nó
tự chạy với số thread vừa phải; ép quá 28 thread làm hỏng kết quả. Hướng sửa hiện tại:
chọn BLAS với cap ≈ 8 thay vì viết lại microkernel AVX2 (việc lớn, nhiều ngày).

**Xác nhận 3 vòng xen kẽ (cùng phiên, cùng `.so`; fwd/bwd):**

| vòng | native TỔNG | BLAS cap8 TỔNG | Δ |
|---:|---:|---:|---:|
| 1 | 8 393 (1 971 / 5 085) | 7 332 (1 983 / 4 031) | −12.6% |
| 2 | 8 409 (1 985 / 5 105) | 7 194 (1 851 / 4 029) | −14.4% |
| 3 | 8 296 (1 928 / 5 084) | 7 221 (1 829 / 4 104) | −13.0% |
| **TB** | **8 366** | **7 249** | **−13.4% (BLAS 1.15×)** |

Lệch ổn định qua 3 vòng ⇒ kết quả thật, không phải nhiễu. Phần cải thiện tới từ **bwd**
(native 5 091 → BLAS 4 055 ms, −20%), fwd gần như bằng nhau (1 961 → 1 887).
Thử nhanh một lần: cap 6 = 7 775 ms, cap 10 = 7 540 ms ⇒ **cap 8 là bắp cầu tốt nhất**
trên máy này. Loss BLAS `25.407887` ở mọi vòng (native `25.407898`).

## 9.33 `min_n` KHÔNG PHẢI NÚT; CẤU HÌNH ĐỀ XUẤT: BLAS + cap 8 + `min_n=512` (2026-11-12)

| Cấu hình | TỔNG | Ghi chú |
|---|---:|---|
| 28T BLAS cap8 `min_n=0` | 7 527 | mọi tile đều đi BLAS |
| 28T BLAS cap8 `min_n=256` | 7 388 | |
| 28T BLAS cap8 `min_n=512` | **7 249** (TB 3 vòng) | **giữ** — an toàn, không call BLAS cho tile `n = T` |
| 2T BLAS `min_n=0` cap 2 | 10 700 | mô phỏng Colab 2 vCPU |
| 2T BLAS `min_n=0` cap 8 | 10 563 | cap vô nghĩa khi pool = 2 |

Chênh lệch `min_n` 0/256/512 nằm trong nhiễu (7 527 vs 7 388 vs 7 249, khoảng 3% ở
phép đơn lẻ) nên **giữ `min_n=512`** vì lý do thiết kế: tránh gọi BLAS cho từng tile
`n = T` (64/256) bên trong vòng lặp packed.

**Cấu hình đề xuất (Linux):** BLAS bật sẵn, `TKV_GEMM_BLAS_THREADS=8`, `min_n=512`;
`TKV_GEMM_BLAS=0` để quay lại kernel nội bộ. Kết quả:

| Môi trường | native | BLAS cap8 | Cải thiện |
|---|---:|---:|---:|
| Colab 2 vCPU (§9.30) | 31 454 ms | 20 477 ms | **−35%** |
| WSL 28 core (§9.32) | 8 366 ms | 7 249 ms | **−13.4%** |

**Đổi lại:** loss lệch ở chữ số thứ 6 (`25.407898` → `25.407887`), tức **không còn
bit-identical** với bản native/Windows. Với Colab, `.tkv` vẫn còn **1.55× chậm torch**
(20 477 vs 13 247 ms) — phần còn lại ở fwd (1.70×) + bwd (1.66×) ngoài GEMM.

**ĐÃ QUYẾT (2026-11-13, theo lựa chọn của user): bật BLAS mặc định trên Linux/macOS.**
`tkv_blas_load` không còn ngăn opt-in trên POSIX; `TKV_GEMM_BLAS=0` để quay lại kernel
AVX2 nội bộ (loss EXACT). `tkv_blas_threads` mặc định **cap 8** trên POSIX, 0 (không
chặn) trên Windows. Windows không đổi hành vi.

**Kiểm chứng WSL2 28 core, mode 2, seq1024, 3 vòng xen kẽ (cùng `.so`, không env):**

| | TỔNG | loss |
|---|---:|---|
| MẶC ĐỊNH mới (BLAS tự nạp) | 7 690 / 7 466 / 7 628 → **TB 7 595** | `25.407887` |
| `TKV_GEMM_BLAS=0` (native) | 8 358 / 8 376 / 8 364 → **TB 8 366** | `25.407898` |

→ Mặc định mới nhanh hơn **9.2%** ở 28 core (đo độc lập 3 vòng khác cho −13.4%; nhiễu
máy WSL ±4%) và **−35%** ở Colab 2 vCPU. Commit `c9becd3` (TokenVector).

## 9.34 COLAB XÁC NHẬN: BLAS MẶC ĐỊNH NHANH HƠN 1.45×; `.tkv` **KHÔNG** CHẠY TRÊN GPU (2026-11-13)

Phiên Colab mới (T4, 2 vCPU; `[5/5]`, cùng phiên, VM nhanh hơn phiên trước ~15%: torch
CPU 11 220.9 → dùng tỷ lệ trong phiên):

| Pha | native (`TKV_GEMM_BLAS=0`) | **MẶC ĐỊNH (BLAS cap 8)** | torch CPU (T=2) |
|---|---:|---:|---:|
| forward | 7 738 ms | 5 246 ms (−32%) | 3 225 ms |
| CE loss | 1 552 ms | 1 009 ms | (gộp trong fwd) |
| **backward** | **17 246 ms** | **11 821 ms** (−31%) | **7 937 ms** |
| opt.step | 430 ms | 469 ms | 594 ms |
| copyback | 87 ms | 91 ms | 17 ms (zero_grad) |
| **TỔNG** | **27 053 ms** | **18 636 ms** | **11 896 ms** |

- BLAS nhanh hơn native **1.45×**; loss `25.407888` vs `25.4079` (như dự kiến).
- `.tkv` còn **1.57× chậm torch**; khoảng cách giờ nằm ở fwd+CE (1.94×) và bwd (1.49×).
  bwd chiếm **63%** TỔNG (11 821/18 636).
- mode 0 (`[4/5]`), mặc định BLAS: `.tkv` **19 977 ms** (fwd 5 706 + bwd 14 271) vs
  torch CPU 11 220.9 → **1.78×**; torch CUDA 292.8 / 322.4 ms.

**Xác nhận tận mắt: `.tkv` không dùng GPU.** Sampler `nvidia-smi` trong lúc `.tkv`
chạy: `117–231 MiB, 0–3 %`; lúc torch CUDA chạy: `2 311 / 3 527 MiB, 99–100 %`.
`TkvCUDA_Available() = 1` chỉ nghĩa là cuBLAS nạp được, **không phải GEMM chạy trên
GPU**: hook P0 chỉ nằm trong `SgemmF32` (kernel tile `T < 512`, điều kiện `n >= 512`
không bao giờ đúng) và mỗi lần gọi lại `cudaMalloc` + 2 H2D + 1 D2H (xem
`docs/PLAN_GPU_BACKEND.md` §0). Muốn GPU thật phải làm P1 (hook đầy đủ + device-resident).

## 9.35 SPIKE P1/P0: `TKV_GEMM_CUDA=1` NHANH HƠN TORCH CPU THẬT SỰ (2026-11-13)

Sau khi wiring `TkvCUDA_RectF32` (cuBLAS row-major, buffer device cache grow-only,
mutex, `beta=1` copy C; commit `3d8d8d8` TokenVector + `1f1de4b` Inference), chạy Colab
T4, cell `[5/5]` mode 2 seq1024, cùng phiên với `[4/5]` (mode 0) và torch `tier_e` 2 thread:

| Pha | native (`BLAS=0`) | MẶC ĐỊNH BLAS cap 8 | **cuBLAS (`CUDA=1`)** | torch T=2 |
|---|---:|---:|---:|---:|
| forward | 8 193 ms | 4 483 ms | **2 108 ms** | 3 089 ms |
| CE loss | 880 ms | 901 ms | 904 ms | (gộp) |
| **backward** | **17 963 ms** | **11 316 ms** | **5 310 ms** | 6 826 ms |
| opt.step | 428 ms | 451 ms | 433 ms | 621 ms |
| copyback | 93 ms | 85 ms | 84 ms | 32 ms (zero_grad) |
| **TỔNG** | 27 557 ms | 17 236 ms | **8 839 ms** | **11 096 ms** |

- cuBLAS nhanh hơn BLAS mặc định **1.95×**, nhanh hơn native **3.1×**.
- cuBLAS **nhanh hơn PyTorch CPU 1.26×** (8 839 vs 11 096 ms) — và vẫn chậm torch CUDA
  **~32×** (273.5 / 303.1 ms). fwd+CE 3 012 vs torch CPU 3 089 → 1.03×; bwd 5 310 vs
  6 826 → **1.29× nhanh hơn**; opt 1.43×.
- Cell `[4/5]` (mode 0), mặc định BLAS: `.tkv` 17 476 ms (fwd 5 482 + bwd 11 994) vs
  torch CPU 10 440 ms (**1.67× chậm**) / torch CUDA 273.5 / 303.1 ms.
- Loss khi dùng cuBLAS: `25.407866` (BLAS `25.407888`, native `25.407898`) — lệch chữ số
  thứ 6, gate `1e-5` vẫn qua. Sampler GPU của `[4/5]` vẫn chỉ thấy `.tkv` ở 117 MiB / 0–4%
  (nhánh cuBLAS nằm ở `[5/5]` nên không được sample) — **nhưng tốc độ 2 108/5 310 ms là
  bằng chứng việc cuBLAS thật sự chạy GPU**, không phải nhiễu CPU (2 thread).

**Chưa công bằng đầy đủ:** nhánh cuBLAS dùng GPU (không phải CPU-vs-CPU). Muốn so sánh
apples-to-apples CPU-vs-CPU thì dùng nhánh BLAS (+full profile). Muốn đánh giá hết hoặc
đuổi torch CUDA cần **P2** (device-resident weight/activation, bỏ H2D mỗi GEMM).

## 9.36 P2-LITE: WEIGHT-B CACHE TRÊN GPU (chưa đo được hiệu năng) (2026-11-13)

Increment đầu của P2 — **đã phải sửa lỗi lớn**: bản đầu (commit `946500e`) suy đoán
"A không Trans ⇒ B là weight" và cache B cho **mọi** GEMM loại đó — kể cả packed path,
trong đó **B có thể là activation** (attention: `attn_xgemm` pack K/V). Kết quả Colab:
loss_val = **13.831469** (đúng phải là ~25.408) — sai kết quả, không phải lệch 1e-6.
Đã **fix (chưa push)**: cache B chỉ bật tường minh qua `cacheB=1`, và chỉ dùng ở
`tkv_try_cuda_rect_tb` (**LinearDxFused: dx = gy @ w^T**, B chắc chắn là weight).
Packed path / full GEMM / dW / SgemmF32 đều `cacheB=0`.

API đổi thành `TkvCUDA_RectF32(..., int transA, int transB, int accumulate, int cacheB)`.
`TkvCUDA_MarkWeightsDirty()` vẫn gọi trong `adamw_step_f32_impl` để epoch tăng sau mỗi
AdamW; cache 96 entry, grow-only, mutex.

**Kiểm chứng:** WSL build OK (GPU-less), không thể tái hiện sai số ở đây; fallback
không đổi (loss `25.407887`). **Phải Colab chạy `TKV_GEMM_CUDA=1` lại** để xác nhận
loss quay về `25.4079xx` — và weight-resident trong LinearDxFused cho fwd/bwd mới biết.

**RAM:** cell `[7/7]` đo `VmHWM` `.tkv` vs `torch tier_d` (`/usr/bin/time -v`):
nhánh `.tkv` (BLAS mặc định) peak = **4 090 192 KB ≈ 4.09 GB**; torch tier_d peak =
**4 299 828 KB ≈ 4.30 GB** → **`.tkv` dùng ÍT hơn torch ~210 MB (5%)**. Cả hai ~4 GB
(chủ yếu là bộ nhớ runtime .NET/Python + tensor 124M params + optimizer states).

**RAM:** cell `[7/7]` mới (Inference `6636f94`) đo `VmHWM` `.tkv` vs `PyTorch tier_d`
(fwd+bwd, T=2, mode 0) bằng `/usr/bin/time -v`; kết quả chưa có (chờ user chạy).

## 9.37 PROTOCOL BENCHMARK CHUẨN (2026-11-13) — MỌI SỐ ĐỀU CẦN ĐIỀU KIỆN BẰNG NHAU

Yêu cầu người dùng: **mọi thứ đo phải bằng nhau về điều kiện mới gọi là benchmark.**
Đối chiếu thực trạng sau khi dọn:

| Điều kiện | `.tkv` | torch | Kết luận |
|---|---|---|---|
| Model | GPT-2 124M, seq1024 | GPT-2 124M, seq1024 | **bằng** |
| Input tokens | `[(i*37+11) % 50257]` | `[(i*37+11) % 50257]` | **bằng** |
| Seed | `gpt2_init(7)` | `manual_seed(7)` | **bằng** (nhưng scheme khác → params khác) |
| Threads | 2 (2 vCPU, cap BLAS 8) | 2 (`TKV_BENCH_THREADS=2`) | **bằng** |
| Timing | mode 0 = **1 bước** (warmup 1); mode 2 = warmup + 1 bước 5 pha | median n=2 | **KHÁC** — single-shot vs median |
| Thứ tự đo | .tkv chạy hết rồi mới torch | — | **KHÁC** — dễ bị VM drift (±15–25% giữa các phiên) |
| Device | BLAS (CPU) hoặc `TKV_GEMM_CUDA=1` (GPU) | cpu (`TKV_BENCH_DEVICE=cpu`) hoặc cuda | **bằng** khi cùng flag |
| Loss | `10.76/25.41` | `9.8851` | **KHÔNG SO ĐƯỢC** (init khác; FLOPs giống nhau) |

**Hệ quả:** các con số §9.24–§9.36 đo đúng hướng nhưng chưa chặt: (a) timing đơn lẻ (nhiễu
±10–20%), (b) không xen kẽ A/B, (c) loss khác nhau gây hiểu sai; (d) sampler GPU chỉ phủ
[4/5].

**Đã làm (notebook Inference):**
- Cell `[8/8]`: A/B **xen kẽ** 3 vòng mỗi bên (`.tkv` mode 0 vs torch tier_d, CPU; và
  `.tkv` cuBLAS vs torch CUDA, GPU), lấy **median** → loại VM drift, cùng điều kiện.
- Cell `[5/5]`: **sampler GPU riêng cho nhánh cuBLAS** (`gpu_cublas.log`) — giờ phủ cả
  mode 0 lẫn (một phần) [5/5].
- Ghi nhận trong HANDOFF: loss `.tkv` ≠ torch là do init khác, **không** phải sai số;
  đúng đắn chỉ được kiểm bởi `tv/tests` fixtures/parity, không phải bench.

**Điều kiện ĐÚNG để so:** chỉ so cùng device (CPU BLAS vs torch CPU; cuBLAS vs torch CUDA);
chỉ so Thời gian, không so loss; mỗi con số phải kèm env flags (`TKV_GEMM_BLAS`,
`TKV_GEMM_CUDA`, `TKV_BENCH_THREADS`).

> **CẬP NHẬT 2026-11-13 (commit `c83545e`, notebook fair-only):** mọi phép đo KHÔNG
> theo protocol trên đã bị **xoá khỏi notebook** — chỉ giữ `[1/2]` (phase BLAS-vs-torch
> T=2, xen kẽ A/B/A/B ×3, median) và `[2/2]` (.tkv cuBLAS vs torch CUDA, xen kẽ ×3) +
> cell RAM. `[4/5]` (mode 0 đơn lẻ + torch), `[5/5]` (mode 2 đơn lẻ + torch tier_e
> không xen kẽ) đã xoá; bản cũ trong git nếu cần.

## 9.38 TÁCH TỪNG BƯỚC — SỐ FAIR INTERLEAVED (notebook #13, 2026-11-13)

Cell `[1/2]`, xen kẽ A/B/A/B ×3 mỗi bên, median:

| Pha | `.tkv` BLAS | torch T=2 | Tỷ lệ |
|---|---:|---:|---:|
| **fwd (+CE)** | **4 216–4 263** (round 1/2) — **5 020** (round 3); median +CE = **5 165** | 3 066–3 109 | **1.67×** |
| **bwd** | 10 643 / 11 186 / 11 359 → **11 186** | 5 931 / 7 122 / 7 412 → **7 122** | **1.57×** |
| opt.step | 419 / 429 / 432 → **429** | 587 / 601 / 636 → **601** | **0.71× — `.tkv` thắng** |
| **TỔNG** | 16 797 / 17 039 / 17 061 → **17 039 ms** | 10 658 / 11 014 / 11 114 → **11 018 ms** | **1.55×** |

GPU `[2/2]`: `.tkv` cuBLAS median **9 728 ms** (vong 8 762 / 9 728 / 9 731; fwd
3 113–3 878, bwd 5 599–6 615 — nhiễu) vs torch CUDA **286.0 ms** → **34.0×**; sampler:
1 009–1 111 MiB, util **0–9%** — GPU gần như nhàn rỗi (CPU-driven).
RAM: `.tkv` 4.09 GB < torch 4.29 GB.

**Hai pha đang thua:** fwd(+CE) 1.67× và bwd 1.57× (xấp xỉ nhau — không còn một pha
duy nhất "thủ phạm"); opt thắng rõ. So với single-shot trước đây, tỉ lệ bwd thực chất
~1.57× chứ không 1.9×; fwd là ~1.67×.

**Điểm đáng chú ý:** `.tkv` bwd rất ổn định (10 643–11 359) trong khi torch bwd dao
động 5 931–7 412 — có thể T=2 tail; ít nhất .tkv không vô cớ ĐỨNG IM.

**Điều chỉnh đề xuất (dựa trên bằng chứng, chưa sửa code):**
1. **Không đoán — instrument per-op** (`TKV_PROF` timers quanh `AttentionBwd*`,
   `TkvTorchLayerNormBwd`, GELU/softmax/CE bwd, `LinearDwAtFused`/`LinearDxFused`) —
   ~1 ngày, để biết chính xác op nào ăn bao nhiêu %.
2. Nhắm **fwd+CE trước** vì 1.67×: CE loss (gpt2_loss + grad) nếu chạy vô hướng trên
   managed list[f32] có thể ~900 ms; kiểm bằng instrument.
3. bwd: phần GEMM đã cuBLAS (trong GPU path 5.4–6.6 s); phần còn lại = attention bwd +
   LN/GELU/softmax — instrument sẽ cho biết.

Ràng buộc: mọi số ở đây chỉ valid trong cùng phiên; giữa-phiên lệch ±15%.
