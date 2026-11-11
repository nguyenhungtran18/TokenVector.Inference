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