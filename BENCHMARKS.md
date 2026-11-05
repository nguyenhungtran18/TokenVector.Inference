# TokenVector.Inference Competitor Benchmark & Stress Test Report

## 1. Executive Summary
Benchmark and stress suites target the pure **TokenVector (`.tkv`)** engine only. Results
must be measured with the `tkvc` toolchain on the hardware under test — **do not quote
numbers from any previous backend**.

```powershell
$tkvc = "D:\TokenVector\3.code\dist\tkvc_patched.exe"
& $tkvc build tv/benchmarks/compare_rivals.tkv --out compare_rivals.exe
.\compare_rivals.exe
# or, with the whole pipeline:
powershell -File build_inference.ps1 -BenchmarkTimeoutSeconds 300
```

**Read this before quoting any number.** As of 2026-09-28 the timing helpers in
`compare_rivals.tkv` (`now_ms`, `now_us`, `gc_allocated_bytes`, `gc_counts`) are still
stubs returning `0`, so the suite prints `0.0 ms`, `0 B` and `req/s` values derived from
a division by zero. **No timing number in this suite is currently meaningful.** The
`QUICK-RUN SCALE` banner in the output says so too. Wire up real timing helpers before
recording anything.

Hardware note: there is **no SIMD**. `tv/optimization/kernels.tkv` uses a hand-rolled
8-wide blocked loop (`VEC_W = 8`) that is still scalar, so a machine with AVX2/FMA gains
nothing today. See `SIMD_NOTES.md` for why and what the compiler would need.

---

## 2. Suite Structure (compare_rivals.tkv)

The suite has six scenarios. The first four are benchmarks, the last two are stress.

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
gates, not a performance comparison against the C# SIMD baseline.

`stress_concurrent` reports `ok=` and `err=`: an `err` count above zero means requests
failed rather than merely ran slowly.

---

## 3. Recording Results

When you run the suite, append a dated table here with:

- Host CPU model and flags
- `tkvc` build id / commit hash
- Scenario, metric, median ± spread
- Rival baseline command lines for the same machine (ONNX Runtime, llama.cpp, PyTorch) if compared

Until a fresh run is recorded, leave result cells empty rather than copying historical
figures. Same rule for the hardcoded ORT/Torch constants in the source: they are
literals in the file, not measurements from this machine.


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
powershell -ExecutionPolicy Bypass -File tools/simd_native/run_cpuid.ps1    # AVX2/AVX-512, XCR0
powershell -ExecutionPolicy Bypass -File tools/simd_native/run_avx512.ps1   # 0xC000001D = không dùng được
```

Máy hiện tại: **AVX2 + FMA, KHÔNG AVX-512** (`AVX512F=0`, `/arch:AVX512` → illegal instruction),
14 logical = **7 core vật lý + SMT**. Đỉnh đo được: **75.5 GFLOP/s 1 luồng**, **~515 GFLOP/s 14 luồng**
(torch 76.2/476.6; numpy OpenBLAS 75.7). tkv GEMM đạt 34.5 GFLOP/s (46% đỉnh) ở 1 luồng và scale 4.52×.
Micro-kernel: `tools/simd_native/bench_micro.c` (4 biến thể, có kiểm bit-exact bằng `memcmp`).

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
python tools/bench_torch.py e <seq>      # phase breakdown T=1 và T=14
python tools/bench_torch.py d <seq>      # fwd+bwd và full step
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
