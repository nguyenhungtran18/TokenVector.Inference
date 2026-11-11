# BENCHMARK — TokenVector (scalar / SIMD) vs PyTorch CPU — 2026-10-19

**Máy**: Intel Xeon E5-2695 v3 @ 2.30 GHz (14C/28T, AVX2),  Windows.
**Phần mềm**: PyTorch 2.14.0+cpu (`torch.get_num_threads()=14`, mặc định), transformers 5.15.1;
tkv: `tkvc-tv.exe` (build 23:58 2026-10-17) + `simd_gemm` AVX2+FMA (`simd_matmul.tkv`).
**Đối tượng**: 1 train step = forward + cross-entropy + backward + AdamW step (fp32,
dropout=0, betas 0.9/0.999, eps 1e-8, wd 0, lr 0.0025 — cùng hợp đồng 2 bên).
**Cách đo**: warmup rồi lấy trung vị; tkv in tổng ms cho N bước (chia tay), mỗi benchmark
chạy ≥2 lần — tkv ổn định <5%; torch tier A dao động 4.3–5.6 ms giữa các lần (bảng lấy 5.6).

Chạy lại (tkv ổn định <5% giữa 2 lần; torch A dao động 4.3–5.6 ms giữa các
lần — số trong bảng là lần đo ghi ở mục "Số đo thô"):

```
python benchmark/bench_torch.py [a|b|c|all]   # hop "ab"/"bc" cung duoc
tkvc tv/tests/simd_bench.tkv %TEMP%/simd_bench.exe main      && %TEMP%/simd_bench.exe
tkvc tv/tests/simd_bench_124m.tkv %TEMP%/b124.exe main       && %TEMP%/b124.exe
```

## A. GEMM: linear 768→2304, 256 token (fwd+bwd, ~2.75 GFLOP)

| Đường | ms / lượt | Hiệu năng |
|---|---|---|
| **PyTorch 2.14** (MKL, 14T) | **5.6** | ~490 GFLOPS |
| tkv **SIMD** | **323** | ~8.5 GFLOPS |
| tkv scalar | 15 800 | ~0.17 GFLOPS |

- **SIMD nhanh hơn tkv scalar ~49×** (chính = số đo gate: fwd 3390→63 ms).
- **PyTorch vẫn nhanh hơn tkv-SIMD ~58×.**

## B. GPT-2 nhỏ (2/4/64, vocab 256, seq 8, 118 528 tham số) — train step

| Đường | ms / step |
|---|---|
| **PyTorch** | **8.8** |
| tkv scalar | 25.8 |
| tkv SIMD | 32.1 |

- **SIMD CHẬM HƠN scalar ~25% ở model nhỏ** (lặp lại ổn định 2/2 lần).
  Nguyên nhân: GEMM cực nhỏ (M=8 token, 64-dim) — chi phí dựng tile 64×64 bằng
  `append` từng phần tử (workaround `list[f32]` cho `simd_gemm`) + đệm 8→64 hàng
  (12% hữu dụng) lớn hơn phần tiết kiệm.
- PyTorch nhanh hơn tkv scalar 2.9×, nhanh hơn tkv-SIMD 3.6×.

## C. GPT-2 124M thật (12/12/768, vocab 50257, seq 8, 124 439 808 tham số)

Cấu hình giống hệt nhau: `transformers` báo `params=124439808` == tkv
`n_param=124439808`; loss bước đầu scalar=tkv-simd=10.674946 (cùng seed).

| Đường | ms / step |
|---|---|
| **PyTorch** (trung vị 3 step) | **999.8** |
| tkv **SIMD** (3 step, warmup 1) | 31 333 |
| tkv scalar (1 step, không warmup) | 52 578 |

- **SIMD nhanh hơn scalar ~1.7×** ở 124M — thấp hơn 49× của tier A vì:
  1. **seq=8 → GEMM "mảnh"** (M=8): tile SIMD đệm 8/64 hàng, copy tile lặp mỗi ô;
  2. **Phần scalar chiếm ưu thế** — AdamW (498M lượt chạm m/v/g/p), copy grad,
     unflatten, softmax head — không đi qua SIMD (giống nhau ở mọi đường);
  3. Matmul chỉ là ~21/52 s delta giữa 2 đường.
- **PyTorch nhanh hơn tkv-SIMD ~31×** (MKL cache-blocking + 14 thread vs tile copy
  + scalar phần lớn).

## Kết luận & hướng tối ưu (chưa làm — chỉ nêu)

1. **Benchmark honestly**: SIMD thắng đậm ở GEMM lớn đơn thuần (49×) nhưng profit
   thật của 1 train step còn bị 2 thứ nuốt: overhead dựng tile và AdamW/scalar phần.
2. **Tối ưu tiếp (theo tác động)**:
   - Bỏ per-element copy: cache tile reuse (A-tile không đổi khi quét j; B-tile
     không đổi khi quét i) hoặc nạp thẳng `List[Single]` thay vì `append((d-0.0)/1.0)`.
   - SIMD/vectorize AdamW + grad-copy (498M chạm/step ≈ phần lớn 30 s còn lại).
   - Thử seq lớn hơn (1024): GEMM chiếm tỉ trọng lớn hơn, tile hữu dụng 16× —
     khoảng cách với PyTorch dự kiến thu hẹp; scalar ở seq 1024 sẽ không khả thi.
3. Bug đo: **`now_ms()` trong local của hàm đọc ra ≈0/garbage** (f64 — probe W);
   timing phải đặt trong `main` — đã ghi HANDOFF bug #8.

## Số đo thô

```
TORCH A: median=5.6 ms (n=10, min5.1 max6.3)
TORCH B: median=8.8 ms (n=10, min8.6 max10.0) first_loss=4.9400 params=118528
TORCH C: median=999.8 ms (n=3, min978.6 max1059.1) first_loss=9.9140 params=124439808
TKV   A scalar: 31688 ms / 2 rounds   (lan 2: 31515)
TKV   A simd:    3219 ms / 10 rounds  (lan 2:  3235)
TKV   B scalar:   250 ms / 10 steps   (lan 2:   266)
TKV   B simd:     313 ms / 10 steps   (lan 2:   328)
TKV   C simd:   94000 ms / 3 steps (warmup1)  losses 10.674946 -> 8.549695 -> 9.7539015
TKV   C scalar: 52578 ms / 1 step (no warmup) loss 10.674946
```

Lưu ý: loss 124M tăng ngược giữa các bước (lr 0.05 + init ngẫu nhiên) — không ảnh
hưởng số đo tốc độ; chuẩn hóa / lr là việc của train, không phải benchmark.
