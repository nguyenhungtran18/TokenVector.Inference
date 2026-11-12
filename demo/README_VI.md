# Demo — Train AI trên Colab (TokenVector.Inference)

Mục đích: chứng minh repo có thể huấn luyện AI bằng một lần chạy thật trên Colab,
không cần biên dịch `.tkv` (trình biên dịch `tkvc-tv` chạy trên Windows).

## Chạy

Mở `colab_train_ai.ipynb` trong Google Colab (File > Upload notebook, hoặc
"Open in Colab" từ GitHub). Notebook sẽ:

1. Cài `.NET 10 runtime` (curl dotnet-install.sh).
2. Clone repo về.
3. Chạy `dotnet demo/trainer_tests.dll`.
4. In tóm tắt: 15/15 PASS, loss cuối ≈ 0.0149, acc = 1.0, khớp PyTorch 1e-07.

## Nội dung demo

| File | Vai trò |
|---|---|
| `trainer_tests.dll` | Managed assembly biên dịch từ `tv/tests/trainer_tests.tkv` (MLP 4→8→3, 400 bước, AdamW + warmup/cosine + clip) |
| `tkv_simd_native.dll` | Linux .so build từ `Release/tools/simd_native/tkv_simd_native.c` (AVX2+FMA, pthread pool) |
| `trainer_tests.runtimeconfig.json` | net10.0 runtimeconfig cho .NET |
| `colab_train_ai.ipynb` | Notebook chạy demo + tóm tắt |

## Bài toán & cách kiểm chứng

MLP 4→8→3, 20 mẫu synthetic (fixture `tv/tests/fixtures/train_cases.txt`,
PyTorch reference). Các bước kiểm chứng (mọi gate phải PASS, tol ≤ 1e-5):

- Bước 1: loss/grad norm/lr/trọng số khớp PyTorch (1.19e-07)
- Lịch lr warmup + cosine, 400 bước
- Clip norm, resume bit-exact
- Overfit 20 mẫu: loss cuối 0.0149, acc 1.0

Đây là **cùng một chương trình training** đã PASS trên Windows và WSL Ubuntu —
xem `docs/HANDOFF.md` §0.153 và `BENCHMARKING.md` §9.22.

> Lưu ý: native lib yêu cầu CPU có AVX2. Hầu hết VM Colab có AVX2.
> Nếu chạy lỗi `Illegal instruction`, máy Colab đó không có AVX2.

## Notebook benchmark

| Notebook | Đo gì | Ghi chú |
|---|---|---|
| `colab_bench_ai.ipynb` | MLP 4→8→3, 400 bước: `.tkv` vs PyTorch | Kết quả §9.23: `.tkv 0.241 s` vs torch `0.495 s` |
| `colab_bench_ai_gpu.ipynb` | GPT-2 124M fwd+bwd @seq1024: `.tkv` vs PyTorch CUDA | Từ 2026-11-12 `.tkv` chạy được trên Linux (Phase A) |

`colab_bench_ai_gpu.ipynb` build `tkv_simd_native.so` từ source (có dispatch cuBLAS),
copy vào `demo/`, in `TkvCUDA_Available()` và log `nvidia-smi` trong lúc chạy để biết
GPU có thực sự được dùng hay không.

> Trên Linux, kernel ASM AVX2 (Windows x64 ABI) không dùng được: `.tkv` tự rơi về
> fallback vô hướng cho op phần tử, còn GEMM/attention/AdamW vẫn chạy trong C `.so`.
> Vì vậy phép so sánh là **.tkv CPU** vs **PyTorch GPU** (trừ khi cuBLAS load được —
> khi đó một phần GEMM của `.tkv` chạy trên GPU).
