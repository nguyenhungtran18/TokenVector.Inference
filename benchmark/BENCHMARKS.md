# Benchmark cuối: train GPT2-medium, TokenVector thuần vs PyTorch

**Trạng thái:** 2026-11-27, Kaggle GPU **Tesla T4** (`torch 2.11.0+cu130`, CUDA 13.0),
cùng một session cho cả hai bên. Repo `TokenVector.Inference@main`, bản trainer
`cuda-backend/Release/cuda_backend/examples/ilgpt2_long.exe`.

Cấu hình: `NL=24, T=512, H=1024, heads=16, head_dim=64, V=50257`, AdamW
`betas=(0.9,0.999), eps=1e-8, wd=0`. Tốc độ là **median ms/step** (PyTorch:
10 bước đo sau 2 bước warm-up, sync CUDA mỗi bước; TokenVector: các bước 4–10
của bản 10 bước).

| Implementation | median ms/step | vs PyTorch | Loss (bước đầu → cuối) |
| :--- | ---: | ---: | :--- |
| PyTorch eager, fp32, AdamW | **447.4** | 1.00× | 9.5467 → 4.2756 |
| **TokenVector pure `.tkv` (bản cuối)** | **181.5** | **2.46×** | 11.1816 → 1.2534 |
| C++-generated bridge (baseline cũ) | 187.8 | 2.38× | — |

TokenVector cuối cũng nhanh hơn bridge C++ **~3.4%** trong cùng lần đo này
(các lần trước: 176.1–181.5 ms/step). Không có C/C++ và không có torch trong
đường chạy TokenVector; torch chỉ là đối chiếu.

## Cách chạy lại

Trên Kaggle (sau khi cài .NET 10 và clone repo này):

```python
import subprocess, sys
d = '/kaggle/working/TokenVectorInfer'
r = subprocess.run([sys.executable, d+'/benchmark/bench_final_torch_tkv.py', d],
                   capture_output=True, text=True, timeout=1800)
print(r.stdout[-3000:])
```

Script `benchmark/bench_final_torch_tkv.py` tự dựng model PyTorch cùng cấu hình,
đo torch trước, giải phóng GPU rồi chạy bản TokenVector 10 bước và in bảng trên.

## Ghi chú về số liệu cũ

Các bảng CPU (`compare_rivals.tkv`, `bseq*.exe`, `fullstep_min*`, log
`_bench_matrix_post14_*`) và các probe GEMM cũ đã bị xoá: chúng đo các phiên bản
cũ hơn, không còn đại diện cho bản cuối. Chỉ giữ script benchmark cuối và bảng
này. Chi tiết hành trình tối ưu xem `../docs/ILGPT2_BLOCK_FORWARD_STATUS_2026-11-26.md`.
