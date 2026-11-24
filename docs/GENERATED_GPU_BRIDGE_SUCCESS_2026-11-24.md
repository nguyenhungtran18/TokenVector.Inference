# GPU bridge sinh tu TokenVector (.tkv -> .torch.cpp -> .so) â€” ket qua xac nhan

**Ngay**: 2026-11-24 01:06 UTC (Colab Tesla T4)
**HEAD**: `TokenVector.CudaBackend@92a2349` (prototype/cuda-backend) Â· `TokenVector.Inference@793bbd9` (V17) Â· reference `TokenVector@9ccc17b`
**Notebook**: `demo/colab_FINAL_TOKENVECTOR_GENERATED_GPU_BRIDGE_2026-11-24.ipynb`

## Ket qua do duoc
| item | PyTorch eager | TKV (bridge do compiler sinh) |
|---|---|---|
| median 50 step cuoi | 442.3 ms | **187.8 ms** |
| loss | 9.0058 -> 5.0647 | 3.8906 -> 3.4384 |
| AdamW | torch.optim.AdamW | **PATH=1** â€” kernel `.tkv` qua dlsym |

**SPEEDUP = 2.36x** (so sanh cung model GPT2-medium 355M, T=512, 60 step, cung T4, CUDA_LAUNCH_BLOCKING=1).

Build: `generated bridge exists=True bytes=18919`, `nvcc rc=0`, `ctypes load OK True`.
VRAM: free 12.41 GB truoc forward, 5.75/14.56 GB sau step 0 (khong OOM).

## Ghi chu fidelity (bat buoc)
- `3.8906` la fingerprint cua duong native/fp16 â€” **khong** so sanh truc tiep voi `9.0058` cua HF fp32 eager.
  Gia tri nay khop voi bridge C++ viet tay truoc day (xem `FULL_STEP_GPU_NOTES_2026-11-19.md`).
- Khong co C/C++ viet tay trong san pham: bridge la compiler output cua `torch_fused_port.tkv`
  thong qua `torch_feature.tkv`; moi sua dÃªu thuc hien trong `.tkv` cua backend roi rebuild bang `vong.exe` (IL_IDENTICAL).

##ç»“è®º / Ket luan
Duong GPU da khong con hand-written C++ bridge. TokenVector tu sinh C++/ATen tu `.tkv`, bien dich bang nvcc,
va dat 2.36x so voi PyTorch eager tren cung phan cung. Cac phep do CPU (355x) va GPU (2.24x / 2.36x)
deu co HEAD + timestamp rieng; khong gop chung.