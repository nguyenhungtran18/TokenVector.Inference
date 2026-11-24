# IL-only GPU â€” trang thai kernel (cap nhat 2026-11-24 05:54 UTC)

**HEAD**: `TokenVector.CudaBackend@f1bfb71` Â· `TokenVector.Inference@431b637`
**Duong chay**: `.tkv -> IL -> ilasm -> PE -> mono 6.8 -> P/Invoke libcudart/libcublas/libcuda`.
Khong `.cpp`, khong `.cu`, khong `import torch`. PTX duoc embed dang `b"..."` trong chinh `.tkv`,
chep vao native buffer bang `write_u8`, nap bang `cuModuleLoadData`, launch bang `cuLaunchKernel`.

## Kernel da xac nhan (MAXERR so voi reference)
| kernel | nguon | MAXERR | notebook |
|---|---|---|---|
| SAXPY | ilgpu_saxpy.tkv | 0.0 | colab_IL_ONLY_GPU_M1_M2_V4 |
| SGEMM 4x4 | ilgpu_sgemm.tkv | 0.0 | colab_IL_ONLY_GPU_M1_M2_V4 |
| SCALE | ilgpu_ptx_scale.tkv | 0.0 | colab_IL_ONLY_GPU_M3_PTX_V3 |
| GELU (tanh approx) | ilgpu_kernels_batch.tkv | 1.8e-6 | colab_IL_ONLY_GPU_KERNELS_BATCH_V2 |
| ADD | ilgpu_kernels_batch.tkv | 0.0 | ( cung notebook ) |
| INDEX_SELECT | ilgpu_kernels_batch.tkv | 0.0 | ( cung notebook ) |
| LayerNorm | ilgpu_norm_softmax.tkv | 6.8e-8 | colab_IL_ONLY_GPU_LN_SM_V2 |
| Softmax | ilgpu_norm_softmax.tkv | 5.8e-8 | colab_IL_ONLY_GPU_LN_SM_V2 |
| AdamW | ilgpu_adamw.tkv | 4.1e-7 | colab_IL_ONLY_GPU_ADAMW |

## Bai hoc ky thuat da fix
- `cublasSaxpy_v2`: alpha la `const float*` (khong phai float value).
- `float()` sinh `conv.r8` (f64) lam sai stack khi lam voi literal f32 -> bo float(), dung bien f32.
- `cuModuleLoadData` can primary context -> goi `cudaMalloc` truoc.
- `cuLaunchKernel` kernelParams la mang CON TRO toi slot arg host (khong phai gia tri arg).
- `sgemm` tinh tong voi vector `ones[1,h]`: lda phai = 1 (khong phai h).
- PTX: `mul.wide.u32` chi nhan `.u32`; offset 64-bit phai dung `mul.wide.u32` roi `add.u64`.

## Con lai (CHUA lam, khong claim xong)
- SDPA/causal attention (co the dung sgemm + softmax PTX + mask PTX).
- gather/log-softmax loss tren device.
- backward: tape/grad tich luy, put grads, fp16 master sync.
- binding Python <-> .NET cho full training loop va do lai toc do.
- So sanh fingerprint 3.8906 cua duong bridge cu.

Bridge C++ sinh (`2.36x` vs torch, HEAD 92a2349 / Inference 5f9d8ac) van la baseline,
KHONG xoa cho den khi chay duoc mot buoc train IL-only hoan chinh.