# IL-only transformer: device kernels + block forward (2026-11-26)

## Các mốc đã xác thực trên T4

| Mốc | HEAD | Notebook | Kết quả |
|---|---|---|---|
| Full GPU train step (toy, host loss) | `b831002` | V4 / notebook 24 | `KET LUAN: OK`; loss maxerr `4.6309e-7`; dG1max `3.19e-9` |
| Full GPU train step + device loss kernel | `93e8bc7` | V4 / notebook 26 | `KET LUAN: OK`; **không D2H `T×V`, không H2D `dL` mỗi step**; chỉ D2H `T` float |
| Timing harness (`time_perf` + `cudaDeviceSynchronize`) | `4ce6343` | — | `MS_TOTAL`/`MS_STEP` in cuối step loop (chưa có số đo scale lớn) |
| 4 kernel device-only cho transformer | `62bffac` | `KAGGLE_ILGPT2_KERNELS` / notebook 27 | `LY 2.29e-6`, `LDX 1.69e-6`, `SP 8.84e-8`, `DS 8.47e-10` |
| Block transformer forward (T=4,H=8) | `f117852` (baseline khôi phục từ `03b23b1`) | `KAGGLE_ILGPT2_BLOCK_FWD` / notebook 39 | `Y1 1.15e-6`, `P 1.20e-7`, `X2 5.53e-7`, `Y2 3.60e-5`, `HH 3.02e-6`, `O2 2.34e-5`, `OUT 2.32e-5`, `KET LUAN: OK` |

## Bug đã tìm và sửa trong block forward

1. **Tràn buffer tên kernel**: `nb = len(FN1)` = 11 B nhưng `"gelu_kernel\0"` cần 12 B → segfault trước khi in gì. Sửa `nb = 32`.
2. **Residual add lần 2**: chỉ đổi `c1=dO2`, `c2=dOUT`, quên đổi `c0=dX → dX2` → `OUT = X + O2`, sai `1.375`. Sửa xong `OUT 2.32e-5`.

## Vấn đề còn mở

`lnf_kernel` (LayerNorm forward device) chạy OK ở lần gọi thứ nhất (LN1). Lần gọi thứ hai (LN2) gây `SIGSEGV` trong `cuLaunchKernel`, dù:

- 6 pointer device đều hợp lệ (đã in ra);
- dùng `kernelParams` buffer riêng (`pl2`) và **entry PTX riêng** `ln2f_kernel` / `CUfunction` riêng vẫn crash;
- không phải lỗi JIT (không có `MODULE FAIL`/`FN ... FAIL`).

Hiện tại LN2 tính trên host (D2H `T×H`, tính mean/rstd/xhat/y, H2D lại) để block chạy được. Phải thay bằng thiết kế device khác trước khi đo tốc độ thật — ví dụ tách reduction mean/rstd thành kernel riêng rồi dùng `norm_kernel`, hoặc dùng shared-memory reduction với block per-row.

## Quyết định kiến trúc

- Độ chính xác: **làm cả hai**. fp32 ở scale trung gian để khoá thuật toán (`MAXERR ~1e-6`), sau đó thêm path fp16 tensor-core (`cublasGemmEx(CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`) + master weight fp32 để so tốc độ với baseline `187.8 ms`.
- Trình tự scale: **trung gian trước** (T=512, ít lớp/V nhỏ rồi mới V=50257, NL=24, 60 steps).
- Baseline cần vượt: torch eager `442.3 ms` vs bridge C++ sinh ra `187.8 ms` = `2.36×`, fingerprint `3.8906` (fp16, không so trực tiếp với HF fp32 `9.0058`).

## Block backward — verified (notebook 54, HEAD `aa13acd`)

Script `Release/cuda_backend/examples/ilgpt2_bwd_clean.tkv` (exe 20 480 B) — tách khỏi forward,
mọi tensor đầu vào sinh deterministic trên host rồi H2D, 16 gradient so với reference double:

```
GW2 2.0e-8   GW1 1.7e-7   GY2 1.7e-7   GX2 6.9e-7
GA  1.7e-6   GWO 2.6e-6   GP  5.5e-6   GV 1.3e-6
GS  1.0e-6   GQ  3.7e-7   GK  7.9e-7
GWQ 4.4e-7   GWK 1.5e-6   GWV 1.1e-5
GY1 3.2e-6   GDX 5.2e-6
KET LUAN: OK
```

Các bug phải sửa (đều là dạng **ghi đè weight/buffer đầu vào bằng gradient** hoặc sai kích thước):

1. `dZ`/`dHH` device buffer cấp `qb` (128 B) thay vì `fb` (512 B).
2. `dHHg` tách riêng; `dZ` dùng `dHHg` (gradient), không dùng `dHH` (input thô).
3. `dW1g`/`dW2g` tách riêng; `dY2` dùng `dW1` gốc, `dHHg` dùng `dW2` gốc.
4. `dV` GEMM đúng `m=T, n=H, ldc=T` (không phải `H,T`).
5. `dK` GEMM đúng `m=T, n=H`.

Bản QKV-backward tối thiểu (`ilgpt2_qkv_bwd.tkv`, notebook 49) cũng sạch:
`GWQ 6e-8, GWK 1e-7, GWV 2e-7, GY1 7e-8`.

**Kết luận**: forward và backward của một block transformer đều đã verify ở mức f32,
dù đang tạm tính LN/dgelu/softmax-backward trên host (cần device hóa để đo tốc độ).

## Tiếp theo

1. Ghép forward + backward thành vòng train 2 lớp ở T=512 (weight deterministic), kiểm tra loss giảm.
2. Thay LN1/LN2 host bằng kernel device thiết kế lại (lần gọi `lnf_kernel` thứ 2 từng segfault).
3. Đo median ms/step; sau đó V=50257, NL=24; thêm path fp16 tensor-core để so `187.8 ms`.
