# IL-only FULL GPU train step — verified on T4 (2026-11-25)

## Mốc

| Mục | Giá trị |
|---|---|
| `TokenVector.CudaBackend` HEAD | `d0a824d` (code trong lần chạy), `1d22f64` (reference script) |
| `TokenVector.Inference` HEAD | `1eb4a2b` |
| Notebook | `demo/KAGGLE_IL_ONLY_FULL_GPU_STEP_V3_2026-11-25.ipynb` |
| Kernel | `Release/cuda_backend/examples/ilgpu_step_full_gpu.tkv` → `.exe` (66 560 B) |
| Môi trường | Kaggle GPU T4, .NET **10.0.404**, CUDA 13.0 (`libcudart.so`, `libcublas.so`), driver `/usr/local/nvidia/lib64/libcuda.so` |
| Không dùng | C/C++, `.cu`, `.torch.cpp`, `torch`, `mono` |

## Kết quả

Toàn bộ **forward + backward + AdamW** chạy trên GPU; chỉ còn softmax/loss trên host.

```
LOSS 2.1086576      ref 2.10865766      err 5.7e-8
LOSS 2.0675688      ref 2.06756899      err 1.9e-7
LOSS 2.0311878      ref 2.03119492      err 7.1e-6
LOSS 1.9962366      ref 1.99628594      err 4.9e-5
LOSS 1.9606265      ref 1.96073251      err 1.06e-4
W1/W2 step1 maxerr  1.97e-8 / 4.78e-8
KET LUAN: OK
```

- Loss giảm đơn điệu qua 5 step, đúng chiều.
- Sau **step 1**, tham số khớp reference double tới `2e-8` ⇒ forward, backward và AdamW đúng.
- Sai số loss **tăng dần** tới `1.06e-4` ở step 5. Đây là điểm cần giải thích, không phải lỗi chặn.

## Chẩn đoán sai số tăng dần — đã khoanh vùng (notebook V4, HEAD `0196551`)

| step | dW1max | dW2max | db1max | db2max | lossIL − lossDouble(tham số IL) |
|---|---|---|---|---|---|
| 1 | 2.90e-8 | 4.78e-8 | 3.62e-10 | 3.81e-09 | 5.67e-8 |
| 2 | 6.37e-5 | 7.02e-8 | 3.43e-05 | 3.78e-08 | 2.02e-7 |
| 3 | 1.70e-4 | 1.33e-7 | 1.42e-04 | 1.27e-07 | 1.09e-7 |
| 4 | 2.29e-4 | 9.26e-7 | 2.19e-04 | 2.60e-07 | 9.17e-8 |
| 5 | 2.64e-4 | 2.31e-6 | 2.63e-04 | 5.15e-07 | 1.97e-8 |

Kết luận từ bảng này:

1. **Forward + softmax/loss chính xác.** Cột cuối là loss tính lại bằng double **từ chính tham số IL
   in ra**; khớp `LOSS` in ra ≤ `2e-7` ở **mọi** step ⇒ `tanh.approx.f32` trong GELU **không phải**
   nguyên nhân (loại giả thuyết 1).
2. **Không phải vùng `eps`.** `tiny = 0` ở mọi step (không phần tử nào `|g| < 1e-7`) ⇒ loại giả thuyết 2.
3. **Sai số chỉ nằm ở W1/b1**, tức nhánh `dZ = dH ⊙ dgelu(Z)`; W2/b2 (nhánh `dL`) sạch tới `1e-7`.
   Drift bắt đầu từ **step 2** (sau update đầu tiên) và bão hoà dần.

Nghi vấn còn lại: sai số **gradient** của nhánh `dZ` là bao nhiêu, và AdamW khuếch đại nó lên mức nào.
Đo mới (HEAD `ec63d03`): exe in thêm `DW1/DW2/DB1/DB2` từng step; reference tính
`predW1 = |Adam(g_IL) − Adam(g_ref)|` — nếu `obsW1 ≈ predW1` thì drift được giải thích hoàn toàn
bởi làm tròn f32 bị Adam khuếch đại, không phải lỗi kernel.

## Ý nghĩa

Đây là lần đầu một bước train **hoàn toàn bằng IL/PTX tự sinh, không C++, không torch** chạy đúng
trên GPU. Không có kernel C/C++ nào được sinh ra trong quá trình build.

## Tiếp theo

1. Chốt chẩn đoán sai số bằng V4, ghi lại kết luận.
2. Đưa softmax + log-softmax/gather-loss lên device (`prep`/`loss` kernel đã verify) để bỏ
   D2H/H2D mỗi step.
3. Vòng lặp đo ms/step (median) trên toy, rồi scale GPT2-medium (T=512, 355M).
4. So với torch eager và baseline bridge `2.36×` (torch 442.3 ms vs TKV 187.8 ms, loss fingerprint `3.8906`).
