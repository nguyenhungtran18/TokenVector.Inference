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

## Giả thuyết cho sai số tăng dần (đang kiểm chứng)

Hai nguồn khả dĩ, tách được bằng phép đo mới (notebook V4):

1. **Forward**: `tanh.approx.f32` trong GELU (sai số ~1e-6 trên mỗi phần tử H) — sẽ làm
   `loss_double(tham số IL)` lệch khỏi `LOSS` in ra.
2. **Đường cập nhật**: AdamW f32 với `eps = 1e-8`; các phần tử gradient rất nhỏ
   (`|g| ~ 1e-8`) rơi vào vùng `sqrt(v/bc2) ≈ eps`, nơi làm tròn f32 đổi hướng cập nhật
   ⇒ tham số lệch dần giữa f32 và f64.

Notebook V4 in **toàn bộ tham số sau từng step** + thống kê `|g|` (min/max/số phần tử < 1e-7),
rồi so sánh từng step và tính lại forward bằng double **từ chính tham số IL** — cho biết
sai số nằm ở forward hay ở update.

## Ý nghĩa

Đây là lần đầu một bước train **hoàn toàn bằng IL/PTX tự sinh, không C++, không torch** chạy đúng
trên GPU. Không có kernel C/C++ nào được sinh ra trong quá trình build.

## Tiếp theo

1. Chốt chẩn đoán sai số bằng V4, ghi lại kết luận.
2. Đưa softmax + log-softmax/gather-loss lên device (`prep`/`loss` kernel đã verify) để bỏ
   D2H/H2D mỗi step.
3. Vòng lặp đo ms/step (median) trên toy, rồi scale GPT2-medium (T=512, 355M).
4. So với torch eager và baseline bridge `2.36×` (torch 442.3 ms vs TKV 187.8 ms, loss fingerprint `3.8906`).
