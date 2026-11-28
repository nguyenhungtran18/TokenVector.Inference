# Benchmark / điều tra: TokenVector vs PyTorch (GPT2-medium, Kaggle T4)

HEAD xác minh cuối `285ed5a` — 2026-11-28. Mỗi bên 12 update, cùng một batch
512 token tinyshakespeare, cùng nhiệm vụ `vị trí t -> token t+1`.
Không có hàng C++ bridge trong bảng.

## Kết quả mới nhất (đã xác minh)

| implementation | weights | median ms/step | speedup | loss trước -> sau 12 update | step tăng |
| :--- | :--- | ---: | ---: | :--- | ---: |
| PyTorch eager, AdamW lr 5e-5, all params | gpt2-medium pretrained | 465.2 | 1.00x | 3.6676 -> **0.1457** | - |
| TokenVector .tkv, AdamW lr 2e-4 constant | gpt2-medium pretrained | 222.4 | **2.09x** | 3.6676 -> 0.1693 | **0** |

Chuỗi loss TKV:
`3.6676 -> 3.4786 -> 2.9517 -> 2.6590 -> 2.3124 -> 1.8646 -> 1.5026 ->
1.0932 -> 0.7609 -> 0.5588 -> 0.3653 -> 0.2512 -> 0.1693` — mọi drop > 0.

**Chưa đạt yêu cầu "loss TKV <= PyTorch"**: 0.1693 vs 0.1457 (chênh 0.0236
nat). TKV thắng tốc độ 2.09x và giảm đơn điệu, nhưng vẫn thua loss. Không
được phép công bố ngược lại. `187.8 ms/step` của C++ bridge chỉ là baseline
lịch sử, không xuất hiện trong bảng so sánh.

## Các bước đã xác minh

1. **Exporter (root cause lớn nhất)**: `qkv[:,0:H].T` và `attn.c_proj.weight.T`
   ghi ra `W^T`, trong khi cublas GEMM cần `B[i,o]=W[i,o]`. Toàn bộ khối
   attention dùng sai trọng số pretrained. Đã bỏ `.T`. Kiểm tra trực tiếp từ
   blob: q/k/v/c_proj/c_fc/mlp_proj/head `maxerr=0.000e+00` (7/7 OK).
2. **Forward khớp chính xác**: torch step0 `3.6676`, TKV step1 `3.6676`.
   Trước khi sửa exporter là `9.0423` vs `8.7249`.
3. **Backward**: sửa dW `attn.c_proj` ghi vào buffer dùng chung `gWo` thay vì
   `AGo[ll]`; bias AdamW cập ngay trong vòng lặp để mỗi tầng dùng gradient của
   chính nó; buffer riêng cho `ln_f`.
4. **Gradient subnormal**: `dL/dlogits ~ 1e-6` nằm sâu vùng subnormal fp16;
   nhân 1024× trong `loss_dx`. AdamW scale-invariant nên không đổi bước cập
   nhật. Loss cải thiện 2.70 -> 0.1693 và không còn step nào tăng.

## Chẩn đoán gradient còn lại (step 1)

| đại lượng | corr |
| :--- | ---: |
| dL/dlogits @ Wh (đầu vào ln_f) | +1.00000 |
| dL/d(hidden cuối) sau ln_bwd | +0.99999 |
| gQ tầng 23 | +0.99999 |
| gDX / gY1 tầng 23 | +0.99999 / +0.99993 |
| gQ tầng 16 / 8 / 0 | +0.733 / +0.420 / +0.223 |
| dWq tầng 0 (256 phần tử) | +0.195 |

Lỗi nằm ở sai số tích lũy qua các block (block 23 đúng, suy dần về layer 0),
nhiều khả năng do cast fp16 ở các GEMM backward. Đây là lý do loss chưa san
bằng PyTorch hoàn toàn.

## Test nội dung đã train (PyTorch)

```
BASELINE pretrained held-out CE = 4.0695
CE loss train batch   = 0.0466
CE loss held-out      = 6.7884   (CAO HON baseline 2.72 nat)
PROMPT 'ROMEO: ' -> vẫn là Shakespeare nhưng lặp "I know not"
```

Loss dưới 1 trên cùng batch 512 token lặp lại = **memorization**, không phải
chất lượng ngôn ngữ thật. Held-out cao hơn pretrained -> model đã overfit;
không được xây dựng claim "AI học tiếng Anh".

## Ghi chú

- `c_fwrite` không có trong runtime; không thể lưu weight của TKV ra file.
  Bước 3 hiện test model PyTorch (`trained_gpt2m.pt`).
- Không có GPU/python local; mọi số liệu đều từ Kaggle T4, Stop session trước
  khi đọc output theo yêu cầu quota.
- Các lần chạy cũ (loss 8.72 -> 6.37, 1.45, 0.3427) dùng exporter/gradient sai
  và không còn đại diện cho kết quả hiện tại.
