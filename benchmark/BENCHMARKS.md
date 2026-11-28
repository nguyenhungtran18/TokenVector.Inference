# Benchmark cuoi: train GPT2-medium, TokenVector thuan vs PyTorch

**Timestamp:** 2026-11-28. Kaggle GPU Tesla T4, torch 2.11.0+cu130, CUDA 13.0.
Repo `TokenVector.Inference@main`, HEAD co verify cuoi: `2f101ac`.

Cau hinh: `NL=24, T=512, H=1024, heads=16, head_dim=64, V=50257`, AdamW
`betas=(0.9,0.999), eps=1e-8, wd=0`. Chi so la **median ms/step**
(PyTorch: 10 step sau 2 step warm-up, co CUDA sync; TokenVector: step 4-10).

## Bang chinh (trong so that gpt2-medium pretrained)

| Implementation | Weights | median ms/step | Speedup | Train loss (dau -> cuoi) |
| :--- | :--- | ---: | ---: | :--- |
| PyTorch eager | gpt2-medium pretrained | 443.5 - 457.3 | 1.00x | 6.0951 -> 3.8234 |
| TokenVector pure .tkv | gpt2-medium pretrained | 216.6 - 219.1 | ~2.05x | 21.2213 -> 11.2158 |

Khong co row C++ bridge trong bang so sanh (theo yeu cau).
So lich su: C++ bridge `187.8 ms/step` (random-init, 2026-11-26) - chi de tham chieu,
khong duoc tinh la ket qua cua lan nay.

## Canh bao quan trong (chua xong)

HF tren cung 512 token cho loss khoi dau `9.0423`. TokenVector cho `21.2213`
(roi giam ve `11.2158`), nghia la **forward TKV voi trong so that chua khop HF**
va khong the xem la ket qua training tuong duong.

Da kiem tra va sua trong qua trinh debug:
- embedding that `wte[xb]+wpe` (truoc do la nhieu ngau nhien `wrand`);
- nhan that `yb` tu `gpt2m_ids.npz`, ghep vao cuoi blob (truoc do la `(ii+1)%V`);
- offset bias: `wo` thieu `+ hb` sau `ALB2`, lam `ABQ` doc `ln_2.bias` thay vi
  `c_attn.bias` (da doi chieu chinh xac voi `CAND ln2b`);
- IL local <= 255, loss_init i64 overflow, PTX bias mul.wide.

Con lai: stage-by-stage dump cho thay embedding va LN1 khop HF chinh xac, nhung
Q truoc bias khac HF; da thu nghiem rieng `fpA`, `fpQKV`, `gemm16` 4x4, `gemm16b`,
`gemm32` va cac compute type 0/66/74 - A/B/GEMM tuong doi nho deu dung, nen
nguyen nhan sai so lon van chua xac dinh. Khong khang dinh loss 9.04 cho TKV.

## Tai hien

Cell 2: load weights (`benchmark/export_gpt2m_weights.py`, blob 1,417,197,568 bytes).
Cell 3: `benchmark/train_real.py` (PyTorch roi TokenVector).
Cell 4: `benchmark/test_trained.py` (held-out + generation).

Lu y: `/kaggle/working` bi xoa giua cac session, export phay lai ~10 phut.
Dung Run menu -> Stop session sau khi chay de tiet kiem quota.
