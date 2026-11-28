# Benchmark cuoi: huan luyen GPT2-medium, TokenVector thuan vs PyTorch

**Thoi gian:** 2026-11-28. Kaggle GPU Tesla T4, torch 2.11.0+cu130, CUDA 13.0.
Repo `TokenVector.Inference@main`, HEAD verify cuoi `2f101ac`.

Cau hinh: `NL=24, T=512, H=1024, heads=16, head_dim=64, V=50257`, AdamW
`betas=(0.9,0.999), eps=1e-8, wd=0`. So lieu la **median ms/step**
(PyTorch: 10 buoc sau 2 buoc warm-up, co CUDA sync; TokenVector: cac buoc 4-10).

## Bang chinh (trong so that gpt2-medium pretrained)

| Cai dat | Weights | median ms/step | Speedup | Train loss (dau -> cuoi) |
| :--- | :--- | ---: | ---: | :--- |
| PyTorch eager | gpt2-medium pretrained | 443.5 - 457.3 | 1.00x | 6.0951 -> 3.8234 |
| TokenVector pure .tkv | gpt2-medium pretrained | 216.6 - 219.1 | ~2.05x | 21.2213 -> 11.2158 |

Khong co row C++ bridge trong bang so sanh (theo yeu cau). So lich su:
C++ bridge `187.8 ms/step` (random-init, 2026-11-26) - chi de tham chieu,
khong tinh la ket qua lan nay.

## Canh bao quan trong (chua xong)

HF tren cung 512 token cho loss khoi dau `9.0423`. TokenVector cho `21.2213`
(roi giam ve `11.2158`). Forward TKV voi trong so that **chua khop HF**,
khong the xem la ket qua training tuong duong.

Da sua trong debug: embedding that, nhan that, offset bias (`ABQ` doc
`ln_2.bias` thay vi `c_attn.bias`), IL local <= 255, loss_init i64 overflow,
PTX bias mul.wide. Stage dump cho thay embedding va LN1 khop HF, nhung Q truoc
bias khac HF. Da thu rieng fpA/fpQKV/gemm16 4x4/gemm16b/gemm32 va compute
0/66/74 - nguyen nhan sai so lon van chua xac dinh. Khong khang dinh loss 9.04.

## Tai hien

Cell 2 load weights, cell 3 train (PyTorch roi TokenVector), cell 4 test.
`/kaggle/working` bi xoa giua cac session nen export chay lai ~10 phut.
Dung Run menu -> Stop session sau khi chay.
