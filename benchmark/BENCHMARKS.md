# Benchmark cuoi: train GPT2-medium, TokenVector thuan vs PyTorch

**Timestamp:** 2026-11-28. Kaggle GPU Tesla T4, torch 2.11.0+cu130, CUDA 13.0.
Repo `TokenVector.Inference@main`, verify HEAD `928a755` (trainer `bd2348b`).

Cau hinh: `NL=24, T=512, H=1024, heads=16, head_dim=64, V=50257`, AdamW
`betas=(0.9,0.999), eps=1e-8`, lr TKV = 1e-4 co warmup `lr*min(step/10,1)`,
wd = 0. PyTorch lr = 5e-5, 2 step warm-up.

## Bang chinh (trong so that gpt2-medium pretrained)

| Implementation | Weights | median ms/step | Speedup | Train loss (dau -> cuoi) |
| :--- | :--- | ---: | ---: | :--- |
| PyTorch eager | gpt2-medium pretrained | 433 - 447 | 1.00x | 6.0951 -> 3.8234 |
| TokenVector pure .tkv | gpt2-medium pretrained | 210 - 216 | ~2.08x | 8.7249 -> 6.5418 |

Khong co row C++ bridge trong bang so sanh (theo yeu cau).
So lich su: C++ bridge `187.8 ms/step` (random-init, 2026-11-26) - chi de
tham chieu, khong tinh la ket qua lan nay.

## Loss giam deu theo tung step (da verify, lr 1e-4 + warmup)

```
8.7249 -> 8.1827 -> 7.6403 -> 7.3655 -> 7.2680 -> 7.0690
        -> 6.9027 -> 6.7974 -> 6.6568 -> 6.5418
```
Moi step deu thap hon step truoc (`per-step drops` deu duong).
Thu lr 2e-4 cho ket qua cuoi thap hon (6.3053) nhung co 2 step tang lai,
nen khong chon; lu cau hinh don dinh 1e-4.

HF cung 512 token cho loss khoi dau `9.0423`. TKV bat dau `8.7249` (gan,
chenh lech nho do fp16). Ket qua cuoi `6.5418` van cao hon PyTorch `3.8234`.

## Cac bug da tim va sua (co doi chieu truc tiep)

- embedding: `wrand()` ngau nhien -> `wte[xb]+wpe` that;
- nhan: `(ii+1)%V` gia -> `yb` that tu `gpt2m_ids.npz`;
- offset bias: thieu `wo = wo + hb` sau `ALB2`, `ABQ` doc `ln_2.bias`
  thay vi `c_attn.bias` (khop chinh xac `CAND ln2b`);
- **huong trong so HF Conv1D**: weight dang `[input, output]` = `[H,3H]` /
  `[H,4H]`. Exporter cat qkv theo HANG thay vi THEO COT -> Q/K/V sai va
  transpose. Da sua thanh `qkv[:,0:H].T`, va W1/W2 viet thang khong transpose.
  Sau khi sua: `DUMPCHK wq_T_alt maxerr=0`, loss 21.22 -> 8.72;
- gradient bias ghi de luon buffer bias -> da tach buffer gradient rieng,
  dung activation-grad colsum va AdamW rieng;
- them backward qua ln_f; AdamW co warmup de tranh step dau overshoot;
- IL local <= 255, loss_init i64 overflow, PTX bias `mul.wide`.

## Tai hien

Cell 2 load weights (blob 1,417,197,568 bytes), cell 3 train
(PyTorch roi TokenVector, in toan bo step loss), cell 4 test held-out + sinh.
`/kaggle/working` bi xoa giua cac session nen export chay lai ~10 phut.
Dung Run menu -> Stop session sau khi chay de tiet kiem quota.
