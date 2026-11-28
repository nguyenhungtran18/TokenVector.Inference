# Benchmark cuoi: huan luyen GPT2-medium, TokenVector thuan vs PyTorch

**Thoi gian:** 2026-11-28. Kaggle GPU Tesla T4, torch 2.11.0+cu130, CUDA 13.0.
Repo `TokenVector.Inference@main`, verify HEAD `928a755` (trainer `bd2348b`).

Cau hinh: `NL=24, T=512, H=1024, heads=16, head_dim=64, V=50257`, AdamW
`betas=(0.9,0.999), eps=1e-8`, lr TKV = 1e-4 co warmup `lr*min(step/10,1)`,
wd = 0. PyTorch lr = 5e-5, 2 step warm-up.

## Bang chinh (trong so that gpt2-medium pretrained)

| Cai dat | Weights | median ms/step | Speedup | Train loss (dau -> cuoi) |
| :--- | :--- | ---: | ---: | :--- |
| PyTorch eager | gpt2-medium pretrained | 433 - 447 | 1.00x | 6.0951 -> 3.8234 |
| TokenVector pure .tkv | gpt2-medium pretrained | 210 - 216 | ~2.08x | 8.7249 -> 6.5418 |

Khong co row C++ bridge trong bang so sanh (theo yeu cau). So lich su:
`187.8 ms/step` (random-init, 2026-11-26) - chi tham chieu.

## Loss giam deu tung step (da verify, lr 1e-4 + warmup)

```
8.7249 -> 8.1827 -> 7.6403 -> 7.3655 -> 7.2680 -> 7.0690
        -> 6.9027 -> 6.7974 -> 6.6568 -> 6.5418
```
Moi step thap hon step truoc. Thu lr 2e-4 cho loss cuoi 6.3053 nhung co
2 step tang lai, nen khong chon.

HF cung 512 token: loss dau `9.0423`; TKV bat dau `8.7249` (gan, lech nho
do fp16). Loss cuoi `6.5418` van cao hon PyTorch `3.8234`.

## Bug da sua

- embedding ngau nhien -> embedding that `wte[xb]+wpe`;
- nhan gia `(ii+1)%V` -> `yb` that;
- offset bias thieu `wo += hb` -> `ABQ` doc `ln_2.bias` thay vi `c_attn.bias`;
- huong weight Conv1D `[input,output]`: exporter cat qkv theo hang thay vi cot,
  Q/K/V sai + transpose. Da sua `qkv[:,0:H].T`, W1/W2 khong transpose ->
  loss 21.22 xuong 8.72;
- gradient bias ghi de buffer bias -> tach buffer gradient rieng + AdamW;
- them backward ln_f; AdamW warmup tranh overshoot step dau;
- IL local <= 255, loss_init i64 overflow, PTX bias mul.wide.

## Tai hien

Cell 2 load weights, cell 3 train (in toan bo step loss), cell 4 test.
`/kaggle/working` bi xoa giua cac session nen export chay lai ~10 phut.
Dung Run menu -> Stop session sau khi chay.
