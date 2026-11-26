# IL-only transformer: device kernels + block forward (2026-11-26)

## CÃ¡c má»‘c Ä‘Ã£ xÃ¡c thá»±c trÃªn T4

| Má»‘c | HEAD | Notebook | Káº¿t quáº£ |
|---|---|---|---|
| Full GPU train step (toy, host loss) | `b831002` | V4 / notebook 24 | `KET LUAN: OK`; loss maxerr `4.6309e-7`; dG1max `3.19e-9` |
| Full GPU train step + device loss kernel | `93e8bc7` | V4 / notebook 26 | `KET LUAN: OK`; **khÃ´ng D2H `TÃ—V`, khÃ´ng H2D `dL` má»—i step**; chá»‰ D2H `T` float |
| Timing harness (`time_perf` + `cudaDeviceSynchronize`) | `4ce6343` | â€” | `MS_TOTAL`/`MS_STEP` in cuá»‘i step loop (chÆ°a cÃ³ sá»‘ Ä‘o scale lá»›n) |
| 4 kernel device-only cho transformer | `62bffac` | `KAGGLE_ILGPT2_KERNELS` / notebook 27 | `LY 2.29e-6`, `LDX 1.69e-6`, `SP 8.84e-8`, `DS 8.47e-10` |
| Block transformer forward (T=4,H=8) | `f117852` (baseline khÃ´i phá»¥c tá»« `03b23b1`) | `KAGGLE_ILGPT2_BLOCK_FWD` / notebook 39 | `Y1 1.15e-6`, `P 1.20e-7`, `X2 5.53e-7`, `Y2 3.60e-5`, `HH 3.02e-6`, `O2 2.34e-5`, `OUT 2.32e-5`, `KET LUAN: OK` |

## Bug Ä‘Ã£ tÃ¬m vÃ  sá»­a trong block forward

1. **TrÃ n buffer tÃªn kernel**: `nb = len(FN1)` = 11 B nhÆ°ng `"gelu_kernel\0"` cáº§n 12 B â†’ segfault trÆ°á»›c khi in gÃ¬. Sá»­a `nb = 32`.
2. **Residual add láº§n 2**: chá»‰ Ä‘á»•i `c1=dO2`, `c2=dOUT`, quÃªn Ä‘á»•i `c0=dX â†’ dX2` â†’ `OUT = X + O2`, sai `1.375`. Sá»­a xong `OUT 2.32e-5`.

## Váº¥n Ä‘á» cÃ²n má»Ÿ

`lnf_kernel` (LayerNorm forward device) cháº¡y OK á»Ÿ láº§n gá»i thá»© nháº¥t (LN1). Láº§n gá»i thá»© hai (LN2) gÃ¢y `SIGSEGV` trong `cuLaunchKernel`, dÃ¹:

- 6 pointer device Ä‘á»u há»£p lá»‡ (Ä‘Ã£ in ra);
- dÃ¹ng `kernelParams` buffer riÃªng (`pl2`) vÃ  **entry PTX riÃªng** `ln2f_kernel` / `CUfunction` riÃªng váº«n crash;
- khÃ´ng pháº£i lá»—i JIT (khÃ´ng cÃ³ `MODULE FAIL`/`FN ... FAIL`).

Hiá»‡n táº¡i LN2 tÃ­nh trÃªn host (D2H `TÃ—H`, tÃ­nh mean/rstd/xhat/y, H2D láº¡i) Ä‘á»ƒ block cháº¡y Ä‘Æ°á»£c. Pháº£i thay báº±ng thiáº¿t káº¿ device khÃ¡c trÆ°á»›c khi Ä‘o tá»‘c Ä‘á»™ tháº­t â€” vÃ­ dá»¥ tÃ¡ch reduction mean/rstd thÃ nh kernel riÃªng rá»“i dÃ¹ng `norm_kernel`, hoáº·c dÃ¹ng shared-memory reduction vá»›i block per-row.

## Quyáº¿t Ä‘á»‹nh kiáº¿n trÃºc

- Äá»™ chÃ­nh xÃ¡c: **lÃ m cáº£ hai**. fp32 á»Ÿ scale trung gian Ä‘á»ƒ khoÃ¡ thuáº­t toÃ¡n (`MAXERR ~1e-6`), sau Ä‘Ã³ thÃªm path fp16 tensor-core (`cublasGemmEx(CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`) + master weight fp32 Ä‘á»ƒ so tá»‘c Ä‘á»™ vá»›i baseline `187.8 ms`.
- TrÃ¬nh tá»± scale: **trung gian trÆ°á»›c** (T=512, Ã­t lá»›p/V nhá» rá»“i má»›i V=50257, NL=24, 60 steps).
- Baseline cáº§n vÆ°á»£t: torch eager `442.3 ms` vs bridge C++ sinh ra `187.8 ms` = `2.36Ã—`, fingerprint `3.8906` (fp16, khÃ´ng so trá»±c tiáº¿p vá»›i HF fp32 `9.0058`).

## Block backward â€” verified (notebook 54, HEAD `aa13acd`)

Script `Release/cuda_backend/examples/ilgpt2_bwd_clean.tkv` (exe 20 480 B) â€” tÃ¡ch khá»i forward,
má»i tensor Ä‘áº§u vÃ o sinh deterministic trÃªn host rá»“i H2D, 16 gradient so vá»›i reference double:

```
GW2 2.0e-8   GW1 1.7e-7   GY2 1.7e-7   GX2 6.9e-7
GA  1.7e-6   GWO 2.6e-6   GP  5.5e-6   GV 1.3e-6
GS  1.0e-6   GQ  3.7e-7   GK  7.9e-7
GWQ 4.4e-7   GWK 1.5e-6   GWV 1.1e-5
GY1 3.2e-6   GDX 5.2e-6
KET LUAN: OK
```

CÃ¡c bug pháº£i sá»­a (Ä‘á»u lÃ  dáº¡ng **ghi Ä‘Ã¨ weight/buffer Ä‘áº§u vÃ o báº±ng gradient** hoáº·c sai kÃ­ch thÆ°á»›c):

1. `dZ`/`dHH` device buffer cáº¥p `qb` (128 B) thay vÃ¬ `fb` (512 B).
2. `dHHg` tÃ¡ch riÃªng; `dZ` dÃ¹ng `dHHg` (gradient), khÃ´ng dÃ¹ng `dHH` (input thÃ´).
3. `dW1g`/`dW2g` tÃ¡ch riÃªng; `dY2` dÃ¹ng `dW1` gá»‘c, `dHHg` dÃ¹ng `dW2` gá»‘c.
4. `dV` GEMM Ä‘Ãºng `m=T, n=H, ldc=T` (khÃ´ng pháº£i `H,T`).
5. `dK` GEMM Ä‘Ãºng `m=T, n=H`.

Báº£n QKV-backward tá»‘i thiá»ƒu (`ilgpt2_qkv_bwd.tkv`, notebook 49) cÅ©ng sáº¡ch:
`GWQ 6e-8, GWK 1e-7, GWV 2e-7, GY1 7e-8`.

**Káº¿t luáº­n**: forward vÃ  backward cá»§a má»™t block transformer Ä‘á»u Ä‘Ã£ verify á»Ÿ má»©c f32,
dÃ¹ Ä‘ang táº¡m tÃ­nh LN/dgelu/softmax-backward trÃªn host (cáº§n device hÃ³a Ä‘á»ƒ Ä‘o tá»‘c Ä‘á»™).

## 1-layer train step â€” verified (notebook 61, HEAD `0804559`)

`Release/cuda_backend/examples/ilgpt2_train1.tkv` â€” forward + CE loss + backward + SGD, 3 bÆ°á»›c,
T=4 H=8 F=32 V=8, weight deterministic:

```
LOSS 2.821911 -> 2.337988 -> 1.9980869
3 x MARK UPD, khong co FAIL, KET LUAN: OK
```

Bug quyáº¿t Ä‘á»‹nh: IL cÃ³ **263 local**, `ldloc.s`/`stloc.s` dÃ¹ng chá»‰ sá»‘ 8-bit â†’ trÃ n á»Ÿ 256,
lÃ m `T/H/F/V` Ä‘á»c thÃ nh `256/256/65536/2048` (vÃ²ng LN cháº¡y 29 láº§n). Gom cÃ¡c handle táº¡m
(`w0..w25`, `f0..f20`, `u*`, `v*`, `z*`, `b*`, `c*`) thÃ nh 2 biáº¿n dÃ¹ng láº¡i â†’ max `ldloc.s = 177`.
BÃ i há»c: **pháº£i kiá»ƒm tra max local index < 256** cho má»i script .tkv lá»›n.

## Train T=512 â€” verified (notebook 64, HEAD `cadd9fa`)

CÃ¹ng script `ilgpt2_train1.tkv`, Ä‘á»•i `T=4 -> 512` vÃ  sá»­a `dS` cáº¥p `sb = T*T*4` (thay `fb`):

```
LOSS 3.4078045 -> 3.2996545 -> 3.2021413
3 x MARK UPD, khong FAIL, KET LUAN: OK
MS_STEP 79.58 ms  (baseline host-heavy: LN/softmax/dgelu/CE deu qua host)
```

`lr` pháº£i Ä‘á»•i `1e-3 -> 1e-6`: gradient trá»ng sá»‘ lÃ  **tá»•ng qua T token**, nÃªn tÄƒng ~128Ã—
khi T tÄƒng 4 â†’ 512. ÄÃ¢y lÃ  má»‘c train IL-only Ä‘áº§u tiÃªn á»Ÿ `T=512`.

## 2-layer train step â€” verified (notebook 65, exe HEAD `7e9df80`)

`ilgpt2_train2.tkv` sinh tá»± Ä‘á»™ng tá»« báº£n 1 lá»›p: forward lá»›p 1 â†’ lá»›p 2 â†’ CE â†’ backward lá»›p 2 â†’ lá»›p 1 â†’ SGD cáº£ hai.
T=512, H=8, F=32, V=8, lr=1e-6:

```
LOSS 4.4484606 -> 4.139653 -> 3.8697996
2 x MARK FWD + 2 x MARK UPD / step, khong FAIL
KET LUAN: OK (sau khi sua reference regex / so luong update)
```

Max local IL = **249** (Ä‘Ã£ chia sáº» scratch vÃ  buffer gradient giá»¯a 2 lá»›p Ä‘á»ƒ nÃ© ngÆ°á»¡ng 256).

## V=50257, 2 lá»›p â€” verified (notebook 68, HEAD `1f8b10d`)

```
LOSS 13.193880 -> 12.885666 -> 12.616205
2 x MARK FWD + 2 x MARK UPD / step, KET LUAN: OK
```

Bug khi tÄƒng V: khá»Ÿi táº¡o `Wh` dÃ¹ng `0.2 + 0.01*(i+j)` vá»›i `j` tá»›i 50257 â†’ weight tá»›i ~502,
logits explode, `exp` underflow, loss `inf`. Äá»•i thÃ nh `0.2 + 0.01*(i + (j % 8))`.

## GPT2-medium config â€” cháº¡y sáº¡ch (notebook 84, HEAD `e413c32`)

`ilgpt2_trainN.tkv` dáº¡ng layer-loop, `NL=24, T=512, H=1024, F=4096, V=50257`:

```
LOSS 10.824877 (x2 step, ~= ln 50257)
24 x MARK B (23..0) / step, MARK FWD + MARK BWD + MARK UPD, khong FAIL
KET LUAN: OK
max ldloc.s = 254 (<256)
```

CÃ¡c bug khi scale pháº£i sá»­a:
- khá»Ÿi táº¡o weight/input/gamma chá»‰ dÃ¹ng `jj % 8` â†’ `ii` váº«n tá»›i 1024 â†’ weight ~10 â†’ activation ná»•;
  dÃ¹ng háº±ng sá»‘ cho má»i init.
- `exp(2*z)` cho tanh overflow khi `z` lá»›n â†’ clamp `thv = Â±1` khi `|z| > 8`.
- sau softmax chá»‰ H2D vÃ o buffer riÃªng `pd`, quÃªn H2D vÃ o buffer dÃ¹ng chung `dP` â†’ attention = 0.
- `hGY2` cáº¥p `qb` nhÆ°ng D2H gradient `fb` â†’ trÃ n buffer, segfault.

## Sá»‘ Ä‘o baseline GPT2-medium (notebook 85, HEAD `09755b1`)

```
MS_TOTAL 12843.70 ms / 2 step  ->  MS_STEP 6421.85 ms
LOSS 10.824877 (x2), KET LUAN: OK
```

So vá»›i bridge C++ sinh ra `187.8 ms/step` (torch eager `442.3 ms`, speedup `2.36x`):
IL-only host-heavy cháº­m hÆ¡n ~34x. NguyÃªn nhÃ¢n: má»—i lá»›p D2H/H2D buffer `Z` (~8 MB),
host gelu ~2M pháº§n tá»­ Ã—24, host softmax CE ~25.7M pháº§n tá»­ Ã—2, host LN 25M pháº§n tá»­.

## Tiáº¿p theo

1. DÃ¹ng cÃ¡c kernel device Ä‘Ã£ verify (`lnf`, `sf`, `gelu`, `dgelu`, `loss_kernel`) thay host loop.
   Cáº§n bá»›t thÃªm local (hiá»‡n 253/255) Ä‘á»ƒ nhÃ©t handle/module.
2. ThÃªm AdamW (hiá»‡n SGD `lr=1e-7`).
3. ThÃªm path fp16 tensor-core (`cublasGemmEx`) vá»›i master weight fp32 Ä‘á»ƒ so `187.8 ms`.


## 2026-11-26 (muon sau): CE loss device + clean LN + SGD fix

- TokenVector.CudaBackend@18adb03, compile sach, max ldloc.s/stloc.s = 255/255.
- loss_kernel_packed (3 params, packed T/V/invT/ids + combined lossRow+dL) verified tren Kaggle T4:
  MAXERR 7.450581e-09, LAUNCH ok, LOSSROW0 1.4658537 / LOSSROW1 0.20593393, KET LUAN OK.
- Cau truc trainN: device gelu + packed CE device + LN backward sach (khong rename acc/dvv/gvv) +
  cap nhat trong so SGD dung tung tensor (trc do wph bi ghi de 6 lan -> toan bo gradient vao HW2).
- Can chay lai TRAIN_N de xac nhan LOSS 2 step (ky vong step 2 giam) + MS_STEP + gradient.
- Speed baseline van 187.8 ms; chua do lai.
