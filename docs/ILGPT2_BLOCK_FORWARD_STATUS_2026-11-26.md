# IL-only transformer: device kernels + block forward (2026-11-26)

## CÃƒÂ¡c mÃ¡Â»â€˜c Ã„â€˜ÃƒÂ£ xÃƒÂ¡c thÃ¡Â»Â±c trÃƒÂªn T4

| MÃ¡Â»â€˜c | HEAD | Notebook | KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ |
|---|---|---|---|
| Full GPU train step (toy, host loss) | `b831002` | V4 / notebook 24 | `KET LUAN: OK`; loss maxerr `4.6309e-7`; dG1max `3.19e-9` |
| Full GPU train step + device loss kernel | `93e8bc7` | V4 / notebook 26 | `KET LUAN: OK`; **khÃƒÂ´ng D2H `TÃƒâ€”V`, khÃƒÂ´ng H2D `dL` mÃ¡Â»â€”i step**; chÃ¡Â»â€° D2H `T` float |
| Timing harness (`time_perf` + `cudaDeviceSynchronize`) | `4ce6343` | Ã¢â‚¬â€ | `MS_TOTAL`/`MS_STEP` in cuÃ¡Â»â€˜i step loop (chÃ†Â°a cÃƒÂ³ sÃ¡Â»â€˜ Ã„â€˜o scale lÃ¡Â»â€ºn) |
| 4 kernel device-only cho transformer | `62bffac` | `KAGGLE_ILGPT2_KERNELS` / notebook 27 | `LY 2.29e-6`, `LDX 1.69e-6`, `SP 8.84e-8`, `DS 8.47e-10` |
| Block transformer forward (T=4,H=8) | `f117852` (baseline khÃƒÂ´i phÃ¡Â»Â¥c tÃ¡Â»Â« `03b23b1`) | `KAGGLE_ILGPT2_BLOCK_FWD` / notebook 39 | `Y1 1.15e-6`, `P 1.20e-7`, `X2 5.53e-7`, `Y2 3.60e-5`, `HH 3.02e-6`, `O2 2.34e-5`, `OUT 2.32e-5`, `KET LUAN: OK` |

## Bug Ã„â€˜ÃƒÂ£ tÃƒÂ¬m vÃƒÂ  sÃ¡Â»Â­a trong block forward

1. **TrÃƒÂ n buffer tÃƒÂªn kernel**: `nb = len(FN1)` = 11 B nhÃ†Â°ng `"gelu_kernel\0"` cÃ¡ÂºÂ§n 12 B Ã¢â€ â€™ segfault trÃ†Â°Ã¡Â»â€ºc khi in gÃƒÂ¬. SÃ¡Â»Â­a `nb = 32`.
2. **Residual add lÃ¡ÂºÂ§n 2**: chÃ¡Â»â€° Ã„â€˜Ã¡Â»â€¢i `c1=dO2`, `c2=dOUT`, quÃƒÂªn Ã„â€˜Ã¡Â»â€¢i `c0=dX Ã¢â€ â€™ dX2` Ã¢â€ â€™ `OUT = X + O2`, sai `1.375`. SÃ¡Â»Â­a xong `OUT 2.32e-5`.

## VÃ¡ÂºÂ¥n Ã„â€˜Ã¡Â»Â cÃƒÂ²n mÃ¡Â»Å¸

`lnf_kernel` (LayerNorm forward device) chÃ¡ÂºÂ¡y OK Ã¡Â»Å¸ lÃ¡ÂºÂ§n gÃ¡Â»Âi thÃ¡Â»Â© nhÃ¡ÂºÂ¥t (LN1). LÃ¡ÂºÂ§n gÃ¡Â»Âi thÃ¡Â»Â© hai (LN2) gÃƒÂ¢y `SIGSEGV` trong `cuLaunchKernel`, dÃƒÂ¹:

- 6 pointer device Ã„â€˜Ã¡Â»Âu hÃ¡Â»Â£p lÃ¡Â»â€¡ (Ã„â€˜ÃƒÂ£ in ra);
- dÃƒÂ¹ng `kernelParams` buffer riÃƒÂªng (`pl2`) vÃƒÂ  **entry PTX riÃƒÂªng** `ln2f_kernel` / `CUfunction` riÃƒÂªng vÃ¡ÂºÂ«n crash;
- khÃƒÂ´ng phÃ¡ÂºÂ£i lÃ¡Â»â€”i JIT (khÃƒÂ´ng cÃƒÂ³ `MODULE FAIL`/`FN ... FAIL`).

HiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i LN2 tÃƒÂ­nh trÃƒÂªn host (D2H `TÃƒâ€”H`, tÃƒÂ­nh mean/rstd/xhat/y, H2D lÃ¡ÂºÂ¡i) Ã„â€˜Ã¡Â»Æ’ block chÃ¡ÂºÂ¡y Ã„â€˜Ã†Â°Ã¡Â»Â£c. PhÃ¡ÂºÂ£i thay bÃ¡ÂºÂ±ng thiÃ¡ÂºÂ¿t kÃ¡ÂºÂ¿ device khÃƒÂ¡c trÃ†Â°Ã¡Â»â€ºc khi Ã„â€˜o tÃ¡Â»â€˜c Ã„â€˜Ã¡Â»â„¢ thÃ¡ÂºÂ­t Ã¢â‚¬â€ vÃƒÂ­ dÃ¡Â»Â¥ tÃƒÂ¡ch reduction mean/rstd thÃƒÂ nh kernel riÃƒÂªng rÃ¡Â»â€œi dÃƒÂ¹ng `norm_kernel`, hoÃ¡ÂºÂ·c dÃƒÂ¹ng shared-memory reduction vÃ¡Â»â€ºi block per-row.

## QuyÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»â€¹nh kiÃ¡ÂºÂ¿n trÃƒÂºc

- Ã„ÂÃ¡Â»â„¢ chÃƒÂ­nh xÃƒÂ¡c: **lÃƒÂ m cÃ¡ÂºÂ£ hai**. fp32 Ã¡Â»Å¸ scale trung gian Ã„â€˜Ã¡Â»Æ’ khoÃƒÂ¡ thuÃ¡ÂºÂ­t toÃƒÂ¡n (`MAXERR ~1e-6`), sau Ã„â€˜ÃƒÂ³ thÃƒÂªm path fp16 tensor-core (`cublasGemmEx(CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`) + master weight fp32 Ã„â€˜Ã¡Â»Æ’ so tÃ¡Â»â€˜c Ã„â€˜Ã¡Â»â„¢ vÃ¡Â»â€ºi baseline `187.8 ms`.
- TrÃƒÂ¬nh tÃ¡Â»Â± scale: **trung gian trÃ†Â°Ã¡Â»â€ºc** (T=512, ÃƒÂ­t lÃ¡Â»â€ºp/V nhÃ¡Â»Â rÃ¡Â»â€œi mÃ¡Â»â€ºi V=50257, NL=24, 60 steps).
- Baseline cÃ¡ÂºÂ§n vÃ†Â°Ã¡Â»Â£t: torch eager `442.3 ms` vs bridge C++ sinh ra `187.8 ms` = `2.36Ãƒâ€”`, fingerprint `3.8906` (fp16, khÃƒÂ´ng so trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p vÃ¡Â»â€ºi HF fp32 `9.0058`).

## Block backward Ã¢â‚¬â€ verified (notebook 54, HEAD `aa13acd`)

Script `Release/cuda_backend/examples/ilgpt2_bwd_clean.tkv` (exe 20 480 B) Ã¢â‚¬â€ tÃƒÂ¡ch khÃ¡Â»Âi forward,
mÃ¡Â»Âi tensor Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o sinh deterministic trÃƒÂªn host rÃ¡Â»â€œi H2D, 16 gradient so vÃ¡Â»â€ºi reference double:

```
GW2 2.0e-8   GW1 1.7e-7   GY2 1.7e-7   GX2 6.9e-7
GA  1.7e-6   GWO 2.6e-6   GP  5.5e-6   GV 1.3e-6
GS  1.0e-6   GQ  3.7e-7   GK  7.9e-7
GWQ 4.4e-7   GWK 1.5e-6   GWV 1.1e-5
GY1 3.2e-6   GDX 5.2e-6
KET LUAN: OK
```

CÃƒÂ¡c bug phÃ¡ÂºÂ£i sÃ¡Â»Â­a (Ã„â€˜Ã¡Â»Âu lÃƒÂ  dÃ¡ÂºÂ¡ng **ghi Ã„â€˜ÃƒÂ¨ weight/buffer Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o bÃ¡ÂºÂ±ng gradient** hoÃ¡ÂºÂ·c sai kÃƒÂ­ch thÃ†Â°Ã¡Â»â€ºc):

1. `dZ`/`dHH` device buffer cÃ¡ÂºÂ¥p `qb` (128 B) thay vÃƒÂ¬ `fb` (512 B).
2. `dHHg` tÃƒÂ¡ch riÃƒÂªng; `dZ` dÃƒÂ¹ng `dHHg` (gradient), khÃƒÂ´ng dÃƒÂ¹ng `dHH` (input thÃƒÂ´).
3. `dW1g`/`dW2g` tÃƒÂ¡ch riÃƒÂªng; `dY2` dÃƒÂ¹ng `dW1` gÃ¡Â»â€˜c, `dHHg` dÃƒÂ¹ng `dW2` gÃ¡Â»â€˜c.
4. `dV` GEMM Ã„â€˜ÃƒÂºng `m=T, n=H, ldc=T` (khÃƒÂ´ng phÃ¡ÂºÂ£i `H,T`).
5. `dK` GEMM Ã„â€˜ÃƒÂºng `m=T, n=H`.

BÃ¡ÂºÂ£n QKV-backward tÃ¡Â»â€˜i thiÃ¡Â»Æ’u (`ilgpt2_qkv_bwd.tkv`, notebook 49) cÃ…Â©ng sÃ¡ÂºÂ¡ch:
`GWQ 6e-8, GWK 1e-7, GWV 2e-7, GY1 7e-8`.

**KÃ¡ÂºÂ¿t luÃ¡ÂºÂ­n**: forward vÃƒÂ  backward cÃ¡Â»Â§a mÃ¡Â»â„¢t block transformer Ã„â€˜Ã¡Â»Âu Ã„â€˜ÃƒÂ£ verify Ã¡Â»Å¸ mÃ¡Â»Â©c f32,
dÃƒÂ¹ Ã„â€˜ang tÃ¡ÂºÂ¡m tÃƒÂ­nh LN/dgelu/softmax-backward trÃƒÂªn host (cÃ¡ÂºÂ§n device hÃƒÂ³a Ã„â€˜Ã¡Â»Æ’ Ã„â€˜o tÃ¡Â»â€˜c Ã„â€˜Ã¡Â»â„¢).

## 1-layer train step Ã¢â‚¬â€ verified (notebook 61, HEAD `0804559`)

`Release/cuda_backend/examples/ilgpt2_train1.tkv` Ã¢â‚¬â€ forward + CE loss + backward + SGD, 3 bÃ†Â°Ã¡Â»â€ºc,
T=4 H=8 F=32 V=8, weight deterministic:

```
LOSS 2.821911 -> 2.337988 -> 1.9980869
3 x MARK UPD, khong co FAIL, KET LUAN: OK
```

Bug quyÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»â€¹nh: IL cÃƒÂ³ **263 local**, `ldloc.s`/`stloc.s` dÃƒÂ¹ng chÃ¡Â»â€° sÃ¡Â»â€˜ 8-bit Ã¢â€ â€™ trÃƒÂ n Ã¡Â»Å¸ 256,
lÃƒÂ m `T/H/F/V` Ã„â€˜Ã¡Â»Âc thÃƒÂ nh `256/256/65536/2048` (vÃƒÂ²ng LN chÃ¡ÂºÂ¡y 29 lÃ¡ÂºÂ§n). Gom cÃƒÂ¡c handle tÃ¡ÂºÂ¡m
(`w0..w25`, `f0..f20`, `u*`, `v*`, `z*`, `b*`, `c*`) thÃƒÂ nh 2 biÃ¡ÂºÂ¿n dÃƒÂ¹ng lÃ¡ÂºÂ¡i Ã¢â€ â€™ max `ldloc.s = 177`.
BÃƒÂ i hÃ¡Â»Âc: **phÃ¡ÂºÂ£i kiÃ¡Â»Æ’m tra max local index < 256** cho mÃ¡Â»Âi script .tkv lÃ¡Â»â€ºn.

## Train T=512 Ã¢â‚¬â€ verified (notebook 64, HEAD `cadd9fa`)

CÃƒÂ¹ng script `ilgpt2_train1.tkv`, Ã„â€˜Ã¡Â»â€¢i `T=4 -> 512` vÃƒÂ  sÃ¡Â»Â­a `dS` cÃ¡ÂºÂ¥p `sb = T*T*4` (thay `fb`):

```
LOSS 3.4078045 -> 3.2996545 -> 3.2021413
3 x MARK UPD, khong FAIL, KET LUAN: OK
MS_STEP 79.58 ms  (baseline host-heavy: LN/softmax/dgelu/CE deu qua host)
```

`lr` phÃ¡ÂºÂ£i Ã„â€˜Ã¡Â»â€¢i `1e-3 -> 1e-6`: gradient trÃ¡Â»Âng sÃ¡Â»â€˜ lÃƒÂ  **tÃ¡Â»â€¢ng qua T token**, nÃƒÂªn tÃ„Æ’ng ~128Ãƒâ€”
khi T tÃ„Æ’ng 4 Ã¢â€ â€™ 512. Ã„ÂÃƒÂ¢y lÃƒÂ  mÃ¡Â»â€˜c train IL-only Ã„â€˜Ã¡ÂºÂ§u tiÃƒÂªn Ã¡Â»Å¸ `T=512`.

## 2-layer train step Ã¢â‚¬â€ verified (notebook 65, exe HEAD `7e9df80`)

`ilgpt2_train2.tkv` sinh tÃ¡Â»Â± Ã„â€˜Ã¡Â»â„¢ng tÃ¡Â»Â« bÃ¡ÂºÂ£n 1 lÃ¡Â»â€ºp: forward lÃ¡Â»â€ºp 1 Ã¢â€ â€™ lÃ¡Â»â€ºp 2 Ã¢â€ â€™ CE Ã¢â€ â€™ backward lÃ¡Â»â€ºp 2 Ã¢â€ â€™ lÃ¡Â»â€ºp 1 Ã¢â€ â€™ SGD cÃ¡ÂºÂ£ hai.
T=512, H=8, F=32, V=8, lr=1e-6:

```
LOSS 4.4484606 -> 4.139653 -> 3.8697996
2 x MARK FWD + 2 x MARK UPD / step, khong FAIL
KET LUAN: OK (sau khi sua reference regex / so luong update)
```

Max local IL = **249** (Ã„â€˜ÃƒÂ£ chia sÃ¡ÂºÂ» scratch vÃƒÂ  buffer gradient giÃ¡Â»Â¯a 2 lÃ¡Â»â€ºp Ã„â€˜Ã¡Â»Æ’ nÃƒÂ© ngÃ†Â°Ã¡Â»Â¡ng 256).

## V=50257, 2 lÃ¡Â»â€ºp Ã¢â‚¬â€ verified (notebook 68, HEAD `1f8b10d`)

```
LOSS 13.193880 -> 12.885666 -> 12.616205
2 x MARK FWD + 2 x MARK UPD / step, KET LUAN: OK
```

Bug khi tÃ„Æ’ng V: khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o `Wh` dÃƒÂ¹ng `0.2 + 0.01*(i+j)` vÃ¡Â»â€ºi `j` tÃ¡Â»â€ºi 50257 Ã¢â€ â€™ weight tÃ¡Â»â€ºi ~502,
logits explode, `exp` underflow, loss `inf`. Ã„ÂÃ¡Â»â€¢i thÃƒÂ nh `0.2 + 0.01*(i + (j % 8))`.

## GPT2-medium config Ã¢â‚¬â€ chÃ¡ÂºÂ¡y sÃ¡ÂºÂ¡ch (notebook 84, HEAD `e413c32`)

`ilgpt2_trainN.tkv` dÃ¡ÂºÂ¡ng layer-loop, `NL=24, T=512, H=1024, F=4096, V=50257`:

```
LOSS 10.824877 (x2 step, ~= ln 50257)
24 x MARK B (23..0) / step, MARK FWD + MARK BWD + MARK UPD, khong FAIL
KET LUAN: OK
max ldloc.s = 254 (<256)
```

CÃƒÂ¡c bug khi scale phÃ¡ÂºÂ£i sÃ¡Â»Â­a:
- khÃ¡Â»Å¸i tÃ¡ÂºÂ¡o weight/input/gamma chÃ¡Â»â€° dÃƒÂ¹ng `jj % 8` Ã¢â€ â€™ `ii` vÃ¡ÂºÂ«n tÃ¡Â»â€ºi 1024 Ã¢â€ â€™ weight ~10 Ã¢â€ â€™ activation nÃ¡Â»â€¢;
  dÃƒÂ¹ng hÃ¡ÂºÂ±ng sÃ¡Â»â€˜ cho mÃ¡Â»Âi init.
- `exp(2*z)` cho tanh overflow khi `z` lÃ¡Â»â€ºn Ã¢â€ â€™ clamp `thv = Ã‚Â±1` khi `|z| > 8`.
- sau softmax chÃ¡Â»â€° H2D vÃƒÂ o buffer riÃƒÂªng `pd`, quÃƒÂªn H2D vÃƒÂ o buffer dÃƒÂ¹ng chung `dP` Ã¢â€ â€™ attention = 0.
- `hGY2` cÃ¡ÂºÂ¥p `qb` nhÃ†Â°ng D2H gradient `fb` Ã¢â€ â€™ trÃƒÂ n buffer, segfault.

## SÃ¡Â»â€˜ Ã„â€˜o baseline GPT2-medium (notebook 85, HEAD `09755b1`)

```
MS_TOTAL 12843.70 ms / 2 step  ->  MS_STEP 6421.85 ms
LOSS 10.824877 (x2), KET LUAN: OK
```

So vÃ¡Â»â€ºi bridge C++ sinh ra `187.8 ms/step` (torch eager `442.3 ms`, speedup `2.36x`):
IL-only host-heavy chÃ¡ÂºÂ­m hÃ†Â¡n ~34x. NguyÃƒÂªn nhÃƒÂ¢n: mÃ¡Â»â€”i lÃ¡Â»â€ºp D2H/H2D buffer `Z` (~8 MB),
host gelu ~2M phÃ¡ÂºÂ§n tÃ¡Â»Â­ Ãƒâ€”24, host softmax CE ~25.7M phÃ¡ÂºÂ§n tÃ¡Â»Â­ Ãƒâ€”2, host LN 25M phÃ¡ÂºÂ§n tÃ¡Â»Â­.

## TiÃ¡ÂºÂ¿p theo

1. DÃƒÂ¹ng cÃƒÂ¡c kernel device Ã„â€˜ÃƒÂ£ verify (`lnf`, `sf`, `gelu`, `dgelu`, `loss_kernel`) thay host loop.
   CÃ¡ÂºÂ§n bÃ¡Â»â€ºt thÃƒÂªm local (hiÃ¡Â»â€¡n 253/255) Ã„â€˜Ã¡Â»Æ’ nhÃƒÂ©t handle/module.
2. ThÃƒÂªm AdamW (hiÃ¡Â»â€¡n SGD `lr=1e-7`).
3. ThÃƒÂªm path fp16 tensor-core (`cublasGemmEx`) vÃ¡Â»â€ºi master weight fp32 Ã„â€˜Ã¡Â»Æ’ so `187.8 ms`.


## 2026-11-26 (muon sau): CE loss device + clean LN + SGD fix

- TokenVector.CudaBackend@18adb03, compile sach, max ldloc.s/stloc.s = 255/255.
- loss_kernel_packed (3 params, packed T/V/invT/ids + combined lossRow+dL) verified tren Kaggle T4:
  MAXERR 7.450581e-09, LAUNCH ok, LOSSROW0 1.4658537 / LOSSROW1 0.20593393, KET LUAN OK.
- Cau truc trainN: device gelu + packed CE device + LN backward sach (khong rename acc/dvv/gvv) +
  cap nhat trong so SGD dung tung tensor (trc do wph bi ghi de 6 lan -> toan bo gradient vao HW2).
- Can chay lai TRAIN_N de xac nhan LOSS 2 step (ky vong step 2 giam) + MS_STEP + gradient.
- Speed baseline van 187.8 ms; chua do lai.


## Ket thuc phien: CE device trong trainN chua xong

- packed loss kernel loss_kernel_packed da verify rieng: MAXERR 7.450581e-09 (Kaggle T4, HEAD 127a3ed).
- Tich hop vao 	rainN (clean LN c8df82f + device gelu + SGD fix) cho ket qua NaN:
  packed params dung (512, 50257, 0.001953125), module/FN rc=0, nhung logits nan -> LOSS nan ->
  backward nan -> step 2 cublas status 13. Da co lap bang marker: NaN xuat hien truoc CE,
  khong phai loi cua kernel loss (kernel da verify standalone).
- Da khoi phuc ban chay duoc cuoi cung (c7de056): LOSS 10.824877, 24 lop, MS_STEP 6828.25 ms.
- HEAD: TokenVector.CudaBackend@4910461 (restore), Inference@374d7c6.
- Chua dat 187.8 ms; fp16 tensor-core / AdamW chua lam.
