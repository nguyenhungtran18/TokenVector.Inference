# IL-only transformer: device kernels + block forward (2026-11-26)

## CÃƒÆ’Ã‚Â¡c mÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ xÃƒÆ’Ã‚Â¡c thÃƒÂ¡Ã‚Â»Ã‚Â±c trÃƒÆ’Ã‚Âªn T4

| MÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc | HEAD | Notebook | KÃƒÂ¡Ã‚ÂºÃ‚Â¿t quÃƒÂ¡Ã‚ÂºÃ‚Â£ |
|---|---|---|---|
| Full GPU train step (toy, host loss) | `b831002` | V4 / notebook 24 | `KET LUAN: OK`; loss maxerr `4.6309e-7`; dG1max `3.19e-9` |
| Full GPU train step + device loss kernel | `93e8bc7` | V4 / notebook 26 | `KET LUAN: OK`; **khÃƒÆ’Ã‚Â´ng D2H `TÃƒÆ’Ã¢â‚¬â€V`, khÃƒÆ’Ã‚Â´ng H2D `dL` mÃƒÂ¡Ã‚Â»Ã¢â‚¬â€i step**; chÃƒÂ¡Ã‚Â»Ã¢â‚¬Â° D2H `T` float |
| Timing harness (`time_perf` + `cudaDeviceSynchronize`) | `4ce6343` | ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â | `MS_TOTAL`/`MS_STEP` in cuÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœi step loop (chÃƒâ€ Ã‚Â°a cÃƒÆ’Ã‚Â³ sÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœ Ãƒâ€žÃ¢â‚¬Ëœo scale lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºn) |
| 4 kernel device-only cho transformer | `62bffac` | `KAGGLE_ILGPT2_KERNELS` / notebook 27 | `LY 2.29e-6`, `LDX 1.69e-6`, `SP 8.84e-8`, `DS 8.47e-10` |
| Block transformer forward (T=4,H=8) | `f117852` (baseline khÃƒÆ’Ã‚Â´i phÃƒÂ¡Ã‚Â»Ã‚Â¥c tÃƒÂ¡Ã‚Â»Ã‚Â« `03b23b1`) | `KAGGLE_ILGPT2_BLOCK_FWD` / notebook 39 | `Y1 1.15e-6`, `P 1.20e-7`, `X2 5.53e-7`, `Y2 3.60e-5`, `HH 3.02e-6`, `O2 2.34e-5`, `OUT 2.32e-5`, `KET LUAN: OK` |

## Bug Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ tÃƒÆ’Ã‚Â¬m vÃƒÆ’Ã‚Â  sÃƒÂ¡Ã‚Â»Ã‚Â­a trong block forward

1. **TrÃƒÆ’Ã‚Â n buffer tÃƒÆ’Ã‚Âªn kernel**: `nb = len(FN1)` = 11 B nhÃƒâ€ Ã‚Â°ng `"gelu_kernel\0"` cÃƒÂ¡Ã‚ÂºÃ‚Â§n 12 B ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ segfault trÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºc khi in gÃƒÆ’Ã‚Â¬. SÃƒÂ¡Ã‚Â»Ã‚Â­a `nb = 32`.
2. **Residual add lÃƒÂ¡Ã‚ÂºÃ‚Â§n 2**: chÃƒÂ¡Ã‚Â»Ã¢â‚¬Â° Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢i `c1=dO2`, `c2=dOUT`, quÃƒÆ’Ã‚Âªn Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢i `c0=dX ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ dX2` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ `OUT = X + O2`, sai `1.375`. SÃƒÂ¡Ã‚Â»Ã‚Â­a xong `OUT 2.32e-5`.

## VÃƒÂ¡Ã‚ÂºÃ‚Â¥n Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã‚Â cÃƒÆ’Ã‚Â²n mÃƒÂ¡Ã‚Â»Ã…Â¸

`lnf_kernel` (LayerNorm forward device) chÃƒÂ¡Ã‚ÂºÃ‚Â¡y OK ÃƒÂ¡Ã‚Â»Ã…Â¸ lÃƒÂ¡Ã‚ÂºÃ‚Â§n gÃƒÂ¡Ã‚Â»Ã‚Âi thÃƒÂ¡Ã‚Â»Ã‚Â© nhÃƒÂ¡Ã‚ÂºÃ‚Â¥t (LN1). LÃƒÂ¡Ã‚ÂºÃ‚Â§n gÃƒÂ¡Ã‚Â»Ã‚Âi thÃƒÂ¡Ã‚Â»Ã‚Â© hai (LN2) gÃƒÆ’Ã‚Â¢y `SIGSEGV` trong `cuLaunchKernel`, dÃƒÆ’Ã‚Â¹:

- 6 pointer device Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã‚Âu hÃƒÂ¡Ã‚Â»Ã‚Â£p lÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¡ (Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ in ra);
- dÃƒÆ’Ã‚Â¹ng `kernelParams` buffer riÃƒÆ’Ã‚Âªng (`pl2`) vÃƒÆ’Ã‚Â  **entry PTX riÃƒÆ’Ã‚Âªng** `ln2f_kernel` / `CUfunction` riÃƒÆ’Ã‚Âªng vÃƒÂ¡Ã‚ÂºÃ‚Â«n crash;
- khÃƒÆ’Ã‚Â´ng phÃƒÂ¡Ã‚ÂºÃ‚Â£i lÃƒÂ¡Ã‚Â»Ã¢â‚¬â€i JIT (khÃƒÆ’Ã‚Â´ng cÃƒÆ’Ã‚Â³ `MODULE FAIL`/`FN ... FAIL`).

HiÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¡n tÃƒÂ¡Ã‚ÂºÃ‚Â¡i LN2 tÃƒÆ’Ã‚Â­nh trÃƒÆ’Ã‚Âªn host (D2H `TÃƒÆ’Ã¢â‚¬â€H`, tÃƒÆ’Ã‚Â­nh mean/rstd/xhat/y, H2D lÃƒÂ¡Ã‚ÂºÃ‚Â¡i) Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ block chÃƒÂ¡Ã‚ÂºÃ‚Â¡y Ãƒâ€žÃ¢â‚¬ËœÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã‚Â£c. PhÃƒÂ¡Ã‚ÂºÃ‚Â£i thay bÃƒÂ¡Ã‚ÂºÃ‚Â±ng thiÃƒÂ¡Ã‚ÂºÃ‚Â¿t kÃƒÂ¡Ã‚ÂºÃ‚Â¿ device khÃƒÆ’Ã‚Â¡c trÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºc khi Ãƒâ€žÃ¢â‚¬Ëœo tÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢ thÃƒÂ¡Ã‚ÂºÃ‚Â­t ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â vÃƒÆ’Ã‚Â­ dÃƒÂ¡Ã‚Â»Ã‚Â¥ tÃƒÆ’Ã‚Â¡ch reduction mean/rstd thÃƒÆ’Ã‚Â nh kernel riÃƒÆ’Ã‚Âªng rÃƒÂ¡Ã‚Â»Ã¢â‚¬Å“i dÃƒÆ’Ã‚Â¹ng `norm_kernel`, hoÃƒÂ¡Ã‚ÂºÃ‚Â·c dÃƒÆ’Ã‚Â¹ng shared-memory reduction vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi block per-row.

## QuyÃƒÂ¡Ã‚ÂºÃ‚Â¿t Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¹nh kiÃƒÂ¡Ã‚ÂºÃ‚Â¿n trÃƒÆ’Ã‚Âºc

- Ãƒâ€žÃ‚ÂÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢ chÃƒÆ’Ã‚Â­nh xÃƒÆ’Ã‚Â¡c: **lÃƒÆ’Ã‚Â m cÃƒÂ¡Ã‚ÂºÃ‚Â£ hai**. fp32 ÃƒÂ¡Ã‚Â»Ã…Â¸ scale trung gian Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ khoÃƒÆ’Ã‚Â¡ thuÃƒÂ¡Ã‚ÂºÃ‚Â­t toÃƒÆ’Ã‚Â¡n (`MAXERR ~1e-6`), sau Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â³ thÃƒÆ’Ã‚Âªm path fp16 tensor-core (`cublasGemmEx(CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`) + master weight fp32 Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ so tÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢ vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi baseline `187.8 ms`.
- TrÃƒÆ’Ã‚Â¬nh tÃƒÂ¡Ã‚Â»Ã‚Â± scale: **trung gian trÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºc** (T=512, ÃƒÆ’Ã‚Â­t lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp/V nhÃƒÂ¡Ã‚Â»Ã‚Â rÃƒÂ¡Ã‚Â»Ã¢â‚¬Å“i mÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi V=50257, NL=24, 60 steps).
- Baseline cÃƒÂ¡Ã‚ÂºÃ‚Â§n vÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã‚Â£t: torch eager `442.3 ms` vs bridge C++ sinh ra `187.8 ms` = `2.36ÃƒÆ’Ã¢â‚¬â€`, fingerprint `3.8906` (fp16, khÃƒÆ’Ã‚Â´ng so trÃƒÂ¡Ã‚Â»Ã‚Â±c tiÃƒÂ¡Ã‚ÂºÃ‚Â¿p vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi HF fp32 `9.0058`).

## Block backward ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verified (notebook 54, HEAD `aa13acd`)

Script `Release/cuda_backend/examples/ilgpt2_bwd_clean.tkv` (exe 20 480 B) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â tÃƒÆ’Ã‚Â¡ch khÃƒÂ¡Ã‚Â»Ã‚Âi forward,
mÃƒÂ¡Ã‚Â»Ã‚Âi tensor Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚ÂºÃ‚Â§u vÃƒÆ’Ã‚Â o sinh deterministic trÃƒÆ’Ã‚Âªn host rÃƒÂ¡Ã‚Â»Ã¢â‚¬Å“i H2D, 16 gradient so vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi reference double:

```
GW2 2.0e-8   GW1 1.7e-7   GY2 1.7e-7   GX2 6.9e-7
GA  1.7e-6   GWO 2.6e-6   GP  5.5e-6   GV 1.3e-6
GS  1.0e-6   GQ  3.7e-7   GK  7.9e-7
GWQ 4.4e-7   GWK 1.5e-6   GWV 1.1e-5
GY1 3.2e-6   GDX 5.2e-6
KET LUAN: OK
```

CÃƒÆ’Ã‚Â¡c bug phÃƒÂ¡Ã‚ÂºÃ‚Â£i sÃƒÂ¡Ã‚Â»Ã‚Â­a (Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã‚Âu lÃƒÆ’Ã‚Â  dÃƒÂ¡Ã‚ÂºÃ‚Â¡ng **ghi Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â¨ weight/buffer Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚ÂºÃ‚Â§u vÃƒÆ’Ã‚Â o bÃƒÂ¡Ã‚ÂºÃ‚Â±ng gradient** hoÃƒÂ¡Ã‚ÂºÃ‚Â·c sai kÃƒÆ’Ã‚Â­ch thÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºc):

1. `dZ`/`dHH` device buffer cÃƒÂ¡Ã‚ÂºÃ‚Â¥p `qb` (128 B) thay vÃƒÆ’Ã‚Â¬ `fb` (512 B).
2. `dHHg` tÃƒÆ’Ã‚Â¡ch riÃƒÆ’Ã‚Âªng; `dZ` dÃƒÆ’Ã‚Â¹ng `dHHg` (gradient), khÃƒÆ’Ã‚Â´ng dÃƒÆ’Ã‚Â¹ng `dHH` (input thÃƒÆ’Ã‚Â´).
3. `dW1g`/`dW2g` tÃƒÆ’Ã‚Â¡ch riÃƒÆ’Ã‚Âªng; `dY2` dÃƒÆ’Ã‚Â¹ng `dW1` gÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc, `dHHg` dÃƒÆ’Ã‚Â¹ng `dW2` gÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc.
4. `dV` GEMM Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Âºng `m=T, n=H, ldc=T` (khÃƒÆ’Ã‚Â´ng phÃƒÂ¡Ã‚ÂºÃ‚Â£i `H,T`).
5. `dK` GEMM Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Âºng `m=T, n=H`.

BÃƒÂ¡Ã‚ÂºÃ‚Â£n QKV-backward tÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœi thiÃƒÂ¡Ã‚Â»Ã†â€™u (`ilgpt2_qkv_bwd.tkv`, notebook 49) cÃƒâ€¦Ã‚Â©ng sÃƒÂ¡Ã‚ÂºÃ‚Â¡ch:
`GWQ 6e-8, GWK 1e-7, GWV 2e-7, GY1 7e-8`.

**KÃƒÂ¡Ã‚ÂºÃ‚Â¿t luÃƒÂ¡Ã‚ÂºÃ‚Â­n**: forward vÃƒÆ’Ã‚Â  backward cÃƒÂ¡Ã‚Â»Ã‚Â§a mÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢t block transformer Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã‚Âu Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ verify ÃƒÂ¡Ã‚Â»Ã…Â¸ mÃƒÂ¡Ã‚Â»Ã‚Â©c f32,
dÃƒÆ’Ã‚Â¹ Ãƒâ€žÃ¢â‚¬Ëœang tÃƒÂ¡Ã‚ÂºÃ‚Â¡m tÃƒÆ’Ã‚Â­nh LN/dgelu/softmax-backward trÃƒÆ’Ã‚Âªn host (cÃƒÂ¡Ã‚ÂºÃ‚Â§n device hÃƒÆ’Ã‚Â³a Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ Ãƒâ€žÃ¢â‚¬Ëœo tÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢).

## 1-layer train step ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verified (notebook 61, HEAD `0804559`)

`Release/cuda_backend/examples/ilgpt2_train1.tkv` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â forward + CE loss + backward + SGD, 3 bÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºc,
T=4 H=8 F=32 V=8, weight deterministic:

```
LOSS 2.821911 -> 2.337988 -> 1.9980869
3 x MARK UPD, khong co FAIL, KET LUAN: OK
```

Bug quyÃƒÂ¡Ã‚ÂºÃ‚Â¿t Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¹nh: IL cÃƒÆ’Ã‚Â³ **263 local**, `ldloc.s`/`stloc.s` dÃƒÆ’Ã‚Â¹ng chÃƒÂ¡Ã‚Â»Ã¢â‚¬Â° sÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœ 8-bit ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ trÃƒÆ’Ã‚Â n ÃƒÂ¡Ã‚Â»Ã…Â¸ 256,
lÃƒÆ’Ã‚Â m `T/H/F/V` Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã‚Âc thÃƒÆ’Ã‚Â nh `256/256/65536/2048` (vÃƒÆ’Ã‚Â²ng LN chÃƒÂ¡Ã‚ÂºÃ‚Â¡y 29 lÃƒÂ¡Ã‚ÂºÃ‚Â§n). Gom cÃƒÆ’Ã‚Â¡c handle tÃƒÂ¡Ã‚ÂºÃ‚Â¡m
(`w0..w25`, `f0..f20`, `u*`, `v*`, `z*`, `b*`, `c*`) thÃƒÆ’Ã‚Â nh 2 biÃƒÂ¡Ã‚ÂºÃ‚Â¿n dÃƒÆ’Ã‚Â¹ng lÃƒÂ¡Ã‚ÂºÃ‚Â¡i ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ max `ldloc.s = 177`.
BÃƒÆ’Ã‚Â i hÃƒÂ¡Ã‚Â»Ã‚Âc: **phÃƒÂ¡Ã‚ÂºÃ‚Â£i kiÃƒÂ¡Ã‚Â»Ã†â€™m tra max local index < 256** cho mÃƒÂ¡Ã‚Â»Ã‚Âi script .tkv lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºn.

## Train T=512 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verified (notebook 64, HEAD `cadd9fa`)

CÃƒÆ’Ã‚Â¹ng script `ilgpt2_train1.tkv`, Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢i `T=4 -> 512` vÃƒÆ’Ã‚Â  sÃƒÂ¡Ã‚Â»Ã‚Â­a `dS` cÃƒÂ¡Ã‚ÂºÃ‚Â¥p `sb = T*T*4` (thay `fb`):

```
LOSS 3.4078045 -> 3.2996545 -> 3.2021413
3 x MARK UPD, khong FAIL, KET LUAN: OK
MS_STEP 79.58 ms  (baseline host-heavy: LN/softmax/dgelu/CE deu qua host)
```

`lr` phÃƒÂ¡Ã‚ÂºÃ‚Â£i Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢i `1e-3 -> 1e-6`: gradient trÃƒÂ¡Ã‚Â»Ã‚Âng sÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœ lÃƒÆ’Ã‚Â  **tÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢ng qua T token**, nÃƒÆ’Ã‚Âªn tÃƒâ€žÃ†â€™ng ~128ÃƒÆ’Ã¢â‚¬â€
khi T tÃƒâ€žÃ†â€™ng 4 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ 512. Ãƒâ€žÃ‚ÂÃƒÆ’Ã‚Â¢y lÃƒÆ’Ã‚Â  mÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœc train IL-only Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚ÂºÃ‚Â§u tiÃƒÆ’Ã‚Âªn ÃƒÂ¡Ã‚Â»Ã…Â¸ `T=512`.

## 2-layer train step ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verified (notebook 65, exe HEAD `7e9df80`)

`ilgpt2_train2.tkv` sinh tÃƒÂ¡Ã‚Â»Ã‚Â± Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã¢â€žÂ¢ng tÃƒÂ¡Ã‚Â»Ã‚Â« bÃƒÂ¡Ã‚ÂºÃ‚Â£n 1 lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp: forward lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp 1 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp 2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ CE ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ backward lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp 2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp 1 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ SGD cÃƒÂ¡Ã‚ÂºÃ‚Â£ hai.
T=512, H=8, F=32, V=8, lr=1e-6:

```
LOSS 4.4484606 -> 4.139653 -> 3.8697996
2 x MARK FWD + 2 x MARK UPD / step, khong FAIL
KET LUAN: OK (sau khi sua reference regex / so luong update)
```

Max local IL = **249** (Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ chia sÃƒÂ¡Ã‚ÂºÃ‚Â» scratch vÃƒÆ’Ã‚Â  buffer gradient giÃƒÂ¡Ã‚Â»Ã‚Â¯a 2 lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ nÃƒÆ’Ã‚Â© ngÃƒâ€ Ã‚Â°ÃƒÂ¡Ã‚Â»Ã‚Â¡ng 256).

## V=50257, 2 lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â verified (notebook 68, HEAD `1f8b10d`)

```
LOSS 13.193880 -> 12.885666 -> 12.616205
2 x MARK FWD + 2 x MARK UPD / step, KET LUAN: OK
```

Bug khi tÃƒâ€žÃ†â€™ng V: khÃƒÂ¡Ã‚Â»Ã…Â¸i tÃƒÂ¡Ã‚ÂºÃ‚Â¡o `Wh` dÃƒÆ’Ã‚Â¹ng `0.2 + 0.01*(i+j)` vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi `j` tÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi 50257 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ weight tÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi ~502,
logits explode, `exp` underflow, loss `inf`. Ãƒâ€žÃ‚ÂÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢i thÃƒÆ’Ã‚Â nh `0.2 + 0.01*(i + (j % 8))`.

## GPT2-medium config ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â chÃƒÂ¡Ã‚ÂºÃ‚Â¡y sÃƒÂ¡Ã‚ÂºÃ‚Â¡ch (notebook 84, HEAD `e413c32`)

`ilgpt2_trainN.tkv` dÃƒÂ¡Ã‚ÂºÃ‚Â¡ng layer-loop, `NL=24, T=512, H=1024, F=4096, V=50257`:

```
LOSS 10.824877 (x2 step, ~= ln 50257)
24 x MARK B (23..0) / step, MARK FWD + MARK BWD + MARK UPD, khong FAIL
KET LUAN: OK
max ldloc.s = 254 (<256)
```

CÃƒÆ’Ã‚Â¡c bug khi scale phÃƒÂ¡Ã‚ÂºÃ‚Â£i sÃƒÂ¡Ã‚Â»Ã‚Â­a:
- khÃƒÂ¡Ã‚Â»Ã…Â¸i tÃƒÂ¡Ã‚ÂºÃ‚Â¡o weight/input/gamma chÃƒÂ¡Ã‚Â»Ã¢â‚¬Â° dÃƒÆ’Ã‚Â¹ng `jj % 8` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ `ii` vÃƒÂ¡Ã‚ÂºÃ‚Â«n tÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi 1024 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ weight ~10 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ activation nÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¢;
  dÃƒÆ’Ã‚Â¹ng hÃƒÂ¡Ã‚ÂºÃ‚Â±ng sÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœ cho mÃƒÂ¡Ã‚Â»Ã‚Âi init.
- `exp(2*z)` cho tanh overflow khi `z` lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºn ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ clamp `thv = Ãƒâ€šÃ‚Â±1` khi `|z| > 8`.
- sau softmax chÃƒÂ¡Ã‚Â»Ã¢â‚¬Â° H2D vÃƒÆ’Ã‚Â o buffer riÃƒÆ’Ã‚Âªng `pd`, quÃƒÆ’Ã‚Âªn H2D vÃƒÆ’Ã‚Â o buffer dÃƒÆ’Ã‚Â¹ng chung `dP` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ attention = 0.
- `hGY2` cÃƒÂ¡Ã‚ÂºÃ‚Â¥p `qb` nhÃƒâ€ Ã‚Â°ng D2H gradient `fb` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ trÃƒÆ’Ã‚Â n buffer, segfault.

## SÃƒÂ¡Ã‚Â»Ã¢â‚¬Ëœ Ãƒâ€žÃ¢â‚¬Ëœo baseline GPT2-medium (notebook 85, HEAD `09755b1`)

```
MS_TOTAL 12843.70 ms / 2 step  ->  MS_STEP 6421.85 ms
LOSS 10.824877 (x2), KET LUAN: OK
```

So vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi bridge C++ sinh ra `187.8 ms/step` (torch eager `442.3 ms`, speedup `2.36x`):
IL-only host-heavy chÃƒÂ¡Ã‚ÂºÃ‚Â­m hÃƒâ€ Ã‚Â¡n ~34x. NguyÃƒÆ’Ã‚Âªn nhÃƒÆ’Ã‚Â¢n: mÃƒÂ¡Ã‚Â»Ã¢â‚¬â€i lÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºp D2H/H2D buffer `Z` (~8 MB),
host gelu ~2M phÃƒÂ¡Ã‚ÂºÃ‚Â§n tÃƒÂ¡Ã‚Â»Ã‚Â­ ÃƒÆ’Ã¢â‚¬â€24, host softmax CE ~25.7M phÃƒÂ¡Ã‚ÂºÃ‚Â§n tÃƒÂ¡Ã‚Â»Ã‚Â­ ÃƒÆ’Ã¢â‚¬â€2, host LN 25M phÃƒÂ¡Ã‚ÂºÃ‚Â§n tÃƒÂ¡Ã‚Â»Ã‚Â­.

## TiÃƒÂ¡Ã‚ÂºÃ‚Â¿p theo

1. DÃƒÆ’Ã‚Â¹ng cÃƒÆ’Ã‚Â¡c kernel device Ãƒâ€žÃ¢â‚¬ËœÃƒÆ’Ã‚Â£ verify (`lnf`, `sf`, `gelu`, `dgelu`, `loss_kernel`) thay host loop.
   CÃƒÂ¡Ã‚ÂºÃ‚Â§n bÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºt thÃƒÆ’Ã‚Âªm local (hiÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¡n 253/255) Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ nhÃƒÆ’Ã‚Â©t handle/module.
2. ThÃƒÆ’Ã‚Âªm AdamW (hiÃƒÂ¡Ã‚Â»Ã¢â‚¬Â¡n SGD `lr=1e-7`).
3. ThÃƒÆ’Ã‚Âªm path fp16 tensor-core (`cublasGemmEx`) vÃƒÂ¡Ã‚Â»Ã¢â‚¬Âºi master weight fp32 Ãƒâ€žÃ¢â‚¬ËœÃƒÂ¡Ã‚Â»Ã†â€™ so `187.8 ms`.


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


## Cuoc dieu tra gradient NaN (ket thuc phien)

- wph bi ghi de 6 lan (tat ca gradient vao HW2) da sua: update dung HWq/HWk/HWv/HWo/HW1/HW2.
- exp(2*z) khong co clamp trong vong gelu recompute cua backward da them clamp.
- Tuy nhien van NaN: step1 forward finite (LOSS 10.8249), gHH0=1.9e-6, gOUT0=8.5e-9 finite,
  nhung dW1[0]/dW2[0] = nan o moi lop; lr=1e-3 lam LOSS step2 = nan.
- Ket luan: loi nam o duong weight-gradient GEMM/update (co the device-buffer aliasing hoac
  cuBLAS status sticky do D2D copy qua wrapper cuda_h2d), KHONG phai CE/softmax/gelu.
- Chua co fix da verify. Da khoi phuc ban chay duoc cuoi (c7de056) tren HEAD 80da502.
- CE device rieng van da verify MAXERR 7.45e-09.
