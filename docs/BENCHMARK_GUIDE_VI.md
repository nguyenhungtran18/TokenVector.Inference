# Huong dan benchmark — GPT-2 124M full training step

Tai lieu nay huong dan tai hien cac so lieu trong `README.md`, do ngay
**2026-11-20** tren Colab **Tesla T4 16 GB**, Python 3.13, torch `2.14.1+cu130`.

HEAD su dung:
- `TokenVector`: `9ef0d91`
- `TokenVector.Inference`: `f1d4b99` (`bseq.dll` SHA256 `4eb13bc3…`)

## 1. Ket qua (GPU time that ca 2 ben)

| Duong chay | seq | TokenVector | PyTorch eager | Ty le |
|---|---:|---:|---:|---:|
| `bseq.dll` fullstep (cell `# 7/9`) | 1024 | **40,0 ms** | 289,1 ms | **nhanh ~7,2 lan** |
| ctypes fullstep (cell `7D/9`) | 2048 | **212,6 ms** | 599,5 ms | **nhanh ~2,8 lan** |

Cong bang thoi gian: worker repo dat `CUDA_LAUNCH_BLOCKING=1`; worker torch goi
`torch.cuda.synchronize()` sau moi `opt.step()`. Khong ben nao chi do thoi gian enqueue.

## 2. File lien quan

- Notebook: `demo/colab_bench_ai_gpu.ipynb`.
- Cau hinh `tv/tests/_seq_cfg.txt`, 4 dong: `seq`, `mode`, `steps`, `co_fullstep`.
  Dong 4 = `1` de mode 2 di qua `_step124_gpu()` -> mot lan goi `TkvFullStep124m`.
- Ma nguon bench: `tv/tests/simd_bench_124m_seq.tkv`.
- Cau noi native: `Release/tools/simd_native/torch_fused.cpp`, `tkv_simd_native.c`.

## 3. Cac buoc

1. Mo notebook tren Colab (runtime T4 GPU).
2. Cell `1/9` kiem tra GPU/torch; cell `2/9` cai .NET 10 SDK.
3. Cell `3/9` clone lai 2 repo; kiem tra HEAD va sha cua `bseq.dll`.
4. Cell `4/9` build `tkv_simd_native.so` + `tkv_step124.so`.
5. Cell `5/9` build `tkv_torch_fused.so` (`build rc = 0`, `ctypes load OK`).
6. Cell `# 7/9`: fullstep qua `bseq.dll`, T=1024, 3 lan trong 1 process. Ky vong:
   ```
   1: FULLSTEP TONG=  40.0 ms | loss_val=8.570312 (T=1024)
   2: FULLSTEP TONG=  37.4 ms | loss_val=8.570312 (T=1024)
   3: FULLSTEP TONG=  40.1 ms | loss_val=8.570312 (T=1024)
   TONG median : 40.0 ms
   ```
   Lan dau chua lazy-load CUDA; notebook lay median cua 2 lan sau.
7. Cell `7D/9`: fullstep ctypes, T=2048 (`FULLSTEP median ms≈200`).
8. Cell `8/9`: torch eager T=2048 (da synchronize).
9. Cell cuoi `# 10/9`: torch eager T=1024 (da synchronize) — moc so sanh cong bang.

## 4. Doc ket qua

- `TONG` = tong thoi gian 1 step train (fwd + loss + bwd + AdamW), khong tinh
  thoi gian khoi dong process.
- Duong GPU dung **tham so fp16 + master fp32 + moment fp32**; torch la model fp32.
  Mot phan toc do den tu tensor-core fp16.
- **Khong so sanh loss** giua cac che do: weights ngau nhien seed 7, lich LR khac
  (`.tkv` `lr_max=0.05`, warmup 2 buoc vs `0.0025` co dinh ben torch).
- Duong CPU native EXACT (loss `25.454`) khong bi thay doi.

## 5. Xu ly su co

| Trieu chung | Nguyen nhan / cach sua |
|---|---|
| `MissingMethodException: TkvFullSetGradM` | `bseq.dll` cu; clone lai HEAD co wrapper int64 |
| `EntryPointNotFoundException: TkvFullBegin` | `.so` cu; chay lai cell 4 |
| `vectorized_gather_kernel index out of bounds` | shim cu; can `TkvFullSetIdsGen` |
| `TONG ≈ 500-600 ms` | lazy-load CUDA o lan dau; lay median 2 lan sau |
| Ty le ~7x bat thuong | kiem tra cell 6 co that su dat `CUDA_LAUNCH_BLOCKING=1` |

## 6. So lieu lich su

- CPU native fullstep @seq1024: `4016 ms` vs torch `4061.1 ms` (0,989x, 2026-11-11).
- GPU bridge 148 op: `~3286 ms/step` — da bi thay the boi fullstep.
- Nhat ky chi tiet: `docs/FULL_STEP_GPU_NOTES_2026-11-19.md`, `docs/HANDOFF.md`.
