# Benchmark: train GPT2-medium, TokenVector thuan vs PyTorch

**Timestamp:** 2026-11-28. Kaggle GPU Tesla T4, torch 2.11.0+cu130, CUDA 13.0.
HEAD verify cuoi `3524900` (trainer `35399d3`).

## So sanh CUNG SO STEP (150 step, moi ben deu 150 step that + 2 warm-up cho PyTorch)

| Implementation | median ms/step | Speedup | Train loss (dau -> cuoi) |
| :--- | ---: | ---: | :--- |
| PyTorch eager, lr 5e-5 | 460 - 508 | 1.00x | 6.0951 -> **0.0001** |
| TokenVector pure .tkv, lr 1e-4 + warmup | 210 - 216 | **~2.2x** | 8.7249 -> **0.3427** |

**Ket quan trung thuc:** TokenVector nhanh hon ~2.2x nhung **loss cuoi cao hon
PyTorch** (0.3427 so voi 0.0001) du cung so step. Yeu cau "TKV loss <= PyTorch
va nhanh hon" **chua dat duoc**. Khong duoc bao cao nguoc lai.

Khong co row C++ bridge trong bang (theo yeu cau). So lich su `187.8 ms/step`
(random-init 2026-11-26) chi de tham chieu.

## Tien trinh loss cua TokenVector (150 step)

Bat dau 8.7249 (HF cung batch: 9.0423), ket thuc 0.3427. Co 21 step tang nhe
trong 150 step (dao dong AdamW khi loss da rat nho). Cau hinh 10 step + warmup
cho chuoi **don dieu**: 8.7249 -> 8.1827 -> 7.6403 -> ... -> 6.5418.

## Luu y ve "loss duoi 1"

Loss 0.3427 tuong ung perplexity ~1.41 tren **chinh batch 512 token duoc lap lai
150 lan** — day la memorize, khong phan anh chat luong nguon van ban. Held-out
loss cua pretrained la ~8.97; con TKV chua do duoc tren weights da train.

## Da thua / chua thanh cong

- `c_fwrite` khong ton tai trong runtime (chi co `c_fopen/c_fread/c_fclose`).
  Thu dung `path_write_text` de luu 1.4 GB weights khong kha thi.
- Da thu tinh held-out loss ngay trong .tkv bang cach ghep `emb_held` + `yt`
  vao blob va eval o step 151: ket qua `HELDOUT 0.3427` nhung TKV thoat
  `rc=1` (CUDA illegal address 717, nhieu kernel fail) -> **khong tin duoc**,
  da revert ve ban chay on dinh.
- Da thu train them tied head weight (50M params): loss 0.3714 -> 0.3427,
  khong du de thua PyTorch.

## Bug da sua (da verify)

embedding that, nhan that, offset bias `ABQ`, huong weight Conv1D `[input,output]`
(exporter cat qkv theo cot + W1/W2 khong transpose: loss 21.22 -> 8.72),
gradient bias rieng + AdamW, backward ln_f, IL local <= 255, loss_init overflow,
PTX bias mul.wide.
