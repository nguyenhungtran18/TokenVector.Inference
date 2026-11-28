# Benchmark / dieu tra: TokenVector vs PyTorch (GPT2-medium, Kaggle T4)

HEAD verify cuoi `5e00a4a` — 2026-11-28. 12 update moi ben, cung mot batch
512 token tinyshakespeare, cung mot nhiem vu `position t -> token t+1`.
Khong co row C++ bridge trong bang.

## Ket qua moi nhat (da xac minh)

| implementation | weights | median ms/step | speedup | loss truoc -> sau 12 update | step tang |
| :--- | :--- | ---: | ---: | :--- | ---: |
| PyTorch eager, AdamW lr 5e-5, all params | gpt2-medium pretrained | 451.0 | 1.00x | 3.6676 -> **0.1457** | - |
| TokenVector .tkv, AdamW lr 2e-4 constant | gpt2-medium pretrained | 219.3 | **2.06x** | 3.6676 -> 0.1693 | **0** |

Chuoi loss TKV (lr 2e-4, 13 gia tri = truoc update + sau moi update):
`3.6676 -> 3.4786 -> 2.9517 -> 2.6590 -> 2.3124 -> 1.8646 -> 1.5026 ->
1.0932 -> 0.7609 -> 0.5588 -> 0.3653 -> 0.2512 -> 0.1693` — moi drop > 0.

**Chua dat yeu cau "loss TKV <= PyTorch"**: 0.1693 vs 0.1457 (chenh 0.0236
nat). TKV thang toc do 2.06x va giam don dieu, nhung van thua loss. Khong
duoc phep cong bo nguoc lai.

`187.8 ms/step` cua C++ bridge la baseline lich su, chi tham chieu; khong
xuat hien trong bang so sanh.

## cac buoc da xac minh

1. **Exporter (root cause lon nhat)**: `qkv[:,0:H].T` va
   `attn.c_proj.weight.T` ghi ra `W^T`, trong khi cublas GEMM can `B[i,o]=W[i,o]`.
   Toan bo khoi attention dung sai trong so pretrained. Da bo `.T`.
   Kiem tra truc tiep tue blob: q/k/v/c_proj/c_fc/mlp_proj/head
   `maxerr=0.000e+00` (7/7 OK).
2. **Forward khop chinh xac**: torch step0 `3.6676`, TKV step1 `3.6676`
   (4 chu so thap phan). Truoc khi sua exporter la `9.0423` vs `8.7249`.
3. **Backward**: sua dW `attn.c_proj` ghi vao buffer dung chung `gWo` thay vi
   `AGo[ll]`; bias AdamW cap nhat ngay trong vong lap de moi lop dung gradient
   cua chinh no; buffer rieng cho `ln_f` (khong giay mau buffer cua lop cuoi).
4. **Gradient subnormal**: `dL/dlogits ~ 1e-6` nam sau vung subnormal fp16
   (~6e-8/step), nhan 1024x trong `loss_dx`. AdamW scale-invariant nen
   khong doi buoc cap nhat. Ket qua: loss 2.70 (`lr 2e-4` co warmup) -> 0.1693
   (constant lr), va khong con step nao tang.

## chan doan gradient con lai (step 1, HEAD 4c9e952)

So sanh 32 diem bang hook PyTorch vs dump trong .tkv:

| dai luong | corr |
| :--- | ---: |
| dL/dlogits @ Wh (dau vao ln_f) | +1.00000 |
| dL/d(hidden cuoi) sau ln_bwd | +0.99999 |
| gQ layer 23 | +0.99999 |
| gDX / gY1 layer 23 | +0.99999 / +0.99993 |
| gQ layer 16 / 8 / 0 | +0.733 / +0.420 / +0.223 |
| dWq layer 0 (256 phan tu) | +0.195 |

Loi nam o sai so tich luy qua cac block (khoi 23 dung, suy dan dan ve layer 0),
nhieu kha nang do cast fp16 o cac GEMM backward. Forward va 2 block dau
khong co loi. Day la ly do loss khong the san bang PyTorch hoan toan.

## Test noi dung da train (PyTorch, HEAD 5e00a4a)

```
BASELINE pretrained held-out CE = <xem output>
CE loss train batch   = 0.0466
CE loss held-out      = 6.7884
PROMPT 'ROMEO: ' -> ... (van la Shakespeare, khong mustanh)
```

Loss duoi 1 tren cungmot batch 512 token lap lai = **memorization**, khong
phai chat luong ngon ngu that. Held-out van cao hon pretrained; khong duoc
xay dung claim "AI hoc tieng Anh".

## Ghi chu

- `c_fwrite` khong co trong runtime; khong the luu weight cua TKV ra file.
  Step 3 hien test model PyTorch (file `trained_gpt2m.pt`).
- Khong co GPU/python local; moi so lieu tren deu tu Kaggle T4, Stop session
  truoc khi doc output theo yeu cau quota.
- Cac lan chay cu (loss 8.72 -> 6.37, 1.45, 0.3427) dung exporter/gradient sai
  va khong con dai dien cho ket qua hien tai.
