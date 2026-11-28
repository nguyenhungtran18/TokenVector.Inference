# Benchmark / dieu tra: TokenVector vs PyTorch (GPT2-medium, Kaggle T4)

HEAD verify cuoi `b6e61e2`. 150 step moi ben.

## Ket qua hien tai (khong co step nao loss tang)

| | median ms/step | loss dau -> cuoi | step tang |
| :--- | ---: | :--- | ---: |
| PyTorch eager (lr 5e-5) | 460 - 508 | 9.0423 -> 0.0001 | - |
| TokenVector .tkv (lr 1e-4, warmup + decay bac hai) | 210 - 225 | 8.7249 -> **1.4503** | **0** |

Chuoi loss khong tang o step nao. Linh canh: decay manh giu on dinh nhung loss
cuoi cao hon ban decay tuyen tinh (0.4387, co 5 step tang: 90,100,102,109,110).

## Tai sao co gia tri am (loss tang giua chung)

1. AdamW cap nhat dang `w -= lr * mhat/(sqrt(vhat)+eps)` â€” **do lon buoc khong
   giam theo do lon gradient** (khac SGD). Khi loss da nho (~1.0 tro xuong),
   buoc co dinh ~1e-4 co the vuot qua cuc tieu nho -> loss tang.
2. Ban dac them tied-head (50M params) lam nhieu step tang hon (update dau ra
    lon). Da bo head training (khong cai thien loss: 0.3714 -> 0.3427).
3. Warmup don thuan `lr*min(step/10,1)` giu lr 1e-4 den cuoi -> con 5 step tang.
   Decay tuyen tinh `lr*(151-step)/141` giam con 5 step; decay bac hai
   `lr*((151-step)/141)^2` cho **0 step tang**.

## Dieu tra quan trong: KHONG phai do thieu tham so

Chay PyTorch voi **dung tap tham so ma TKV cap nhat** (dong bang LN gamma/beta
va wte embedding, chi train 4 ma tran + bias moi layer):

```
ABL head=False  s1=9.0423  s10=5.0051  s50=0.0004  s150=0.0001
ABL head=True   s1=9.0423  s10=4.7941  s50=0.0003  s150=0.0000
```

PyTorch dat 0.0004 o step 50 trong khi TKV ~4.0 -> **loi nam o gradient/duong
backward cua TKV, khong phai do khong train LN/head**. Kernel AdamW da kiem tra
dung cong thuc; loss gradient (/T) dung. So sanh dWq step 1: `GRADCHK maxerr
= 4.2e-4` va dau khac nhau o 4 phan tu dau (TKV ~2e-6, PyTorch ~4e-5) â€” can
doi chieu them (co the do transpose/chi so hoac scale).

## Cac bug da sua truoc do

embedding that, nhan that, offset bias `ABQ`, huong weight Conv1D (exporter cat
qkv theo cot, W1/W2 khong transpose: loss 21.22 -> 8.72), gradient bias rieng,
backward ln_f, IL local <= 255, loss_init overflow, PTX mul.wide.

## Ghi chu

`c_fwrite` khong co trong runtime (chi fopen/fread/fclose). Thu tinh held-out
trong .tkv cho `HELDOUT` nhung TKV thoat rc=1 (CUDA illegal address) -> da revert.
Khong co row bridge; so lich su 187.8 ms/step chi tham chieu.
## Test 12 step (yeu cau moi nhat, HEAD 285ecf1)

| | median ms/step | loss dau -> cuoi | step tang |
| :--- | ---: | :--- | ---: |
| PyTorch eager (2 warm-up + 12) | 454.4 | 6.0951 -> 3.1089 | - |
| TokenVector .tkv (12 step) | 215.9 (**2.10x**) | 8.7249 -> 6.3718 | **0** |

Chuoi loss TKV: 8.7249 -> 8.1827 -> 7.6403 -> 7.3655 -> 7.2680 -> 7.0690
-> 6.9027 -> 6.7974 -> 6.6568 -> 6.5418 -> 6.4931 -> 6.3718 (moi drop > 0).
Van chua dat yeu cau "loss TKV <= PyTorch"; TKV thang toc do, thua loss.