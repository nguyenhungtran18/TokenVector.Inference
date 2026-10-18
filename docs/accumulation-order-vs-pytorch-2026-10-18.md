# So sánh thứ tự tích luỹ fp32 — TokenVector vs PyTorch — 2026-10-18

Mục đích: kiểm chứng xem parity (forward + gradient GPT-2) có thể nâng lên
**bit-exact 100% với PyTorch** như `numerics-contract.md` §3 yêu cầu hay không,
và nếu không thì vì sao — kèm bằng chứng đo được, không suy diễn.

Môi trường đo: `torch 2.14.0+cpu`, **Intel oneAPI MKL 2026.1 + MKL-DNN v3.12**,
`CPU capability usage: AVX2` (xem `torch.__config__.show()`).
Tái lập: `python tools/probe_accumulation_order.py`.

## 1. Bảng kết quả (bit-diff = số phần tử khác DUY NHẤT bit, so với torch)

| Phep toan (torch) | Ung vien từ TokenVector | bit-diff | ket qua |
|---|---|---|---|
| `mm [8,64]@[64,256]` | tich luy tuan tu k=0..63, f32 | **1499 / 2048** (max 5.7e-06) | KHAC |
| `mm [8,64]@[64,256]` | FMA tuan tu (tich f64, round 1 lan) | 1618 / 2048 (max 7.6e-06) | KHAC (tham hon!) |
| `mm [8,64]@[64,256]` | 4 accumulator lien thap + tree | 1618 / 2048 | KHAC (giong f64sum) |
| `mm [8,64]@[64,256]` | BLAS cua numpy (khac MKL) | 1609 / 2048 | KHAC |
| `mm [4,4]@[4,4]` | tuan tu f32 | 8 / 16 | KHAC |
| `mean(sum)` 64 pt | tuan tu f32 | 1 khac | KHAC |
| `mean` 64 pt | f64-ruut | khac | KHAC |
| `mean` 64 pt | tree 8 lane (giong AVX2) | khac | KHAC |
| `exp` f32 | `exp(double)` roi round f32 | **0 / 6** | **KHOP** |
| `sqrt` f32 | IEEE sqrt | **0 / 64** | **KHOP** |
| `erf` f32 | libm double round f32 | 1 / 64 | gan khop (1 ulp thinh thoang) |
| `add`/`mul` elementwise | (fixture G1) | 0.0 | **KHOP** |

Doc bo tro: fixture G1 da do duoc `gelu(erf) 0.0`, `cross_entropy 0.0`,
`matmul fwd 6.0e-08` — dung voi bang tren (elementwise/structure khop,
matmul/sum lech vang ulp).

## 2. Phan tich

1. **MKL sgemm khong tich luy tuan tu.** sgemm nhieu accumulator
   FMA + packing + blocking theo k, thu tu phu thuoc MKL version va ISA
   (tren may AVX512 torch se dispatch AVX512 — khac nua). Khong the
   tai hien dung thu tu nay bang vong lap scalatru TokenVector ma khong
   "hack" theo tung phien ban MKL.
2. **Ung vien "dung hon" lai sai hon.** Tinh dung qua f64 roi round 1 lan
   (1618 khac) xa MKL hon ca tuan tu f32 (1499 khac) — vi MKL cung tich
   luy f32, chi khac thu tu/kenh. **Thu tu hien tai cua TokenVector da la
   ung vien tot nhat** trong cac ung vien test duoc.
3. **`mean`/`sum` giong matmul**: torch dung vectorized reduction
   (8/16 lane + horizontal add, FMA) — cung khong tich lai tuan tu.
4. **Nhom khop bit**: elementwise (add/mul), `sqrt` (IEEE), `exp`
   (torch cung dung double exp roi round ve f32 — giong .NET
   `Math.Exp`), gan dung cho `erf`.

## 3. Ket luan vs numerics-contract §3 ("CPU: bit-exact 100% voi PyTorch")

- **Kha thi bit-exact**: phep toan elementwise, `exp`, `sqrt`,
  cau truc CE/softmax sau khi cac dau vao da bit-exact.
- **KHONG kha thi bit-exact** voi matmul va moi reduction
  (mean/sum/layernorm stats) — chan ve MKL sgemm + vectorized reduce,
  phu thuoc phien ban MKL va dispatch CPU (AVX2/AVX512). Day la
  han che cua PyTorch (BLAS) chu khong phai bug cua TokenVector.
- **Do lech thuc te hien tai** (gpt2_tests, `run_cases` [8/8]):
  forward `logits 2.68e-07`, `loss 0.0`; grad `wte 1.79e-07`,
  `ln1_g0 3.3e-09`, `attn_w0 3.4e-08`, `mlp_w1_0 2.5e-08`,
  `lnf_g 9.3e-09` — tat ca 1-3 ulp, **<< tol 1e-5** va << 5.7e-06
  roundoff rieng cua 1 phep mm MKL.

## 4. De xuat sua contract (chot boi chu khoa, khong tu sua)

Gia tri bit-exact nen co pham vi ro rang thay vi 100% tuyet doi:

| Nhom | Cam ket |
|---|---|
| elementwise (add/mul/nhân), `exp`, `sqrt` | **bit-exact** voi PyTorch |
| matmul, mean/sum, layernorm stats | thu tu tich luy TUAN TU co dinh (f32, khong fast-math); dung sai do duoc: <= 1 ulp/ptu, ky vong cuoi-toi-toi 3e-07 << 1e-5 |
| toan bo duong parity CPU | deterministic: cung input -> cung ket qua TREN MOI MAY (thu tu khong phua thuoc parallelization/MKL) |

Yeu cau "bit-exact 100%" doi voi matmul nen doi thanh dung sai 1e-5
(mac dinh hien tai cua `grad_tests`/`parity_tests`), vi MKL khong cong
bo va khong dung thu tu tich luy.

## 5. Trang thai quyet dinh (2026-10-18)

**CHAP NHANH CO KIEM SOAT** — do lech 1-3 ulp so voi PyTorch la binh thuong
(va do chinh MKL/PyTorch gay ra), khong anh huong muc tieu chung minh thuat toan:
- Bug that (thieu residual, sai attention, thieu LN g/b) gay lech 5.3e-03 -> 0.284
  — lon hon roundoff ~10^5 lan; tolerance 1e-5 bat duoc bug that, bo qua roundoff.
- Determinism noi bo giu nguyen (thu tu tuan tu co dinh) — cac gate resume
  bit-exact/fixture tai lap van dat duoc, du khong khop torch tung bit.
- Khong doi code; khong doi thu tu tich luy (contract cam; doi se tham hon).

### Cong cu xac nhan lai ket qua: `tools/verify_parity.ps1` — DA XAY (2026-10-18)

Moi lan chot lai mot trong cac muc tren, can mot lenh chay duoc lap lai
thay vi nhac lai tung lenh rieng le. De xuat:

| Cap | Noi dung | Phat hien | Can gi |
|---|---|---|---|
| [1/3] Fixture oracle | sinh lai fixture vao thu muc tam bang `gen_fixtures.py` + `gen_gpt2_fixture.py`, **diff bit-level** voi ban da check-in | fixture drift, torch upgrade doi ket qua | python + torch/transformers |
| [2/3] Bang thu tu tich luy | chay `probe_accumulation_order.py`, so bit-diff voi baseline da chot (1499/2048, ...) | MKL/torch dispatch doi (AVX2->AVX512...), nghi van khi ket qua con lai | python + torch |
| [3/3] Gate parity .tkv | chay `run_cases.ps1`, doc max_abs tung case, doi chieu tolerance 1e-5 | regression code `.tkv` | khong can python |

- Exit code tuong minh (0 = toan bo xanh), bao cao `reports/parity-<ngay>.md`
  de luu vet; chi chay [1/3] khi co torch (neu khong thi bo qua co bao ro).
- **Lam rieng le** de `run_cases` van chay duoc o may khong co Python
  (nguyen tac hien tai giu nguyen).

```powershell
powershell -NoProfile -File tools\verify_parity.ps1
powershell -NoProfile -File tools\verify_parity.ps1 -SkipRegen   # bo qua [1/3]
```

Lan chay dau tien (2026-10-18): **TONG XANH, exit 0** —
100/100 fixture khop 100% voi ban sinh lai, bit-diff khop baseline (AVX2),
run_cases 26/26 + gpt2 10/10. Bo sung kem:
`tools/compare_fixtures.py` (diff 2 thu muc fixture o moi dong — 9 chu so
nghia trong = bit-exact cho f32), 2 generator nhan `--out <thu muc>`.

GHI CHU KY THUAT (da gap khi xay): `.ps1` PHAI thuần ASCII (PS 5.1 doc
UTF-8 khong BOM thanh cp1252, byte 0x94 cua dau "—" tro thanh ngoac thong
minh `"` ma PowerShell coi la delimiter -> loi parser); fence markdown khong
duoc de tay `"`n````" (backtick thu 3 escape dau nhay dong) — dung bien.

## 6. Cac cap nhat tu doi nay

- Khong doi thu tu tich luy trong code (contract cam doi thu tu; do cho
  thay hien tai da la ung vien tot nhat — doi se THAM hon).
- `gpt2_tests` van 10/10 tol 1e-5; khong gate nao thay doi.
