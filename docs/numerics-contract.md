# HOP DONG SO HOC — TokenVector

Ngay chot: **2026-10-17** · Trang thai: **DA DUYET (2026-10-17)**
Ap dung cho: moi thuat toan huan luyen LLM trong `TokenVector.Inference`
Lien quan: `docs/ROADMAP_TRAIN.md`

Muc dich: chot truoc cac quyet dinh so hoc, vi sau nay doi la phai sinh lai
fixture va sua lai moi kernel.

---

## 1. Nguyen tac

1. Kieu tinh mac dinh: `f32`
2. Tich luy: `f32`, thu tu co dinh
3. Cam: fast-math, doi thu tu phep cong, f64 "cho chac" o duong parity
4. Phep chia: IEEE-754 dung dau

## 2. Parity va GPU

**CHOT LAI 2026-10-18** (phe duyet boi chu khoa, bang chung:
`docs/accumulation-order-vs-pytorch-2026-10-18.md`, tao lai bang
`tools\verify_parity.ps1`). Muc cu truoc do la "CPU: bit-exact 100% voi
PyTorch" — do cho thay KHONG kha thi doi voi matmul/reduction (MKL sgemm
+ vectorized reduce, phu thuoc phien ban MKL/ISA), nen co pham vi ro rang:

| Nhom phep toan | Cam ket CPU |
|---|---|
| elementwise (add/mul/nhan), `exp`, `sqrt` | **bit-exact** voi PyTorch |
| matmul, mean/sum, reduction cua LN/RMSNorm | thu tu tich luy TUAN TU f32 co dinh, khong fast-math; do lech do duoc <= 1 ulp/ptu, ky vong cuoi duong <= 3e-07 (thuc te: forward 2.68e-07, grad 1.79e-07) |
| toan bo duong parity CPU | **deterministic**: cung input -> cung ket qua TREN MOI MAY (thu tu khong phu thuoc parallelization/MKL) |

- **Gate parity: dung sai 1e-5** (mac dinh `parity_tests`/`grad_tests`/
  `gpt2_tests`) ket hop lop **gradcheck sai phay doc lap** (khong phu thuoc
  fixture torch) de bat loi cong thuc.
- GPU: deterministic, khong TF32, rel <= 1e-5 (nguyen trang).

### 2.1 SUA DOI 2026-11-13 — Linux mac dinh dung OpenBLAS (matmul)

Theo quyet dinh do hieu nang (`benchmark/BENCHMARKS.md` §9.30–§9.33), tren
Linux/macOS GEMM **mac dinh di qua BLAS** (`libopenblas.so.0` hoac `libmkl_rt`),
so voi kernel AVX2 noi bo truoc day. Do la **ly do duy nhat** thay doi hop dong:

| | Truoc 2026-11-13 | Tu 2026-11-13 (Linux/macOS) |
|---|---|---|
| Matmul | kernel noi bo, thu tu co dinh | BLAS (OpenBLAS/MKL) |
| Bit-exact giua cac may Linux | Co | **Khong con** (thu tu tich luy phu thuoc BLAS) |
| Loss fixture | `10.760399` | `10.760399` (khong doi — test o §9.21/§9.22 chay tren duong native) |
| Do lech loss native vs BLAS | — | lech chu so thu 6 (`25.407898` vs `25.407887`), trong nguong gate 1e-5 |

- **Muon khoi phuc dung nghia cu (matmul kernel noi bo, deterministic tren moi
  may): dat `TKV_GEMM_BLAS=0`**. Windows khong doi (khong co DLL BLAS trong
  `demo/` nen van native; va duong `TKV_LINK_MKL` co kha nang doi).
- Hieu nang: BLAS + `TKV_GEMM_BLAS_THREADS=8` nhanh hon 13.4% o 28 core va 35% o
  2 vCPU so voi kernel noi bo.

## 3. Bang quyet dinh tung thanh phan

| Thanh phan | Chot | Ghi chu |
|---|---|---|
| GELU | erf chinh xac | tanh-approx lech 4.12e-04 |
| RMSNorm / LayerNorm | tru mean, chia var+eps, fp32 |
| Adam/AdamW | betas 0.9/0.999, eps 1e-8, weight decay tach |
| Init | normal(0, 0.02) theo thu tu co dinh |
| Softmax / CrossEntropy | giam theo thu tu co dinh |
| Dropout | RNG tu viet, co seed |
| Embedding backward | sap xep chi so truoc cong |
| Gradient accumulation | thu tu micro-batch co dinh |

## 4. Bau rieng GPT-2 HF (phai biet truoc)

| Van de | Chi tiet |
|---|---|
| Conv1D | luu `[in, out]`, nguoc chieu Linear |
| Scale attention | khong scale theo tang (`scale_attn_by_inverse_layer_idx = False`) |
| Dropout | `embd_pdrop=0.1`, `attn_pdrop=resid_pdrop=0.1` |
| Tokenizer | phai trung GPT-2 BPE (tiktoken / HF `gpt2`) |

## 5. Dropout — tach khoi parity

- Parity ban dau: dropout = 0 (che do eval HF)
- Parity co dropout: sau, doi hoi phat lai Philox
- RNG tu viet: chi de tai lap duoc

## 6. Trang thai do duoc

| Phep tinh | max_abs_diff | Ket qua |
|---|---|---|
| gelu(tanh) | 0.0 | bit-exact |
| cross_entropy(mean) | 0.0 | bit-exact |
| softmax | 1.86e-09 | dat |
| rmsnorm | 1.19e-07 | dat |
| gelu vs erf | 4.12e-04 | sai — da co `gelu_erf` |

Nguon: fixture `tv/tests/fixtures/` (torch 2.14.0+cpu, fp32, seed 0) va doi chieu `parity_tests.tkv`, `grad_tests.tkv`.
