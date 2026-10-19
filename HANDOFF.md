# HANDOFF — TokenVector.Inference (Pure .TKV) — 2026-10-19

Verified: compiler build 23:58 (2026-10-17) — B-01..B-07 fixed + SIMD GEMM updates.
Repo HEAD: `b28bdb7` (registry kernel + numerics-contract); G3 tokenizer `8ac487d`, G3 pack `d8a203a`.
Not staged: `COMPILER-NOTES-2026-10-16.md` (official compiler team update, not ours).

**Luu y:** muc 1..6 ben duoi la snapshot 2026-10-17 (truoc G2/G3). Trang thai
hien tai cua G3 nam o muc 0 ngay duoi day.

---

## 0. G3 STATUS (2026-10-19) — PACKING + SHARD XANH 6/6

Gate: `tv/tests/pack_tests.tkv` (reference: `tools/gen_pack_fixture.py`, fixture
`tv/tests/fixtures/pack_cases.txt`). Chay:

```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/pack_tests.tkv %TEMP%/pack_tests.exe main
%TEMP%/pack_tests.exe
```

Ket qua THAT (2026-10-19): **cases=6 fails=0**

| Gate | Ket qua |
|---|---|
| [1] packing 8 seq x 32 (seqs + doc_ids tung phan) | PASS |
| [1b] cat tail 12 pair -> 2 pair, token nguyen ven (regression guard bug #2) | PASS |
| [2] masks: block_seq0 32x32 flat + causal 4x4 | PASS |
| [3] shard write/read: 16 shard, 256 token roundtrip | PASS |
| [4] stream seed42: 16 sample + checksum 1184745045 | PASS |
| [5] determinism: cung seed -> cung checksum; 42/43 khac thu tu | PASS |

### 0.1 Ba nguyen nhan da tim ra va sua (that, khong doan)

1. **`pack_sequences` tao list phang THU HAI (`pk2`) de cat tail** -> list thu 2
   trong cung ham bi cap backing `i32[]` (bug compiler #2, muc 0.2) -> token corrupt
   (1,2,3 -> 0,1,0) nhung CHI khi co tail bi cat. Fixture 8 chunk day du nen
   gate [1] van xanh, che mat loi. Sua: dem tong pair truoc, chi append phan
   day du vao MOT list duy nhat. Da them gate [1b] co tail le lam regression
   guard cho dung bug nay (fixture khong bao gio cat tail).
2. **Test [3] dung sai `flat`**: sau khi doi sang interleaved pairs, vong lap cu
   lay `pk[0], pk[64], ...` (1 token/chunk) -> 8 token -> `shard_write n=1`.
   Sua: them `flat_tokens(pk)` trong `tv/data/dataset.tkv` (mot list phang,
   return truc tiep) tra token cua toan bo chunk day du.
3. **Test [2] tai su dung bien `t0`**: `t0 = now_ms()` o dau `main` khien `t0`
   bi suy `f32`; sau do `t0 = len(_d0)` (=32) thanh `32.0` -> so sanh mask
   fail du gia tri mask DUNG (probe: bad=0). Sua: doi bien timing thanh
   `t_start`; them canh bao: KHONG gan i64 vao bien da suy f32.

### 0.2 Bug compiler list (ban day du — HANDOFF tung bi revert mat muc 5.7)

- **#1 nested `list[list[i64]]` qua return/field**: noi dung thanh rac dia chi
  stack hoac crash `ArrayTypeMismatchException`; cung code voi nested locals
  thi DUNG. Workaround: moi boundary chi qua `list[i64]` PHANG.
- **#2 list phang THU HAI trong cung ham** bi cap backing `i32[]` (doc dung,
  doc i64 ra `4294967297 = 1|1<<32`). Workaround: moi ham chi MOT list phang,
  return truc tiep.
- **#3 nested literal append**: `docs.append([1, 2, 3])` voi `docs: list[list[i64]]`
  -> phan tu thanh rac dia chi (`55454752`), khong crash. Cung du lieu qua ham
  tra `list[i64]` (`docs.append(mk(...))`) thi DUNG. Repro toi thieu:
  `docs = []` / `docs.append([1,2,3])` / `docs.append([4])` / in `docs`.
  Workaround: moi list con phai qua mot ham return `list[i64]`.
- **#4 gan i64 vao bien da suy f32**: `x = now_ms()` roi `x = len(lst)` -> x thanh
  `32.0`; dung x lam index/so sanh cho ket qua sai lang le. Khong co chan o
  compile-time. Workaround: bien rieng cho timing.

### 0.3 G4 TRAINER (2026-10-19) — XANH 15/15

File: `tv/training/trainer.tkv` (+ mirror `tv/tests/` & `tv/tools/`),
gate `tv/tests/trainer_tests.tkv`, reference `tools/gen_trainer_fixture.py`,
fixture `tv/tests/fixtures/train_cases.txt`.

```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/trainer_tests.tkv %TEMP%/trainer_tests.exe main
%TEMP%/trainer_tests.exe
```

Ket qua THAT (cases=15 fails=0), model MLP 4-8-3, 20 mau, 400 buoc:

| Gate | Ket qua |
|---|---|
| 1 buoc: loss / grad norm / lr / trong so | 1.19e-07 / **0.0** / **0.0** / 2.98e-08 vs PyTorch |
| lich lr warmup+cosine 400 buoc | khop tung buoc (0.0) |
| clip norm: vuot nguong / duoi nguong | 0.0 / 0.0 |
| overfit 20 mau: loss tung buoc / trong so cuoi | 2.38e-06 / 1.91e-06 vs PyTorch |
| overfit ket qua | eval_loss **0.01495**, acc **1.000**, ppl 1.0151 |
| resume bit-exact 3 + save/load + 3 | **bit-for-bit** bang 6 buoc lien (p, m, v, t, RNG) |
| resume vs PyTorch sau 6 buoc | 3.58e-07 |
| throughput | ~8.5k buoc/s (47 ms cho 400 buoc, may nay) |

Khac biet con lai voi PyTorch la **2e-6** (khong bit-exact tuyet doi): PyTorch
tinh norm/clip va cap nhat Adam trong f32 theo thu tu khac o vai buoc lam tron.
Gate ROADMAP ghi "bit-exact" — thuc te do duoc la 2.4e-6 tren loss tung buoc;
**resume trong noi bo .tkv thi bit-exact tuyet doi**.

Checkpoint: text, tham so/m/v ghi bang **bit IEEE-754 f32** (khong phai so thap
phan) + `t` + `draws/state` + `perm` + sieu tham so -> load lai bit-for-bit.

Lenh CLI: `train --steps N --lr X --samples N --accum K --seed S --log-every K
--out ck.txt` trong `tv/tools/cli.tkv` (demo MLP tren du lieu tong hop LCG).
Smoke test 100 buoc: loss 1.133 -> 0.143, eval 0.2236, acc 0.95, checkpoint
2.3 KB, 31 ms.

### 0.4 Bug compiler moi (tiep 0.2)

- **#5 gan lai field trong method roi doc ngay**: `self.perm = []` (hoac bat ky
  list moi) trong `shuffle()` xong doc `self.perm[...]` ngay sau do -> doc
  **tham chieu CU** (index dau tien sau khi xao lay tu perm cu). Workaround:
  sua TAI CHO (`self.perm[k] = k`) thay vi gan lai field.
- **#6 constructor khong export qua module**: file khac `import` module roi goi
  `AdamW(...)` / `Sampler(...)` -> `il_codegen: ham 'AdamW' khong ton tai`.
  Workaround: factory function `new_adamw(...)`, `new_sampler(...)`.
- Nho lai: import tu `tv/tests/*.tkv` chi thay module khi co mirror
  `tv/tests/tv.<duong.dan>.tkv` (va tuong tu cho `tv/tools/`).
- **#7 hai bien cung ten trong 2 nhánh if cung ham, khac kieu**: trong
  `gpt2_backward`, `gg` la **Tensor** o nhánh `simd==1` nhung **f64** o nhánh
  `simd==0` (2 bien gop 1 slot) -> toan bo 28 grad hong 1e12–1e15, GIA TRI THAY
  DOI giua 2 lan chay (khong deterministic). Trace bang cach tach tung vung ra
  probe thi moi vung deu DUNG — chi ro khi chay toan bo. Fix: doi ten bien
  (`gsc`). Cung ten o 2 nhánh ma CUNG kieu (mxf/tot/e/inv_tot/inv_n) thi an toan.
  Quy tac: ten bien rieng per nhánh khi kieu khac nhau.
- **#8 `now_ms()` (f64) gan vao local trong HAM roi doc trong phep tru**: doc ra
  ≈0 / gia tri absolute dong ho (probe `simd_probe_w.tkv`: main delta=1453ms DUNG,
  func delta=37854359 = absolute). Hau qua: ham tra tong thoi gian cho ra so
  vo nghia (37547187ms / 9ms). Workaround: **toan bo timing dat trong `main`**
  (pattern gate [8]/[9]), chi nhan ket qua ve. `now_ms()` trong main binh thuong.

### 0.5 G5 HARNESS + SEAM BACKEND (2026-10-19) — XANH 4/4

File: `tv/training/backend.tkv` (seam), `tv/training/parity_harness.tkv`
(bo test doi chieu), `tv/tests/harness_tests.tkv`, wrapper
`tools/verify_train_parity.ps1`.

**MOT lenh:**

```
powershell -NoProfile -File tools\verify_train_parity.ps1
```

Ket qua THAT (exit 0, bao cao o `reports/train-parity-<stamp>.md`):

```
=== G5 parity harness (backend=cpu) ===
  [PASS] gelu_erf        max_abs_diff=1.192e-07
  [PASS] gelu_tanh       max_abs_diff=0.0
  [PASS] softmax_row     max_abs_diff=1.863e-09
  [PASS] rmsnorm         max_abs_diff=1.192e-07
  [PASS] cross_entropy   max_abs_diff=0.0
=== 5/5 khop (100.0%) ===
  [PASS] cpu: 5/5 khop (100%)
  [PASS] bat backend gia (tanh thay erf): dung 1/5 case
  [PASS] bat backend gia (rmsnorm /(n-1)): dung 1/5 case
  [PASS] bat backend gia (+1e-3 moi output): 5/5 case
=== G5 harness cases=4 fails=0 ===
```

Backend gia chung minh harness THAT SU kiem tra (khong phai luon xanh) va bat
**dung** case: dung tanh thay erf chi lam hong `gelu_erf`; rmsnorm chia `n-1`
chi lam hong `rmsnorm`; backend "gan dung" +1e-3 bi bat ca 5 case.

Seam: `backend.tkv` dispatch theo `kind` (0 = cpu reference; 1.. = backend gia
chi de test). Them backend moi (SIMD/GPU/WASM) = them module + nhanh `kind`,
khong sua `ops.tkv`/`trainer.tkv`/harness. Fixture van la PyTorch check-in
(`tools/gen_fixtures.py`).

Con lai cua G5 (khong thuoc gate, chua lam): notebook Colab de chay lai tren
GPU khi co may.

### 0.6 G2 PHAM VI BO SUNG (2026-10-19) — XANH 9/9

File: `tv/training/gpt2_ref.tkv` (them `gpt2_init`, duong padding mask),
gate `tv/tests/gpt2_scope_tests.tkv`, fixture `g2_m_*` sinh boi
`tools/gen_gpt2_fixture.py` (HF gpt2 thu nho + `attention_mask`).

Ket qua THAT (cases=9 fails=0):

| Gate | Ket qua |
|---|---|
| `gpt2_init(7)` hai lan | bit-for-bit giong nhau; seed 8 khac |
| `gpt2_init` thong ke 16384 mau | mean=0.00012, std=0.02009 (dung ~0 / ~0.02) |
| mask: hidden h0/h1/h2 | 0.0 / 2.2e-08 / 3.9e-08 vs HF |
| mask: sau ln_f / logits / loss | 1.2e-06 / 2.7e-07 / **0.0** |
| invariant: them 2 pad o CUOI khong doi 6 dong dau | 0.0 |

`gpt2_tests` cu van **10/10** (khong hoi quy sau khi sua gpt2_ref).

**Chua lam duoc — BLOCKER 124M (so do that, khong phai uoc luong suong):**

1. Fixture: `wte` cua 124M = 50257x768 = 38.6M phan tu ~ **425 MB text**; m/v
   cua optimizer them ~850 MB -> khong the check-in theo quy uoc repo.
2. Toc do: forward GPT-2 nho (2 lop/64/seq 8) do duoc **3.75 ms** (200 lan, may
   nay). Cung thuat toan o 124M (12 lop/768/vocab 50257) nang hon ~**1266x** theo
   so phep nhan -> ~0.6 s/token, ~4.8 s cho 1 forward 8 token, ~15 s cho 1 buoc
   train (chua tinh backward). Mot gate parity 10 buoc se mat vai phut.
3. Bit-exact voi HF khong the dat khi GELU dung xap xi A&S `erf`
   (docs/numerics-contract.md): sai so ~1.5e-7 luon lon hon 0.

=> Ket luan: giu ban thu nho lam bang chung thuat toan (dung tinh than "chung
minh thuat toan" cua ROADMAP); muon 124M phai co che do fixture ngoai repo +
thoi gian do tren may khoac. So do tren ghi lai de quyet dinh sau, khong che
giau.

### 0.7 G4 x G2 — TRAIN THAT TREN GPT-2 (2026-10-19) — XANH 5/5

File: `tv/training/gpt2_train.tkv` (cau noi), gate `tv/tests/gpt2_train_tests.tkv`.
Trainer `trainer.tkv` duoc tong quat hoa: `AdamW.groups` + `new_adamw_g(...)` de
clip grad norm theo TUNG tensor (giong PyTorch), thay vi cung layout MLP.

Ket qua THAT (cases=5 fails=0), GPT-2 nho **118.528 tham so / 28 tensor**:

| Gate | Ket qua |
|---|---|
| so tham so + 28 nhom grad norm | n=118528, tong nhom = n |
| overfit 1 chuoi 12 buoc | loss **5.083 -> 0.500** (375 ms cho 12 buoc) |
| loss theo moc 1/6/12 | 5.083 / 2.390 / 0.500 (giam deu) |
| resume bit-exact | 3 + save/load + 3 == 6 lien, **tung bit** tren 118.528 tham so |
| loss sau resume | giong chay lien |

Khong hoi quy: `trainer_tests` 15/15, `gpt2_tests` 10/10, `gpt2_scope_tests`
9/9, CLI compile OK, harness 5/5 (100%).

### 0.8 G1 — DONG NOT 2 MUC CON THIEU (2026-10-19) — XANH 8/8

| Muc | File | Ket qua |
|---|---|---|
| `swiglu` fwd/bwd | `tv/training/ops.tkv` (`op_swiglu_fwd/bwd`) | fwd 5.96e-08, bwd 2.38e-07 vs PyTorch; gradcheck sai phan 1.97e-04 |
| phat hien in-place nguy hiem | `tv/training/inplace_guard.tkv` | 5/5: bat ghi de khi backward con can, cho qua sau release, khong bao dong sai |

Fixture swiglu: `tools/gen_swiglu_fixture.py` -> `tv/tests/fixtures/g_sw_*.txt`.
Gate: `tv/tests/swiglu_tests.tkv` (3) + `tv/tests/inplace_tests.tkv` (5).

Luu y ve guard: day la lop API (`guard_consume/release/write/check`) phai duoc
GOI tu op/tape; chua tu dong chen vao moi op.

### 0.9 G2 124M — CHE DO FIXTURE NGOAI REPO (2026-10-19) — XANH 20/20

Mo khoa BLOCKER 124M cua 0.6. Mot lenh:

```
powershell -NoProfile -File tools\verify_gpt2_124m.ps1
```

4 buoc: (1) sinh/sinh lai fixture `gen_gpt2_fixture.py --full --out
tv\tests\fixtures-124m` (~2.3 GB text, .gitignore — khong len GitHub duoc vi
shard 119-231MB vuot 100MB cua GitHub khi khong co LFS), (2) ghi pointer
`tv/tests/g2_full_dir.txt`, (3) bien dich + chay gate `gpt2_full_tests.tkv`,
(4) chong hoi quy `gpt2_tests`. Exit: 0 = khong FAIL (SKIP duoc phep). Bao cao
`reports/gpt2-124m-<stamp>.md`.

Ket qua THAT (19-10-2026, 2m49s, exit=0):

| Case | Ket qua |
|---|---|
| hidden h0 (embed) | **0.0** |
| hidden h1..h12 (12 tang) | 6.1e-07 -> **2.1e-06** (tich luy 12 tang) |
| logits | 5.56e-06 (tol 1e-4) |
| loss | 9.5e-07 (tol 1e-5) |
| grad wte (tie) | 4.11e-06; cac grad khac <= 6.1e-07 |

3 luong do THAT (khong uoc luong):
1. **argv() khong hoat dong** tren build tkvc hien tai (ca cli.exe cung in
   usage voi arg) -> gate doc fixture dir qua **file pointer**
   `tv/tests/g2_full_dir.txt` (chu MISSING -> SKIP, exit 0).
2. **Import resolve ve flattened copy**: `tv/tests/tv.training.gpt2_ref.tkv`
   (256/64) thang, canonical `tv/training/gpt2_ref.tkv` (124M) KHONG duoc
   chon; module chi co ban canonical khong resolve. -> 124M can flattened copy
   rieng: `tv.training.gpt2_ref124` (copy 2 ban, giong le quy uoc copy).
3. **Nguong OOM .NET cua text file**: 533 MB split duoc (26s), 594 MB bi
   OutOfMemoryException sau 4.9s (UTF-16 ~2x + LOH). -> generator `--full`
   **shard 15M gia tri/man** (`g2_shards.txt` manifest, file `*.0/*.1/*.2`),
   gate doc manifest va gop. Parse wte 38.6M phan tu = 26.3s do (P4).

Guard generator: `--full` vao thu muc trong repo -> loi exit 2 (khong de
fixture 2.3 GB lo vao git); thieu transformers -> exit 2. Khong `--full` van
la ban thu nho 256/64 (verify_parity.ps1 khong bi anh huong).

Files: `tools/gen_gpt2_fixture.py` (`--full` + `--out` + shard),
`tv/tests/gpt2_full_tests.tkv` (SKIP/doc pointer/shard loader), flattened
`tv.training.gpt2_ref124` (2 ban copy),
`tools/verify_gpt2_124m.ps1`. Khong hoi quy: 6/6 gate nho xanh.

### 0.10 Tiep theo

G1-G5 xong gom ca parity 124M ngoai repo (0.9). SIMD train da xong (0.11).
G5 thieu notebook Colab; G6 (phan tan) nam ngoai pham vi theo ROADMAP. G4 da
noi vao GPT-2 nho (0.7).

### 0.11 SIMD TRAIN (2026-10-19) — XANH 20/20

Muc 8 ROADMAP ("dung SIMD de tang toc") — da dua SIMD vao duong train, opt-in.

**File**
- `tv/training/simd_matmul.tkv` (+ 2 mirror): `smm_fwd(a,b,m,n,k)` — B luu
  `[k hang][n cot]` (duong chuan A×B, cung layout `gpt2_linear`); `smm_dot(a,b,m,n,k)`
  — B luu `[n hang][k cot]` (C[i,j]=dot dong, dong y nghia `op_matmul_fwd`);
  `smm_transpose(t,rows,cols)`.
- Boc native `simd_gemm` (AVX2+FMA, `tkv_simd_native.dll` kep exe): chi nhan ma
  TRAN VUONG T×T -> chia o (T = roundup64(k), toi da 1024 -> tile ≤ 4MB), bo sung
  0 o le, moi o cong vao C ngay (simd_gemm GAN, khong tich luy).
- `gpt2_ref.tkv` (canonical + 4 mirror) + `gpt2_train.tkv`: tham so
  `simd: 'i32' = 0` — **mac dinh scalar, moi gate cu nguyen ven**; truyen 1 de
  bat SIMD. Linear/attention/mlp/block/forward/backward deu noi qua.

**Gate** `tv/tests/simd_matmul_tests.tkv` — **20/20 PASS**:

| Nhom | Ket qua |
|---|---|
| smm_dot/fwd vs scalar (5x7x11, 64³, 70x130x300, 40x96x250) | PASS ≤4.3e-6 |
| smm_transpose + roundtrip (16x33, 64x64) | PASS 0.0 (bit-exact) |
| gpt2_linear fwd+bwd (32x64→192) simd vs scalar | PASS 9.5e-7 |
| forward logits + loss (nho + mask) | PASS 2.4e-7 / 0.0 |
| gpt2_backward 28 grad simd vs scalar | PASS ≤1e-4 |
| train 3 buoc (loss buoc 1 + buoc 3 tich luy AdamW) | PASS ≤1e-4 |
| SPEED linear 256x768x2304 | scalar 3390 ms → simd 63 ms (**~53×**) |
| SPEED smm_dot head-like 64x768x4096 | 641 → 94 ms (~6.7×) |

Hoi quy: `gpt2_train_tests` 5/5, `gpt2_tests` 10/10, `gpt2_scope_tests` 9/9,
`gpt2_full_tests` 124M **20/20**.

**2 bug tim duoc do THAT (probe, khong doan)** — xem 0.4 bug #7:

1. **Layout B**: `smm_dot` doc `b[j*k+p]` (B [n][k]) — 3 call site truyen B
   dang [k][n] (x cua linear_bwd, wte cua dxf/dwte) -> cu SAI 9.478449 lap lai
   moi chuoi test, dung mo phong tung moc deu PASS (vi ref cung doc sai theo).
   Fix: dung `smm_fwd` cho 3 cho do (`dwT`, `dxf`, `dwte`) — 2 cho con lai
   (logits, dx) dung `smm_dot` la DUNG vi B tuong ung co layout [n][k].
2. **Bug compiler #7** (0.4): bien `gg` Tensor/f64 trung ten 2 nhánh -> hong
   1e12–1e15 toan 28 grad, fix doi ten `gsc`. Tach probe (head, block, linear
   tung vung) thi PASS het — chi toan bo chay moi hong.

**Ky luat so**: simd khong bit-exact (FMA + thu tu tich luy), do lech do
~3e-6 (512², ±1) -> chot dung sai **1e-4** (cung nguong parity 124M). Doc file
`simd_matmul.tkv` truoc khi sua (ky luat tile + kieu list[f32] cho simd_gemm).

Chay gate:

```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/simd_matmul_tests.tkv %TEMP%/simd_matmul_tests.exe main
%TEMP%/simd_matmul_tests.exe
```

Probe tim loi con lai (simd_probe*.tkv o goc va tv/tests/ — ghi nhan phan tich,
khong phai gate).

### 0.12 BENCH SIMD vs PYTORCH (2026-10-19)

Bao cao day du: **`reports/bench-simd-vs-pytorch-20261019.md`**. So chinh
(Xeon E5-2695 v3 14C, torch 2.14+cpu 14 threads, fp32, cung AdamW contract):

| Tier | PyTorch | tkv SIMD | tkv scalar |
|---|---|---|---|
| A linear 768→2304 x256 tok (fwd+bwd) | **5.6 ms** | 323 ms (49× scalar) | 15 800 ms |
| B GPT-2 nho seq8 train step | **8.8 ms** | 32.1 ms | 25.8 ms |
| C GPT-2 **124M** seq8 train step | **999.8 ms** | 31 333 ms | 52 578 ms |

- Params 124M 2 ben = 124439808, loss dau scalar=simd=10.674946 — cau hinh thuc.
- **SIMD 49× tren GEMM lon, nhung 1.7× thuc te tren 1 buoc 124M** va **CHAM hon
  scalar o model nho** (tile copy per-element + padding 8/64 hang nuot loi).
- **TRANG THAI P0/P1 (cung ngay, sau doan tren)**:
  - **P0 XONG** — `simd_matmul.tkv` (Ct reuse + cache A-tile i→p→j + T theo
    min(m,n)): tier A 321.9→**262.5 ms**, B 31.3→**21.8 ms** (SIMD vuot scalar
    26.6), C 31.3s→**24.2-24.5 s/buoc**. Gate simd_matmul 20/20 + hoi quy 4/4
    (gpt2_train 5/5, gpt2 10/10, scope 9/9, full 20/20).
  - **P1.2 fused step+unflatten: DA PROBE + REVERT** — probe `simd_probe_ah`
    (best-of-5 tren 124M): fused moi hinh thuc deu CHAM hon 2 vong rieng
    (4704-7547ms vs 4516ms). Vao luc do bat duoc 2 bug: for-range sót
    `jj+1` = skip nua phan tu (loss sai), while-counter local = TkvInt
    boxed (~+1s). `gpt2_train_step` van 2 vong; khong lam lai.
  - **P1.4 f32 AdamW: da probe (`simd_probe_ae`) → lu** (m/v 2.9× nhung
    p-loop sqrt tu choi P6).
  - **TIEP THEO: P1.1** bwd ghi thẳng opt.g (~−2.5s, can probe truoc) + P1.3
    `gc_collect` (builtins co that — compare_rivals goi; REFERENCE chua liet ke).
  - `t_zeros`/zero-init: list-multiplication `[0.0]*n` CHAM 9× append (probe
    AF) — giu append; `reserve` co trong REFERENCE 2.13 (chua dung).
  - So cu + 2 buoc toi uu ban dau (chua lam): cache/reuse tile, vectorize
    AdamW + grad-copy (498M cham/step) — doc bao cao + plan.
- Timing trong tkv PHAI o `main` (bug #8, xem 0.4).
- **Nguyen nhan co so (do duoc) + plan uu tien: `docs/perf-plan-simd-vs-pytorch.md`**
  — cham that = dung tile (probe X: build 43.8ms vs gemm 7.8ms/fwd), BWD 63%
  buoc 124M, plumbing flat ~27%, AdamW 5s scalar. Moc: P0 cat dung tile
  (simd_matmul), P1 bo grad materialization + fuse unflatten + f32 AdamW,
  P2 floor scalar (phu thuoc compiler team), P3 do/sua so luong simd_gemm,
  P4 bench lai seq1024. Ky vong trung thuc: 31s -> ~19-21s sau P0+P1.

File: `tools/bench_torch.py` (torch A/B/C), `tv/tests/simd_bench.tkv` (nho),
`tv/tests/simd_bench_124m.tkv` (124M — import `gpt2_ref124` + `trainer`, khong
import gpt2_train vi gpt2_fwd_ps gan voi ban nho).

---

## 1. STATUS (snapshot 2026-10-17): FULLY VERIFIED ACROSS ALL GATES

| Gate | Command / Path | Result |
|---|---|---|
| Inference tests | `tools\run_cases.ps1` (26 cases) | PASS 26/26 |
| Gap tests (xUnit mirror) | `gap_tests.tkv` (16/16) | PASS |
| Build pipeline | `build_inference.ps1` | Completed |
| Module import (all 48) | module import probe | 48/48 import OK |
| Module compile (all 48) | `--verify-all` sweep | 55/55 PASS |
| Function body verify (all callable) | enhanced function probe | 366/366 compiled OK |
| Function body verify (library modules) | `funprobe3.py` enhanced (adapted) | 46/48 clean (2 false alarm harness artifacts, not source errors) |
| Full pipeline (all 5 entries) | `build_inference.ps1` full + benchmark | Completed; benchmark real timing restored (B-05 fixed) |
| Parity (forward) vs PyTorch fixtures | `parity_tests.tkv` ([4/4] in run_cases) | 4/4 PASS |
| Gradient parity vs PyTorch autograd | `grad_tests.tkv` ([5/5] in run_cases) | 20/20 PASS |
| Gradient chain (tape backward on multi-layer) | `tape_tests.tkv` ([6/6] in run_cases) | 8/8 PASS (6-layer: rms_norm*w1, gelu_erf, silu, loss, gx, gw1, gb1) |
| Gradient corruption catch (sensitivity) | `gc_tests.tkv` ([7/7] in run_cases) | 7/7 PASS: detects deliberate corruption (0.045) + 1% perturbation (0.0007) |

---

## 2. WHAT CHANGED IN REPO (G1 + fixtures + docs)

### Source edits (repo canonical, 3 files edited for compiler changes only):
`tv/benchmarks/compare_rivals.tkv`: removed `now_ms()`/`now_us()`/`gc_*` stubs (new compiler adds them); `gc_collect()` now called as bare statement (B-05 fixed); added second `gc_collect()` before `used` read for Zero-GC gate.
`tv/tests/inference_tests.tkv`: removed `gc_*` stubs.
`tv/tools/cli.tkv`: removed `now_us()` stub.

### Source additions (G1 core):
`tv/training/*.tkv`: 5 canonical files (`tensor.tkv` 165 lines; `ops.tkv` 321; `tape.tkv` 426; `kernels_reg.tkv` 47; `gpt2_ref.tkv` 83 lines — G2 skeleton).
`tools/gen_fixtures.py`: 185 lines (fixture generator from torch autograd, fp32, seed 0).
`docs/linux-portability-spike.md`: WSL/mono test results.
`docs/G1_RESULT_STATE.md`: G1 verification results.
`docs/numerics-contract.md`: updated from proposal to verified (DUYET).
`docs/ROADMAP_TRAIN.md`: updated status + next steps.
`ROADMAP_TRAIN.md`: included in commit message.
`HANDOFF.md`: updated (section 4.1 closed list, 4.2 fixed, 4.3 workarounds, 6 remaining items, 7 quick commands, 5 important notes).

### Fixture + test additions (47 fixtures + 60 total):
`tv/tests/fixtures/`: 60 .txt files (gelu_tanh/erf, softmax, rmsnorm, layernorm, cross_entropy, matmul weights/grad, embedding, chain fixtures for tape, etc.).
`tv/tests/*.tkv` test files: `grad_tests.tkv` (126 lines), `tape_tests.tkv` (114), `parity_tests.tkv` (139), `gc_tests.tkv` (288).
`tv/training/*.tgv` colocated copies: created by sync; should NOT be manually edited.

### Compiler document updates (not committed in repo, but in compiler repo):
`D:\TokenVector\Release\docs\e_handoff\*` (4 notes: 2026-10-16 list-alias, record-container, 2026-10-17 fix-request, 2026-10-17 verify-after).
`COMPILER-NOTES-2026-10-16.md`: committed (`8129c73`). The note's suggested workaround (`annotated local` with single quotes) was incorrect; corrected by user/official compiler docs.

---

## 3. COMPILER STATE (confirmed 2026-10-17)

New compiler exe: `D:\TokenVector\Release\dist\tkvc-tv.exe` (build 2026-10-17 23:58, commit `40a3b6f`).
Changes: B-01 regression fix (`il_codegen` literal-arg branch), B-04 SIMD (dll copy + version match), B-07 (`--verify-all`), B-05 (`now_ms`/`gc_*` builtins + timing), and G0-related diagnostics.

Verified fixes applied to repo:
- `simd_*` no longer crashes (dll present); `simd_gemm` works (14.6x faster, square matrix); `simd_*` 3-arg (add/sub/mul/scale/axpy) compile but are NOT bit-exact and allocate 16 B/element (2.4x slower vs scalar) — must NOT be used for parity.
- `now_ms()`/`now_us()`/`gc_*` are real; benchmark timing is real (not stub 0); `gc_counts()` returns `2/2/2` (actually real GC counts).
- `--verify-all` works for 5 entry points + library modules (verified 48/48 modules); but requires scalar entry params for library modules (not an issue for entry points).
- Annotated locals (`"list[f32]"`) work; single quotes (`'list[f32]'`) rejected.
- List `f64` → `f32` silent corruption is now caught and reported (`SyntaxError` at call site).

Remaining compiler issues (documented in `docs/e_handoff/`):
- P1-1: `gc_collect()` statement form missing from `_BUILTIN_LENH_DOC_LAP` (`file_io.tkv:190`); workaround in repo: `_gc_ret = gc_collect()`.
- P1-1: `gc_counts()` returns `""` (stub); gate always `0/0/0`; fixed by compiler to return real counts `2/2/2`.
- P3 (D-01): annotated local message should mention double quotes; not a bug.
- P2-2: `simd_*` 3-arg form allocates; `simd_gemm` requires square matrices; `v_*` names whitelisted but not registered.
- P2 (feature): library module verify (`--verify-all`) requires scalar entry params for library modules; solved by entry-based verify.
- G0 compiler portability: compiler runs but `.tkv` compile fails on Linux (IL `stloc.s 63` rejected by mono JIT at line 110 of `record_feature.tkv`); `.tkv` output files run fine on Linux/mono; `ilasm` missing; fix requires compiler team.

---

## 4. G1 — CORE TRAINING COMPLETED

### 4.1 Completed deliverables

- `tv/training/tensor.tkv`: `Tensor` record with zero-copy `view`, `reshape`, `t_row`, `t_slice_rows`, `t_at`/`t_set`, `t_sum`, `t_max`, `t_zeros`, `t_new`, `t_zero_grad`, `t_grad_add`, `t_alloc_grad`, `t_require_grad`, `erf` (Abramowitz-Stegun), `gelu_tanh`, `gelu_erf`, `silu`.
- `tv/training/ops.tkv`: 12 operations — `add_fwd/bwd`, `mul_fwd/bwd`, `matmul_fwd/bwd` (m,n,k with sequential accumulation for bit-exact), `rmsnorm_fwd/bwd`, `layernorm_fwd/bwd`, `gelu_erf_fwd/bwd`, `gelu_tanh_fwd/bwd`, `silu_fwd/bwd`, `relu_fwd/bwd`, `softmax_fwd/bwd`, `cross_entropy_fwd/bwd`, `embed_fwd/bwd` (index-add sorted for determinism).
- Fixtures: `tools/gen_fixtures.py` (185 lines) generates 60 fixtures (`gpt2_*` architecture fixtures + `g_*` op fixtures + `c_*` chain fixtures) from `torch.autograd` (torch 2.14.0+cpu, fp32, seed 0, eager attn).
- Parity: `tv/tests/parity_tests.tkv` (4/4 PASS: gelu_tanh bit-exact 0.0, cross_entropy 0.0, softmax 1.86e-09, rmsnorm 1.19e-07, info: tanh vs erf gap 4.12e-04).
- Gradient parity: `tv/tests/grad_tests.tkv` (20/20 PASS: matmul fwd 5.96e-08, dA 2.38e-07, dB 1.19e-07, gelu_erf 1.19e-07, gelu_tanh 8.34e-07, silu 1.19e-07, relu 0.0, rmsnorm fwd 1.19e-07, bwd 5.96e-08, layernorm fwd 0.0, bwd 1.19e-07, softmax fwd 1.49e-08, bwd 1.49e-08, CE loss 0.0, CE bwd 3.73e-09, add/mul bwd 0.0, embed bwd 0.0).
- Tape chain: `tv/tests/tape_tests.tkv` (8/8 PASS: chain h1 (rms_norm*w1) diff=1.86e-09, h2 (gelu_erf) diff=5.96e-08, h5 (silu) diff=2.98e-08, loss diff=1.49e-08, tape gx (6 layers) diff=7.45e-09, tape gw1 diff=5.59e-09, tape gb1 diff=4.66e-09, nodes diff=0.0).
- Gradcheck sensitivity: `tv/tests/gc_tests.tkv` (7/7 PASS: detects deliberate corruption max_abs_diff=0.04508522; detects 1% perturbation max_abs_diff=0.00070177; h values 0.01/0.005/0.002 show relative error < 0.015 for good h).
- Kernel registry sample: `tv/training/kernels_reg.tkv` (47 lines: `KernelReg` class with `register/get`, default_reg maps 10 ops to `simd_*` builtins + `hal_*` builtins, `backend='cpu'` default).
- Contract: `docs/numerics-contract.md` — approved, covers fp32, bit-exact CPU, deterministic accumulation, simd backend separation, dropout=0 for parity, GELU erf vs tanh distinction, double-quote annotations, and compiler source references.
- Linux portability: `docs/linux-portability-spike.md` — `.tkv` runs under mono; compiler fails (IL of compiler itself rejected by mono JIT at `stloc.s 63`); `ilasm` missing.
- Architecture skeleton: `tv/training/gpt2_ref.tkv` (embedding lookup via `op_embed_fwd`, MHA skeleton using `op_matmul_fwd` + causal mask, MLP skeleton, LM head skeleton with weight tying reference, full forward skeleton, norm helpers, `t_zeros` helper, `t_shape[0]` quirk handled with `_sh0` alias).

### 4.2 Actual verification results (all green)

- `run_cases.ps1`: ALL PASS (26/26 + gap 16/16 + parity 4/4 + grad 20/20 + tape 8/8 + gc 7/7 = 0 FAIL).
- `build_inference.ps1`: Completed Successfully (full pipeline + benchmarks with real timing: cold start 62ms, single 30us, throughput 133k req/s, FP32 641ms vs INT8 1735ms, Zero-GC −728 B).
- `--verify-all` 55/55 modules PASS (new compiler 2026-10-17 23:58, includes G1 + fixtures + mirror sync).
- No source regression (`run_cases` passes after all G1 source additions and mirror sync).

### 4.3 Key real-world findings (not guesses, measured)

- `simd_gemm` (square 256x256): **14.6x faster** than scalar triple-loop, **0 B alloc**.
- `simd_*` 3-argument forms (`simd_add`/`sub`/`mul`/`scale`/`axpy`): compile OK but **allocate ~16 B/element** (total 16.78 MB for 1M floats), making them **slower (2.4x)** than scalar loops and **breaking Zero-GC** — unusable for inference parity.
- `simd_*` requires list shapes (not container shapes) — `list[f32]` accepted for `simd_*` but **not** for `simd_gemm` in square form; `shape=list` rejected by compiler unless through array/matrix interface.
- `v_*` (`v_load8`, `v_mul8_add8`, etc.): listed in `_BUILTIN_LENH_DOC_LAP` but **no registered builtins found** in compiler source — dead names.
- `docs/numerics-contract.md` approved 2026-10-17; annotation double-quote rule (`"list[f32]"`) documented; `docs/linux-portability-spike.md` documents `.tkv` runs under mono but compiler fails (IL error, missing `ilasm`).
- Fixture pipeline (`gen_fixtures.py`): generates 60 fixtures from `torch.autograd` with deterministic seed; fixtures stored as `.txt` (readable via `read_file`); parity and gradient fixtures share same format; fixtures are small (~6.5 KB for 60 files) and version-controlled.
- `grad_tests.tkv` and `tape_tests.tkv` verify algorithm-level correctness independently of fixtures (gradcheck by finite differences); `gc_tests.tkv` verifies gradcheck is sensitive (detects deliberate corruption 0.045 and 1% perturbation 0.0007).

---

## 5. Commit status (final before G2 full architecture)

| Commit | Message |
|---|---|
| `53e817f` | Constructor migration (~130 sites) |
| `70256f2` | LSTM overflow + LoRA overflow fixes |
| `5d8a270` | compare_rivals: pool type, list-field, race |
| `8129c73` | docs: official compiler-notes update + handoff |
| `6f8ee13` | G1 core + fixtures + docs update |

Working tree (after `git rm --cached` leftover `docs/G1_RESULT_STATE.md`): all tracked/added; untracked: fixtures dir + mirror copies (`tv.training.*.tkv`) + `COMPILER-NOTES-2026-10-16.md` (official compiler team file, not ours) + `docs/linux-portability-spike.md`.

---

## 6. Next — G2 full architecture (GPT-2 124M)

Skeleton complete; fixtures exist for 4 parity + 20 gradient + 8 tape + 7 gradcheck cases. Next steps (user to approve or proceed):

1. Complete `gpt2_ref.tkv`: MHA attention real (manual QKV + mask + projection + residual), MLP real (linear up/down + GELU + residual), LM head weight tying real (`op_embed_fwd` with tied `wte`).
2. Generate architecture-level fixtures from `transformers.gpt2` (small config: same architecture, vocab 256, hidden 64, heads 4, layers 2, seq 32); fixtures include hidden-state stats per layer + loss + logits + gradients per parameter group.
3. Write architecture-level parity (`gpt2_tests.tkv`) comparing 1-layer forward and backward against fixtures.
4. Verify no regression: full `run_cases` (26/26 + gap + parity + grad + tape + gc), `--verify-all` 55/55, build Inference Completed.
