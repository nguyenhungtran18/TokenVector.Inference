# HANDOFF — TokenVector.Inference (Pure .TKV) — 2026-10-20

Verified: compiler build 23:58 (2026-10-17) — B-01..B-07 fixed + SIMD GEMM updates.
Repo HEAD: `f85ea3d` (P5.3 consolidation — benchmark full seq + recorded figures).
Truoc do: `b28bdb7` (registry kernel + numerics-contract); G3 tokenizer `8ac487d`, G3 pack `d8a203a`.
Not staged: `COMPILER-NOTES-2026-10-16.md` (official compiler team update, not ours).

**Luu y:** muc 1..6 ben duoi la snapshot 2026-10-17 (truoc G2/G3). Trang thai
hien tai cua G3 nam o muc 0 ngay duoi day.

**Dinh huong hien tai (2026-10-20):** muc 7 (TRIZ roadmap) + file
`docs/perf-plan-triz-roadmap.md` — muc tieu seq1024 full step 154.234 s
(43.5x vs torch) -> ~85 s (~20x), giữ nguyên 84/84 parity assertions.
**P5.4 (2026-10-20) da ha 154.234 -> 141.219 s (39.9x), 84/84 gate — xem 0.14
(5 file gpt2_ref sua chua commit, 4 probe untracked, chua push).**
**P5.5 (2026-10-20) gop forward: 141.219 -> 118.641 s (33.5x), 84/84,
bit-exact — xem 0.15.**

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
  - **PROBE AJ (P1.3 GC) — XONG**: builtins THAT = `gc_collect()` /
    `gc_allocated_bytes()` / `gc_counts()` (tra string "gen0/gen1/gen2" —
    KHONG index duoc, `t[i]` = AccessViolation). 1 buoc 124M = 23.2s:
    BWD +1577 MB live (grads materialized), GRAD_ACC free 1577 MB (1.42s),
    UNFLAT free 2837 MB (gen2+1). **743 gen0 + 8 gen1 + 7 gen2 collections/
    buoc — churn ~8 GB boxed f64 tu AdamW la THAT** (giai thich wall > cpu).
  - **P1.1 XONG (2026-10-19) — bwd ghi thẳng opt.g: −1.4s that**
    (bench C 24.17 → **22.78s/buoc**; losses bit-exact 10.674946 -> 8.5497055
    -> 9.754064). Refactor backward nhan optional `gflat`/offsets (default
    off — duong cu parity nguyen ven); bo GRAD_ACC + mat grads. Gate 5/5 xanh
    + probe AL: **grads flat vs cu maxdiff=0.0 ca 28 nhom**. It hon uoc tinh
    2.5s: dwte/gg/smm intermediates van mat, copy-add simd ~0.3-0.5s.
  - **CANH BAO mirror**: `tv/tests/tv.training.gpt2_ref.tkv` va
    `tv/training/tv.training.gpt2_ref.tkv` la **ban NHO 256/64/4heads/2layers**
    (khac 3 file 124M con lai) — sync phai giu hang so rieng, khong cp blind.
    (Lan nay da loi: init 124M trong test nho, phai git checkout + sed hang.)
  - Probes moi: `simd_probe_aj` (GC per phase: 743 gen0 + 7 gen2/buoc,
    BWD +1577MB live), `simd_probe_al` (grads parity old-vs-flat),
    `simd_probe_ah` (P1.2 fused measurements), `simd_probe_ak` (cu phap
    default param None: `x: 'list[i64]' = None` + `== None` OK, list literal
    default BI TU CHOI, bare 'list' annotation khong parse),
    `simd_probe_af` ([0.0]*n cham 9x append).
  - **PROBE AM (P2) — XONG, ket luat dao nguoc**: phan rach block_bwd 124M
    (best-of-3): **MLP bwd 391ms (57%) + attention bwd 219ms (32%) + fwd
    recompute ~77ms + LN bwd = 0.0ms** — block tong 687ms (x12 = 8.2s/13.25s
    BWD). **Scalar residue <5%** (attention scalar dctx+dw_proj ≈ 1ms/block)
    → KHONG con vung scalar >1s de sua. "Bwd 4.5x fwd" = GEMM volume 3x
    (moi linear_bwd = dx+dw+fwd-linear), khong phai residue.
  - **PROBE AN (P4a) — XONG**: chi phi 124M that theo seq (best-of-2; do lai
    2026-10-20, cung xu huong, +-10%):
    | n | block fwd | block bwd | head logits GEMM | head GFLOPS (flop huu ich) |
    | 8 | 125 ms | 657 ms | 531 ms | 1.16 |
    | 64 | 406 ms | 1578 ms | 1031 ms | 4.79 |
    | 256 | 1375 ms | 5062 ms | 1187 ms | 16.65 |
    => head GEMM **build-bound** (chi phi gan nhat hang theo seq) -> hieu suat
    huu ich tang **14x** khi seq x32. XAC NHAN gia thuyet P2: seq1024 het la
    nut that cua head.
  - **PROBE AO (P3) — XONG, gia thuyet bi bac bo**: do bang
    `tools/probe_threads.py` (poll `psutil.Process.cpu_times()` cua tien trinh
    con -> cpu/wall = so core dang chay THAT; control scalar ratio 0.97):
    | Pha | wall | cpu | ratio |
    | pure simd_gemm 768^3 x800 | 1.397 s | 29.63 s | **21.20 luong** (539 GFLOPS) |
    | pure simd_gemm 64^3 x200000 (shape that cua seq8) | 2.178 s | 2.16 s | **0.99 (1 luong, 49 GFLOPS)** |
    | 1 buoc 124M seq8 (fwd+bwd) | 28.77 s | 28.77 s | **1.00 (1 luong)** |
    => gemm **DA da luong** khi tile lon (768^3), nhung **tile nho 64^3 (shape
    that cua seq8) lai 1 luong**; **ca buoc hieu 1 luong** => phan ngoai gemm
    (dung tile/append, attention scalar, AdamW) chiem phan lon wall time.
    Moi toi uu sau phai nham phan 1 luong.
  - **P4 — XONG, KET LUAN DAO NGUOC** (bench `tv/tests/simd_bench_124m_seq.tkv`
    doc cau hinh seq qua `tv/tests/_seq_cfg.txt`; doi chieu torch tier `d` cua
    `tools/bench_torch.py`, CUNG data ids[i]=(i*37+11)%50257, pos[i]=i):

    | seq | tkv SIMD full step | torch full step | tỉ lệ |
    |---|---|---|---|
    | 8 | 22.78 s | 1.0216 s | 22.3× |
    | 256 | **109.4 s** | 1.4262 s | 76.7× |
    | 1024 | **674.6 s** | 3.8401 s | **175.7×** |

    (tkv mode 0 = fwd+bwd: **666.2 s** @1024 = fwd 120.0 + bwd 546.2 → plumbing
    optimizer chi 8.4 s, khong phu thuoc seq. torch fwd+bwd: 0.8228 s @256,
    3.2387 s @1024 → tỉ lệ fwd+bwd 205.7×.) **Khoảng cách KHÔNG thu hẹp
    mà NỞ RỘNG theo seq** — nguyên nhân (đo được): o seq1024 phan lon thoi gian
    cua tkv la **scalar O(n^2) 1 luong** (attention + proj + dung tile), trong
    khi gemm native da chay da luong (P3) va torch vectorize het. Buoc seq1024
    ton ~6.1 GB RAM (may 12.7 GB) — chay duoc, nhung can 12-13 GB cho 1 buoc.
  - **PROBE AP (P5a) — XONG, XAC NHAN CO LOI** (`tv/tests/simd_probe_ap.tkv`;
    124M thật, layer 6, 2 bản copy **giống hệt** code library + biến `mode`
    tắt từng vùng, self-check copy-vs-library; cấu hình `tv/tests/_seq_cfg.txt`
    dong 1 = n, dong 4 = reps):
    | Vùng (ms/layer) | n=8 fwd | n=8 bwd | n=1024 fwd | n=1024 bwd | proto GEMM @1024 |
    |---|---|---|---|---|---|
    | qkv linear (SIMD) | 26 | 26 | 187 | 187 | — |
    | core O(n^2) scalar | ~0 | ~1 | **5858** | **15999** | 756 (12 head) |
    | proj scalar (n·h^2) | **31** | — | **3970** | — | **78** |
    | dctx scalar (n·h^2) | — | **19** | — | **2204** | **47** |
    | dw_proj scalar (n·h^2) | — | **51** | — | **7032** | **78** |
    | attention total (library) | 54 | 225 | 9188 | 24703 | — |
    Cross-check: vung n·h^2 scale **tuyen tinh** theo n (proj 3970/128 = 31.0 vs
    do 31.1) => do dung. **Tiet kiem P5.1**: seq8 **0.75 s/buoc (3.3%)**,
    seq1024 **279 s/buoc (41%)** (chi P5.1; `dp/ds/dqkv` 10141 ms/layer van
    scalar). seq1024: attention scalar = 406 s / 666 s fwd+bwd = **61%**.
    **SUA ƯỚC LƯỢNG SAI CỦA P2**: "dctx/dw_proj chi loi ~1ms/block" **SAI ~6x**
    (thuc 70 ms/layer @n=8) — uoc luong cu sai vi "10M MAC ≈ 1 ms"; scalar
    thuc chi **~134 MMAC/s** (boxed f64, 1 luong — dung P3).
  - **TIEP THEO**: **P5.1 — code proj/dctx/dw_proj → `smm`** (khong can
    compiler team, `smm` co san). Sau do **P5.2** (score/ctx theo head → `smm`
    + mask causal). Gate moi thay doi: `simd_matmul_tests` 20/20 + 4 suite hoi
    quy + bench lai seq8 (22.78 s) va seq1024 (674.6 s).
    **BAI HOC DO (probe AP)**: (1) `now_ms()` do phan giai **~15.6 ms** → o
    seq8 phai trung binh N lan (reps), mot lan do vo nghia; (2) vong hot cua
    library dung **LIST cuc bo** (`ctx[r]`), KHONG dung `tensor.data[...]` → ban
    copy dung tensor indexing cham ~1.8x → phep "total - qkv - proj" ra so
    **AM**; (3) do ca ham roi tach bang bien `mode`, khong do tung token.
    GC/boxing 8GB/buoc van can native buffer tu compiler team.
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

File: `tools/bench_torch.py` (torch A/B/C + tier `d` = seq thực), `tv/tests/simd_bench.tkv` (nho),
`tv/tests/simd_bench_124m.tkv` (124M — import `gpt2_ref124` + `trainer`, khong
import gpt2_train vi gpt2_fwd_ps gan voi ban nho),
`tv/tests/simd_bench_124m_seq.tkv` (124M theo seq — cau hinh `_seq_cfg.txt`:
dong 1 = seq, dong 2 = mode 0=fwd+bwd / 1=full step, dong 3 = steps),
`tools/probe_threads.py` (do cpu/wall cua tien trinh con = so luong that),
`tv/tests/simd_probe_ap.tkv` (P5a: tach attention theo vung + prototype GEMM).

---

### 0.13 P5.2 + P5.3 — ATTENTION VE GEMM (2026-10-20) — XANH 84/84

Lam tiec tieu perf-plan P5. Doan `gpt2_attention` / `gpt2_attention_bwd` —
5 ban mirror (3 x 124M + 2 x nho) da dong bo, khac nhau **chi 6 hang thong so**
(cung quy uoc nhu HEAD):

- **P5.2** score/ctx theo tung head → `smm_dot(Q,K)` + scale `1/sqrt(hd)` +
  mask causal (`u > t` → 0) + softmax hang + `smm_fwd(S,V)`.
  **Vong head phai NGOAI vong token** — ban draft 2026-10-20 chen no *trong*
  vong token ⇒ moi `(t, head)` tinh ca ma tran n×n → **O(n³)**, thieu scale,
  thieu mask, va `import math` long trong vong lap.
- **P5.3** `dp/ds/dqkv` theo tung head → `smm`:
  `P = softmax_causal(inv·Q·Kᵀ)`, `dp = dctx·Vᵀ`,
  `dot[t] = Σ_d dctx[t,d]·ctx[t,d]` (ong thuc cung `Σ_u P·dp` nhung chi
  O(n·hd), va `ctx` da can cho `dw_proj`), `ds = P∘(dp − dot)`, roi
  `dq = inv·ds·K`, `dk = inv·dsᵀ·Q`, `dv = Pᵀ·dctx`.
  Ban draft cu sai 4 cho: doc `sc.data` tren mot **list** cuc bo (list khong co
  `.data`), truyen `dctxT [n,h]` vao `smm_dot` voi `k = hd` (sai stride), dao
  chu/dk, va **bo luon nhánh `dw_proj/db_proj` GEMM cua P5.1** → `simd=1` cho
  gradient `w_proj` = 0.
- Duong **scalar (`simd=0`) giu NGUYEN VANC**: chi bo 2 lo `if simd == 0/1`
  boc quanh code (simd=1 ra khoi ham bang `return` som) — cac cau thuc chay
  giong het → parity 1e-5 cua gate cu van xanh.

**Do A/B/C — cung process, cung trong so, cung dau vao**
(`tv/tests/_bench_p53.tkv`; A = 2 ham `gpt2_attention(_bwd)` chuyen **nguyen van**
tu `git HEAD` chi doi ten `def`; B = ma hien tai `simd=1`; C = `simd=0`),
attention **1 layer 124M**:

| seq | A: P5.1-only | B: P5.1+P5.2+P5.3 | **B/A** | C: scalar | B/C |
|---|---|---|---|---|---|
| 8 | 227.7 ms | 232.4 ms | **0.98×** | 725.8 ms | 3.12× |
| 32 | 285.3 ms | 279.3 ms | 1.02× | 2 908 ms | 10.4× |
| 64 | 398.5 ms | 347.8 ms | 1.15× | 5 902 ms | 17.0× |
| 128 | 679.5 ms | 492.0 ms | 1.38× | 11 328 ms | 23.0× |
| **1024** | **20 390** | **5 437** | **3.75×** | 110 016 ms | 20.2× |

seq1024 tach doi: **fwd (P5.2) 4 937 → 1 281 ms = 3.85×**;
**bwd (P5.3) 15 453 → 4 156 ms = 3.72×**. Giao diem ≈ seq 16-32; duoi do lo
~2% (4.7 ms/layer = +0.25% mot buoc seq8 ca 12 tang) — mac dinh van `simd=0`.
Self-check `|A−B|`/`|B−C|` ≤ 3.3e-8 (ca ba tinh cung mot phep). seq8/32/64
trung binh 40/8/4 lan vi `now_ms()` chi do duoc 15.6 ms.

**Do thang CA BUOC seq1024** — 2 lan chay cung binary `simd_bench_124m_seq`,
cung config `1024/1/1`, **chi khac 2 ham attention** (tam `git show 4a14acc:`
ve 2 mirror 124M roi tu khôi phục bang trap EXIT; `diff` xac nhan khôi phục
nguyen ven, backup da xoa):

| Chuan ma | full step seq1024 | ty le |
|---|---|---|
| P5.1-only (`4a14acc`) | **420.6 s** | — |
| P5.1 + P5.2 + P5.3 | **157.1 s** | **2.68×** (**−263.5 s**) |

Loss khop toi chu so cuoi (`10.760399 -> 13.44291` / `13.442912`) → hai chuan
tinh cung mot phep. Torch cung may cung ngay (`tools/bench_torch.py d 1024`):
fwd+bwd **3.084 s**, full step **3.672 s** → ty le tkv/torch **49.9× / 42.8×**
(truoc P5.3: 114.5×).

**Bench lai TOAN DAI seq (2026-10-20, lan 2 — `simd_bench_124m_seq` rebuild tu
nguon hien tai vs `bench_torch.py d <seq>`, torch chay 2 lan/seq spread ≤3%):**

| seq | tkv fwd+bwd | torch fwd+bwd | tkv full step | torch full step | **full step ty le** |
|---|---|---|---|---|---|
| 8 | 15.812 s | 0.302 s | **22.219 s** | 0.934 s | **23.8×** |
| 256 | 35.172 s | 0.782 s | **41.218 s** | 1.404 s | **29.4×** |
| 1024 | 150.625 s (fwd 25.281 + bwd 125.344) | 2.987 s | **154.234 s** | 3.542 s | **43.5×** |

- Loss `simd=1` khop tung chu so voi lan do truoc trong ngay (`10.572291` /
  `10.756927` / `10.760399`) — cung chuan ma, chi khac trang thai may. So seq1024
  lan do sang (153.8 / 157.1 s) cao hon ~2% — trong dao dong may; bang tren la so
  chot.
- **Backward chiem 83-88%** fwd+bwd cua tkv; **plumbing/optimizer** = full step −
  fwd+bwd = **6.4 / 6.0 / 3.6 s** (torch AdamW 124M: 0.63 / 0.62 / 0.55 s — tkv
  nang hon ~10×, gan nhu khong phu thuoc seq). O seq8 no chiem **29%** ca buoc.
- Ty le fwd+bwd gan phang **45-52×**, ty le full step tang dan 23.8 → 43.5× theo
  seq (optimizer co dinh cua torch che phan compute o seq nho).
- **seq256: 109.4 s (P5.1-only, dau ngay) → 41.2 s = 2.65×** nho P5.2+P5.3,
  ty le vs torch 76.7× → **29.4×** — hieu nang P5.x khong chi co o seq1024.

**Cach doc so:** bang A/B/C cô lap tren (3.75×, −179.5 s) cho so **thap hon**
do ca buoc (2.68×, −263.5 s) — do mot ham roi khoi allocator/cache cua pipeline
lam uoc luong thap. **So 420.6 → 157.1 la so quyet dinh.** Con lai 157.1 s chua
tach duoc attention / khong-attention bang do truc tiep (uoc theo ty le bang cô
lap ≈ 96 s attention / 61 s khac — **chua chot**); muon chon dung muc tieu tiep
hay chay `simd_bench_124m_profile` (do tung pha) truoc.

**Tat ca gate XANH — 84/84, fails=0.** Chay (tu goc repo):

```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/<gate>.tkv out/<gate>.exe main
./out/<gate>.exe
```

| Gate | Truoc P5.2/P5.3 | Nay |
|---|---|---|
| `simd_matmul_tests` | 20/20 | **20/20** |
| `gpt2_tests` | 10/10 | **17/17** (+7 assert `simd=1` doi chieu **ngay fixture PyTorch**, ≤2.4e-7) |
| `gpt2_scope_tests` | 9/9 | **9/9** |
| `gpt2_train_tests` | 5/5 | **5/5** |
| `gpt2_full_tests` | 20/20 | **33/33** (+`SIMD logits/loss` 124M = 2.6e-6 / 0.0; +attention **n=192** da o 3×3 fwd/bwd simd-vs-scalar ≤3.6e-5; +**backward TOAN BO 124M `simd=1` vs fixture PyTorch** — 5 grad ≤8.3e-7) |

**Diem quan trong cho nguoi ke:** truoc P5.2/P5.3 **khong gate nao chay
attention GEMM voi `simd=1`** — nen ban draft hong van “xanh” vi khong ai do no.
So chi tiet: `docs/perf-plan-simd-vs-pytorch.md` (muc **P5.2 + P5.3**),
`ROADMAP_TRAIN.md` (dong “SIMD train”).

### 0.14 P5.4 — ATTACK ATTENTION BWD (2026-10-20) — XANH 84/84, −8.8%

**Vi sao:** probe AQ (muc 1/TRIZ truoc do ghi "chua do duoc") tach duoc
backward @seq1024: attention_bwd 48.9s + fwd recompute 15.4s = **51%**,
LN chi 0.6%, embed 0.8%, gelu 2.6% — chon attack attention bwd (qyet dinh
goc 2026-10-20). 4 probe tao ra:

- `tv/tests/simd_probe_aq.tkv` — phan rach backward theo PHA, seq doc dong 1
  cua `tv/tests/_seq_cfg.txt` (8/256/1024), self-check SUM vs TOTAL ≤3%.
- `tv/tests/simd_probe_ar.tkv` — phan rach TUNG PHEP trong attention_bwd
  (1 tang, 19 pass; P13 build-vs-gemm), self-check −2.8/−2.2/+1.1%.
- `tv/tests/simd_probe_at.tkv` — now_us = now_ms×1000, phan giai ~15.6ms
  → vung do phai gop qua nhieu cua so (reps × NHeads) — bai hoc do luong.
- `tv/tests/simd_probe_au.tkv` — kiem `<=` + if/else lon (DUNG, khong phai loi ngon ngu).

**Bang quyet dinh (attention_bwd, 1 tang @seq1024, lib 4265 ms):**

| Vung | ms/layer | % |
|---|---|---|
| 6 GEMM/head (score/ctx/dp/dq/dk/dv) | 2469 | 58% |
| **2 transpose n^2/head (gDts+gPts)** | **672** | **16%** |
| qkv linear_bwd | 454 | 11% |
| softmax + gDs scalar | 375 | 9% |
| qkv fwd + extract + dctx + proj + writeback | 345 | 8% |

**DA LAM (giu — 5 mirror gpt2_ref dong bo):**

1. **Bo 2 transpose n^2/head** trong gpt2_attention_bwd: dk/dv tinh qua
   chuyen vi NHO cua Q/dctx (n×hd = 65K thay vi n×n = 1M) + writeback doc
   strided `gDkT[d*n+t2]`. Cung tich luy p-tile tang dan → **bit-exact**.
2. **dw = xᵀ·gy truc tiep** trong gpt2_linear_bwd: bo gyT (n×out_d) + dwT
   (in_d×out_d) — seq1024 qkv 4.13M → 0.79M copy/layer. Bit-exact.

**DA REVERT (do chung minh regression):** thu tu vong `_smm_core`
p0→j0→i0 cho m>n (ly thuyet P13 "it build tile hon") — that te **cham hon**:
FWD_BODY +101%, moi pha GEMM +40-100% @seq256 (tach bang git stash tung
nhom: revert 3 file simd_matmul thi ve baseline). `git checkout` 3 file.
**Bai hoc: so luong build tile khong phai chi phi thuc — chi do A/B moi chot.**

**Do A/B (cung phien, loss khop tung chu so 10.760399 → 13.442912):**

| seq | full step TRUOC | SAU | Δ |
|---|---|---|---|
| 8 | 22.219 s | **19.297 s** | −13.1% |
| 256 | 41.218 s | **37.813 s** | −8.3% |
| 1024 | 154.891 s (ctrl) | **141.219 s** | **−8.8%** |

- fwd+bwd @1024: 150.750 (ctrl) → **136.469 s** (−9.5%);
  **bwd 125.969 → 111.625 s (−11.4%)**.
- Pha AQ @1024 truoc/sau: ATTN_BWD×12 48.86 → **38.20 s (−21.8%)**,
  MLP_BWD −8.1%; LN/GELU/EMBED khong doi ✓ (dung code khong sua).
- Tyle vs torch (full step, khong do lai): 43.5× → **39.9×**
  (seq8 20.7×, seq256 26.9×; torch 0.934/1.404/3.542 s cung ngay).
- 5 gate trang thai cuoi: **84/84** (20+17+9+5+33, fails=0);
  SIMD grads 124M vs PyTorch ≤8.3e-7.

**Chua lam (ung vien ke theo so do):** gop forward loss+backward (−19s
@1024), cache attn-fwd trong block_bwd (−15s), Phase 2 flatten/lazy
materialize (~12s), lm-head (~11s). KPI ~20× can ca 3 nhom nay + Phase 3.

**Trang thai git:** 5 file gpt2_ref sua chua commit (P5.4), 4 probe untracked,
simd_matmul da tro lai HEAD. Chua push.

### 0.15 P5.5 — GOP FORWARD: backward dung saves (2026-10-20) — 84/84, −16.0%

**Van de (so do AQ/AR 0.14):** moi full step forward chay 2 lan — 1 lan cho
loss + 1 lan TINH LAI trong gpt2_backward (FWD_BODY ~19-21s @seq1024 = 15%
bwd) + GEMM logits lm-head tinh lai (~2-3s). Tong ~22-25s/lang thua.

**Thay doi (5 mirror gpt2_ref, dong bo):**

1. `gpt2_forward_saved(...)` — forward + LUU [logits, xf, xf_in, xins[0..11]]
   (~250MB @seq1024: z 206MB + activations ~40MB).
2. `gpt2_backward(..., saves=None)` — param moi CUOI; saves != None thi dung
   lai forward (khong recompute) + saves[0] thay GEMM logits lm-head;
   saves == None duong cu nguyen ven (toan bo gate/fixture khong doi).
3. 2 bench (`simd_bench_124m`, `simd_bench_124m_seq`) + probe AQ chuyen
   sang duong saves.

**BIT-EXACT (probe `tv/tests/simd_probe_as.tkv`, n=8, config 124M):**
logits=0.0, gflat (toan bo n_param)=0.0, out list=0.0, goi lan 2=0.0
(saves khong bi mutate). Overhead fwd_saved = 0.0%.

**Do A/B:**

| seq | full step TRUOC (P5.4) | SAU (P5.5) | Δ |
|---|---|---|---|
| 8 | 19.297 s | **17.469 s** | −9.5% |
| 256 | 37.813 s | **32.578 s** | −13.8% |
| 1024 | 141.219 s | **118.641 s** | **−16.0%** |

- mode 0 @1024: fwd 24.625 + bwd **86.219** (truoc 111.625, **−22.8%**)
  = 110.844 s (truoc 136.469, −18.8%).
- AQ @1024 sau: TOTAL_bwd **84.328s** (truoc 103.2s recompute = −18.9s —
  dung so fwd recompute da bo); LHEAD van 11.7s (con softmax+dxf+dwte).
- Loss khop tung chu so P5.4 (10.760399 → 13.442912) — xac nhan bit-exact
  o seq1024 (khong chi o n=8).
- Gate: **84/84** (duong saves=None). Ty le vs torch: 39.9× → **33.5×**
  (seq8 18.7×, seq256 23.2×; torch 0.934/1.404/3.542 s cung ngay).

**Ghi chu do luong:** may drift ~5-8% so voi luc do P5.4 cung ngay
(FWD_BODY 19.7→21.3, ATTN_BWD 38.2→42.4) -> ty le % A/B bench co the lech
vai phan tram; so ground truth cung tien trinh: probe AS bwd @n8
11812 → 9829 ms (−16.8%), va AQ TOTAL_bwd −18.9s @1024.

**Con lai (ung vien ke theo so AQ moi @1024):** ATTN_BWD×12 42.4s
(cache attn-fwd trong block_bwd — P5.6), MLP_BWD 16.2s (mlp fwd tinh lai
trong mlp_bwd), LHEAD 11.7s, flatten ~2s, LN_BWD 0.77s. KPI ~20× (~85s)
can tiep 2-3 attack nua.

**Trang thai git:** 5 gpt2_ref + 2 bench + probe AQ sua, probe AS moi —
chua commit/push.

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
`ROADMAP_TRAIN.md` (repo root): updated status + next steps.
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

---

## 7. TRIZ OPTIMIZATION ROADMAP (2026-10-20) — tu "thua" sang "khac"

**Context:** P5.3 consolidated (commit `f85ea3d`). Benchmark re-record cung may /
cung ngay tren seq 8/256/1024. tkv **van thua** PyTorch nhung gap da hep nhieu:
full step seq1024 **154.234 s vs 3.542 s = 43.5×** (truoc P5.3: 114.5× fwd+bwd).

### 7.1 Benchmark chot (2026-10-20, torch chay 2 lan/seq spread ≤3%)

| seq | tkv fwd+bwd | torch fwd+bwd | tkv full step | torch full step | **full step ty le** |
|---|---|---|---|---|---|
| 8 | 15.812 s | 0.302 s | **22.219 s** | 0.934 s | **23.8×** |
| 256 | 35.172 s | 0.782 s | **41.218 s** | 1.404 s | **29.4×** |
| 1024 | 150.625 s (fwd 25.281 + bwd 125.344) | 2.987 s | **154.234 s** | 3.542 s | **43.5×** |

**Sau P5.4 (cung ngay 2026-10-20, do A/B cung phien): full step 19.297 /
37.813 / 141.219 s — ty le 20.7x / 26.9x / 39.9x. Chi tiet: 0.14.**

**Sau P5.5 (cung ngay 2026-10-20): full step 17.469 / 32.578 / 118.641 s —
ty le 18.7x / 23.2x / 33.5x. Chi tiet: 0.15.**

### 7.2 Doc so (cai gi can sua)

- **Backward chiem 83-88%** fwd+bwd cua tkv — sau P5.3, **attention khong con la
  nut that**; nut that la backward KHONG attention (LN / gelu / embed-LM-head).
- **Plumbing/optimizer nang hon torch ~10×**: tkv 6.4 / 6.0 / 3.6 s vs torch
  0.63 / 0.62 / 0.55 s — gan nhieu khong phu thuoc seq, o seq8 no chiem **29%**
  ca buoc (day la ly do ty le full step seq8 thap hon ty le compute).
- **seq256 la bang chung P5.x khong chi co o seq1024**: 109.4 s (P5.1-only) →
  41.2 s = **2.65×**; ty le vs torch 76.7× → **29.4×**.
- **Loss van khop tung chu so** (`10.760399 → 13.442912`) ⇒ cung thuat toan,
  chi khac FMA/thu tu tich luy.
- Ty le fwd+bwd gan phang **45-52×** theo seq; ty le full step tang dan
  23.8 → 43.5× (optimizer co dinh cua torch che phan compute o seq nho).

### 7.3 Huong sua theo TRIZ

| TRIZ principle | Huong di | Ky vong |
|---|---|---|
| **Segmentation (1)** | Tach backward thanh stage rieng: `ln_bwd` / `gelu_bwd` / `proj_bwd` — moi stage toi uu doc lap | Lon nhat — backward = 83-88% chi phi |
| **Asymmetry (15)** | Khac phuong phap theo stage: attention da GEMM (xong); LN bwd → FMA tiled; gelu bwd → erf approx (trong numerics contract); optimizer → fused | Rl cua LN/gelu chua GEMM hoa la lon nhat con lai |
| **Merging (25)** | Lazy materialize (chi copy `ps → opt.p` 1 lan truoc `opt.step()`); fused AdamW (gop momentum + WD vao 1 vong lap) | Rl lon — plumbing dang ~10× torch, khong nen vay |
| **Nature of invention (40)** | Policy adaptiv theo stage: FMA neu co hardware, scalar fallback; FP16 accumulate + rescale FP32 truoc softmax neu throughput giu trong loss tolerance | Trung binh, sau khi segmentation + merging xong |

### 7.4 Loi trinh muc tieu (3 thang, CHUA commit)

- **Phase 1 (6 tuan)** — backward LN + gelu + fusion proj: seq1024 fwd+bwd
  150.6 s → **~70 s**, full step → **~85 s**, ty le **43.5× → ~24×**.
- **Phase 2 (5 tuan)** — lazy materialize + fused AdamW: plumbing
  6.4/6.0/3.6 s → **~2/1.9/1.1 s**, full step → **~80 s**, ty le **~23×**.
- **Phase 3 (2 tuan)** — policy adaptiv + mixed precision: them **5-10%**,
  ty le **~20-22×**.
- **Muc tieu chung: ty le tkv/torch ~20× o seq1024 full step** (tu 43.5× hien tai)
  — tuc "thua nang" sang "khac duoc".

### 7.5 Parity phai giu (khong duoc break)

| Gate | Truoc P5.2/P5.3 | Nay |
|---|---|---|
| `simd_matmul_tests` | 20/20 | **20/20** |
| `gpt2_tests` | 10/10 | **17/17** (+7 assert `simd=1` vs fixture PyTorch ≤2.4e-7) |
| `gpt2_scope_tests` | 9/9 | **9/9** |
| `gpt2_train_tests` | 5/5 | **5/5** (loss bit-exact) |
| `gpt2_full_tests` | 20/20 | **33/33** (+SIMD logits/loss 124M 2.6e-6/0.0; +attention n=192 3×3 fwd/bwd ≤3.6e-5; +backward TOAN BO 124M `simd=1` vs PyTorch, 5 grad ≤8.3e-7) |

### 7.6 Rui ro + contingency

| Rui ro | Contingency |
|---|---|
| GELU erf approx vuot 2.4e-7 logits | Default dung erf khong approx; chi giu tanh-approx lam duong debug |
| Lazy materialize lam vo parity (tai dung gradient giua cac buoc) | Gate tung stage: `gpt2_train_tests` phai 5/5 bit-exact; revert tung op ve eager materialize |
| Chien thang seq256 khong transfer sang seq8 | Benchmark ca 3 seq moi ung vien; giu mac dinh `simd=0` cho den khi chung minh net win tren ca 3 seq |
| Policy adaptiv them latency kiem tra moi call | Do chi phi policy; neu qua lon, bake policy vao hang compile-time constant |

---

*Cap nhat 2026-10-20 · Hanh dong tiep theo: Phase 1 — trien khai `layernorm_bwd` FMA tiled.
Chi tiet day du: `docs/perf-plan-simd-vs-pytorch.md` (muc P5.2 + P5.3) va
`docs/perf-plan-triz-roadmap.md`.*
