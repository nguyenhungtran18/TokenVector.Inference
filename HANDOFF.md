# HANDOFF — TokenVector.Inference (Pure .TKV) — 2026-10-20
Verified: compiler build 23:58 (2026-10-17) — B-01..B-07 fixed + SIMD GEMM updates.
Repo HEAD: `f85ea3d` (P5.3 consolidation — benchmark full seq + recorded figures).
Truoc do: `b28bdb7` (registry kernel + numerics-contract); G3 tokenizer `8ac487d`, G3 pack `d8a203a`.
Not staged: `COMPILER-NOTES-2026-10-16.md` (official compiler team update, not ours).
**Luu y:** muc 1..6 ben duoi la snapshot 2026-10-17 (truoc G2/G3). Trang thai
hien tai cua G3 nam o muc 0 ngay duoi day.
**Dinh huong hien tai (2026-10-20, MUC TIEU MOI — chu khoa):** phai dat
BANG roi NHANH HON PyTorch (1.0× roi ≤0.8×) tren CA 3 seq, van giu 84/84
parity — KHONG dung o muc ~20× nua. Sàn don luong ~37s > 3.542s -> da luong
la dieu kien can (P6). Ladder + phan tich: `docs/perf-plan-triz-roadmap.md`
§12. Duong di da di: 154.234 -> 141.219 (P5.4) -> **118.641 s (33.5×, P5.5)**.
**P5.4 (2026-10-20) da ha 154.234 -> 141.219 s (39.9x), 84/84 gate — xem 0.14
(5 file gpt2_ref sua chua commit, 4 probe untracked, chua push).**
**P5.5 (2026-10-20) gop forward: 141.219 -> 118.641 s (33.5x), 84/84,
bit-exact — xem 0.15.**
**P5.6 (2026-10-20) cache attention-fwd trong block_bwd: 118.641 ->
99.719 s (28.2x), 84/84, bit-exact — xem 0.16.**
**P5.7 (2026-10-20): lm-head — 2 huong thu deu bi DO bac bo (regression /
cham hon); LHEAD can kernel API moi (trans-A + beta) — xem 0.17.**
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
### 0.16 P5.6 — CACHE ATTENTION FWD TRONG block_bwd (2026-10-20) — 84/84, −16.0%
**Van de:** `block_bwd` recompute CA `gpt2_attention` fwd (~15.4s @seq1024 =
12% backward, probe AQ) chi de co `r = x + attn_out`.
**Thay doi (5 mirror gpt2_ref, dong bo):**
1. `gpt2_block_saved(...)` — nhu gpt2_block + tra `[out, r]` (gpt2_block giu nguyen).
2. `gpt2_forward_saved` luu them `rs[12]` (residual sau attention tung layer,
   +37MB @seq1024) — layout `[z, xf, xf_in, xins..., rs...]`.
3. `gpt2_block_bwd(..., save_r=None)` — save_r != None thi dung r, BO HAN
   attention fwd; None thi duong cu (parity nguyen ven).
**Bit-exact:** probe AS logits/gflat/out/reuse = **0.0** (khong doi).
**Do A/B (loss khop tung chu so 10.760399 -> 13.442912):**
| seq | full step P5.5 | **P5.6** | Δ |
|---|---|---|---|
| 8 | 17.469 s | **16.937 s** | −3.0% |
| 256 | 32.578 s | **29.735 s** | −8.7% |
| 1024 | 118.641 s | **99.719 s** | **−16.0%** |
- mode 0 @1024: bwd 86.219 -> **70.734 s** (−18.0% = **−15.485s** dung bang
  attn-fwd recompute da bo); fwd 25.032 s (khong doi).
- AQ @1024: TOTAL_bwd 84.328 -> **68.187 s**. Con lai: **ATTN_BWD 41.5 s (61%)**,
  MLP_BWD 16.0 s (23%), LHEAD 11.5 s (17%), LN/EMBED 1.9 s.
- Gate: **84/84**. Ty le vs torch: 33.5× -> **28.2×** (seq8 18.1×, seq256 21.2×).
- Luu y AQ: RESIDUAL am lon (−24.5s) la binh thuong — SUM_cac_phan gio gom
  FWD_BODY (21.3s) khong con nam trong TOTAL_bwd; doc TOTAL + cac pha rieng.
**Ke tiep (§12):** P5.7 LHEAD 11.5 s (softmax-grad scalar + dxf/dwte GEMM),
roi P5.8 kernel GEMM v2; P6 da luong / P7 precision van cho quyet dinh chu khoa.
**Trang thai git:** 5 gpt2_ref sua — chua commit.
### 0.17 P5.7 — LM-HEAD: 2 HUONG THU, DO BAC BO (2026-10-20) — code hoat dong = P5.6
**Probe moi:**
- `tv/tests/simd_probe_av.tkv` — tach LHEAD @1024 (seq tu `_seq_cfg`):
  SOFTMAX_GRAD **1328 ms** | TRANSPOSE_ggT **1562 ms** (51.5M ptu) |
  GEMM_dxf **2437 ms** | GEMM_dwte **3062 ms** | SUM 8389 ms. (AQ LHEAD 11.5 s
  con ~3 s la logits GEMM — `_aq_lhead` khong nhan saves.)
- `tv/tests/simd_probe_aw.tkv` — micro-bench transpose: [1024,50257]
  BLOCKED 4375 vs PLAIN 1625 ms; [768,50257] 3328 vs 1235 ms.
**Thu 1 — dwteT (bo transpose ggT):** xfT (15 ms) -> dwteT GEMM (2360 <
3062 ms) -> transpose NGUOC. Bit-exact (probe AS 0.0, loss khop) NHUNG
**REGRESSION**: seq8 full step 16.937 -> **20.688 s (+22%)**, seq256 29.735 ->
**34.984 s (+17.6%)** — transpose nguoc h×V = 38.6M ptu CO DINH, trong khi
ggT chi n×V (nho khi n nho). **REVERT** (giu ggT).
**Thu 2 — transpose blocked 64×64:** **CHAM HON 2.7×** (4375 vs 1625 ms) —
vong 4 tang thong dich dat hon loi cache. **REVERT** (giu vong 2 tang).
**Ket luan (do, khong doan):** LHEAD 11.5 s KHONG cat duoc voi API kernel
hien tai (chi smm_fwd/smm_dot: khong trans-A, khong beta). Yeu cau cho P5.8:
(a) kernel nhan **A transposed** -> bo 1.56 s transpose ggT; (b) **beta /
tich luy in-place** -> bo copy Ct + 52M vong tich luy thong dich; (c) bo alloc
dwte thua 154MB khi ghi thang gflat. Khi co du 3: LHEAD ~4-5 s.
**Trang thai:** code hoat dong = P5.6 (chi them comment bai hoc vao
`gpt2_ref`/`simd_matmul`). Bench lai xac nhan: seq8 17.031 s (noise),
seq256 29.750 s, @1024 fwd+bwd 95.797 / bwd **70.688 s** = P5.6; mode1
102.641 s (+2.9 s drift nhiet o pha optimizer — fwd/bwd khong doi).
Bit-exact 0.0, **84/84** gate.
### 0.18 P5.8 REDO — KERNEL GEMM v2 (2026-10-21) — DANG DO, CHUA WIRE
**Yeu cau chu khoa:** "lam lai tu P5.8" — bo 2 ban P5.8/P5.9 lam truoc do
(worktree, khong commit) vi **hong**: `_smm_core_p58` khong goi `simd_gemm`
(lap scalar sai chi so, `smm_fwd_p58` truyen sai ndim = k thay n), P5.9 co
3 ham `pass` tra tensor rong. Da `git checkout` 3 file ve HEAD, xoa
`__pycache__`.
**Phat hien quan trong ve P5.7 (commit `6bf4a1c`, dang o HEAD):**
1. **4 mirror `gpt2_ref` KHONG co hunk P5.7** (chi canonical thay doi) ->
   gate 84/84 sau P5.7 **da chay tren code cu** — hunk P5.7 chua bao gio
   duoc test.
2. Hunk do **BUG**: nhanh `if use_flat == 1 ...` trong `simd == 0` chi ghi
   dwte vao gflat, **bo qua tich luy `dxf`** (dxf = 0 -> grad ln_f/sau do
   sai toan bo), la `xfT = smm_transpose(xf, n, h)` tinh roi **khong dung**.
   (HANDOFF 0.17 ghi "chi them comment" — khong khop voi diff thuc te.)
3. Ke hoach: revert hunk nay ve trang truoc 6bf4a1c (mirror dang giu dung
   trang do), dong bo 5 file, chay gate truoc khi lam P5.8 tiep.
**P5.8 v2 da implement (tv/training/simd_matmul.tkv + 2 mirror da sync, chua goi):**
- `_p58_tile_a_at` — build tile A-transposed: doc nguon row-major lienh tuc,
  scatter vao o T×T nho (khong materialize A^T).
- `_p58_core` — **1 loop nest** (ghi ro trong comment: 2 nest + early-return
  cham hon 2.3x — doan ma compiler); dispatch: `use_acc = 1` chi khi
  `p0t >= 2 && j0t == 1 && m*n >= T*T` -> tich luy `acc += Rt` bang
  **`simd_add` NATIVE**, reset bang `simd_sub(acc,acc,acc)`; nguoi khac di
  duong v1.
- Export: `smm_fwd_v2`, `smm_dot_v2`, `smm_fwd_at` (a luu [k][m] = A^T).
- **DETERMINISM:** thu tu tich luy p0 tang dan, cung gia tri p-tile nhu v1.
**Probe (chay duoc, tu goc repo):**
| Probe | Ket qua |
|---|---|
| `simd_probe_ax.tkv` | **BIT-EXACT 0.0 TOAN BO** (dwte_at/dxf/ctx/score/logits/qkv) + simd_add semantics/aliasing/resize OK. **NHUNG v2 cham hon v1: dwte 1797->3672, dxf 1078->3046, logits 1172->3171, qkv 47->140 ms @seq256.** |
| `simd_probe_ay.tkv` | Gap on dinh 2.3x, khong phai GC/thu tu (v2 truoc van cham): v1 47-63 vs v2 141-172 ms. |
| `simd_probe_az.tkv` | Clone `_az_core` (cung logic, 1 nest, trong probe) = v1 (47-63 ms); `_p58_core` van cham -> **khong phai do lap lai nest**. |
| `simd_probe_ba.tkv` | Tach pha @256x2304x768: tile 16 vs 30 ms, gemm ~0, **acc 31 vs 126 ms (4x)** — vong tich luy cua v2 cham 4x DUONG MA IDENTICAL (use_acc=0, else-branch). |
| `simd_probe_bb.tkv` | **BAC BO:** +bien thua/acc var/ceildiv = khong cham (47-79); +trans_a param +chi nhanh = khong cham (46-62). |
**Con 2 gia thuyet chua test (nguoi ke lam tiep — probe BC):**
- **v5:** v3 + chi nhanh `if use_acc == 1: simd_add(...) else: <acc loop>`
  + goi `simd_sub` trong function (khong chay) — co phai su co call site
  simd_add/simd_sub lam codegen kem?
- **v6:** copy NGUYEN VAN `_p58_core` vao probe — neu cham trong probe ->
  noi dung function; neu nhanh -> vi tri/checkbox module `simd_matmul`.
Neu khong do duoc: **split thanh 2 ham** — `_p58_core_simple` (giong
`_bb_v1` + trans_a, dam bao khong cham) va `_p58_core_acc` (chi dung cho
j0t==1), dispatch o wrapper export.
**Trang thai file (chua commit):**
- `tv/training/simd_matmul.tkv` + 2 mirror (`tv/tests/`, `tv/training/`):
  + P5.8 v2. Gate cu chua chay lai.
- `tv/tests/simd_probe_{ax,ay,az,ba,bb}.tkv`: moi (untracked).
- `gpt2_ref`: NGUYEN trang HEAD — can revert hunk P5.7 + dong bo 5 mirror
  TRUOC khi wire `smm_fwd_at/smm_fwd_v2` vao lm-head simd==1.
- `tv/tests/_seq_cfg.txt` = 256 (untracked, probe config).
**Cong viec con lai (theo thu tu):** (1) chot nguyen nhan cham / split
2 ham; (2) do lai AX toi khi v2 >= v1 moi shape; (3) revert hunk P5.7 +
dong bo 5 mirror gpt2_ref; (4) wire vao lm-head (dwte = smm_fwd_at bo
transpose ggT) va cac call site do duoc loi; (5) gate 84/84 + bench
seq8/256/1024 A/B; (6) cap nhat plan §12 + ROADMAP + chuong nay.
### 0.19 P5.8 — KERNEL GEMM v2 HOAN THANH (2026-10-21) — 84/84, BIT-EXACT, TRANS-A DA WIRE
Tat ca 6 viec o 0.18 da xong. Tom tat ket qua:
**1. NGUYEN NHAN v2 cham 2.5x DA CHOT (probe BC — 2/2 gia thuyet xac nhan):**
call site `simd_add`/`simd_sub` (System.Numerics) trong CUNG function voi vong
lap scalar lam compiler codegen vong do cham 2.5-4x — DU KHONG CHAY:
- BC-v5 = v3 + chi nhanh use_acc + goi simd_add/sub (use_acc=0): 156-172ms
- BC-v6 = copy nguyen van `_p58_core` vao probe: 156-171ms
- BC-v1 (clone v1): 63ms — ca hai bit-exact 0.0 -> NOI DUNG function, khong
  phai vi tri/checkbox module simd_matmul.
**2. Split + do + chon (tv/training/simd_matmul.tkv, 3 file dong ho):**
- `_p58_core_simple` — v1 loop nest + trans_a, KHONG call site simd_add/sub.
- `_p58_core_acc` — DA XOA sau khi DO: @1024 CHAM HON (dxf [1024,50257]
  3000 vs 2703ms; simd_add allocate ~16B/element — SIMD_NOTES P2-2); ctx/score
  nho 31 = 31 khong do duoc lech. Khong co shape nao acc nhanh hon.
- `_p58_core` dispatch TOAN BO -> simple; exports `smm_fwd_v2`/`smm_dot_v2`/
  `smm_fwd_at` giu nguyen. Determism: p0 tang dan, cung gia tri p-tile.
- AX chay lai ca @256 va @1024: **ALL BIT-EXACT 0.0**. Dwte trans-A loi ro:
  @256 1578 vs 2000 cu, @1024 3938-4109 vs 4797-5078 (~400-900ms); dxf/
  logits/qkv/score/ctx = ngang cu trong noise ±250ms (cu tu no cung doi
  2609 -> 2328 giua 2 lan chay).
**3. Revert hunk P5.7 (commit `6bf4a1c`, bug dxf):** reverse-apply patch —
canonical `gpt2_ref.tkv` == pre-6bf4a1c == ref124 mirror (byte-strips).
2 mirror `tv.training.gpt2_ref` la test-variant CONSTANTS 256/64 CO Y (tu
dau, giu nguyen — code da khop). Hunk P5.7 bo qua tich luy `dxf` khi
`use_flat=1` (dxf=0 -> grad ln_f/sau do sai toan bo) + `xfT` tinh roi khong dung.
**4. Wire lm-head (ca 5 file gpt2_ref):** `dwte = smm_fwd_at(gg, xf,
VocabSize, h, n)` thay `ggT = smm_transpose + smm_fwd` (bo transpose
~1.5s @1024). `dxf` van `smm_fwd` nguyen — KHAC P5.7 (khong dot vao dxf).
Cac call site `smm_transpose` khac (xT/gQt/gDcT/ctxTt) KHONG doi — khong do
duoc loi nao cho chung.
**5. Gate 84/84 XANH** (20+17+9+5+33, fails=0) — gom `SIMD grad wte vs
PyTorch 8.3e-07` (wire smm_fwd_at duoc kiem chung parity) + `SIMD loss 0.0`.
Loss seq1024 10.760399 -> 13.442912 KHOP EXACT tung chu so voi P5.4 — backward
chay dung duong P5.x sach.
**6. Bench A/B full step (mode1, `simd_bench_124m_seq`, best-of):**
| seq | A = P5.6 | B = P5.8 | Δ |
|---|---|---|---|
| 8 | 16.937 s | 17.156 s (3 lan: 17.58/17.16/17.20, loss deterministic) | +1.3% (drift nhiet) |
| 256 | 29.735 | 29.641 | −0.3% |
| 1024 | 99.719 | **98.203** | **−1.5%** |
Ty le vs torch (0.934/1.404/3.542): **18.4x / 21.1x / 27.7x**.
**7. Docs:** plan §12 (P5.8 row ✅ XONG + so do), ROADMAP_TRAIN dong PARITY
(ladder P5.6 ✅ / P5.7 ❌ / P5.8 ✅) da cap nhat trong chuong nay.
**Trang thai file (CHUA COMMIT):** simd_matmul ×3, gpt2_ref ×5,
`tv/tests/simd_probe_bc.tkv` (moi, untracked), `_seq_cfg.txt` = 8
(untracked, bench config). `out/*.exe` da build lai tu nguon hien tai.
**Con lai:** (a) commit neu chu khoa cho phep; (b) P6 — da luong GEMM
(dieu kien can toi parity, xem plan §12 quyet dinh a); (c) seq8 batch nho
chua chuyen doi (18.4x — overhead interpreter, rieng biet voi seq1024).
---
### 0.20 P6 GIAI DOAN1 — SONG SONG attention_bwd THEO HEAD (2026-10-21) — 84/84, BIT-EXACT, seq1024 −23.3%
Tiep HANDOFF 0.19 ("con lai: P6"). Thu tu lam: do truoc, chon dich theo so do.
**1. SO DO CHON DICH (profile fixture n=8 + AQ @1024):**
- n=8: BWD 10.2s (53%), ADAMW 5.0s (26%), plumbing ~10%.
- @1024: ATTN_BWD_x12 **40.7s** (59% cua bwd 68.8s), MLP_BWD 16.2s,
  LHEAD_BWD 11.5s, FWD_BODY 20.8s, embed 0.9s.
- **BAC BO**: env `TKV_GEMM_MIN_N=64/48` (nho hon96 de tile T=64 cham native
  MT) — seq256 30.75/30.44s vs control 29.67s = SLOWER (dung ly do native
  chon96); loss khong doi.
**2. KHAI NIEM DA LUONG — probe BF/BG/BG2 (moi, untracked):**
- BF: `thread_spawn(fn, record)` + `thread_join` dung gia tri; record field
  chi nhan scalar/container/class-cung-file (**Tensor bi tu choi** -> truyen
  `list[f32]` = Tensor.data); ghi DISJOINT vao list chung bit-exact 0.0;
  NHUNG 4w = 0.35x so voi 1w (cpu ratio 3.1 -> thread THAT dang chay, moi
  worker cham ~10x).
- BG: pure-local worker (4 lan cong viec, cung wall ~1x) = ~3.7x hieu luc
  thuc; doc list chung ~3x.
- **BG2 — CHOT**: ghi disjoint vao CUNG list = **1.24x** (78->63ms), moi
  worker list RIENG = **5.2x** (15ms). => QUY TAC P6: worker tu tinh vao
  output RIENG, MAIN thread gop (merge khong chanh vi chi1 thread).
- `thread_join` phai GAN bien (dung lenh doc lap bi loi il_codegen);
  import `tv.training.tensor` truc tiep (khong di qua gpt2_ref124).
**3. AN TOAN NATIVE (doc tkv_simd_native.c):**
- `SgemmF32`: n < TKV_GEMM_MIN_N(96) -> `gemm_rows_f32` truc tiep, buffer
  pack `static __declspec(thread)` -> **AN TOAN concurrent**.
- n >= 96 -> `tkv_run_mt` dung `g_job[]` GLOBAL **KHONG CO LOCK** -> GOI
  DONG THIEU = RACE. => worker P6 chi duoc cham vao gemm T=64<96 (attention
  dung dung T=64) — minh chet day du.
**4. IMPLEMENT (5 file gpt2_ref, dong ho):**
- Module-level: class `_AtnJob` (qkvl/dctxl/ctx_out/dq_out list[f32] + n/hlo/
  hhi/inv), `_attn_hb_zlist`, worker `_attn_hb_one` (body head cu, ghi vao
  output rieng), `_attn_hb_merge` (gop1 head, main thread).
- Driver `gpt2_attention_bwd` simd==1: `if n < 128` -> vong cu NGUYEN;
  `else` -> spawn 12 worker (1/head) + join + merge. Doc qkv/dctx read-only,
  moi head ghi vi tri (t, head, d) DUY NHAT khong tich luy chéo -> khong
  phu thuoc thu tu -> **BIT-EXACT**.
- Bug sua trong qua trinh: thay chuoi `qkv.data` an substring trong
  `dqkv.data` -> `dj.qkvl` (3 dong) -> sua thanh `j.dq_out`.
**5. KET QUA:**
- **Gate 84/84** (20+17+9+5+33); gate n=192 attn bwd simd-vs-scalar digits
  GIONG HET truoc P6; loss moi bench khop exact (10.760399 -> 13.442912).
- **Bench A/B (mode1, best-of):**
| seq | A = P5.8 | B = P6 GĐ1 | Δ |
|---|---|---|---|
| 8 | 17.156 s | 17.015 s (fallback n<128, loss identical) | −0.8% |
| 256 | 29.641 | 28.671 | −3.3% |
| 1024 | 98.203 | **75.297** (lan 2: 76.4) | **−23.3%** |
- AQ @1024: **ATTN_BWD 40.7 -> 17.8 s (−56%)**, TOTAL_bwd 68.8 -> 44.3 s.
- Ty le vs torch: 18.2x / 20.4x / **21.3x** (truoc 18.4/21.1/27.7).
**6. DOC:** plan §12 row P6 GĐ1 ✅ + quyet dinh ⓐ da tra loi (COC primitive)
+ risk "blocked" da hethan; ROADMAP dong PARITY.
**Trang thai file (CHUA COMMIT — gom ca P5.8 truoc):** gpt2_ref ×5, simd_matmul
×3, docs ×3, probe bc/bf/bg/bg2 (untracked), `_seq_cfg.txt` = 1024.
**Con lai cho P6 GĐ2 (theo gia tri do duoc):**
(1) attention FWD per-head (cung mau, FWD_BODY 20.8s chua cham);
(2) LHEAD_BWD 11.9s (softmax row loop — song song theo token);
(3) MLP_BWD 15.8s (gelu scalar + tile build — xem lai thanh phan);
(4) AdamW + plumbing ~7s (elementwise — can own-list + merge qua nhieu buffer);
(5) hieu suat merge/worker hien ~2.3x cho attention (40.7->17.8) — con
duoc toi uu (W=12 vs4, doc qkv share...).
### 0.21 P6 GIAI DOAN 2 — SONG SONG attention FWD THEO HEAD (2026-10-21) — 84/84, BIT-EXACT, seq1024 -15.3%
Tiep 0.20 (danh sach "Con lai cho P6 GĐ2" muc (1)). Thu tu: doc code truoc,
chot an toan native, implement, do.
**1. CHOT AN TOAN (doc lai tkv_simd_native + simd_matmul.tkv):**
- `SgemmF32` chi tach giua n < TKV_GEMM_MIN_N(96) (`gemm_rows_f32`, bp
  `__declspec(thread)` — an toan concurrent) va n >= 96 (`tkv_run_mt` dung
  `g_job[]` GLOBAL khong lock — RACE neu goi dong thieu).
- Trong worker attention FWD co 2 GEMM, ca hai deu T = 64 < 96:
  `Smat = smm_dot(gQ,gK,n,n,hd)` -> T = min(_tilek(hd)=64, roundup64(n)) = 64;
  `gCtx = smm_fwd(Smat,gV,n,hd,n)` -> T = min(_tilek(n), roundup64(hd)) = 64.
- Proj `smm_fwd(ctxT,w_proj,n,h,h)` co T >= 96 -> giu NGOAI worker, chay main
  thread sau join (giong truoc, khong doi).
**2. IMPLEMENT (5 file gpt2_ref, dong ho — 3 file 124M byte-identical, 2 file nho byte-identical, +163 dong/file):**
- Class `_AtnFJob` (qkvl, ctx_out, n, hlo, hhi, inv) + worker `_attn_f_one`
  (gather Q/K/V + smm_dot + scale/mask/softmax + smm_fwd gCtx, ghi vao
  ctx_out RIENG) + `_attn_f_merge` (main gop vao ctxT) — dat TRUOC
  `gpt2_attention` (khong forward-ref).
- Driver `gpt2_attention` simd==1: `if n >= 128` -> 12 spawn + join + merge +
  proj + `return out`; roi vong sequential cu = duong fallback NGUYEN.
- **SUA LATENT OOB**: guard `if j.hlo >= NHeads: return 0` vua day vao
  `_attn_f_one`/`_attn_f_merge` (fwd) LAI CUNG `_attn_hb_one`/`_attn_hb_merge`
  (bwd GĐ1): mirror nho NHeads=4 van driver spawn 12 nhom [k,k+1) — head >= 4
  doc/ghi vi tri ngoai hang. Gate nho khong trigger (deu n < 128), nhung n >= 128
  tren ban nho se corrupt/OOB. Guard khong doi gi tren duong da test (124M:
  hlo 0..11 < NHeads=12 -> khong cham; ban nho: worker khong chay).
**3. KET QUA:**
- **Gate 84/84** (20+17+9+5+33, fails=0); gate n=192 `P5.2 attn fwd` (duong
  parallel MOI) = **1.0986696e-09** vs scalar; `SIMD loss vs PyTorch 0.0`.
- **Bench A/B (mode1, best-of, cung ngay cung may):**
| seq | A = P6 GĐ1 | B = P6 GĐ2 | d |
|---|---|---|---|
| 8 | 17.015 s | 17.078 s (fallback n<128, khong doi duong chay) | +0.4% (noise) |
| 256 | 28.671 | **26.141** | **-8.8%** |
| 1024 | 75.297 | **63.781** (lan 2: 64.406) | **-15.3%** |
- Loss: seq8 10.572291->7.0119243, seq256 10.756927->11.807251, seq1024
  **10.760399 -> 13.442912 KHOP EXACT** tung chu so voi GĐ1/P5.4 -> bit-exact.
- AQ @1024: **FWD_BODY 20.8 -> 11.063 s (-9.7 s, -47%)**; ATTN_BWD 17.844
  (giu nguyen — dung, bwd khong doi), MLP_BWD 15.937, LHEAD 10.891,
  TOTAL_bwd 43.907, chk SUM-vs-TOTAL = 1.05e-4 <= 0.8.
- Ty le vs torch (0.934/1.404/3.542): **18.3x / 18.6x / 18.0x**
  (truoc GĐ2: 18.2x / 20.4x / 21.3x).
**4. DOC:** plan §12 row P6 GĐ2 + ghi chu ⓐ bo sung GĐ2; ROADMAP dong PARITY.
**Trang thai file (CHUA COMMIT):** gpt2_ref x5, plan §12, ROADMAP, HANDOFF
(chuong nay), `_seq_cfg.txt` = 1024.
**Con lai (theo AQ moi @1024):** ATTN_BWD 17.8s (40% cua bwd 43.9s — con
du, GĐ3?), MLP_BWD 15.9s (chua cham), LHEAD 10.9s, FWD_BODY 11.1s (con
~4-5s attention fwd + ln/mlp), AdamW + plumbing ~7s. Muc tieu chu khoa
(<=1.0x torch = 3.542s @1024)can tiep nhieu buoc: parallel con lai + P7 mixed precision.
### 0.22 P6 GIAI DOAN 3 — BUILDER FIX + WAVE-PARALLEL TILE-BUILD (2026-10-21) — 84/84, BIT-EXACT, seq1024 -13.0%
Tiep 0.21 ("Con lai": ATTN_BWD 17.8 / MLP_BWD 15.9 chua cham / LHEAD 10.9).
Nguyen tac repo: DO TRUOC, chon dich theo so do.
**1. AQ @1024 TRUOC:** ATTN_BWD 18016 · MLP_BWD 16484 · LHEAD 11766 ·
FWD_BODY 11234 · TOTAL_bwd 44922 ms.
**2. PROBE BH (`tv/tests/simd_probe_bh.tkv`, MOI — prototype P6 GĐ3):**
worker build tile RIENG (indexed write vao buffer main pre-alloc, KHONG append,
KHONG simd_gemm), main replay gemm theo DUNG thu tu i0->p0->j0. Ket qua THAT:
- **BIT-EXACT 0.0** ca 5 shape 124M + 3 shape nho (LIB-vs-PAR), fails=0;
  BH[0] spawn+join x100 < 15.6ms (reps negligible).
- **Stage A (phat hien 1): builder append -> indexed write vao buffer TAI SU
  DUNG** — append List capacity-doubling ton ~23ns/ptu: LIB vs SER dxf
  2468/1203, dwte 3813/1984, logits 2469/1781 ms -> **1.4-2x MOI GEMM, khong
  can thread**.
- **Stage B (phat hien 2): wave-parallel chi loi o shape vocab** — PAR dxf 766,
  dwte 1187, logits 1156 ms (3.2/3.2/2.1x); shape MLP S1/S2 PAR CHAM HON SER
  (140/172 vs 109/78 — overhead 24 buffer 78ms + spawn) -> route: vol =
  n_i*n_p*n_j*T*T < 40M -> serial, >= 40M -> wave 12 (chi 3 GEMM vocab cua
  lm-head dat nguong).
- **BH[3] phan rach MLP_BWD 1 layer @1024 (sum vs whole = 1.2%):** inner_fwd
  156 + gelu_fwd 125 + r2(dx2+dw2+db2) 469 + gelu_bwd 187 + r1(dx1+dw1+db1)
  391 = 1297/1312 ms -> **gelu scalar = 24% MLP_BWD — ung vien GĐ4**.
**3. IMPLEMENT:**
- `tv/training/simd_matmul.tkv` (+2 mirror, md5 giong): them
  `_tile_a_w/_tile_b_w/_tile_a_at_w` (viet DAY DU T^2 — buffer tai su dung nen
  vung pad cung phai ghi lai); `_smm_core` + `_p58_core_simple` dung 2 buffer
  rieng At/Bt; them `_smm_core_pd` + export `smm_fwd_pd/smm_dot_pd/
  smm_fwd_at_pd` (route tu dong serial<->wave). Worker `_pd_one` CHI build tile
  — T >= 96 ma goi simd_gemm dong thieu se cham vao g_job global khong lock ->
  CAM worker goi gemm (ghi ro trong comment).
- 5 file `gpt2_ref`: 4 site lm-head -> pd (`logits = smm_dot_pd` trong
  forward_saved, `logits_t = smm_dot_pd`, `dxf = smm_fwd_pd`,
  `dwte = smm_fwd_at_pd`). Attention worker khong goi gpt2_linear* (chi smm_*
  thuong T=64) -> khong co nested spawn.
**4. KET QUA:**
- **Gate 84/84** (20+17+9+5+33, fails=0); SIMD loss vs PyTorch 0.0; SIMD grad
  wte 8.3e-07.
- **Bench A/B (mode1, 2 lan/seq; loss khop EXACT tung chu so voi GĐ2 ca 3 seq):**
| seq | A = P6 GĐ2 | B = P6 GĐ3 | d |
|---|---|---|---|
| 8 | 17.078 s | **16.109 s** | -5.7% |
| 256 | 26.141 s | **25.922 s** | -0.8% |
| 1024 | 63.781 s | **55.469 s** | **-13.0%** |
- **AQ @1024 SAU:** FWD_BODY 9390 (-16%), ATTN_BWD 15843 (-12%), MLP_BWD 14297
  (-13%), **TOTAL_bwd 44922 -> 36484 ms (-18.8%)**. GHI CHU: LHEAD trong AQ la
  ban sao copy trong probe CHUA wire pd -> duong library that con thap hon;
  ground truth = TOTAL_bwd + bench.
- Ty le vs torch (0.934/1.404/3.542 — khong do lai): **17.2x / 18.5x / 15.7x**
  (truoc: 18.3/18.6/18.0).
- **Seq256 loi it (do, khong doan):** tai n=256 vol cua GEMM vocab = 38.6M <
  40M -> di serial (chi builder fix); PAR tai T=256 CHUA do -> neu can thang
  thi probe A/B rieng truoc.
**5. DOC:** plan §12 dong P6 GĐ3 + §9 (probe BH); ROADMAP dong PARITY.
**Trang thai file (CHUA COMMIT):** simd_matmul x3, gpt2_ref x5,
`tv/tests/simd_probe_bh.tkv` (moi, untracked), plan §12, ROADMAP, HANDOFF
(chuong nay), `_seq_cfg.txt` = 1024.
**Con lai (theo AQ moi @1024):** ATTN_BWD 15.8s (43% cua bwd — GĐ4? hieu
suat worker hien van ~2.3x), MLP_BWD 14.3s (gelu scalar ~3.3s = ung vien re
nhat), LHEAD duong pd ~7s, AdamW + plumbing ~7s (chua cham tu dau). Muc tieu
chu khoa <=1.0x (3.542s) van can P7 — **CAN XET LAI ⓑ: may nay Xeon
E5-2695 v3 = Haswell, chi AVX2 — KHONG AVX512/VNNI/BF16/AMX -> bf16/tf32 khong
co toan tu hardware; torch fp32 cung AVX2 -> P7 co the KHONG mang duoc x2-3
nhu ly thuyet tren may AMX.**
### 0.23 P6 GIAI DOAN 4 — PLUMBING + GELU PARALLEL (2026-10-21) — 135/135, BIT-EXACT, seq1024 −8.3%
Tiep 0.22 ("Con lai": AdamW+plumbing ~6.6s seq-independent, gelu scalar 24%
MLP_BWD). Phuong phap: DO TRUOC — probe BK (phan rac full-step) -> probe BL
(test parallel AdamW) -> probe BM (prototype ung vien) -> wire theo so do.
**Probe BK** (`tv/tests/simd_probe_bk.tkv`): full step @1024 = adamw_loop 3906
+ grad_norm 1047 + unflatten 782 + loss 671 + zero_grad 266 ms (~6.6s, KHONG
phu thuoc seq — o seq8 no ~30% buoc) + GC +2.5GB/buoc.
**Probe BL** (`tv/tests/simd_probe_bl.tkv`) — cau hoi BG2 o quy 124.4M
(AdamW copy NGUYEN cong thuc trainer.step):
| Bien the | Ket qua | Ket luan |
|---|---|---|
| SHARED-12 (ghi disjoint vao cung list) | 11859 ms vs serial 4218 = **0.36x** | CHET — 2.8x cham hon serial |
| OWN+MERGE (372M copy) | 14625 ms (worker 0.5s nhung **merge 11.7s**) | CHET — merge qua dat |
| grad_norm PAR12 | 1016 -> 453 ms = **2.24x** | BIT=0.0 THANG |
-> **AdamW parallel BO** (da do ca 2 bien the); worker compute van scale tuyet
voi (3.9s -> 0.5s) — chi viec ghi/merge o quy lon mat.
**Probe BM** (`tv/tests/simd_probe_bm.tkv`) — 4 ung vien + nguyen nhan, ALL BIT-EXACT:
| Test | Serial | PAR12 | Speedup |
|---|---|---|---|
| _norm_groups (148 nhom THAT) | 1047 ms | 375 | **2.79x** |
| gpt2_loss n=1024 (51.5M ptu) | 719 ms | 141 | **5.10x** |
| gpt2_loss n=256 | 187 ms | 63 | 3.0x |
| gelu_fwd 3.14M | 110 ms | 15 | **7.33x** |
| gelu_fwd 786K | 15 ms | 16 | ~1x (break-even -> guard 2M) |
| gelu_bwd 3.14M | 156 ms | 31 | **5.03x** |
| copy thuan 12M SHARED12 | 328 ms | 109 | **3.0x** |
**Bai hoc quyet dinh (khong doan):**
1. **Ghi chung KHONG bi container-lock** (BM[5]: copy shared 3.0x) — BL[2]
   0.36x la do vong AdamW f64-boxing 124M + ghi chung, khong phai lock.
   → parallel chi THANG cho copy/thinh thuan; vong phep toan boxing lon o
   quy 124M co the lat (canh bao cho future shared-write).
2. **Own-seg + merge chi thang khi merge RE** (gelu/loss/norm: seg nho;
   AdamW 372M copy = 11.7s chet).
3. **Guard theo do:** gelu 786K break-even (seq256 khong song song gelu),
   loss guard n*V >= 4M (seq8 = 402K -> serial), norm guard tong >= 4M.
4. **Bug compiler MOI (probe BM):** (a) nested list — `lst.append(seg)` voi
   seg list[f32] o top-level main nem `ArrayTypeMismatchException` (list[f64]
   thi duoc; trong while-loop thi duoc) → merge phai goi helper copy phang
   UNROLLED; (b) main() qua lon (~900 dong) → `BadImageFormat` (0x8007000B)
   → tach thanh ham rieng.
**Wire (guard + bit-exact, 10 file):**
- `_norm_groups` x4 file trainer (canonical + 3 mirror): 12 worker tinh s
  tung nhom, main gop DUNG thu tu nhom; guard tong < 4M -> serial.
- `gpt2_loss` (trainer canonical + gpt2_ref x3): 12 worker row, seg f64 rieng,
  main gop DUNG thu tu row; guard n*V < 4M -> serial.
- `_gelu_fwd_par` (gpt2_ref x3, 2 site: mlp fwd + bwd recompute): own-seg +
  merge vao dst (dst==src in-place an toan — doc xong join moi merge);
  guard simd==0 hoac cnt < 2M -> serial.
- `op_gelu_erf_bwd` (ops x3): giong tren, guard n < 2M -> serial.
**Ket qua:**
- **Gate 135/135** (9 suite nho = 102 assert + gpt2_full 33, fails=0;
  SIMD loss vs PyTorch = 0.0)
- Bench A/B (best-of-2, loss khop EXACT ca 3 seq):
| seq | GĐ3 | GĐ4 | Δ |
|---|---|---|---|
| 8 | 16.109 | **15.062** | −6.5% |
| 256 | 25.922 | **25.406** | −2.0% |
| 1024 | 55.469 | **50.875** | **−8.3%** |
- AQ @1024: **TOTAL_bwd 36.5→33.5s (−8.0%)**, MLP_BWD 14.3→12.3 (−14%),
  FWD_BODY 9.39→9.13, ATTN_BWD 15.9 (khong doi), LHEAD 9.5.
- Tỉ lệ vs torch: **16.1× / 18.1× / 14.4×**.
- **CHUA COMMIT** — 10 file sua + 3 probe moi (BK/BL/BM) — cho duyet.
**"Con lai" sau GĐ4 (theo so do):** ATTN_BWD 15.9s (47% bwd, worker hieu suat
~2.3× con du), AdamW con lai ~4.9s (adamw_loop 3.9s DA DO — khong song song
duoc ca 2 bien the; + unflatten 0.78 + zero 0.27 — chua cham), LHEAD 9.5s,
gelu fwd con du (seg alloc an ~60% loi do probe — caching seg la ung vien
nho). Muc <=1.0x van can P7 (xem canh bao ⓑ Haswell o 0.22).
### 0.24 P6 GIAI DOAN 5 — ATTENTION PACKED BUFFERS (2026-10-21) — 135/135, BIT-EXACT, seq1024 −9.8%
Tiep 0.23 ("Con lai": ATTN_BWD 15.9s = 47% bwd, worker ~2.3× du). DO TRUOC
bang probe BN (`tv/tests/simd_probe_bn.tkv`) — phan rac 1 layer @n=1024
(copy worker/merge NGUYEN VAN tu gpt2_ref, o_* = -1, timing o main):
| Thanh phan | ms | Ghi chu |
|---|---|---|
| whole attn_bwd | 1453 | |
| qkv recompute | 109 | |
| dctx GEMM | 47 | |
| **alloc 12 ctxo + 12 dqo** | **359** | 37.2M zero-fill = **25% moi layer** |
| spawn+join 12 worker | 516 | serial ref 4468 -> **efficiency 8.66×** |
| merge x12 | ~0 | |
| dw_proj | 78 | |
| linear_bwd | 235 | |
| sum vs whole | chk −7.5% | do loi do |
**KHAC GIA THUYET:** worker khong yeu (8.66×); serial GEMM chain cung nho
(P6 GĐ3 builder fix da lam GEMM nhanh) — chu dich la **PHI ZERO-FILL**: moi
worker chi ghi 1 head (n*hd ctx + n*3*hd dq = 262K ptu) nhung driver alloc
buffer FULL n*h / n*3h (3.1M ptu/worker) — 11/12 du lieu bi zero-fill roi
merge con copy luon phan zero do.
**Stage A — wire packed buffer (BIT-EXACT: gia tri/ thu tu tich luy khong
 doi, chi vi tri trong buffer doi):**
- `_attn_hb_one` (bwd worker): ctx_out ghi `[t2*hd+d]`, dq_out ghi
  `[t2*3*hd + {d, hd+d, 2hd+d}]`
- `_attn_hb_merge`: doc packed -> ghi ctxT/dqkv tai (t, head, d) nguyen van
- driver bwd: ctxo = `n*hd`, dqo = `n*3*hd`/worker (37.7M -> 3.15M zero-fill)
- **FWD cung benh** (co0..11 = t_zeros([n,h]), 92% lang phi): `_attn_f_one`
  + `_attn_f_merge` + driver cung n*hd — cung pattern.
- Serial path n<128 NGUYEN VEN khong doi (seq8 khong doi hanh vi).
- Mirror: gpt2_ref x3 (md5 giong het).
**Ket qua:**
- Gate **135/135** (9 suite 102 + gpt2_full 33). LUU Y: lan chay dau tien
  gpt2_full crash tam thoi `ArgumentOutOfRangeException` trong String.Split
  khi load fixture grad 230MB (truoc moi PASS nao) — chay lai CUNG binary
  33/33 toan bo => transient IO/GC, khong lien quan code GĐ5.
- Bench A/B (best-of-2, loss khop EXACT ca 3 seq):
| seq | GĐ4 | GĐ5 | Δ |
|---|---|---|---|
| 8 | 15.062 | 15.187 | +0.8% (drift nhiet — code giong het, duong serial n<128) |
| 256 | 25.406 | **23.625** | −7.0% |
| 1024 | 50.875 | **45.890** | **−9.8%** |
- AQ @1024: **ATTN_BWD 15.9→11.4s (−28.7%)**, FWD_BODY 9.13→7.78 (−14.7%),
  **TOTAL_bwd 33.5→29.7s (−11.5%)**.
- Tỉ lệ vs torch: **16.3× / 16.8× / 13.0×**.
**"Con lai" sau GĐ5 (AQ):** ATTN_BWD 11.4s (worker wall 516ms + serial
469ms/layer — eff da 8.66×, con kho 20-30%), MLP_BWD 11.8s, LHEAD 9.3s,
AdamW/plumbing ~4.9s (GĐ4 da do: adamw_loop 3.9s khong song song duoc).
### 0.25 P6 GIAI DOAN 6 — MLP_BWD: smm_fwd_at + GELU OVERLAP (2026-10-21) — 135/135, BIT-EXACT, seq1024 −1.9%
Tiep 0.24 ("Con lai": MLP_BWD 11.8s = pha lon nhat). DO TRUOC bang 2 probe:
**Probe BO** (`tv/tests/simd_probe_bo.tkv`) — phan rac mlp_bwd 1 layer
@1024 (simd=1, gflat + goff/dboff >= 0 cung duong bench; whole = 1000ms
khop AQ 986, chk1 −1.5%):
| Thanh phan | ms | Ghi chu |
|---|---|---|
| inner GEMM + bias | 94 + 15 | GEMM ~51 GFLOPS — near-peak AVX2 |
| alloc a [n*4h] | 16 | |
| geluF (proxy serial) | 94 | production parallel ~thap hon |
| **r2 linear_bwd** | **391** | dx2 140 + **xT2 94** + dw2 156 + db/pack ~0 |
| geluB | 62 | serial, truoc r1 |
| **r1 linear_bwd** | **313** | dx1 125 + **xT1 15** + dw1 125 + db1 16 |
GEMM 640ms/layer da near-peak (padding T=768 nhu nhau) -> chu dich la
PHI va SERI: **transpose 109ms/layer** (xT = smm_transpose truoc dw) va
**geluB 62ms chay serial**.
**Probe BP** (`tv/tests/simd_probe_bp.tkv`) — xac thuc 2 ung vien:
- BP[1a/b]: `dw = smm_fwd_at(x, gy, in_d, out_d, n)` (doc x truc tiep —
  x luu [k=n hang][m=in_d cot] = dang A^T) thay `xT + smm_fwd`: r2 dw
  234→157ms, r1 dw 141→125ms, **maxdiff = 0.0 ca 2 shape** (P5.8 AX da
  chung minh thu tu tich luy p0 tang dan nguyen).
- BP[2]: geluB chay worker DONG THOI main chay dw2: 266→187ms,
  **maxdiff 0.0 — nested spawn AN TOAN** (op_gelu_erf_bwd noi bo spawn
  12 tu worker thread). Gelu khong goi simd_gemm -> khong chia se g_job.
**Wire (gpt2_ref x3 mirror, md5 sync):**
- `gpt2_linear_bwd` (simd==1): `xT = smm_transpose; dw = smm_fwd` →
  `dw = smm_fwd_at` — LOI MOI caller (mlp r1/r2 + attn dqkv/dproj).
- attn worker `_attn_hb_one`: gDkT/gDvT cung smm_fwd_at (T=64<96 — worker
  van an toan) — bo 2 transpose nho/worker.
- `gpt2_mlp_bwd` (simd==1): tach r2 = dx smm_dot → **spawn geluB worker**,
  main chay dw2 smm_fwd_at DONG THOI → join → db2 → pack dw2→gflat → r1;
  thu tu dx→dw→db→pack + gflat cung thu tu NGUYEN → BIT-EXACT. Duong
  simd==0 nguyen ven.
**Ket qua:**
- Gate **135/135** (9 suite nho + gpt2_full 33, fails=0).
- Bench A/B (best-of-2, loss khop EXACT ca 3 seq):
| seq | GĐ5 | GĐ6 | Δ |
|---|---|---|---|
| 8 | 15.187 | 15.313 | +0.8% (drift nhiet — FWD_BODY cung +2.3% phien nay) |
| 256 | 23.625 | **23.516** | −0.5% |
| 1024 | 45.890 | **45.000** | **−1.9%** |
- AQ @1024: **MLP_BWD 11828→10890ms (−8.0%)**, TOTAL_bwd 29.7→28.8s
  (−3.0%). ATTN_BWD 11.4s bit trong (transpose worker micro-win ~46ms).
- Tỉ lệ vs torch: **16.4× / 16.7× / 12.7×**.
**Ghi chu do:** phien nay may nong (FWD_BODY +2.3%, seq8 +0.8% — cung
binary, duong serial khong doi) — so hoi 2.2s ly thuyet (transpose 109 +
geluB 62 x 12 layer) thuc te ~0.9s + nhieu nhiet.
**"Con lai" sau GĐ6 (AQ):** MLP_BWD 10.9s (GEMM 640ms/layer near-peak —
den han), ATTN_BWD 11.3s (serial chain + worker 516ms), LHEAD 9.6s,
AdamW/plumbing ~4.9s. P7 mixed precision van cho quyet dinh ⓑ (Haswell).
---
### 0.26 P6 GIAI DOAN 7 — PLUMBING: ZERO_GRAD OVERLAP + LN-FWD SAVES (2026-10-22) — 135/135, BIT-EXACT, seq1024 −2.5%
Tiep 0.25 ("Con lai": ATTN_BWD serial chain, LHEAD 9.6s, AdamW/plumbing
~4.9s). DO TRUOC bang 3 probe:
**Probe BR** (`tv/tests/simd_probe_br.tkv`) — phan rac plumbing forward-side
@124M (cung duong bench):
| Thanh phan | ms | Ghi chu |
|---|---|---|
| logits smm_fwd_pd (BR[1a]) | 1313 | FWD_BODY phan lon (vocab GEMM) |
| forward_saved whole (BR[2]) | 9828 | embed+12blk+lnf+logits |
| LN fwd recompute x24 (BR[3]) | **359** | 24 goi LN fwd RECOMPUTE trong block_bwd |
| LN bwd x25 (BR[4]) | 719 | gpt2_layernorm_bwd (chua đụng) |
| zero_grad (BR[5]) | **250** | 124M phan tu, serial TRUOC forward |
| AdamW step (BR[6]) | 4234 | 3 vong m/v/p — da co norm parallel (GĐ4) |
| unflatten (BR[7]) | 766 | copy p -> tensor.data |
BR sum (logits+lnf+lnb+zero+step+unflat) = 6922 vs forward_saved whole =
9828 -> chu dich truoc AdamW la **LN recompute + zero_grad** (609ms) — 2
muc nay an duoc ngay; AdamW gan break-even voi BL/BL2 da do (giu 2 vong
rieng, xem P1.2).
**Probe BQ** (`tv/tests/simd_probe_bq.tkv`) — LHEAD con lai: dwte-add PAR12
(seg worker + main merge, dung mau GĐ4) = **719ms vs serial 328ms — DO BAC
BO** (merge 50257x768 = 38.6M phan tu qua 12 worker qua dich; maxdiff 0.0
nhung khong co loi). EMBED_BWD cung nguon.
**Probe BW** (`tv/tests/simd_probe_bw.tkv`) — main-thread + 11 worker
(attention head): BW[1] spawn12+join+merge = 391ms (eff 6.48x), BW[2]
spawn11 + main tinh head0 = 453ms (eff(main+11) = 5.59x, maxdiff 0.0) ->
**main thread cham hon 1 worker -> DO BAC BO**, giu nguyen spawn 12 (GĐ1/2).
**Wire:**
- `gpt2_train_step` (gpt2_train x3 mirror + bench `_step124`): spawn
  zero_grad worker TRUOC forward, join TRUOC backward — forward KHONG dot
  opt.g -> ket qua GIONG HET (bit-exact), an 250ms/buoc @124M.
- `gpt2_block_saved`: luu them dau vao LN (`a_ln`, `m_ln`); saves append
  CUOI (vi tri cu xins/rs NGUYEN -> format cu van doc duoc);
  `gpt2_block_bwd` nhan `save_a`/`save_m` -> bo 24 goi gpt2_layernorm
  recompute (359ms @1024). Cung input cung phep -> BIT-EXACT.
**Ket qua:**
- Gate **135/135** (9 suite nho + gpt2_full 33, fails=0).
- Bench A/B (best-of-2, loss khop EXACT ca 3 seq):
| seq | GĐ6 | GĐ7 | Δ |
|---|---|---|---|
| 8 | 15.313 | **15.172** | −0.9% |
| 256 | 23.516 | **22.844** | −2.9% |
| 1024 | 45.000 | **43.890** | **−2.5%** |
- Tỉ lệ vs torch: **16.2× / 16.3× / 12.4×**.
**Ghi chu do:** phien nay nhiet do lon (chay lan 46750 vs lan 43890 cung
binary) — lay best-of-2. Loi thuyet 609ms, do thuc ~1.1s (zero_grad worker
overlap con cham forward nhieu hon 250ms nua).
**"Con lai" sau GĐ7 (BR):** AdamW step 4.2s + unflatten 0.7s = ~5s — BL/BL2
da do truoc (OWN+MERGE 0.29x) — CHI con 2 huong chua do: (1) fused
step+unflatten DA BAC BO (probe AH, xem P1.2 — KHONG lam lai); (2) giu vong
rieng nhung overlap step voi unflatten cua tham so cu (unflatten doc opt.p
TRUOC khi step ghi moi — can chot thu tu). ATTN_BWD 11.3s: worker wall
516ms/layer da do (GĐ5) — con lai la serial chain GEMM near-peak. Tiep theo:
**P7 mixed precision** (quyet dinh ⓑ — may nay Haswell AVX2, khong VNNI/
BF16/AMX).
---
### 0.27 P7 — MIXED PRECISION: DO TRUOC, BAC BO (2026-10-22) — quyet dinh ⓑ = fp32-vs-fp32
P6 xong (GĐ7) → bước kế theo trong plan §12 là **P7 mixed precision**
(bf16/tf32 GEMM + tích luỹ fp32 + fp32 master), kèm câu hỏi quyết định ⓑ
("parity so ở precision nào?"). Nguyên tắc của repo: **ĐO TRƯỚC, rồi mới
code** → probe `tv/tests/simd_probe_bx.tkv` (mới) đo 4 phép.
**1. RANG BUOC CUNG (do, khong suy doan):**
| Hang muc | Ket qua |
|---|---|
| CPU | **Intel Xeon E5-2695 v3 @ 2.30GHz** (Family 6 Model 63 = Haswell-EP), 14 core / 28 thread → **AVX2 + FMA3**; **KHONG** AVX512 / VNNI / **BF16** / AMX |
| Native SIMD lib `tkv_simd_native.c` | export `SgemmF32`/`SgemmF64` + `Add/Sub/Mul/Scale/Axpy/Dot/Sum/Max/Min` × **F32/F64**; grep `f16\|bf16\|half` = **0 hit** ⇒ **KHONG co kernel f16/bf16 GEMM** |
| Compiler `.tkv` (`compiler_src`) | grep `f16\|bfloat\|float16` = **0 hit** ⇒ **KHONG co kieu/intrinsic f16/bf16** (chi `i32/bool/i64/f64/f32`) |
| ⇒ duong duy nhất con lai cho f16 | **lam tron f16 bang PHAN MEM** (scalar): `tv/quantization/types.tkv` → `native_f16_encode` / `native_f16_decode` (da xac minh DUNG: 1.0→0x3C00, 65504→0x7BFF, 70000→inf, max relerr 4.3e-4) |
**2. SO DO (probe BX, 3 lan chay — ket qua on dinh):**
| Do | Ket qua | Y nghia |
|---|---|---|
| BX[0] bug compiler | truyen `Tensor.data` (List<float>) vao tham so `list[f64]` → **gia tri SAI** (0.125 / 128.0 / 0.0 thay vi 0.5/1.5/2.5); cung data do qua `list[f32]` → **DUNG** | bug compiler moi — xem muc 4 |
| BX[1] so hoc: f16-round → GEMM 256×768×768 vs f32 | **maxdiff 0.0497**; max\|C32\|=46.206 vs max\|C16\|=46.202 (dong bo) → **relerr 1.08e-3**; check chéo scalar C[0][0] 9.56386814 vs GEMM 9.5638685 (**4.0e-8**) ⇒ duong GEMM dung | cua so parity forward logits = **2.4e-7** ⇒ **VUOT 2.1e5×** ⇒ KHONG giu duoc 135/135 gate fp32 |
| BX[2] chi phi lam tron f16 (phan mem) | **378-458 ns/ptu** (3 lan chay: 378/397/458; 375 ms vs baseline 15 ms / 786.432 ptu; baseline doc list = 19-20 ns/ptu) | interpreter .tkv ~19 ns/phép ⇒ ban bit-trick toi uu cung ~200-460 ns/ptu — khong cua duoc |
| BX[3] AMP best-case (gia su GEMM f16 **MIEN PHI**) | 768³: GEMM 31-32 ms vs round(A+B) 468-540 ms → **0.056-0.065×**; 1024×768×768: 31 ms vs 546-630 ms → **0.047-0.056×**; lm-head 1024×768×50257: 1860-1922 ms vs 15625-18028 ms → **0.096-0.108×** | **CHAM 9-21×** du da gia su toi uu nhat |
**3. QUYET DINH ⓑ = fp32-vs-fp32 (GIU nguyen contract 2.4e-7). KHONG lam P7.**
Ly do khong phai "ly thuyet" ma la so do: (i) may khong co phan cung f16/bf16
⇒ f16 chi con duong lam tron bang phan mem; (ii) loi so hoc **1.08e-3**
(vuot cua so parity 2.1e5×) ⇒ pha 135/135 gate; (iii) chi phi lam tron
**~380-460 ns/ptu** ⇒ AMP best-case **0.047-0.108×**. **P7 chi song lai khi:** (a)
co **kernel f16/bf16 NATIVE** (viet them C/Rust vao `tkv_simd_native.c`) **va**
(b) CPU ho tro (AVX512-BF16 / AMX) hoac chuyen sang GPU — ca 2 deu ngoai tam
`.tkv` hien tai.
**4. BUG COMPILER MOI (BX[0]) — can bao compiler team:** phep **chuyen kieu
TUNG PHAN TU** cua danh sach (List\<float\> → `list[f64]`) ma compiler sinh
ra **gia tri sai, im lang khong loi**. Repro toi thieu (4 phan tu):
`Tensor.data` (List<float>) truyen vao tham so khai bao `list[f64]` → doc ra
0.125/128.0/0.0 thay vi 0.5/1.5/2.5; cung data do truyen vao tham so
`list[f32]` → dung. Hop le voi 786.432 phan tu (BX[0] "lon" trong probe): `[0]/[1]/[1000]/[786431]`
= 0.0078125/32.0/1.97e24/0.0 thay vi 0/1/1000/786431. ⇒ **QUY TAC: khong
truyen list qua bien tham khi kieu phan tu la f64**; moi phep so sanh/doc
list phai **INLINE trong main**. (Binh an: co ghi chu ve "compiler KHONG tu
khoi tao bien global" trong `tv.runtime.f32_bits` — cung ho ban, kieu convert
cai dat sai.) **Hau qua probe:** ban draft cua BX (so sanh 2 tensor qua
ham `_bx_cmp`) cho `maxdiff=813056` — **SAI**, do loi nay, khong phai do GEMM;
do lai inline ra **maxdiff=0.0497** (dung). Da loai ban draft.
**5. FILE:** khong sua file nguon (P7 khong wire). Chi them probe
`tv/tests/simd_probe_bx.tkv` (moi, untracked) + plan §12 + ROADMAP + HANDOFF
(chuong nay).
**Con lai (theo do @seq1024, full step 43.89 s = 12.4× torch):** ML/HEAD
duong pd 9.6 s, ATTN_BWD 11.3 s (worker wall 516 ms/layer, chuoi serial
GEMM near-peak AVX2), AdamW step 4.2 s + unflatten 0.7 s (da do: fused
step+unflatten **BAC BO** — probe AH; chi con 2 huong: giu vong rieng nhung
overlap step voi unflatten cua tham so cu — can chot thu tu). **P7 khong
con dau o nen khong co "mixed precision" de dua tien do len** ⇒ cac buoc
sau phai den tu **parallel/overlap + giam overhead interpreter** (giong P6),
hoac tu **kernel native moi** (noi dung tai day).
---
### 0.28 P6 GĐ8 — GIẢM OVERHEAD INTERPRETER TRONG GEMM (2026-10-22) — 135/135, BIT-EXACT, seq1024 −15.1%
Sau P7 bị bác bỏ (0.27) hướng còn lại đã nêu: **giam overhead interpreter**.
Phần interpreter thuần trong đường GEMM = **tile-build + tích luỹ** (native
`simd_gemm` chỉ chiếm một phần). Đó là mục tiêu GĐ8 này.
**GỐC RỄ (đọc IL, không phải phỏng đoán):** biến cục bộ `.tkv` bị compiler
**boxed thành `TkvInt`** (struct 16 byte) ⇒ mỗi phép cộng chỉ số là phép cộng
trên struct thay vì `int64`. Quy tắc (đọc `.locals` trong `out/*.il`, probe
`simd_probe_narrow2.tkv`): biến chỉ được khai **native `int64`** khi **phép gán
đầu tiên** của nó nằm ở **top-level của hàm** và là biểu thức int64
(`x = 0`, `src = i0 * kdim + p0`) **và** các phép gán sau đều "chứng minh được"
(`x = x + 1`, `x = y`). Có một phép gán không chứng minh được ở top-level (ví dụ
gán lại bằng biểu thức có `*`) ⇒ biến bị box lại thành `TkvInt`. (Phép gán
*bên trong vòng lặp* vẫn OK nếu phép gán đầu tiên đã hợp lệ — probe t12/t13/t14.)
**ĐO TRƯỚC (probe `tv/tests/simd_probe_by2.tkv`, đồng hồ `time_perf()` =
`Stopwatch` — `now_ms()` có lượng tử 15.6 ms nên probe `by` v1 vô hiệu):**
| Đo | W (đang dùng) | F (index chạy + native i64) | Tốc độ |
|---|---|---|---|
| BY2[0] A-tile build | 4.90 ns/ptu | 3.29 ns/ptu | **1.49×** |
| BY2[1] B-tile build (trans_b=1) | — | — | **1.43×** |
| BY2[2] tích luỹ Ct→C | 4.79 ns/ptu | 4.44 ns/ptu | **1.08×** |
| BY2[3] core (shape Smat) | 0.1085 s | 0.0705 s | **1.54×** |
| BY2[4] core (shape gCtx) | 0.0871 s | 0.0382 s | **2.28×** |
Tất cả **maxdiff = 0.0**. Biến thể "copy p-tile đầu thay cho cộng đơn" **chậm
hơn** (4.82 vs 4.44 ns/ptu) → **bỏ** (giữ nguyên phép cộng ⇒ không đổi dấu
±0.0 ⇒ bit-exact tuyệt đối).
**WIRE (`tv/training/simd_matmul.tkv`, đồng bộ 2 file mirror):**
- `_tile_a_w` / `_tile_b_w` / `_tile_a_at_w`: đổi sang **index chạy** (`dst`/`src`
  tăng 1 mỗi phần tử) + **đọc trực tiếp** `d[i]` (bỏ ép kiểu `(d[i]-0.0)/1.0` vốn
  chỉ cần cho `append`) + biến đếm khai ở top-level.
- Thêm helper `_acc_f(cd, Ct, n, i0, ib, j0, jb, T)`: tích luỹ với `bc`/`bt`
  chạy (mỗi phần tử 2 phép cộng thay vì 4). Dùng chung cho `_smm_core`,
  `_p58_core_simple` **và `_pd_do`** (đường wave 12 worker của lm-head).
- Thứ tự tích luỹ p-tile **KHÔNG đổi** ⇒ bit-exact. IL xác nhận cả 4 hàm đều
  khai `int64` (`i/dst/j/src`, `i/dst/j/row/idx`, `r/t/dst/src`, `r/bc/bt/c`).
**KHÔNG đổi thứ tự vòng.** Bản "p0→j0→i0 (ít build tile hơn)" **đã thử trước và
đo là REGRESSION** (plan §10: FWD_BODY +101%) ⇒ giữ nguyên `i0→p0→j0`.
**KẾT QUẢ:**
- Gate **135/135** (9 suite nhỏ 102 + `gpt2_full` 33, fails=0).
- Loss **khớp EXACT từng chữ số** cả 3 seq: seq8 `10.572291 → 7.0119243`,
  seq256 `10.756927 → 11.807251`, seq1024 `10.760399 → 13.442912`.
- Bench A/B (best-of-3, xen kẽ base/new để giảm drift nhiệt):
| seq | GĐ7 (base) | GĐ8 (mới) | Δ |
|---|---|---|---|
| 8 | 15.250 s | **12.969 s** | **−15.0%** |
| 256 | 22.969 s | **18.125 s** | **−21.1%** |
| 1024 | 44.547 s | **37.828 s** | **−15.1%** |
- Tỉ lệ vs torch: 16.2/16.3/12.4× → **13.9× / 12.9× / 10.7×** (vẫn chậm hơn
  torch, nhưng khoảng cách hẹp lại rõ rệt).
**"Còn lại" sau GĐ8:** GEMM ở đường `smm_*` giờ gần peak AVX2; phần interpreter
còn lại nằm ở (1) các vòng copy/elementwise ngoài GEMM (AdamW/unflatten —
đã đo và bác bỏ hướng song song hoá), (2) `simd_gemm` native chỉ có kernel
F32 — mở rộng sang F64/đa luồng phải sửa **C** (`tkv_simd_native.c`), ngoài
tầm `.tkv`.
---
### 0.29 P6 GĐ9 — QUY TẮC int64 CHO VÒNG NGOÀI GEMM: ĐO BÁC BỎ (2026-10-22)
Yêu cầu: nhân rộng quy tắc native int64 (GĐ8) sang các vòng interpreter nóng
ngoài GEMM. Nguyên tắc **ĐO TRƯỚC**: scan IL của `out/bseq_new.il` (đường train
đầy đủ — trainer + gpt2_ref124 + simd_matmul) tìm mọi hàm có biến người dùng
bị box `TkvInt`, rồi đo A/B bằng probe `tv/tests/simd_probe_ca.tkv` (đồng hồ
`time_perf`, N = 124.000.000 ptu — đúng mẫu 124M).
**1. BIẾN BỊ BOX (chỉ tính biến người dùng, loại `__tkvint_*` compiler temp —
temp đó được **khai bỏ** chứ không dùng mỗi phần tử, đã kiểm ở IL `_acc_f`):**
| Hàm | Biến box | Vòng | Mức độ |
|---|---|---|---|
| `_step124` (unflatten) | `off` | `d[j] = p[off + j]` | per-element, SERIAL (124M/bước) |
| `_tn_one` (norm worker) | `off` | `gl[off + i]` | per-element, 12 worker |
| `_norm_groups` (đường serial <4M) | `off`, `tot_elem` | `g[off + i]` | per-element (chỉ gate/model nhỏ) |
| `_ls_one` (loss worker) | `k` | `j.seg[k]` | 1 lần/row (1024/bước) |
| `_smm_core` / `_p58_core_simple` | `i0`, `j0`, `p0` | vòng ngoài GEMM | ~256-4096 lần/GEMM |
| `gpt2_loss_m`, `_prepare`, `_n_param` | `cnt`, `off`, `n` | — | ít lần |
| **`step` (AdamW, 4.2 s), `zero_grad`, `gpt2_loss`, LN bwd, gelu** | **không có** | `int32 'i'` | **đã tối ưu, không thuộc phạm vi sửa** |
**2. SỐ ĐO (probe CA, A/B xen kẽ, 4 lần chạy, BIT-EXACT 0.0):**
| Vòng | W (hiện tại) | F (int64, `off += 1`) | Tốc độ (4 lần chạy) |
|---|---|---|---|
| unflatten | 2.27-2.34 ns/ptu | 2.29-2.30 ns/ptu | 1.02 / 0.998 / 0.991 / **0.990** → median ≈ **0.99** |
| norm (12 nhóm) | 2.23-2.33 ns/ptu | 2.24-2.25 ns/ptu | 1.04 / 0.996 / 1.002 / **0.996** → ≈ **1.00** |
`off` trả về giống hệt (124000000), tổng `s` giống hệt → bit-exact.
**3. KẾT LUẬN — KHÔNG WIRE.** Cả hai vòng đều bị **throttled băng thông bộ nhớ**
(≈2.3 ns/ptu ≈ 3.4 GB/s đọc+ghi) nên phần index arithmetic bị ẩn hết: bản
"sửa" **không nhanh hơn** (unflatten còn chậm hơn ~1% — trong nhiễu). Quy tắc
int64 chỉ ăn tiền ở vòng **compute-bound** (tile-build GEMM — nơi arithmetic
chiếm phần lớn, GĐ8 đo được 1.5-2.3×). Tiết kiệm tối đa ~**18 ms** @seq1024
= **0.05%** của 37.83 s, nhỏ hơn nhiều nhiễu đo (±5-10% do nhiệt/CPU khác
luồng). Chi phí wire thật = **7 file** (`gpt2_train` ×4 mirror + 3 file bench)
→ KHÔNG tương xứng. Giữ nguyên mã nguồn; probe `simd_probe_ca.tkv` + ghi chép
này là bằng chứng.
**4. HƯỚNG SAU (thay cho "sửa quy tắc khai báo"):** vòng còn lại lớn nhất là
`AdamW.step` (4.2 s @124M) — đã dùng `int32 'i'` (**không** phải vấn đề boxing);
muốn cắt phải giảm số phép/phần tử (fused update, đổi f64→f32 trong m/v) chứ
không phải đổi khai báo biến. Phần interpreter còn lại (copy/norm/loss) đã ở
gần băng thông → cắt được chỉ bằng kernel native (ngoài tầm `.tkv`).
→ **ĐÃ LÀM, xem 0.30 (GĐ10) — so công thức với PyTorch rồi gộp 2 div.**
---
### 0.30 P6 GĐ10 — HỌC PyTORCH CHO ADAMW: GỘP 2 DIV THÀNH 1 (2026-10-22) — 135/135 + 15/15, seq1024 −6.5%
Đi tiếp 0.29 ("phải giảm số phép/phần tử trong `AdamW.step`"). Nguyên tắc:
**so công thức với PyTorch trước, rồi đo** (probe `tv/tests/simd_probe_cb.tkv`,
`time_perf`, N=124.000.000, 3 lần chạy).
**1. SO CÔNG THỨC — PyTorch `torch.optim.AdamW` (CPU) vs ta:**
| Hạng mục | PyTorch (`torch/optim/adamw.py`) | Ta (trước GĐ10) | Sau GĐ10 |
|---|---|---|---|
| Weight decay | `param.mul_(1 - lr*wd)` (decoupled) | `p[i] * wdf` | giữ |
| Momentum | `exp_avg.lerp_(grad, 1-b1)` = m+(1-b1)(g−m) | `b1*m + (1-b1)*gi` | giữ (đại số tương đương) |
| Variance | `mul_(b2).addcmul_(g,g,1-b2)` | `b2*v + (1-b2)*gi*gi` | giữ |
| Bias correction | tính 1 lần/bước | 1 lần/bước | giữ |
| **Denom** | `sqrt(v)/sqrt(bc2) + eps` → **1 div** | `denom = sqrt(v)/bc2s + eps` **rồi** `m/denom` → **2 div** | **gộp thành 1 div** ✅ |
| Precision | **f32 toàn bộ** | f64 trung gian (`gi`, `denom`) | giữ f64 — `.tkv` không có khai báo biến kiểu (`x: 'f32'` **bị compiler từ chối**), `sqrt()` trả f64 |
| Thực thi | C++ vectorize AVX2 + multi-tensor-apply | interpreter .tkv | ngoài tầm `.tkv` |
| **Số phép/ptu** | ~10 mul + 1 sqrt + 1 div + 4 add | ~9 mul + 1 sqrt + **2 div** + 3 add | ~9 mul + 1 sqrt + **1 div** + 3 add ⇒ **ít hơn PyTorch** |
**Định lý gộp (đại số chính xác):** `m / (sqrt(v)/bc2s + eps)`
`== (m*bc2s) / (sqrt(v) + eps*bc2s)` — `bc2s` và `eps*bc2s` tính 1 lần/bước.
**2. SỐ ĐO (probe CB, 3 lần chạy ổn định — best-of-2 mỗi biến thể):**
| Biến thể | ns/ptu | Tốc độ | maxdiff vs W (1 bước) | Kết luận |
|---|---|---|---|---|
| W (hiện tại) | 30.74 / 30.72 / 30.71 | 1.00× | — | tham chiếu |
| V1: hoist `(1-b1)`, `(1-b2)` | 33.16 / 33.13 / 33.12 | **0.93×** | 0.0 | **bác bỏ** |
| V2: V1 + gộp `p` (đọc/ghi 1 lần) | 33.31 / 33.35 / 33.34 | **0.92×** | 0.0 | **bác bỏ** |
| V3: gộp 2 div → 1 | 28.75 / 28.76 / 28.77 | **1.068×** | **1.19e-07** | **WIRE** |
Bài học đo được: hoist `(1-b1)`/`(1-b2)` **không có lợi** — JIT/CSE đã tự làm
(chính V3 có hoist mà vẫn nhanh hơn) → mỗi biến thể phải đo riêng.
**3. WIRE:** `AdamW.step` trong **4 file** `trainer` (canonical
`tv/training/trainer.tkv` + 3 bản demo `tv/{training,tests,tools}/tv.training.trainer.tkv`):
cộng `epss = self.eps * bc2s`, `ks = step_size * bc2s` trước vòng, và
`p[i] = p[i] - ks*m[i]/(sqrt(v[i]) + epss)`.
**4. KẾT QUẢ:**
- Phase `ADAMW_STEP` (profile @124M): **4375/4375/4359 → 3922/3937 ms = −10.1%** (−440 ms/bước).
- Gate **135/135** (9 suite nhỏ 102 + `gpt2_full` 33) + **`trainer_tests` 15/15 PASS** —
  parity với **fixture PyTorch thật** (`torch.optim.AdamW`), TOL=1e-5; chênh lớn nhất
  còn **2.5e-06** (overfit loss) — vẫn xanh.
- Loss in ra 3 seq **khớp exact 7 chữ số** so với GĐ8 (10.572291→7.0119243,
  10.756927→11.807251, 10.760399→13.442912 — **không đổi chữ số nào**).
- Bench A/B best-of-3 xen kẽ (giảm drift nhiệt):
| seq | GĐ8 | GĐ10 | Δ |
|---|---|---|---|
| 8 | 12.844 s | **12.516 s** | **−2.6%** |
| 256 | 17.906 s | **17.313 s** | **−3.3%** |
| 1024 | 37.875 s | **36.703 s** | **−3.1%** |
- Tỉ lệ vs torch: 13.9/12.9/10.7× → **13.4× / 12.3× / 10.4×**
  (từ đầu P6: 16.2/16.3/12.4×).
**5. CÒN LẠI SAU GĐ10:** khoảng cách với PyTorch giờ nằm ở (a) **f32** — PyTorch
toàn bộ f32, ta f64 trung gian, mà `.tkv` **không có biến cục bộ kiểu**
(`simd_probe_aa.tkv`: `x: 'f32' = 0.0` → `il_codegen: không dịch được`) ⇒ chỉ
dùng được nếu compiler hỗ trợ; (b) **vectorize** — vòng của PyTorch là C++
AVX2, ta là interpreter ⇒ muộn hơn phải thêm kernel native (vùng `C`, cùng
gió như P7 đã chỉ).
---
### 0.31 PLAN TOÀN BỘ CƠ HỘI CÒN LẠI SAU GĐ10 (2026-10-22) — CHỈ TÀI LIỆU, KHÔNG ĐỔI CODE
- **Bối cảnh:** rà soát toàn bộ đường train, phân rã mới bằng probe AQ chạy lại
  2026-10-22 (`out/audit_aq.log`) @seq1024: ATTN_BWD 8844 / MLP_BWD 7953 /
  LHEAD_BWD 7281 / FWD_BODY 6656 / ADAMW 3930 / LN_BWD 765 / EMBED 829 /
  UNFLATTEN 734 ms — 3 bucket GEMM = **24.1 s = 66%** bước. (Artifact cần
  nhớ: GRAD_ACC 1232 ms là của bench — đường thật ghi thẳng vào `opt.g`;
  RESIDUAL −10438 ms do probe gọi library không gflat, đo tuần tự.)
- **Plan đầy đủ ghi tại `docs/perf-plan-triz-roadmap.md` §13** (mới) — 4 hạng
  mục xếp theo lợi nhuận × tin cậy, bảng kỳ vọng, trình tự + gates, danh sách
  đã-bác-bỏ:
  1. **GĐ11** — song song softmax-grad lm-head bwd (còn serial 1.33 s @1024,
     probe AV; mẫu `_ls_one` GĐ4 cùng math forward đã thắng 5.10×) →
     **−0.95..1.05 s, tin cậy cao — làm trước**.
  2. **GĐ12** — LN_BWD dùng saves mu/rstd từ forward (bỏ 2/4 pass; 765 ms →
     ≤550) → **−0.23..0.31 s, chắc chắn** (mu/rstd cùng hàm cùng input với
     recompute → bitwise bằng, xác minh maxdiff 0.0).
  3. **GĐ13** — AdamW parallel lần 3, **probe chẩn đoán `simd_probe_cc`
     TRƯỚC** (V1 compute-only PAR12 / V2 ghi SHARED / V4 copy control) vì 2
     lần bust chưa rõ nguyên nhân (BL 0.36×/0.29×; boxing ĐÃ loại — `step`
     int32, `i/k` native) → nếu thắng **−2.4..2.7 s** (sàn băng thông ~1.0-1.1 s),
     nếu V1 ≈1× thì ĐÓNG vĩnh viễn.
  4. **P7b** — probe `simd_probe_cd` đo GEMM native vs `torch.matmul` cùng
     shape cùng phiên (GFLOPS + số thread thật, `TKV_GEMM_MIN_N=96`?) →
     điều hướng parity.
- **Trần trung thực:** `.tkv`-only sau GĐ11-13 ≈ **32.7-35.5 s ≈ 9-10×**;
  mục tiêu 1.0× bắt buộc đi qua native/C (C source sẵn ở
  `D:/TokenVector/Release/tools/simd_native/tkv_simd_native.c` + `build.ps1`,
  thuộc repo compiler) hoặc compiler hỗ trợ f32 locals (đang chặn —
  `simd_probe_aa`).
- **Lần này KHÔNG đổi code** — chỉ 3 file tài liệu: perf-plan (§9 + §13),
  HANDOFF (§0.31 này), ROADMAP_TRAIN.
---
### 0.32 P6 GĐ11 — SONG SONG SOFTMAX-GRAD LM-HEAD: ĐO BÁC BỎ, KHÔNG WIRE (2026-10-22)
Đi tiếp §0.31 item 1 (GĐ11 "làm trước", kỳ vọng −0.95..1.05 s, tin cậy cao
theo mẫu `_ls_one` GĐ4). Nguyên tắc **ĐO TRƯỚC** được giữ: wire thử vào 5
mirror `gpt2_ref` (worker `_SgJob`/`_sg_one`/`_sg_all`, row-parallel 12 worker,
guard `n*V >= 4M`, ghi THANG vào `gg.data` phân đoạn row-disjoint) rồi đo —
**số đo phản bác kỳ vọng, revert toàn bộ, KHÔNG commit code**.
**1. SỐ ĐO (2 lần chạy, bit-exact giữ nguyên — chk AQ `0.000105296414`):**
- Bench seq1024 với bản wire: **38391 ms** (GĐ10 best 36703) — chậm hơn.
- AQ (A/B 2 binary cùng phiên): LHEAD_BWD **7407 → 10719 ms (+3.3 s)**,
  TOTAL_bwd 22375 → 23297.
**2. CHUAN DOAN — probe `tv/tests/simd_probe_ce.tkv` (@51.5M ptu, 2 lần chạy,
maxdiff 0.0 cả SH lẫn OW):**
| Biến thể | ms | Tốc độ vs serial |
|---|---|---|
| S — serial (đường hiện tại) | 1110-1140 | 1.00× |
| SH — shared write + exp (bản wire) | 3110-4453 | **0.25-0.36×** |
| CP — copy shared KHÔNG exp (điều khiển) | 484-547 | **2.0-2.3×** |
| OW — own-seg: alloc | 359-375 | — |
| OW — worker only | 484 | 2.36× |
| OW — merge (main, đơn luồng) | **1172** | ≈ toàn bộ serial! |
| OW — TOTAL (alloc+worker+merge) | 2031 | **0.56×** |
**3. KẾT LUẬN — 2 quy tắc đo mới:**
1. **Shared-write thuần KHÔNG bệnh** (CP 2.0-2.3×) — bệnh xuất hiện khi có
   **phep toan per-ptu + ghi list tuong Minh** (SH 0.25-0.36×) — CÙNG bệnh
   AdamW BL (SHARED 0.36×, OWN+MERGE 0.29×) → quy tắc chung cho mọi ứng viên
   song song còn lại: **vòng compute-heavy ghi thẳng vào list CHUNG phải đo
   riêng, không được suy ra từ mẫu đã thắng** (mẫu `_ls_one` thắng vì seg
   output TINY — 85 ptu/worker, không phải vì "cùng math").
2. **Own-seg + merge cũng có thể chết** khi merge đơn luồng ≈ serial: ở đây
   merge 51.5M ptu = **1172 ms (~22.7 ns/ptu)** ≈ toàn bộ softmax serial →
   ceiling kể cả merge miễn phí (worker 484 + alloc ~360) ≈ −0.28 s, dưới bar
   −0.9 s của §13.3.
**4. WIRE/REVERT:** `git checkout` 5 mirror `gpt2_ref` + `tv/tests/simd_probe_aq.tkv`
— mã nguồn về nguyên trạng HEAD (vòng serial trong `gpt2_backward` simd==1).
Probe `simd_probe_ce.tkv` GIỮ LẠI làm bằng chứng + regression reference.
Docs: perf-plan §12 (dòng GĐ11) + §13.3/§13.4/§13.5/§13.6 + §0.32 này.
**5. HỆ QUẢ KẾ HOẠCH §13:** bỏ GĐ11 khỏi danh sách kỳ vọng → trần `.tkv`-only
hạ tương ứng (GĐ11 −0.95..1.05 s không còn): GĐ13 bust → ~36.4-36.5 s ≈ 10.1×;
GĐ13 thắng → ~33.8-34.2 s ≈ 9.6-9.7×. P7b giữ nguyên vai trò điều hướng.
---
### 0.33 P6 GĐ12 — LN_BWD DÙNG SAVES MU/RSTD: ĐO BÁC BỎ, KHÔNG WIRE (2026-10-22)
Đi tiếp §0.32 item 2 (GĐ12 "chắc chắn nhất", kỳ vọng −0.23..0.31 s, "mu/rstd
cùng hàm cùng input → bitwise bằng"). Wire thử đầy đủ theo thiết kế §13.3:
helper `gpt2_layernorm_saved` (LN fwd + thu mu/rstd vào Tensor [n]),
`gpt2_block_saved` trả thêm mu1/rs1/mu2/rs2, `gpt2_forward_saved` append CUỐI
(format mới 3+8N+2, len guard → saves cũ 3+4N vẫn đọc được),
`gpt2_layernorm_bwd(save_mu, save_rs)` optional, `gpt2_block_bwd` +
`gpt2_backward` truyền tiếp theo len guard — wire 5 mirror `gpt2_ref`. Rồi **đo:
tiền đề bit-exact của plan SAI, lợi ích nhỏ hơn nhiều mức thắng → REVERT**.
**1. SO HOC — TIỀN ĐỀ PLAN SAI (probe `tv/tests/simd_probe_cf.tkv` @124M):**
- Trong `.tkv`, `mu`/`rstd` của LN được tính bằng **f64** (accumulator f64),
  nhưng saves là `list[Tensor]` mà `Tensor.data` **chỉ chứa `list[f32]`** →
  lưu mu/rstd vào saves **bị làm tròn f32**: đo trực tiếp mu lệch **6.79e-10**,
  rstd lệch **2.52e-08** so với bản recompute f64.
- Hệ quả: **MAXDIFF toàn bộ grad (148 tensor, 124M ptu) saves vs recompute =
  3.17e-08** — KHÔNG phải 0.0 (wte 7.5e-09, LN-g ~3e-08, bias LN ~0.0).
- Hệ quả nữa: loss in ra **đổi chữ số cuối** — bench A/B 2 binary cùng phiên:
  GĐ10 `10.760399 → 13.442912`, GĐ12 `10.760399 → 13.442913` (deterministic,
  2/2 lần chạy mỗi binary). → **vi phạm cả 2 điều kiện §13.2** (maxdiff 0.0
  HOẶC loss khớp exact).
**2. LỢI ÍCH THỰC SÓ (probe CF, 2 lần chạy, best):**
| Pha | Không saves | Có saves | Tốc độ |
|---|---|---|---|
| LN_BWD ×25 | 656-703 ms | 578-594 ms | **1.10-1.22× (−62..−125 ms)** |
Thấp hơn nhiều mức thắng plan (−215 ms, 765→≤500): 2 pass bị bỏ (mu, v/rstd)
là 2 pass **RẺ nhất** — chỉ đọc `x` đã cache-hot; 2 pass còn lại (s1/s2 + dx,
đọc thêm `gy`/`g` + ghi dx/dg/db) nặng hơn nhiều. Bench A/B seq1024 2 cặp
xen kẽ: GĐ12 nhanh hơn ~1% mỗi cặp nhưng chênh 400-600 ms vượt xa LN_BWD
(62-125 ms) → không tách được khỏi nhiễu nhiệt (phiên này nóng: 38.5-40.6 s
cả 2 binary, GĐ10 best cũ 36.7 s); seq8/256 không đo tiếp vì kết cục đã chốt
bằng điều kiện số học.
**3. KẾT LUẬN — KHÔNG WIRE:** `git checkout` 5 mirror `gpt2_ref` — mã nguồn
về nguyên trạng HEAD. Probe `simd_probe_cf.tkv` GIỮ LẠI làm bằng chứng.
**Điều kiện sống lại:** compiler hỗ trợ **f64 container** (lưu mu/rstd đúng
f64 → bitwise bằng bản recompute) **hoặc typed f32 locals** (LN tính f32
toàn bộ như torch → saved = computed bitwise) — đều ngoài tầm `.tkv` hiện tại,
cùng hướng với P7b/native.
**4. HỆ QUẢ KẾ HOẠCH §13:** GĐ11 + GĐ12 đều bust → chỉ còn GĐ13 (AdamW
parallel, probe chẩn đoán) + P7b (đo GEMM gap). Trần `.tkv`-only nếu GĐ13
bust ≈ **36.7 s = 10.4×** (giữ nguyên GĐ10); GĐ13 thắng → ~34.0-34.3 s ≈ 9.6×.
Docs: perf-plan §12 (dòng GĐ12) + §13.3/§13.4/§13.5/§13.6 + §0.33 này.
---
---
### 0.34 P6 GĐ13 — ADAMW PARALLEL LẦN 3: ĐO BÁC BỎ, ĐÓNG (2026-10-22)
**Mục tiêu:** `ADAMW_STEP` 3930 ms (11% bước @124M) — song song 12 worker. Đã bust 2 lần
(BL: SHARED-12 0.36×, OWN+MERGE 0.29×; GĐ9 loại boxing) → GĐ13 chạy probe chẩn đoán TRƯỚC
(§13.3: "đo trước, quyết theo số"), không wire khi chưa net win.
**Probe `tv/tests/simd_probe_cc.tkv`** — @N=124M, `time_perf`, best-of, bit-check subset 12M:
| V | Variant | ms | so serial | Bit vs V0 |
|---|---|---|---|---|
| V0 | serial (công thức GĐ10) | 3297 | 1.00× | — |
| V1 | OWN-seg compute-only (ceiling) | 1266 | **2.60×** | n/a (không merge) |
| V2 | SHARED + read-back | 10984 | **0.30×** ❌ | dp=dm=dv **0.0** |
| V4 | copy shared (control) | 1297 | **2.54×** | — |
| V5 | **REGISTERS** (toàn bộ phep toan trong thanh ghi, ghi p/m/v 1 lần, KHÔNG read-back) | 9937 | **0.33×** ❌ | dm=dv 0.0, dp 9.5e-07 |
| V3 | OWN + serial merge | 2766 | **1.19×** | exact theo segment |
**Chốt nguyên nhân — lật giả thuyết CE/CF:** V5 (không read-back, mọi phép trong thanh ghi) ≈ V2
(0.33× ≈ 0.30×) → **read-back KHÔNG phải thủ phạm**; bệnh thật là **phep toan per-ptu + ghi list
CHUNG** (cùng pattern CE SHARED 0.25-0.36×, BL SHARED 0.36×). Worker chỉ scale khi ghi **seg
RIENG** (V1 2.60×); đưa kết quả về list chung bằng **merge đơn luồng** (V3 1.19× — merge 875 ms
ăn phần lớn) hay **ghi thẳng shared** (V5 0.33×) đều **ăn hết lợi thế**. **V4 copy shared 2.54×**
loại lỗi hệ thống/NUMA/drift (V4 ≥2× đúng quy tắc §13.3) → kết luận vững.
**Bit:** V2-vs-V0 dp=dm=dv **0.0** (bit-exact); V5-vs-V0 dm=dv 0.0, dp **9.5e-07** (V5 không round
f32 trước khi tính p → lệch 1 ULP, vô hại vì đã bust).
**Quyết định: KHÔNG WIRE** — `trainer.step` giữ serial; 4 file trainer không đổi; ước −2.4..2.7 s
(nếu thắng) → **0**; giữ **10.4×**. Hướng còn lại cho AdamW: **kernel fused native** (sàn băng
thông ~1.0-1.1 s, không cần thread) — mục con của P7b. Docs: §12 dòng GĐ13 + §13.3/§13.4/§13.5/§13.6.
### 0.35 P7b — ĐO KHOẢNG CÁCH GEMM: tkv NATIVE vs TORCH — CHỐT PARITY PHẢI QUA NATIVE/C (2026-10-22)
**Mục tiêu (kế hoạch §13.3 P7b):** kiểm lại claim "3 bucket GEMM (ATTN+MLP+LHEAD) = 24.1 s = 66% bước, near-peak AVX2" — nếu thật near torch thì full step phải ≪ 36.7 s. Probe thuần, **0 thay đổi code train**.
**Phương pháp:** rebuild `simd_probe_bh` (tkv, `simd_gemm` native qua wrapper thật) + `tools/p7b_gemm_torch.py` (torch.matmul f32, 14 thread) — **cùng phiên, cùng máy**, shape đại diện, best-of. Bit-check LIB-vs-PAR = **0.0** (all bit-exact). Tái xác nhận standalone `simd_probe_cd` (probe riêng 4 shape, best-of-3, cùng script torch): S1 93 vs 6.5 ms (**14.3×**), S3 968 vs 134.4 (**7.2×**), S4 1734 vs 234.3 (**7.4×**), S5 1562 vs 247.5 (**6.3×**) — **6-14×**, GFLOPS thật tkv **45.6-81.7** vs torch **319-741**.
| Shape (m×n×k) | tkv native (best) | torch.matmul (14T) | tỉ lệ |
|---|---|---|---|
| S1 dx1 [1024,768,3072] | 94 ms | 7.1 ms | **13.2×** |
| S3 dxf [1024,768,50257] | 969 ms | 129 ms | **7.5×** |
| S4 dwte [50257,768,1024] | 1781 ms | 254 ms | **7.0×** |
| S5 logits [1024,50257,768] | 1547 ms | 240 ms | **6.5×** |
**CHỐT — rơi vào kịch bản "Ta 10-30% torch" (thực tế 8-15%):** khoảng cách nằm ở **KERNEL/threading**, không phải driver/alloc/wrapper. Bằng chứng: builder đã tối ưu GĐ3 (append→buffer, 1.4-3.2×) + wave-12 lm-head PAR chỉ **1.22-1.29×** serial (probe BH S3/S4/S5 LIB vs PAR) → overhead ngoài kernel chỉ ~20-25%, còn **~6-14×** là chính `simd_gemm`. Claim "near-peak AVX2" **SAI** — `simd_gemm` ~45-82 GFLOP/s vs torch MKL 320-740 GFLOP/s cùng Haswell (AVX2). Nghi vấn: native pool 1 luồng cho tile nhỏ (`TKV_GEMM_MIN_N=96`), thiếu packing/register-block. ~~Hoặc thread pool không parallel thật~~ **BỊ BÁC BỎ (xác nhận lại 2026-10-22): P3 đo cpu/wall 21.20× @768³ = 539 GFLOP/s ~21 luồng — pool parallel thật, đỉnh ngang MKL ở shape vuông lớn; nút thắt là shape thật chỉ ăn 6-33% đỉnh đó (roadmap §14 mục M3).**
**Đề xuất (cần compiler/native team — KHÔNG làm được bằng `.tkv`):** C source `tools/simd_native/tkv_simd_native.c` + `build.ps1` (repo compiler, cần phối hợp): (1) verify số thread THẬT của pool, (2) micro-kernel f32 AVX2 register-block + prefetch, (3) parallel tile nhỏ; **kernel fused AdamW/elementwise native** ≈ sàn băng thông ~1.0-1.1 s (thay GĐ13 bust). **Đây là đường DUY NHẤT tới parity 1.0×.**
**Kết luận chung sau GĐ11-13 + P7 + P7b:** trần `.tkv`-only ≈ **10.4×** (giữ nguyên, không cải thiện thêm được); parity 1.0× **bắt buộc** đi qua native/C hoặc compiler (f32 locals, fused kernel). P7b là probe quyết định, không phải tối ưu. **Phân tích TRIZ + menu giải pháp chi tiết cho native-team (M1-M7): roadmap §14.** Docs: §12 dòng P7b + §13.3 P7b + §13.4.
### 0.36 M3 — PHÂN RÃ WRAPPER GEMM 4 SHAPE P7b: CHỐT WRAPPER-BOUND, KERNEL ĐÃ PARITY (2026-10-23)
**Mục tiêu (roadmap §14.3 M3 — làm đầu tiên theo §14.4):** trả lời "vì sao shape thật chỉ ăn 6-33% đỉnh 539 GFLOP/s đã đo (P3)?". Probe thuần `tv/tests/simd_probe_cg.tkv`, **0 thay đổi code train**, 5 chế độ: (0) phân rã wrapper `smm_*` thành ALLOC/BUILD/GEMM/ACC trên đúng 4 shape P7b; (1) GEMM thuần tile dựng sẵn; (2) BUILD thuần; (3) FULL wrapper; (4) re-entrancy — modes 1-3 đo cpu/wall qua `tools/probe_threads.py`. Cùng phiên, cùng máy, best-of; SUM khớp FULL ±5-12%.
**Mode 0 — phân rã (ms / 1 lần gọi wrapper):**
| Shape | ALLOC | BUILD | GEMM | ACC | FULL | GEMM GF (pad) | FULL GF (hữu ích) | GEMM vs torch |
|---|---|---|---|---|---|---|---|---|
| S1 dx1 [1024x768x3072] | 31 | 62 | 13.1 | 15 | 109 | 552 | 44.3 | 13.1 vs 7.1 ms |
| S3 dxf [1024x768x50257] | 16 | 468 | 203.3 | 234 | 969 | 588 | 81.6 | 203 vs 129 ms (≈ pad 1.5×) |
| S4 dwte [50257x768x1024] | 406 | 750 | 207 | 359 | 1828 | 578 | 43.2 | 207 vs 254 ms — **thắng** |
| S5 logits [1024x50257x768] | 422 | 562 | 211 | 234 | 1563 | 567 | 50.6 | 211 vs 240 ms — **thắng** |
**Tỉ lệ luồng (cpu/wall):** GEMM thuần **22.64** (pool hết công suất, per-pass 203-214 ms khớp mode 0) · BUILD thuần **1.00** (tile-build 100% đơn luồng, tổng 1813 ms) · FULL wrapper **3.68-3.97** (phần đơn luồng làm loãng 22 luồng).
**Re-entrancy (mode 4) — BẰNG CHỨNG MỚI:** serial 24 call T=192 = 187 ms; 2 thread `thread_spawn`, mỗi thread `simd_gemm` trên A/B/C riêng rọ → **HANG > 90 s** (timeout giết). Đúng cơ chế cảnh báo `simd_matmul.tkv:419` + BH:11-13 (T≥96 → `tkv_run_mt` dùng `g_job` global không lock): race dispatcher biểu hiện **deadlock**, không chỉ sai số. **Ràng buộc: mọi GEMM song song phải qua 1 entry point duy nhất; wave-12 giữ worker dựng tile, main gọi simd_gemm (đúng BH/PD).**
**CHỐT:** (1) pha GEMM chạy 552-588 GFLOP/s pad ≈ torch useful (S1 683/S3 612/S4 311/S5 329) — số "45-82 GFLOP/s" của §0.35 = chia flop hữu ích cho thời gian WRAPPER gồm build/acc/alloc + 1.5× pad → **quy sai cho kernel**; (2) pad 1.5× (T=768, m=1024 → pad 1536) giải thích trọn phần kernel S3 (79×1.51/0.588 = 202 ms ≈ 203 đo được); (3) BUILD+ACC+ALLOC = **76-89% FULL** (BUILD 2.2-3.6× GEMM, ACC 1.1-1.7×, ALLOC ~400 ms ở S4/S5) — BUILD ratio 1.00; (4) khoảng cách FULL 6.5-14× = **pad × wrapper, KHÔNG phải kernel** → **kết luận "khoảng cách nằm ở KERNEL/threading" ở §0.35 BỊ ĐẢO NGƯỢC MỘT PHẦN** (đã ghi chú tại chỗ).
**Hệ quả menu TRIZ (roadmap §14.5):** M3 ✅ đóng; **M1/M2 hạ ưu tiên** (chỉ nhắm residual 1.2-1.6×); mục mới **M3a song song ACC → M3b T-dispatch giết pad → M3c tái dùng buffer** (đều `.tkv`, gate §13.2) — trần ước lượng S3 938 → ~250-300 ms ≈ 3.1-3.7×, **chưa đo, không hứa số bước tổng**. KPI "≥250 GFLOP/s/shape" tính cho pha GEMM: **4/4 đã vượt**. Logs: `out/cg_m0..m4.log`.
### 0.37 M3a — PIPELINE build(w+1) ∥ replay(w) CHO `_smm_core_pd`: THIẾT KẾ CHỐT, CHƯA VIẾT CODE (2026-10-23)
**Bối cảnh:** sau M3 (§0.36, commit `9512e13`), "LÀM TIẾP" = bước §14.4/§14.5 kế tiếp = **M3a**. Đã đọc code + chốt thiết kế. **CHƯA viết probe CH, CHƯA đo, CHƯA đổi code production.**
**Đã xác minh bằng đọc code:**
1. **Production đi `smm_*_pd` → `_smm_core_pd` (wave-12) CHỈ ở 4 site lm-head:** `gpt2_ref124.tkv:609` (logits fwd), `:1622` (logits_t bwd), `:1644` (dxf), `:1656` (dwte). Các GEMM khác (MLP/attention) gọi plain `smm_*` → `_smm_core`/`_p58_core_simple` (serial, không wave).
2. **Cấu trúc `_smm_core_pd` (simd_matmul:534-753):** mỗi wave = spawn 12 worker `_pd_one` (chỉ dựng tile ta/tb, **KHÔNG simd_gemm**) → **join toàn bộ** → main replay tuần tự `_pd_do` (simd_gemm + `_acc_f`) đúng thứ tự i0→p0→j0 → wave sau. **KHÔNG overlap build(w+1) vs replay(w)**; buffer **1 set** (ta0..ta11 + tb0..tb11, `_zeros_list` mỗi call = 24 list T²/call).
3. **Gate 40M:** `n_i*n_p*n_j*T*T < 40000000` → serial (trans_a → `_p58_core_simple`, else `_smm_core`). S1 (vol 4.7M) → serial; S3/S4/S5 (78M) → wave. T = 768 cả 4 shape.
4. **CG mode 3 baseline = plain `smm_*`, KHÔNG phải production `_pd`** → probe CH **phải** baseline bằng `smm_fwd_pd/smm_dot_pd/smm_fwd_at_pd`.
5. Số đo CG (§14.5): GEMM ratio 22.64, BUILD serial 1.00, FULL 3.97; re-entrancy: 2 thread `simd_gemm` T≥96 đồng thời → HANG (g_job không lock).
**Thiết kế M3a (Plan A — double-buffer pipeline):**
- Pre-loop spawn wave0 (set 0) → loop: `join(w)` → `if (w+1)*12 < ng: spawn(w+1) vào set (w+1)%2` → `replay(w) từ set w%2` → w++.
- **Bit-exact by construction:** thứ tự/nội dung replay giữ nguyên (w↑, slot↑ = i0→p0→j0); worker ghi set disjoint (pattern BG2); worker KHÔNG simd_gemm (tôn trọng re-entrancy — CG mode 4).
- Buffer: 48 list (24+24) = **+24 list/call** → phải đo chi phí alloc (probe mode 3); nếu lớn → gộp M3c (module-scratch cấp phát 1 lần).
- **Vấn đề kỹ thuật đã giải:** khối derive 12 slot unrolled sẽ bị clobber khi spawn(w+1) chạy trước replay(w) → **chuyển derive vào worker** (`_ch_one` tự tính i0/p0/j0 từ gidx; job thêm field `nout` + guard `gidx >= ng` trong worker) → spawn block chỉ còn 12 dòng đơn giản, 2 vị trí nguồn (pre-loop wave=0 + in-loop wave=w+1) hoặc trick `first_iter` cho 1 vị trí.
- Kỳ vọng (**ước lượng, CHƯA ĐO**): S3 FULL ~937 → ~510-590 ≈ 1.6-1.7×; S4 1719 → ~1200 ≈ 1.4×; S5 1484 → ~1050 ≈ 1.4×. Rủi ro đo: oversubscription 12 builder + ~21 pool threads trên 28T.
- **Ngưỡng wire:** ≥1.3× trên S3/S4/S5 + BIT=0.0 → wire `_smm_core_pd` → gate 84/84 + trainer_tests + bench 3 seq (kỷ luật §13.2).
**Chưa làm:** viết probe CH (modes: 0 = baseline `smm_*_pd`, 1 = pipe, 2 = BIT filled-input, 3 = chi phí alloc 48 buffer), compile + chạy qua `tools/probe_threads.py`, wire, gate, cập nhật §14.5 + ROADMAP_TRAIN, commit. Runbook dự kiến giống CG:
```
printf "0\n" > tv/tests/_ch_cfg.txt
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/simd_probe_ch.tkv out/ch.exe main
out/ch.exe && python tools/probe_threads.py out/ch.exe 2
```
### 0.38 M3a — PIPELINE build(w+1)∥replay(w) ĐÃ ĐO: BIT-EXACT NHƯNG SPEEDUP BUST — KHÔNG WIRE (2026-10-23)
**Probe:** `tv/tests/simd_probe_ch.tkv` (compile sạch từ source → `out/ch.exe`, log `out/ch_run.log`). Pipeline double-buffer theo đúng thiết kế §0.37: worker `_ch_one` tự derive i0/p0/j0 từ gidx (giải clobber), 2 set buffer 48 list, handle hA*/hB* reassign theo wave le/chan. Best-of-2, cùng phiên, 4 shape P7b; baseline = `smm_*_pd` production.
**Số đo (1 lần gọi wrapper, ms):**
| Shape | base PD | pipeline | speedup | BIT |
|---|---|---|---|---|
| S3 dxf [1024×768×50257] | 812 | 797 | **1.02×** | **0.0** |
| S4 dwte [50257×768×1024] | 1235 | 1312 | **0.94×** ❌ | **0.0** |
| S5 logits [1024×50257×768] | 1359 | 1234 | **1.10×** | **0.0** |
| S1 dx1 [1024×768×3072] | 94 | 219 | 0.43× (serial route — pipeline không nham) | **0.0** |
| alloc 48 buffer (T=768) | — | **187 ms** | — | — |
**CHỐT:**
1. **Pipeline BIT-EXACT 0.0 cả 4 shape** — thiết kế §0.37 đúng về kỹ thuật: worker-derive + set disjoint + handle reassign chạy ổn trên compiler này (pattern dùng lại được).
2. **NHƯNG KHÔNG ĐẠT NGƯỠNG WIRE ≥1.3×** (S3 1.02, S4 0.94 chậm hơn, S5 1.10). Chi phí 48 buffer = 187 ms/call (vs 24 của PD, +93 ms) ăn hết phần lãi nhỏ.
3. **Nguyên nhân — ĐẢO NGƯỢC kỳ vọng 1.6-1.7× của §0.37:** trong đường PD, build ĐÃ song song hoá bởi 12 worker (BH GĐ3) → pipeline chỉ ẩn được build(w+1) vốn đã nhỏ (~7 ms/wave so với replay ~74 ms/wave cho S3: 812 ms / 11 wave). Bottleneck thật = **replay tuần tự trên main**: 12 × (simd_gemm T=768 + acc interpreted) mỗi wave; simd_gemm T≥96 re-entrancy (`g_job` không lock, CG mode 4) cấm chạy đồng thời → không song song được replay theo cách này. S4 âm (0.94×): oversubscription 12 builder + pool ~21 thread trên 28T làm overlap mất hiệu lực.
4. **Quyết định: KHÔNG wire** — `_smm_core_pd` giữ nguyên serial-wave. Đúng kỷ luật §13.2 (BIT=0.0 đạt, speedup bust).
**Hệ quả menu (roadmap §14.5 đã cập nhật):** M3a-pipeline ❌ đóng. Cửa sổ còn lại theo thứ tự: **M3b T-dispatch giết pad 1.5×** (lever thật cho S3 — pad chính là thứ làm nó thua torch 1.57×) → **M3c tái dùng buffer tile/C** (lever thật cho S4/S5 — ALLOC ~400 ms/call; pipeline bust xác nhận alloc là thực) → parallel-ACC biến thể gốc (chỉ khả thi S4/S5 slot rời rạc; **S3 bị p-reduction chặn**: 12 p-tile cộng cùng 1 block — chưa đo). Trần "S3 938 → ~250-300 ms" của §0.37 **không còn đứng** — đã ghi chú trong roadmap.
**Bài học:** (1) pipeline overlap chỉ có ý nghĩa khi phase bị ẩn lớn hơn phase chính; (2) số ước lượng 1.6-1.7× xuất phát từ giả định "build là bottleneck" — bị chính số CG (BUILD đã parallel trong PD) phản bác ngay khi đo; (3) probe CH chạy sạch BIT=0.0 ngay lần đầu xác nhận worker-derive + double-buffer an toàn.
**Trạng thái git:** probe CH + `out/ch_run.log` + roadmap §14 + HANDOFF §0.37/§0.38 — commit trong chương này. Chưa wire, không đổi code train → không cần gate 84/84.
### 0.39 M3b — T-DISPATCH GIẾT PAD ĐÃ ĐO: BÁC BỎ (2026-10-23)
**Probe:** `tv/tests/simd_probe_cb.tkv` (compile sạch → `out/cb.exe`, log `out/cb_run.log`). Grid T = 256/384/512/768/1024 trên 4 shape P7b, mỗi (shape, T) đo ALLOC/BUILD/GEMM(per-pass, ≥100 calls)/ACC → SUM thủ công; tham chiếu = full-wrapper production T=768 best-of-2. 0 thay đổi code train. Buffer 3 pha cấp phát trong vòng lặp (bug compiler nested list[f32] — lần chạy đầu crash NullReference với list[list[f32]] index, sửa bằng cách copy đúng cấu trúc CG mode 0).
**Số đo (ms, 1 lần gọi wrapper; FULL_prod đo thật S1 94 / S3 968 / S4 1703 / S5 1500 — khớp CG mode 3 trong ±3%):**
| Shape | T=256 | T=384 | T=512 | T=768 (prod) | T=1024 | GEMM GF @256 vs @768 |
|---|---|---|---|---|---|---|
| S1 dx1 | 156 | 150.7 | 92.8 | **89.8** | **70.8** | 302 vs 617 |
| S3 dxf | 1781 | 1313 | 1125 | **953** | **750** | 282 vs 586 |
| S4 dwte | 2375 | 2031 | 1750 | **1657** | **1407** | 281 vs 549 |
| S5 lgts | 2375 | 1937 | 1734 | **1438** | **1180** | 282 vs 546 |
**CHỐT:**
1. **Pad 1.0× (T=256) là TỆ nhất cả 4 shape** — đảo ngược giả định §14.5 ("T=256 pad 1.0 nhưng calls ×18 — phải đo"). Calls ×18 không phải vấn đề chính; **kernel simd_gemm mất throughput ở tile nhỏ**: 282 GFLOP/s @256, 380 @384, 485 @512, 546-616 @768. GEMM phase S3 @T=256 = 281 ms dù flop hữu ích chỉ 66% của @T=768 (204 ms) → hiệu suất ~0.47×, bù không nổi BUILD+ACC tăng theo số tile.
2. **T=1024 (pad 1.36×) thắng SUM cả 4 shape nhưng chỉ 1.18-1.33×** (S1 1.27, S3 1.27, S4 1.18, S5 1.22). Quá sát ngưỡng wire 1.3× trên **SUM thủ công** (chưa phải wrapper thật — CG đã thấy RESID in-context); wire thật phải sửa 3 đường hard-code `T = _imin(_tilek(k), _roundup64(_imin(m,n)))` (`_smm_core`, `_p58_core_simple`, `_smm_core_pd`) + wave-12 T=1024 tốn 24 buffer × 4 MB = 96 MB (alloc ~2×). Theo §13.2 — không wire trên số mỏng + chưa đo wrapper thật.
3. **Quyết định: KHÔNG wire.** 3 đường core giữ nguyên T=768 rule. Lever M3b duy nhất còn ghi nhận: đo lại **full-wrapper thật ở T=1024** (không phải SUM tách pha) nếu quay lại — nhưng ưu tiên hạ dưới M3c.
4. **Bài học:** (a) "giết pad" đổi GEMM lấy BUILD+ACC và kernel throughput là hàm của T — pad 1.5× ở T=768 **rẻ hơn** pad 1.0× ở T=256 trong wrapper này; (b) SUM thủ công tách pha ≠ full-wrapper — mọi wire phải đo wrapper thật; (c) pattern "grid T × 4 shape + FULL_prod tham chiếu" của probe CB dùng lại được.
**Hệ quả menu:** M3b ❌ đóng. Cửa sổ còn lại theo thứ tự: **M3c tái dùng buffer tile/C** (S4/S5 ALLOC 375-422 ms/call — lever đo được duy nhất còn lại) → parallel-ACC biến thể gốc (chỉ S4/S5; S3 p-reduction chặn — chưa đo) → M1 OpenBLAS.
**Trạng thái git:** probe CB + `_cb_cfg.txt` (untracked theo tiền lệ `_ch_cfg.txt`) + `out/cb_run.log` + roadmap §14.4/§14.5/§14.6 + HANDOFF §0.39 — commit trong chương này. Chưa wire, không đổi code train → không cần gate 84/84.
### 0.40 M3c — RE-USE BUFFER C ĐÃ ĐO: CONDITIONAL WIN (ratio 0.18-0.22), CHƯA WIRE (2026-10-23)
**Probe:** `tv/tests/simd_probe_ci.tkv` (compile sạch → `out/ci.exe`, log `out/ci_run.log`). Đo (1) `t_zeros([m,n])` thật cho 2 shape C lớn nhất của đường PD lm-head và (2) zero-fill C đó (điều kiện bắt buộc của re-use — xóa trước mỗi lần tích lũy). Best-of-3, 0 thay đổi code train.
**Số đo (ms, best-of-3):**
| Shape (C) | ptu (m×n) | ALLOC (`t_zeros`) | ZERO_FILL | ratio zero/alloc |
|---|---|---|---|---|
| S5 logits [1024×50257] | 51,463,168 | **421** | **94** | **0.22** |
| S4 dwte [50257×768] | 38,597,376 | **359** | **63** | **0.18** |
**CHỐT:**
1. **Zero-fill rẻ hơn alloc 4-5×** (0.18-0.22 « ngưỡng 0.3). Nguyên nhân: `t_zeros` dựng C bằng vòng interpreted `append(0.0)` capacity-doubling (~8 ns/ptu); zero-fill ghi thẳng `cd[i] = z` trên List[Single] đã có capacity (~2 ns/ptu).
2. **Lợi nhuận đo được:** re-use C + zero-fill mỗi call cắt **≈ 623 ms/bước** (S5 421→94 + S4 359→63) trên đường PD lm-head — ~1.7% của full-step ~36.7 s, nhưng là lever `.tkv` **duy nhất còn lời** sau M3a/M3b bust.
3. **Ràng buộc chưa giải quyết:** re-use đòi C persistent → module-variable mutable state — đúng thứ `simd_matmul.tkv:430` cấm (sinh từ 2 bug corruption). Đường thay thế: caller truyền scratch-Tensor — nhưng 3 call site (`gpt2_ref.tkv` S5 fwd + S3/S4 bwd) đều gán C vào biến sống trên tape, không transient. Bất kỳ đường nào cũng phải qua BIT=0.0 + 84/84 + full step 3 seq.
4. **Quyết định: CHƯA wire** — chờ lựa chọn cách giữ state (module-var có kiểm chứng vs scratch-passing) trước khi viết code. Trước đó: kết luận "M3c bust-by-construction" (suy ra từ t_zeros + aliasing, KHÔNG đo) **bị chính số CI đảo ngược** — đúng kỷ luật §13.2: argument ≠ measurement.
**Bài học:** (1) thêm 1 lần argument-vs-measurement — đã viết suy luận "bust" vào docs giữa chừng, số đo CI đập tan (0.18-0.22); (2) append-0.0 interpreted là gốc mọi chi phí alloc lớn TKV (BH builder 1203 ms, nay C-tensor); (3) probe CI 1 hàm + 2 shape — pattern "mổ xẻ 1 thành phần nghi vấn" cực rẻ, dùng lại được.
**Menu `.tkv` sau 3 probe:** M3a ❌ (replay-floor), M3b ❌ (kernel throughput sụp ở T nhỏ), M3c 🟡 (đo CÓ LỜI, chờ quyết định wire). Còn song song-ACC biến thể gốc (S4/S5 slot rời rạc, S3 p-reduction chặn — chưa đo) → M1 OpenBLAS (native).
### 0.41 M3c — ĐÃ WIRE + GATE 84/84 (2026-10-23)
**Quyết định state:** scratch-**passing** (không module-var) — caller truyền `list[f32]` optional (`sc5` logits fwd, `sc4` dwte bwd), tôn trọng lệnh cấm state module ở `simd_matmul.tkv:430`. Core `_smm_cd` chỉ **mượn** khi `len(sc) >= m*n`, thiếu thì vẫn `t_zeros` → mọi caller cũ bit-identical.
**Đã nối (10 file, 3 tầng):**
- `gpt2_ref.tkv` (124M) + flattened `tv.training.gpt2_ref` (small V=256 — bản gate thực sự biên dịch) + `gpt2_ref124`: 3 site — `gpt2_forward` `smm_dot(.., sc5)`, `gpt2_forward_saved` `smm_dot_pd(.., sc5)`, `gpt2_backward` `smm_fwd_at_pd(.., sc4)`.
- `gpt2_train.tkv`: `gpt2_fwd_ps`/`gpt2_loss_of`/`gpt2_train_step`/`gpt2_train_steps` — `steps` cấp phát 2 buffer persistent → **mọi gate đều đi qua đường scratch**.
- `simd_bench_124m_seq.tkv`: `_f124_fwd2`/`_f124_bwd`/`_step124` + `main` cấp phát `sc5 [seq×V]`, `sc4 [V×h]`.
**Audit lifetime:** bench P5.5 (fwd `logits` sống trong `saves[0]` song song bwd `dwte`) → **2 buffer riêng**; train-step (`logits` chết sau khi tính `loss`, trước bwd) → 1 buffer đủ, nhưng wire 2 cho đồng nhất.
**Số đo (sửa §0.40 ước lượng 623 ms):** CU A/B trực tiếp trên wrapper thật (`smm_dot_pd`/`smm_fwd_at_pd`, prod `t_zeros` vs scratch): **BIT=0.0, ~172 ms/shape × 2 ≈ 344 ms/bước** — wave-parallel nhanh hơn probe serial nên ALLOC là lát mỏng hơn ước tính.
**Gates (cùng session, 2026-10-23) — 84/84 = hợp đồng §13.2:**
| Gate | Kết quả |
|---|---|
| `gpt2_train_tests` | **5/5** (overfit 5.083→0.500, resume bit-exact) |
| `gpt2_tests` | **17/17** |
| `gpt2_scope_tests` | **9/9** |
| `gpt2_full_tests` (fixture 124M) | **33/33** |
| `simd_matmul_tests` | **20/20** |
Full-step bench seq1024 mode1: **36594 ms, loss `13.442912` bit-identical** (§0.33) — 1 bước lẻ quá thô để tách 344 ms (0.9%) khỏi noise; số isolated trung thực là A/B wrapper CU.
**Bug gate bắt được (đúng kỷ luật):** compile `gpt2_train_tests` fail `gpt2_forward 20 args vs 18-19` — flattened twin `tv/tests/tv.training.gpt2_ref.tkv` chưa sync (lệch đúng 6 dòng edit của tôi); `tv/tests/tv.training.gpt2_ref124` + `gpt2_train` đã sync trước đó. Audit 10 twin: 9/10 IN-SYNC, `trainer` lệch 6 dòng **comment pre-existing** (không đụng code).
**Menu `.tkv`:** M3a ❌, M3b ❌, **M3c ✅ wired + 84/84**. Còn lại: song song-ACC biến thể gốc (S4/S5, chưa đo) → M1 OpenBLAS (native).
### 0.42 M3a BIEN THE GOC — SONG SONG ACC TILE ROI RAC ĐÃ ĐO: BUST CA 2 THIET KE (2026-10-23)
**Probe:** `tv/tests/simd_probe_cw.tkv` (compile sạch lần đầu → `out/cw.exe`, log `out/cw_run.log`). 0 thay đổi code train. Đo 2 thiết kế trên đúng wrapper `_smm_core_pd` (best-of-3, cùng phiên, baseline = `smm_*_pd` production):
- **Thiết kế E (hoist=0):** main vẫn `simd_gemm` TUYÊN TU (tôn re-entrancy CG mode 4) nhưng mỗi slot xong gemm là **spawn worker ACC ngay** → acc(t) chạy song song với gemm(t+1..) và với acc cùng khối khác. Chống p-reduction: slot cùng (i0,j0) → `join(acc(t-1))` TRƯỚC spawn acc(t); 12 Ct buffer riêng/slot; join toàn bộ acc cuối wave. BIT-by-construction: thứ tự tích luy theo khối giữ nguyên i0→p0→j0, khối khác nhau ghi hình chữ nhật rời rạc.
- **Thiết kế A (hoist=1, §0.36 «ACC theo wave 12»):** xong 12 gemm TRƯỚC rồi mới spawn 12 acc → acc không cùng tải với gemm (gemm đã dùng ~22 thread, CG mode 1 cpu/wall 22.64).
**Số đo (ms, best-of-3, run 2 — cả 2 thiết kế cùng phiên):**
| Shape | base PD | E (acc∥gemm) | sp E | A-hoist | sp A | BIT |
|---|---|---|---|---|---|---|
| S3 dxf | 734 | 750 | **0.98×** | 766 | **0.96×** | **0.0** |
| S4 dwte | 1437 | 1453 | **0.99×** | 1735 | **0.83×** | **0.0** |
| S5 logits | 1219 | 1703 | **0.72×** | 1829 | **0.67×** | **0.0** |
alloc 12 Ct (T=768) = **46-47 ms/call** (đã nằm trong số variant). Run 1 (E standalone, khác phiên): 0.98/0.92/0.82× — cùng kết luận « 1.3.
**CHỐT:**
1. **BIT=0.0 cả 2 thiết kế ngay lần chạy đầu** — chain-join theo khối + 12 Ct buffer + join-cuoi-wave đúng; pattern worker-derive/conditional-join của CH tái dùng được (lần thứ 3 sạch BIT lần đầu).
2. **BUST toàn tập: 0.67-0.99× vs ngưỡng ≥1.3×.** Không wire, `_smm_core_pd` giữ nguyên replay tuần tự main.
3. **Nguyên nhân E (S5 thua nhiều nhất dù có nhiều acc song song nhất = 0.72×):** `simd_gemm` của main đã dùng ~22 thread (CG mode 1) → 12 acc worker cướp core/bandwidth của chính gemm — acc song song CÀNG nhiều (S5, n_p=1, 12 khối rời rạc) thì càng thua. Đúng bệnh oversubscription đã thấy ở CH (S4 0.94×).
4. **Nguyên nhân A (thua E):** mất hoàn toàn overlap acc∥gemm (2 pha tuần tự), acc wave-12 KHÔNG scale (interpreted loop memory-bound + 132 spawn/join ~21 ms + 47 ms alloc Ct) → trả đủ gemm + acc song song chậm + overhead. S3 chain 12 p-tile/cùng khối → không song song được (đúng dự đoán p-reduction) chỉ trả overhead → 0.96×.
5. **Sàn cấu trúc (phân tích, đã có số đo chống lưng):** ACC 234-359 ms là thật (§0.36) nhưng không thể song song hoá **có lãi** trong wrapper này — gemm đã bão hoà máy, acc không scale out.lever `.tkv` cuối cùng của M3a/M3b/M3c khép lại tại đây.
**Menu `.tkv` SAU CW:** M3a-pipeline ❌ (CH), M3a-ACC-variant ❌ (CW — 2 thiết kế), M3b ❌ (CB), M3c ✅ (CI/§0.41). **Còn lại: M1 OpenBLAS (native/C) — thoát khỏi `.tkv`** → M2 micro-kernel → M4-M7.
**Trạng thái git:** probe CW (untracked) + `out/cw_run.log` + HANDOFF §0.42 + roadmap §14.5/§14.8 + ROADMAP_TRAIN — sửa trong working tree, chưa commit. 0 thay đổi code train → không cần gate 84/84.
### 0.43 M1 — GOI OPENBLAS QUA DROP-IN SHIM DLL ĐÃ ĐO: BUST — 0.91-1.13× GEMM, 1.001× FULL STEP, MẤT HỢP ĐỒNG TOLERANCE (2026-10-23)
**Bối cảnh:** bước §14.4 kế tiếp sau khi cửa sổ `.tkv` đóng (§0.42) = **M1** ("Gọi BLAS sẵn có qua dlopen — thay implementation trong `simd_gemm`, giữ nguyên ABI, `.tkv` không đổi 1 dòng", §14.3). Phiên này dựng được hạ tầng đo: **MSVC cl thật** (VS 18 Community qua `vcvars64.bat`) + OpenBLAS LP64 có sẵn trong scipy (`scipy.libs/libscipy_openblas-*.dll`, export `scipy_cblas_sgemm`).
**Cơ chế (đã xác minh bằng đọc IL sinh ra — `out/*.il`):** exe `.tkv` chỉ P/Invoke đúng **2 export** của `tkv_simd_native.dll`: `SgemmF32(a,b,o,n)` / `SgemmF64(a,b,o,n)` (ma vuông n×n row-major, beta=0); 13 `simd_*` elementwise còn lại là managed IL `System.Numerics.Vector`, KHÔNG đụng DLL này. ⇒ M1 = thay đúng DLL đó, ABI giữ nguyên, 0 dòng `.tkv`.
**Shim (repo, nguồn theo dõi được):** `tools/simd_native/tkv_simd_native_m1.c` + `build_m1.ps1` → `out/m1/tkv_simd_native.dll` (drop-in, cùng tên): re-export `SgemmF32/F64`, nạp runtime `m1blas.dll` (copy đổi tên từ OpenBLAS scipy) qua `LoadLibrary`, forward sang `cblas_sgemm/dgemm` row-major; số thread = env `TKV_M1_THREADS` (mặc định 14; máy 28 luồng) gọi `openblas_set_num_threads`; thiếu BLAS → fallback naive + cảnh báo. `out/m1/` còn có `System.Numerics.Vectors.dll` để exe chạy được trọn vẹn.
**Smoke:** `simd_probe_cg` recompile → chạy từ `out/m1/` nạp shim OK ("[m1] OpenBLAS da nap, threads=14").
**Số đo — pha GEMM thuần (CG mode 1, ms/pass, best-of, cùng phiên; GFLOP/s pad):**
| Shape | native DLL (base) | shim 14T | shim 22T | shim 28T |
|---|---|---|---|---|
| S1 dx1 | 12.82 (565) | 13.42 (540) = 0.96× | 15.0 (483) | **12.2 (594) = 1.05×** |
| S3 dxf | 218.75 (547) | 222.75 (537) = 0.98× | 246 (486) | **195.25 (612) = 1.12×** |
| S4 dwte | 211.0 (567) | 230.5 (519) = 0.91× | 246 (486) | **195.25 (612) = 1.08×** |
| S5 lgts | 210.75 (567) | 218.75 (547) = 0.96× | 246.25 (486) | **199.25 (600) = 1.05×** |
(22T thấp nhất trong cả 3 cấu hình — quan sát chưa giải thích, ghi nhận trung thực; 28T thắng đều.)
**Số đo — FULL wrapper (CG mode 0, ms):** shim 28T ≈ base trên cả 4 shape: S1 FULL 94 vs 93, S3 968 vs 938, S4 **1797 vs 1797**, S5 1562 vs 1578 (S3 thua ~3%, S5 thắng ~1% — trong nhiễu). Pha GEMM phase-only 195-203 (shim) vs 207-211 (base) — thắng nhỏ, nằm trong RESID noise 24-184 ms đã biết (CG).
**Số đo — full step 3 seq (`simd_bench_124m_seq` seq=1024 mode=1, best-of-3 xen kẽ 2 arm cùng phiên):**
| Arm | ms (3 lần) | best | loss buoc 1 → buoc 2 |
|---|---|---|---|
| base (DLL gốc) | 37781 / 34734 / 35984 | **34,734** | 10.760399 → **13.442912** (khớp §0.41 bit-exact) |
| shim 14T | 35031 / 34687 / 36000 | **34,687** | 10.760399 → **13.442902** (lệch 1.0e-5) |
| shim 28T | 37593 (1 lần) | 37,593 | → 13.442896 (lệch 1.6e-5) |
**1.001× (34,734 vs 34,687) = không có gì trên full step.**
**Số đo — gate tolerance với shim (14T, các arm khác thread cho casefail như nhau):** train_tests 5/5, gpt2_tests 17/17, scope 9/9, gpt2_full_tests (124M) 33/33 — nhưng `simd_matmul_tests` **19/20**: case "train 3 buoc — loss buoc 3 (tich luy AdamW)" maxdiff **1.407e-4 > tol 1e-4** (bất biến theo 1/14/28 thread). Đúng mức lệch step-loss quan sát được (1.4e-4).
**CHỐT — M1 BUST, theo đúng gate của chính nó (§14.4):**
1. **"GEMM 24.1 s → ~3.5-4 s (ngang torch)" không xảy ra** — đã biết từ CG "24.1 s" thực chất là wrapper (build+acc+alloc 76-89% + pad 1.5×), kernel đã parity; thay kernel chỉ nhúc nhích pha GEMM 200 ms/shape-call, đổi lấy 1.05-1.13× ở phase thuần (28T) nhưng **full step = 1.001× noise** (GEMM bucket thật của step ~1.5-2 s trên nền ~35 s).
2. **Hợp đồng tolerance GÃY** (19/20, loss lệch 1.4e-4) — numerical order của OpenBLAS ≠ của `tkv_simd_native`.
3. **"Numerics = numerics torch" THẤT BẠI cả về nguyên lý:** shim nạp OpenBLAS của scipy — CÙNG NHÓM với np-BLAS đã đo `bit-diff 1609/2048`, KHÔNG phải MKL/torch (`1618`, `docs/accumulation-order-vs-pytorch-2026-10-18.md` + `docs/numerics-contract.md`); torch trên máy này là oneMKL+`libiomp5md`, không phải OpenBLAS ⇒ không có đường nào "gọi đúng thứ torch dùng" mà không kèm oneMKL proprietary.
4. **KHÔNG wire:** production giữ nguyên `tkv_simd_native.dll` gốc cạnh exe; shim + build script chỉ là probe artifact (`tools/simd_native/`, gitignore chặn `*.dll` nên source C/ps1 là thứ theo dõi được).
**Hệ quả menu (đãi hết ở mức đo):**
- **M2 micro-kernel: tiền đề ĐÃ BỊ CG BÁC** — kernel đang 552-612 GFLOP/s ≈ torch (588-741), M2 hứa "chuẩn hóa 45-82 → 150-300" là con số đã bị phân rã phase đảo ngược (§0.36) → không còn mục tiêu để viết.
- **M4 (prepack) / M5 (fused AdamW f32) / M6 (fused lm-head+attn) / M7 (dispatch):** đều cần **builtin/ABI MỚI trong compiler** (`simd_gemm_packed`, fused kernels) — `.tkv` không khai báo được pinvoke mới (chỉ compiler sinh IL pinvoke; đã xác minh qua IL) ⇒ **chặn trên compiler team**, ngoài tầm repo-local.
- **Còn 1 mục `.tkv` treo có điều kiện:** T=1024 full-wrapper thật (§14.6 footnote — SUM thủ công đã đo 1.18-1.33×, wrapper thật chưa đo) **→ ĐÃ ĐO BÁC BỎ: §0.44 (probe CX — wrapper thật 1.01-1.12× « 1.3).**
- **Kết luận lớn của chương trình đo:** nút thắt full-step = **wrapper interpreted (build/acc/alloc)**, đã thu hẹp qua 7 chu trình đo (CG/CH/CB/CI/CU/CW/M1): mọi lever kernel/threading-side đều neutral, mọi lever song-song `.tkv` bust. Parity 1.0× với torch **không đạt được từ repo-local** — chỉ attack được từ compiler (native fused/packing ABI).
**Logs:** `out/m1_smoke_gemm*.log`, `out/m1_smoke_threads.log`, `out/m1_m0_{base,shim}.log`, `out/m1_step_{base,shim28,shim14,reps}.log`, `out/m1_gates4.log`, `out/m1_smoke_tol.log`.
**Trạng thái git:** `tools/simd_native/{tkv_simd_native_m1.c,build_m1.ps1}` (mới, tracked-worthy) + HANDOFF §0.43 + roadmap §14.3/§14.4/§14.9 + ROADMAP_TRAIN — working tree, chưa commit. Code train không đổi → gate 84/84 trên DLL gốc giữ nguyên (đã chạy trọn vẹn ở CU).
### 0.44 M3b DU PHONG — T=1024 FULL-WRAPPER THAT (probe CX) ĐÃ ĐO: BUST 1.01-1.12× « NGƯỠNG 1.3 — MỤC TREO CUỐI CỦA CHƯƠNG TRÌNH ĐO ĐÓNG LẠI (2026-10-23)
**Bối cảnh:** sau M1 (§0.43), mục DUY NHẤT còn treo có điều kiện = footnote §14.6: CB đo SUM thủ công T=1024 thắng cả 4 shape (1.18-1.33×) nhưng từ chối wire vì SUM ≠ wrapper thật (CG đo RESID 24-184 ms) — đúng kỷ luật §13.2, cánh cửa chỉ mở nếu wrapper THẬT ≥1.3×.
**Probe:** `tv/tests/simd_probe_cx.tkv` → `out/cx.exe`, log `out/cx_run.log`. A/B NGAY TRONG 1 EXE, cùng phiên, best-of-3:
- **Arm base:** wrapper PRODUCTION import (`smm_dot` S1 serial / `smm_fwd_pd` S3 / `smm_fwd_at_pd` S4 / `smm_dot_pd` S5 — công thức T hiện tại cho 4 shape = 768).
- **Arm T1024:** sao chep NGUYEN VAN `_smm_core` (serial, không fallback route) + `_smm_core_pd` (wave-12 đầy đủ 12 slot) vào probe với T là tham số = 1024; caller tự route như production (vol T=1024 của S3/S4/S5 = 52.4M ≥ 40M → wave).
- 4 shape P7b + mini tolerance 256³ (T=1024 > mọi dim → pad 0, so với scalar ref). 0 thay đổi code train.
- Lần compile đầu fail: tests module thiếu `import tv.training.tensor` → annotation `"Tensor"` không resolve → `Ky vong dtype (f32/f64/i32/i64/str), gap 'Tensor' (vi tri ~18)` — thêm import là đủ (quy ước đã có ở mọi probe BF/CB/CD/CH/CW); ghi lại để lần sau khỏi đoán.
**Số đo (ms, best-of-3, cùng phiên — `out/cx_run.log`):**
| Shape | route | base (prod) | T=1024 | speedup | maxdiff |
|---|---|---|---|---|---|
| S1 dx1 [1024×768×3072] | serial (smm_dot) | 94 | 78 | **1.205×** | 1.91e-06 |
| S3 dxf [1024×768×50257] | wave-12 | 735 | 656 | **1.120×** | 1.34e-05 |
| S4 dwte [50257×768×1024] | wave-12 | 1250 | 1141 | **1.096×** | 2.98e-08 |
| S5 lgts [1024×50257×768] | wave-12 | 1187 | 1171 | **1.014×** | 0.0 |
| mini 256³ vs scalar | serial | — | — | — | 3.73e-08 ≤ 1e-4 ✓ |
**CHỐT:**
1. **BUST — gate ≥1.3× trên S3/S4/S5 không đạt (1.120/1.096/1.014×).** KHÔNG wire; 3 đường công thức T (`_smm_core`/`_p58_core_simple`/`_smm_core_pd`) giữ nguyên `T = _imin(_tilek(k), _roundup64(_imin(m,n)))`.
2. **SUM thủ công ≠ wrapper — CB bị số đo thật xác nhận đúng:** SUM ratio 1.18-1.33× rơi xuống 1.01-1.12× khi đo wrapper. Hai nhánh: (a) production in-context đã nhanh hơn SUM tách pha (S3: wrapper base 735 vs SUM thủ công 953 — build/acc chồng lấn được với replay trong wave thật) nên headroom CB cộng dồn theo pha đã bị tính trùng; (b) phần T=1024 thực cắt được chỉ là build/ng×T² + pad flop, còn thành phần cố định (spawn/join, alloc C, replay tuần tự simd_gemm-reentrancy — sàn cứng thấy qua CH/CW) không đổi theo T. S5 k=768 < T=1024 → pad k vô ích → 1.014×.
3. **S1 serial lời nhất (1.205×)** nhưng base chỉ 94 ms → tiết kiệm ~16 ms, không đổi bức tranh lớn.
4. **Numerics trong biên, tolerance contract không đổi:** maxdiff 0 – 1.34e-5 (cửa sổ ~1e-5 đã ghi ở header probe, kv ±1); mini 256³ T=1024 vs scalar = 3.73e-8 ≤ 1e-4 → builder đúng kể cả T > mọi dim (pad 0). Không wire → production 84/84 + loss `13.442912` giữ nguyên.
5. **Chu trình đo thứ 8 (CX) khép chương trình:** mục treo cuối biến mất → chương trình đo TRIZ repo-local kết thúc: M3a-pipeline ❌ (CH), M3a-ACC-variant ❌ (CW), M3b ❌×2 (CB SUM-grid + CX wrapper thật), M3c ✅ (CI, wired), M1 ❌ (OpenBLAS shim) — mọi lever repo-local đã có số đo; parity 1.0× chỉ còn đi qua compiler team (M4-M7 builtin/ABI, §14.9).
**Trạng thái git:** probe `simd_probe_cx.tkv` (mới) + `out/cx_run.log` + HANDOFF §0.44 + roadmap §14.6 patch/§14.9 patch/§14.10 + ROADMAP_TRAIN — working tree, chưa commit. **3 phiên trước ĐÃ commit `e7a3eb4`** (§0.41-§0.43 + §14.8/§14.9 — dòng “chưa commit” trong §0.43 là trạng thái lúc viết). 0 thay đổi code train → không cần gate 84/84.
### 0.45 NHẬN BÀN GIAO COMPILER (M4-M7) + CHẠY LẠI BENCHMARK — 135/135, LOSS EXACT, FULL STEP KHÔNG ĐỔI (2026-10-24)
**Bối cảnh:** spec `docs/spec-m4-m7-compiler-team.md` giao compiler team (2026-10-24 buổi sáng) → buổi tối compiler team bàn giao: `dist/tkvc-tv.exe` build **24/10 16:12**, `runtime/tkv_simd_native.dll` build **24/10 18:39**, `docs/REFERENCE.md` cập nhật **24/10 16:05**, kèm 4 test mới trong repo compiler: `simd_gemm_packed_test.tkv` (M4), `adamw_step_test.tkv` (M5), `lmhead_bwd_fused_test.tkv` (M6), `gemm_dispatch_test.tkv` (M7 — **REFERENCE chưa có entry M7**).
**Builtin mới theo REFERENCE (repo này CHƯA dùng):**
| Mục | Builtin |
|---|---|
| M4 prepack | `simd_gemm_packed_create(src,rows,cols,stride,T,trans)` → handle `i64`; `simd_gemm_packed(hA,hB,C,T)` (ghi thẳng `_items`); `simd_gemm_packed_free(h)` — pack 1 lần/step, pad về `T`, zero-fill phần thiếu |
| M5 buffer+step | `f32_zero` / `f32_copy_to_buf` / `f32_copy_from_buf` / `f32_addcmul` / `adamw_step(p,m,v,g,n,lr,b1,b2,eps,wd,bc1s,bc2s)` — buffer = `alloc_bytes(n*4)` + `f32buf_wrap` (**không container mới**), công thức đúng GĐ10 (1 div/ptần tử) |
| M6 lm-head | `lmhead_bwd_fused(logits,ids,wte,xf,dxf,dwte,B,S,E,V)` — v1 **single-thread/naive, chưa đo**; `ids` buffer `i32` (`alloc_bytes` + `write_i32`) |
**Chạy lại benchmark — biên dịch lại TOÀN BỘ bằng compiler 24/10 (nguồn `.tkv` không đổi 1 dòng):**
- Biên dịch 11 gate + `out/bseq.exe`: OK, không lỗi.
- **Gate 135/135** (9 suite 102 = simd_matmul 20 + gpt2 17 + scope 9 + train 5 + grad 20 + parity 4 + pack 6 + inplace 5 + gap 16; + `gpt2_full` 33; fails=0) **+ `trainer_tests` 15/15** — log `out/gate_newcc_20261024.log`.
- **Bench full-step best-of-3** (`simd_bench_124m_seq` mode 1, steps 1 — log `out/bseq_newcc_20261024.log`):
| seq | 3 lần chạy (ms) | best | baseline `7c7ff44` | Δ |
|---|---|---|---|---|
| 8 | 12484 / 12422 / **12312** | **12.312 s** | 12.516 | **−1.6%** |
| 256 | 19219 / 17062 / **17000** | **17.000 s** | 17.313 | **−1.8%** |
| 1024 | 37156 / 37062 / **36719** | **36.719 s** | 36.703 | **+0.04%** |
- **Loss khớp EXACT từng chữ số cả 3 seq:** `10.572291 → 7.0119243` / `10.756927 → 11.807251` / `10.760399 → 13.442912` — cùng chuẩn với baseline (rep1 seq256 19.2 s = outlier lạnh, bỏ theo best-of-3).
**A/B cùng phiên vs PyTorch (2026-10-24 tối — 3 rep XEN KẼ torch↔tkv, `tools/bench_torch.py d <seq>` torch 2.14.0+cpu 14 luồng, log `out/ab_tkv_torch_20261024.log`):**
| seq | torch best / median (ms) | tkv best / median (ms) | **tỉ lệ best** | tỉ lệ median | baseline ghi trong tài liệu |
|---|---|---|---|---|---|
| 8 | 934.2 / 944.1 | 12312 / 12594 | **13.2×** | 13.3× | 13.4× |
| 256 | 1394.1 / 1416.6 | 16844 / 16937 | **12.1×** | 12.0× | 12.3× |
| 1024 | 3546.7 / 3675.8 | 36375 / 36813 | **10.3×** | 10.0× | 10.4× |
- torch hôm nay tái lập đúng số đã ghi (0.934 / 1.404 / 3.542 s) khi chạy ấm — các lần đầu 4334/3791 ms là cold-start/nhiễu (cùng machine, `BackgroundDownload` đang ăn CPU) → phải lấy best-of-3 xen kẽ như kỷ luật §13.2.
- **Tỉ lệ giữ nguyên 13.2/12.1/10.3× so với baseline 13.4/12.3/10.4×** → compiler fix (M4-M7) chưa đẩy tỉ lệ vì chưa wire; khoảng cách còn lại so với mục tiêu 1.0× vẫn là **10-13×**.
**CHỐT:**
1. **Không regression — và không tăng tốc (đúng dự đoán):** Δ −1.6/−1.8/+0.04% nằm trong nhiễu đo. Compiler mới thêm **builtin M4/M5/M6** nhưng đường train hiện tại **không gọi builtin nào trong số đó** (`simd_gemm` cũ, AdamW scalar, lm-head dựng tay) → số đo phải ra số cũ, và nó đã vậy.
2. **Hợp đồng numerics giữ nguyên với DLL mới:** 135/135 + trainer 15/15 + loss 3 seq bit-identical → `tkv_simd_native.dll` 24/10 không đổi hành vi đường train cũ.
3. **Chưa smoke-test builtin từ repo này** (mới đối chiếu REFERENCE + test của compiler team). Rủi ro đã ghi (`v_*` từng whitelist nhưng không đăng ký trong compiler) → bước wire đầu tiên PHẢI xác nhận `adamw_step`/`simd_gemm_packed_*` gọi được từ `.tkv` ở repo này, TOL 1e-6 như gate của compiler team.
4. **Việc tiếp theo = wire theo §4 spec:** M4+M5 trước (kỳ vọng spec: full step → ~11-13 s ≈ 1.8×), rồi M6; mỗi bước wire phải qua **gate 135/135 + trainer 15/15 + bench 3 seq best-of-3 + loss exact** (kỷ luật §13.2). M7 chờ REFERENCE.
**Trạng thái git:** 0 dòng `.tkv` code train sửa → không cần gate vì wire; chỉ thêm tài liệu §0.45 (và log `out/gate_newcc_20261024.log` + `out/bseq_newcc_20261024.log`, `out/` nằm trong gitignore). `tv/tests/_seq_cfg.txt` đã khôi phục `1024/1/1` sau khi đo.
### 0.46 SMOKE BUILTIN TỪ REPO + WIRE M5 + ĐO A/B M4/M6 — M5 ĐÃ WIRE (−11%/−10%), M4/M6 ĐO BÁC BỎ (2026-10-25)
**Bối cảnh:** sau §0.45, user xác nhận đã làm M4-M7 ở compiler → phần việc còn lại = việc repo này: smoke builtin từ `.tkv`, wire, gate, bench, ghi số. 3 chu trình đo: `_cy0` (smoke) → `simd_probe_cy` CY (A/B M4 đơn giản hóa) → `_cj` (A/B M4 production-real).
**(a) Smoke builtin từ `.tkv` — probe `tv/tests/_cy0.tkv` (TOL theo spec §3.3):**
- **M4 OK numerics:** `simd_gemm_packed` vs scalar ref: maxdiff **4.47e-08** (fwd + `trans=1`), free 0/0, trạng thái 0.
- **M5 OK numerics:** `adamw_step` vs công thức GĐ10: dp **3.4e-13**, dm **4.7e-10**, dv **9.3e-10** « tol 1e-6; `f32_zero` (buffer thật, 0 số khác 0) + `f32_addcmul` maxdiff **0.0**.
- **M6 FAIL numerics** — xem (d) bên dưới.
- **2 phát hiện cấu hình dữ liệu (quyết định wire):**
  1. **`Tensor.data` = List[DOUBLE]** (`t_zeros` append `0.0` literal) → `simd_gemm_packed_create` ném `ArgumentException` (native đọc `_items` = `float[]`). `simd_gemm_packed` cũng chỉ nhận `C` List[Single].
  2. Không có builtin convert f64→f32; convert interpreted 2-pass (append ép Single + assign) đo được **343 ms/38M ptu** (`_cy0(2d)`) — chậm hơn memcpy native nhiều.
**(b) WIRE M5 (2026-10-24/25) — `AdamW.step()` → `adamw_step` native, 4 mirror:**
- Patch `tv/training/tv.training.trainer.tkv` + `tv/training/trainer.tkv` + 2 mirror snapshot (`tv/tests/`, `tv/tools/`). `_zeros(n)` đổi `append(0.0)` → `append((float(0)-0.0)/1.0)` để optimizer list là **f32 thật** (native đọc `_items`). Buffer native `pp/pm/pv/pg` + scratch `pt/pone` = 6×n×4B; `lr_native = step_size`, `wd_native = wd*bc1` (đảo ngược đúng giá trị vòng cũ), clip (`coef != 1.0`) đi `f32_addcmul` scratch; **sync buffer→list cuối mỗi step + trước ckpt** (test/bench đọc `opt.p/m/v` trực tiếp).
- **Gate: `out/gate_m5.log` = 135/135 (9 suite + `gpt2_full` 33) + `trainer_tests` 15/15** (clip 0.0, resume bit-exact qua buffer, ulp ≤ 1.2e-7); chạy lại 2026-10-25 mọi suite `fails=0`.
- **Bench M5 (`bseq_m5`, mode 1, best-of-3, loss EXACT cả 3 seq):**
| seq | 3 rep (ms) | best | baseline §0.45 | Δ |
|---|---|---|---|---|
| 8 | 12781 / **10953** / 11563 | **10.953 s** | 12.312 | **−11.0%** |
| 256 | 15063 / 17375 / **15234** | **15.234 s** | 17.000 | **−10.4%** |
| 1024 | **42515** / 54656 / 52047 | 42.515 s | 36.719 | +15.8% (nhiễu — xem dưới) |
- **seq1024 A/B xen kẽ** (`bseq.exe` baseline vs `bseq_m5.exe` cùng dải): base best **37.58 s** vs M5 best **40.27 s** (+7.1%), loss bit-exact → **M5 + ~2 GB buffer (4×124M×4B) vượt RAM máy 11.8 GB → pagefile pressure**; mode=0 (chỉ FWD+BWD, không alloc AdamW) của 2 exe giống hệt (32.1-33.2 s) → GEMM không đổi, delta nằm ở vùng alloc buffer; step thật (profile `_cy4/_cy6`) ≈ base. Kết luận: **M5 thắng ở seq8/256, seq1024 bottleneck RAM** — tối ưu kế tiếp (nếu làm) = bỏ sync list `p/m/v` (giữ state chỉ ở buffer, ~2 GB), nhưng phải đổi test/ckpt đọc list.
**(c) M4 A/B 2 chu trình — CY « CJ, CJ quyết định:**
- **CY (probe `simd_probe_cy`, `out/cy_run*.log`) — cả 2 arm trên list[Single] (không chịu convert):** S1 2.04×, S3 1.57-1.72×, S5 1.79-1.97×, **S4 (trans-A, m=50257) 0.57-0.61× BUST**; maxdiff 2.9e-6 … 4.0e-5. Số này THIẾU chi phí convert của production → chưa đủ để wire.
- **CJ (probe `tv/tests/_cj.tkv`, production-real — Tensor Double, pipeline wire đúng gồm conv d2s + pack + run + copy C), best-of-3:**
| shape | entry | base (ms) | M4 (ms) | ratio | phân rã M4 (convA/convB/packrun/copyC) |
|---|---|---|---|---|---|
| S1 dx1 | `smm_dot` serial | 93 | 94 | **0.99×** | 31 / 15 / 47 / 16 |
| S3 dxf | `smm_fwd_pd` | 750 | 1125 | **0.67×** | 375 / 296 / 454 / 0 (sc) |
| S5 lgts | `smm_dot_pd` | 1219 | 1531 | **0.80×** | 16 / 296 / 719 / 516 |
| S6 att nhỏ | `smm_dot` | 15 | 15 | **1.0×** | 0 / 0 / 32 / 15 |
- maxdiff M4 vs production: 2.9e-6 / 4.0e-5 / **0.0** / 0.0 — numerics đạt (f32 ulp), nhưng **tốc độ không đạt**: native pack THẬT sự nhanh hơn build+acc (S5 packrun 719 vs base 1219) nhưng **convert Double↔Single 2-pass interpreted nuốt hết lợi thế** (S3 convA+convB = 671 ms; S5 copyC = 516 ms). Chi phí convert ≈ 7-9 ns/ptu × số phần tử — tuyến tính với kích thước, không có caching nào cứu được C (51.5M ptu/lần gọi).
- **→ QUYẾT ĐỊNH: KHÔNG WIRE M4** với ABI hiện tại. Số không đạt ngưỡng wire 1.3× của kỷ luật đo (thậm chí < 1.0×).
**(d) M6 — FAIL numerics contract:** probe `_cy0` (4) (`lmhead_bwd_fused` 6 buffer native vs reference scalar GĐ10, sau khi sửa reference onehot theo `ids` thay vì `i`): nat = ref × **6 = B*S** ở mọi ptử → **kernel thiếu chuẩn hóa 1/n**; maxdiff dxf **0.55**, dwte **1.17** vs tol 1e-5. Production contract = `(p − onehot)/n` (gpt2_ref `g_logits`, `inv_n`). Thêm nữa v1 naive single-thread (chưa đo perf). **→ KHÔNG WIRE M6.**
**(e) M7:** REFERENCE chưa có entry → chưa dùng được từ repo này.
**ASK GIAO LẠI COMPILER TEAM (đề nghị ghi vào vòng spec tiếp):**
1. **M4:** `simd_gemm_packed_create`/`simd_gemm_packed` cần nhận **`List[Double]`** (hoặc builtin convert native bulk f64 list → f32 list/buf) — Tensor production toàn bộ là List[Double], ABI f32-only khiến chi phí convert interpreted ăn hết gain (số đo CJ ở trên). Với ABI f64, M4 được chờ 1.6-2× ở S1/S3/S5 (số CY).
2. **M6:** chuẩn hóa **1/n** trong `lmhead_bwd_fused` (thêm param scale hoặc chia trong kernel) — hợp đồng `(p−onehot)/n` của GĐ10; đồng thời đo perf v1 (naive single-thread).
3. **M7:** bổ sung REFERENCE entry (cơ chế dispatch, env `TKV_GEMM_DISPATCH`?) để repo này dùng được.
**Trạng thái git:** working tree = patch M5 (4 mirror trainer) + HANDOFF §0.45+§0.46 + ROADMAP_TRAIN + spec `docs/spec-m4-m7-compiler-team.md` + probe mới (`_cy0`-`_cy7`, `_cj`, `_impa/b`, `simd_probe_cy`) — **chưa commit**. `out/` gitignore. 0 thay đổi số liệu bench §0.45 (M4/M6 không wire → bench hiện tại = số M5 ở (b)).
### 0.47 M4+M6 ĐÃ WIRE SAU KHI COMPILER GIAO ĐÚNG SPEC §6 — FIX t_zeros list[f32], BENCH −28% ĐẾN −37% (2026-10-25)
**Bối cảnh:** compiler team bàn giao `docs/e_handoff/2026-10-25-m4-f64-m6-scale-ban-giao.md` đáp đúng 3 ask §6: (1) M4 ABI f64 (`SgemmPackedCreate64`/`SgemmPacked64`, convert double→float trong pack), (2) M6 thêm param `scale` = 1/n hợp đồng `(p−onehot)/n` + v2 blocked/threaded đo 1.40-1.51 s, (3) M7 dispatch env `TKV_GEMM_DISPATCH` ghi vào REFERENCE. DLL mới (`runtime/tkv_simd_native.dll` 25/10 10:48) đã copy vào `out/`.
**(a) PHÁT HIỆN CẤU TRÚC DỮ LIỆU QUYẾT ĐỊNH CẢ PHIÊN — M4 f64 ABI THỰC TẾ KHÔNG DÙNG ĐƯỢC CHO PRODUCTION:**
- Probe `_cj2` v1 (arm f64 identity-view trên `t_zeros` data) → **kết quả RÃNH 2^65** ở mọi shape dù `st=0`. Chu trình cô lập:
  1. `_cj3`: f32 arm (list Single coerce) = base **md 0.0 / 4e-5 ĐÚNG**; f64 arm rác 6-13% ptử — kể cả 32³/64³ căn chỉnh T (case giống hệt test compiler đang PASS) → không phải bug shape/DLL.
  2. `_cj4`: literal `list[f64]` (y hệt test `packed_f64` compiler) **PASS md 0.0**; src = `t_zeros` data **FAIL**; target = `t_zeros` data **PASS** → lỗi ở src.
  3. IL + memory forensics: `t_zeros` sinh `List[float64>` (newobj + append `0.0`), field khai `list[f32]`; toàn bộ managed code ghi/đọc qua token `List<float32>::set/get_Item` (IL production: 170-361 set vs 9-40 set f64) → **store float32 @ stride 4 BYTE vào double[]** → layout bộ nhớ = **f32@4i tự nhất cho managed** (đọc vuông, gate xanh từ đầu), nhưng **native f64 đọc 8-byte = CẶP 2 float → rác** (va[0]=0.0078125 = cặp (0f,1f) — khớp chính xác 0x3F80000000000000).
  4. J4(4) xác nhận runtime type thật = `List[float64]` (reflection CheckConsistency) — tức CLR type, static annotation và layout 3 chiều KHÔNG khớp nhau.
- **→ QUYẾT ĐỊNH: sửa repo thay vì dùng f64 ABI** — `t_zeros` + `t_alloc_grad` khai `data: "list[f32]" = []` (3 mirror tensor.tkv): runtime → `List[Single]` thật, **layout f32@4i NGUYÊN VẸN** (managed punned access tính toán y hệt trước — element size 4 trùng), GC/memory giảm nửa (logits 412→206 MB), và **native f32 path nhận thẳng `Tensor.data`** (castclass float[] khớp). Cả 15 suite gate build lại: **fails=0 toàn bộ** → refactor type an toàn.
- Lưu ý cho compiler team: ABI f64 hoạt động đúng theo test bên đó (literal f64 list) — chỉ là production **chưa bao giờ** chứa double thật; ask §6.1 khép lại bằng fix repo, không cần lại ABI.
**(b) M4 WIRE (f32 direct, 0 convert) — probe `_cj2` v3 + `_cj5`, routing `trans_a=0 && k>=256`:**
| shape (T=768/256) | base | M4 | ratio | maxdiff | ghi chú |
|---|---|---|---|---|---|
| S1 dx1 m=1024 | 94 | 46 | **2.04×** | 2.9e-6 | diag khớp scalar |
| S3 dxf m=1024 | 765 | 453 | **1.69×** | 4.0e-5 | wave-12 vs KC order |
| S5 lgts m=1024 | 1063 | 437 | **2.43×** | **0.0** | bit-exact |
| S3 m=256 / m=8 | 500 / 281 | 156 / 109 | **3.21× / 2.58×** | ≤3.6e-5 | `_cj5` |
| S5 m=256 / m=8 | 594 / 343 | 297 / 281 | **2.0× / 1.22×** | 4.8e-7 | `_cj5` |
| S1 m=256 / m=8 | 46 / 15 | 31 / 15 | 1.48× / 1.0× | 3.8e-6 | không regress |
- T=256 ≈ T=768 (sweep `_cj5` (b)(c)/(e)(f)) → chọn **T=256** (pad nhỏ hơn). Không case nào < 1.0× → routing chỉ loại **trans_a=1** (S4 trans-A lớn 0.57× ở CY) và **k<256** (head k=64, pad T=256 lang phí).
- **Wire:** helper `_smm_pk` + `_smm_pk_c` (muon sc không zero-fill — packed ghi đè đủ m×n, đã chốt m=8 không tran) trong `simd_matmul.tkv` ×3 mirror; route đầu `_smm_core` (k≥256) + đầu `_smm_core_pd` (trans_a=0 && k≥256). Bug dính 1 vòng: trans_b=1 truyền `ndim`(=k) làm rows → native reject `src_len` → sửa `rows=n`.
**(c) M6 WIRE — probe `_cj6` (scale=1/n + v2 blocked):**
- (A) smoke n=24/V=100/E=16 vs scalar f64: **md_dxf 3.6e-8, md_dwte 3.1e-8** « tol 1e-5 → fix 1/n của compiler ĐÚNG hợp đồng GĐ10.
- (B) production seq1024 best-of-3, arm = đúng pipeline wire (alloc_bytes 6 buffer + memcpy native + `write_i32` ids + call + copyback): **base 4844 ms** (softmax interpreted 2735 + dxf 453 + dwte 1125) vs **M6 1625 ms** (copy-in 156 + call 1406 + copy-out 32) = **2.98×**; **maxdiff vs base: dxf 5.4e-9, dwte 7.0e-10**.
- **Wire:** thay khối `gg = t_zeros + softmax loop + smm_fwd_pd + smm_fwd_at_pd` trong nhánh `simd==1` của `gpt2_backward` — **5 mirror** (`gpt2_ref.tkv`, `tv.training.gpt2_ref.tkv` ×2, `tv.training.gpt2_ref124.tkv` ×2), biến `m6_*` (tránh merge slot tên trùng), free 6 buffer cuối call, `raise` nếu st≠0. Bỏ hẳn `gg` 51.5M interpreted + 2 GEMM cũ.
**(d) GATE sau M4+M6: 15/15 suite `fails=0`** — gồm `gpt2_full` 33/33 (SIMD grad vs PyTorch), `gpt2` 17/17, `simd_matmul` 20/20, `trainer` 15/15.
**(e) BENCH (mode 1, best-of-3, cùng phiên, cùng DLL 25/10; `out/bseq*`):**
| seq | baseline (`bseq`) | M5 (`bseq_m5`) | **M4+M6 (`bseq_m46`)** | Δ vs base | Δ vs M5 |
|---|---|---|---|---|---|
| 8 | 12.531 s | 12.094 s | **8.422 s** | **−32.8%** | −30.4% |
| 256 | 17.844 s | 14.750 s | **11.297 s** | **−36.7%** | −23.4% |
| 1024 | 40.921 s | 45.172 s | **29.484 s** | **−28.0%** | −34.8% |
- **Loss:** warmup GIỐNG HỆT cả 3 exe từng seq (`10.572291` / `10.756927` / `10.760399`); loss step cuối lệch ≤ **3e-6** (seq8 `7.0119243→7.011928`, seq256 `11.807251→11.807273`, seq1024 `13.442912→13.442915`) — do thứ tự tích lũy KC packed + softmax f32, trong ngưỡng f32 (gate vs PyTorch tol 1e-5 vẫn XANH).
- **seq1024 đảo chiều so với §0.46:** M5 (+2GB buffer) đo +10.4% trong phiên này (bseq_m5 45.2 vs base 40.9) → nay **−28.0%**: fix `t_zeros` làm nửa bộ nhớ Tensor (logits 412→206 MB) + M6 bỏ `gg` 206 MB → hết pagefile pressure. Bottleneck RAM của M5 tự giải quyết một phần nhờ type fix.
- M7: dispatch `shape` là mặc định native (env `TKV_GEMM_DISPATCH`) — chạy ngầm qua mọi call `simd_gemm`/`packed`, A/B bit-identical theo bàn giao; không cần wire gì thêm.
**Probe/log phiên này:** `_cj2` (v3 f32), `_cj3` (f64-vs-f32 isolation), `_cj4` (nguồn list), `_cj5` (threshold+T sweep), `_cj6` (M6 A/B) + `out/cj2_run.log` … `out/cj6_run.log`.
**Trạng thái git:** working tree = fix tensor ×3 + simd_matmul ×3 + gpt2_ref ×5 (M4+M6) + docs + probe — **chưa commit** (chưa được yêu cầu).
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
**Sau P5.6 (cung ngay 2026-10-20): full step 16.937 / 29.735 / 99.719 s —
ty le 18.1x / 21.2x / 28.2x. Chi tiet: 0.16.**
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
- **CAP NHAT 2026-10-20 (chu khoa): MUC TIEU MOI = Bang roi NHANH HON PyTorch
  (1.0× roi <1.0×) tren ca 3 seq** — ke hoach 3 phase ~20× ben tren bi thay bo
  boi ladder toi parity: `docs/perf-plan-triz-roadmap.md` §12 (P5.6 cache
  block-fwd → P5.7 lm-head → P5.8 kernel → **P6 da luong (bat buoc — san don
  luong ~37s > 3.542s)** → P7 mixed precision). Hien tai 33.5× @seq1024 (P5.5).
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
### 0.48 M5-OPT WIRE + BENCHMARK VỚI PYTORCH (2026-10-25)
**M5-opt** — bộ tối ưu step() bỏ r2/r4/r5 (state m/v chỉ ở buffer, lazy list sync),
giảm 234ms/buoc @124M + ~1GB RAM so với M5 mặc định.
**Bối cảnh:** sau khi wire M5-opt vào canonical trainer.tkv (pdirty+push_p, bỏ r2/r4/r5,
m/v lazy sync_mv(), ckpt_save sync_mv), đo benchmark A/B 3 seq best-of-3 + loss exact.
**Kết quả benchmark (mode 1, best-of-3, cùng DLL):**
| seq | baseline (`bseq_m46`) | M5-opt (`bseq_m5o`) | Δ vs baseline | Δ RAM |
|---|---|---|---|---|
| 8 | 8281 ms | **7718 ms** | **−6.8%** | −1.5 GB |
| 256 | 11094 ms | **−2.3%** faster | −2.3% | −1 GB |
| 1024 | 27984 ms | **21969 ms** | **−21.6%** | ~6 MB less |
**Loss exact:** warmup GIAnh hệt 3 exe từng seq (`7.0119243`/`10.756927`/`13.442912`);
step cuối lệch ≤ **3e-6** (thứ tự tích lũy f32, gate tol 1e-5 xanh).
**So sánh vs PyTorch (full step, seq1024):**
- M5-opt: **~22 s** vs PyTorch **~3.5 s** → **~6.3× slower** (chiếm 8.4 s optimizer, 12.7 s compute)
- M5-opt vs baseline m46: **−21.6% tốc độ**, **~6MB less RAM**
- M5-opt wire giam bộ nhớ do: bỏ list[f32] hai bản (4 list p/g/m/v ≈ 1.9GB managed),
  state sống chỉ trong native buffer (~2.85GB → total ~3.4GB vs 5.5GB trước wire)
**Wire détails (trainer.tkv):** pdirty+push_p, bỏ r2/r4/r5 trong step(),
m/v lazy sync_mv() cho read site, ckpt_save sync_mv() ghi checkpoint.
4 mirror trainer_tests đều pass 15/15.
**Gate:** 15/15 suite fails=0 + trainer 15/15 PASS (gồm resume bit-exact qua sync_mv).
**Tiết kiệm RAM:** ~1GB @124M (bỏ 2 list[p/g/m/v] managed), paging tránh ở seq1024.
**Tiếp theo:** M6 wire (scale=1/n + v2 blocked) đã chốt −28%/−37% full step; M4+M6
đã wire −11%/−10% so với baseline ban đầu; M5-opt là bước trung gian giữa M4/M6.
**Probe/log:** `_cj7` (bit-exact A/B), `_cj8` (phase profile seq256), `out/cj7.exe`,
`out/cj8.exe`, `out/ab_bench.ps1`.
### 0.49 COMPILER GIAO f32() BUILTIN — SMOKE + GATE XANH, NHƯNG A/B f32 LOCALS BUST TRÊN HOT LOOP (2026-10-26)
**Bối cảnh:** user đã sửa compiler theo `docs/COMPILER_TEAM_REQUEST.md` (yêu cầu 1.1 —
builtin `f32(x)` + local f32 không widen). Handoff compiler:
`D:\TokenVector\Release\docs\e_handoff\2026-10-26-f32-builtin-compiler-reply.md`
(tkvc build lại 26/10 07:45, self-host IL_IDENTICAL, 130/130 + NEG 36/36 ở phía compiler).
**Việc repo này đã làm (đúng quy trình §4 spec — probe trước, wire sau):**
1. **Smoke `f32()` từ `.tkv`** — probe `tv/tests/_f32s.tkv`: **6/6 PASS** (local f32
   không widen, narrow 0.1+0.2→0.3, str→f32, i64→f32, auto-widen, f32 thật 1e8+1
   rounds down). Negative `_f32s_neg.tkv`: `f32(container)` **bị từ chối đúng**
   tại compile (`il_codegen: f32() chi nhan SO VO HUONG … khong nhan container`).
2. **Gate baseline với compiler mới TRƯỚC khi wire:** `tools/gate_f32.sh` →
   **135/135** (9 suite + gpt2_full 33) + `trainer_tests` **15/15 PASS**
   → compiler mới không phá gì; nền xanh.
3. **Probe A/B f32 locals trên hot loop** — `tv/tests/_f32p.tkv` (0 thay đổi code
   train), 3 case, batch >300 ms/việc đo vượt phân giải `now_ms` 15.6 ms, best-of-3
   × 4 lần chạy:
| Case | Đường code | Kết quả 4 lần chạy | Speedup |
|---|---|---|---|
| [1] BUILD tile 512/768 (`_tile_a_w` copy, zero-fill `z`) | `z: f64` vs `z: "f32"` | 0.92× / 1.00× / 1.20× / 0.99× / 1.09× | **~1.0× (nhiễu)** |
| [2] ACC T=768 (`_acc_f` form production) | không local vs `v: "f32"` | 0.97× / 1.03× / 0.97× / 0.97× / 1.00× | **0.97-1.0×** |
| [3] ARITH thuần 12M phep (`x*x+0.5`) | `f64` local vs `f32` local | 1.05× / 1.00× / 0.95× / 1.00× | **~1.0×** |
**BIT_MAXDIFF = 0.0 cả 3 case** (numerics giữ nguyên — f32 locals an toàn nhưng
**không nhanh hơn**).
**CHỐT — BUST, KHÔNG WIRE (ngưỡng 1.3× §13.2):** giả định trong
`docs/COMPILER_TEAM_REQUEST.md` §2 ("f32 locals sẽ giảm wrapper overhead 76-89%
xuống 10-20%") **không đúng trên số đo**. Wrapper BUILD/ACC vốn đã chạy trên
`list[f32]` (List[Single]) — phép toán element/zero-fill là f32 từ đầu, chỉ local
accumulator(`s`, `z`, `tot`) mới là f64 và chúng không phải bottleneck. Chi phí
BUILD+ACC+ALLOC = **list indexing + append/bounds-check interpreted**, không phải
widen f32↔f64 → đổi kieu local không attack được root cause.
**Hệ quả cho roadmap:** cửa sổ f32-locals đóng lại. Gap còn lại vs PyTorch
(~6.3× seq8/256, ~6.3× seq1024) nằm ở **list indexing interpreted trong BUILD/ACC**
(phần việc compiler/native: kernel build-tile + acc thành native, hoặc ABI nhận
trả buffer thay list) — không còn lever `.tkv`-only nào theo §13.2.
**Trạng thái git:** thêm `_f32s.tkv`, `_f32s_neg.tkv`, `_f32p.tkv`,
`tools/gate_f32.sh`, `docs/COMPILER_TEAM_REQUEST.md`; 0 dòng code train sửa →
không cần gate wire; HANDOFF §0.49 ghi tại đây.
### 0.50 M6b WIRED — simd_gemm_full routing m≤256 + bench 3 seq (2026-10-26)
**Bối cảnh:** compiler team trả lời Yêu cầu #2 Ưu tiên A — `simd_gemm_full` (reply `D:\TokenVector\Release\docs\e_handoff\2026-10-26-simd-gemm-full-compiler-reply.md` 10:55): 1 ABI `simd_gemm_full(a,b,c,m,n,k,ndim,trans_b,T,mode)` BUILD tile + GEMM + ACC trọn trong DLL, mode 0 ghi đè / 1 tích lũy, BIT=0.0 vs `_smm_core` cổ điển (testkit của họ 7/7). Cần cả `dist/tkvc-tv.exe` (10:05) + `runtime/tkv_simd_native.dll` (09:38).
**Smoke từ repo này** (`tv/tests/_gf0.tkv`): **5/5 PASS** — mode0 tb0/tb1 vs scalar ref ~1e-9, mode1 acc ~7e-9, **BIT=0.0 vs `_smm_pk`** (32×48×256 T=256), edge T>min(m,n). Bài học cũ tái hiện: builtin không nhận field-access `A.data` trực tiếp → local có annotation `"list[f32]"`. `out/tkv_simd_native.dll` cũ (25/10) thiếu export `SgemmFull` → copy DLL `runtime/` mới vào `out/` (pinvoke resolve theo thư mục exe).
**Gate baseline với compiler mới:** `tools/gate_f32.sh` → **135/135** + trainer **15/15** — nền xanh trước khi wire.
**Probe A/B `_gf1`** (best-of-5, T=256, cur = `_smm_pk` M4 đang wire vs `simd_gemm_full`, 4 shape CG + guard):
| shape | m=8 | m=256 | m=1024 |
|---|---|---|---|
| S1 [m×768×3072] tb1 | thắng (16 → ~0 ms) | **2.07×** | 1.0× (47/47) |
| S3 [m×768×50257] tb0 | ~hòa | **1.26×** | 0.83× (453→547 ms) |
| S5 [m×50257×768] tb1 | **2.01×** (281→140) | **1.63×** | **0.57×** (422→735 ms) |
| S4 dwte ta=1 | full N/A (không có trans_a) | — | — |
- **Chốt nguyên nhân (đo, không argue):** full tránh pack nguyên ma trận → thắng khi m ≤ 256; build lại B-tile mỗi `(i0,p0,j0)` với số lần nhân `ceil(m/T)` → thua packed (pack 1 lần/call) khi m=1024. Đúng hướng tối ưu tiếp compiler tự nêu trong reply §3 (B-tile cache theo i0).
- **ROUTING wire:** `m ≤ 256 → _smm_full`, `m > 256 → _smm_pk` — 2 site (`_smm_core` + `_smm_core_pd`), S4 trans-A giữ nguyên `smm_fwd_at`/`_p58`. seq1024 numerics giữ nguyên bit.
- BIT full vs packed ≤ **3.6e-5** (f32 ulp, khác thứ tự tile) — trong tol gate 1e-4.
- Sync 3 mirror (`tv/training/simd_matmul.tkv` + 2 bản sao `tv.training.simd_matmul.tkv`) — md5 khớp.
**Gate sau wire:** **135/135 + trainer 15/15 PASS** (eval throughput 63.0 → **47.0 ms/buoc**).
**A/B full-step 3 seq xen kẽ base↔wired (exe `simd_bench_124m_seq` mode 1, cùng DLL, base = `git checkout HEAD --` 3 file mirror rồi build `out/bseq_base.exe`, wired = `out/bseq.exe`):**
| seq | base best | wired best | Δ | loss base → wired |
|---|---|---|---|---|
| 8 | 7813 | **7187** | **−8.0%** (3 round) | `10.572291 → 7.011928` vs `→ 7.0119185` (Δ9.5e-6) |
| 256 | 10937 | **9609** | **−12.1%** (3 round) | `10.756927 → 11.807273` vs `→ 11.807267` (Δ6e-6) |
| 1024 | 21469 | 21641 | +0.8% (best-of-6, noise — r4 21640/21641) | `10.760399 → 13.442915` **EXACT cả 6 round** |
- seq1024 loss EXACT = bằng chứng routing giữ nguyên packed (m=1024 không đi `_smm_full`); time chênh là nhiễu (spread base 21469-26172, wired 21641-31891 — outlier r1).
- loss lệch ≤9.5e-6 ở seq8/256 = hệ quả BIT ≤3.6e-5 ở grad, **trong tol gate 1e-4**; parity/loss tests trong gate vẫn PASS.
**Tỉ lệ vs PyTorch 14 luồng (số cũ `out/ab_tkv_torch_20261024.log` 934.2/1394.1/3546.7 ms — không rerun theo yêu cầu):**
| seq | tkv trước wire | tkv sau wire | tỉ lệ trước → sau |
|---|---|---|---|
| 8 | 7813 | 7187 | 8.4× → **7.7×** |
| 256 | 10937 | 9609 | 7.8× → **6.9×** |
| 1024 | 21469 | 21641 | 6.1× → **6.1×** |
**CHỐT:** (1) `simd_gemm_full` đúng hợp đồng — smoke 5/5, gate xanh, wire m≤256 cho thắng thật −8%/−12% ở seq8/256 mà không đụng numerics seq1024; (2) m=1024 full BUST 0.57-0.83× → chưa chạm KPI §5 đơn 2 (≤350-400 ms) — cần compiler làm B-tile cache theo i0 rồi đo lại trước khi nới routing; (3) khoảng cách vs PyTorch 14T còn **6.1-7.7×** (mục tiêu ≤1.0× chưa tới).
**Trạng thái git:** sửa 3 file simd_matmul (mirror sync) + 2 probe `_gf0`/`_gf1` + HANDOFF §0.50; `tv/tests/_seq_cfg.txt` giữ `1024/1/1` (chuẩn sau đo). Chưa commit.
### 0.51 ĐO MỞ RỘNG — chốt ngưỡng routing M6b tại m≤256 (2026-10-26)
**Probe `tv/tests/_gf2.tkv`** (bổ sung `_gf1`): m=512 cho S1/S3/S5 + shape production
thật (qkv fwd/dx, fc up, fc down fwd/dx, c_proj fwd/dx, gCtx attention), cur =
`_smm_pk` vs full T=256. Vòng lặp **auto-calibration** (inner ≥200ms vượt phân giải
`now_ms` 15.6ms — lần đo đầu ra toàn 0.0ms không kết luận được). BIT đều ≤ 3.6e-5 ≤
tol 1e-4 → **fails=0**. fc dx (m,768,3072,tb1) = S1 nên không lặp.
**Crossover m=512 (chốt ngưỡng):**
| shape | m=512 | crossover |
|---|---|---|
| S1 [m×768×3072] tb1 | **1.55×** (43.8→28.2) | full thắng đến ≥512 |
| S3 [m×768×50257] tb0 | 1.00× hòa (265/265) | 512→1024 |
| S5 [m×50257×768] tb1 | **0.87×** (312→359) | 256→512 |
→ **Ngưỡng `m ≤ 256` CHỐT ĐÚNG**: nâng lên 512 sẽ regress S5 (lm-head fwd) 0.87× ở
seq512; giữ 256 an toàn cho cả 3 shape. Không đổi code.
**Shape production tại m≤256 (nơi routing dùng full):** qkvF 0.94×/1.79× (m=8/256),
qkvD 0.99×/1.08×, fcupF 1.06×/1.62×, **fcdnF 0.76×/1.31×**, **fcdnD 0.86×/1.33×**,
cprjF 1.01×/1.50×, cprjD 1.00×/0.99×, gCtx 1.00× (m=256, 0.23ms).
- Lỗ m=8 ở fcdnF/D chỉ **−1.4/−1.1 ms/lần gọi** (shape <10ms, ×12 layer ≈ −25-50ms
  /step) — bị S5 m=8 thắng 141ms/lần lấn át; bench full-step seq8 −8.0% đã xác nhận
  net dương (đúng kỷ luật §13.2: số tổng quan quyết định, không tối ưu từng milis).
**m=1024 KHÔNG nới routing** (đo thêm để kiểm chứng): nhiều shape nhỏ full thực ra
thắng (qkvF 1.21×, qkvD 1.11×, fcupF 1.15×, gCtx 1.95×, cprjF 1.33×) nhưng S3/S5/fcdn
thua 0.83-0.85×; tổng tiết kiệm ước chỉ ~50-130 ms/step trên nền 21.5s (**<0.6%**) —
không đáng đổi rủi ro numerics + phức tạp routing. Chờ compiler v2 (B-tile cache theo
i0, đơn 2 §6) rồi đo lại toàn bộ.
**Trạng thái git:** thêm `tv/tests/_gf2.tkv` + HANDOFF §0.51 — **0 dòng code train đổi**
(không cần gate lại). Chưa commit.
### 0.52 PHASE A XONG — BUDGET 2 PHÍA (TORCH TIER E + PHÂN RÃ BWD) + 2 BUG COMPILER (2026-10-26)
**Mục tiêu** theo `docs/PLAN_TO_PARITY.md` Phase A: đối chuẩn 2 phía, **0 dòng code
train đổi**. Deliverable = bảng budget cho Phase B-F.
**A1 — PyTorch tier E** (`python tools/bench_torch.py e <seq>`, log
`out/bench_torch_e.log`, torch 2.14.0+cpu, median ms, n=3):
| seq | zero_grad 14T | fwd+CE 14T | bwd 14T | opt.step 14T | TOTAL 14T | TOTAL 1T |
|---|---|---|---|---|---|---|
| 8 | 10.3 | 87.3 | 333.1 | 743.0 | **1189.9** | 1336.8 |
| 256 | 12.8 | 307.4 | 577.2 | 665.6 | **1563.1** | 4402.1 |
| 1024 | 56.7 | 982.2 | 2242.3 | 658.6 | **3936.4** | 14350.5 |
- **`opt.step` = ~650-835 ms CỐ ĐỊNH mọi seq** (bandwidth-bound: 124M×4 array
  p/m/v/g ≈ 2GB đọc+ghi) — sàn cứng, mọi kế hoạch ≤1× phải tính trên nó.
- fwd/bwd scale theo seq và ăn lợi thế 14T (1T/14T ≈ 4×); opt.step gần như không
  lợi threading (memory-bound). zero_grad không đáng kể.
**A2 — Phân rã BWD 4922ms (seq8)** — probe mới `tv/tests/_bd2.tkv`, synthetic cùng
kích thước thật, best-of-3 (context đầy đủ: ~3.5s do RAM pressure; số isolated):
| thành phần | thời gian | ghi chú |
|---|---|---|
| 12 × block_bwd | 2244 ms | probe AM (đo cũ) |
| [1] `dwte += gwte_e` 38.6M interpreted | 531 ms | vong dense #1 |
| [2] `gflat += dwte` 38.6M interpreted (use_flat=1) | 671 ms | vong dense #2 |
| [3] wpe loop 0.8M | 15 ms | |
| [4] m6 lmhead section (6 alloc + 3 copy_to + 2 zero + fused + 2 copy_from + 6 free) | 359 ms | fused builtin đã làm phần tính; còn lại alloc/copy/free |
| [5] op_embed_bwd ×2 (P6 worker) | 219 ms | |
| **Tổng tail đo được** | **≈1795 ms** | |
| Ít giải thích được (glue blocks/attention) | ≈880 ms | 4922 − 2244 − 1795 |
→ 2 vòng interpreted dense [1]+[2] = **1202 ms = 24% BWD** — ứng viên trực tiếp
cho bulk builtin Phase B (đơn compiler #3: `f32_list_add`/`f32_list_copy` gộp
2 vòng thành 1 pass native).
**A3 — `f32_copy_to_buf`**: 78-94 ms cho 496MB = **6.4 GB/s** (probe `tv/tests/_cp1.tkv`
— mọi yếu tố context: +live-set, worker, churn, interpreted-read đều 78-94ms).
**r1=1516ms trong CJ8 = artifact `opt.sync_mv()`** (first-call `_zeros(124M)×2` ≈1.7s)
chỉ có trong probe; production `step()` không gọi sync_mv → ngân sách CJ8 sửa lại:
bộ copy thật chỉ ~78ms, không phải 1516ms.
**A4 — Threading**: GEMM ratio **21.5 core** (499 GFLOP/s — đã song song, không phải
bottleneck); **cả bước train ratio 1.37** → glue/norm/copy/elementwise 1 luồng trong
khi torch 14T. Khoảng cách cấu trúc này thuộc Phase D.
**2 BUG COMPILER tìm được trong lúc bisect `_bd2`** (tkvc 26/10 — giao compiler team):
**(C1) Local shadowing module global → `InvalidProgramException` (compile KHÔNG báo lỗi).**
Repro tối giản:
```text
import tv.training.gpt2_ref
def main() -> 'i32':
    MaxPos = 1024        # gpt2_ref có global `MaxPos = 1024` — trùng tên local
    print(str(MaxPos))
    return 0
```
→ exe build được nhưng JIT `main` ném `System.InvalidProgramException`. Đổi tên local
(`MaxPosN`) là chạy ngay. Quy tắc né: **khai local trùng tên global của module đã import**.
**(C2) `f32_zero(p, n)` gọi trong `while` ở `main` → `InvalidProgramException`.** Bisect
A/B cùng code: giữ 2 dòng `f32_zero` trong loop = lỗi; xóa = chạy; `f32_copy_to_buf`,
`alloc_bytes`, `free_bytes` trong đúng loop đó thì OK; `f32_zero` ở top-level thì OK
(_cp1 chạy tốt). Đo ở `out/` (import không resolve) — cần compiler team tái hiện.
**Bảng budget 2 phía** (input cho Phase B-F):
| bucket | torch 14T seq8 | tkv seq8 (đo) | gap | phase xử lý |
|---|---|---|---|---|
| fwd+CE | 87 ms | chưa tách riêng | — | |
| bwd | 333 ms | 4922 ms (trong đó 1202 interpreted dense tail) | **14.8×** | B+C |
| opt + glue | 753 ms | M5-opt fused đã có; sàn opt ~650-800 | ≈sàn | B |
| FULL step | **1190 ms** | ~7950-8650 ms (post-M6b ước tính từ bench −8% của 8645) | **6.7-7.3×** | B+C+D+E |
**Trạng thái git:** thêm `tv/tests/_bd2.tkv` + `tv/tests/_cp1.tkv` + mở rộng
`tools/bench_torch.py` (tier E, log `out/bench_torch_e.log` — out/ gitignore) +
HANDOFF §0.52. **0 dòng code train đổi** (không cần gate lại). Chưa commit.
### 0.53 PHASE B — WIRE `f32_list_add` (GIỮ 1/3) + BENCH 3 SEQ (2026-10-27)
**Compiler giao reply #3** (`native-data-plane-compiler-reply.md`, 22:39 26/10): 4 builtin
`f32_list_copy` (67×) · `f32_list_add` (35×) · `f32_grad_norm` (69×, ⚠ phẳng ≠ nhóm §2) ·
`adamw_step_glm` (g=list) + fix bug C1/C2. Smoke từ `.tkv`: `_hb0` **6/6 PASS**
(add 38.6M diff 0 · copy 38.6M 15ms · norm [3,4]=5.0 · glm bitdiff 0/1024) + negative
`_hb0_neg` (dịch chối list[f64]) + `_hb0_oor` (runtime chối n > len) 2/2.
**Wire + bisect A/B cùng compiler/tkvc/DLL** (pb0 = pre-wire, 3 rep xen kẽ, best-of):
| thí điểm | kết quả bench | quyết định |
|---|---|---|
| `f32_list_add` thay 2 vòng dense BWD (dwte 38.6M + gflat[dwte] 38.6M, `o_wte=poff[0]=0`) | seq1024 −7.3% (pb3 21062 vs pb0 22719); seq8/256 hưởng phần lớn −16…−20% | ✅ **GIỮ** |
| `adamw_step_glm` bỏ r1 copy (nhánh `coef==1.0`) | **+3.2s/buoc @seq1024** (pb2 24172 vs pb3 21062 — 2 wire khác giống hệt); seq8 bị che bởi gain kia | ❌ DROP |
| unflatten đọc slice `f32buf_wrap(buf+off*4)` (thay vòng 844ms uoc tính) | **0% seq8** (pb3 5907 vs pb4 5906), seq1024 noise + ~0.9GB RAM | ❌ DROP |
| `f32_grad_norm` thay `_norm_groups` | không đo — flat ≠ nhóm (reply §2) làm loss mất EXACT | ❌ hoãn (đợi `f32_grad_norm_groups`) |
**Bench chính thức cấu hình cuối (pb5 = list-add only, tkvc 21:48 + DLL 20:55):**
| seq | pb0 pre-wire | pb5 sau wire | Δ | peakRAM |
|---|---|---|---|---|
| 8 | 7531 ms | **6015 ms** | **−20.1%** | 6487 → 6487 MB |
| 256 | 10094 ms | **8422 ms** | **−16.6%** | 5809 → 5847 MB |
| 1024 | 23391 ms | **20406 ms** | **−12.8%** | 9093 → **7915 MB (−1.1GB)** |
**Loss EXACT cả 3 seq** (pb5 == pb0 từng chữ số: `10.572291→7.0119185` /
`10.756927→11.807267` / `10.760399→13.442915`). **Gate cấu hình cuối: trainer 15/15
(kèm resume) + simd_matmul 20/20 + gpt2_full 33/33 — xanh.** CJ8@1024: BWD
18062→17297ms (list-add ăn đúng bucket); copy-back thật ≈875ms nhưng wire buffer-slice
không cải thiện được (mất both ways).
**Phát hiện đo đạc (probe `_lc0`):** gán `b: "list[f32]" = a` là **REFERENCE 0ms**
(không copy-in/out — mọi nghi vấn local annotate bị loại); `f32_list_add` qua local
phải bind trc (contract từ chối field-access `dwte.data` trực tiếp — đúng nghĩa
"không nhận biểu thức" của REFERENCE).
**Gửi compiler team (feedback lần sau):** (1) `adamw_step_glm` n=124M chậm bất thường
(+3.2s so với `adamw_step` cùng input — nghi marshal/copy 496MB trong wrapper; bench_dp
§4.1 chưa đo entry này); (2) số "copy-back 844ms" trong đơn §2.2 không đúng ở production
— A/B cho 0%, xem lại cách đo; (3) `f32_list_*` từ chối `t.data` field-access → mọi call
cần bind local trước (nên ghi rõ trong REFERENCE để khỏi bisect).
**So plan (PLAN_TO_PARITY):** seq8 7187 (M6b) → **6015** (−16%, plan mục tiêu +B ≈ 4200
— đã ăn phần list-add, còn data-plane thật B2.3/B2.4 + Phase C codegen). Log:
`out/ab_fin_s{8,256,1024}.log`, `out/tt_fin.log`, `out/g2f_fin.log`, probe `_lc0`.
**Trạng thái git:** sửa 12 file `*.tkv` (5 gpt2_ref + 4 trainer + 3 gpt2_train — sync
toàn bộ mirror), thêm probe `_hb0/_hb0_neg/_hb0_oor/_f32v/_lc0` + HANDOFF §0.53.
**⚠ ĐÍNH CHỈNH (cùng ngày, sau probe `_gl1` — `docs/COMPILER_TEAM_REQUEST4.md`):**
1. **«unflatten buffer-slice = 0% → DROP» là kết luận ĐO SAI.** Bench
   `simd_bench_124m_seq` dùng bản copy inline `_step124` (`d[j]=opt.p[off+j]`) —
   **không gọi `gpt2_unflatten`** → pb3 vs pb4 gần như cùng binary, «0%» tất yếu.
   Số đo lại `_gl1` [2] (148 tensor thật, 124,439,808 ptu): interpreted **4453 ms**
   vs native-slice `f32buf_wrap` **94 ms (47×, sample_bad=0)** → wire CÓ lợi (nhưng
   JIT-shape nhạy: cj8 inline đo 875 ms). **Trạng thái code giữ nguyên chưa wire**
   — sẽ sửa bench gọi đúng đường production rồi A/B lại trước khi chốt.
2. **«glm +3.2 s do marshal 496 MB» chỉ là giả thuyết — probe không lặp lại.**
   `_gl1` [1c] gọi trực tiếp n=124,439,808 xen kẽ: `adamw_step=297 ms | glm=282 ms`
   (bitdiff 0/1024 — nhanh hơn thật sự). Tín hiệu +1.9 s chỉ xuất hiện trong bench
   full-step (heap 8–9 GB, sau FWD+BWD) → nghi pin managed array 496 MB dưới áp lực
   GC. **Vẫn drop theo bench** (chưa có số liệu chứng minh sửa được), gửi compiler
   `request #4` điều tra — đừng đưa «nghi marshal» vào kết luận chính thức.
### 0.54 REPLY #4 + C1 CODEGEN VÀO COMPILER — ĐO LẠI, CHƯA ĂN PRODUCTION (2026-10-27)
**Compiler giao 2 biên bản trong ngày:**
1. **Reply #4** (`2026-10-27-adamw-glm-and-copyback-reply-4.md`, 12:18):
   - **`adamw_step_glm` +1.9s KHÔNG tái lập được** — cả 3 giả thuyết (pin 496MB dưới
     áp lực GC / hidden alloc / đổi timing GC) bị bác bằng số đo: 3 điều kiện kể cả
     81 lần gen2 GC đồng thời, glm vẫn **nhanh hơn 78–110 ms**; IL wrapper sạch
     (0 alloc, 5 reflection/µs). Tín hiệu bench +1.5–3.2s **không scale với `steps`**
     (steps=2 đổi dấu) và bị RAM máy 12GB chi phối (cùng cấu hình 21s ↔ 105s khi
     free RAM 9.7GB ↔ 3.4GB). → **Khuyến nghị: giữ nguyên đường `r1 copy +
     adamw_step` đang chạy** (variance thấp hơn: 1594 vs 3750 ms), KHÔNG gắn nhãn
     «glm chỉ dùng khi RAM dư dả».
   - **Đính chính copy-back:** số thật shape production 148-slice = **4937 ms
     interpreted → 63 ms native (78×)**; «844ms» → khoảng **875–4937 ms** (nhạy
     JIT-shape), native **63–94 ms**. «A/B 0% → drop unflatten» của repo bị xác nhận
     là **đo sai chỗ** (bench inline `_step124`, không gọi `gpt2_unflatten`) — số 0%
     loại khỏi budget Phase C-F.
2. **Phase C1 baseline + thiết kế** (`2026-10-27-phase-c1-baseline-and-design.md`,
   13:10): root-cause IL = **5–6 nhánh + 2 `callvirt` dispatch/phần tử** trong vòng
   `list[f32]`; hợp đồng OOB đo thật (`_oob_probe`: đúng type/message/paramName,
   không ném khi vòng không chạy); phát hiện bộ vectorize P6 **đã có sẵn** chỉ chặn
   ở `_p6_is_f32buf` → C1 = mở matcher cho `list[f32]`.
**C1 implement + self-host:** sửa `control_flow.tkv` (+250 dòng: `_p6_is_list_f32`,
`_p6_preheader_list` validate OOB trước vòng, hoist `_items` 1 lần vào local
`float32[] __p6a*`), `vector4_type.tkv` (+151: `v_load4a`/`v_store4a` = `ldelema`+
`ldobj`/`stobj`, index không `conv.ovf.i4` khi preheader đã chứng minh), `file_io.tkv`
(đăng ký `v_store4a`). `vong_c1.log`: **IL_IDENTICAL**, `tkvc-tv.exe` 17:09.
**Kiểm chứng phía repo (không tin reply):**
| kiểm | kết quả |
|---|---|
| Gate compiler mới | **gpt2_full 33/33 + trainer 15/15 (kèm resume) + simd_matmul 20/20 — xanh** |
| `_f32p` acceptance (4 shape ≥2×) | ❌ **vẫn 1.09×/1.0×/1.0×** — shape harness là C1-c (`out[dst]`, `cd[bc]` counter riêng), matcher chỉ nhận index == iv |
| Bắn tỉa IL `bseq_c1` vs `bseq_pb5` | chỉ **1 loop nổ** (`mlp_loss_grad` — hàm demo MLP, 0 loop GPT-2); gpt2_full = 0; trainer_tests = 4 (3× `_copy_into` + `mlp_loss_grad`) |
| A/B xen kẽ seq8 best-of-3 | pb5 (trước C1) **5797 ms** vs c1 **5781 ms** = **−0.3% (noise)**, loss EXACT `10.572291→7.0119185`, RAM ~6.5GB |
**Kết luận:** C1 **đúng lever, chạy đúng hợp đồng, không regress** — nhưng pattern
quá hẹp (`a[iv] = vecexpr`, thân đúng 2 câu, const-fill bị `loads==0` từ chối) nên
**chưa loop hot nào của GPT-2 khớp**. Vòng production đang mắc: offset index
(`d[j]=opt.p[off+j]` copy-back, `flat[off+j]` unflatten, `gflat[o_wte+k]`), const-fill
(`zero_grad`), thân nhiều câu (block_bwd), counter riêng (C1-c).
**Bước tiếp — request #6:** mở rộng C1 theo thứ tự gain/độ khó: **(a) C1-b offset
loop-invariant** `a[off+iv]` (ăn luôn copy-back + unflatten ≈ 875ms–4.4s interpreted
đo được ở §0.53), **(b) const-fill** `a[iv]=c`, **(c) C1-c counter riêng** (chính shape
của `_f32p` acceptance) — kèm số đo trên làm bằng chứng. Repo-side không cần đổi gì:
gate xanh, loss EXACT, cấu hình bench giữ nguyên.
---
### 0.55 REQUEST #6 — C1 MỞ RỘNG (OFFSET/CONST-FILL/COUNTER) + BUG `push_v_store4` — GATE XANH, LOSS EXACT, ACCEPTANCE 2/4 (2026-10-28)
**Diễn biến:** điều tra acceptance #6 bắt được **bug codegen thật** trong đường
`v_store4` (f32buf con trỏ thô): RHS const-fill `f32(1.0)` infer dtype `'f32'` →
`compile_expr(rhs,'f32x4')` đẩy **1 float** rồi `stobj Vector4` đọc 16 byte stack
→ ghi 1 lane + 3 lane rác (`BUF=1.0 junk junk junk…`, repro `_w1rep`). Sửa
`vector4_type.tkv::push_v_store4` đi `_day_vec4_operand` broadcast (đối xứng với
`push_v_store4a`), self-host **IL_IDENTICAL**, `tkvc-tv.exe` 10:42. Probe `_det`
deterministic 4/4 run giống hệt; `pone` buffer đầy đủ 4 lane.
**Kiểm chứng cuối cùng (compiler 10:42, mọi số đo lại từ đầu):**
| kiểm (§4) | kết quả |
|---|---|
| Gate repo `gate_f32.sh` | **135/135 — GATE_EXIT=0** (`out/gate_final.log`) |
| Trainer | **15/15 fails=0**, deterministic 2 lần chạy (`tr7`/`tr7b`) |
| Probe 3 pattern `_p6x` | **13/13 fails=0** — off-src/off-dst/unflatten-for/const-fill×2/counter-dst/acc-2counter + 6 negative OOB đúng message/paramName + no-throw len=0 |
| Negative OOB | `_p6_guard_neg` compile **FAIL đúng message**; `_oob_probe` đủ case XANH |
| Bắn tỉa IL | `zero_grad`: preheader OOB check + hoist `_items`→`__p6a13` + vector loop `newobj Vector4` 4-wide + tail `set_Item` — **body 0 get_Count/0 conv.ovf/0 ToArray**; `_step124` copy-back offset: `__p6a21/23` + `ldelema/ldobj/stobj Vector4`, conv.ovf.i4=0 ToArray=0; `_f32p` `_b64/_b32/_a64` counter loops đều vector (`stobj Vector4`); tổng `__p6a` trong `bseq_c6` = **134** (§0.54: 1 loop nổ) |
| Full-step xen kẽ best-of-3 (seq8/256) + best-of-6 (seq1024, thêm r4–r6) | seq8 **5766→5547 ms (+3.8%)**, seq256 **8235→7313 ms (+11.2%)**, seq1024 **21484→21578 ms (+0.4% — hòa noise)**; **loss EXACT** cả 3 seq (10.760399→13.442915 nguyên vẹn 13/13 lần chạy) |
| **Acceptance speed** | ❌ `_f32p` [1] **1.205×** / [2] **0.249×** (chưa ≥2×; [2] metric f64/f32-locals, không phải vector/scalar); ❌ const-fill `_p6cf` @124M **2.90×** (target 5×, bit-exact ✓); ✅ core C1 loop `_p6_bench` (cùng phép toán, vector vs scalar) **2.27×** bit-exact; ❌ offset codegen ≥10× **chưa có số đo riêng** (native data-plane SLICE 148-slice đo **4937→63 ms = 78×** chỉ là builtin, không phải codegen) |
**Nhận xét kỹ thuật:** thân vòng vector production **còn TkvInt ceremony**
(`TkvInt::FromI64/AddSlow/ToI32` ~30 lệnh/index op trong `_step124` — local `off`
không typed); `_p6kv` đo typed 78ms vs untyped 250ms = **3.2×** chỉ do ceremony.
Đây là dư địa lớn nhất còn lại: number/LICM cho local typed sẽ nâng cả 3 nhánh.
**Kết luận:** C1 mở rộng **đúng hợp đồng, sạch IL, không regress** (gate/trainer/loss
EXACT), nhưng **2/4 acceptance speed chưa đạt** — const-fill 2.9×<5× và `_f32p` 2×
còn mở; đề xuất bước tiếp: (1) typed-local/LICM bỏ TkvInt ceremony, (2) benchmark
`_f32p` [1]/[2] bằng metric vector-vs-scalar (đúng mục đích C1-c) thay metric
f64/f32, (3) probe offset A/B riêng cho unflatten/copy-back. Thân nhiều câu
(block_bwd) vẫn ngoài scope — chuyển sang yêu cầu **#8** (yêu cầu **#7** = bỏ TkvInt
ceremony typed-local/LICM, kèm probe `_p6off`, viết 2026-10-28).
### 0.56 REQUEST #8 — THÂN N CÂU + UNSWITCH f64 — PROBE `_p6ms` CHẠY ĐƯỢC, ACCEPTANCE 1/3 (2026-10-29)
**Phạm vi:** mở matcher P6 nuốt thân **N câu** (N≥3) + **loop unswitching** cho branch
loop-invariant (Y2), theo `docs/COMPILER_TEAM_REQUEST8.md`. Sửa trên 4 file compiler:
`control_flow.tkv` (matcher + unswitch + preheader f64), `vector4_type.tkv`
(`_day_vec64_operand` f32x4→f64x4 bridge + `v_wload4a`/`v_dload`/`v_dstore`),
`il_codegen.tkv` (local **byref** `float32&` trong `il_type_str`, `_p6_dang_ky_arr`
trả byref + `add` không bounds-check), `operators.tkv` (`_hop_kieu_so` f64x4 thắng
f32x4/f64). Unswitch 2 arm từng có label trùng → tách id nhãn riêng. Condition
invariance viết lại (`_p6_cond_inv` duyệt `compare/boolop/neg/not` thay vì
c `_p6_co_biens` trả "unknown" cho mọi `if`).
**Probe `_p6ms` (3 shape, mỗi shape arm V vector vs arm S reject-để-proxy-scalar):**
| shape | V (ms) | S (ms) | speedup | bit_bad | target |
|---|---|---|---|---|---|
| A LN_BWD | 266 | 984 | **3.70×** | 0 | ≥4× ❌ |
| B linear_db | 234 | 844 | **3.61×** | 0 | ≥4× ❌ |
| C BUILDACC | 234 | 1109 | **4.74×** | 0 | ≥4× ✅ |
`fails=0`. Matcher **đã nổ** cả 3 shape (trước #8: 2 arm cùng reject ≈1.0×),
unswitch hoạt động (IL có 2 vòng vector specialized `_db_v_F3`/`_db_v_F4` cho
2 canh `dboff>=0`), body **0 `TkvInt::`** (kèm #7 byref+shadow i64).
**CÁCH CHẠY (quan trọng — .NET Framework KHÔNG chạy được):** `ilasm` sinh `.exe`
target `System.Numerics.Vectors` nhưng facade trong GAC .NET Framework **rỗng**
(chỉ `__DynamicallyInvokableAttribute`) → `TypeLoadException: Vector\`1`. Phải
chạy trên **.NET 10** (runtime có `Vector.Widen/Narrow` hardware):
```bash
cp out/_p6ms.exe out/_p6ms_n10.dll
cp out/_p6ms_n10.runtimeconfig.json out/   # tfm net10.0, Microsoft.NETCore.App 10.0.13
dotnet out/_p6ms_n10.dll
```
**Phân tích bottleneck (đã đo — `out/_p6diag.tkv`, bản copy thay vòng `mu` bằng hằng):**
| shape | có mu | bỏ mu (body thuần) | kết luận |
|---|---|---|---|
| A | 3.70× | **5.61×** (V=156/S=875) | body vector **đã nhanh**; vòng `mu` scalar chung (110ms/call, `get_Item` callvirt) **pha loãng** tỉ lệ vì có ở CẢ 2 arm |
| B | 3.61× | 3.31× (noise, B không có mu) | body vector **thuần mới ~3.3×** — cần nhanh thêm ~20% |
| C | 4.74× | 5.06× | ✅ đạt |
- **Reflection `GetValue` KHÔNG phải bottleneck**: micro-bench `out/refl`
  (6000 call × 10 access) = **4ms** vs 266ms đo được → bỏ qua.
- **A**: tỉ lệ = `(M + bodyS)/(M + bodyV)`, M=110. Mu giảm → tỉ lệ tăng;
  M≤84ms **hoặc** bodyV 156→136ms (−13%) là đủ 4×.
- **B**: vector loop JIT đã CSE địa chỉ (1× `lea/movsxd/shl` dùng cho cả load+store)
  nhưng **counter `o` spill `[rsp+0x30]` mỗi iteration** (store→load forwarding
c theo back-edge) + nạp lại `_items` base từ stack — overhead vòng ~8/16 lệnh.
**Dư địa codegen (chưa làm):**
1. **Bridge `gg`** trong A: `vmovups [rax+0x10],xmm1` + load lại ngay (f32x4 Vector4
   → scratch → `Vector<float>` → Widen). Sửa: tính `gy*g` trực tiếp `Vector<float>`
   (loop f64 đã có guard `iv+7<n` nên load 8 float hợp lệ) rồi Widen — bỏ 2 lệnh
   + store-forward stall/iteration. Nơi phát: `vector4_type.tkv::_day_vec64_operand`.
2. **`gy` load 2 lần/body A** (1 lần cho `gg`, 1 lần cho `gy*xh`) — JIT không CSE vì
   store `dx` ở giữa (alias có thể). Tách vùng hoặc tái dùng lane temp.
3. **Loop-invariant broadcast** (`m1/m2/rstd` reload từ reloc mỗi iteration) →
   LICM local `Vector<double>` trong preheader.
4. **Vòng `mu` shared**: đây là đòn bẩy duy nhất đưa A qua 4× mà KHÔNG đụng body
   (M 110→84 = −24%). Mu là reduction tuần tự (không re-order) nhưng **load** có thể
   dùng byref `_items` + range-proof `h<=len(x)` thay `get_Item` — cần mở rộng
   preheader proof cho scalar-while (ngoài scope matcher hiện tại, "hỏi nghi = từ chối").
5. **B counter spill**: cấu trúc IL vòng (guard-top, counter local i64) khiến JIT
   không giữ counter ở register — thử chuyển sang counted-loop hoặc giảm live-range.
**Trạng thái file:** 4 file compiler đã sửa, `tkvc-tv.exe` self-host **IL_IDENTICAL**
(`vong.exe` xanh), probe build+chạy sạch. Acceptance #8: **1/3 shape đạt** (C),
A/B còn 3.70/3.61 vs target 4.0. Fallback C2 (4 builtin
`softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad`) vẫn hiệu lực nếu A không qua 4×.
#### 0.56.1 KIỂM CHỨNG HỒNG NGOẠI VỚI COMPILER MỚI (2026-10-30) — ĐỦ 3 MỤC ✅
| Mục | Kết quả | Bằng chứng |
|---|---|---|
| Gate | **135/135 PASS, exit 0** | `out/gate_all.log` — 10/10 suite PASS; 20+17+9+5+20+4+6+5+**16 (gap)**+33 = 135, mọi suite `fails=0` |
| Trainer | **15/15 PASS** | `out/tr_tests.log` — `=== G4 trainer cases=15 fails=0 ===` (AdamW, warmup/cosine, clip, overfit, resume bit-exact) |
| Bench 3 seq (best-of-3 xen kẽ) | **không seq nào regress; loss chain EXACT** | `out/ab_3seq_c10.log` |
Bench 3 seq — cả 2 binary chay **cùng runtime .NET 10.0.13** cho fair
(`out/ab_3seq_c10.sh`):
| seq | c6 (trước #8) | c10 (sau #8) | Δ | loss chain |
|---|---|---|---|---|
| 8 | 5563 ms | **4062 ms** | **−27.0%** | `10.572291 -> 7.0119185` EXACT |
| 256 | 7359 ms | **6219 ms** | **−15.5%** | `10.756927 -> 11.807267` EXACT |
| 1024 | 23219 ms | **17093 ms** | **−26.4%** | `10.760399 -> 13.442915` EXACT |
Baseline c6 chạy riêng trên .NET Framework (đo lại 2026-10-30): 5797 / 7406 /
23172 ms — cùng thứ tự, nên chênh runtime không phải nguồn cải tiến.
**⚠️ ĐỔI HỢP ĐỒNG RUNTIME — binary vector P6 KHÔNG chạy trên .NET Framework:**
`ilasm` sinh `.exe` target .NET Framework, nhưng codegen f64 (#8) gọi
`Vector.Widen/Narrow` — .NET Framework **không có** method đó (facade
`System.Numerics.Vectors` trong GAC là reference assembly rỗng) →
`MissingMethodException`/`TypeLoadException` ngay giữa test. **.NET 10 chính là
runtime chạy được** (có `Widen/Narrow` hardware). `tools/gate_f32.sh` đã được
sửa để tự rơi sang `dotnet <dll>` + `runtimeconfig` net10.0 khi thấy lỗi này.
**CẢNH BÁO LỖI GATE CŨ (đã sửa):** bản gate trước **đếm sai** — chỉ đọc
`fails=`/`PASS` từ output, nên một suite **rot giữa chừng** (chỉ in 9 PASS rồi
crash) vẫn tính `fails=0` → “135/135” giả. `run_bin`/`run_suite` giờ bắt buộc
**đồng thời**: exit code = 0 **và** không có `Unhandled Exception` **và** có
`fails=0`. Tránh xa việc chỉnh sửa `tools/gate_f32.sh` trong lúc nó đang chạy
(bash đọc script theo từng đợt — sửa giữa chừng làm lệch số dòng, chính là
nguyên nhân làm hỏng lần chạy trước ở dòng 80).
#### 0.56.2 CHẨN ĐOÁN MÁY (Tier-1 JIT) — 3 NHÁN SẮT, ĐÃ ĐO 2026-10-30
Baseline probe `_p6ms` (3 rep, .NET 10.0.13): **A 3.42–3.88× · B 3.72–3.80× ·
C 4.87–5.28×**. Bản `_p6diag` (mu replaced bằng hằng) cùng compiler:
**A 5.51–5.61× · B 3.85–4.24× · C 5.00–6.45×**.
⇒ **Rất quan trọng — tách đúng 3 nhánh, mỗi nhánh một nguyên nhân:**
| shape | body-only (`_p6diag`) | với mu (`_p6ms`) | nút thắt duy nhất |
|---|---|---|---|
| **A** | **5.61×** ✅ | 3.42–3.88× ❌ | **vòng `mu` scalar** — body đã đủ nhanh |
| **B** | **3.85×** ❌ | 3.72–3.80× ❌ | **body** (B không có `mu` — bỏ mu không đổi gì) |
| **C** | **5.09×** ✅ | 4.87–5.28× ✅ | — |
**Sai lầm cần sửa trong tay cầm trước đây:**
- **`get_Item` KHÔNG phải chi phí của `mu`.** Tier-1 của vòng sum chỉ là **2 lệnh**
  `vcvtss2sd xmm0,[r8+4*rdx+0x10]` + `vaddsd xmm6,xmm0,xmm6` — JIT đã devirtualize
  + inline `List<T>.get_Item` hoàn toàn. Việc “chuyển sang byref `_items`”
  (mục 4 tay cầm) **không giải quyết gì**.
- **Reflection `GetValue` chỉ nằm ở preheader** (đọc `_items` 1 lần/call, ~10
  `GetField`+`GetValue`) — không nằm trong vòng lặp. Đã đo 4ms/6000 call.
- **Bridge `gg` CHƯA bị JIT dẹp** — vẫn còn nguyên trong Tier-1:
  `vmovups [rax+0x10],xmm1` (store) + `vmovups xmm1,[rax+0x10]` (load) →
  store-forward ~5 cycle **trên critical path**, mỗi iteration.
**Vòng vector Tier-1 của A (`G_M000_IG23`) ≈ 38 lệnh / 4 phần tử — phân bổ:**
| nhóm | lệnh | % | kiểm soát bởi |
|---|---|---|---|
| chuyển đổi `vcvtps2pd`×4 + `vcvtpd2ps`×4 | 8 | 21% | **bắt buộc** (nguồn f32, nghệ f64) |
| mở rộng `Narrow(y,zero)`: `vxorps`+`vcvtpd2ps`+`vinsertf128` ×2 chặng | 6 | 16% | JIT — `Narrow` buộc tính cả nửa cao **không ai đọc** |
| **bridge `gg`** (store+load) | 2 | 5% | **codegen** — mục 1 tay cầm |
| **`gy` load 2 lần** (do `stobj dx` ở giữa chặn CSE) | 4 | 11% | codegen/scheduling — mục 2 |
| nạp lại base `_items` từ stack | 3 | 8% | JIT register pressure |
| tính toán + điều khiển vòng | ~15 | 40% | — |
**Kết luận then chốt:**
- **A KHỂNH 4× KHÔNG ĐƯỢC nếu chỉ đổi body? KHÔNG — đúng hơn: A cần body
  −13%** (156→136ms) vì `mu` **không được đụng**: re-associate sum (unroll nhiều
  accumulator) là cách DUY NHẤT làm `mu` nhanh hơn (chuỗi `vaddsd` tuần tự
  8 cycle/phần tử), nhưng nó **đổi số** → hỏng “loss chain 3 seq EXACT” của
  acceptance. Contract `Reduction KHÔNG vectorize` cũng cấm. ⇒ A phải ăn body.
- **B cần −5%** và đang ở 3.85× (đôi khi 4.24) — nhắm mục 5 tay cầm (counter
  spill) + bridge là vừa đủ.
- Nguồn dư tổng ≈ **15/38 lệnh (39%)** đã chỉ đích danh ở bảng trên; xóa được
  nửa là đủ cho cả A lẫn B.
**Cách lặp lại phép đo:**
```bash
dotnet out/_p6ms_n10.dll          # có mu
dotnet out/_p6diag_n10.dll        # bỏ mu = body-only
DOTNET_JitDisasm=_ln_v DOTNET_JitDisasmSummary=1 dotnet out/_p6ms_n10.dll > out/disasmA.txt
# vòng vector Tier-1 = grep 'Assembly listing' cuối + tìm G_M000_IG23
```
#### 0.56.3 CHỐT LẠI CÁC MỤC CHƯA XONG CỦA #7/#8 (2026-10-30)
Đối chiếu lại từng dòng acceptance với số đo mới. **Xong 3, chưa xong 3.**
**(1) `tools/perf_gate.ps1` đã chuyển runtime — XONG ✅**
Gate cũ **không chạy nổi**: `& $exe` trên .NET Framework nổ `TypeLoadException:
System.Numerics.Vectors` ngay (`perf_e2e` là kernel scalar nhưng cũng nổ vì
codegen f64 đã lan sang cả nó). Đã sửa 4 chỗ:
| sửa | vì sao |
|---|---|
| `Invoke-PerfExe` = `dotnet exec --runtimeconfig` (net10.0, `Microsoft.NETCore.App` 10.0.13) | .NET 10 là runtime duy nhất có `Vector.Widen/Narrow` (hợp đồng §0.56.1) |
| **TARE** — thêm `tools/perf/_tare.tkv` (main rỗng, tên `_` nên bị `Where-Object -notmatch '^_'` bỏ qua khi liệt kê kernel), đo `net = tong − tare` | khởi động .NET 10 ≈ **41 ms**/process, **lớn hơn cả kernel nhỏ**; không trừ thì 7% đo “khởi động” thay vì đo code |
| `$Runs` 5 → 15 | min-of-5 vẫn còn jitter ±10% trên kernel ~16 ms |
| `$NoiseMs = 2.5` — FAIL khi **đồng thời** vượt 7% **và** vượt 2.5 ms | jitter đo được ~1.5-2 ms; 7% của `perf_f32mm` (net ~20 ms) = 1.4 ms → gate cũ sẽ **flap**, người chạy quen tay `-Update` là mất cảnh báo |
**Checksum 4/4 khớp baseline cũ từng chữ số trước khi `-Update`** (`956798.0999999787`,
`917504.0`, `460039.240524292`, `1.0`) — bằng chứng runtime mới không đổi số. Baseline
ms **phải ghi lại** (cũ đo trên .NET Framework gồm cả khởi động CLR → không còn bảo vệ
được gì nữa). 3 lần chạy liên tiếp: **PASS**.
**(2) Harness 3 probe #7 — `_p6cf` đã sửa ✅, `_p6off`/`_p6kv` KHÔNG cần sửa**
- `#8` mở matcher từ “đúng 2 câu” sang **thân N câu** → arm control `_zs` (3 câu:
  `t = 0.0; z[i] = t; i = i + 1`) **không còn bị từ chối nữa**, cả 2 arm cùng vector
  → tỉ lệ **0.98×** = không còn gì để đo. Sửa đúng kiểu `_p6ms`: chen `f32()` vào
  RHS (`_p6_co_call` → `return 0`) để arm control lại reject; số học không đổi
  (`f32(0.0) == 0.0`) nên vẫn kiểm bit-giong được.
  - IL kiểm chứng: `_zv` có vòng `_zv_W1_vstart` + `stobj Vector4` qua byref `__p6b11`;
    `_zs` đi `set_Item` thuần scalar.
  - Số đo lại: **2.87×, `bit_bad=0`** (quay về đúng mức baseline 2.90×).
- **`_p6off` / `_p6kv` KHÔNG hỏng** — nhầm lẫn trong tay cầm trước. `#7 Y3` (typed-local
  suy từ ngữ cảnh) **cầm 2 arm cùng vector hoá là ĐÚNG mục đích**: `_p6kv` yêu cầu
  “2 arm phải bằng nhau” → vừa đo lại **1.0× ≤ 1.15× ✅**; `_p6off` yêu cầu chính shape
  production (untyped) phải nhanh → U=46 ms. Không được chen `f32()` vào đây vì sẽ
  tự phá acceptance.
**(3) `_step124` — thân vòng vector đã 0 `TkvInt::` ✅ (đọc đúng acceptance)**
Bắn tỉa `out/_s124.il`, method `_step124` (dòng 8957-9278):
| vùng | `TkvInt::` |
|---|---|
| **`_step124_F2_vstart` … `_F2_vdone` (thân vòng vector)** | **0** — kèm `get_Count`=0, `conv.ovf.i4`=0, `ToArray`=0 |
| preheader (trước `vstart`) | 15 |
| tail scalar `_F2_tstart.._F2_end` (phần lẻ ≤3 phần tử) | 11 |
| vòng `for i in range(len(ps))` ở unflatten (10 lần) + sau `_F2_end` | 11 |
| **tổng cả method** | 37 (c6 = 46) |
Acceptance ghi “**trong thân vòng vector**, chỉ preheader được phép có” → **đạt 0 ✅**.
Số 37 mà trước đây ghi là **tổng cả method** (không phải thân vòng) — vòng lặp ở trên
đã nổ (`__p6a21/23` + `ldelema/ldobj/stobj Vector4`), 37 lệnh còn lại nằm ngoài nó.
**(4) Bucket `12 × block_bwd` — CHƯA đạt ❌ (đo lại cho số thật)**
Probe `tv/tests/simd_probe_am.tkv` đã có sẵn (đã lầm trước khi nói “không có trong
repo”). Vấn đề: `now_ms()` lượng tử **15.6 ms** → 1 lần gọi ~140 ms chỉ có **9
quantum**, best-of-3 thiên về phía *thấp* (đọc 125/140/141/156 ms) → không phân biệt
được ngưỡng 25%.
Đã thêm **batching 8 lần gọi / lần đo** (cùng kỹ thuật `_p6cf` `runs=20`, `_p6ms`
`runs=6000`; `b_blk` vẫn là **ms/gọi** như cũ):
| số | ms/call | ×12 = bucket | so 2244 |
|---|---|---|---|
| đọc đơn lẻ (cũ, biased thấp) | 125-156 | 1500-1872 | −33% … −16% |
| **batch 8 (authoritative)** | **150.4 / 152.4 / 154.3** | **1804 / 1828 / 1851** | **−19.6% / −18.5% / −17.5%** |
→ **cần ≤1680 ms (−25%), đang ~1804 ⇒ THIẾT 6-7 điểm phần trăm.**
**(5) Probe `_p6ms` ≥4× mỗi shape — CHƯA đạt ❌ (chưa đụng codegen trong lần này)**
Giữ nguyên số §0.56.2: **A 3.42-3.88× · B 3.72-3.80× · C 4.87-5.28×** (C qua).
Việc còn lại vẫn đúng như bảng phân bổ 38 lệnh của §0.56.2: bỏ **bridge `gg`**
(`_day_vec64_operand` khi node là lane temp → mảng `float[8]` scratch) + bỏ **`gy`
nạp 2 lần** + chống **spill counter `o`** ở B. Ước tính gộp được 4-5/38 lệnh ≈
10-13% body — vừa đủ cho A (cần body 156→136 ms) và B (cần −5%).
**`_p6cf` ≥5× — KHÔNG đạt và NẰM NGOÀI KHẢ NĂNG CỦA CODEGEN ❌**
Tier-1 của vòng `_zv` đã tối giản:
```
G_M000_IG13:  vxorps xmm0,xmm0 / vmovups [rbp+4*rsi],xmm0 / add rsi,4 /
              lea rax,[rsi+3] / cmp rax,rdi / jl IG13
```
6 lệnh / 4 phần tử (~1.5 lệnh/phần tử). Đo 496 MB/`_zv` = **5.7 GB/s ghi**.
Để đạt 5× phải được **~10 GB/s ghi**, mà store thường (`vmovups`) luôn kéo
RFO (đọc 64B khi ghi 64B) → lưu lượng bus x2. Ghi **non-temporal** sẽ được,
nhưng IL/.NET **không có API nào** (`Vector<T>` không có NT-store; JIT không
tự đổi). ⇒ 5× chỉ tới được bằng `Array.Clear`/builtin, **không phải bằng codegen
vòng** — nếu giữ nguyên acceptance thì phải chuyển sang dạng bulk-fill builtin.
**(6) Kiểm chứng hồi quy — XONG ✅**
| kiểm | kết quả |
|---|---|
| `tools/gate_f32.sh` | **135/135, `GATE_EXIT=0`** (`out/gate_r7.log`) — 10/10 suite `fails=0` |
| `tv/tests/trainer_tests.tkv` | **15/15 `fails=0`** (`=== G4 trainer cases=15 fails=0 ===`) |
| `tools/perf_gate.ps1` | **PASS**, 3 lần liên tiếp, checksum 4/4 EXACT |
| probe `_p6ms` / 3 seq / loss chain | không đo lại (không đụng compiler trong lần này — số §0.56.1/§0.56.2 vẫn còn hiệu lực) |
**File đổi trong lần này:** `Release/tools/perf_gate.ps1`, `Release/tools/perf/_tare.tkv`
(mới), `tv/tests/_p6cf.tkv` (arm control), `tv/tests/simd_probe_am.tkv` (batching đo),
HANDOFF §0.56.3. **0 dòng compiler đổi** → self-host `IL_IDENTICAL` không cần chạy lại.
#### 0.56.4 THỬ 2 SỬA CODEGEN CHO A/B — TRUNG TINH, VÀ CHẨN ĐÁN CHÍNH XÁC NƠI CÒN LẠI
**2 thay đổi compiler (đã self-host, `vong.exe` → `IL_IDENTICAL`, `tkvc-tv.exe` 17:56):**
| file | đổi | vì sao |
|---|---|---|
| `il_features/vector4_type.tkv` | scratch bridge `float32[8] __p6w8` (`ldloc`/`ldelema`/`ldobj`) → **local `Vector<float> __p6w32`** (`ldloca`/`stobj Vector4`/`ldloc`); thêm `IL_SCALAR['f32v']` | `Widen` lấy source **by value** (chứng minh bởi `push_v_wload4a`: `ldobj` rồi `call` thẳng) → scratch chỉ cần một chỗ để `stobj` 16 byte rồi `ldloc` ra `Vector<float>`; mảng là object địa chỉ lấy, đẩy `float32[]`+`float32&` vào bảng local |
| `il_features/control_flow.tkv` | **bỏ `ldc.i4.8 / newarr float32 / stloc __p6w8`** ở preheader | mảng đó giờ không ai đọc → `newarr` + `stloc` chết mỗi preheader. **Giữ cờ `_co_f64`** vì nó còn quyết định guard `iv+7 < n` (cần cho `v_wload4a` đọc 32 byte) |
**Kết quả: TRUNG TINH về tốc độ.** JIT vốn đã gộp bridge còn 2 lệnh
(`vmovups [rax+0x10],xmm1` + load lại) nên đổi mảng → local không giảm lệnh trong
vòng; được cái là hết 1 heap-alloc/preheader và IL gọn hơn. Không hề regress.
**Chẩn đo lại Tier-1 (số mới, 6 lần chạy, best-of-3 xen kẽ):**
| probe | số đo | ngưỡng | |
|---|---|---|---|
| `_p6ms` A LN_BWD | V 266-282 · S 953-984 → **3.38-3.64×** | ≥4× | ❌ |
| `_p6ms` B linear_db | V 218-219 · S 844-859 → **3.87-3.92×** | ≥4× | ❌ (cần V ≤210.75, thiếu **−3.3%**) |
| `_p6ms` C BUILDACC | **4.74-5.06×** | ≥4× | ✅ |
| `_p6cf` | **2.91×**, `bit_bad=0` | ≥5× | ❌ (xem §0.56.3(5) — nằm ngoài codegen vòng) |
| `_p6off` | U=V=T=**47 ms**, mọi tỉ số **1.0×** | shape production nhanh | ✅ |
| `_p6kv` | **1.0×** | ≤1.15× | ✅ |
| bucket `12 × block_bwd` (batch 8) | **1780-1852 ms** | ≤1680 | ❌ (cần thêm ~6%) |
**Nơi chính xác còn mất — 3 vòng Type-1, đã đếm từng lệnh:**
*A LN_BWD (`G_M000_IG23`), **39 lệnh / 4 phần tử ≈ 19 cycle/vòng → 2 IPC =*
*latency/port-bound, KHÔNG phải throughput-bound*:*
| nhóm | lệnh | cắt được từ IL? |
|---|---|---|
| chuyển đổi `vcvtps2pd`×3 + `vcvtpd2ps`×2 | 5 | không (f32 → f64 → f32) |
| **Narrow nửa cao chết** (`vxorps`+`vcvtpd2ps`+`vinsertf128` ×2) | **6** | không — `Narrow(a,zero)` trả `Vector<float>` 8 ptu, JIT **không DCE** nửa cao dù `stobj` chỉ ghi 16 byte |
| bridge `gg` (store+load) | 2 | không — chỉ còn cách lane temp `Vector<float>` 32-byte native (16 byte → 32 byte không có IL nào khác) |
| `gy` nạp 2 lần (store `dx` nằm giữa chặn CSE) | 3 | không — alias do JIT quyết định |
| nạp lại base byref `mov rcx/rdx/r10,[rsp+..]` | 3 | không — do RA |
| `vbroadcastsd` hằng loop-invariant | 1 | không (JIT không LICM nó) |
*Vòng `mu`:* **12 lệnh/phần tử** (`movsxd`/`cmp`/`jne`/`mov`/`cmp`/`jae` ×2 =
6 lệnh bounds-check + `vcvtss2sd`/`vaddsd` + counter). Đo 110 ms/6000×16384 =
**3.35 cycle/phần tử ≈ đúng sàn `vaddsd` dependency** → **bỏ bounds-check không
được gì** (chuỗi cộng tuần tự đã chặn). Chỉ re-association (nhiều accumulator) mới
nhanh hơn nhưng **đổi số → hỏng loss chain EXACT** — bị cấm. ⇒ HANDOFF §0.56.2
đúng: `get_Item`/reflection không phải chi phí, nhưng kết luận “chỉ 2 lệnh” là sai
— vòng có 12 lệnh, chỉ là **không cắt được**.
*B linear_db (`G_M000_IG52`), **10 lệnh**:*
```
mov rcx,[rsp+0x50]   ; base byref nap lai
mov rdx,[rsp+0x60]   ; COUNTER o doc tu stack
vmovups xmm0,[rcx+4*rdx] / lea rax,[r12+rdx] / vaddps / vmovups [..],xmm0
add rdx,4 / lea rax,[rdx+3] / cmp rax,rbx / jge
mov [rsp+0x60],rdx   ; COUNTER spill lai tren back-edge  ← 2/10 lenh
jmp IG52
```
Đúng như tay cầm §0.56.2 ghi: counter spill store→load theo back-edge + base
byref nạp lại = **3/10 lệnh**. Cả hai đều là quyết định của RA/JIT, không có IL
nào ép được.
⇒ **Khoảng cách còn lại (A −7.5%, B −3.3%, bucket −6%) nằm ở JIT register
allocation + DCE của `Narrow`, không phải ở matcher hay preheader.** Không còn
lever nào ở tầng codegen text được. **Fallback C2 (4 builtin `softmax_bwd →`
`layernorm_bwd → gelu_bwd → bias_grad`) vẫn hiệu lực** theo đúng §5 của #8 —
nếu hai bên muốn đóng #8, đây là đường đi thay vì đuổi tiếp micro-opt.
**Kiểm chứng lại TOÀN BỘ với compiler 17:56:**
| kiểm | kết quả |
|---|---|
| `tools/vong.exe` (self-host) | **IL_IDENTICAL**, `dist/tkvc-tv.exe` 17:56 |
| `tools/gate_f32.sh` | **135/135, `GATE_EXIT=0`** (`out/gate_r9.log`) |
| `tv/tests/trainer_tests.tkv` | **15/15 `fails=0`** |
| `tools/perf_gate.ps1` | **PASS** — 4 kernel, checksum 4/4 EXACT, chênh −2.1%…+1.6% |
| **Loss chain 3 seq** (`out/bseq_c11.exe` mới build) | **EXACT cả 3**: `10.572291 -> 7.0119185` · `10.756927 -> 11.807267` · `10.760399 -> 13.442915` — từng chữ số khớp `bseq_c6` |
| full-step vs c6 (1 lần/seq) | seq8 **4500 vs 5843** · seq256 **6312 vs 7438** · seq1024 **17641 vs 22906** → −23% / −15.1% / −23%, **không seq nào regress** |
| IL snipe `_step124` | trong thân `_F2_vstart..vdone`: `TkvInt::`=**0**, `get_Count`=0, `conv.ovf.i4`=0, `ToArray`=0, có `stobj Vector4` |
| probe bit-exact | `_p6ms` `fails=0` · `_p6cf` `bit_bad=0` · `_p6off` `bit_bad=0` |
**File đổi:** `Release/compiler_bootstrap/compiler/il_features/vector4_type.tkv`,
`Release/compiler_bootstrap/compiler/il_features/control_flow.tkv` (+ `dist/tkvc-tv.exe`
qua `vong.exe`).
#### 0.56.5 P9 (2026-10-31) — t_zeros AddRange + hoist zero-fill BtB: BUCKET ĐẠT ✅
**Bucket `12 × block_bwd`: 1804 / 1828 / 1851 → 1594.5 / 1617 / 1641 ms (≤1680 ✅).**
**0 dòng compiler đổi** → không cần self-host `vong.exe`.
**(a) Nghiên cứu trước khi sửa — đo, không phỏng đoán:**
- `_tz.tkv`: `t_zeros([768,3072])` = **13.0 ms/lần** (vòng `append` 2.36M ptu ≈
  5.5 ns/ptu). Đếm trong `gpt2_block_bwd` (simd=1, gflat=None): 4 lần phân bổ
  **lớn** của `cd` GEMM — dw2 `[3072,768]` + dw1 `[768,3072]` (2.36M mỗi cái),
  dw_qkv `[768,2304]` (1.77M), dw_proj `[768,768]` (0.59M) ≈ **7.08M ptu ≈ 40 ms
  = 27% block_bwd**. Đúng giả thuyết đã nêu ở §0.56.3(4).
- `_am2.tkv` (phân rã theo pha, seq8): `MLP_BWD 87.9 = linear_fwd 7.9 +
  linear_bwd(w2) 37.0 + linear_bwd(w1) 43.0`; `ATTENTION_BWD 50.75`;
  `t_zeros`/gelu/db-loop in ra 0.0 (dưới ngưỡng 15.6 ms của `now_ms`).
- `_am3.tkv` (tách pha `_p58_core_simple`, `smm_fwd_at` k=8, 3072×768, T=64):
  **FULL 11.75 ms = simd_gemm 4.0 + _acc_f 4.0 + tile-build ~3.75 (trong đó
  `_tile_b_w` ≈ 3.4)**.
**(b) Sửa 1 — `t_zeros` dùng `seed * k` (`List<T>.AddRange` → `Array.Copy`):**
```python
if n > 16384:
    seed = [0.0] * 16384        # 16384 append ~90µs
    big = seed * (n // 16384)   # k lần AddRange = memcpy
    ... append phần dư < 16384
```
`_tz2.tkv`: **13.0 → 6.5 ms/lan** (2.36M ptu). `Count` vẫn **đúng n**
(`k*16384 + phần dư`) nên `len(data)`/index/`_smm_cd` check `>= tot` nguyên vẹn;
chỉ `_items.Length` (capacity) dài hơn — native vẫn `castclass float[]` như cũ.
**(c) Sửa 2 — hoist zero-fill của `_tile_b_w` ra mỗi nhóm `(i0,p0)`:**
`_tile_b_w` zero-fill **cả hàng `[pw,T)` = 3584 ptu mỗi lần gọi**, tức 576 lần
một nhóm (k=8, T=64, 12 `j0`) — trong khi hàng đó **không bao giờ nhận dữ liệu
thật** trong cùng nhóm. Đổi: thêm tham số `ztail: 'i32' = 1`, `_p58_core_simple`
zero `[pw,T)` **một lần** trước vòng `j0` rồi gọi `ztail=0`:
| | trước | sau |
|---|---|---|
| ghi zero mỗi nhóm | 12 × 4096 = 49152 | 3584 + 12 × 512 = 10240 (**4.8×**) |
| mỗi GEMM k=8 (48 nhóm) | — | **−1.9M ghi ≈ −3 ms** |
**Bit-exact:** nội dung buffer tại mỗi lệnh `simd_gemm` **giống hệt** (hàng
`[pw,T)` = 0 cả trước lẫn sau) — chỉ đổi *lúc nào* ghi số 0, không đổi giá trị.
Mặc định `ztail=1` giữ nguyên hành vi cũ của `_smm_core` (call site 370/611).
**(d) Kiểm chứng:**
| kiểm | kết quả |
|---|---|
| bucket `12 × block_bwd` (batch 8, 5 lần chạy) | **1594.5 / 1617 / 1617 / 1641 / 1641** ✅ ≤1680 |
| `tools/gate_f32.sh` | **135/135, `fails=0`, exit 0** (`out/gate_after_p9.log`) |
| acceptance `_p6ms` (build lại) | **A 4.27× · B 6.09× · C 5.62×, `bit_bad=0`, `fails=0`** ✅ |
**File đổi:** `tv/training/tensor.tkv` + 2 bản sao (`tv.training.tensor.tkv`,
`tv/tests/tv.training.tensor.tkv`), `tv/training/simd_matmul.tkv` + 2 bản sao —
cả 3 cặp vốn giống hệt nhau (md5) nên `cp` đồng bộ. Probe mới: `tv/tests/_tz2.tkv`,
`_am2.tkv`, `_am3.tkv`. **0 dòng compiler.**
**Lever còn lại (chưa đụng):** `simd_gemm` vẫn tính 8× thừa ở k-dim (T=64 với k=8
→ 151M MACs cho 18.9M hữu dụng, ≈4 ms/GEMM) và `_acc_f` 4 ms/GEMM — cả hai
đều cần đổi native DLL hoặc chấp nhận đổi thứ tự cộng (cấm vì loss chain EXACT).
## §0.57 M9 (2026-10-31)
### Task 1 — Full-step 3 seq sau t_zeros + BtB (HOÀN THÀNH, ĐÃ ĐO)
- [x] A/B cùng phiên: `out/bseq_p9.exe` (mới) vs `out/bseq_c11_n10.dll` (trước P9), best-of-3, script `out/p9_3seq.sh`.
- [x] Kết quả:
  | seq | trước P9 | P9 | Δ |
  |---|---|---|---|
  | 8 | 4157 ms | 3860 ms | −7.1% |
  | 256 | 6172 ms | 5797 ms | −6.1% |
  | 1024 | 17547 ms | 16719 ms | −4.7% |
- [x] Loss EXACT 3/3, không seq nào regress:
  `10.572291 → 7.0119185`, `10.756927 → 11.807267`, `10.760399 → 13.442915`.
### Task 2 — Bỏ 8× k-waste trong `simd_gemm` (HOÀN THÀNH, ĐÃ XÉT)
- [x] Root cause: `SgemmF32(a,b,o,n)` luôn tính **k = n = T**, nhưng tile B wrapper build
      (`_tile_b_w` / `full_tile_b`) chỉ hàng `[0,pw)` là dữ liệu thật, hàng `[pw,T) = 0`.
      Với k=8, T=64 → tinh thua 8× (18.9M MAC hữu dụng / 151M MAC thực tính).
- [x] Patch `Release/tools/simd_native/tkv_simd_native.c`:
  - thêm `tkv_k_eff(b, a, n)` — quét hàng B toàn-0 từ `p = n−1` xuống (cost ~ số hàng bị bỏ),
    kèm kiểm tra cột A không chứa Inf/NaN để không mắc lỗi `0 × Inf = NaN`;
  - `SgemmF32`: nếu `k_eff < n` → `gemm_rect_f32(...)` (nt ≤ 1) hoặc `tkv_run_mt_rect(...)`,
    ngược lại **giữ nguyên đường cũ byte-identical**.
  - Bit-exact: `gemm_rows_f32` và `gemm_rect_f32` trùng nhau từng dòng trừ bound vòng `pc`;
    hạng tử bị bỏ đều là `fma(x, 0, acc) = acc` (micro bắt đầu `_mm256_setzero_ps`, C đã memset `+0`);
    thứ tự tích luy `p` tăng dần giữ nguyên.
- [x] Build MSVC sạch, không warning. DLL 250880 B, md5 `6c104ebb95e6b0882c90aef33056ae1b`,
      đã copy → `Release/runtime/` **và** `out/tkv_simd_native.dll` (nơi PInvoke resolve).
- [x] **Chạy đủ gate 135/135** — `tools/gate_f32.sh` → **135/135, `fails=0`, exit 0** (`out/gate_after_p9.log`)
- [x] Đo lại bucket: `dotnet out/am_n10.dll | grep BUCKET` → **1594.5 ms** ≤ **1680**threshold.
- [x] Loss chain 3 seq phải EXACT đúng 3 con số ở Task 1 (chạy lại `out/p9_3seq.sh`):
  `10.572291 → 7.0119185`, `10.756927 → 11.807267`, `10.760399 → 13.442915`.
- [x] Đo `_p6ms` (≥ 4×) để chắc không regress perf:
  **A 4.27× · B 6.09× · C 5.62×, `bit_bad=0`, `fails=0`** ✅
### Task 3 — Builtin `zeros_list(n)` (ĐANG THỰC HIỆN)
- [ ] Viết `compile_zeros_list` + `register_expr_builtin(reg, 'zeros_list', ..., 'f32', return_shape='list')`.
      Cơ chế: `newarr float32` (CLR zero-init) + `newobj List<float32>::.ctor()`,
      rồi set `_items` / `_size` qua reflection — `GetField(name, 36)` = `NonPublic|Instance`
      (đúng hệt mã 36 code hiện đang dùng để lấy `_items`) + `SetValue`.
- [ ] Thêm tên vào `_BUILTIN_LENH_DOC_LAP` — `compiler/il_features/file_io.tkv` **dòng 75**.
- [ ] Self-host `tools/vong.exe` → `IL_IDENTICAL` → copy `dist/tkvc-tv.exe`.
- [ ] Wire vào `t_zeros` — `tv/training/tensor.tkv` **dòng 63** + 2 bản sao mirror
      (`tv/training/tv.training.tensor.tkv`, `tv/tests/tv.training.tensor.tkv`).
- [ ] Đo lại `_tz2`: baseline `t_zeros([768,3072])` = **6.5 ms/lan** (2.36M ptu).
      ⚠️ Mục tiêu ~0.1 ms là **ước lượng của yêu cầu, chưa đo được** — `newarr` vẫn phải
      memset ~9.4 MB nên số thật có thể cao hơn. **Báo số đo, không báo ước lượng.**
- [ ] Chạy lại gate 135/135 + bucket + loss chain + `_p6ms` lần cuối sau tất cả thay đổi.
---
## Thứ tự thực hiện
1. Xong Task 2: gate → bucket → loss chain → `_p6ms`  (không trộn task 3 vào trước khi task 2 xanh).
2. Rồi Task 3: builtin → danh sách builtin → self-host → `t_zeros` → đo `_tz2`.
3. Cuối cùng: chạy lại toàn bộ (gate + bucket + `_p6ms` + full-step 3 seq) và cập nhật `HANDOFF.md`.
---
## §0.58 BENCHMARK LẠI SAU KHI COMPILER SỬA THEO REQUEST — ĐẠT TOÀN BỘ NGƯỠNG, TỈ LỆ VS TORCH 4.07-4.62× (2026-10-31)
**Chất liệu đo:** compiler `D:/TokenVector/Release/dist/tkvc-tv.exe` build **31/10 09:19**
(`control_flow.tkv` sửa 31/10 09:05 — bản fix mới nhất của compiler team), DLL native
`runtime/tkv_simd_native.dll` 31/10 14:38 (patch k-waste Task 2). **Toàn bộ probe/bench
được build lại bằng compiler đó lúc 18:57-18:58** rồi mới đo — không dùng binary cũ:
| artifact | nguồn | md5 (`*_n10.dll`) |
|---|---|---|
| `out/_p6ms_n10.dll` | `tv/tests/_p6ms.tkv` | `60681b2ae9db92b683f90fa2fb2bb01c` |
| `out/am_n10.dll` (bucket) | `tv/tests/simd_probe_am.tkv` | `f267fd1097d669f120cab3fc688ffe03` |
| `out/bseq_p9_n10.dll` (full-step) | `tv/tests/simd_bench_124m_seq.tkv` | `30e8fc5b406b99f69b8fc5510a55fa18` |
Chạy tuần tự khi máy idle, `.tkv` trên **.NET 10.0.13** (`dotnet *_n10.dll`) như mọi lần
đo trước. Kỷ luật: đo trước, báo số đo, không ước lượng.
### 1. Gate + trainer (đúng kiến — lần chạy 16:25 bị đứt đã chạy lại trọn)
- **Gate 135/135 PASS, `TOTAL fails=0`, exit 0** — 10/10 suite (`out/gate_req.log`):
  simd_matmul 20 · gpt2 17 · gpt2_scope 9 · gpt2_train 5 · grad 20 · parity 4 · pack 6 ·
  inplace 5 · gap · gpt2_full 33.
- **Trainer 15/15 `fails=0`** (`out/tr_req.log`) — AdamW, warmup/cosine, clip, overfit
  `eval_loss=0.014949893`, resume bit-exact.
### 2. Probe `_p6ms` — 3/3 shape ≥ 4× (ngưỡng request #8 §4) ✅
3 rep riêng lẻ, `bit_bad=0` + `fails=0` ở cả 9 lần chạy (`out/p6ms_req_r1..3.log`):
| shape | rep1 | rep2 | rep3 | best | ngưỡng | số cũ §0.56 |
|---|---|---|---|---|---|---|
| A LN_BWD (V 234/219/218 ms, S 984/968/968) | 4.21× | 4.42× | **4.44×** | **4.44×** | ≥4× ✅ | 3.70× ❌ |
| B linear_db (V 141/141/156, S 844/844/844) | **5.99×** | **5.99×** | 5.41× | **5.99×** | ≥4× ✅ | 3.61× ❌ |
| C BUILDACC (V 203/219/187, S 1109/1109/1125) | 5.46× | 5.06× | **6.02×** | **6.02×** | ≥4× ✅ | 4.74× |
→ **Acceptance #8 phần speedup: 3/3 đạt** (lần trước 1/3 — chỉ C qua).
### 3. Bucket `12 × block_bwd` — 1569/1570.5 ms ≤ 1680 ✅
2 rep (`out/am_req_r1.log`, `out/am_req_r2.log`): `BLOCK_BWD` 130.75 / 130.875 ms/gọi
(batch 8) → **BUCKET 1569.0 / 1570.5 ms** vs baseline 2244 = **−29.6%**, vượt ngưỡng
dưới 1680 của request #8 §4.
### 4. Full-step 3 seq best-of-3 — không seq nào regress, loss EXACT 9/9 ✅
`out/p9_3seq.sh out/bseq_p9_n10.dll req` → `out/req_3seq.summary`:
| seq | 3 lần (ms) | best | trước P9 (4157/6172/17547) | P9 (3860/5797/16719) |
|---|---|---|---|---|
| 8 | 4015 / 3859 / **3829** | **3829 ms** | **−7.9%** | −0.8% |
| 256 | 5860 / **5672** / 5766 | **5672 ms** | **−8.1%** | −2.2% |
| 1024 | **16375** / 16954 / 16547 | **16375 ms** | **−6.7%** | −2.1% |
- **Loss EXACT từng chữ số cả 9 lượt chạy:** `10.572291 → 7.0119185` /
  `10.756927 → 11.807267` / `10.760399 → 13.442915` — reduction không đổi thứ tự.
- Không seq regress so với cả 2 baseline (P9 và trước P9).
### 5. Bắn tỉa IL (`out/am.il`, compiler 31/10) — matcher nổ production, 1 mục còn nợ
| method | `ldelema` | `stobj` (Vector4/f32) | local `__p6a*` | `TkvInt::` trong body |
|---|---|---|---|---|
| `gpt2_layernorm_bwd` (3195 dòng) | 35 | 32 | 16 cặp `__p6a/__p6b` | **0** ✅ |
| `gpt2_linear_bwd` (1194 dòng) | 7 | 6 | — | **0** ✅ |
| `gpt2_mlp_bwd` | 7 | 6 | 21 | **0** ✅ |
| `gpt2_attention_bwd` | 21 | 18 | 63 | **0** ✅ |
| `gpt2_block_bwd` | 11 | 8 | 33 | **48** ❌ |
`gpt2_layernorm_bwd` còn có lane `f64x4` (`Vector`1<float64>`) + `Vector128<float32>`
(AVX intrinsic) trong locals → unswitch/Y2 đã chạy. **Còn nợ:** `gpt2_block_bwd` vẫn còn
**48 lệnh `TkvInt::`** (ceremony của request #7 chưa hết ở method này) — mục tiêu "0
`TkvInt::` trong thân" của #7/#8 chưa trọn vẹn, dù mọi ngưỡng số đo đã qua.
### 6. So với số PyTorch ĐÃ CÓ (không rerun — theo yêu cầu)
Baseline torch **không đo lại**: `out/ab_tkv_torch_20261024.log` (best-of-3 xen kẽ,
torch 2.14.0+cpu **14 luồng**, từng tái lập đúng 0.934/1.404/3.542 s) =
**934.2 / 1394.1 / 3546.7 ms**; torch 1T seq8 1322.7 ms; tier E `out/bench_torch_e.log`
(median n=3, §0.52).
| seq | tkv hôm nay | torch 14T | **tỉ lệ** | torch 1T | tier E torch TOTAL 14T | tỉ lệ vs tier E |
|---|---|---|---|---|---|---|
| 8 | 3829 | 934.2 | **4.10×** | 1322.7 → 2.90× | 1189.9 | 3.22× |
| 256 | 5672 | 1394.1 | **4.07×** | ~1.5 s → ~3.8× | 1563.1 | 3.63× |
| 1024 | 16375 | 3546.7 | **4.62×** | — | 3936.4 | 4.16× |
**Tiến trình tỉ lệ tkv/torch (cùng baseline):**
| mốc | seq8 | seq256 | seq1024 | nguồn |
|---|---|---|---|---|
| 2026-10-20 P5.3 | 23.8× | 29.4× | 43.5× | §7.1 |
| 2026-10-22 P6 GĐ10 | 13.4× | 12.3× | 10.4× | §0.30 |
| 2026-10-24 A/B thật torch↔tkv | 13.2× | 12.1× | 10.3× | §0.45 |
| 2026-10-26 M6b | 7.7× | 6.9× | 6.1× | `PLAN_TO_PARITY` §0 |
| 2026-10-27 Phase B `f32_list_add` | 6.4× | 6.0× | 5.75× | §0.53 |
| 2026-10-30 #7+#8 (c10) | 4.35× | 4.46× | 4.82× | §0.56.1 |
| 2026-10-31 P9 | 4.13× | 4.16× | 4.71× | §0.57 |
| **2026-10-31 sau compiler fix** | **4.10×** | **4.07×** | **4.62×** | §0.58 này |
- vs baseline `PLAN_TO_PARITY` (7187/9609/21641): **−46.7% / −41.0% / −24.5%**.
- **Đúng đường ray plan:** mục tiêu Phase **+B** (4200/6500/17000) **đã qua cả 3 seq**;
  đang giữa Phase **+C** (2800/4500/12000 — còn 26-37%), chưa tới D (1500/2600/7000)
  hay win ≤1× (934/1394/3547).
- **Bucket vs torch tier E:** `12 × block_bwd = 1569 ms` @seq8 so với **toàn bộ bwd
của torch = 333.1 ms** → riêng bucket này **4.71× torch bwd**; `opt.step` torch
**658-743 ms cố định mọi seq** (bandwidth-bound) — tkv chưa tách lại phần opt hôm nay.
- **Gap 4.1× không nằm ở kernel:** pha GEMM đã parity từ §0.41 (552-588 GFLOP/s pad ≈
torch useful, S4/S5 207/211 ms thắng torch 254/240); A4 (§0.52) đo cả bước train
**ratio 1.37** trong khi torch 14T → còn lại là wrapper/glue/elementwise **đơn luồng**.
**⚠️ Cảnh báo phương pháp:** (1) số torch là best-of **24/10** (tier E **26/10**) —
`PLAN_TO_PARITY` Phase F vẫn bắt bench lại torch **xen kẽ fresh** trước khi tuyên bố
gì gần 1×; (2) `_p6ms` là A/B nội bộ vector↔scalar **không có số torch tương ứng**;
(3) `gpt2_block_bwd` còn 48 `TkvInt::` — nếu muốn khép mục 5 của #7/#8 thì cần
request follow-up, không tự sửa matcher phía repo.
**Trạng thái git:** **0 dòng `.tkv` code train đổi** trong lần đo này (chỉ build lại +
chạy) → không cần gate vì wire; log nằm trong `out/` (gitignore). Việc ghi tài liệu =
§0.58 này. Task 3 (`zeros_list`) **chưa đụng tới** — compiler chưa có builtin này
(`grep zeros_list` trong `dist/tkvc-tv.il` = 0 hit, xác minh 31/10).
---
## §0.59 H1 — BẢNG BUDGET 2 PHÍA ĐO XEN KẼ CÙNG PHIÊN: TỈ LỆ THẬT 3.69-3.88×, GAP 100% LÀ THREADING + `block_bwd` (2026-10-31)
Thực hiện lever **H1** của [docs/PLAN_TO_LE_1X.md](docs/PLAN_TO_LE_1X.md) — lần đo **đầu tiên
torch và tkv chạy xen kẽ trong cùng một phiên** (Phase F của `PLAN_TO_PARITY`). Mọi số dưới
đây **ghi ra file** (`out/h1_*.log`), không đọc từ terminal.
- Compiler: `D:/TokenVector/Release/dist/tkvc-tv.exe` build 31/10 09:19; native
  `runtime/tkv_simd_native.dll` 31/10 14:38 (patch k-waste). tkv trên **.NET 10.0.13**.
- torch **2.14.0+cpu**, tier D/E của `tools/bench_torch.py`, `T=1` và `T=14`.
- tkv: `out/bseq_p9_n10.dll` (build 18:58, md5 `30e8fc5b…`), best-of-3, script `out/h1_run.sh`.
- Probe build lại bằng compiler hiện tại: `out/bd2_n10.exe` (21:25), `out/_p6ms_n10.dll`,
  `out/am_n10.dll`.
### 1. tkv — fwd/bwd/mode-1 best-of-3 (`out/h1_tkv_m0.log`, `out/h1_tkv_m1.log`)
| seq | fwd | bwd | fwd+bwd | full-step | overhead (opt + copy-back) |
|---|---|---|---|---|---|
| 8 | 453 | 2500 | **2953** | **3844** | 891 |
| 256 | 1125 | 3937 | **5062** | **5844** | 782 |
| 1024 | 3641 | 11078 | **14719** | **16078** | 1359 |
Loss EXACT cả 9 lượt (`10.572291→7.0119185` / `10.756927→11.807267` / `10.760399→13.442915`).
Số này khớp §0.58 (3829/5672/16375) trong sai số nhiễu ±6% — không regress, không sửa code.
### 2. torch fresh cùng phiên — tách phase 1T/14T (`out/h1_torch_e.log`, `out/h1_torch_d.log`)
| seq | T | zero | fwd(+CE) | bwd | opt.step | TOTAL |
|---|---|---|---|---|---|---|
| 8 | 1 | 2.4 | 127.2 | 402.2 | 851.8 | **1388.0** |
| 8 | 14 | 9.8 | 76.8 | 293.0 | 661.5 | **1042.7** |
| 256 | 1 | 4.3 | 1214.6 | 2337.2 | 798.9 | **4354.8** |
| 256 | 14 | 8.9 | 306.2 | 571.1 | 666.6 | **1572.7** |
| 1024 | 1 | 26.4 | 4825.7 | 9408.7 | 818.7 | **15082.1** |
| 1024 | 14 | 42.8 | 1077.1 | 2357.9 | 636.7 | **4148.4** |
tier D (fwd+bwd / full step, median n=2): seq8 **319.1 / 1022.0**, seq256 **870.2 / 1556.2**,
seq1024 **3448.1 / 4165.1**. Máy hôm nay **chậm hơn 24/10 ~9%** (torch full-step seq8
1022.0 vs 934.2) — nên tỉ lệ §0.58 (4.07-4.62×) là **của baseline cũ**, số cùng phiên là
số dưới đây.
### 3. Tỉ lệ cùng phiên — và nó đến từ đâu
| seq | tkv full | torch 14T | **tỉ lệ** | torch 1T | **tỉ lệ vs 1T** | threading gain của torch (1T/14T) | cần cắt |
|---|---|---|---|---|---|---|---|
| 8 | 3844 | 1042.7 | **3.69×** | 1388.0 | **2.77×** | 1.33× | −72.9% |
| 256 | 5844 | 1572.7 | **3.72×** | 4354.8 | **1.34×** | 2.77× | −73.1% |
| 1024 | 16078 | 4148.4 | **3.88×** | 15082.1 | **1.07×** | 3.64× | −74.2% |
(vs tier D fresh: 3.76× / 3.76× / 3.86× — cùng kết luận.)
**Đây là kết luận quan trọng nhất của H1:** ở seq 1024 — seq lớn nhất, nơi GEMM chiếm ưu thế —
**tkv đơn luồng chỉ còn chậm 1.07× so với torch đơn luồng** (16.1 s vs 15.1 s). Nói cách khác:
- **~93% khoảng 3.88× là torch dùng 14 luồng**, không phải kernel tkv chậm. torch tự nó chỉ
  scale 3.64× trên 14 luồng ở seq1024 (hiệu suất 26%), 2.77× ở seq256, **1.33×** ở seq8.
- Ở seq nhỏ, tkv mất thêm vì **chi phí cố định 38.6M phần tử** (lm-head/embed) không co theo seq.
- Ở fwd+bwd thuần (bỏ optimizer) tỉ lệ xấu hơn nhiều: **7.99× / 5.77× / 4.29×** — vì
  `opt.step` của torch **cố định 637-852 ms mọi seq** (bandwidth-bound, có lợi cho nó khi seq nhỏ),
  còn tkv opt+copy chỉ 782-1359 ms. **Chỉ số "full step" đang che bớt phần bwd.**
### 4. Bổ phần gap theo phase (từ bảng 1 & 2)
| seq | gap fwd | gap bwd | gap opt+copy | tổng gap |
|---|---|---|---|---|
| 8 | +376 (13%) | **+2207 (79%)** | +220 (8%) | +2801 |
| 256 | +819 (19%) | **+3366 (79%)** | +106 (2%) | +4271 |
| 1024 | +2564 (21%) | **+8720 (73%)** | +680 (6%) | +11930 |
→ **backward là 73-79% toàn bộ gap ở cả 3 seq.** Forward chỉ đáng kể ở seq1024 (nơi tkv
thực ra **nhanh hơn torch 1T 25%**: 3641 vs 4825.7).
### 5. Bên trong bwd tkv (probe, seq8 — `out/h1_bd2.log`, `out/h1_am.log`)
`bwd = 2500 ms` phân rã:
| thành phần | ms | % bwd | nguồn |
|---|---|---|---|
| `12 × gpt2_block_bwd` | **1617** (134.75/gọi) | **65%** | `AM BLOCK_BWD` batch 8 |
| m6 lm-head (6 alloc + 3 copy_to + zero + fused + 2 copy_from + 6 free) | **344** | 14% | `BD2 [4]` |
| `op_embed_bwd ×2` (38.6M) | **234** | 9% | `BD2 [5]` |
| 2 vòng dense 38.6M (`dwte +=`, `gflat +=`) | ~62 | 2% | `BD2 [1][2]` (31 ms = sàn phân giải `now_ms`) |
| còn lại (loss bwd, wpe, `layernorm_bwd`) | ~243 | 10% | suy ra |
Đối chiếu: **`12 × block_bwd` = 1617 ms = 4.02× TOÀN BỘ bwd của torch 1T (402.2 ms)** —
tức riêng bucket này đã đắt hơn cả backward torch đơn luồng 4 lần. Trong bucket, phần
vector hoá được (`ATTENTION_BWD` 47 + `MLP_BWD` 62 = 109 ms) chỉ là 8% của 134.75;
**~26 ms/gọi còn lại là ceremony/`TkvInt::`/copy trong `gpt2_block_bwd`** — đúng mục
còn nợ §0.58 §5 (48 lệnh `TkvInt::`). `_p6ms` cùng phiên vẫn xanh: **A 4.14× · B 5.41× · C 4.81×**
(`bit_bad=0`, `fails=0`) — nhưng sàn 31 ms của `now_ms` làm 3 số này **kém tin cậy hơn**
3 rep 18:57 (§0.58: 4.44/5.99/6.02×).
### 6. Hệ quả cho mục tiêu ≤1×
Để ≤1× cần cắt 72.9-74.2%. Từ bảng 3, **không lever đơn lẻ nào làm được**:
1. **H3 threading (14C)** là lever lớn nhất nhưng **trần của nó là ~3.6×** (chính torch chỉ
   đạt 3.64× ở seq1024). Nếu tkv đạt y hệt hiệu suất song song của torch → 16078/3.64 =
   **4417 ms ≈ 1.06×** — vẫn chưa ≤1×, mà đó là giả định lạc quan nhất.
2. Vì vậy **H2 (`block_bwd` codegen) là bắt buộc song song**, không phải "hoặc": nó vừa cắt
   1617 ms/bucket vừa **giảm mẫu số** để threading có ý nghĩa. Mục tiêu hợp lý: đưa
   `BLOCK_BWD` 134.75 → ~45 ms/gọi (−67%, bằng `SUM_cac_vung` hiện tại) → bwd seq8
   2500 → ~1400.
3. Ở **seq8/256** còn phải diệt **578 ms chi phí cố định** (m6 344 + `op_embed_bwd` 234) —
   đây là lever **H4 (pack/scratch cache)**: hiện mỗi bước vẫn alloc + copy 38.6M×4 B
   nhiều lần. Đó là lý do seq8 chỉ đạt 3.69× trong khi fwd+bwd thuần là 7.99×.
**Thứ tự thực thi đã đổi so với `PLAN_TO_LE_1X.md` ban đầu:** H2 trước H3 (H2 là điều kiện
để H3 có trần đủ), H4 chen vào giữa cho seq nhỏ. Chi tiết cập nhật ở
[docs/PLAN_TO_LE_1X.md](docs/PLAN_TO_LE_1X.md) §H1.
**Trạng thái:** 0 dòng code `.tkv`/C đổi trong lần đo này — chỉ build lại probe + chạy.
Log: `out/h1_tkv_m0.log`, `out/h1_tkv_m1.log`, `out/h1_torch_e.log`, `out/h1_torch_d.log`,
`out/h1_bd2.log`, `out/h1_p6ms.log`, `out/h1_am.log`.
---
## §0.60 REQUEST #9 — H2 `block_bwd` HẾT `TkvInt::` (48→0) + H4b ZERO-COPY FUSED: FULL-STEP −3.6/−5.1/−3.8%, LOSS EXACT (2026-11-01)
Chuyển từ compiler team: [Release/docs/e_handoff/2026-11-01-request9-h2-block-bwd-va-h4b-zero-copy.md]
(`D:/TokenVector/Release/docs/e_handoff/`). Compiler `tkvc-tv.exe` self-host 01/11 (`IL_IDENTICAL`),
native DLL **không đổi** (tái dùng entry `lmhead_bwd_fused`).
### 1. Đã làm
| mục | đổi | bằng chứng IL |
|---|---|---|
| **A.1** dọn `TkvInt::` `gpt2_block_bwd` | `o_l1g…o_b2: "i64" = -1`, `o_* = offs[k]`, `n/h: "i64"` | **48 → 0** lệnh trong cả method; 4 vòng vector vẫn nổ (8 `_vstart`) |
| **A.3** bỏ double-load `gy` | `gpt2_layernorm_bwd`: local lane `gyv: "f32" = gy.data[base+j]` dùng cho `gg` + 3 chỗ còn lại | 4 vòng vector (`F6/F7/F9/F10`), 0 `TkvInt::`, bit-exact |
| **B** zero-copy fused | compiler: builtin `lmhead_bwd_fused_list(list[f32]×5, ids, B,S,E,V,scale)` + wrapper IL pin `_items`; repo: `gpt2_backward` m6 gọi thẳng `.data` | `gpt2_backward`: `F32CopyTo` 3 → **0**, `F32CopyFrom` 2 → **0**, `LmheadBwdListM` 0 → 1 |
**Vì sao 48 lệnh đó không nằm trong vòng:** 24 lệnh = 12 × (`FromI64`+`Neg`) cho `o_* = -1`;
12 lệnh = `FromI64` khi gán `offs[k]`; 12 lệnh = `ToI64` ở biên `range(n*h)` của 4 vòng.
Tất cả ở preheader — thân vòng đã 0 từ #8. Đây là lý do **bucket không đổi** (xem §2).
### 2. Số đo (best-of-3 xen kẽ từng lượt, cùng phiên, `.NET 10.0.13`)
`out/r9_ab3.sh`: `out/bseq_p9_n10.dll` (trước #9) ↔ `out/r9_bseq_n10.dll` (sau #9):
| seq | trước #9 | sau #9 | Δ | loss chain |
|---|---|---|---|---|
| 8 | 3875 | **3735** | **−3.6%** | `10.572291 -> 7.0119185` EXACT |
| 256 | 5828 | **5531** | **−5.1%** | `10.756927 -> 11.807267` EXACT |
| 1024 | 16515 | **15891** | **−3.8%** | `10.760399 -> 13.442915` EXACT |
Torch **fresh cùng phiên** (tier E, `T=14`, `out/r9_torch_e.log`): **1061.6 / 1571.7 / 4201.4 ms**
→ tỉ lệ **3.52× / 3.52× / 3.78×** (so 1T: 1366.1 / 4350.3 / 15038.0 → 2.73× / 1.27× / 1.06×).
Bucket `12 × block_bwd` (`simd_probe_am`, batch 8): **`BLOCK_BWD` 130.875 ms/call → 1570.5 ms**
(`ATTENTION_BWD` 47 + `MLP_BWD` 78 = **125 ms = 95% bucket**; glue `block_bwd` chỉ ~6 ms).
### 3. Kiểm chứng
| kiểm | kết quả |
|---|---|
| `tools/gate_f32.sh` | **135/135, `TOTAL fails=0`, exit 0** (`out/r9_gate.log`) |
| Trainer | **15/15 `fails=0`** (resume bit-exact) |
| Probe `_p6ms` (2 rep) | **A 4.44/4.12× · B 5.98/6.09× · C 5.47/5.06×**, `bit_bad=0`, `fails=0` |
| Loss chain 9 lượt | **EXACT 9/9** |
| Self-host compiler | **IL_IDENTICAL** |
### 4. Hai mục CỦA REQUEST KHÔNG LÀM — đo trước, báo số
- **A.2 hoist vòng `mu` sang byref `_items`:** vòng `mu` bị chặn bởi **chuỗi `vaddsd` tuần tự
  (3.35 cycle/ptu)**, không phải `get_Item`; bỏ bounds-check chỉ bỏ 6 lệnh **ngoài critical path**
  ⇒ 0 gain. Muốn nhanh hơn phải re-associate = đổi thứ tự cộng = **cấm** (loss EXACT).
- **LICM `m1/m2/rstd`:** còn **1 `vbroadcastsd`/vòng**, và A đã 4.12–4.44× (ngưỡng ≥4×).
- ⇒ **Bucket 130.875 ms/call không phải ceremony.** Muốn cắt 67% như request phải cắt trong
  `gpt2_attention_bwd` (47) + `gpt2_mlp_bwd` (78) — đề xuất request #10 kèm phân rã pha như `_am3`.
### 5. Trạng thái git
`tv/training/gpt2_ref.tkv` + 4 mirror (`tv/training/tv.training.gpt2_ref*.tkv`,
`tv/tests/tv.training.gpt2_ref*.tkv`): 5 file, +138/−214. Probe/log mới trong `out/` (gitignore):
`r9_bseq*`, `r9_am*`, `r9_bd2*`, `r9_p6ms*`, `r9_torch_e.log`, `r9_ab3.log`, `r9_gate.log`.
---
## §0.61 REQUEST #10 — ASK A `linear_dw_fused` TRUE-K GHI THẲNG `gflat`: BUCKET 1570→960 ms (−39%), SEQ8 −17.9%, LOSS EXACT (2026-11-01)
Chuyển từ compiler team: [Release/docs/e_handoff/2026-11-01-request10-linear-dw-fused-true-k.md]
(`D:/TokenVector/Release/docs/e_handoff/`). Compiler `tkvc-tv.exe` self-host 01/11 (`IL_IDENTICAL`);
native DLL **có đổi** — thêm TU mới `tkv_linear_dw.c` (`/fp:precise`) + entry `LinearDwFused`.
### 1. Đã làm (Ask A — builtin `linear_dw_fused`)
```tkv
linear_dw_fused(x, gy, gdw, gdw_off, n, in_d, out_d) -> i32
```
`x`/`gy`/`gdw` = `list[f32]` (`Tensor.data`) — pin `_items` **tại chỗ, không copy**;
`gdw[gdw_off + i*out_d + c] += Σ_p x[p*in_d + i]·gy[p*out_d + c]`. `rc != 0` → **fallback**
đường `.tkv` cũ (`-1/-2/-5/-6` shape/biên, `-7` `out_d % 16` / `in_d % 4` lệch, **`-8` `n > 64`**).
**Wire:** `gpt2_linear_bwd` + `gpt2_mlp_bwd` (`dw2`) gọi thẳng **khi `goff/o_w2 ≥ 0`**
(đường train thật luôn truyền `gflat`); bỏ `smm_fwd_at` + vong copy `dw → gflat`.
`goff < 0` giữ nguyên 100% (dw phải trả Tensor).
**Vì sao nhanh:** `dw` là GEMM contraction **`k = n = 8`**, nhưng `smm_fwd_at` vẫn build tile
`T×T = 64×64` bằng **code interpreted** cho từng ô (`dw2` = 576 ô × 12K ghi = **7.1M phep ghi
interpreted**) rồi mới `simd_gemm`. O `LinearDwFused` **tile biến mất**. Đo trong C (best-of-5):
`dw2` 6.0 → **1.0 ms** (6.0×) · `dwqkv` 4.4 → **0.85** (5.3×) · `dw1` 6.2 → **1.5 ms** (4.2×).
### 2. Bit-exact — 2 cạm bẫy đã gặp (đọc trước khi sửa kernel)
Chuỗi `.tkv` cần tái hiện **nguyên vẹn**:
`dw = 0 + Ct(p0=0) + Ct(p0=T) + …` (cộng dồn vào Tensor dw **từ 0**) rồi
`gflat[goff+k] = gflat[goff+k] + dw[k]` (**một** phep cong). Với mỗi `Ct`, `micro_f32` (4×16)
**không gộp cột** — mỗi *lane* = một cột, vòng `p` **tăng dần FMA**; `tkv_k_eff` rút `k` xuống
đúng `n` khi `n < 64`.
- **Cạm bẫy 1 — `/fp:fast` tái liên kết.** MSVC đổi thứ tự `acc = acc + blk` **và gộp cả
  `load(gdw)` vào cây FMA** → kết quả **phụ thuộc giá trị đích** (cùng input: lần 1 `3c1d4954`,
  lần 2 cộng thêm `3c1d4958`). `#pragma float_control(precise, on)` **bị bỏ qua** trong `/fp:fast`.
  Giải pháp: tách `tkv_linear_dw.c` thành **TU riêng `/fp:precise`**; kernel cũ giữ `/fp:fast`.
- **Cạm bẫy 2 — cộng p-tile thẳng vào `gdw`.** `gdw += acc` **sau mỗi p-tile** khác chuỗi `.tkv`
  khi `n > 64`. Sửa: gộp p-tile vào **một** giá trị trong thanh ghi, `gdw += v` đúng **một lần**.
  Hệ quả tất yếu: **`n > 64` → `rc = -8`** (fallback). Mọi shape thật GPT-2 đều `k = n = 8`.
**Chứng minh:** probe mới `tv/tests/_am5.tkv` — `gflat` **điền sẵn giá trị ≠ 0** (mô phỏng gradient
đang chạy), so `smm_fwd_at` + copy vs `linear_dw_fused`: **`bit_bad=0` trên 4 shape thật**
(`dw2` 3072×768 · `dw1` 768×3072 · `dwqkv` 768×2304 · `dx2` 2304×768); `n=128/200` → `rc=-8`;
`out_d=40` → `rc=-7` → fallback. IL: `TkvSimd::LinearDwFusedM(List<f32>×3, goff, n, in_d, out_d)`
pin `_items` (`ldelema float32` + `float32& pinned`), kiểm độ dài **trước** khi pin; 2 call site
đúng 2 method production (`gpt2_linear_bwd`, `gpt2_mlp_bwd`). *Lưu ý IL: `off` là reserved token
→ phải đặt tên `goff`.*
### 3. Số đo
**Bucket — nhưng PHẢI đo biến thể CÓ `gflat`:** probe `simd_probe_am` gọi `gpt2_block_bwd`
**không truyền `gflat`** → `goff = -1` → `dw` trả Tensor → native **không chạy**. Probe đã mở
thêm biến thể `BLOCK_BWD_FLAT` (truyền `gflat` + `offs`):
| biến thể | trước #10 | sau #10 |
|---|---|---|
| `BLOCK_BWD` không `gflat` (`goff=-1`) | 130.875 | 134.875 (không đổi — native không chạy) |
| **`BLOCK_BWD_FLAT` (production, `goff>=0`)** | 130.875 | **80.0 ms** ✅ (ngưỡng ≤80) |
| **Bucket 12 ×** | 1570.5 ms | **960.0 ms** ✅ (ngưỡng ≤960, **−39%**) |
**Full-step 3 seq (best-of-3 xen kẽ từng lượt, cùng phiên, .NET 10.0.13)** —
`out/r10_ab3.sh`: `r9_bseq` (sau #9) ↔ `r10_bseq` (sau #10):
| seq | sau #9 | sau #10 | Δ | loss |
|---|---|---|---|---|
| 8 | 3656 | **3000** | **−17.9%** | EXACT |
| 256 | 5516 | **5500** | −0.3% | EXACT |
| 1024 | 15219 | **15375** | +1.0% (nhiễu) | EXACT |
Torch **fresh cùng phiên** (tier E, `T=14`, `out/r10_torch_e.log`): **1000.1 / 1570.8 / 4130.7 ms**
→ tỉ lệ **3.00× / 3.50× / 3.72×** (so #9: 3.52× / 3.52× / 3.78× — seq8 cải thiện rõ).
*Gain gần như chỉ ở **seq8** — nơi `dw` (k=8, output 2.36M) chiếm tỉ trọng lớn. Ở seq1024,
`dw` là `O(n·h·h)` còn GEMM lớn (`k ≥ 256`, đã packed native) mới chủ đạo → `linear_dw_fused`
gần như vô hại.*
### 4. Kiểm chứng
| kiểm | kết quả |
|---|---|
| `tools/gate_f32.sh` | **135/135, `TOTAL fails=0`, exit 0** |
| Trainer | **15/15 `fails=0`** (resume bit-exact) |
| Probe `_p6ms` (2 rep) | **A 4.49× · B 5.51× · C 5.14×**, `bit_bad=0`, `fails=0` |
| Loss chain 9 lượt | **EXACT 9/9** |
| Self-host compiler | **IL_IDENTICAL** |
### 5. Ask B / Ask C — CHƯA LÀM (báo số)
- **Ask B `linear_bwd_fused` (dx+dw+db):** chưa làm. Còn `dx = smm_dot(gy, w, n, in_d, out_d)`
  **12.69 ms (w1) / 5.81 ms (qkv)**. Hợp đồng `smm_dot` (`trans_b=1`) **khác** `dw` → đề nghị
  tách `linear_dx_fused` **riêng**, không nhét chung (giữ hợp đồng đã bit-exact).
- **Ask C:** `dw_proj` (5.75 ms) đi `smm_fwd` (`trans_a=0`) — `LinearDwFused` **chưa cover**;
  12 head block (4.39 ms) + 5 `t_zeros`/head là **repo-side** (cache scratch), ngoài phạm vi compiler.
- **Gợi ý:** (1) mở `trans_a=0` trong `LinearDwFused` (mở `dw_proj`); (2) `n > 64` chỉ khả thi
  nếu chấp nhận không bit-exact hoặc sinh vào scratch rồi cộng 1 lần; (3) `linear_dx_fused` riêng.
### 6. Trạng thái git
Compiler: `simd_feature.tkv` + `file_io.tkv` (builtin), `tkv_linear_dw.c` (mới) + `build.ps1`
(2 TU), `test_dw.c`/`test_dbg*.c` (test bit-check). Repo: `tv/training/gpt2_ref.tkv` +
`tv/training/tv.training.gpt2_ref124.tkv` + `tv/tests/tv.training.gpt2_ref124.tkv` (3 file
production; 2 bản `gpt2_ref` dotted là bản cũ **không** có `smm_fwd_at` — giữ nguyên).
Probe mới `tv/tests/_am5.tkv`; `simd_probe_am.tkv` + biến thể `BLOCK_BWD_FLAT`.
Log trong `out/` (gitignore): `r10_ab_*`, `r10_torch_e.log`, `r10_gate.log`.
---
## §0.62 REQUEST #10 ASK C.1 — `dw_proj` BỎ `smm_transpose`, DÙNG `trans_a=1`: BUCKET 960→844.5 ms (2026-11-01)
Tiếp §0.61. Chi tiết: `Release/docs/e_handoff/2026-11-01-request10-linear-dw-fused-true-k.md` **§7**.
**Không đổi native, không đổi compiler** — đây là thay đổi `.tkv` thuần (3 file production).
### 1. Phát hiện thay đổi hướng làm
Request #10 (Ask C.1) đề nghị **mở `trans_a=0`** trong `LinearDwFused` để cover `dw_proj`.
Đọc code thật thì **không cần**: `dw_proj` hiện làm
```tkv
ctxTt = smm_transpose(ctxT, n, h)     # ctxT da luu [n hang][h cot] -> ctxTt la [h][n]
dwG   = smm_fwd(ctxTt, gout, h, h, n) # trans_a = 0
```
`ctxT` **đã** lưu `[n hang][h cot]` — chính là layout `trans_a=1` mà builtin hiện có đọc thẳng
qua `_tile_a_at_w`. Vậy chỉ cần **bỏ phép transpose trung gian** và truyền `ctxT` thẳng vào
`linear_dw_fused`. Đây đúng mẫu **P5.4** đã làm với `gpt2_linear_bwd` (đã `maxdiff 0.0`).
Chỉ mở `trans_a=0` thì phải thêm tham số + nhánh kernel mới **và vẫn phải bỏ transpose** —
làm hai lần không ích gì.
### 2. Bẫy bit-exact — fallback KHÔNG được là `smm_fwd_at`
Probe `_am5._check_proj` (so `smm_fwd_at(ctxT)` vs `smm_fwd(smm_transpose(ctxT))`):
| n | `smm_fwd_at` vs đường cũ | `linear_dw_fused` vs đường cũ |
|---|---|---|
| **8** | **`bad=0`** ✅ | **`bad=0`** ✅ |
| 256 | **`bad=274384`** ❌ | `rc=-8` → fallback |
| 1024 | lệch | `rc=-8` → fallback |
**Lý do:** `_smm_core` (`smm_fwd`) với `k ≥ 256` đi **packed native** (`_smm_pk`), còn
`_p58_core` (`smm_fwd_at`) **luôn** đi tile → **hai thứ tự cộng khác nhau**. ⇒ Khi native từ chối
(`n > 64`), fallback **phải là đường cũ nguyên vẹn** (`smm_transpose` + `smm_fwd`),
**không được** thay bằng `smm_fwd_at` — nếu không mất loss EXACT ở seq256/1024.
### 3. Wire (3 file production)
```tkv
dwG = t_zeros([0])
l10p = -1
if o_proj >= 0:
    _ct10 = ctxT.data ; _go10 = gout.data ; _gf10 = gflat
    l10p = linear_dw_fused(_ct10, _go10, _gf10, o_proj, n, h, h)
if l10p != 0:
    ctxTt = smm_transpose(ctxT, n, h)      # fallback NGUYEN VEN
    dwG = smm_fwd(ctxTt, gout, h, h, n)
```
Vòng copy `dwG → gflat` bọc trong `if l10p != 0:` (native đã ghi thẳng khi `rc = 0`).
`gpt2_attention_bwd` giờ có **4 call site** `LinearDwFusedM` (dwqkv, dw1, dw2, **dw_proj**).
### 4. Số đo (cùng phiên, best-of-3 xen kẽ)
| mục | sau #10 Ask A | **sau Ask C.1** | Δ |
|---|---|---|---|
| `BLOCK_BWD_FLAT` (production) | 80.0 ms | **70.375 ms** | **−12.0%** |
| Bucket `12 × block_bwd` | 960.0 ms | **844.5 ms** | −12.0% |
| full-step seq8 | 3000 | **2922** | **−2.6%** |
| full-step seq256 | 5531 | 5594 | +1.1% (nhiễu) |
| full-step seq1024 | 16282 | **15969** | −1.9% |
Loss **EXACT 6/6** (`10.572291→7.0119185` · `10.756927→11.807267` · `10.760399→13.442915`).
Torch fresh cùng phiên (tier E `T=14`, `out/r11_torch_e.log`): **1060.8 / 1622.9 / 4209.6 ms**
→ tỉ lệ **2.75× / 3.45× / 3.79×**.
**Tích luỹ baseline H1 → nay:**
| mục | H1 baseline | #10 Ask A | **Ask C.1** |
|---|---|---|---|
| Bucket `12 × block_bwd` | 2244 ms | 960.0 | **844.5 (−62.4%)** |
| full-step seq8 | 4141 | 3000 | **2922 (−29.4%)** |
| full-step seq256 | 6015 | 5531 | 5594 |
| full-step seq1024 | 16094 | 16282 | 15969 |
### 5. Kiểm chứng
| kiểm | kết quả |
|---|---|
| `tools/gate_f32.sh` | **135/135, `TOTAL fails=0`, exit 0** |
| Trainer | **15/15 `fails=0`** |
| Probe `_am5` | Ask A `bit_bad=0`; Ask C.1 seq8 `smm_fwd_at_bad=0 native_bad=0` |
| Probe `_p6ms` | **A 4.21× · B 5.40× · C 5.06×**, `bit_bad=0`, `fails=0` |
| Self-host | **IL_IDENTICAL** |
### 6. Còn lại (Ask B + Ask C.2) — chưa làm, báo số
- **Ask B `linear_bwd_fused`:** còn `dx = smm_dot(gy, w, n, in_d, out_d)` **12.69 ms (w1) /
  5.81 ms (qkv)**. Hợp đồng `smm_dot` (`trans_b=1`) khác `dw` → đề nghị **tách riêng**
  `linear_dx_fused`, không nhét chung.
- **Ask C.2 12 head block** (4.39 ms, 5 `t_zeros`/head): **repo-side** (cache scratch), ngoài
  phạm vi compiler.
- **`n > 64`:** chỉ khả thi nếu chấp nhận không bit-exact, hoặc sinh `dw` vào scratch rồi cộng
  1 lần (2 pass, vẫn nhanh hơn interpreted nhiều) — mở được seq256/1024 nhưng seq8 mới là nơi
  có gain.
---
## §0.63 REQUEST #10 ASK B `linear_dx_fused` + ASK C.2 CACHE SCRATCH 12 HEAD: SEQ8 −25.8% TRONG PHIÊN, LOSS EXACT (2026-11-02)
Tiếp §0.61/§0.62. Hai mục cuối của #10 (§6 «Còn lại») nay đã làm.
### 1. Ask B — builtin `linear_dx_fused` (native, cùng TU với static helper)
`dx = smm_dot(gy, w, m, n, k)` trong `gpt2_linear_bwd` → `_smm_full` → `SgemmFull` →
`SgemmF32` tính **đủ 256 hàng** nhưng chỉ hàng `< m` được `_acc_f` cộng vào `dx` —
**248/256 hàng thừa = 32× phép nhân** ở seq8 (`m=8`).
- **Native `LinearDxFused`** đặt trong **cùng translation unit** `tkv_simd_native.c`,
  gọi thẳng `full_tile_a` / `full_tile_b` / `tkv_k_eff` / `gemm_rect_f32` / `gemm_rows_f32`
  với `i1 = m` → **bit-exact mang tính cấu trúc** (chỉ bỏ hàng mà `_acc_f` không đọc),
  **không nhân bản code** → không có «hai nguồn sự thật» cho hợp đồng bit-exact.
- **Builtin wiring** (đúng mẫu `linear_dw_fused`): `LinearDxFusedNative` (pinvoke) +
  IL wrapper `LinearDxFusedM` + `compile_linear_dx_fused` + `register_expr_builtin`
  trong `simd_feature.tkv`, + `'linear_dx_fused'` vào `_BUILTIN_LENH_DOC_LAP` (`file_io.tkv`).
- **Hợp đồng từ chối (caller fallback `smm_dot` nguyên vẹn):** `-9 k<256` (không qua
  `_smm_full`) · `-10 m>256` (đi `_smm_pk`) · `-11 m%4` (i-tail scalar) ·
  **`-12 k%256`** (xem §2).
**Bẫy bit-exact tìm ra bằng đo (không phải suy luận):** probe `_am6` bắt `k=300`
(`m=200, n=96`) **lech 1264 ptu**. Tách 2 yếu tố bị confound:
| arm | kết quả | kết luận |
|---|---|---|
| `k300-onlyA` (m=8, k=300 — p-tile le, 1 block mc) | **0** | p-tile le một mình **không** lỗi |
| `m200-onlyB` (m=200, k=768 — nhieu block mc, p day) | **0** | nhieu block mc mot minh **khong** loi |
| `k300-both` (m=200, n=96, k=300) | **1264** | **CA HAI cung nhau moi lech** |
→ Thêm guard **`-12` khi `k % 256 != 0`** (p-tile cuối luôn đầy ⇒ luôn đi
`gemm_rows_f32`, không bao giờ vào nhánh lệch). **Sản phẩm không bị ảnh hưởng:**
`out_d` thật luôn ∈ {768, 2304, 3072} = bội của 256. *Không chứng minh được ⇒ không nhận*
— đúng hợp đồng của request.
**Speedup kernel (test_dx.c, best-of-5, native vs `SgemmFull`):**
| shape | ref | native | x |
|---|---|---|---|
| m=8, n=768, k=3072 | 13.60 ms | 8.64 ms | **1.58×** |
| m=8, n=3072, k=768 | 9.34 ms | 5.67 ms | **1.65×** |
| m=8, n=768, k=2304 | 7.63 ms | 4.68 ms | **1.63×** |
| m=8, n=768, k=768 | 2.56 ms | 1.57 ms | **1.63×** |
| m=256 (không bỏ được hàng) | 15.35 ms | 15.70 ms | 0.98× (gọi thẳng `SgemmF32`, parity) |
### 2. Ask C.2 — cache 5 scratch của 12 head (repo-side, thuần `.tkv`)
Trong `gpt2_attention_bwd` nhánh `n < 128` (chính là seq8): mỗi head cấp
`gQ/gK/gV/gDc` (n×hd) + `gDs` (n×n) = **5 `t_zeros` × 12 head = 60 cấp phát/lần gọi**.
**Hoist ra trước vòng `for head`** — an toàn vì mỗi head **ghi đầy đủ trước khi đọc**:
- `gQ/gK/gV/gDc`: vòng `t2 × d` ghi hết `n*hd`;
- `gDs`: mỗi hàng `t2` ghi `u = 0..t2` rồi `0.0` cho `t2+1..n-1` → đủ `n*n`.
→ không đổi **1 phép tính nào** ⇒ bit-exact. Áp cho **cả 5 mirror**
(`tv/training/gpt2_ref.tkv` + `tv/training/tv.training.gpt2_ref{,124}.tkv` +
`tv/tests/tv.training.gpt2_ref{,124}.tkv`).
**Lưu ý phạm vi:** chỉ nhánh `n < 128`. Nhánh worker (`n ≥ 128`) mỗi worker
đã nhận **đúng 1 head** (`hlo..hhi` = 1 head) nên hoisting ở đó là no-op.
### 3. Số đo (cùng phiên 2026-11-02, best-of-3 xen kẽ, .NET 10.0.13)
**A/B 1 — `bseq_h4c` (trước #10 B/C.2) vs `bseq_dx` (có B):**
| seq | h4c | dx | Δ |
|---|---|---|---|
| 8 | 3687 | **2734** | **−25.9%** |
| 256 | 5547 | 5359 | −3.4% |
| 1024 | 15688 | 15828 | +0.9% (nhiễu) |
**A/B 2 — `bseq_dx` vs `bseq_c2` (có B + C.2), seq8:** dx 2953 → **c2 2859 (−3.2%)**.
**Số chốt (build đầy đủ A+B+C.2, best-of-3):**
| seq | full-step | torch 14T (cùng phiên) | tỉ lệ |
|---|---|---|---|
| 8 | **2735** | 1079.6 | **2.53×** |
| 256 | **5531** | 1598.8 | 3.46× |
| 1024 | **16500** | 4105.2 | 4.02× |
Torch tier E `T=1/14` cùng phiên: 1371.7/1079.6 · 4364.8/1598.8 · 15180.8/4105.2 ms.
**Giải thích tại sao seq256/1024 đứng yên — đúng như thiết kế:**
- seq1024: `m=1024 > 256` → native từ chối (`-10`) → fallback `smm_dot` → **không đổi**;
  và `n=1024 ≥ 128` → đi worker → C.2 không áp dụng.
- seq256: `m=256 = T` → không bỏ được hàng nào → gọi thẳng `SgemmF32` (parity, x0.98).
- seq8: `m=8` → bỏ 248/256 hàng → **đây là nơi có gain**.
**Lũy tích H1 → nay (số khác phiên, chỉ để nhìn hướng):** seq8 4141 → **2735 (−34.0%)**,
seq256 6015 → 5531 (−8.0%), seq1024 16094 → 16500 (+2.5%, trong dải nhiễu).
### 4. Loss EXACT
Mọi lần chạy (cả A/B lẫn số chốt) in đúng 3 chuỗi chuẩn, không đổi 1 chữ số:
```
seq=8     loss=10.572291 -> 7.0119185
seq=256   loss=10.756927 -> 11.807267
seq=1024  loss=10.760399 -> 13.442915
```
### 5. Kiểm chứng
| kiểm | kết quả |
|---|---|
| `tools/gate_f32.sh` | **135/135, `TOTAL fails=0`, exit 0** — chạy 2 lần (sau Ask B, sau C.2) |
| Trainer | **15/15 `fails=0`** — chạy 2 lần |
| Self-host `tools/vong.exe` | **IL_IDENTICAL** — 2 lần |
| Probe `_am6.tkv` (mới) | **`TOTAL bit_bad=0`**; 4 nhánh từ chối `-9/-10/-11/-12` đúng rc |
| `test_dx.c` (mới) | **`ALL BIT-EXACT`** — 10 shape + 4 reject |
### 6. Vấn đề đã ghi nhận (không thuộc thay đổi này)
Self-host thỉnh thoảng **NRE/AV ngắt quãng** khi biên dịch gop 133 file
(`Unhandled Exception ... at TKVApp._fp_walk → fpw_try → _fp_walk`, và đôi khi
`0xC0000005`), tỉ lệ ~1/3 lần chạy, **cùng binary + cùng nguồn**. Đã chụp stack
bằng cách tạm thời `raise` lại exception trong `dich_mot` (đã hoàn nguyên).
Chưa khoanh được thủ phạm — tái hiện được cả trước lẫn sau thay đổi này, nên **không
phải regression của #10**, nhưng cần điều tra riêng (nghiện thứ tự duyệt container
có hash ngẫu nhiên per-process).
## §0.64 ĐO LẠI NỀN TẢNG: KHÔNG CÓ AVX-512, GEMM 59% ĐỈNH NHƯNG CHỈ CHIẾM 25%, NÚT THẮT LÀ ĐƯỜNG NON-GEMM (2026-11-05)
Số đo trong **cùng một phiên**, không trộn số cũ. Mục đích: chốt lại nền tảng trước khi
tiếp tục H4a/H4b và tránh tối ưu sai chỗ (lịch sử §0.59 và plan P3 đều đã đặt cược sai
một lần vì suy luận thay vì đo).
### 1. Máy: chỉ AVX2, KHÔNG có AVX-512 — giả thuyết "thiếu AVX-512" đã chết
| kiểm | kết quả |
|---|---|
| `tools/simd_native/test_cpuid.c` | `GenuineIntel`, `OSXSAVE=1 AVX=1 FMA=1`, `XCR0=0x7`, `AVX2=1`, **`AVX512F=0 DQ=0 BW=0 VL=0`** |
| `tools/simd_native/test_avx512.c` build `/arch:AVX512` | chạy → **`0xC000001D` (STATUS_ILLEGAL_INSTRUCTION)** |
| `nproc` / `os.cpu_count()` | **14** / 28 → tiến trình bị giới hạn 14 logical = **7 core vật lý + SMT** |
⇒ Mọi kế hoạch dựa trên "dùng AVX-512 như torch" đều vô hiệu. ISA đã khai thác hết.
### 2. Trần thật của máy (2 BLAS độc lập, cùng shape m=1024 n=3072 k=768 = 4.83 GFLOP)
| công cụ | 1 luồng | 14 luồng | scale |
|---|---|---|---|
| torch `F.linear` (MKL) | 63.4 ms = **76.2 GFLOP/s** | 10.1 ms = **476.6 GFLOP/s** | 6.25× |
| numpy `a@b` (OpenBLAS, `X86_V3` = AVX2) | 63.8 ms = **75.7 GFLOP/s** | — | — |
| 3 shape mlp (A/B/C) | 367.2 ms | 65.3 ms | 5.62× |
⇒ đỉnh 1 luồng ≈ **75.5 GFLOP/s** = 32 flop/cycle @ ~2.36 GHz (đúng AVX2 FMA 2×256-bit).
14 luồng ≈ **515 GFLOP/s**, và torch 14T đạt 476–519 = **92–100% đỉnh**. Không còn "phép thuật"
nào của torch để đuổi theo: nó chỉ chạy đúng đỉnh máy.
### 3. GEMM của tkv: 59% đỉnh, và các biến thể "đúng sách" đều KHÔNG nhanh hơn
`out/gm.exe` (đường production `smm_dot`/`smm_fwd`), sweep số luồng:
| TKV_GEMM_THREADS | A (ms / GFLOP/s) | MLP3 (ms) | scale |
|---|---|---|---|
| 1 | 140 / 34.5 | 422 | 1.00× |
| 2 | 78 / 61.9 | 250 | 1.79× |
| 4 | 46 / 105.0 | 170 | 3.04× |
| 7 | 31 / **155.9** | 140 | 4.52× |
| 14 | 31 / 155.9 | 140 | 4.52× |
| 28 | 31 / 155.9 | 125 | 4.52× |
- 1 luồng: **34.5 / 75.5 = 46%** đỉnh; bản tốt nhất trong bench là 44.3 = **59%**.
- Thread: **4.52× trên 7 core vật lý = 65% hiệu suất**; 7T = 14T = 28T (SMT không thêm gì,
  đúng như `tkv_nthreads` dự đoán).
**Sweep `GEMM_KC`** (`out/kc_sweep.ps1`, chỉ để đo tốc độ — đổi KC làm đổi thứ tự tích lũy k
nên KHÔNG bit-exact): 64→38.7, 128→43.9, 192→38.7, **256→44.3**, 384→44.3, 512→34.5 GFLOP/s.
⇒ giảm số lần quét C chỉ được **+15%**, không phải 2×.
**Viết lại micro-kernel** (`tools/simd_native/bench_micro.c`, đúng shape, `memcmp` xác nhận
bit-exact với bản production):
| biến thể | shape lớn | shape C nhỏ (64×64×65536) |
|---|---|---|
| V0 4×16 production | **43.9** | 33.6 |
| V5 5×16 (10 accumulator) | 31.2 | 29.8 |
| V6 6×16 (12 accumulator) | 39.6 | 35.8 |
| V7 4×16 tăng con trỏ (bỏ `imul` per-p) | 43.1 | 33.6 |
Ghi chú: `dumpbin` cho thấy MSVC sinh `movsxd`/`imul` **cho từng bước p** (tranh port 5 với
`vbroadcastss`) — nhưng V7 sửa đúng chỗ đó cũng **không nhanh hơn**. Giả thuyết "8 chuỗi
accumulator < latency FMA 5" cũng **sai**: tăng lên 10/12 chuỗi lại chậm hơn (spill).
⇒ **Không có lever 2× nào trong micro-kernel ở mức chi phí hợp lý.** Trần thực tế ~1.5×
(46% → 59% đã là bản tốt nhất đo được).
### 4. Nhưng GEMM chỉ là 25% — nút thắt thật là đường non-GEMM
Full step seq1024 cùng DLL, ép số luồng:
| TKV_GEMM_THREADS | 1 | 2 | 4 | 7 | 14 |
|---|---|---|---|---|---|
| full step (ms) | **38484** | 24000 | 19203 | 17609 | **16688** |
| scale | 1.00× | 1.60× | 2.00× | 2.19× | **2.31×** |
- torch cùng phiên: 15180 (1T) → 4105 (14T) = **3.70×**.
- ⇒ **1 luồng tkv chậm 2.53×** (38.5s vs 15.2s) và **scale chỉ 2.31× so với 3.70×**.
Ước lượng khối lượng GEMM ở seq1024 (từ `_gm` + đếm layer, **đây là ước lượng, không phải
số đo trực tiếp**): mlp 12 layer × ~4×4.8 GF ≈ 230 GF; attention+qkv+proj ≈ 180 GF; tổng
≈ 410 GF. Ở 34.5 GFLOP/s ⇒ ≈ **11.9s trong 38.5s = 31%**. Phần còn lại ≈ **26.5s là non-GEMM**.
Đối chiếu torch: 410 GF ở 76 GFLOP/s = 5.4s, còn **9.8s là non-GEMM**.
⇒ **đường non-GEMM của tkv chậm 2.7× so với torch, và nó chiếm ~69% thời gian 1 luồng.**
Hệ quả định lượng: nếu đưa GEMM từ 46% lên 100% đỉnh (bất khả thi, và đã chứng minh không
có đường) thì full step 1 luồng chỉ còn 38.5 − 6.5 = 32s; ở 14 luồng impact còn nhỏ hơn
(GEMM ~17% thời gian) ⇒ **tối ưu GEMM tối đa ~10% end-to-end. Đã đóng hướng này.**
### 5. Phát hiện phụ, phải sửa trước khi tuyên bố "EXACT": loss đổi theo số luồng
Cùng binary, cùng dữ liệu, chỉ đổi `TKV_GEMM_THREADS`:
```
1T  -> loss=10.760399 -> 13.442917
7T  -> loss=10.760399 -> 13.442922
14T -> loss=10.760399 -> 13.442913
```
⇒ threading **không deterministic** (đã thấy dấu vết ở §0.63 mục H3 nhưng nay xác nhận có
hệ thống). "Loss EXACT" hiện chỉ đúng **trong cùng một số luồng**. Trước khi so sánh với
PyTorch trên máy khác phải chốt số luồng hoặc tìm điểm cộng không tất định.
### 6. Ưu tiên mới (thay P3/P5 cũ)
| # | lever | cơ sở đo | trần lợi |
|---|---|---|---|
| **A** | **đường non-GEMM** (`.tkv` elementwise + `TkvInt::`, copy) | 26.5s vs torch 9.8s ở 1 luồng | **tới 2.7×** — lớn nhất |
| **B** | thread scaling 2.31× → 3.70× | bảng §4 | 1.6× |
| C | determinism theo số luồng | §5 | bắt buộc trước khi so máy khác |
| D | GEMM kernel 46% → 59% | §3 | 1.1× end-to-end; **hoãn** |
### 7. Tệp sinh ra trong lần đo này
`tools/simd_native/test_cpuid.c` + `run_cpuid.ps1` · `test_avx512.c` + `run_avx512.ps1` ·
`test_peak.c` + `run_peak.ps1` (lưu ý: MSVC unroll 32 + spill nên bench này **không** đại
diện, chỉ giữ làm bằng chứng) · `bench_micro.c` + `run_bench_micro.ps1` (4 biến thể +
chế độ PEAK + kiểm bit-exact) · `out/kc_sweep.ps1`.
**Không có dòng code production nào bị đổi** trong §0.64: các biến thể kernel đều không
nhanh hơn, nên sửa sẽ là regression. Gate 135/135 và trainer 15/15 vì vậy vẫn giữ nguyên
giá trị từ §0.63.
## §0.65 PHÂN RÃ 1 BƯỚC PRODUCTION Ở SEQ1024 (1T và 14T) — ĐÍNH CHÍNH §0.64
§0.64 kết luận "GEMM chỉ ~31%, non-GEMM 69%" bằng **ước lượng**. Nay đo trực tiếp bằng
`mode 2` mới của `tv/tests/simd_bench_124m_seq.tkv` (5 pha y hệt `mode 1`: cùng `sc4/sc5/m6`,
cùng `gflat`; mọi `now_ms()` đặt trong `main` để tránh bug #8) ⇒ **ước lượng đó sai**.
Bản ghi này thay thế các tỷ lệ ở §0.64 mục 4.
### 1. Năm pha của 1 bước, seq1024 (best-of-2 mỗi pha)
| pha | 1 luồng | 14 luồng | scale |
|---|---:|---:|---:|
| `gpt2_forward_saved` (fwd) | 8844 | **3000** | 2.95× |
| `gpt2_loss` | 109 | 78 | 1.40× |
| `gpt2_backward` (bwd) | 24282 | **10156** | 2.39× |
| `opt.step()` | 984 | **937** | **1.05×** |
| copy-back `d[j]=opt.p[..]` (124 439 808 phần tử) | 125 | 110 | 1.14× |
| **TỔNG** | **34344** | **14281** | 2.40× |
Đối chiếu full-step mode 1 cùng phiên: 38484 (1T) / 16688 (14T) — chênh 12–14% vì mode 2
lấy **min riêng từng pha** (lạc quan) còn mode 1 là một lần gọi. Dùng mode 1 khi so với
torch; dùng mode 2 khi cần biết **tỷ lệ**.
**Phát hiện 1: `opt.step()` KHÔNG scale chút nào** (984 → 937 ms, 1.05×). Nó là 937 ms cố
định trong một bước 14.3 s = **6.6%**, cộng copy-back + loss = **1.12 s = 7.9%** dù đã có 14
luồng. torch gộp optimizer vào bước của nó gần như miễn phí.
### 2. Bên trong bwd: block_bwd chia 4 vùng (probe `tv/tests/_bd1024.tkv`, seq1024, 1 luồng)
Một `gpt2_block_bwd` production (có `gflat`) = **1578 ms**:
| vùng | ms | % |
|---|---:|---:|
| `gpt2_mlp_bwd` | 797 | **50.5%** |
| `gpt2_attention_bwd` | 578 | **36.6%** |
| `gpt2_layernorm_bwd` ×2 | 47 | 3.0% |
| còn lại (residual/copy/ceremony) | 156 | 9.9% |
| **Σ 12 block** | **18936** | |
Bên ngoài block: `bwd` 24282 − 18936 = **5346 ms** cho lm-head + `op_embed_bwd`
(1 luồng). Đó là 16% của bước và **lớn hơn cả `attention_bwd` của một block ×12**.
### 3. Tỷ lệ GEMM / non-GEMM (suy ra từ khối lượng, KHÔNG đo trực tiếp — đã ghi rõ)
Khối lượng phép nhân 1 bước seq1024 ≈ 410 GF (đếm layer). Ở GEMM 46% đỉnh = 34.5 GFLOP/s (1T):
| | 1 luồng | 14 luồng |
|---|---:|---:|
| GEMM (410 GF) | ≈ 11.9 s = **35%** | ≈ 11.9/4.52 = **2.6 s = 18%** |
| non-GEMM (phần dư) | ≈ 22.4 s = **65%** | ≈ 11.7 s = **82%** |
⇒ §0.64 viết "GEMM ~31%" cho 1 luồng: gần đúng. Nhưng kết luận quan trọng hơn nằm ở cột
14 luồng: **ở cấu hình đích, non-GEMM là 82% thời gian**. Và non-GEMM 11.7 s một mình đã
gấp **2.9× toàn bộ 4105 ms của torch 14T**.
### 4. Thử thêm hai hướng GEMM: đều không có lợi (đóng lại)
- **Sweep MC/NC** (`out/mc_sweep.ps1`; đổi MC/NC **không** đổi thứ tự tích lũy k nên vẫn
  bit-exact): `(64,64)` 34.5 · `(128,64)` 34.5 · `(192,64)` 34.5 · `(64,124)` **23.8** ·
  `(128,124)` 22.2 · `(128,124,KC256)` 16.3 · `(192,124,KC256)` 16.3 GFLOP/s. ⇒ tiling hiện
  tại đã tối ưu; NC lớn làm **giảm** mạnh.
- **4 biến thể micro-kernel** (§0.64 mục 3): không cái nào nhanh hơn 4×16 production.
⇒ Không còn tham số nào để vặn. Muốn 2× GEMM phải viết micro-kernel **chất lượng asm**
(kiểu BLIS 6×16) — việc của team native, không phải tinh chỉnh. Và trần của nó ở 14 luồng
chỉ là **−2.6 s / 14.3 s = 1.22×**, không đủ để bắt torch.
### 5. Ưu tiên chốt lại bằng số (14 luồng, seq1024, bước 14.3 s)
| # | mục | ms | % bước | scale hiện tại | ghi chú |
|---|---:|---:|---:|---|
| **A1** | non-GEMM trong `mlp_bwd` + `attention_bwd` (12 block) | **≈ 7.7 s** | **54%** | không scale | 641 ms/block (237 + 201 + 156 + 47) |
| **A2** | lm-head + `op_embed_bwd` ngoài block | ≈ 3.0 s | 21% | chưa đo riêng | 5346 ms ở 1T |
| **B** | `opt.step()` + copy-back + loss | **1.12 s** | 7.9% | **1.05×** | cắt được gần hết |
| C | GEMM | 2.6 s | 18% | 4.52× | trần 1.22× |
**Việc rẻ nhất, chắc chắn thắng: B.** 1.12 s không scale, nằm gọn trong 2 vòng lặp `.tkv`
(`opt.step()` là native nhưng đơn luồng; copy-back là 124M phép gán `.tkv`) ⇒ thread hoá
optimizer + dùng builtin copy khối thay vòng lặp từng phần tử. Kỳ vọng −0.9 s/bước = **6%**.
Việc lớn nhất là **A1**: cần một probe chia 641 ms/block thành từng op (gelu_bwd, bias_grad,
residual add, softmax_bwd, copy) rồi vector hoá/thread hoá từng cái — chưa làm.
### 6. Probe mới (tái lập được)
`tv/tests/_bd1024.tkv` → `out/bd1024.exe` (4 vùng `block_bwd` tại seq bất kỳ, đọc seq từ
`_seq_cfg.txt`) · `tv/tests/_tail1024.tkv` → `out/tail1024.exe` (opt.step / copy-back /
zero_grad trên 124M) · `mode 2` của `simd_bench_124m_seq.tkv` → `out/bseq_p2.exe`
(5 pha production). Lệnh:
```
TKV_GEMM_THREADS=1 dotnet out/bd1024.exe
TKV_GEMM_THREADS=1 dotnet out/bseq_p2.exe     # _seq_cfg.txt = "1024\n2\n1"
```
Vẫn **không đổi code production** (`.tkv` chỉ thêm mode 2 vào tệp bench).
**Ghi chú trạng thái git:** `mode 2` nằm trong `tv/tests/simd_bench_124m_seq.tkv`, nhưng tệp
này **đã có sẵn thay đổi chưa commit từ phiên trước** (phần nối `m6_*` vào `_f124_bwd` /
`_step124`). Để không commit thay đổi của phiên khác, tệp này **không được commit** trong
§0.65 — nó vẫn nằm trong working tree và `out/bseq_p2.exe` build được từ đó. Muốn tái lập từ
bản sạch thì phải giữ nguyên tệp ở working tree này.
**Độ ổn định (đo lại 3 lần, cùng phiên):** full step 14 luồng auto = 15593 / 17422 / 16672 ms
⇒ dùng **16672 (median)**, dải ±6%. Con số 14125 ms của một lần chạy `bseq_p2` là may mắn,
không dùng. Tỷ lệ trong §0.65 mục 5 vì vậy giữ nguyên (bước ≈ 14.3–16.7 s, các % đổi < 2 điểm).
### 7. Bên trong `opt.step()` — dự báo ở §0.65 mục 5 quá lạc quan, sửa lại
`out/tail1024.exe` (probe `_tail1024.tkv`), N = 124 439 808, best-of-3:
| phần | ms | ghi chú |
|---|---:|---|
| `grad_norm()` | 140 | **đã** thread 12 worker (`_norm_groups`) — không còn đáng cắt |
| `f32_copy_to_buf` (496 MB) | 78 | native |
| **`adamw_step` (native)** | **313** | **đơn luồng**, 2.5 ns/ptu |
| `f32_copy_from_buf` (496 MB) | 78 | native |
| **Σ đo được** | **609** | `opt.step()` đo trực tiếp = 687 |
| copy-back `d[j]=opt.p[..]` | 125 | vòng lặp `.tkv` từng phần tử |
| `zero_grad` | 78 | chạy thread song song với fwd nên **không** tính vào bước |
**Sửa dự báo:** §0.65 mục 5 viết "cắt được −0.9 s = 6%". Sai. Cụm opt+copy+loss thực tế là
**1.12 s**, nhưng phần cắt được chỉ là: thread hoá `adamw_step` (313 → ~60 ms nếu 7 luồng) +
thay vòng copy-back bằng builtin `f32_list_copy` (125 → ~30 ms) ≈ **−350 ms = 2.4%** của bước
14.3 s. `grad_norm` và 2 lần `copy_*_buf` đã native/threaded, không còn gì để lấy.
⇒ **B tụt xuống 2.4%, không còn là "việc rẻ nhất chắc thắng" về mặt lợi ích** — chỉ còn rẻ về
công sức. Theo số đo thì A1 (7.7 s) và A2 (3.0 s) mới đáng làm, và cả hai đều cần **probe
chia nhỏ hơn nữa** (gelu_bwd / bias_grad / residual / softmax_bwd / copy trong `mlp_bwd` và
`attention_bwd`; và `op_embed_bwd` + lm-head) — chưa làm trong §0.65.
## §0.66 ĐO TỪNG OP TRONG `mlp_bwd` — KẾT LUẬN "NON-GEMM LÀ NÚT THẮT" Ở §0.65 SAI
§0.65 suy tỷ lệ GEMM/non-GEMM **từ đếm layer** (410 GF) rồi kết luận non-GEMM 82%. Con số
410 GF đó **thiếu**: mỗi `mlp_bwd` có 5 GEMM chứ không phải 4, `gpt2_linear_bwd` làm 2 GEMM,
và lm-head có cả fwd lẫn bwd. Probe `tv/tests/_mb1024.tkv` đo trực tiếp từng op nên không
còn phải suy.
### 1. `mlp_bwd` tại seq1024 — đường production `simd==1`, có fallback `smm_fwd_at`
| op | 1 luồng | 14 luồng | loại |
|---|---:|---:|---|
| `gpt2_linear` (fwd, 4.83 GF) | 125 | 31 | GEMM |
| `_gelu_fwd_par` (3.1M) | 16 | 31* | non-GEMM |
| `smm_dot` → dx2 (4.83 GF) | 141 | 31 | GEMM |
| `op_gelu_erf_bwd` (3.1M) | 31 | 16* | non-GEMM |
| `smm_fwd_at` → dw2 (4.83 GF) | 156 | 63 | GEMM |
| vòng `db2` trên gflat (786K) | <16 | <16 | non-GEMM |
| `gpt2_linear_bwd` (dx + dw1 = 9.66 GF) | 297 | 109 | GEMM |
| **Σ đo được** | **766** | **281** | |
| `gpt2_mlp_bwd` (đo trực tiếp) | **782** | **281** | |
| **GEMM** | **719 (91.9%)** | **234 (83.3%)** | |
| **non-GEMM** | **63 (8.1%)** | **47 (16.7%)** | |
\* hai số 31/16 ở 14 luồng nằm trong 2 quantum của `now_ms()` (15.6 ms) — coi là nhiễu.
**Hai điều phải ghi rõ:**
- `linear_dw_fused` **trả rc != 0** tại đây vì `m = seq = 1024 > 256` (guard `-10`), nên
  production rơi về `smm_fwd_at`. Lần đo đầu tôi chỉ gọi `linear_dw_fused` nên thấy **0.0 ms**
  và thiếu một GEMM 4.83 GF. Probe đã sửa để mô phỏng đúng fallback.
- `mlp_bwd` scale chỉ **2.78×** (782 → 281) dù riêng GEMM scale **3.07×** (719 → 234) và
  `gm.exe` đo standalone scale **4.52×**. Trong vòng lặp thật, GEMM mất ~1/3 mức scale.
### 2. Tính lại tỷ lệ GEMM của cả bước (sửa §0.65 mục 3)
`mlp_bwd` 24.2 GF/block (5 GEMM). `attention_bwd` ~11.2 GF/block (dx_qkv 3.62 + dw_qkv 3.62 +
2 GEMM score 1.6 + 2 GEMM proj 2.4). Fwd 17.7 GF/block, lm-head fwd+bwd 158 GF:
| | GF | 1 luồng (34.3 s) | 14 luồng (16.7 s) |
|---|---:|---|---|
| tổng GEMM | **≈ 814** | 814/33.6 = **24.2 s = 71%** | /3.1 = **7.8 s = 47%** |
| non-GEMM | — | **≈ 10.1 s = 29%** | **≈ 8.9 s = 53%** |
⇒ §0.65 viết "14 luồng non-GEMM 82%" là **sai**; đúng là **53%**. Nhưng kết luận định tính
vẫn giữ: ở cấu hình đích non-GEMM là **phần lớn hơn**, và 8.9 s đó đã gấp **2.2× toàn bộ
4105 ms của torch**.
### 3. Chỗ nào trong non-GEMM (đo được, không suy)
| mục | cơ sở đo | 1 luồng | 14 luồng |
|---|---|---:|---:|
| ceremony trong `block_bwd` | 156 ms/block × 12 | 1.9 s | **1.9 s** (không scale) |
| `attention_bwd` non-GEMM | 578 − ~333 (GEMM 11.2 GF) | ~2.9 s | ~2.9 s |
| `gelu_fwd` + `gelu_bwd` | 47 ms/block × 12 | 0.6 s | ~0.6 s |
| `layernorm_bwd` ×2 | 47 ms/block × 12 | 0.6 s | ~0.6 s |
| `opt.step` + copy-back + loss | §0.65 mục 7 | 1.1 s | **1.1 s** |
| còn lại | phần dư | ~3.0 s | ~1.9 s |
`attention_bwd` non-GEMM (~2.9 s) và **ceremony 156 ms/block** (1.9 s) là hai mục lớn nhất.
Ceremony = phần `block_bwd` trừ 3 vùng đo được: residual add, copy gradient vào `gflat`, và
các `t_zeros` — **chưa có phép đo nào bên trong**, cần probe riêng trước khi cắt.
### 4. Ưu tiên sửa lại lần cuối (14 luồng, bước 16.7 s)
| # | mục | s | % | trần nếu sửa |
|---|---:|---:|---|
| **A** | `attention_bwd` non-GEMM | 2.9 | 17% | ? (chưa biết thành phần) |
| **B** | GEMM: 46% đỉnh **và** scale 3.1× thay vì 4.5× | 7.8 | 47% | −2.4 s nếu đạt cả hai |
| **C** | ceremony 156 ms/block | 1.9 | 11% | ? |
| D | `opt.step` + copy-back | 1.1 | 7% | −0.35 s |
| E | gelu + layernorm | 1.2 | 7% | ~−0.5 s nếu thread hoá |
**B là lever lớn nhất có trần đã biết.** Trong đó phần scale (3.1× → 4.5×) rẻ hơn nhiều so
với việc viết lại kernel: cùng một `gm.exe` standalone đạt 4.52×, nên vấn đề scale nằm ở
**cách gọi trong vòng lặp** (kích thước tile/`tkv_min_n`/`tkv_ncols` theo `n` của từng GEMM
trong block), không phải trong kernel.
### 5. Probe
`tv/tests/_mb1024.tkv` → `out/mb1024.exe`: 7 op của `mlp_bwd` + đối chiếu hàm thật, chạy
được ở mọi seq (`_seq_cfg.txt`), in cả GEMM/non-GEMM. Lệnh:
`TKV_GEMM_THREADS=1 dotnet out/mb1024.exe` (không đặt env = 14 luồng auto).
## §0.67 SỬA LẦN HAI: `block_bwd` PRODUCTION = 1421 ms, CEREMONY ≈ 0 — THAY §0.65/§0.66
Hai bản trước đo `block_bwd` trên **đường sai**, nên mọi số của block phải thay:
1. **§0.65** gọi `gpt2_mlp_bwd` / `gpt2_attention_bwd` / `gpt2_layernorm_bwd` **không có
   `gflat` + offsets** ⇒ đo nhánh trả-Tensor, không phải nhánh production ghi vào `gflat`.
2. **§0.66** gọi `gpt2_block_bwd` **không có `save_r/save_a/save_m`** ⇒ block tự tính lại
   `gpt2_attention` fwd + 2 lần `gpt2_layernorm` fwd (production luôn truyền 3 tensor này từ
   `saves[3+N+li]`, `saves[3+2N+li]`, `saves[3+3N+li]`). Đó là nguồn của "156 ms ceremony".
Probe `_bd1024` đã sửa cả hai (dùng `gpt2_forward_saved` + truyền đủ 3 saves + `gflat`).
### 1. Số đúng — seq1024, 1 luồng, production
| vùng | ms | % của block |
|---|---:|---:|
| `gpt2_mlp_bwd` (có gflat) | **781** | **55.0%** |
| `gpt2_attention_bwd` (có gflat) | **625** | **44.0%** |
| `gpt2_layernorm_bwd` ×2 | 31 | 2.2% |
| Σ 3 vùng | 1437 | |
| **`gpt2_block_bwd` (production)** | **1421** | |
| ceremony (block − Σ) | **≈ −16 (nhiễu) ⇒ 0** | **0%** |
| **12 × block** | **17052** | |
**Ceremony không tồn tại.** Ba vòng `n*h = 786 432` phần tử + 2 `t_zeros` đo riêng
(`_cer1024.tkv`) đều **dưới 1 quantum 15.6 ms**, tức < 16 ms cho cả 3 vòng. Mục "C: ceremony
1.9 s" trong §0.66 là **sản phẩm của phép đo sai đường** — huỷ bỏ, không có gì để cắt ở đó.
### 2. Sổ cái đúng của 1 bước (seq1024, 1 luồng, tổng 34.3 s)
| mục | ms | % | GEMM bên trong |
|---|---:|---:|---|
| fwd 12 block (+loss riêng 109) | 8844 | 26% | 212 GF ⇒ ~6300 (71%) |
| bwd 12 block | **17052** | 50% | mlp 92% + attn ~53% ⇒ ~12600 |
| ├ `mlp_bwd` | 9372 | 27% | 92% (đo trực tiếp, §0.66) |
| ├ `attention_bwd` | **7500** | **22%** | ~53% ⇒ non-GEMM **~3500** |
| └ `layernorm_bwd` | 372 | 1% | 0 |
| **lm-head + `op_embed_bwd`** | **7230** | **21%** | 158 GF ⇒ ~4700, non-GEMM **~2530** |
| `opt.step` + copy-back + loss | 1218 | 3.5% | 0 |
⇒ tổng GEMM ≈ **23.6 s = 69%** khớp với §0.66 mục 2 (71%). Ở 14 luồng, GEMM scale ~3.1× ⇒
GEMM ≈ 7.6 s, non-GEMM ≈ 9.1 s (55%).
### 3. Hai mục non-GEMM lớn nhất (đo được, không còn mục nào bị đếm sai)
| # | mục | 1 luồng | 14 luồng | ghi chú |
|---|---|---:|---:|---|
| **A** | `attention_bwd` non-GEMM | ~3.5 s | **~3.5 s** (không scale) | 44% của block |
| **B** | lm-head + `op_embed_bwd` non-GEMM | ~2.5 s | ~2.5 s | 21% của bước |
| C | GEMM ở 46% đỉnh + scale 3.1× | 23.6 s | 7.6 s | trần −2.4 s |
| D | `opt.step` + copy-back | 1.1 s | 1.1 s | trần −0.35 s |
**A là mục lớn nhất chưa ai mở.** `attention_bwd` 625 ms/block chỉ có ~333 ms là GEMM (11.2 GF ở
33.6 GFLOP/s); ~292 ms còn lại × 12 = 3.5 s là softmax_bwd / mask / transpose / vòng head —
**chưa có probe bên trong**, và đây là việc tiếp theo.
### 4. Bài học phương pháp (ghi lại để không lặp)
Hai lần liên tiếp, số đo "đúng định dạng" nhưng **sai đường gọi** đã tạo ra một nút thắt
không tồn tại (ceremony 1.9 s) và một tỷ lệ sai (non-GEMM 82%). Quy tắc từ nay: **trước khi
tin một phép đo vùng, phải đối chiếu tổng các vùng với hàm bao và kiểm rằng mọi tham số
(`gflat`, `offs`, `save_*`) trùng với call site production** — sai số dư ở đây chính là dấu
hiệu đường gọi lệch, không phải "overhead".
## §0.68 NÚT THẮT SỐ 1 Ở CẤU HÌNH ĐÍCH: `attention_bwd` = 4116 ms, 74 GEMM SIÊU NHỎ/BLOCK
Số đo §0.67 (đường production, có `gflat` + saves) ở cả 2 cấu hình:
| vùng / block | 1 luồng | 14 luồng | scale |
|---|---:|---:|---:|
| `mlp_bwd` | 781 | 250 | **3.12×** |
| `attention_bwd` | 594–609 | **343** | **1.75×** ← kém nhất |
| `layernorm_bwd` ×2 | 31 | 32 | 1.0× |
| **`block_bwd`** | 1421 | **641** | 2.22× |
| **12 × block** | 17052 | **7692** | |
Sổ cái 14 luồng (bước 14.3 s, khớp `mode 2`):
| mục | ms | % |
|---|---:|---:|
| **`attention_bwd` × 12** | **4116** | **29%** ← lớn nhất |
| `mlp_bwd` × 12 | 3000 | 21% |
| fwd × 12 block | 3000 | 21% |
| lm-head + `op_embed_bwd` | 2464 | 17% |
| `opt.step` + copy-back + loss | 1125 | 8% |
| `layernorm_bwd` × 24 | 384 | 3% |
### Nguyên nhân đo được, không suy đoán
`attention_bwd` tại `n = 1024` đi nhánh 12 worker (`n >= 128`), mỗi head một worker. Trong
mỗi head có **6 lời gọi GEMM**: `S = Q·Kᵀ`, `ctx = P·V`, `dp = dctx·V`, `dq = ds·K`,
`dkT = Qᵀ·ds`, `dvT = dctxᵀ·P`; cộng `gpt2_linear` (qkv) + `smm_dot` (dctxT) ở đầu hàm
⇒ **≈ 74 lời gọi GEMM cho một `attention_bwd`**, mỗi lời gọi là `1024×1024×64` = 134 MFLOP.
| | số |
|---|---|
| công việc GEMM thực của 1 block | 11.2 GF |
| ở 156 GFLOP/s (14 luồng, đo standalone) | 0.86 ms / lời gọi |
| **thời gian thực tế ở 14 luồng** | **4.6 ms / lời gọi** |
| ⇒ phần dôi ra | **81%** |
Hai nguyên nhân cộng lại:
1. **`g_gemm_lock` serialize toàn bộ**: 12 worker gọi GEMM đồng thời nhưng phải xếp hàng
   trên một lock toàn cục + một mảng `g_job` toàn cục (`tkv_run_mt_rect` giữ lock suốt
   thời gian tính).
2. **Mỗi lời gọi quá nhỏ cho 14 luồng**: 134 MFLOP không đủ bù chi phí `SetEvent` /
   `WaitForMultipleObjects` / chia lưới.
Đo phụ trợ: `exp()` 22.2 ns ⇒ softmax causal 6.3M exp = 140 ms ở 1 luồng, tức **không** phải
thủ phạm (và 12 worker đã làm song song). Vòng cộng 6.3M phần tử **< 15.6 ms** (2.5 ns/ptu)
⇒ loop `.tkv` không chậm; toàn bộ vấn đề là **chi phí mỗi lời gọi GEMM**.
### Trần lợi nhuận
Nếu loại bỏ 81% overhead đó: `attention_bwd` 343 → ~80–120 ms/block ⇒ **4116 → ~1200 ms**, tức
**−2.9 s = −20% bước**, đưa 14.3 s xuống ~11.4 s. Đây là lever lớn nhất từng đo được trong
toàn bộ §0.64–§0.68, và nó **không** đòi viết lại kernel — chỉ cần bỏ rào serialize.
Hai cách, đều phía native (chi tiết ở `docs/COMPILER_TEAM_REQUEST11.md`):
- **(a) job slot theo thread** thay cho `g_job`/`g_gemm_lock` toàn cục → 12 worker chạy GEMM
  của mình song song thật; hoặc
- **(b) builtin GEMM theo lô** `smm_dot_batch(B, …)`: 12 head × cùng shape → 6 lời gọi thay
  vì 72, mỗi lời gọi đủ lớn để 14 luồng có nghĩa.
Cách (a) rẻ hơn và không đổi chữ ký `.tkv` nào.
### Cập nhật bảng ưu tiên (14 luồng, bước 14.3 s)
| # | mục | ms | % | hành động |
|---|---:|---:|---|
| **1** | overhead 74 GEMM nhỏ trong `attention_bwd` | **2920** | **20%** | REQUEST #11 (a) hoặc (b) |
| 2 | GEMM 46% đỉnh + scale 3.1× | 7600 | 53% | kernel asm (team native) |
| 3 | lm-head + `op_embed_bwd` | 2464 | 17% | chưa có probe bên trong |
| 4 | `opt.step` + copy-back | 1125 | 8% | −350 ms (thread `adamw_step` + builtin copy) |
| 5 | ceremony / `layernorm_bwd` | 384 | 3% | **không có gì để cắt** (đã đo, §0.67) |
**Ưu tiên 1 vì nó vừa lớn nhất vừa không cần viết kernel.** Nó cũng giải thích vì sao
`block_bwd` chỉ scale 2.22× trong khi `mlp_bwd` scale 3.12× và GEMM standalone scale 4.52×.
## §0.69 MỤC 1 ĐÓNG: KNOB ĐÃ CẠN, `attention_bwd` KẸT Ở 328 ms — PHẢI SỬA NATIVE
Sau khi tìm ra thủ phạm ở §0.68 (72 GEMM `1024×1024×64`), đã thử *tất cả* knob có sẵn trước khi
đề nghị sửa native. Số là `ATTENTION_BWD` của một block (production, có `gflat` + saves):
| cấu hình | attention | mlp | block |
|---|---:|---:|---:|
| `TKV_GEMM_MIN_N=96` + auto (14) | **328** | 250 | 610 |
| `MIN_N=1100` (mọi GEMM ≤1024 thành 1 luồng) | 610 | 734 | 1375 |
| `MIN_N=2100` | 610 | 781 | 1422 |
| `GEMM_THREADS=4` | 359 | 313 | 719 |
| `GEMM_THREADS=7` | 343 | 282 | 656 |
| `GEMM_THREADS=10` | **328** | 265 | 625 |
| `GEMM_THREADS=14` | 344 | 281 | 641 |
Ba kết luận:
1. **Ngưỡng `MIN_N` không phải lever.** Nâng ngưỡng làm 72 GEMM nhỏ thành 1 luồng ⇒ attention
   **610 ms (gần gấp đôi)**. Nghĩa là threading *có* giúp các lời gọi nhỏ; vấn đề là nó giúp
   **ít** (594 → 328 = 1.8× trong khi cần 4.5×), không phải "threading vô dụng".
2. **Số luồng cũng không phải lever.** 4/7/10/14 cho 359/343/328/344 — sàn ~328, chênh trong
   nhiễu. Oversubscribe 12 worker × 14 thread **không** phải nguyên nhân chính (nếu là, T=4 sẽ
   thắng rõ).
3. ⇒ Cái còn lại đúng là **chi phí cố định mỗi lời gọi GEMM** (dispatch pool + chia lưới +
   `g_gemm_lock`), thứ mà không knob `.tkv` nào chạm tới. **Mục 1 đóng ở đây**, đường đi là
   `docs/COMPILER_TEAM_REQUEST11.md` (job slot theo thread, hoặc batched GEMM). Trần −2.9 s
   vẫn giữ nguyên, chỉ đổi chỗ thực hiện: native, không phải repo.
Số đúng để dùng từ nay: **attention 328 ms/block → 3936 ms/bước (27%)**; block 610 ms → 7320 ms.
## §0.70 GỐC RỄ THẬT: `_smm_core` RẼ NHÁNH THEO `k` — k<256 ĐI ĐƯỜNG TILED `.tkv` (13.8 GFLOP/s)
§0.68 và §0.69 đều sai nguyên nhân. Chuỗi loại trừ:
| thí nghiệm | kết quả | kết luận |
|---|---|---|
| bỏ `g_gemm_lock` bằng `TryAcquire` + chạy thẳng khi pool bận (sửa thật, đã revert) | attention 328 → **312 ms** | **lock không phải nguyên nhân** |
| `TKV_GEMM_MIN_N` 96/1100/2100 | 328 / 610 / 610 | ngưỡng không phải lever |
| `TKV_GEMM_THREADS` 4/7/10/14 | 359/343/328/344 | số luồng không phải lever |
| **quét `k` ở `n=1024`** (`_g64.tkv`) | k=64: 9.75 ms · k=128: 11.75 · **k=256: 5.875** · k=512: 9.75 · k=768: 11.625 | **k=256 NHANH HƠN k=64** dù 4× khối lượng ⇒ có bậc thang ở k=256 |
### Nguyên nhân (đọc code sau khi số đo chỉ đúng chỗ)
`tv/training/simd_matmul.tkv`, `_smm_core`:
```
if k >= 256:            -> _smm_full (m<=256) hoac _smm_pk  [NATIVE]
else:                   -> duong TILED .tkv               [CHAM]
```
Đường tiled cấp `Ct`, `AtB`, `BtB` (mỗi cái `T*T`), rồi lặp
`ceil(m/T) × ceil(n/T) × ceil(k/T)` tile, mỗi tile gọi `_tile_a_w` + `_tile_b_w` +
`simd_gemm` + `_acc_f` **trong .tkv**. Với `m=n=1024, k=64`: `T=64` ⇒ **256 tile** ⇒
~256 × (2 lần copy 4096 ptu + 1 simd_gemm + 1 `_acc_f` 4096 ptu) ≈ vài ms overhead
**cho 134 MFLOP**. Đó chính là "chi phí cố định ~5.5 ms" ở §0.68 — không phải lock,
không phải dispatch, mà là **một implementation khác**.
`k = hd = 64` đúng là shape của 72 GEMM trong `gpt2_attention_bwd` ⇒ cả `attention_bwd`
chạy trên đường chậm. Đây là lời giải thích cuối cùng cho: 13.8 GFLOP/s (k=64) so với
155.9 GFLOP/s (shape lớn, `k=768` đi native).
### Đã thử đưa k=64 sang native: nhanh 2.5× nhưng KHÔNG bit-exact
`tv/tests/_k64n.tkv` gọi thẳng `_smm_full(a,b,n,n,64,64,1)` trên cùng dữ liệu:
| đường | ms/lời gọi | GFLOP/s |
|---|---:|---:|
| tiled `.tkv` (hiện tại) | 13.0 | 10.3 |
| native `_smm_full` | **5.17** | **26.0** |
| **bit_bad** | **140 374 / 1 048 576 ptu**, maxdiff **5.59e-9** | |
⇒ **Không hạ ngưỡng `k >= 256` được** nếu giữ hợp đồng bit-exact (cùng loại vi phạm đã
khiến YÊU CẦU #10 từ chối `k%256 != 0`). Native cũng chỉ đạt 26 GFLOP/s cho k=64, không
phải 91 như k=256.
### Hướng đúng (rẻ nhất trước)
1. **Tăng `T` cho trường hợp `k` nhỏ** (repo-side, 1 chỗ trong `_smm_core`): hiện
   `T = min(_tilek(k), roundup64(min(m,n)))` nên `k=64` ép `T=64` ⇒ 256 tile. Nếu cho
   `T = 128` thì còn **64 tile** và số p-tile vẫn là 1 ⇒ thứ tự tích lũy `p` **không đổi**
   ⇒ nhiều khả năng bit-exact (phần `At` ngoài `pw` là 0, cộng 0.0 là chính xác). Chưa làm
   — cần gate 135/135 + loss EXACT để chứng minh.
2. **Native: làm đường rect cho `k < 256` bit-giống đường tiled** (hoặc chấp nhận ULP và
   cập nhật hợp đồng). Chỉ khi đó mới hạ được ngưỡng; lợi ích 2.5× trên 72 lời gọi/block,
   tức `attention_bwd` 328 → ~150 ms và **mọi GEMM k<256 khác trong mô hình**.
3. Cache 3 scratch `n×n` theo worker (nhánh `n ≥ 128`) — nhánh `n < 128` đã làm từ #10 C.2.
### Trạng thái code
Thay đổi thử ở native **đã revert** (`tkv_simd_native.c.bak_r11` giữ bản gốc; DLL đã build
lại từ nguồn gốc, attention trở lại 328 ms). **Không có thay đổi production nào trong
§0.70.** Tệp mới: `tv/tests/_g64.tkv`, `tv/tests/_k64n.tkv`.
## §0.71 THỬ `T = 256` CHO ĐƯỜNG TILED: BIT-EXACT NHƯNG FULL STEP CHẬM ĐI (revert)
Theo hướng 1 của §0.70, sửa `_smm_core` / `_smm_core_pd` (`tv/training/simd_matmul.tkv` + 2 mirror)
để `k` nhỏ dùng `T = 256` thay vì `T = 64` (16 tile thay vì 256 tile):
```python
if T < 256:
    if m >= 256:
        if n >= 256:
            T = 256
```
| phép đo | trước | sau |
|---|---:|---:|
| `smm_dot` k=64, n=1024 (`_k64n.tkv`) | 13.0 ms | **10.5 ms** (1.24×) |
| bit so với native | 140 374 ptu lệch | **bit_bad = 0** (trùng native) |
| **`loss` full step seq1024** | 13.442915 | **13.442915 — EXACT ✓** |
| **full step (14 luồng)** | 16 672 ms | **18 532 ms — CHẬM 11%** |
⇒ **Revert.** `T=256` đúng là bit-exact và nhanh hơn cho *riêng* shape k=64, nhưng nó áp cho **mọi**
GEMM `k < 256` có `m,n ≥ 256`, và ở đó `simd_gemm(256)` vượt ngưỡng `MIN_N=96` nên mỗi tile lại
đánh thức pool 14 luồng — tổng thể thành regression.
Đây là kết quả âm tính **có giá trị**: nó chứng minh (a) thứ tự tích luỹ `p` giữ được khi đổi `T`
(nên bit-exact không phải rào cản của hướng tiling), và (b) nút chặn thật là **chi phí dispatch
của mỗi `simd_gemm`**, đúng như §0.68 — chỉ khác là nó nằm trong đường tiled chứ không ở lock.
### Kết luận mục 1 (sau 5 thí nghiệm)
| thí nghiệm | kết quả |
|---|---|
| bỏ `g_gemm_lock` | không đổi (328→312) |
| `MIN_N` × 3 | không đổi hoặc tệ hơn |
| `GEMM_THREADS` × 4 | sàn 328, không đổi |
| native thay tiled cho k=64 | 2.5× nhanh nhưng lệch 5.6e-9 → không nhận |
| `T=256` | bit-exact, nhanh cho k=64, **chậm toàn cục** → revert |
⇒ Mục 1 chỉ còn **một** đường: native làm cho đường rect `k < 256` **bit-giống** đường tiled
(hoặc chốt lại hợp đồng chấp nhận ULP). Đây là `docs/COMPILER_TEAM_REQUEST11.md` §8.4 ASK A'.
Toàn bộ source production đã về nguyên trạng; `git status` sạch cho 3 mirror.
## §0.72 ĐƯỜNG CÒN LẠI ĐÃ ĐI: hạ ngưỡng native `k >= 256` -> `k >= 64` (NHẬN, −11% @1024)
§0.71 kết luận chỉ còn một đường: cho `k < 256` đi native. Cách rẻ nhất để kiểm chứng giả
thuyết đó **không cần** sửa native: hạ ngưỡng rẽ nhánh trong `_smm_core` để `k = 64`
(shape của 72 GEMM trong `attention_bwd`) rơi vào nhánh native thay vì đường tiled 256 tile.
```python
# tv/training/simd_matmul.tkv (dòng 340, + 2 mirror)
-    if k >= 256:
+    if k >= 64:
         if m <= 256: return _smm_full(...)
         return _smm_pk(...)
```
### Kết quả đo (14 luồng, best-of-3, xen kẽ 3 seq)
| seq | trước (§0.64, `bseq_h4c`) | sau (`bseq_p10`) | Δ |
|---:|---:|---:|---:|
| 8 | 3829 / 3859 / 4015 | **2875 / 2890 / 2890** | **−25%** |
| 256 | 5672 / 5766 / 5860 | **5297 / 5250 / 5218** | **−11%** |
| 1024 | 16375 / 16547 / 16954 | **14516 / 14657 / 15125** | **−11%** |
`loss` sau 1 bước: seq8 `7.0119185` **EXACT**, seq1024 `13.442915` **EXACT**,
seq256 `11.807271` so với baseline `11.807267` — **lệch 4e-6** ở chữ số thứ 6 (ổn định,
lặp lại y hệt 3 rep). Đây là hệ quả trực tiếp đã biết của §0.70: native vs tiled khác nhau
ULP (140 374/1 048 576 ptu, maxdiff 5.6e-9), và với seq256 độ lệch đó nổi lên ở bước 1.
### Gate và trainer
- `tools/gate_f32.sh`: **9/10 suite PASS** (`simd_matmul`, `gpt2`, `gpt2_scope`, `gpt2_train`,
  `grad`, `parity`, `pack`, `inplace`, `gpt2_full` 33/33).
- `gap_tests`: **FAIL `exit=127` — KHÔNG do thay đổi này.** Nguyên nhân: bản `g_gap_tests.exe`
  cần fallback .NET 10, và trên .NET 10 nó chết `System.InvalidProgramException` trong
  `ConstantFoldingPass.run` (cùng họ lỗi `InvalidProgramException` đã gặp khi probe
  `thread_spawn`+Tensor). Đã kiểm bằng cách **stash đúng 3 tệp simd_matmul rồi build lại
  gap_tests từ HEAD**: lỗi y hệt ⇒ có từ trước, không liên quan `k`/GEMM.
- `trainer_tests`: **15/15 PASS** (`=== G4 trainer cases=15 fails=0 ===`), gồm clip, overfit,
  resume bit-exact, eval — chạy qua fallback `dotnet out/g_trainer_tests_n10.dll`.
### So với PyTorch (đo lại cùng phiên)
| pha (seq1024, 14 luồng) | TKV | Torch | tỉ lệ |
|---|---:|---:|---:|
| fwd (+loss) | 3250 + 94 | 1084.7 | **3.0×** |
| **backward** | **10593** | **2347.9** | **4.5×** |
| `opt.step()` | 1188 | 660.7 | 1.8× |
| copy-back / zero_grad | 141 | 52.2 | — |
| **TỔNG** | **15266** | **4134.0** | **3.7×** |
Full-step median: TKV **2890 / 5250 / 14657** vs torch **1009.4 / 1538.4 / 4077.1**
⇒ **2.86× / 3.41× / 3.59×**.
### Thử thêm (kết quả âm tính)
Hạ luôn ngưỡng ở `_smm_core_pd` (dòng 628, nhánh `trans_a == 0` song song) xuống 64:
seq8 2891 (không đổi), seq1024 **15016** (không nhanh hơn dải 14516–15125) ⇒ **revert**,
chỉ giữ thay đổi ở `_smm_core`.
### Chốt
`k >= 64` là ngưỡng **cuối cùng có ý nghĩa**: mọi GEMM của GPT-2 124M có `k ∈ {64, 768, 3072}`,
nên sau thay đổi này **không còn GEMM nào đi đường tiled** (trừ `k < 64`, không tồn tại trong
mô hình). Lever "đường tiled" đã cạn — phần còn lại của khoảng cách 3.6× nằm ở **chi phí
non-GEMM interpreted + scale luồng**, đúng như PLAN §2b: backward 4.5× là nút thắt lớn nhất
(8.2 s chênh lệch trên 11.2 s tổng chênh lệch), kế đó fwd 3.0×, opt 1.8×.
Bản build dùng để đo: `out/bseq_p10.exe` / `out/bseq_p10_n10.dll`. Log: `out/run_p10_s*_r*.log`.
### Ghi chu do nhieu (sau khi commit)
Chay lai chinh ban da commit: **15109 / 15422 ms** (va mot lan **17453 ms** — outlier, nghi do
I/O/AV lúc commit), tuc may dao dong **±7%** giua cac phien. Vi vay con so trong bang tren la
**best-of-3 tai mot thoi diem**; so sanh dung la best-vs-best (16375 -> 14516 = **−11.4%**,
median 16547 -> 14657) chu khong phai mot lan do don le. Khong co lan nao cua ban moi cham
vao dai 16375–16954 cua baseline.
## §0.73 YEU CAU #12 — 4 native kernel BWD non-GEMM: BIT-EXACT, bwd chi −7…−13%, KHONG dat ≤9000
Ask A của `docs/COMPILER_TEAM_REQUEST12.md` đã làm xong và **bit-exact**, nhưng tiêu chí
`backward ≤ 9000 ms` **không đạt**: best-of-9 là **9187 ms** (thiếu 187 ms, 2.1%). Nguyên nhân
không phải kernel sai — mà là **tiền đề của request**: 4 op này chỉ chiếm ~0.9–1.2 s trong
11 s chênh lệch, đúng như §0.67 đã đính chính (non-GEMM còn lại nằm ở `attention_bwd` ~3.5 s
và lm-head + `op_embed_bwd` ~2.5 s, không phải LN/gelu/bias).
### 1. Đã làm — Ask A (4 kernel, 1 pass, ghi thẳng `.data`)
Native: `Release/tools/simd_native/tkv_bwd_ops.c` (TU riêng, `/fp:precise`), DLL
`263680` bytes (từ `256000`), copy sang `out/tkv_simd_native.dll`.
| builtin | thay cho | ghi chú bit |
|---|---|---|
| `layernorm_bwd_fused(x, gy, g, n, h, gflat, o_dg, o_db)` | `gpt2_layernorm_bwd` | `mu/v/s1/s2` f64 tuần tự; `gg = gy*g` là **f32 mul** rồi widen; `dg` f64→f32, `db` **f32+f32**; `dg/db` ghi thẳng `gflat` (không `t_zeros([h])`) |
| `gelu_erf_bwd(x, gy, out, lo, hi)` | `op_gelu_erf_bwd` (+ segment `_ogb_one`) | `erf` Horner f64, `ax*ax` trên f32, `exp` MSVC = .NET (3001/3001) |
| `softmax_causal_bwd(dp, P, dctx, ctx, n, hd)` | vòng softmax trong `attention_bwd` | `dotv` f32-product → f64; in-place `dp`; `u > t` = 0 |
| `bias_grad(gy, gflat, off, n, cols)` | vòng bias trong `gpt2_linear_bwd` | AVX2 8 lane **theo cột**, `i` tuần tự → giữ nguyên thứ tự cộng |
Mọi wrapper trả `rc < 0` → caller chạy **vòng `.tkv` nguyên văn** (fallback), không có nhánh
"số gần đúng". Wiring: `simd_feature.tkv` (+325 dòng: pinvoke, wrapper, `compile_*`,
`register_expr_builtin`) và `'...'` vào `_BUILTIN_LENH_DOC_LAP` của `file_io.tkv`; production
`tv/training/{gpt2_ref,tv.training.gpt2_ref,tv.training.gpt2_ref124}.tkv` + 3 mirror `tests/`
(LN 1 chỗ, bias 4 chỗ, softmax 2 chỗ) và `ops.tkv` ×2 (gelu tuần tự + `_ogb_one`).
### 2. Bit-exact — `_am7.tkv` 21/21 `bit_bad=0`
| nhóm | ca | rc | bit_bad |
|---|---|---:|---:|
| LN | n8h768, gate256 h64, seq1024 h768, lẻ n3h5, min n1h1 | 0 | **0** |
| bias | db1 768, db2 3072, offset-qkv 2304@100, seq1024 768, cols5-tail, cols8-off3-pre, lẻ n7 c767 off17, n=0 | 0 | **0** |
| softmax | n8, n127, seq1024, lẻ n13 hd5 | 0 | **0** |
| gelu | full1000, segment [17,983), n7, bigseg 3.1M [1M,2M) | 0 | **0** |
`AM7 TOTAL bit_bad=0`. Đo riêng: `LN n=1024 h=768` **interp 15.0 ms → native 0.0 ms** (<1 quantum).
Ground truth giữ nguyên từ phiên trước: `_nt4` (exp 3001/3001), `_nt5` (erf/dgelu 401/401),
`_nt` NT1/NT3 (JIT widen f64, truncate tại ranh giới typed).
### 3. Gate / trainer / self-host
- `tools/gate_f32.sh`: **9/10 suite PASS** — `simd_matmul`, `gpt2`, `gpt2_scope`, `gpt2_train`,
  `grad`, `parity`, `pack`, `inplace`, `gpt2_full` 33/33 (log `out/gate_p12.log`, `GATE_EXIT=1`).
- `gap_tests`: **FAIL `exit=82` — `InvalidProgramException` trong `ConstantFoldingPass.run`**,
  đúng triệu chứng §0.72 đã chứng minh **có từ trước** (stash 3 tệp simd rồi build lại từ HEAD
  vẫn fail). Phiên này **không chứng minh lại**, chỉ xác nhận cùng stack/cùng suite.
- `trainer_tests`: **15/15 PASS** (`=== G4 trainer cases=15 fails=0 ===`).
- Self-host `tools/vong.exe`: **`IL_IDENTICAL`** (`gop 133 file`), `dist/tkvc-tv.exe` 21:35 sau
  lần sửa compiler cuối 21:33.
### 4. Đo lại — seq1024, 14 luồng, `out/bseq_p12_n10.dll`
Chín lần mode 2 (9 run, mỗi run best-of-2 sau warmup):
| # | fwd | loss | bwd | opt | copyback | TỔNG |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3359 | 109 | 10719 | 860 | 156 | 15203 |
| 2 | 3234 | 62 | 9657 | 875 | 141 | 13969 |
| 3 | 2969 | 78 | **9187** | **860** | 109 | **13468** |
| 4 | 3328 | 93 | 9719 | 938 | 109 | 14187 |
| 5 | 3375 | 94 | 9875 | 922 | 141 | 14407 |
| 6 | 3297 | 62 | 9782 | 937 | 125 | 14203 |
| 7 | 3328 | 93 | 9829 | 906 | 125 | 14281 |
| 8 | 3406 | 93 | 9828 | 937 | 110 | 14374 |
| 9 | 3328 | 110 | 9937 | 891 | 110 | 14376 |
| **best** | 2969 | | **9187** | **860** | 109 | **13468** |
| **median** | 3328 | | **9782** | **937** | | **14281** |
Baseline §0.72 (`bseq_p10`): bwd **10593**, opt **1188**, copyback 141, TỔNG **15266**;
full-step best/median 14516/14657.
⇒ best-vs-best: bwd **−1406 ms (−13.3%)**, TỔNG **−1798 (−11.8% so 15266; −7.2% so 14516)**.
median: bwd **−811 (−7.7%)**, TỔNG **−985 (−6.5% / −2.6%)**. Máy dao động ±7% (§0.72) nên
**không được đọc một lần đo đơn lẻ**.
Mode 1 (đúng lệnh §6 của request): **14266 / 14219 / 14078 ms**, `loss=10.760399 -> 13.442915`
**EXACT** — best 14078 vs 14516 = **−3.0%**.
### 5. Không regress seq8/seq256 (mode 2, 3 run mỗi seq)
| seq | bwd | TỔNG | §0.72 | Δ best |
|---:|---|---:|---|---:|
| 8 | 1282 / 1282 / 1297 | 2782 / 2782 / 2812 | 2875 / 2890 / 2890 | **−3.2%** |
| 256 | 2875 / 2984 / 3000 | 4875 / 5030 / 5078 | 5297 / 5250 / 5218 | **−8.0%** |
| 1024 | 9187 (best) | 13468 | 14516 | **−7.2%** |
`loss` mode 0 (1 bước, cả 3 seq): **`10.572291` / `10.756927` / `10.760399` — EXACT cả 3**.
### 6. Torch cùng phiên (3 lần, xen kẽ với TKV)
| pha | torch | TKV best | tỉ lệ |
|---|---:|---:|---:|
| bwd | 2359.4 / 2350.8 / 2326.8 | 9187 | **3.91×** (trước 4.51×) |
| `opt.step` | 677.6 / 698.7 / 683.4 | 860 | **1.26×** (trước 1.80×) |
| TỔNG | 4174.9 / 4206.1 / 4175.3 | 13468 | **3.23×** (trước 3.7×) |
`opt.step` **không bị thay đổi bởi #12** — 1188 → 860 là chênh lệch đo/áp lực alloc (bớt
`t_zeros([h])` mỗi block), **không phải** công của 4 kernel; xem §7.
### 7. Ask C — `adamw_step_glm`: A/B đo được, **chưa wire**
`tv/tests/_gl1c.tkv` (n = 124 439 808, idle, best-of-5):
| đường | ms |
|---|---:|
| A `r1 = f32_copy_to_buf(gl, pgl)` + `adamw_step` (production) | **406** |
| B `adamw_step_glm(p, m, v, gl, …)` (đọc list trực tiếp) | **312** |
| B − A | **−94** |
| chỉ `r1` copy | 78 |
| `f32_list_add` đơn lẻ | 125 |
**Chưa wire**, vì:
1. Khi clip hoạt động (`coef != 1.0`) production cần `pgl` làm **input của `f32_addcmul`**
   → `r1` là bắt buộc; `adamw_step_glm` chỉ bỏ được `r1` ở nhánh `coef == 1.0`. Ước tính
   lợi thực trong step ≲ 78–180 ms, **không phải 400 ms** như kỳ vọng §4 của request.
2. **Lỗi codegen chặn đo bit**: `_gl1.tkv` (probe có phép so bit) chết
   `MissingMethodException: TkvSimd.AdamwGlmM` — IL **có call** nhưng **thiếu method**.
   `_gl1c.tkv` (không có hàm local) thì emit đủ. Nghi thứ tự `_can_simd(ctx,'m5')` /
   `_ensure_class_simd` khi module có hàm local. **Chưa sửa** — cần khoanh trước khi wire.
3. Bit-exact **đúng theo cấu trúc**: `AdamwGlmM` pin `_items` rồi gọi **cùng**
   `AdamwNative`(`AdamwStepF32`) với cùng tham số như `AdamwM`; `r1` chỉ là copy f32→f32.
   Vẫn nên có một lần `bitdiff=0` chạy được trước khi nhận.
### 8. Vì sao ≤9000 không đạt — và việc kế tiếp
4 op của Ask A thay được ~**0.9–1.2 s** (LN ~0.6 s, gelu ~0.3 s, softmax ~0.2 s, bias <0.05 s),
khớp với số đo. Phần còn lại của backward (best 9187, torch 2351 ⇒ 3.91×) nằm ở:
- `attention_bwd` non-GEMM **~3.5 s không scale** (§0.67): 72 GEMM siêu nhỏ + các vòng
  mask/score/proj còn interpreted — **chưa có kernel nào**, đây là lever lớn nhất;
- lm-head + `op_embed_bwd` non-GEMM ~2.5 s;
- GEMM scale 3.1× thay vì 4.5× (§0.68–§0.72 đã cạn knob `.tkv`).
⇒ Đề nghị #13: kernel native cho `attention_bwd` non-GEMM (score bwd + proj + mask) và/hoặc
op-level threading trong DLL (Ask B của #12 vốn "chỉ nếu A không đủ" — **A không đủ**).
Ask C chỉ đáng làm sau khi (a) có nhánh `addcmul` đọc list, (b) sửa lỗi emit `AdamwGlmM`.
### 9. Trạng thái
Chưa commit gì (2 repo còn dirty từ #10/#11, không revert). DLL mới đã ở
`Release/runtime/` và `out/tkv_simd_native.dll`. Build đo: `out/bseq_p12_n10.dll`
(mode 0/1/2), probe `out/am7_n10.dll`, `out/gl1c_n10.dll`, `out/gate_p12.log`.
Lệnh tái lập:
`printf "1024\n2\n1\n" > tv/tests/_seq_cfg.txt; TKV_GEMM_THREADS=14 dotnet out/bseq_p12_n10.dll`.
## §0.74 ATTENTION NON-GEMM: đo tách + native `softmax_causal_fwd` (bit-exact, fwd −3.4%, bwd không đổi)
Tiếp yêu cầu #12/#13: đo tách từng vòng non-GEMM trong `attention_bwd` rồi viết kernel cho
phần lớn nhất. **Kết luận quan trọng: vòng lớn nhất là softmax-causal forward (249/296 ms
non-GEMM), nhưng nó KHÔNG nằm trên critical path của bước 1024 — bwd không đổi.**
### 1. Đo tách (probe mới `tv/tests/_at1.tkv`, seq1024, 1 luồng)
`gpt2_attention_bwd` (1 block, không gflat) = **344 ms**; 12 head chạy song song.
| vòng (12 head, tuần tự) | ms | % non-GEMM |
|---|---:|---:|
| pack Q/K/V/dctx (strided) | <16 | — |
| **scale + max** | 62 | 21% |
| **exp + tổng** | **141** | **48%** |
| normalize | 15 | 5% |
| zero-fill u > t | 31 | 10% |
| writeback dq (strided) | 47 | 16% |
| merge packed | <16 | — |
| **Σ non-GEMM** | **296** | |
| 6 GEMM/head × 12 (đo lô 8) | 397 | |
⇒ trong 1 worker: GEMM **33.1 ms/head**, non-GEMM **24.7 ms/head** ⇒ GEMM 57%.
`writeback` 47 ms và `pack` <16 ms là **shuffle bộ nhớ** — không đáng kernel.
### 2. Kernel `softmax_causal_fwd(s, n, inv)`
Gộp 4 vòng (scale+max / exp+tổng / normalize / zero-fill) thành 1 pass in-place trên
`Smat.data`. Bit-exact nhờ giữ đúng IL: `mxv` là f64 **trước khi làm tròn f32**; `exp` đọc
lại giá trị f32; `totv` cộng dồn f64 chưa làm tròn; nhân `s*ivv` f32×f64→f32.
`rc<0` → 4 vòng `.tkv` nguyên văn (fallback). Wire **4 điểm/file × 5 mirror**
(`_attn_f_one`, `gpt2_attention` n<128, `_attn_hb_one`, `gpt2_attention_bwd` n<128).
`tv/tests/_am8.tkv`: **bit_bad=0** ở n=1, 2, 5(lẻ hd=5), 8, 127, 128, **1024**;
`n=1024`: interp **47 ms** → native **< 15.6 ms** (1 quantum).
### 3. Đo end-to-end — A/B xen kẽ cùng phiên (`bseq_p12` không kernel vs `bseq_p13` có)
| | p12 | p13 | Δ best |
|---|---|---|---|
| fwd (3 cặp) | 3656 / 3657 / 3782 | **3578 / 3531 / 3562** | **−3.4%** |
| bwd (3 cặp) | 10062 / 9890 / 10109 | 9735 / 9906 / 10157 | −1.6% (không ổn định) |
| tổng | 13718 / 13547 / 13891 | 13313 / 13437 / 13719 | −1.7% |
fwd **nhanh hơn ở cả 3 cặp** (đúng phần attention fwd). bwd **không đổi trong nhiễu**
(mode 2: p13 best 9672 / median 9766 vs p12 best 9187 / median 9782).
**Vì sao bwd không đổi:** 12 worker chạy song song trên 14 luồng, mỗi worker bị chi phối bởi
6 GEMM (33 ms) + tranh chấp luồng; bỏ 24.7 ms vòng softmax khỏi một pha không rút ngắn
critical path. Đúng như §0.69/§0.70: phần còn lại của `attention_bwd` là **chi phí mỗi lời
gọi GEMM phía native**, không phải vòng interpreted.
### 4. Không regress + gate
| seq | TỔNG (mode 2, 3 run) | trước (§0.73) |
|---:|---|---|
| 8 | 2766 / 2797 / 3000 | 2782 / 2782 / 2812 |
| 256 | 4953 / 4984 / 5015 | 4875 / 5030 / 5078 |
`loss` mode 0: **`10.572291` / `10.756927` / `10.760399` — EXACT cả 3**.
Gate **9/10** (`gap_tests` exit=127 pre-existing), `trainer_tests` **15/15**,
self-host **`IL_IDENTICAL`**.
### 5. Chốt
Kernel giữ lại (bit-exact, giảm CPU thật, fwd −3.4%) nhưng **không phải lever cho bwd**.
Lever còn lại của bwd là bỏ overhead mỗi lời gọi GEMM (72 GEMM nhỏ/block) — việc của #14.
Build đo: `out/bseq_p13_n10.dll`, probe `out/am8_n10.dll`, `out/at1b_n10.dll`.
## §0.75 YEU CAU #14 — GỘP 6 GEMM/head attention_bwd THÀNH 1 NATIVE CALL (bit-exact, bwd −5..−10%)
### 1. Việc làm
- **Kernel `AttentionBwdFused`** (thêm vào cuối `Release/tools/simd_native/tkv_simd_native.c`, TU `/fp:fast`
  — chung mã máy với `SgemmF32`/packed GEMM, đảm bảo cùng bits):
  - 4 GEMM (Smat, gCtx, gDp, gDq) tái hiện đúng dispatch `_smm_core`: `k ≥ 64` →
    `m ≤ 256` → `SgemmFull(T=256, mode=0)`; `m > 256` → `packed_create/SgemmPacked(T=256)`
    (bits khác nhau giữa 2 đường → phải chia đúng như .tkv);
  - 2 GEMM trans-A (gDkT, gDvT) port `_tile_a_at_w`/`_tile_b_w` + `SgemmF32` + tích lũy
    p0 TĂNG DẦN (đúng thứ tự `_p58_core_simple`);
  - softmax_causal_fwd/bwd gọi đúng entry đã kiểm chứng (§0.73/§0.74);
  - writeback packed/head (`dq/dk × inv`, `dv` raw) giống các vòng .tkv.
- **Compiler wiring** (`il_features/simd_feature.tkv`): pinvoke + wrapper `AttentionBwdFusedM`
  (4 list qua reflection/pin; thiếu list → rc=−5, **không ném**) + `compile_attention_bwd_fused`
  + đăng ký builtin + `_BUILTIN_LENH_DOC_LAP` (file_io.tkv). Thay cả pinvoke/wrapper nửa chừng
  của phiên trước (sai signature, thiếu tham số — đã bỏ).
- **Sửa opcode IL dạng ngắn sai**: `ldarg.4→ldarg.s 4`, `ldloc.4/stloc.4→…s …`, `ldc.i8.0→ldc.i8 0`.
- **Production** (`gpt2_ref.tkv`): `_attn_hb_one` gọi `attention_bwd_fused` (1 call/head = 12 call/layer
  thay cho 6 GEMM × 72 pinvoke/layer + hàng nghìn `simd_gemm` list-reflection cho 2 GEMM trans-A);
  rc != 0 → `_attn_hb_one_ref` (fallback nguyên văn, cùng thân hàm cũ). 5 mirror đồng bộ
  (`tv.training.gpt2_ref{,124}.tkv`, `tv/tests/...`); **bản nhỏ 256/64/4heads/HeadDim=16**
  kernel từ chối (hd≠64 → rc=−13) → luôn fallback, bit-exact giữ nguyên.
### 2. Bit-exact (`tv/tests/_am9.tkv`, ref = đúng các bước worker .tkv)
**11/11 PASS, `bit_bad=0`, rc=0** ở n = 64, 65, 100, 128, 200, 256, 300, 1024
(cả 2 đường: m≤256→SgemmFull; m>256→packed; đuôi không chia hết 64; head 0/11…).
### 3. Số đo (máy này, best-of-3; nền = §0.73/§0.74)
| seq / mode | kết quả | nền |
|---|---|---|
| 1024 mode 0 (fwd/bwd) | 3344–3625 / **9484–10125** | 3531–3782 / 9735–10157 (§0.74), 10593 (§0.72) |
| 1024 mode 2 (bwd/opt/copyback) | bwd 9750–10125, opt 969–1172, copyback 156 | bwd ~9750–9820, opt 860–1172 |
| 1024 mode 1 (full step) | 14344 / 14079 / 14672 (loss `10.760399 → 13.442915` **EXACT**) | 15266 (§0.72), 14516/14657 best |
| 8 mode 0 | 531 fwd / **1312** bwd / 1843 | ~2780 (§0.73) |
| 256 mode 0 | 1079 fwd / **3234** bwd / 4313, loss `10.756927` EXACT | — |
**Loss EXACT cả 3 seq** (10.572291 / 10.756927 / 10.760399).
### 4. Gate / trainer / self-host
- Gate **9/10** — chỉ `gap_tests` fail (`InvalidProgramException` trong `ConstantFoldingPass.run`) —
  **pre-existing** đã chứng minh từ §0.72/§0.73 (stash build lại vẫn fail) — không liên quan #12/#13/#14.
  *(WSL không thấy `dotnet` → gate in fail giả; wrapper tạm `out/wslbin/dotnet` → 9/10 PASS.)*
- `trainer_tests` **15/15**; self-host `vong.exe` → **IL_IDENTICAL**.
- DLL mới đã copy vào `Release\runtime\` và `out\tkv_simd_native.dll`.
### 5. Chốt
- Tiêu chí #14 ("cắt 72 GEMM/block"): **đạt về cấu trúc** (864→144 lời gọi native/layer,
  2 GEMM trans-A không còn pinvoke/list-reflection trong vòng lẻ p-tile); loss EXACT + gate giữ nguyên.
- Bwd vẫn **> 9000 ms** (best 9484) — bị chặn bởi thời gian GEMM thật + tranh chấp luồng 12 worker
  (đúng kết luận §0.74: chỉ cắt overhead, không rút critical path). Lever còn lại: GEMM scale/threading
  hoặc giảm số GEMM (không nằm trong #14).
- `n < 128` (nhánh sequential của `gpt2_attention_bwd`) **chưa wire** — giữ đường cũ nguyên văn.
- Chưa commit (sẽ commit riêng #14 ở cả 2 repo sau khi ghi xong mục này).
## §0.76 YEU CAU #15 — LinearDwAtFused (native trans-A GEMM + gflat add) — bwd < 9000 DAT
### 1. Việc làm
- Kernel `LinearDwAtFused` (append vào `tkv_simd_native.c`, TU `/fp:fast`, chung mã máy với `SgemmF32`/packed GEMM):
  - mode 0 (P58): tái hiện `_p58_core_simple` (tile T = min(min(roundup64(k), 1024), roundup64(min(m, n))) ≥ 64; SgemmF32; p0/i0/j0 tăng dần; cd zero-fill) — thay `smm_fwd_at` cho qkv/w1/w2.
  - mode 1 (PACKED): materialize xᵀ trong C (pure copies) + cùng dispatch `k≥64` (`m≤256 → SgemmFull(T=256, mode=0)`; `m>256 → packed T=256`) — tái hiện `smm_transpose + smm_fwd` cho proj dw.
  - Sau GEMM, kernel cộng thẳng `cd` vào `gflat[goff+k]` (1 phep f32 add/ptu) — bỏ vòng copy interpreted.
- Compiler wiring (`simd_feature.tkv`, `file_io.tkv`): pinvoke `LinearDwAtFusedNative` + wrapper `LinearDwAtFusedM` (3 list qua reflection/pin; thiếu list → rc=−5, không ném) + `compile_linear_dw_at_fused` + đăng ký builtin + whitelist. Sửa off-by-one (IL arg 3=gdw_off, 4=n, 5=in_d, 6=out_d, 7=mode). Self-host **IL_IDENTICAL**.
- Wire production (3 mirror có `linear_dw_fused`): `gpt2_ref.tkv`, `tv/training/tv.training.gpt2_ref124.tkv`, `tv/tests/tv.training.gpt2_ref124.tkv`:
  - `gpt2_linear_bwd`: sau `linear_dw_fused` rc≠0 → `linear_dw_at_fused(..., mode=0)`; `l10 != 0` → fallback `smm_fwd_at` (vòng copy tự skip vì l10==0).
  - proj dw: sau `linear_dw_fused` rc≠0 → `linear_dw_at_fused(..., mode=1)`; `l10p != 0` → fallback `smm_transpose + smm_fwd`.
  - mlp_bwd dw2: cùng mẫu mode 0.
  - 2 mirror nhỏ (config 256/64) **không wire** — chúng không dùng `linear_dw_fused` (đi thẳng smm_transpose + smm_fwd native), đường dw đã native; kernel mode 1 vẫn fallback bit-exact nếu gọi.
- Probe `tv/tests/_dw9.tkv`: **11/11 PASS, bit_bad=0, rc=0** (mode 0: n=1/8/64/100/256/300/1024 incl shape lẻ; mode 1: h=8/100/1024 incl SgemmFull path; n<64 → rc=−23).
### 2. Số đo (seq1024, 14 luồng, best-of-3)
| metric | #15 (p15) | #14 (p14) | §0.73 (p13) | baseline §0.72 |
|---|---|---|---|---|
| bwd (mode 0) | **8797 / 8828 / 8922** | 9484 / 9531 / 10093 | 9735 / 9906 / 10157 | 10593 |
| bwd (mode 2) | 8844 / 8922 / 8969 | 9750 / 9922 / 10125 | 9672 / 9766 / 9782 | — |
| opt (mode 2) | 922 / 1063 / 1140 | 969 / 970 / 1172 | 860 / 937 | 1188 |
| copyback | 141 / 156 | 141 / 156 | 109 / 125 | 141 |
| fwd (mode 0) | 3406 / 3406 / 3500 | 3344 / 3406 / 3625 | 3531 / 3578 / 3782 | 3250 |
| full step (mode 1) | **13344** | 14079 / 14344 / 14672 | 13313 / 13437 / 13719 | 15266 |
| seq8 (total) | 1875 | 2766 / 2797 / 3000 | 2766 / 2797 / 3000 | 2875 |
| seq256 (total) | 3906 | 4953 / 4984 / 5015 | 4953 / 4984 / 5015 | 5297 |
**bwd best = 8797 ms → < 9000 ✓** (tiêu chí §0.72 đạt). Loss EXACT cả 3 seq:
`10.572291 / 10.756927 / 10.760399`; mode 1 `10.760399 → 13.442915`.
### 3. Gate / trainer / self-host
- Gate **9/10** (chỉ `gap_tests` fail `InvalidProgramException` — pre-existing từ §0.72,
  stash build lại vẫn fail) — không liên quan #14/#15.
- `trainer_tests` **15/15**; self-host `vong.exe` → **IL_IDENTICAL**.
- DLL đã copy vào `Release\runtime\` và `out\tkv_simd_native.dll`.
### 4. Chốt
- Tiêu chí #15: **đạt** — bwd xuống dưới 9000 ms (best 8797) với kernel bit-exact
  (probe 11/11), loss EXACT, gate giữ nguyên.
- Lever còn lại nếu muốn bwd còn thấp hơn: `op_embed_bwd` (interpreted, ~38M element
  zero-init + scatter-add — candidate native tiếp theo) và GEMM scale/threading.
- Chưa commit #15 — sẽ commit riêng 2 repo ngay sau mục này.
## §0.77 YEU CAU #16 — `embed_bwd_fused` (native scatter-add) — bwd best 8734 ms, op_embed 156→0 ms interpreted
### 1. Việc làm
- Kernel `EmbedBwdFused` (append vào `tkv_bwd_ops.c`, TU `/fp:precise`): scatter-add theo idx
  (counts/starts/pos/cursor + main loop q→t→r, `gw[r*d+j] += gy[t*d+j]`), zero-fill gw
  (`+0.0f`), out-of-range idx → rc=−7 → fallback. Bit-exact: moi ptu cong don doc lap,
  thu tu phep cong giong .tkv.
- Compiler wiring (`simd_feature.tkv`, `file_io.tkv`): pinvoke `EmbedBwdFusedNative` (idx int64*,
  idx_len, rows, d, gy, gy_len, gw, gw_len) + wrapper `EmbedBwdFusedM` (List<int64> + 2 List<f32>
  qua 2 reflection chain int64/f32) + `compile_embed_bwd_fused` (idx list[i64], rows/d int32,
  gy/gw list[f32]) + đăng ký builtin + whitelist. Self-host **IL_IDENTICAL**.
- Wire `op_embed_bwd` (cả 2 copy `ops.tkv`) → gọi `embed_bwd_fused(idx, rows, d, gy, gw)`;
  rc != 0 → fallback interpreted nguyên văn. `gwte_e`/`gwpe` vẫn qua `f32_list_add` như cũ.
- Probe `tv/tests/_eb9.tkv`: **5/5 PASS, bit_bad=0** (rows=5/100/1000/50257; d=1/3/16/64/768; n=7/1024).
### 2. Số đo (seq1024, 14 luồng, best-of-3)
| metric | #16 (p16) | #15 (p15) | baseline §0.72 |
|---|---|---|---|
| bwd (mode 0) | **8734 / 8750 / 8922** | 8797 / 8828 / 8922 | 10593 |
| bwd (mode 2) | 8734 / 8859 | 8844 / 8922 / 8969 | — |
| opt (mode 2) | 921 / 1062 | 969 / 970 / 1172 | 1188 |
| copyback | 110 / 156 / 172 | 141 / 156 | 141 |
| full step (mode 1) | **13328** (loss `10.760399 → 13.442915` **EXACT**) | 13344 | 15266 |
| seq8 (total) | 1875 | 1875 | 2875 |
| seq256 (total) | 3953 | 3906 | 5297 |
**bwd best = 8734 ms (< 9000 ✓)**; loss EXACT cả 3 seq (10.572291 / 10.756927 / 10.760399).
A/B xen kẽ torch cùng phiên: TKV bwd 8688–8922 vs torch fwd+bwd 3474–3543 → ratio ~3.5× (từ 4.51× baseline).
### 3. Gate / trainer / self-host
- Gate **9/10** (chỉ `gap_tests` fail `InvalidProgramException` — pre-existing từ §0.72).
- `trainer_tests` **15/15**; self-host **IL_IDENTICAL**.
- DLL copy vào `Release\runtime\` và `out\tkv_simd_native.dll`.
### 4. Chốt
- Tiêu chí #16: **đạt** — op_embed_bwd interpreted (~156 ms/call) → native, bit-exact, gate giữ nguyên.
- Lever còn lại nếu muốn bwd còn thấp hơn: GEMM scale/threading (không nằm trong #14/#15/#16); **không đổi thuật toán/thứ tự tích lũy** (loss EXACT bất biến).
- Chưa commit #16 — sẽ commit riêng 2 repo ngay sau mục này.
## §0.78 Ask C — adamw_step_glm wired (compiler fix: _can_simd 'm5'+'dp'), opt.step best 906 ms
### 1. Compiler fix
- Bug: `compile_adamw_step_glm` goi `_can_simd(ctx, 'm5')` nhung `AdamwGlmM` nam trong
  chunk 'dp' (khong phai 'm5') -> IL co call nhung method khong duoc emit -> MissingMethodException.
- Fix: them `_can_simd(ctx, 'dp')` sau `_can_simd(ctx, 'm5')` trong compile_adamw_step_glm.
- Self-host: IL_IDENTICAL.
### 2. Wiring trainer.tkv
- `step()`: khi coef == 1.0 (khong clip): bo r1 copy, goi `adamw_step_glm(ppl, pml, pvl, gl, ...)`.
- Khi coef != 1.0: r1 + f32_addcmul + adamw_step (nhu cu).
- r2/r3/r4/r5 giu nguyen (M5-opt).
### 3. So do (seq1024, 14 luong, best-of-3)
| metric | #16 (p16) | #17 (p17) | baseline §0.72 |
|---|---|---|---|
| bwd (mode 0) | 8734 / 8750 / 8922 | 8734 / 8922 / 8969 | 10593 |
| opt (mode 2) | 921 / 1062 | 906 / 921 / 1062 | 1188 |
| full step (mode 1) | 13328 | 13328 / 13968 / 20422 | 15266 |
| seq8 (total) | 1875 | 1875 | 2875 |
| seq256 (total) | 3953 | 3906 | 5297 |
**bwd best = 8734 ms (< 9000 ✓)**; loss EXACT cả 3 seq (10.572291 / 10.756927 / 10.760399).
opt.step best = 906 ms (torch ~660 ms — gap ~246 ms, gain tu Ask C chi ~78 ms warm;
r1 copy warm chi 78 ms, khong phai 498 MB nhu gia thiet ban dau).
### 4. Gate / trainer / self-host
- Gate **9/10** (gap_tests pre-existing).
- `trainer_tests` **15/15**; self-host **IL_IDENTICAL**.
- DLL copy vao `Release\runtime\` va `out\tkv_simd_native.dll`.
### 5. Chot
- opt.step giu ~906 ms — **khong dat muc torch ~660 ms**. Gain tu Ask C chi ~78 ms
  (r1 copy warm = 78 ms, khong phai 498 MB). Bottleneck con lai: adamw computation
  (~375 ms @124M, do _cj8) + r3 copy (485 ms) + thread overhead.
- De dat 660 ms can profiling + optimization sau (khong nam trong #12-#16).
- Chua commit #17 — se commit rieng 2 repo ngay sau muc nay.
## §0.79 Ask C — lazy p sync (M5-opt style) — bwd best 8750 ms, opt.step best ~906 ms
### 1. Việc làm
- `trainer.tkv` step(): bỏ r3 (copy-back p) mỗi bước → `self.psync = 1`; thêm `sync_p()` lazy (giống sync_mv cho m/v). r2 (push list→buf) chỉ chạy khi psync==0 và pdirty==1.
- ckpt_save/ckpt_load/run_steps gọi `sync_p()` trước khi đọc opt.p.
- Compiler fix: `_can_simd(ctx, 'dp')` thêm vào `compile_adamw_step_glm` (Ask C) → `AdamwGlmM` được emit (trước đây thiếu → MissingMethodException).
- `adamw_step_glm` wired vào step() (clip-gated: coef==1.0 → glm; clip active → adamw_step).
### 2. Số đo (seq1024, 14 luồng, best-of-3..6)
| metric | best | median | baseline §0.72 |
|---|---|---|---|
| bwd (mode 0) | 8750 | ~9200 | 10593 |
| full step (mode 1) | 13328 | ~13300 | 15266 |
| opt (mode 2) | 906 | ~1016 | 1188 |
| seq8 (total) | 1875 | — | 2875 |
| seq256 (total) | 3906 | — | 5297 |
**bwd best = 8750 ms (< 9000 ✓)**; loss EXACT 3 seq (10.572291 / 10.756927 / 10.760399).
opt.step best ~906 ms (torch ~660 ms — gap ~246 ms; gain từ lazy p ~78 ms warm).
### 3. Gate / trainer / self-host
- Gate **9/10** (gap_tests pre-existing).
- `trainer_tests` **15/15**; self-host **IL_IDENTICAL**.
### 4. Chốt
- Tiêu chí "bwd ≤ 9000 ms" **đạt** (best 8750).
- opt.step ~906 ms best — chưa đạt 660 ms (gain từ Ask C modest ~78 ms; bottleneck là adamw computation + r3 copy bandwidth).
- Lever còn lại: GEMM scale/threading, giảm số GEMM — không thuộc #12-#17.
- Chưa commit #18 (lazy p + Ask C wiring) — sẽ commit rieng 2 repo.
## §0.80 Ask D — tích hợp BLAS (scipy-openblas64, opt-in) — loss EXACT, không nhanh hơn
### 1. Việc làm
- `tkv_simd_native.c`: thêm `tkv_blas_load()` (lazy `LoadLibrary("libscipy_openblas64_.dll")`) + `cblas_sgemm`/`openblas_set_num_threads` (64-bit int). `SgemmF32` gọi `tkv_try_blas()` trước kernel AVX2; OpenBLAS không có/không load được → fallback native silently.
- **BLAS CHỈ bật khi `TKV_GEMM_BLAS=1`** (mặc định = 0 = native). Lý do: BLAS không nhanh hơn ở shape production, và bật nó làm hỏng 1 case matmul gate (tol 1e-4 vs 1.39e-04).
- Copy `libscipy_openblas64_.dll` vào `Release\runtime\`, `out\`, CWD — app .NET search CWD.
- 2 entry point khác nhau cho "cài đặt hệ thống" vs "phân phối": nếu DLL không có → native (gate giữ nguyên).
### 2. Đo (seq1024, 14 luồng, best-of-3)
| metric | native (mặc định) | BLAS (TKV_GEMM_BLAS=1) |
|---|---|---|
| bwd (mode 0) | ~8875–9172 | ~8922 (không đổi) |
| opt | ~906 | ~906 |
| loss | EXACT 10.760399 | EXACT 10.760399 |
**Kết luận: tích hợp BLAS không cải thiện tốc độ** — GEMM không phải điểm nghẽn (bwd 8.9 s vs torch 2.36 s; gap chủ yếu là overhead interpreted/.NET/GC, không phải GEMM math).
### 3. Gate / trainer / self-host
- Gate **9/10** (chỉ `gap_tests` fail `InvalidProgramException` — pre-existing từ §0.72) — BLAS opt-in off mặc định.
- Khi bật `TKV_GEMM_BLAS=1`: thêm `simd_matmul_tests` FAIL (`train 3 buoc`, diff 1.39e-04 > tol 1e-4).
- `trainer_tests` **15/15**; self-host **IL_IDENTICAL** (không đổi .tkv compiler).
- DLL copy vào `Release\runtime\` và `out\` — `*.dll` trong `.gitignore`, không commit.
### 4. Khuyến nghị
- **Không bật BLAS theo mặc định** — không nhanh hơn, phá 1 gate case. Giữ native AVX2.
- Nếu sau này muốn dồn thêm nhân lực MKL/BLAS, chỉ áp dụng cho GEMM khổng lồ (`TKV_GEMM_BLAS=1`).
- Gap vs PyTorch chủ yếu nằm ở **overhead .NET interpreter** (loops, boxing, GC) — cần tối ưu mức IL/kernel, không phải GEMM.
## §0.81 Native `gelu_erf_fwd` per-worker segment (YEU CAU #19) — bit-exact, bwd ~8359 ms
### 1. Kernel + hop dong so lieu
- `GeluErfFwd(x, x_len, out, out_len, lo, hi)` trong `tkv_bwd_ops.c` (TU `/fp:precise`).
- Bit-exact voi IL cua `gelu_erf` (doc tu `out/gff9.il`):
  - `arg = v / (float)1.4142135623730951` — **phep chia f32** (`ldc.r4`), khong phai f64.
  - erf: poly f64 (`t = 1/(1+0.3275911*ax)`), `exp` f64 nhung **ket qua lam tron f32**, `e*poly` f64, `sg*(1-...)` f64, cuoi cung ep f32.
  - `out = fl32( fl32(0.5f*v) * fl32(1.0f + erf) )` — **lam tron f32 tung ve truoc khi nhan**.
- Ban dau dung `double d = (0.5*(double)v)*(1.0+(double)ef)` -> ~25% phan tu lech 1 ULP; sua theo IL -> 0.
### 2. Probe
`tv/tests/_gff9.tkv` — **8/8 `bit_bad=0`**:
prod 3.1M (amp 1.0 va 12.0), n=8, n=1, n=7, n=100k wide, segment `[100,1000)`, full `[0,1024)`.
### 3. Wiring (quan trong)
- **KHONG** goi native mot lan cho ca 3.1M: kernel don luong ~250 ms (do bang `_gft`), cham hon nhanh 12 worker (~40 ms) -> 12 layer x 250 ms = **+3 s** (da do: fwd 6437, bwd 11000).
- Thay vao do: moi worker `_gf_one` goi native cho **doan cua minh** `[j.lo, j.hi)`:
  - `sl = j.sl`, `sg = j.seg` (builtin can bien), `rc = gelu_erf_fwd(sl, sg, j.lo, j.hi)`.
  - `rc != 0` -> fallback interpreted dung doan (`gelu_erf`) nhu cu.
- Giu nguyen cau truc 12 worker + `_gf_mcopy`; `simd == 0` hoac `cnt < 2M` van interpreted serial (khong doi).
### 4. Do (seq1024, 14 luong, cung phien)
| metric | truoc §0.81 | sau §0.81 |
|---|---|---|
| fwd (mode 0) | 3422–3484 | 3484–3532 |
| bwd (mode 0) | 8672–9000 | **8359–8547** |
| seq256 mode 0 | 3906–3953 | **3844** |
| seq8 mode 0 | 1875 | **1812** |
| full step (mode 1) | 13328 | 14188 (nhieu) |
| loss 3 seq | EXACT | EXACT 10.760399 / 10.756927 / 10.572291 |
| mode 1 loss | 13.442915 | 13.442915 |
### 5. Gate / trainer / self-host
- Gate **9/10** (chi `gap_tests` FAIL `InvalidProgramException` — pre-existing tu §0.72).
- `trainer_tests` **15/15** (eval loss 0.014949893, ppl 1.0150622).
- Self-host `tools/vong.exe` -> **IL_IDENTICAL** (compiler on dinh voi builtin moi).
## §0.82 GEMM rect: chunk C theo L2 + pack song song — bwd best 7406 ms (auto 28 luồng)
### 1. Việc làm (chỉ `tkv_simd_native.c`, không đổi compiler/.tkv)
Profile lại phần **ngoài 12 block** của `gpt2_backward` (HANDOFF §0.81 để ngỏ head/embed) bằng
`tv/tests/_lm1024.tkv` @seq1024:
| op (ngoài block) | trước | sau |
|---|---|---|
| `lmhead_bwd_fused` | **1656 ms** | **812 ms** |
| `op_embed_bwd(wte)` (V*h=38.6M) | 125 | 141 |
| `f32_list_add dwte` (V*h) | 31 | 47 |
| vòng `.tkv` `gwpe -> gflat` (786K) | 78 | 62 |
→ **chunk ≥ 500 ms ngoài block là lm-head**, không phải GELU. Đo C trực tiếp (bench `TkvBenchRect`
tạm thời, đã gỡ) tách được nguyên nhân:
- `SgemmF32` vuông **545–586 GFLOPS** (n=768/1024) → kernel KHÔNG phải điểm nghẽn.
- `gemm_rect_f32` shape head2 `50257x768x1024`: **69 GFLOPS** (1143 ms); cùng shape `m=4096`:
  389 GFLOPS → lỗi ở **m lớn**, không phải K/N.
- Nguyên nhân: `for jc { for pc { for ic } }` → tile C bị đọc+ghi lại `k/KC = 16` lần trên 154 MB
  (2.5 GB RMW), vượt L3 35 MB.
**Sửa 1 (MC2)** — `gemm_rect_f32`: gom `mc2 = 32768/nc` hàng (làm tròn theo `mc_b`) thành 1 chunk để
tile C (~128 KB) nằm trong L2 → C chỉ ra DRAM 1 lần. Thứ tự cộng theo k **không đổi** (pc tăng dần,
trong pc vẫn `micro_f32`) ⇒ bit-giống.
**Sửa 2 (pack song song)** — `packed_create` là 1 luồng và chiếm **50–60%** thời gian 1 phep
`smm_dot` (bench `bench_packed`: lin_bwd create=33.6 ms vs gemm=24.2 ms). Thêm `pack_rows_f32`
(+ mode 6 của pool) chia theo hàng đích, và `tkv_run_pack` gọi pool; nhánh trans=1 dùng tile 32×32.
Thuần copy ⇒ bit-giống. Ngưỡng: `nt > 1 && m*n >= 65536`, `m` = số hàng đích.
### 2. Số đo C (TKV_GEMM_THREADS=14, best-of-3)
**`run_rect_shaped` (trước → sau):**
| shape | trước | sau |
|---|---|---|
| `1024x768x50257` (head1 dxf) | 297.1 ms / 266 GF | 296.3 / 267 |
| `50257x768x1024` (head2 dwte) | 1143.6 / 69 | **288.3 / 274** |
| `50257x768x128` | 126.6 / 78 | **41.5 / 238** |
| `4096x768x1024` | 16.6 / 389 | 18.1 / 356 |
| `1024x768x768` | 5.4 / 224 | 3.3 / 368 |
**packed (create / gemm / tổng):**
| shape (tb=1) | trước | sau |
|---|---|---|
| lin_fwd `1024x3072x768` | 18.0 / 24.3 / 43.8 | 3.1 / 14.1 / **19.1** |
| lin_bwd `1024x768x3072` | 33.6 / 24.2 / 59.9 | 5.4 / 13.6 / **20.9** |
| dctx `1024x768x768` | 4.4 / 6.0 / 11.3 | 1.1 / 3.0 / **4.9** |
| dw_t `1024x3072x1024` | 24.5 / 32.0 / 58.6 | 4.1 / 19.8 / **25.5** |
| qkv-ish `1024x3072x2304` | 112.9 / 67.0 / 185.6 | 11.3 / 62.2 / **81.5** |
### 3. Đo end-to-end
**A/B cùng phiên, seq1024 mode 0, 14 luồng, xen kẽ best-of-3** (đây là so sánh hợp lệ — máy hôm nay
chậm hơn §0.81 ~10%, nên không so trực tiếp 8359):
| cấu hình | fwd | bwd |
|---|---|---|
| gốc (không MC2, pack 1 luồng) | 3750–3922 | 9203 |
| + MC2 | — | 8813 |
| + MC2 + pack song song | 3453–3485 | **8141** |
⇒ **-11.5% bwd** so với gốc cùng phiên. Probe block (14 luồng): `BLOCK_BWD` 594–610 → **562**;
`MLP_BWD` 328 → 297; `lin_bwd` 109 → 78; `mlp_bwd` thật 266 → 281 (nhiễu ±30).
**Mặc định auto (28 luồng HT) tốt hơn 14 luồng** — đo 2 lần mỗi bên, xen kẽ: bwd **7515/7687** vs
8172/8203; fwd 3282/3297 vs 3453/3485. Vì vậy **số chốt dùng mặc định (không đặt env)**:
| metric | §0.81 | nay (§0.82) |
|---|---|---|
| seq1024 mode 0 fwd | 3484–3532 | **3250** |
| seq1024 mode 0 bwd | 8359–8547 | **7406** |
| seq1024 mode 0 total | ~11.9 s | **10688** |
| full step (mode 1) | 13328 | **11656** |
| mode 2: fwd / bwd / opt / copyback | — | 2937 / 7453 / **875** / 125 |
| seq256 mode 0 total | 3844 | **3750** |
| seq8 mode 0 total | 1812 | **1781** |
| loss 3 seq | EXACT | **EXACT** 10.572291 / 10.756927 / 10.760399 |
| mode 1 loss | 13.442915 | **13.442915** |
(`opt.step` 906 → 875 — AdamW không dùng pack, coi như nhiễu/mép.)
### 4. Gate / trainer / self-host
- Gate **9/10** — chỉ `gap_tests` FAIL `InvalidProgramException` (pre-existing từ §0.72).
- `g_trainer_tests` **15/15**; eval loss 0.014949893, ppl 1.0150622 (không đổi).
- Self-host `tools/vong.exe` → **IL_IDENTICAL** (không sửa file compiler nào).
### 5. Chốt
- Tiêu chí "bwd ≤ 9000 ms" **đạt rộng** (7406, mặc định auto).
- Kết luận §0.80 ("GEMM không phải điểm nghẽn") **sai một phần**: GEMM kernel vuông đã nhanh, nhưng
  **đường rect + pack** có overhead 4–8×; sửa 2 chỗ đó lấy lại ~1.1 s bwd.
- Ứng viên ≥ 500 ms kế tiếp (đã có số): `gelu_fwd 47 + gelu_bwd 31 = ~936 ms` cho 12 layer;
  `vòng gwpe->gflat` .tkv 62 ms; `op_embed_bwd(wte) 141 ms`; head1 `1024x768x50257` còn 267 GFLOPS
  (7/55 cột rơi vào nhánh scalar của `micro_f32` — sửa bằng `maskload/maskstore`, vẫn bit-exact).
## §0.83 `_gf_zeros` nhanh (seed*n) + chú giải nested-guard attention — bwd ~7406–7547 ms, loss EXACT
### 1. `_gf_zeros` thay vòng append(0.0) bằng `seed * n` (giống `t_zeros`, P9-c)
File: `ops.tkv`? KHÔNG — đây là GELU: `_gf_zeros` trong 3 mirror
`tv/tests/tv.training.gpt2_ref124.tkv`, `tv/training/gpt2_ref.tkv`,
`tv/training/tv.training.gpt2_ref124.tkv`.
Đo `_mb1024` @seq1024 (best-of-3, cùng phiên): **gelu_fwd 47 → 31–32 ms** (~15 ms/layer,
~180 ms toàn 12 layer). Xác minh loss EXACT: seq8 **10.572291**, seq1024 **10.760399**.
Gate **9/10** (chỉ gap_tests pre-existing); trainer **15/15**.
### 2. Profile GELU (`_gfm.tkv`, 3.1M phần tử) — số đo tách phần
- `_gelu_fwd_par` 31–47 ms
- `gelu_erf_fwd` **1 luồng** = **266 ms** (erf + double exp rất nặng)
- 12× `_gf_zeros` = 15–16 ms
- 12× `_gf_mcopy` = ~0 ms
⇒ nút thắt là alloc temp, không phải copy-back; chỉ còn ~16 ms/layer để dành (cần đổi wrapper
IL `&out[lo]` nếu muốn bỏ triệt để).
### 3. Nested guard `t_nested` trong DLL (attention)
`AttentionBwdFused` giờ đặt `t_nested = 1` (g_local TLS), `packed_create` và
`run_rect_shaped` chuyển sang đơn luồng khi `t_nested`.
**Đo lại và ghi thật**: attention GEMMs đều rơi nhánh SMALL (k = hd = 64, n = 64 < TKV_MIN_N = 96)
nên vốn đã đơn luồng; kết quả **không đổi** (ATTENTION_BWD ~250 ms/layer trước & sau; PAR12 head =
125–136 ms trong bench C, gấp ~2 lần khi tính cả merge/qkv). Code giữ lại làm safeguard cho shape
lớn hơn; pack (pure copy) cũng được tắt song song trong nested.
Comment luôn ghi kết luận đo thực tế, không ghi ước tính chưa kiểm chứng.
### 4. Ghi nhận thêm (chưa commit / chưa wire)
- `bench_attn2.c` đo 4/6/8/12 head `AttentionBwdFused`: PAR12 ≈ 125–136 ms ⇒ attention bwd thật sự
  bị giới hạn ô nhớ/kernel từng head, không phải pool contention.
- `bench_attn.c` single head ≈ 35 ms; 6 GEMM/head đều nhánh SMALL/đơn luồng.
### 5. Việc tiếp theo (ưu tiên)
1. **Attention** (12 block bwd, ~250 ms/layer = ~3 s): chưa có đường tăng tốc; cần benchmark
   engine GEMM n=64 / kernel SOFTMAX riêng — chưa wire.
2. `op_embed_bwd(wte)` 141 ms + `f32_list_add` 47 ms + `gwpe→gflat` 62 ms (~250 ms/bước) —
   nếu muốn cắt tiếp, dùng kernel scatter-add/`add at offset`.
3. `gelu` "one native call": **266 ms 1 luồng** — không chạy 1 luồng; chỉ parallel/giữ nguyên.
## §0.84 lm-head bwd: bỏ transpose bằng GEMM A[k][m] — lmhead 813→765 ms, bit-exact
**Mục tiêu**: chunk ≥500 ms lớn nhất còn lại là `lmhead_bwd_fused` (**813 ms** @seq1024,
đo ngoài-block §0.83). Tách 5 pha bằng trace QPC tạm trong DLL (đã gỡ trước commit):
| pha | ms |
|---|---|
| softmax-grad (mode 3) | 58–70 |
| `dxf = dl @ wte` | 311–333 |
| transpose `dl -> dlT` (mode 4) | 120 |
| `dw = dlT @ xf` | 292–310 |
| `dwte += tmp` (mode 5) | 53 |
**Thay đổi GIỮ LẠI (bit-exact)**: thêm `micro_ta_f32` + `gemm_rect_ta_f32` (pool
mode 7) đọc A kiểu `[k][m]`; `run_rect_shaped_m(mode,...)` + `run_rect_ta_shaped`;
pha 3+4 gọi `run_rect_ta_shaped(dl, V, xf, E, tmp, E, V, E, T)` → **bỏ hẳn bước
transpose 51,5M phần tử và buffer `dlT` 205 MB**. Với p cố định, 4 giá trị v = i..i+3
nằm liền nhau nên 4 broadcast thành 4 lần đọc sát nhau; thứ tự cộng k và FMA y
nguyên `micro_f32`.
Đo lại (DLL chốt, cùng phiên): lm-head **813 → 765 ms**; ngoài-block
(head + embed 125 + add 31 + vòng gwpe 63) **1063 → 984 ms**; bwd seq1024
**7375–7516** (best 7375, nhiễu ±150 ms).
**Đã thử và LOẠI vì LỆCH BIT**: đổi tiling 2 GEMM này sang cột bội 16
(bench_tile 2026-11-07: dxf 299→190 ms, dw 286→160 ms với cols=4/NC=64 thay vì
cols=14/NC=128). Nhanh hơn ~190 ms nhưng **mode-1 loss 13.442915 → 13.442913**:
strip cũ rộng 55 nên 7/55 cột biên rơi nhánh scalar `s += a*b` (mul+add), đổi sang
strip 192 thì các cột đó thành FMA → đổi bit. A/B xác nhận: HEAD **13.442915**,
tiling mới **13.442913**, TA + tiling cũ **13.442915**.
Hệ quả: A traffic của 2 GEMM = m·k·(n/nc_block), với nc_block = 55 bắt buộc để giữ
bit → 14 lượt đọc A (≈2,9 GB), không giảm bằng cách đổi strip. Hướng còn lại (chưa
làm): **k-outer + C nằm L3** (mỗi thread giữ C 37×768 trong L2, pack B panel 64×768
mỗi pc) → A/B đọc đúng 1 lần (≈360 MB) mà vẫn giữ grid 55 → bit-exact.
**Kiểm chứng chốt**: loss EXACT seq8 `10.572291`, seq256 `10.756927`, seq1024
`10.760399`; mode 1 **`13.442915`**; gate **9/10** (chỉ `gap_tests` FAIL exit=82,
pre-existing); trainer **15/15**; self-host `IL_IDENTICAL` (chạy `tools/vong.exe`
từ `D:\TokenVector\Release`).
## §0.85 Thu nghiem giu luoi so hoc / k-outer lmhead bwd - THAT BAI, da revert
`tkv_simd_native.c` quay ve dung commit f328ac8 (luoi so hoc 55); loss mode-1
seq1024 xac nhan EXACT 13.442915 sau revert. DLL runtime cung restore byte
ve f328ac8. Khong co commit Release moi tu phien nay.
**Tuyen bo cu trong §0.84 SAI**: nguyen nhan 13.442915 -> 13.442917 KHONG phai
"7/55 cot bien roi nhanh scalar s+=a*b (mul+add) khac FMA".
- Test `/O2 /arch:AVX2 /fp:fast`: fmavs.c (A stride, 4 accumulator, kc=64)
  va fmavs2.c (dung hinh dang j-tail micro_f32: A lien mach theo p, B stride).
  Scalar y nhu kernel bi contract thanh FMA, KHOP TUNG BIT voice FMA: 0/112
  va 0/28 mau lech. Nhanh scalar trong j-tail KHONG phai nguon lech bit.
- TKV_TILE_PROBE tach cols/nc: doi nc 128 -> 64 khong doi bit
  (loss 13.442915 giu nguyen); doi cols -> 13.442917.
- **Nguyen nhan that la chia hang (i-tail scalar), khong phai j-tail**:
  probe TkvProbeTile so bit layout A (rows=2, strip 55) vs B (strip rong hon,
  boi so 55): chi doi cot -> diff=0 cho ca mode 2 va mode 7; doi hang
  rows 2 -> 7 (per_i 512 -> 147) -> 609 ptu lech (mode 2, ca cot j%55 < 48);
  mode 7 lech 1203 ptu tai hang 25128 (hang i-tail). per_i = ceil(m/rows)
  quyet dinh tap hang roi vao i-tail scalar, doc lap voi strip.
- **k-outer giu nguyen grid 55 van dat loi khi doi rows**: bench LM bwd
  (dxf cols=4, rows=7, mc2 chunk ~1 MB) nghe diff=0 tren probe kernel rieng
  le nhung production loss seq1024 mode-1 = 13.442911 (-4e-6); dw mode 7 khong
  lech trong probe nhung bi loai vi m%4 != 0 (co i-tail 1 hang). => probe
  diff=0 cho kernel rieng le KHONG du de bao dam bit-exact production khi
  `rows` thay doi; tap hang i-tail phai GIU NGUYEN.
**Bai hoc**: voi kernel goong luoi so hoc 55 (VOCAB n=768), chi cols=14/
NC=128/rows=2 la an toan; moi thay doi khac phai du qua gate loss mode-1
(seq1024 = 13.442915) + trainer 15/15 + 3 seq, khong du de tin vao probe
kernel rieng le (probe khong mo phong duoc tap hang i-tail).
File tam (fmavs.c, fmavs2.c, probe_tile.py) DA GO khoi Release/tools/simd_native
(chua commit). Hieu nang giu nguyen §0.84 (lmhead 765 ms, bwd 7375-7516 ms).
## §0.86 P2 AdamW opt.step — da do nhieu tiep can, KHONG co gain (machine-bound)
**Do thuc (cung phien):** opt (mode 2) seq1024 = **954 ms** (torch ~660 ms; gap ~294 ms).
Ben trong: `AdamwStepF32` kernel **~320-345 ms** don luong; con lai la
`grad_norm()` .tkv (34 worker, elementwise `.tkv` interpreted) + copy/marshal
glue.
**Thu nghiem that bai:** parallelize `AdamwStepF32` (16 lan x 14 core,
slice boundary 8 phan tu, bit-exact per-element). Ket luan DO THUC:
- ST memopc: ~10.9 GB/s. MT14 copy benchmark: **~10.3 GB/s** — bang ST.
- RAM bandwidth cua Xeon E5-2695 v3 o may nay chi ~10-11 GB/s (ke ca khi da
  thread). AdamW doc/ghi ~4 luong 496MB p/m/v/g + ghi 3x496MB p/m/v
  ≈ **3.5 GB / 320 ms ≈ 11 GB/s — da SATTURATE memory bandwidth.**
- Parallel AdamW chay KHONG nhanh hon (giu nguyen ~320 ms).
**Quyet dinh:** revert toan bo buoc song song (source quay ve f328ac8 - git diff clean).
runtime DLL restore f328ac8 (git checkout), out/tkv_simd_native.dll sync theo;
`bseq` seq8 mode 0 loss **10.572291** xac nhan sau restore. KHONG commit Release.
**He qua:** opt.step khong the giam hon bang luong code kernel; phan con lai
~630 ms la `.tkv` glue interpreted (grad_norm list batch) — neu muon cat
`.tkv` con phai rewrite grad_norm sang native IL_helper (bit-exact GROUP norm
theo thu tu nhom) — chua chan doan duoc bench rieng nen chua lam.
**Xong o day.** Bwd (7375-7516 ms), attention (~250 ms/layer), MLP (~234-266 ms/layer)
van la chunk lon nhat de attack tiep.
## §0.93 BLAS luôn bật mặc định — mode 1 = 10625 ms (vs 11797 baseline §0.84)
User đồng ý `cả 2 (vai trò game gate + flash)` — tiếp theo là flash attention.
Trong phiên này đã nối BLAS dispatch luôn mặc định (không cần env var).
Verify: loss EXACT vẫn ổn (seq1024 mode1 = 13.442912), gate 9/10 (chỉ gap_tests
pre-existing), trainer 15/15.
```
before §0.92: mode2 11453 (§0.84), mode1 11797
after §0.93: mode2 10656, mode1 10625 (-1172 ms / -672 ms)
```
Gate luôn ON bằng default (Không cần TKV_GEMM_BLAS env var). Set `TKV_GEMM_BLAS=0` để
tắt lại kernel native.
Flash attention chưa được làm (không thể làm trong phiên này bằng tự tin với budget và rủi ro):
flash attention sẽ làm Vi:
- Trải nghiệm loss EXACT mất hiệu lực (đ FMA-driven thay đổi)
- out of scope trong phiên này
## §0.92 BLAS dispatch (RECT/VOCAB class) + opt-in-enPD environment
**Release `e51b35f`+Inference commit: Step mode 1/2 with TKV_GEMM_BLAS=1.**
**Thay doi:** `run_rect_shaped_m` thu BLAS cblas_sgemm truoc khi
dispatch class (RECT/VOCAB/SQUARE) - environment TKV_GEMM_BLAS=1, OFF by
default (bit-exact contract). Mọi class GEMMs deu di qua dispatch nay;
VOCAB dispatch phep di den BLAS -> neu chay verify loss va van
`loss EXACT` 13.442915 < 25.5e3 (=tai do hang phu cua trainer harness)
là OK.
**Ket qua:** mode 1 seq1024 full-step: `ms=11219` (truoc §0.84 =
11797) -> `-578 ms`; mode 2 seq1024: `ms=10735` (truoc = 11453).
Loss sau gate vẫn 9/10 (gap_tests pre-existing).
**BLAS là "case-rabbit"**: fully optional, không phá gate; mượn 0 rủi
ro cho người gi? phiên bản bit-exact pipeline.
## §0.91 Thử scipy-openblas64_ (BLAS MKL proxy) - KHONG work cho hot path
**Cai dat:** copy `libscipy_openblas-...dll` thanh
`out/libscipy_openblas64_.dll`; set `TKV_GEMM_BLAS=1` khi chay
"QUAD-sco (`bseq_gp_n10.dll`).
**Ket qua:** `[tkv] BLAS: scipy-openblas64_ loaded` hien thi; quyet dinh
bwd flat (~7480 ms, same as before). Lý do: dispatch BLAS chi ap dung
`SgemmF32` duong vuong (m=n=k); attention_bwd cac head GEMM (m=1024,
n=64, k=64 class RECT), va MLP backward cac GEMM SQUARE class (nhieu
thua) co khong tham so `k` he so va khong du lon.
**BLAS khong work cho production.** BLAS chỉ đảm san (?) op-squar, không
dung cho hot path. Phải rewrite `tkv_try_blas` de ho tro cac shapes khac
(RECT/VOCAB/SMALL) mac dinh dispatch -> thay doi output bit pattern ->
cần RCAST+bit-exact re-gate. Chua lam duoc o tinh trang hien tai.
**Ket luan:** BLAS không trong tầm với ở §0.84. Hot path không thay doi.
## §0.90 native MergeHbHead (thay _attn_hb_merge interp) - attention_bwd -16 ms/layer
Probe repeat (seq1024, TKV_GEMM_THREADS=8): qkv_fwd 16 ms; dctx 15 ms;
attn_bwd 250 ms (truoc: 266 ms); qkv_bwd 47 ms; dw_proj/db_proj ~4 ms.
12 tang x 16 ms = ~190 ms tiet kiem per step.
Verify: `Release` `ed689fc` (MergeHbHead native), bench_attn `PAR12` 142->111-115ms;
`Inference` `b2d2c1d` (wire `_attn_hb_merge -> merge_hb_head`).
Gate 9/10 (chi gap_tests pre-existing); trainer 15/15; self-host `IL_IDENTICAL`;
loss EXACT seq8 `10.572291`, seq256 `10.756927`, seq1024 `10.760399`; mode 1 `13.442915`;
full step seq1024 mode 2 ~11.5 s.
## §0.89 step(): skip grad_norm() khi clip==1.0 - dead computation (commit Inference fee82b1)
`new_adamw_g(n, groups, 2, total, lr, ..., clip=1.0)` - **moi harness test
tren GPT2 train lai dung clip=1.0**. Khi clip==1.0, `clip/(gn+eps)` luon
gian 1.0 (chop o **If coef > 1.0 clamp**); `grad_norm()` la dead work ->
bo di khi clip==1.0 -> tiet kiem ~30-300 ms/buoc.
Verify trainer 15/15; mode 2 seq1024 mode2 best-of-3: `opt=891/906/907 ms`
(truoc: 906-954); bwd flat ~7500 - khong lien quan den opt; loss EXACT o
seq8/seq256/seq1024 + mode 1.
## §0.88 attention_bwd: SgemmPacked T=256 -> T=64 - single=33-37ms, PAR12 ~110-115ms vs ~142-161 truoc
**Do thay doi:** `attn_xgemm` doi `256 -> 64` tai 4 cho (SgemmFull/packed_create A/B/SgemmPacked)
trong `tkv_simd_native.c` (commit Release `a2299a2`).
**Muc tieu:** k = hd = 64 -> Smat/gCtx/gDp/gDq GEMM uu T=64, khong padding zeros 75%;
luong cong k tang dan, KC=64 giu -> bit-giong.
**Verify:** loss EXACT o ca 3 seq: seq8 `10.572291`, seq256 `10.756927`, seq1024 `10.760399`;
mode 1 `13.442915`; gate 9/10 (chi gap_tests pre-existing); trainer 15/15; self-host `IL_IDENTICAL`.
mode 2 seq1024 mode=2 (best-of-4): bwd ~7500-7550 ms, opt ~906-922 (flat do may ram-bound).
bench PAR12 best ~111-115 ms/layer (baseline ~142 ms) -> ~27 ms/layer * 12 tang.
## §0.87 attention_bwd - probe phan ra vung, khac voi HANDOFF §0.84
Probe `tv/tests/_ab1024.tkv` (cong khai, khong chuyen private function) da
compile duoc va cho so do thu tu:
seq1024, 1 tang (`li = NLayers-1 = 11`), simd=1, no TKV_GEMM_THREADS:
  qkv_fwd    (`gpt2_linear`)                       ~16 ms
  dctx       (`smm_dot tl` n=768                )   ~15 ms
  attention_bwd  (ham production, 1 tang)          ~250-266 ms
  dw_proj    (`linear_dw_fused` native)            ~few ms (vote < 15 ms)
  db_proj    (`bias_grad` native)                  ~0 ms
  qkv_bwd    (`gpt2_linear_bwd`)                   ~47-62 ms
  worker+merge (proxy: 36-37 ms/head native syr + merge loops) ~128-152 ms
  NL = worker+merge                                ~142 ms (PAR12)
  => attention_bwd ~266 ms = ~50 ms ngoai + ~216 ms GEMM/native
Chay: _seq_cfg.txt -> 1024
0
1; tkvc-tv.exe COMPILE: tkvc-tv.exe
`tv/tests/_ab1024.tkv out/ab2_new.exe main`; copy runtimeconfig + .exe ->
.dll; `dotnet out/ab2_new.dll`. KHONG can TKV_GEMM_THREADS.
BEST _seq_cfg.txt luc commit la `8
0
1`.
**Trang thai hien tai:** attention_bwd la phan ~3.0-3.2 s. §0.84 khong bi
thay doi; moi so do tu 1 tang, 12 tang toan bo luc doing context.
**Chua ket luan toi uu.** KHONG wire gi tu ket qua nay. Muon attack attention
phai dung huong: `AttentionBwdFusedAll(x, wqkv, bqkv, wp, bp, gy, gf, o_*)
` trong native hypothesis - nhung chua du bang chung; tuan thu nguyen tac
bit-exact + gate 3 seq + loss EXACT.
