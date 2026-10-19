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

### 0.9 Tiep theo

G3, G4, G5 xong (G5 thieu notebook Colab). G6 (phan tan) nam ngoai pham vi
theo ROADMAP. Viec con lai: noi trainer G4 vao GPT-2 nho cua G2 de co vong
train that, va viet notebook Colab.

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
