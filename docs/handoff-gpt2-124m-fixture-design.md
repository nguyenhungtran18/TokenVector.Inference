# HANDOFF: THIET KE CHE DO FIXTURE NGOAI REPO CHO PARITY GPT-2 124M

Trang thai: **DA TRIEN KHAI + CHAY XANH (2026-10-19)** — gate 20/20, harness
4 buoc XANH exit=0, khong hoi quy 6/6 gate nho. Chi tiet ket qua: HANDOFF.md
0.9. Tai lieu nay giu lai su thiet ke + cac so do (doc truoc khi sua code).

Muc tieu: mo khoa BLOCKER 124M trong HANDOFF.md 0.6 — chay parity GPT-2 124M
THAT (50257/768/12/12) voi fixture khong check-in vao repo, dung lai bo gate
nho 256/64/4/2 hien tai.

## 1. SU THAT DA DO (khong phai uoc luong)

1. **Co tien le `--out`**: `tools/gen_gpt2_fixture.py` da ho tro `--out <dir>`
   (duoc verify_parity.ps1 su dung de sinh fixture vao %TEMP% roi diff voi ban
   check-in). Che do fixture ngoai repo da co 50% — chi thieu preset 124M.

2. **May nay du dieu kien sinh fixture 124M**: python + torch 2.14.0+cpu +
   transformers 5.15.1 da cai (da verify bang lenh import).

3. **Import resolve ve flattened copy, KHONG phai canonical** (probe P1 da
   chay, ket qua that): khi bien dich tu `tv/tests/`, `import tv.training.gpt2_ref`
   lay `tv/tests/tv.training.gpt2_ref.tkv` (256/64/4/2), khong phai
   `tv/training/gpt2_ref.tkv` (dang 50257/768/12/12/1024). => gate 124M khong
   the import canonical truc tiep neu resolve van the; co 2 huong:
   (a) flattened copy rieng `tv/tests/tv.training.gpt2_ref124.tkv`, hoac
   (b) chua biet compiler co tim module chi co ban canonical khong -> PROBE P2.

4. **`argv()` la intrinsic compiler** (stub trong cli.tkv duoc thay khi chay
   that; cli.tkv dung de parse flag `train --steps N`). Neu P3 xanh thi gate
   nhan fixture dir tu lenh goi: `gpt2_full_tests.exe <fixdir>`.

5. **IO da du trong .tkv**: `read_file` (text, gate hien tai dang dung),
   `ckpt_file_read_all_bytes` + `ckpt_native_read_f32` (binary f32, dang dung
   trong save/load_checkpoint — da bit-exact qua trainer gate), `file_write_bytes`.
   => neu text 425 MB qua cham thi chuyen wte sang binary f32 (~155 MB) duoc
   ngay, khong can compiler moi.

6. **Thoi gian du kien** (so do HANDOFF 0.6, da do tren may nay): forward 124M
   ~0.6 s/token, 1 buoc train ~15 s. Mot gate parity 8-token (forward+backward
   + parse) du kien vai phut — chap nhan duoc voi gate CHAY TAY, khong dua vao
   bo gate mac dinh.

7. **Bit-exact voi HF khong the dat** (HANDOFF 0.6, blocker 3): GELU erf xap
   xi -> sai so ~1.5e-7 luon > 0. Gate 124M phai dung TOLERANCE, theo
   docs/numerics-contract.md, khong phai 0.0 nhu muc dich nho.

8. **Luu y da xu ly**: vua roi da dao config gen_gpt2_fixture.py tu 124M ve
   thu nho (256/64/4/2) vi luot regenerate 124M de dang checkpoint (wte 533 MB
   text, fixture 1.6 GB, gate do). Phan 124M tu day di PHAI la che do --full
   ngoai repo, KHONG dua vao fixture check-in.

## 2. THIET KE DE XUAT (chua trien khai)

### 2.1 Generator — them preset `--full`
- `python tools/gen_gpt2_fixture.py --full --out <dir>`: VOCAB=50257,
  N_EMBD=768, N_HEAD=12, N_LAYER=12, MAX_POS=1024, SEED=7, ids/pos/m_* giong
  ban nho. Giu nguyen ham write_case/format text `%.9g`.
- **GUARD bat buoc**: neu `--full` ma `--out` tro vao `tv/tests/fixtures/`
  (repo) -> in loi va exit 2, khong ghi. Nguoi dung PHAI chi ro dir ngoai repo
  (mac dinh hien tai: `tv/tests/fixtures-124m` trong repo nhung .gitignore —
  truoc do la `%TEMP%/tv-gpt2-124m`, da doi 19-10-2026 vi TEMP bi don bo).
- Khong doi default (khong --full van la ban thu nho) -> verify_parity.ps1
  va gate nho khong bi anh huong.

### 2.2 Gate — `tv/tests/gpt2_full_tests.tkv`
- Doc fixture dir tu `argv()[0]`; neu rong -> in "SKIP: can fixture dir (chay
  gen_gpt2_fixture.py --full --out <dir>)" va return 0 (SKIP khong phai FAIL,
  dung quy uoc verify_parity.ps1).
- Cau truc giong gpt2_tests.tkv (hidden tung tang, logits, loss, 5 grad mau:
  wte, ln1_g0, attn_w0, mlp_w1_0, lnf_g) + ca duong mask (g2_m_*).
- Tol: dung 1e-5 cho ban nho; ban 124M chua do duoc -> chay lan dau lay so
  thuc te, chot tol moi (kho ~1e-4/1e-3 theo ly thuyet tich luy 12 tang), ghi
  ro vao HANDOFF, khong che.
- Config 124M: tuy ket qua P2/P3 (xem 3.2, 3.3).

### 2.3 Harness — `tools/verify_gpt2_124m.ps1`
Mau theo verify_parity.ps1:
1. Kiem tra python + torch + transformers -> thieu thi SKIP (exit 0 + bao ro).
2. Chay `gen_gpt2_fixture.py --full` (mac dinh `tv\tests\fixtures-124m`; cache: chi
   sinh lai neu thieu hoac `-Regen`).
3. Bien dich gate bang `D:\TokenVector\Release\dist\tkvc-tv.exe`, chay exe voi
   fixture dir, ghi `reports/gpt2-124m-<stamp>.md`, giu exit code
   (0 = khong FAIL, SKIP duoc phep; 1 = co FAIL).

### 2.4 Bo sung (sau khi gate chay xanh)
- HANDOFF.md 0.6: doi ket luan "bi chan" thanh "co che do --full ngoai repo,
  chua chay chinh thuc"; ghi ket qua do duoc vao day.
- ROADMAP_TRAIN.md: dong G2 ghi ro "124M: fixture ngoai repo, gate rieng".

## 6. KET QUA TRIEN KHAI (19-10-2026) — DA CHAY THAT

- P2 DA CHAY: module chi co ban canonical KHONG resolve -> tao flattened
  `tv/tests/tv.training.gpt2_ref124.tkv` + `tv/training/tv.training.gpt2_ref124.tkv`.
- P3 DA CHAY: argv() khong hoat dong (stub return [], ca cli.exe in usage
  voi arg) -> bo huong argv, dung file pointer `tv/tests/g2_full_dir.txt`.
- P4 DA CHAY: parse 38.6M/533MB = 26.3s, scan 31ms -> text du nhanh,
  khong can binary.
- SU THAT MOI (quan trong hon ca): file 533MB split OK, 594MB bi
  OutOfMemoryException sau 4.9s (du RAM 7.7GB free — do UTF-16 ~2x + LOH,
  khong phai thieu RAM). -> `--full` shard 15M gia tri/man + manifest
  `g2_shards.txt`; gate `load_shards`/`load_fx` doc manifest.
- Quirk compiler gap khi viet gate (ghi lai de khong lap lai):
  + ham long doc bien cua ham cha -> bat `nonlocal` (tw phai ra top-level).
  + bien module-level khong gan lai trong ham (`global` khong ho tro), khong
    doc file o module level (chi hang so don gian) -> truyen `d`/`sk` qua tham so.
  + phai annotation `FIX_DIR: "str" = ...` cho bien module.
  + `.strip()` tren ket qua split hoat dong (dung cho manifest CRLF).
- Ket qua gate 124M (2m49s): h0=0.0, h1..h12 <=2.1e-06, logits 5.56e-06,
  loss 9.5e-07, grad wte 4.11e-06 — TOL_HID=1e-4 / TOL_LOSS=1e-5 hop ly
  (max do / tol ~ 1:18, du bien cho moi truong khac).
- Generator: --full + GUARD (vao repo -> exit 2; thieu transformers -> exit
  2; khong --full van ban nho, verify_parity.ps1 khong anh huong).
- Harness `tools/verify_gpt2_124m.ps1`: 4 buoc, report
  `reports/gpt2-124m-<stamp>.md`, exit 0/1 nhu verify_parity. LUU Y: file
  .ps1 PHAI ASCII (ky tu `—`/tieng viet trong string lam PS 5.1 parse loi
  vi khong BOM — da gap va sua).
- 6/6 gate nho xanh lai sau tat ca sua doi.

## 3. CAC BUOC CON LAI (theo thu tu) — DA HOAN TAT PHAN 3.1-3.4

### 3.1 PROBE P3 — argv() (da co san file, chua chay)
```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/probe_argv.tkv %TEMP%/tvgate/probe_argv.exe main
%TEMP%/tvgate/probe_argv.exe mot hai ba
```
Ky vong: argc=3 + in ra cac tu. Neu argv() rong trong gate exe -> huong thay:
fixture dir dat vi tri CO DINH (D:\TokenVector\fixtures-124m) hoac doc tu
file pointer.

### 3.2 PROBE P2 — resolve module chi co ban canonical (da co san file, chua chay)
```
D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/probe_resolve.tkv %TEMP%/tvgate/probe_resolve.exe main
%TEMP%/tvgate/probe_resolve.exe
```
Ky vong: `probe_value=4242` (resolve duoc canonical tv/training/probe_mod.tkv).
Neu FAIL -> tao flattened copy `tv/tests/tv.training.gpt2_ref124.tkv` (copy
canonical, doi ten module) cho gate 124M.

### 3.3 PROBE P4 — do toc do parse text 425 MB (chua co file)
Sinh wte 124M bang python ra %TEMP%, roi gate .tkv nho doc bang
`read_file` + split + float() hoi `now_ms()`, in MB/s. Neu qua cham (>5 phut)
-> chon luong binary f32 (`ckpt_file_read_all_bytes` + `ckpt_native_read_f32`),
generator ghi them `.bin`.

### 3.4 Trien khai (theo Muc 2), chay toan bo gate nho xanh truoc khi commit
Gate nho phai van: trainer 15/15, gpt2 10/10, gpt2_scope 9/9, gpt2_train 5/5,
swiglu 3/3, inplace 5/5.

## 4. FILE PRODE DANG DE TRONG REPO (chua commit, xoa sau khi chay xong)
- `tv/tests/probe_argv.tkv` (P3)
- `tv/tests/probe_resolve.tkv` (P2)
- `tv/training/probe_mod.tkv` (doi xung P2)
- `tv/tests/probe_import.tkv` + `.probe/probe_import.tkv` (P1 — DA CHAY XONG,
  co the xoa ngay)

## 5. NGUON THAM CHIEU
- HANDOFF.md 0.6 (blocker 124M + so do toc do), 0.7 (G4xG2), §3 (compiler
  state: tkvc-tv.exe, intrinsic argv/now_ms/gc, binh thuong .tkv khong chay
  tren mono).
- docs/numerics-contract.md (fp32, GELU erf, dropout=0, bit-exact CPU).
- tools/verify_parity.ps1 (mau harness SKIP/PASS/FAIL + report).
