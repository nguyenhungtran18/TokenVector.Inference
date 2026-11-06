# TASK CHI TIET — PLAN_TO_LE_1X.md (<=1x PyTorch, GPT-2 124M full train step)

Ngay: 2026-10-31. Baseline do xen ke cung phien (HANDOFF §0.59):
- full step tkv: 3844 / 5844 / 16078 ms (seq8/256/1024)
- torch 14T: 1042.7 / 1572.7 / 4148.4 ms → ti le 3.69× / 3.72× / 3.88×
- torch 1T: 1388.0 / 4354.8 / 15082.1 ms → tkv gan parity o seq1024 (1.07×)

Dieu kien khong thuong luong (ap dung moi task):
- Loss EXACT, khong doi thu tu cong.
- `tools/gate_f32.sh` xanh 135/135 + trainer 15/15 sau moi thay doi.
- Doi compiler → self-host `tools/vong.exe` → `IL_IDENTICAL`.
- Do truoc, sua sau; moi phase do xen ke torch cung phien roi moi ghi HANDOFF.
- Kill criterion: bucket <5% buoc → loai khoi plan.

---

## Phase H2 — Codegen nuot bucket `block_bwd` (don bay lon nhat da do)

Muc tieu: bucket 1617 ms → ~45 ms/goi (−67%), bwd seq8 2500 → ~1400 ms.

### H2.1 Audit `TkvInt::` trong `gpt2_block_bwd`
- [ ] Grep/IL dump `gpt2_block_bwd` de dem chinh xac so `TkvInt::` (baseline 48).
- [ ] Liet ke tung call site theo nhom: ceremony, copy, reduction, double-load.
- [ ] Do `_p6diag` (body-only) de so sanh A/B/C ratio (baseline A 5.61×, B 3.85–4.24×, C 5.00–6.45×).
- Dau ra: bang `call site | ms | loai | lever`.

### H2.2 Request compiler team — don `TkvInt::`  ✅ DA SOAN (2026-11-01)
- [x] Soan `docs/COMPILER_TEAM_REQUEST9.md` (gop H2 + H4b).
- [ ] Yeu cau: bo ceremony #7 con lai trong `gpt2_block_bwd`.
- [ ] Yeu cau: hoist vong `mu` reduction sang byref `_items` + range-proof.
- [ ] Yeu cau: LICM invariant `m1/m2/rstd` + bo double-load `gy`.
- [ ] Fallback C2: 4 builtin `softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad` neu matcher khong an shape.
- [ ] Self-host `tools/vong.exe` → `IL_IDENTICAL` → copy `dist/tkvc-tv.exe`.

### H2.3 Wire + do
- [ ] Rebuild toan bo entry point (khong build module le).
- [ ] Gate 135/135 + trainer 15/15.
- [ ] Loss chain 3 seq EXACT.
- [ ] Do bucket: muc tieu `134.75 → ~45 ms/goi`.
- [ ] Bench 3 seq xen ke torch (mode 0 + mode 1), ghi `out/h2_*.log`.
- [ ] Cap nhat HANDOFF §0.60 + TODO_M9.

Kill criterion: neu bucket chi giam <20% → dung, danh gia lai matcher.

---

## Phase H3 — Threading doi xung 14C/28T

Muc tieu: GEMM + elementwise/norm/optimizer scale theo torch (1.33×/2.77×/3.64×). Tran thuc te: seq1024 16078/3.64 ≈ 4417 ms = 1.06× → phai sau H2.

### H3.1 Sweep `TKV_GEMM_THREADS`  ✅ DA DO (2026-11-01, seq1024 mode1)
- [x] Probe CG mode 1: sweep 1/4/8/14.
- [x] Ket qua: 1T=34312 ms, 4T=18015 ms, 8T=15625 ms, 14T=15969 ms, default=16203 ms.
- [x] 8T nhanh nhat (-3.6% vs default).
- [!] Rui ro exactness: loss in 13.442913–13.442917 (khac 13.442915 baseline) → threading GEMM thay doi thu tu tich luy. Can danh gia tolerance truoc khi wire vao training.

### H3.2 Native threading trong DLL
- [ ] Elementwise/norm/optimizer thread trong native (memcpy + bandwidth ops).
- [ ] BWD parallel op-level **trong native** (khong merge interpreted).
- [ ] Kiem tra re-entrancy theo §3.1 `simd_gemm_full`.

### H3.3 Do va gate
- [ ] Gate 135/135 + trainer 15/15.
- [ ] Loss chain EXACT.
- [ ] Bench 3 seq xen ke torch, ghi `out/h3_*.log`.
- [ ] Cap nhat HANDOFF.

Rui ro: da bust 8 lan → phai do truoc, khong assume.

---

## Phase H4 — Chi phi co dinh 38.6M (m6 + `op_embed_bwd`) — bat buoc cho seq8/256

Muc tieu: 578 ms → ~150–200 ms/step.

### H4a — Scratch step-level cho m6  ✅ DA LAM (2026-11-01)
- [x] Giu scratch `sc4/sc5` cap step (da co trong `_step124`).
- [x] Them scratch cho m6: `plg/pids/pwte/pxf/pdxf/pdwe` (hoist len step, truyen vao gpt2_backward).
- [x] Bo 6 alloc + 6 free moi lop-buoc (chi alloc 1 lan/step, free bo qua khi reuse).
- Ket qua best-of-3: seq8 3750 ms, seq256 5734 ms, seq1024 16203 ms (vs 4141/6015/16094).

### H4b — Bo `copy_to` 154 MB  ⏸ DEFERRED (can native/compiler team)
- [ ] `fused` doc/ghi thang buffer persistent cua `gpt2_backward`.
- [ ] Bo copy 154 MB vao/ra.
- Ly do: `lmhead_bwd_fused` la builtin native, chi nhan handle `alloc_bytes`; du lieu managed `.data` khong co handle zero-copy. Can team native them API hoac builtin moi.

### H4c — `op_embed_bwd` ×2 (234 ms)  ✅ DA LAM (2026-11-01)
- [x] Thay bubble sort O(n²) bang counting sort O(n+V) trong `op_embed_bwd`.
- [x] Wire canonical + mirror (training + tests).
- [x] grad_tests PASS (embed bwd diff=0.0).

### H4 gate
- [ ] Gate 135/135 + trainer 15/15.
- [ ] Loss EXACT.
- [ ] Bench 3 seq xen ke torch, ghi `out/h4_*.log`.
- [ ] Cap nhat HANDOFF.

Kill criterion: neu giam <100 ms → loai, chuyen sang H5.

---

## Phase H4' — GEMM side (phu, khong chan)

### H4'.1 B-tile cache theo i0
- [ ] Implement cache (don 2 §6, `_gf1/_gf2`).
- [ ] Muc tieu S3/S5 m=1024 ≤400 ms → noi routing `m ≤ 1024`.

### H4'.2 Pack 1 lan/step
- [ ] Persistent packed-handle cache theo step (weight bat bien trong step).
- [ ] Free o bien step.

### H4'.3 Pad/T-dispatch
- [ ] Chi tai mo neu budget table chi ra pad con ≥5% step.

Gate + bench + HANDOFF sau moi muc.

---

## Phase H5 — Peel cuoi theo budget table

- [ ] Sau H2+H3+H4, liet ke gap con lai theo format: `bucket | ms | lever`.
- [ ] Khong nhan chung chung; moi muc phai co so do.
- [ ] Muc tieu ≤1× o ca 3 seq; sau do tiep tuc <1× (vd 0.8×).

---

## Task 3 — Builtin `zeros_list(n)` (khong chan performance)

- [ ] Viet `compile_zeros_list` + register builtin (`file_io.tkv` dong 75).
- [ ] Self-host `tools/vong.exe` → `IL_IDENTICAL` → copy `dist/tkvc-tv.exe`.
- [ ] Wire vao `t_zeros` (`tensor.tkv` dong 63 + 2 ban sao mirror).
- [ ] Do `_tz2`: baseline `t_zeros([768,3072])` = 6.5 ms/lan.
- [ ] Gate 135/135 + bucket + loss chain.

---

## Protocol do (bat buoc)

1. Do truoc: baseline xen ke cung phien, luu `out/h*_*.log`.
2. Sua theo task.
3. Gate 135/135 + trainer 15/15.
4. Loss chain 3 seq EXACT.
5. Bench 3 seq xen ke torch (mode 0 + mode 1), best-of-3.
6. Cap nhat HANDOFF §0.x + TODO_M9.md.
7. Chi chuyen task tiep khi task truoc co so do.

---

## Bang muc tieu (torch mốc = so fresh §0.59)

| Giai doan | seq8 | seq256 | seq1024 |
|---|---:|---:|---:|
| Hom nay | 3844 (3.69×) | 5844 (3.72×) | 16078 (3.88×) |
| +H2 | ~3000 (2.9×) | ~4700 | ~14000 |
| +H4 | ~2600 (2.5×) | ~4300 | ~13600 |
| +H3 | ~1950 (1.9×) | ~1550 | ~3740 (1.06×) |
| Peel cuoi | ≤1043 (≤1×) | ≤1573 | ≤4148 |

Cac dong +H2/+H4/+H3 la du bao tu so do tung phan, chua phai ket qua. Phai do lai sau moi lever.