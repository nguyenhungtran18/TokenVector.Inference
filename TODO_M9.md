# TODO — TokenVector · P9 → M9 (2026-10-31)

Quy tắc giữ nguyên cho mọi mục dưới đây:
- Loss phải EXACT — **không đổi thứ tự cộng**.
- `tools/gate_f32.sh` phải xanh **135/135** sau mỗi thay đổi.
- Đổi compiler → phải self-host `tools/vong.exe` → `IL_IDENTICAL`.
- **Đo trước, sửa sau** — không nhận định hiệu năng khi chưa có số đo.

---

## ✅ Task 1 — Full-step 3 seq sau t_zeros + BtB  (HOÀN THÀNH, ĐÃ ĐO)

- [x] A/B **cùng phiên**: `out/bseq_p9.exe` (mới) vs `out/bseq_c11_n10.dll` (trước P9), best-of-3, script `out/p9_3seq.sh`.
- [x] Kết quả:

  | seq | trước P9 | P9 | Δ |
  |---|---|---|---|
  | 8 | 4157 ms | 3860 ms | −7.1% |
  | 256 | 6172 ms | 5797 ms | −6.1% |
  | 1024 | 17547 ms | 16719 ms | −4.7% |

- [x] Loss EXACT 3/3, không seq nào regress:
  `10.572291 → 7.0119185`, `10.756927 → 11.807267`, `10.760399 → 13.442915`.

---

## ✅ Task 2 — Bỏ 8× k-waste trong `simd_gemm`  (HOÀN THÀNH, ĐÃ XÉT — 135/135 + bucket + loss + `_p6ms` xanh 31/10)

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
- [x] **Chạy đủ gate 135/135** — chạy lại trọn sau compiler fix (lần 16:25 bị đứt):
      `bash tools/gate_f32.sh > out/gate_req.log 2>&1` → **10/10 suite PASS,
      `TOTAL fails=0`, exit 0**; kèm trainer **15/15** (`out/tr_req.log`).
- [x] Đo lại bucket: `dotnet out/am_n10.dll | grep BUCKET` → **1569.0 / 1570.5 ms**
      ≤ **1680** ✅ (baseline 2244 = −29.6%); `out/am_req_r1.log`/`_r2.log`.
- [x] Loss chain 3 seq EXACT cả 9 lượt: `7.0119185` / `11.807267` / `13.442915`
      (best 3829 / 5672 / 16375 ms — không seq regress); `out/req_3seq.summary`.
- [x] Đo `_p6ms` (≥ 4×): **A 4.44× · B 5.99× · C 6.02×**, `bit_bad=0`, 3 rep;
      `out/p6ms_req_r1..3.log`.
- [x] Cập nhật `HANDOFF.md` → **§0.58** (kèm đối chiếu torch 4.07-4.62× + IL snipe).

---

## ✅ H1 — Bảng budget 2 phía xen kẽ torch fresh  (HOÀN THÀNH 31/10 — HANDOFF §0.59)

- [x] tkv: `simd_bench_124m_seq` mode 0 (fwd/bwd) + mode 1 (full step), best-of-3, 3 seq,
      binary `out/bseq_p9_n10.dll` (`out/h1_run.sh tkv`).
- [x] torch: `bench_torch.py e` + `d` tại 3 seq, **T=1 và T=14**, cùng phiên (`out/h1_run.sh torch`).
- [x] Probe: `_bd2` rebuild bằng compiler 31/10 09:19 (`out/bd2_n10.exe`) + `_p6ms` + `simd_probe_am`.
- [x] Kết quả chốt: tỉ lệ cùng phiên **3.69× / 3.72× / 3.88×**; gap = **bwd 73-79%**;
      `12 × block_bwd` 1617 ms = 65% bwd; fixed-cost 38.6M (m6 344 + embed 234) = 578 ms;
      tkv đơn luồng **1.07×** torch đơn luồng tại seq1024 ⇒ 93% khoảng cách là threading của torch.
- [x] Kill criterion: 2 vòng dense 38.6M (~62 ms, 2.5%) + wpe loop (dưới sàn `now_ms`) **loại khỏi plan**.

---

## ⬜ Task 3 — Builtin `zeros_list(n)`

- [x] Thí nghiệm IL thật (`out/_zl.il`, `ilasm.exe` → chạy `dotnet`):
      **`stfld` vào field private `_items` / `_size` của `List<float32>` ném
      `MissingFieldException: Field not found`** → cách viết trong yêu cầu gốc không chạy được.
- [ ] Viết `compile_zeros_list` + `register_expr_builtin(reg, 'zeros_list', ..., 'f32', return_shape='list')`.
      Cơ chế đã chốt: `newarr float32` (CLR zero-init) + `newobj List<float32>::.ctor()`,
      rồi set `_items` / `_size` qua **reflection** — `GetField(name, 36)` = `NonPublic|Instance`
      (đúng hệt mã 36 code hiện đang dùng để lấy `_items`) + `SetValue`.
- [ ] Thêm tên vào `_BUILTIN_LENH_DOC_LAP` — `compiler/il_features/file_io.tkv` **dòng 75**.
- [ ] Self-host `tools/vong.exe` → `IL_IDENTICAL` → copy `dist/tkvc-tv.exe`.
- [ ] Wire vào `t_zeros` — `tv/training/tensor.tkv` **dòng 63** + 2 bản sao mirror
      (`tv/training/tv.training.tensor.tkv`, `tv/tests/tv.training.tensor.tkv`).
- [ ] Đo lại `_tz2`: baseline `t_zeros([768,3072])` = **6.5 ms/lan** (2.36M ptu).
      ⚠️ Mục tiêu ~0.1 ms là **ước lượng của yêu cầu, chưa đo được** — `newarr` vẫn phải
      memset ~9.4 MB nên số thật có thể cao hơn. **Báo số đo, không báo ước lượng.**
- [ ] Chạy lại gate 135/135 + bucket + loss chain lần cuối sau tất cả thay đổi.

---

## Thứ tự thực hiện

1. Xong Task 2: gate → bucket → loss chain → `_p6ms`  (không trộn task 3 vào trước khi task 2 xanh). ✅
2. H1 (budget 2 phía) ✅ → tiếp **H4a/H4b** (scratch m6 + bỏ copy 154 MB) → **H2** (request
   compiler: 48 `TkvInt::` trong `gpt2_block_bwd`) → **H3** (threading 14C) → H5 peel.
3. Rồi Task 3: builtin → danh sách builtin → self-host → `t_zeros` → đo `_tz2`.
4. Cuối cùng: chạy lại toàn bộ (gate + bucket + `_p6ms` + full-step 3 seq xen kẽ torch) và cập nhật `HANDOFF.md`.
