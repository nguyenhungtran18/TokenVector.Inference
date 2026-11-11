# PLAN TỐI THƯỢNG: ĐẠT ≤1× PYTORCH (VÀ <1×) — GPT-2 124M FULL TRAIN STEP

**Ngày:** 2026-10-31 · **Căn cứ:** HANDOFF §0.59 (H1 — đo xen kẽ cùng phiên), §0.58, PLAN_TO_PARITY.md (giai đoạn A–F), docs/COMPILER_TEAM_REQUEST8.md.  
**Mục tiêu:** full step **≤ torch 14 luồng tại CẢ 3 seq** (1042.7 / 1572.7 / 4148.4 ms — số fresh cùng phiên 31/10), giữ gate 135/135 + trainer 15/15 + loss EXACT. Mục tiêu tối thượng: sau khi chốt ≤1×, tiếp tục cắt để **<1×** (ví dụ 0.8×).

---

## 0. Chân dung gap phải đóng (số §0.59 — torch và tkv đo xen kẽ cùng phiên)

| seq | tkv full | torch 14T | tỉ lệ | **phải cắt** | torch 1T | tỉ lệ vs 1T |
|---|---|---|---|---|---|---|
| 8 | 3844 ms | 1042.7 ms | **3.69×** | −2801 ms (−72.9%) | 1388.0 ms | 2.77× |
| 256 | 5844 ms | 1572.7 ms | **3.72×** | −4271 ms (−73.1%) | 4354.8 ms | 1.34× |
| 1024 | 16078 ms | 4148.4 ms | **3.88×** | −11930 ms (−74.2%) | 15082.1 ms | **1.07×** |

**Bốn sự thật đã đo phân bổ gap:**

1. **Backward là 73–79% toàn bộ gap** ở cả 3 seq (seq8 +2207/2801, seq256 +3366/4271, seq1024 +8720/11930). Forward chỉ đáng kể ở seq1024 và ở đó tkv **nhanh hơn torch 1T 25%** (3641 vs 4825.7 ms).
2. **`12 × block_bwd` = 1617 ms = 4.02× toàn bộ bwd torch 1T (402.2 ms)** và chiếm **65% bwd tkv**. Trong bucket, phần đã vector hoá (`ATTENTION_BWD` 47 + `MLP_BWD` 62) chỉ 109/134.75 ms → **~26 ms/gọi là ceremony/`TkvInt::`/copy**.
3. **GEMM đã parity** (S4/S5 207/211 ms thắng torch 254/240 ms; 552–588 GFLOP/s pad ≈ torch useful) → kernel GEMM **không phải** thủ phạm.
4. **93% khoảng 3.88× ở seq1024 là threading của torch**, không phải kernel tkv chậm: torch tự nó chỉ scale **1.33× / 2.77× / 3.64×** (seq8/256/1024) trên 14 luồng. Ở seq nhỏ, tkv còn mất vì **578 ms chi phí cố định 38.6M** (m6 344 + `op_embed_bwd` 234).

> **Hệ quả trực tiếp:** trần của lever threading = đúng hiệu suất song song của torch (3.64×) ⇒ 16078/3.64 ≈ **4417 ms ≈ 1.06×**, *không đủ ≤1×* ngay cả trong giả định lạc quan nhất. Vì vậy H2 và H3 phải đi **cùng nhau**, không phải "hoặc".

---

## 1. Lộ trình lever (ưu tiên theo đòn bẩy đã đo)

### H1 — Bảng budget 2 phía  ✅ HOÀN THÀNH 31/10 (HANDOFF §0.59)

Đã đo xen kẽ cùng phiên: tkv mode 0 + mode 1 (best-of-3), `_bd2` + `simd_probe_am` + `_p6ms`
rebuild bằng compiler 31/10 09:19, torch tier D/E ở **T=1 và T=14**. Log `out/h1_*.log`.

**Kết quả chốt:** tỉ lệ cùng phiên **3.69× / 3.72× / 3.88×** (không phải 4.07–4.62× của
baseline 24/10 — máy hôm nay chậm hơn ~9% và tỉ lệ phải đo cùng phiên). Phân bổ gap:
**bwd 73–79%**, trong đó `12 × block_bwd` 65% của bwd; fwd chỉ 13–21%; opt+copy 2–8%.

**Kill criterion đã áp:** bucket nào <5% bước bị loại. Kết quả: 2 vòng dense 38.6M
(`dwte +=`, `gflat +=`) chỉ ~62 ms = 2.5% bwd ⇒ **loại khỏi kế hoạch**; wpe loop 0.8M
nằm dưới sàn phân giải `now_ms` ⇒ loại.

### H2 — Phase C còn lại: codegen nuốt hết bucket block_bwd (đòn bẩy đã đo rõ nhất)

**Bằng chứng:** `_p6diag` (body-only) đo **A 5.61× · B 3.85–4.24× · C 5.00–6.45×** — thân vòng đã vectorize thì nhanh gấp 4–5×, nhưng production mới nuốt phần. `gpt2_block_bwd` **vẫn còn 48 lệnh `TkvInt::`** (§0.58 mục 5).

- **Hành động (request compiler team, tương tác theo COMPILER_TEAM_REQUEST8.md):**
  1. **Dọn nốt `TkvInt::` trong `gpt2_block_bwd`** — ceremonia #7 chưa hết ở method này; là điều kiện để #8 (thân N câu) có giá trị.
  2. **Hoist vòng `mu` reduction sang byref `_items` + range-proof** (§0.56.2 mục 4 — đang chặn bởi «hỏi nghi = từ chối»).
  3. **LICM invariant `m1/m2/rstd` + bỏ double-load `gy`** (§0.56.2 mục 2–3).
  4. Fallback **C2** (4 builtin `softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad`) vẫn hiệu lực nếu matcher không ăn shape nào.
- **Kỳ vọng (ước lượng — phải đo):** bucket 1617 → **540 ms** nếu nuốt trọn ceremony về
  mức `SUM_cac_vung` (109 ms/lớp × 12 = 1308 → nhưng `BLOCK_BWD` đã gồm cả phần đó, nên
  mục tiêu thực tế là **134.75 → ~45 ms/gọi** = −67%) ⇒ bwd seq8 2500 → ~1400 ms.
  Lever rẻ nhất, nhưng chưa đủ đơn lẻ.

### H3 — Phase D: threading đối xứng 14C/28T — lever lớn nhất, **trần đã đo được**

**Bằng chứng:** A4 đo cả bước **1.37 thread**; §0.59 đo torch tự scale **1.33× / 2.77× /
3.64×** (seq8/256/1024) trên 14 luồng. Trần tuyệt đối nếu tkv đạt y hệt: seq1024
16078/3.64 ≈ 4417 ms = **1.06×**; seq256 5844/2.77 ≈ 2110 = 1.34×; seq8 gần như không
có gì để lấy (1.33×). **⇒ threading một mình không thể đưa về ≤1×; phải sau H2.**

- **Hành động:**
  1. `TKV_GEMM_THREADS` sweep 4/8/14 (A4 đã xác minh mặc định): probe CG mode 1 + full-step A/B. Nếu GEMM đã full-thread → skip.
  2. Elementwise/norm/optimizer native ở Phase B thêm thread **trong DLL** (memcpy + bandwidth ops scale gần tuyến tính).
  3. BWD parallel op-level **trong native** — không còn bài học CW/CE (merge interpreted chết) vì merge nằm trong DLL, kế thừa re-entrancy §3.1 của `simd_gemm_full`.
- **Kỳ vọng:** seq8 → **~1300–1700 ms** (1.1–1.3×), seq256/1024 giảm theo GEMM threads.
- **Rủi ro:** đã từng bust 8 lần — phải đo trước, không assume.

### H4 — Chi phí cố định 38.6M (m6 + `op_embed_bwd`) — **bắt buộc cho seq8/256**

§0.59 đo: `BD2 [4]` m6 section **344 ms** (6 alloc + 3 copy_to + zero + fused + 2 copy_from
+ 6 free, mỗi buffer 154 MB) và `BD2 [5]` `op_embed_bwd ×2` **234 ms** = **578 ms/step**
không co theo seq. Đó là lý do seq8 chỉ 3.69× trong khi fwd+bwd thuần 7.99×.

- **H4a.** Giữ scratch `sc4/sc5` ở cấp **step** (đã có trong `_step124`) nhưng cấp thêm
  scratch cho m6 (`plg/pids/pwte/pxf/pdxf/pdwe`) — bỏ 6 alloc + 6 free mỗi lớp-bước.
- **H4b.** Bỏ các `copy_to` không cần: `fused` đọc/ghi thẳng buffer persistent của
  `gpt2_backward` thay vì copy 154 MB vào rồi copy 154 MB ra.
- **H4c.** `op_embed_bwd` ×2 (234 ms) — gộp scatter `wte`/`wpe` một lượt, hoặc native
  scatter (`gx` 8×768 nên đây là bandwidth thuần 38.6M × 4 B × 2).
- **Kỳ vọng:** 578 → ~150–200 ms ⇒ seq8 −380~−430 ms. **Đây là lever duy nhất giúp seq8/256 nhiều hơn seq1024.**

### H4' — GEMM side: pack cache + compiler v2 (phụ, không chặn)

- **E1.** B-tile cache theo i0 (đơn 2 §6, số `_gf1/_gf2`) → S3/S5 m=1024 ≤400 ms → nới routing `m ≤ 1024`.
- **E2.** Pack 1 lần/step (M4 spec gốc — hiện pack mỗi call; weight bất biến trong step): persistent packed-handle cache theo step, free ở biên step.
- **E3.** Pad/T-dispatch chỉ tái mở nếu budget table chỉ ra pad còn ≥5% step (hiện chưa thấy).

### H5 — Peel cuối theo budget table

Sau H2+H3+H4, gap còn lại (nếu có) liệt kê thành mục 1 dòng: «bucket · ms · lever», không nhận chung chung.

---

## 2. Mốc định lượng (dự kiến — chốt bằng đo)

Torch mốc = số fresh §0.59: **1042.7 / 1572.7 / 4148.4 ms**.

| Giai đoạn | seq8 | tỉ lệ | seq256 | seq1024 |
|---|---|---|---|---|
| Hôm nay (§0.59) | 3844 | 3.69× | 5844 | 16078 |
| +H2 codegen block_bwd (−67% bucket) | ~3000 | 2.9× | ~4700 | ~14000 |
| +H4 fixed-cost 38.6M (578→~180) | ~2600 | 2.5× | ~4300 | ~13600 |
| +H3 threading 14C (hiệu suất = torch) | ~1950 | 1.9× | ~1550 | **~3740** |
| Peel cuối (H5) | **≤1043** | **≤1×** | **≤1573** | **≤4148** |
| <1× (tối thượng) | **<1043** | **<1×** | **<1573** | **<4148** |

⚠️ Các dòng +H2/+H4/+H3 là **dự báo dựa trên số đo từng phần**, chưa phải kết quả. Sau mỗi
lever phải đo lại xen kẽ torch cùng phiên rồi mới ghi tiếp — đúng luật §3.

---

## 3. Điều kiện không thương lượng (giữ từ TODO_M9.md)

- Loss phải **EXACT** — không đổi thứ tự cộng (reduction không re-order).
- `tools/gate_f32.sh` phải xanh **135/135** sau mỗi thay đổi.
- Đổi compiler → phải self-host `tools/vong.exe` → `IL_IDENTICAL`.
- **Đo trước, sửa sau** — không nhận định hiệu năng khi chưa có số đo.
- Mỗi phase = **đo → requirement/wire → gate 135/135 + trainer 15/15 + loss → bench 3 seq xen kẽ → HANDOFF**. Không phase nào đi tiếp khi phase trước chưa có số.

---

## 4. Lựa chọn tiếp theo (cần bạn xác nhận)

H1 đã xong (§0.59). Các bước còn lại, xếp theo tỉ lệ «ms cắt được / ngày công» đã đo:

| Lựa chọn | Mô tả | Ưu tiên |
|---|---|---|
| **A (Recommended)** | **H4a/H4b** — giữ scratch m6 + bỏ copy 154 MB trong repo (không cần compiler team, tự làm được ngay, đo được ngay) | Cao |
| **B** | **H2** — soạn request compiler team: dọn 48 `TkvInt::` trong `gpt2_block_bwd` + hoist `mu` + LICM | Cao (phụ thuộc team khác) |
| **C** | **H3** — sweep `TKV_GEMM_THREADS` 4/8/14 + thread native cho elementwise/norm/optimizer | Trung bình (sau H2) |
| **D** | Task 3 (`zeros_list` builtin) — không chặn performance | Thấp |