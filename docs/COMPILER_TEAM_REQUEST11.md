# YÊU CẦU COMPILER TEAM #11 — bỏ serialize GEMM giữa các worker: `attention_bwd` 4116 → ~1200 ms/12 block

**Ngày:** 2026-11-05 · **Nguồn:** HANDOFF §0.68 + probe `tv/tests/_bd1024.tkv`, `_mb1024.tkv`
**Đối tượng:** Compiler team (native runtime, `tkv_simd_native.c`) · **Ưu tiên:** cao — lever lớn nhất từng đo được
**KHÁC #10:** #10 là thêm builtin fused. Lần này **không cần builtin mới** (nếu chọn cách a), và
**không đổi một dòng `.tkv` nào**. Vấn đề là **thread-safety của dispatcher GEMM**.

---

## 1. Triệu chứng đo được

Cùng binary, cùng dữ liệu, chỉ đổi số luồng (`tv/tests/_bd1024.tkv`, production `gflat` + saves):

| vùng / block | 1 luồng | 14 luồng | scale |
|---|---:|---:|---:|
| `gpt2_mlp_bwd` | 781 | 250 | **3.12×** |
| `gpt2_attention_bwd` | 594–609 | **343** | **1.75×** |
| `gpt2_block_bwd` | 1421 | 641 | 2.22× |
| GEMM standalone (`out/gm.exe`, shape `1024×3072×768`) | 34.5 GFLOP/s | 155.9 GFLOP/s | **4.52×** |

Ở cấu hình đích (14 luồng, seq1024), `attention_bwd` × 12 block = **4116 ms = 29% của bước
14.3 s** — mục lớn nhất. Nó cũng là mục scale kém nhất.

## 2. Nguyên nhân

`gpt2_attention_bwd` với `n ≥ 128` spawn **12 worker, mỗi worker một head**. Trong mỗi head có
6 lời gọi GEMM (`S = Q·Kᵀ`, `ctx = P·V`, `dp = dctx·V`, `dq = ds·K`, `dkT = Qᵀ·ds`,
`dvT = dctxᵀ·P`), cộng `gpt2_linear` (qkv) và `smm_dot` (dctxT) ở đầu hàm
⇒ **≈ 74 lời gọi GEMM / block**, mỗi lời gọi `1024×1024×64` = **134 MFLOP**.

| | số |
|---|---|
| công việc GEMM thực / block | 11.2 GF |
| ở 155.9 GFLOP/s | **0.86 ms / lời gọi** |
| thời gian thực tế ở 14 luồng | **4.6 ms / lời gọi** |
| phần dôi ra | **81%** |

Hai nguyên nhân, cả hai đều trong `tkv_simd_native.c`:

1. **`g_gemm_lock` (SRWLOCK) + `g_job[TKV_MAXT]` toàn cục.** `tkv_run_mt` / `tkv_run_mt_rect`
   `AcquireSRWLockExclusive` rồi giữ lock **suốt thời gian tính**. 12 worker gọi GEMM đồng
   thời ⇒ 11 worker đứng chờ, mỗi lần chờ trả giá thêm chi phí dispatch.
2. **Kích thước quá nhỏ cho 14 luồng.** 134 MFLOP không đủ bù `SetEvent` × nt +
   `WaitForMultipleObjects` + chia lưới 2D; `tkv_nthreads(1024)` trả 14 nên mỗi lời gọi lại
   đánh thức cả pool.

Bằng chứng phụ (đo riêng, `tv/tests/_exp1024.tkv`): `exp()` = 22.2 ns ⇒ softmax causal 6.3M
exp = 140 ms ở 1 luồng, và nó **đã** chạy song song trong 12 worker. Vòng cộng 6.3M phần tử
**< 15.6 ms** (2.5 ns/ptu). Vậy **không phải** softmax, **không phải** loop `.tkv` chậm —
đúng là chi phí mỗi lời gọi GEMM.

## 3. ASK A (khuyến nghị) — job slot theo thread, bỏ lock toàn cục

Mục tiêu: **N thread gọi `SgemmF32`/`smm_dot` đồng thời phải chạy song song thật**, không
xếp hàng. Đề nghị:

1. `g_job` / `g_start` / `g_done` chuyển thành **mảng theo "chủ sở hữu" (tls index hoặc slot
   lấy từ pool)** thay vì một bộ toàn cục. Mỗi caller chiếm một slot trong lúc chạy
   (`InterlockedIncrement` một counter slot), nhả khi `WaitForMultipleObjects` xong.
2. Bỏ `AcquireSRWLockExclusive` khỏi đường nóng; chỉ giữ lock ở **khởi tạo pool**
   (`InitOnce`) và ở cấp phát slot.
3. Nếu `nt == 1` (trường hợp phổ biến của GEMM đầu head), **không** đi qua pool: gọi thẳng
   `gemm_rect_f32` trên thread gọi. Điều này một mình đã cắt được phần lớn overhead khi 12
   worker đã bão hoà core.

**Ràng buộc bit-exact:** thay đổi này **không** được đổi cách chia vùng `[i0,i1)×[j0,j1)` của
từng lời gọi. Thứ tự tích luỹ `k` cho mỗi phần tử vẫn là `p` tăng dần trong một `micro_f32`
⇒ kết quả **bit-giống**. Cần xác nhận bằng `tools/gate_f32.sh` 135/135 + trainer 15/15 +
`_p6ms`/`_am` `bit_bad=0` như các request trước.

**Trần lợi nhuận:** `attention_bwd` 343 → ~80–120 ms/block ⇒ **4116 → ~1200 ms**,
tức **−2.9 s = −20% bước** (14.3 s → ~11.4 s). Không đòi viết lại kernel, không đổi ABI.

## 4. ASK B (thay thế) — builtin GEMM theo lô

Nếu ASK A không khả thi: `smm_dot_batch(B, a, b, n, in, out)` với `B = 12` head cùng shape,
hoặc tương đương `smm_fwd_batch`. Khi đó số lời gọi giảm **72 → 6** cho một `attention_bwd`,
và mỗi lời gọi đủ lớn (1.6 GF) để chia 14 luồng có nghĩa.

Chi phí: builtin mới + wiring codegen như #10 Ask B (pinvoke + wrapper IL + register).
Chữ ký `.tkv` đổi ở 6 chỗ trong `gpt2_attention_bwd` × 5 mirror.

## 5. Cách đo lại sau khi sửa

```
TKV_GEMM_THREADS=1 dotnet out/bd1024.exe     # 1 luồng: ATTENTION_BWD ~594
dotnet out/bd1024.exe                        # 14 luồng: ATTENTION_BWD hiện 343
```

Kỳ vọng sau ASK A: số 14 luồng giảm còn ~80–120 ms/block trong khi số 1 luồng **gần như
không đổi** (vì ở 1 luồng GEMM đã single-thread và không có tranh chấp).
Tiêu chí đạt: `ATTENTION_BWD` ≤ 150 ms ở 14 luồng, `BLOCK_BWD` ≤ 400 ms.

## 6. Điều KHÔNG cần làm

- Không tối ưu softmax/`exp` — đã đo, 140 ms/block ở 1 luồng và đã song song.
- Không đổi tile `GEMM_MC/NC/KC` cho shape lớn — đã sweep (HANDOFF §0.65 mục 4), tiling hiện
  tại tối ưu, NC lớn còn chậm hơn.
- Không viết lại micro-kernel cho mục này — 4 biến thể đã thử đều không nhanh hơn 4×16.

---

## 7. Kiểm chứng bằng số học từ 2 phép đo độc lập

Không cần tin vào việc "đọc code thấy có lock": tách `attention_bwd` theo khối lượng đo được.

Một block có 2 GEMM lớn + 72 GEMM nhỏ:

| thành phần | GF | ở 1 luồng (33.6 GFLOP/s) | ở 14 luồng (155.9) |
|---|---:|---:|---:|
| `gpt2_linear` qkv (`1024×2304×768`) | 3.62 | 108 ms | 23 ms |
| `smm_dot` dctxT (`1024×768×768`) | 1.21 | 36 ms | 8 ms |
| 72 GEMM nhỏ `1024×1024×64` (6 × 12 head) | 9.66 | 288 ms | **?** |
| Σ GEMM | 14.5 | 432 ms | |
| **`attention_bwd` đo được** | | **594 ms** | **343 ms** |

- 1 luồng: 432 ms GEMM + **162 ms** non-GEMM = 594 ✓
- 14 luồng: 2 GEMM lớn = 31 ms; 343 − 31 − (162 ms non-GEMM, đã song song sẵn nên ~40 ms)
  ⇒ **còn ~270–310 ms cho 72 GEMM nhỏ ⇒ 3.8–4.3 ms/lời gọi**, trong khi công việc chỉ
  0.86 ms ⇒ **đúng như dự đoán 78–82% overhead, và không hề giảm khi tăng số luồng.**

Nếu 12 worker thật sự chạy song song 72 GEMM đó trên 7 core: 288 ms/1 luồng ÷ ~6 = **48 ms**.
Đó là toàn bộ chênh lệch giữa 343 ms và ~90 ms.

**Lưu ý cho người muốn chứng minh trực tiếp:** probe `thread_spawn` + `smm_dot` với `Tensor`
truyền qua job **không biên dịch được** (`InvalidProgramException` — job chỉ nên mang
`list[f32]`/số, Tensor phải tạo bên trong worker; và `thread_join` phải gán vào biến).

---

## 8. ĐÍNH CHÍNH (2026-11-05, sau khi đo thêm): KHÔNG phải lock, mà là chi phí cố định mỗi lời gọi

Mục 2 ở trên (và HANDOFF §0.68) quy tội cho `g_gemm_lock`. **Đã thử sửa và số đo bác bỏ.**

### 8.1 Thí nghiệm 1 — bỏ lock (làm thật, đã revert)

Sửa `tkv_run_mt` / `tkv_run_mt_rect` / `tkv_run_aux` sang `TryAcquireSRWLockExclusive`; nếu pool
bận thì thread gọi **chạy thẳng kernel cho cả vùng** (`gemm_rect_f32(a,lda,b,ldb,c,ldc,m,n,k,
0,m,0,n,…)`), an toàn vì buffer panel B là `__declspec(thread)` và nhánh đó không chạm
`g_job`/`g_start`/`g_done`.

Kết quả `ATTENTION_BWD` (seq1024, 14 luồng): **328 → 312 ms**. Trong nhiễu.

⇒ **Lock không phải nguyên nhân.** (Đã revert source + rebuild về nguyên trạng.)

### 8.2 Thí nghiệm 2 — quét `k` ở `n = 1024` (`tv/tests/_g64.tkv`, 14 luồng)

Cùng `smm_dot(a[n,k], b[n,k], n, n, k)` ⇒ `[n,n]`:

| k | ms / lời gọi | GFLOP/s |
|---:|---:|---:|
| 64 (đúng shape attention) | 9.75 | **13.8** |
| 128 | 11.75 | 22.8 |
| **256** | **5.875** | **91.4** |
| 512 | 9.75 | 110.1 |
| 768 | 11.625 | 138.5 |

**k=256 nhanh hơn k=64 về thời gian tuyệt đối** dù khối lượng gấp 4 lần. Hồi quy
`(11.625−5.875)/(768−256) = 0.0112 ms/k` ⇒ **chi phí cố định ≈ 3.0–5.5 ms mỗi lời gọi**, độc lập
với `k`. Với `k = 64`, công việc thật chỉ **0.86 ms** ⇒ 85% thời gian là chi phí cố định.

Đây mới là con số đúng: **72 lời gọi × ~9.75 ms = 702 ms tuần tự**; đo được 328 ms vì 12 worker
chạy song song ~2.1×. Muốn `attention_bwd` xuống ~120 ms thì phải **diệt chi phí cố định**, không
phải diệt lock.

### 8.3 Chi phí cố định nằm ở đâu — 4 nghi phạm, xếp theo khả năng

| # | nghi phạm | bằng chứng / cách kiểm |
|---|---|---|
| 1 | `memset` C `n×n = 4 MB` trong `gemm_rect_f32` **+** epilogue đọc–sửa–ghi C (thêm 8 MB) mỗi lời gọi | 12 MB/lời gọi; nếu đúng thì giảm bằng cách bỏ memset khi caller đã zero |
| 2 | cấp phát + zero `n×n` tạm trong wrapper `.tkv` (`Smat`, `gDp`, `gDs` = 4 MB mỗi cái, mỗi head) | thử cache scratch theo worker như nhánh `n < 128` đã làm |
| 3 | `tkv_k_eff()` — quét **O(n²)** (`for p from n-1`, mỗi p đọc cả hàng `b` **và** cột `a`) ⇒ ~2M lượt đọc bất kể `k` | đây là hàm duy nhất trong code có chi phí đúng dạng "hằng số theo k, tỉ lệ n²"; đo bằng cách gọi `SgemmF32` vuông với dữ liệu không có hàng 0 cuối |
| 4 | dispatch pool + chia lưới (đã loại phần lớn bằng 8.1) | — |

Nghi phạm 3 khớp chính xác hình dạng số đo (hằng số theo `k`, `n²`): 2×1M lượt đọc ở ~2.5 ns =
5 ms. **Đề nghị kiểm nó trước** — nếu đúng, chỉ cần một cờ "k đã biết, bỏ k_eff" cho đường rect
là xong, và nó cũng tăng tốc mọi GEMM của mô hình.

### 8.4 Đề nghị sửa lại (thay mục 3/4)

- **ASK A'**: cho `smm_dot`/`smm_fwd` đường rect **bỏ `tkv_k_eff`** (hoặc chỉ gọi khi caller báo
  `k` chưa chắc). Đây là thay đổi vài dòng, không đổi chữ ký `.tkv`, và nếu đúng là thủ phạm thì
  tiết kiệm ~3–5 ms cho **mỗi** lời gọi GEMM rect trong toàn bộ mô hình.
- **ASK B'** (repo-side, không cần native): cache 3 scratch `n×n` theo worker trong
  `gpt2_attention_bwd` nhánh `n ≥ 128` (nhánh `n < 128` đã làm đúng việc này từ YÊU CẦU #10 C.2).
- **ASK C'**: batched GEMM theo head (như mục 4) vẫn có giá trị, nhưng **không còn là ưu tiên 1**
  vì chi phí cố định mới là thứ chi phối.
