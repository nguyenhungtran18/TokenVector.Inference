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
