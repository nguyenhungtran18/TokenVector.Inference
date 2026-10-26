# PLAN TỚI ≤1× PYTORCH — GPT-2 124M full train step

Ngày: 2026-10-26 · Bằng chứng nền: HANDOFF §0.36–0.51, CJ8 profile, probe `_gf0`–`_gf2`,
bảng CG, `out/ab_tkv_torch_20261024.log` · Kỷ luật: **mỗi phase = đo → requirement/wire →
gate 135/135 + trainer 15/15 + loss → bench 3 seq best-of xen kẽ → HANDOFF**. Không phase
nào đi tiếp khi phase trước chưa có số (§13.2).

---

## 0. Điểm xuất phát và mục tiêu

| seq | tkv hiện tại (M6b) | torch 14T | torch 1T | tỉ lệ hiện tại |
|---|---|---|---|---|
| 8 | 7187 ms | 934.2 ms | 1322.7 ms | **7.7× / 5.4×** |
| 256 | 9609 ms | 1394.1 ms | ~1.5 s | **6.9×** |
| 1024 | 21641 ms | 3546.7 ms | — | **6.1×** |

**Mục tiêu:** full step **≤ torch 14 luồng tại CẢ 3 seq** (934 / 1394 / 3547 ms), giữ gate
135/135 + 15/15 + loss parity. Tham chiếu phụ: ≤ torch 1 luồng (1322.7 ms @seq8).

**Sàn vật lý (vì sao ≤1× là khả thi ở seq8):** step seq8 di chuyển ~4–5 GB RAM
(weights 496 MB × ~3 lần đọc/ghi qua fwd/bwd/optimizer 7 lượt p/m/v/g) — trên DDR3 của
Xeon E5-2695 v3 (~10–15 GB/s thật) sàn ≈ **400–600 ms**, dưới cả 934 ms của torch.
Seq1024 là GEMM-bound (762 GFLOP/step, torch đạt ~215 GFLOP/s tổng 14T ≈ 18% peak
Haswell) — sàn vẫn để ngỏ. Tức là thua KHÔNG phải do vật lý, mà do 3 thứ đã đo:
**copy round-trip ~40%**, **loop list[f32] interpreted trong BWD**, **threading không đối xứng**.

## 1. Vì sao plan cũ dừng ở 3–5× — và plan này đi tiếp bằng gì

Plan cũ chỉ dùng menu «1 builtin fused 1 lúc» → cắt từng miếng nhỏ. 3 lever lớn chưa đụng:

- **(a) Native data plane** — params/grads/opt state chuyển hẳn sang native buffer,
  biến mất toàn bộ round-trip copy (nửa chưa làm của spec M5).
- **(b) Codegen `list[f32]` nhanh trong tkvc** — đánh trúng root cause §0.49 **toàn cục**
  (bounds-check + indexing chậm) thay vì bóc từng builtin.
- **(c) Threading đối xứng 14 core** — máy 14C/28T, torch xài 14T; glue/elementwise/copy
  của ta 1 luồng (BUILD cpu/wall = 1.00).

---

## Phase A — Đối chuẩn 2 phía + phân rã (0 code change, 1–2 ngày)

Chưa từng làm bên torch — trước đây chỉ so total.

- **A1.** Profile torch 2.14: tách `forward / backward / optimizer` tại 3 seq, cả 1T và
  14T (`tools/bench_torch.py` mở rộng). Biết bucket nào ta đang thắng/thua thật.
- **A2.** Phân rã **BWD 4922 ms** (CJ8): matmul wrapper vs softmax-bwd / LN-bwd / gelu-bwd /
  bias-gradd / residual — cái nào interpreted, cái nào native.
- **A3.** Định phận bucket copy: `f32_copy_to_buf` 1516 ms (native mà 0.33 GB/s — **nghi
  timer/đường đi, phải sốc lại**), copy-back 844 ms (production hay bench-only?),
  `grad_norm` 390 ms.
- **A4.** Xác minh threading thật: `simd_gemm` pool đã chạy mấy thread (env
  `TKV_GEMM_THREADS` mặc định?), BUILD/ACC/elementwise có threads không.
- **Deliverable: BUDGET TABLE 2 phía** — mỗi bucket: tkv ms · torch ms · loại
  (native-slow / interpreted / threading) · lever đề nghị.

## Phase B — Native data plane: hết copy round-trip (seq8 mục tiêu −3.0÷−3.4 s)

- **B1. Compiler request #3:**
  1. `f32_list_copy(dst, src, n)` bulk **list[f32]→list[f32]** (thay copy-back interpreted);
  2. `grad_norm` fused 1-pass native (390 → ~20–50 ms);
  3. điều tra `f32_copy_to_buf` (số A3) — nếu thật chậm thì fix phía DLL;
  4. `adamw_step` nhận `g` dạng `list[f32]` trực tiếp → **bỏ hẳn r1** (1516 ms);
  5. mở rộng `f32buf` thành data plane: Tensor.data là handle vào native buffer
     (`f32buf_wrap` mẫu đã có), list↔buf bulk ở biên.
- **B2. Wire dần theo pattern M5-opt** (`pdirty`/`push_p`/`sync_mv` làm mẫu): grads +
  params sống native → biến mất r1/r2/r3/r4/r5/copy-back/zero_grad interpreted.
- **Rủi ro đã lường:** ckpt format + resume bit-exact — mọi bước phải qua trainer 15/15
  (resume case) + loss 3 seq.
- **Dự kiến:** 7187 → **~4000–4500 ms**.

## Phase C — tkvc codegen `list[f32]` nhanh (đánh root cause toàn cục)

- **C1.** Yêu cầu compiler tối ưu codegen vòng `while` trên `list[f32]`: bỏ lặp bounds-check
  (hoist phạm vi), truy cập `_items` trực tiếp, auto-vectorize AVX2. Bằng chứng kèm:
  probe `_f32p` §0.49 (f32 locals 1.0× → chi phí thật = indexing/bounds-check, **không
  phải widen kiểu** — nên lever phải là indexing, không phải typing).
- **Acceptance:** BUILD+ACC+ALLOC wrapper 4 shape giảm ≥2×, full-step giảm theo, BIT/tol
  gate giữ nguyên.
- **C2. Fallback (nếu codegen chưa đủ):** builtin từng op theo thứ tự A2 chỉ ra —
  `softmax_bwd`, `layernorm_bwd` (gộp dγ/dβ), `gelu_bwd`, `bias_grad` — request #4, #5…
  Ưu tiên bucket chiếm >5% step.
- **Dự kiến:** BWD 4922 → **~2500–3000 ms**, seq8 → **~2500–3000 ms**.

## Phase D — Threading đối xứng 14 core (đo trước, không assume)

- **D1.** `TKV_GEMM_THREADS` sweep 4/8/14 (A4 đã xác minh mặc định): probe CG mode 1 +
  full-step A/B. Nếu GEMM đã full-thread → skip.
- **D2.** Elementwise/norm/optimizer native ở Phase B thêm thread **trong DLL** (memcpy +
  bandwidth ops scale gần tuyến tính).
- **D3.** BWD parallel op-level **trong native** — không còn bài học CW/CE (merge interpreted
  chết) vì merge nằm trong DLL, kế thừa re-entrancy §3.1 đã có của `simd_gemm_full`.
- **Dự kiến:** seq8 → **~1300–1700 ms** (1.1–1.3×), seq256/1024 giảm theo GEMM threads.

## Phase E — GEMM side: compiler v2 + pack cache (đầu việc đã giao)

- **E1.** B-tile cache theo i0 (đơn 2 §6, số `_gf1/_gf2`) → S3/S5 m=1024 ≤400 ms →
  **nới routing `m ≤ 1024`**, re-gate, re-bench.
- **E2.** Pack 1 lần/step (M4 spec gốc — hiện pack mỗi call; weight bất biến trong step):
  persistent packed-handle cache theo step, free ở biên step.
- **E3.** Pad/T-dispatch chỉ tái mở nếu budget table chỉ ra pad còn ≥5% step (CB/CX từng
  bust — không reopen không có số).

## Phase F — Chốt ≤1×

- Re-bench 3 seq **xen kẽ torch fresh** (bỏ số cũ 24/10), gate, HANDOFF §0.52+.
- **Win condition:** ≤934 / ≤1394 / ≤3547 ms, loss parity, RAM seq1024 không tăng.
- Gap còn lại sau D+E (nếu có): liệt kê thành Phase G từ budget table — mỗi mục 1 dòng
  «bucket · ms · lever», không nhận chung chung.

---

## Mục tiêu định lượng theo phase (DỰ KIẾN — mọi con số chốt bằng đo ở phase tương ứng)

| Giai đoạn | seq8 | tỉ lệ (vs 14T) | seq256 | seq1024 |
|---|---|---|---|---|
| Hiện tại (M6b) | 7187 | 7.7× | 9609 | 21641 |
| +B native data plane | ~4200 | 4.5× | ~6500 | ~17000 |
| +C codegen list[f32] | ~2800 | 3.0× | ~4500 | ~12000 |
| +D threading 14C | ~1500 | 1.6× | ~2600 | ~7000 |
| +E GEMM v2 + pack cache | ~1200 | 1.3× | ~1900 | ~5000 |
| Peel cuối theo budget table | **≤934** | **≤1×** | **≤1394** | **≤3547** |

**Rủi ro lớn nhất:** Phase C (codegen) là đòn bẩy 40–50% nhưng thuộc phía compiler team —
nếu họ đánh giá không cải thiện được ≥2×, rơi về C2 (builtin từng op) → lộ trình dài hơn
vẫn đi đúng hướng nhưng thêm vài request. Mọi phase còn lại không phụ thuộc C.