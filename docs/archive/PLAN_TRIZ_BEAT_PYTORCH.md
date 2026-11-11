# PLAN TRIZ: dùng phân tích TRIZ tìm hướng tiếp theo — beat PyTorch (GPT-2 124M)

Mục tiêu: xác định các hướng thử ưu tiên để rút ngắn khoảng cách
**TokenVector ~9312 ms** (mode1 seq1024, §0.98) vs **PyTorch ~4081-4191 ms**,
theo đúng nguyên tắc S6 (điều kiện dừng) và không tự ý nới contract bit-exact
(user đã relax BLAS/flash, mọi thay đổi đều cần verify mode0 EXACT + gate).

## 0. Bối cảnh (số đo thực 2026-11-07 §0.98)

| seq | TV fwd+bwd | TV mode1 | PyTorch full step | Loss mode0 |
|-----|-----------|----------|-------------------|------------|
| 8   | 2250      | —        | 1042.7            | EXACT      |
| 256 | 4672      | —        | ~1573             | EXACT      |
| 1024| 8656      | 9312     | ~4081-4191        | EXACT      |

Mode2: fwd 2281 + bwd 6157 + opt 953 + copyback 125 = 9626 ms.

## 1. TRIZ: mâu thuẫn kỹ thuật cốt lõi

| # | Cải thiện | Làm tệ đi | Hiện thân |
|---|-----------|-----------|-----------|
| TC1 | Song song ↑ (12 worker × op × layer) | Spawn/join overhead, nhiễu latency | Hàng trăm thread spawn/pair nhỏ mỗi step |
| TC2 | BLAS/native dispatch ↑ | Alloc + copy round-trip ↑ | `dw` tạm -> gflat -> r3 -> list p/p2 |
| TC3 | Managed `list[f32]` đơn giản | Pin/unpin từng call native ↑ | Mọi `_pt_*` wrapper gọi `_items` pin→call→unpin mỗi layer ×12 |
| TC4 | Tolerance fidelity ↑ | Không đổi được FMA order/GELU/exp | Accumulate phải y nguyên order |
| TC5 | Scratch reuse ↑ (sc4/sc5) | Chưa phủ hết intermediate | Nhiều `t_zeros` tạo/phá lại LLC |

**Mâu thuẫn vật lý:** entity tính toán vừa cần "nhỏ, stateless" (rẻ, đơn giản) vừa cần
"lớn, stateful" (bền vững, cache-friendly, reuse scratch).
**Tách biệt (TRIZ §Separation):** interpreter chỉ điều phối; native stateful giữ vùng
computed; managed lists chỉ là façade mỏng, không sở hữu compute.

## 2. Tài nguyên (Resources đã có nhưng chưa khai thác)

- Copies 3 lớp (`dw` tạm → gflat → copy r3 → list p copy-back) ~350-600 ms/bước.
- Pin/unpin `_items` mỗi call native ~ vài trăm lần/bước (reflection + barrier mem).
- Join rải rác thread hàng trăm/step thay vì barrier-pool.
- Scratch chưa phủ hết intermediate (sc4/sc5/p-data đã có — mở rộng).
- `poff` dựng lại mỗi step; `_AtnJob` setup mỗi layer.
- `_f124_bwd`/`gpt2_block_bwd` còn 2 vòng `n*h` interpreted residual add.
- `f32_copy_to_buf`/`f32_copy_from_buf` trên `opt.g` 37M phần tử mỗi step (r1 copy).

## 3. Hướng thử (D1-D7) — TRIZ operators & ước lượng

| # | Hướng thử | TRIZ operator | Vùng tác động | Lợi ước tính | Rủi ro |
|---|-----------|---------------|---------------|--------------|--------|
| D1 | Pin con trỏ 1 lần/step trong IL wrappers (per-step pinned pool thay vì pin-per-call) | Separation in time | Mọi native call (lmhead/gelu/adamw/attention/bias_grad/...) | 150-300 ms | Trung bình (IL codegen) |
| D2 | GEMM `beta=1` ghi thẳng vào gflat/dw (không buffer tạm, không copy-back) | Nested doll + Copying | lmhead/MLP/Attention GEMMs | 120-250 ms | Thấp |
| D3 | Gộp attention fwd+bwd thành 1 vùng native (12 head/call, dùng lại flash kernels; merge tức thì vào ctxT) | Segmentation | Attention fwd+bwd + merge loops | 200-400 ms | Trung bình |
| D4 | Persistent scratch pool cho intermediate (sc4/sc5 phủ hết; gflat/p trực tiếp; bỏ `t_zeros` hot) | Nested doll | Alloc hot/t_zeros trong fwd+bwd | 100-200 ms | Thấp |
| D5 | **Step-level native emit**: vong compile `_step124` core thành 1 native unit (no IL dispatch mỗi op) | Dynamics / Other way round | Toàn bộ step | 2-5× (≫ PyTorch gap) | Cao — nhiều phiên |
| D6 | Relax chính xác chọn lọc (erf→tanh trong dgelu, giữ erf gốc ở fwd) | Partial action | GELU chain | 100-180 ms | Trung bình (drift; cần phép user trước khi đổi erf gốc) |
| D7 | Persistent SPMD worker pool cho mọi elementwise (barrier sync vs spawn/join per op) | Segmentation + continuity | Mọi op spawn/join | 200-400 ms | Trung bình-cao |

## 4. Thứ tự thực hiện đề xuất

`D1 → D2 → D3 → D4` (gain/effort tốt, có thể tắt từng cái) → `D7` → `D6` (chỉ nếu user cho phép) → `D5` (khi sẵn sàng dự án nhiều phiên).

Mỗi bước: đo baseline (best-of-3 mode1 + mode2) → fix → A/B cùng phiên → verify mode0 EXACT 3 seq + mode1 `13.44291x` + trainer 15/15 + gate 9/10 (gap_tests pre-existing) → ghi HANDOFF → commit đúng file.

## 5. Điều kiện dừng

- Mỗi phase PASS loss EXACT + trainer + gate; tổng gain < ~50 ms => dừng phase kế.
- Nếu PyTorch vẫn nhanh hơn: ghi nhận & giữ baseline hiện tại — không ép.
- Bất kỳ phase nào làm gate regression => revert nguyên văn, ghi bài học vào HANDOFF.

## 6. Không làm (trừ khi user cho phép)

- Đổi erf gốc hoặc FMA order ở GEMM/softmax (loss drift ảnh hưởng `EXACT` mode0).
- Đổi `softmax_causal_fwd/bwd`, `flash_*` bit semantics.
- Reorder flatten layout AdamW.

## 7. Kết quả thực thi (cập nhật khi có phiên)

- §0.98 Flash attention fwd+bwd qua IL builtin: DONE (mode1 9312 ms; PAR12 52-82 ms).
- §0.99 GELU erf 4-way unroll: DONE nhưng trong noise — không tính là gain.
