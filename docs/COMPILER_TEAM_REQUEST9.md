# YÊU CẦU COMPILER TEAM #9 — H2 block_bwd + H4b zero-copy managed data

**Ngày:** 2026-11-01 · **Nguồn:** PLAN_TO_LE_1X.md §1 (H2, H4) + HANDOFF §0.59 + docs/PLAN_TO_LE_1X_TASKS.md
**Đối tượng:** Compiler team (tkvc codegen P6 + native runtime) · **Ưu tiên:** cao — bucket lớn nhất còn lại

## 1. Bối cảnh

- Gate 135/135 + trainer 15/15 xanh sau H4a/H4c.
- Best-of-3 sau H4a+H4c: seq8 3750 ms, seq256 5734 ms, seq1024 16203 ms (baseline H1: 4141/6015/16094).
- Gap còn lại chủ yếu: `12 × block_bwd` = 1617 ms = 65% bwd; fixed-cost 578 ms (đã giảm một phần qua H4a/c, còn copy_to 154 MB ×3 + copy_from 2×).

## 2. Ask A — H2: dọn `TkvInt::` trong `gpt2_block_bwd`

- File: `tv/training/gpt2_ref.tkv` / `gpt2_ref124` mirrors, hàm `gpt2_block_bwd`.
- Baseline: **48 lệnh `TkvInt::`** còn sót (§0.58 mục 5).
- Yêu cầu:
  1. Dọn nốt ceremony #7 trong `gpt2_block_bwd`.
  2. Hoist vòng `mu` reduction sang byref `_items` + range-proof (§0.56.2 mục 4).
  3. LICM invariant `m1/m2/rstd` + bỏ double-load `gy` (§0.56.2 mục 2–3).
- Kỳ vọng: bucket 1617 ms → ~45 ms/gọi (−67%); bwd seq8 2500 → ~1400 ms.
- Fallback C2 (4 builtin `softmax_bwd/layernorm_bwd/gelu_bwd/bias_grad`) vẫn hiệu lực nếu matcher không ăn shape.

## 3. Ask B — H4b: zero-copy managed tensor data cho native fused

- Hiện tại `lmhead_bwd_fused` chỉ nhận handle `alloc_bytes` (i64). Để gọi fused phải `f32_copy_to_buf` 3×154 MB + `f32_copy_from_buf` 2×154 MB.
- Yêu cầu: cung cấp cách fused đọc/ghi trực tiếp buffer persistent của `gpt2_backward` (managed `.data` hoặc handle zero-copy), để bỏ copy 154 MB ×5.
- Nếu không thể zero-copy, thêm builtin fused variant nhận `list[f32]` trực tiếp.

## 4. Điều kiện

- Giữ loss EXACT (không reorder reduction).
- Gate 135/135 + trainer 15/15 sau mỗi thay đổi.
- Self-host `tools/vong.exe` → `IL_IDENTICAL`.
- Báo số đo xen kẽ torch cùng phiên trước khi ghi HANDOFF.