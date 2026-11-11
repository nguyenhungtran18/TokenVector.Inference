# docs/archive — Tài liệu lịch sử (đã hoàn thành / bị thay thế)

Các file trong thư mục này **không còn là kế hoạch hiện hành**. Chúng được
lưu lại làm hồ sơ quyết định và bằng chứng đo — đọc kèm `docs/HANDOFF.md`
(journal theo thời gian) và `benchmark/BENCHMARKS.md` §9 (bảng đo mới nhất).

Tất cả đều là snapshot có ngày tháng; con số trong đó đã lỗi thời và
**không được trích dẫn** làm trạng thái hiện tại. Trạng thái hiện tại
(2026-11-11): GPT-2 124M full train step @seq1024 = **4016 ms**, torch
2.14.0+cpu cùng phiên = **4061.1 ms** → TKV nhanh hơn (0.989×), loss
EXACT, gate 135/135. Chi tiết: `benchmark/BENCHMARKS.md` §9.21–§9.22.

| File | Vai trò ban đầu | Vì sao lưu trữ |
|---|---|---|
| PLAN_TO_PARITY.md | Kế hoạch đầu tiên chạm ≤1.0× torch (2026-10-26) | Đã thực thi xong qua request #2–#14 |
| PLAN_TO_LE_1X.md + PLAN_TO_LE_1X_TASKS.md | Kế hoạch H1–H5 (2026-10-31) | H1–H4 đã xong; seq1024 đã đạt ≤1.0× |
| PLAN_TRIZ_BEAT_PYTORCH.md | Menu TRIZ D1–D7 (2026-11-07) | Gap ~2.5× mà nó nhắm đã đóng (0.989×) |
| PLAN_BEAT_PYTORCH_2026-11-07.md | Snapshot kế hoạch 2026-11-07 | Kết luận "CHƯA đạt ~2.5×" bị bác bởi bench 2026-11-11 |
| perf-plan-simd-vs-pytorch.md | Kế hoạch hiệu năng P0–P5 (2026-10-19) | P5.x xong; 43.5× đã thành 0.989× |
| perf-plan-triz-roadmap.md | TRIZ roadmap + GD10–13 ladder (2026-10-20) | KPI ≤1.0× đã đạt |
| spec-m4-m7-compiler-team.md | Spec builtin/ABI M4–M7 (2026-10-24) | M4–M7 delivered + wired; 10–13× gap đã đóng |
| G1_RESULT_STATE.md | Snapshot G1 tensor core/autodiff (2026-10-17) | G1/G2 xong từ lâu, xem docs/ROADMAP_TRAIN.md |
| D5_STEP_NATIVE_EMIT.md | Design doc native emit `_step124` (2026-11-08) | Mục tiêu seq1024 ≤~5000 ms đã đạt mà không cần D5 |

Quy tắc: không dùng lại nội dung trong thư mục này làm hướng dẫn thực thi;
nếu một mảnh còn giá trị, nó đã được nâng cấp vào doc hiện hành (`docs/ROADMAP_TRAIN.md`,
`benchmark/BENCHMARKS.md`, `NUMERICS_CONTRACT`, `USER_GUIDE.md`, `docs/COMPILER_TEAM_REQUEST*.md`).
