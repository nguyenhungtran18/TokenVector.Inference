# SIMD trong TokenVector — trạng thái hiện tại (2026-11-11)

Bản khảo sát cũ (2026-10-15, kết luận "chưa dùng được SIMD") đã hết hiệu lực.
Hồ sơ đầy đủ các bước đi và mọi lần đo sai nằm ở `HANDOFF.md` (journal) và
`BENCHMARKS.md` §9 (bảng đo hiện tại). Tài liệu gốc được lưu trong git history.

## Ba tầng SIMD tồn tại hôm nay

1. **Tầng ngôn ngữ** — builtin `simd_*` (`simd_add/sub/mul/scale/axpy`,
   `simd_dot/sum/max/min`, `simd_lanes_*`, `simd_hw`, `simd_isa`); compiler
   tự phát lớp IL `TkvSimd` bằng `System.Numerics.Vector<T>`.
2. **Tầng native** — `tkv_simd_native.dll` (repo compiler `D:\TokenVector`),
   export AVX2+FMA: `SgemmF32/F64`, `Add/Sub/Mul/Scale/Axpy/Dot/Sum/Max/Min`
   (mở rộng F64/đa luồng phải sửa C, ngoài tầm repo này). Các builtin nóng
   (`adamw_step_glm`, `f32_*`, fused bwd…) P/Invoke vào DLL này.
3. **Tầng codegen** — matcher P6 của compiler vec hóa thân vòng `list[f32]`
   (`VEC_W=8` là fallback lịch sử; đường production đi native).

## Số đo hiện tại (cùng phiên, 2026-11-11, BENCHMARKS §9.22)

- Full step GPT-2 124M @seq1024: **4016 ms** vs torch 4061.1 → **0.989×**.
- opt.step: **453 ms** @seq1024 (trước request #14: 781 ms).
- fwd nhanh hơn torch (−55 ms), bwd xấp xỉ (−29 ms), `opt` nhanh hơn −193 ms.

## Quy tắc không đổi

- Không quote số SIMD cũ (trước 2026-10) — kiến trúc đã đổi 3 lần.
- `gc_*` / `native_arena_alloc` chưa có counter thật; arena **không** phải
  zero-alloc (tensor vẫn là boxed `list[f32]` ở nhiều chỗ) — xem `USER_GUIDE.md`.
- Mọi kết luận về hiệu năng phải đo trên binary build từ `D:\TokenVector\Release\dist\tkvc-tv.exe`
  với native DLL export mới nhất; binary thiếu export sẽ âm thầm rơi về scalar.
