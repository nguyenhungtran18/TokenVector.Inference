# D5 — Step-level native emit cho GPT-2 124M training step (Design doc)

Ngày viết: 2026-11-08 (sau §0.108)

## 0. Tại sao D5 tồn tại, khoảng cách hiện tại vs PyTorch (đo thật 2026-11-07/08)

| seq | TV full step (§0.105/0.106/0.107) | PyTorch full step | Ratio |
|---|---|---|---|
| 8   | ~1120 ms (63+1047+~10) | 1042.7 ms | ~1.07x |
| 256 | ~2406 ms (469+1937+~10) | ~1573 ms (ước lượng) | ~1.53x |
| 1024| ~6328 ms (2031+4297+875+125) | ~4190 ms | ~1.51x |
| mode0 (no opt) | fwd 2031+bwd 4297 = 6328 (có OPT) | 4190.6 (PyTorch full step) | ~1.5x |

**Profile mới nhất (dotnet-trace §0.107):** ~99.6% samples nằm trong
`UNMANAGED_CODE_TIME` — native GEMM/native kernel. Khoảng cách còn lại so với
PyTorch **không phải** ở GEMM mà ở:
- Overhead managed list (alloc, pin/unpin, boxing `float32`, `List<float32>`
  layout chứa read của `.data` mỗi call).
- IL dispatch per op: `.callvirt get_Item/set_Item` — 51M+ lần/step.
- Thread spawn/join quanh mỗi op parallel (~2M threads/bước).
- AdamW copy_back 37M phẩn tử/bước (f32_list_add / p -> gflat round-trip).
- `Tensor::offset` / save_abi / list/box lớn gây G1 GC spikes.

=> Đổi kernel BLAS/oneDNN không đóng được gap. Phải **thay đổi runtime architecture**
bên trên managed `list[f32]`.

## 1. Mục tiêu (win condition)

- full step seq1024 ≤ **~5000 ms** (~1.2x PyTorch 4190.6, từ ~6328 hiện tại — tối đa
  giảm ~1300 ms từ D5; phanh lớn hơn cần D7/D6 trước).
- seq8 ≤ **900 ms**, seq256 ≤ **1850 ms**.
- **Loss EXACT 3 seq** giữ nguyên; gate 9/10 (gap_tests pre-existing); trainer 15/15;
  selfhost `IL_IDENTICAL`.
- RAM seq1024 không tăng (có thể giảm bằng concat `saves`).

Không chấp nhận:
- Option non-deterministic thread completion order.
- Native kernel mới chưa qua probe diff=0 (trừ phi là bản tương đương bit-exact).
- PR push không qua gate.

## 2. Phạm vi — cái gì là D5, cái gì KHÔNG phải D5

**D5 (theo docs gốc, phiên toy):** "vong compile _step124 core thành 1 native unit,
không dispatch IL mỗi op; toàn bộ step." => ATA cụ thể hơn, khả thi:

**Định nghĩa D5 khả thi** = thay `TKVApp._step124` từ một call graph IL-recursive
bao quanh mọi op native thành **1 native unit với pipelined dispatch**:

1. Native pipeline điều phối tất cả core ops GPT2 (embed → 12 blocks → ln_f →
   lmhead_bwd → AdamW) trong 1 lượt, không quay qua IL mỗi op.
2. Stack của param/grad được pre-allocated 1 lần/step thay vì `t_zeros` &
   `_gi_zeros` lặp đi lặp lại.
3. `_step124` vẫn giữ nguyên signature (loss float `i32` status) — `_f124_*` vẫn
   gọi được từ .tkv.
4. **Bit-exact bởi thiết kế**: pipeline gọi cùng các `TkvSimd::*Native` wrappers
   với cùng argument order & cùng buffer — không thay 1 kernel nào.

**KHÔNG** là D5 (cần project riêng, không ở đây):
- X64 emission tay từ IL (LLVM backend).
- Đổi model training
- Đổi numerics (erf→tanh, FMA reorder) — vào D6/D4, không D5.
- Đổi layout/struct của `Tensor` (tốt nhất giữ nguyên).

## 3. Rủi ro chính & mitigation

| # | Rủi ro | Khả năng | Mitigation |
|---|---|---|---|
| R1 | Full step native unit trở thành ~2000 dòng C mão manually, không feasible | Cao | Sinh tự động từ IL đã có; giữ pattern `existing *_wrapper`; **drives từ .tkv template, không gõ tay** |
| R2 | Bit drift sau wiring | Trung bình | EXACT 3 seq phải pass trước khi merge; nếu lệcfy nhỏ, revert |
| R3 | Managed escape (user vẫn gọi .tkv thêm function trong training) | Thấp | Wrapper giữ public `TKVApp::_step124` signature; g1/g2 IL_IDENTICAL verify |
| R4 | Debug khó hơn (crash native mới không có stack) | Trung bình | Assert theo`ENV VAR: TKV_NATIVE_ASSERT=1` để rollback nhanh; parity-test dần |
| R5 | gflat offset layout swapped | Thấp | `use_flat` + `o_*` luôn use-site wiring, tránh hardcode |
| R6 | Token code drift (vd. `'n'` args type) | Thấp | chép nguyên từ `compile_*` exports hiện có |
| R7 | Profile không chứng minh win → dự án phí | Trung bình | Milestone-gate mọi bước; không commit nếu gain < ~100 ms/bước |

## 4. Kiến trúc đề xuất

### 4A. Bắt buộc phải hiểu trước khi bắt tay

GPT2 step hierarchy hiện tại:
```
TKVApp.main
  └ _step124 (TKVApp.il)
      ├ _f124_fwd2
      │    ├ gpt2_forward_saved (12 blocks + lmhead bwd + loss)
      │    │    ├ gpt2_embed + simd_add*
      │    │    ├ 12 * (gpt2_attention / gpt2_layernorm / mlp)  — via native wrappers
      │    │    ├ gpt2_loss → gpt2_loss PARALLEL worker / CE readout
      │    │    └ gpt2_backward (12 blocks) — native fused bwd chain
      │    └ AdamW opt
      └ copy_back (n_param → m6_* pinned buffers)
```

Hiện này các op gọi qua **IL interpreter** → .NET IL → JIT → native. Mỗi op đi
qua marshal wrapper (`Pin x, Call Native, Unpin x`). D5 gấp các op đó vào 1
**native block** có chuỗi pre-compiled.

### 4B. Phiên bản khả thi nhất — "Native trampoline" precompiled

Không viết codegen. Thay vào đó:

1. Tạo `tkv_step_124.cpp` chứa `_step124_native(...)` với cùng chữ ký/ semantics.
2. Nó gọi lần lượt (theo đúng thứ tự) toàn bộ native workers hiện có (`tkv_simd_native.dll` exports + `tkv_torch_fused.dll` exports), giữ đúng thứ tự/argument order.
3. Gọi như hàm `void* _step124_native_addr()` được `tkv_simd_native.dll` export, hoặc một DLL riêng `tkv_step124.dll`.
4. `.tkv` wiring: `_step124` → `#if USE_NATIVE_STEP` (compile-time `native_int` flag từ registry/env) → `torch_step124(...)` else `ˊphiên cũ`.
5. IL `compile_torch_step124` mới + `register_expr_builtin(reg, 'torch_step124', compile_torch_step124, 'i64')`.
6. Copyback `m6_*` pinned buffers — reference qua `Marshal.AllocHGlobal` tại native trampoline thay vì mỗi step alloc mới.

**Lợi ích:** giữ nguyên bit-exact vì call order không đổi;
thoát IL dispatch có thể show ~10-20% thời gian, không 40%.

**Phải thực sự đo** được IL dispatch cost trước khi cam kết D5; nếu < ~500 ms,
D5 chỉ marginally hợp lý và nên giữ tôn trọn architecture hiện tại.

### 4C. Phiên bản đầy đủ (nếu 4B chứng minh win)

Bước native đầy đủ: tất cả inner op loops (e.g. `_ls_core_bwd_bcast` …) chuyển
sang C++ + OpenMP/oneTBB để tận dụng multi-thread hơn; chỉ còn native kernels
là những lệnh scalar GEN1.

**Lưu ý:** 4C là project riêng; document này chốt 4B trước.

## 5. Milestones & gates

### D5-0: Pre-requisites (nothing to commit yet)
- [ ] **Profile phụ trợ**: chạy `dotnet-trace` trên bench §0.108 seq1024,
  export `speedscope.json`; xác nhận thực sự overhead IL/managed là ≥ 500 ms/step.
- [ ] **Inventory** tất cả `tkv_*. Native` exports + IL wrappers hiện có
  (`stack=grep`, danh sách: ~40 exports).
- [ ] **Vị trí inner hot op**: `gpt2_loss WARSX`, `adamw_step`, `bias_grad`,
  `lmhead_bwd_fused_list`, `layernorm_bwd_fused`, `attention_bwd_native` … xác định
  cái nào đã native 100%, cái nào wrap IL loop per-element.

### D5-1: wire native trampoline empty (calls nothing)
- [ ] Add `'torch_step124'` IL wrapper + `.tkv` entry (NO-OP → returns 0).
- [ ] Verify: build + gate 9/10 + trainer 15/15 + loss EXACT + `vong IL_IDENTICAL`.
- [ ] Commit #1 (~1 commit).

### D5-2: `tkv_step124.cpp` skeleton trong `simd_native/`
  Build luôn từ `tools/simd_native/build.ps1`. Bước này gọi chỉ 1 op native
  (e.g. `layernorm_fwd`) làm sanity. Không safety hơn — tránh chỉ cần thêm
  fields/vintages bên native.
- [ ] Compile clean AVX2 + link `tkv_simd_native` → copy `runtime/` + `out/`.
- [ ] Gate + trainer + loss EXACT → commit #2.

### D5-3: pipeline segment I — embed + first (1-2) blocks
- [ ] `_step124_native` gọi `op_embed_fwd`-equivalent: `op_embed_fwd` đã ở dạng
  native trong `.il` (via `t_zeros`+list). Wire fast path.
- [ ] 1 block attention/mlp: so sánh loss EXACT hiện trạng vs native run.
- [ ] Đảm bảo bit-exact trước khi mở rộng lên 12.
- [ ] Gate + trainer + loss EXACT → commit #3 (chỉ sau khi ổn ở 1 block).

### D5-4: Pipeline segment II — toàn bộ fwd cho seq=8
- [ ] Hoàn tất pipeline cho `N_Layers=12` phía fwd. seq=8 .tkv kiểm tra tồn tại mode EXACT.
- [ ] Gate + trainer + loss EXACT 3 seq (ít nhất 8/256/1024) + mode1 `13.442915`.
- [ ] Commit #4.

### D5-5: Pipeline segment III — backward pass
- [ ] `gpt2_backward` convert native call-by-call.
- [ ] loss EXACT + trainer 15/15 + gate.
- [ ] Commit #5.

### D5-6: Pipeline segment IV — AdamW + copyback
- [ ] `_adamw_step_native` + `m6_* copy` native; xóa `_f124_fwd2`/`backward` helper completely.
- [ ] Gate + trainer + loss EXACT 3 seq + mode1 EXACT.
- [ ] Commit #6.

### D5-7: Performance closure
- [ ] A/B seq=1024 best-of-3 (mode0 & mode1), goal ≤ 5000 ms full step.
- [ ] Update HANDOFF + revert plan note nếu không đạt win threshold.

## 6. Những thứ KHÔNG làm

- Không viết X64 emission từ IL (khiến project bloat, không bit-exact guarante).
- Không thay đổi `saves` ABI (6N+3 -> 5N+3 trong §0.108).
- Không dùng threadpool `Task.Run` ad-hoc cho non-hot path.
- Không tự ý chép `.cpp` từ ATen project (license risk).

## 7. Đo lường — tiến hành hỗ trợ (pre/post)

**Trước khi bắt đầu (cần gấp):**
1. `dotnet-trace collect` seq1024 mode0, lấy breakdown managed/unmanaged.
2. `bench_attn.c`/`bench_attn2.c`(untracked) đo attention bwd.
3. `mb1024_g` cho MLP bwd baseline.
4. `out/gate_f32.sh` ghi baseline đầy đủ.

**Gate tiêu chí (sau commit D5-N nếu N<5):**
- 3 seq loss EXACT mode0
- mode1 `13.442915` EXACT
- `g_trainer_tests_n10.dll` 15/15
- gate 9/10 PASS (chỉ `gap_tests` exit=82 pre-existing)

## 8. Ước tính

| Milestone | Thời lượng ước tính |
|---|---|
| D5-0 | 2 giờ |
| D5-1 | 2 giờ |
| D5-2 | 3 giờ |
| D5-3 | 6–10 giờ |
| D5-4 | 8–16 giờ |
| D5-5 | 8–16 giờ |
| D5-6 | 4–6 giờ |
| D5-7 | 2–3 giờ |
| **Total** | **33–54 giờ** (≈ 4–7 ngày làm việc) |

Đây chưa tính nếu D5-3/D5-5 phát hiện win không rõ (thì rollback về hiện tại);
các section `IF ALL ELSE FAILS — rollback` sau.

## 9. Rollback

- Mỗi commit D5 độc lập; nếu D5-N regression bắt buộc: revert commit đó,
  HANDOFF ghi nhận thất bại và dừng toàn bộ D5.
- Nếu `_step124` luôn phải chạy fallback (native call error), test-suite phải
  xanh với fallback đó (§0.108 pattern: `rc != 0 → fallback nguyên văn`).
- TKVApp::_step124 không bao giờ mất API — `TKVApp.il` giữ cũ cho đến khi D5-6
  hoàn tất, sau đó có thể strip def-fallback sau khi đã ổn định 1 tuần.

## 10. Chủ sở hữu

- User (duyệt mục tiêu, rollback decision)
- Agent (thiết kế chi tiết + triển khai, commit từng milestone)
- docs (cập nhật HANDOFF §0.109..§0.115)
- Benchmark logs untracked (`out/dd_*`,`bench_*`)
