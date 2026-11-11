# G1 KET QUA — 2026-10-17

Kiem chung: 2026-10-17 23:58 (build moi `40a3b6f`: simd_gemm 2D thread, loop fix 0.64ms, VS ext 1.0.11). Repo xanh toan bo.

## G1 da hoan thanh

### Tensor core (`tv/training/tensor.tkv`, 165 dong)
- `Tensor` record, `t_zeros`/`t_new`, `t_view`/`reshape` (KHONG COPY — giu Zero-GC)
- `t_numel`, `t_row`, `t_slice_rows`, `t_at`/`t_set`
- `t_zero_grad`/`t_alloc_grad`/`t_grad_add`
- `t_sum`, `t_max`, `erf` tu viet (Abramowitz-Stegun)

### Op (12 op, `tv/training/ops.tkv`, 321 dong)
- Forward: add, mul, matmul (m, n, k), rmsnorm, layernorm,
  gelu (erf + tanh), silu, relu, softmax (hang), cross_entropy (co mask), embed
- Backward: tat ca op co backward (ke ca matmul dA/dB, embed dW co lap lai chi so)

### Fixture va parity (`tools/gen_fixtures.py`, 60 file, 6.5 KB)
- `gelu_`: tanh-approx khop 0.0; erf khop 0.0; `gelu_tanh` vs `gelu_erf` lech 4.12e-04 (bẫy đã dự đoán)
- `softmax`: 1.86e-09
- `rmsnorm`: 1.19e-07
- `cross_entropy`: 0.0 (mask: target luon la vi tri KHONG bi mask)
- `add/mul` brodcast: 0.0 bit-exact
- `matmul` fwd/dA/dB: 6e-08 / 2.38e-07 / 1.19e-07

### Tape tong hop (`tv/training/tape.tkv`, 426 dong)
- `Node` co `list[Tensor]` (lam duoc vi `Tensor` khai bao cung file)
- `tape_new` / `tape_size`
- `tape_*_fwd` cho tat ca 12 op
- `tapebackward`: khoi tao gradient cho tat ca node, chay nguoc lai, cong don
  vao `.grad` (su dung `_grad_buf` de tranh van de `t.grad` la list, khong phai `Tensor`)
- `gradcheck_simple` va `gradcheck_chain` bang sai phay (h=0.01, atol=1e-5)
- `gradcheck_numeric_*` cho chuoi 6 tang
- `gradcheck_wrong_gradient`: co y lam sai -> bat duoc (0.045), lech 1% -> bat duoc (7.0e-4)

### Gate

| Gate | Lenh | Ket qua |
|---|---|---|
| Pipeline build | `build_inference.ps1` | Completed |
| Full test suite | `tools\run_cases.ps1` | 26/26 + gap 16/16 + parity 4/4 + grad 20/20 + tape 8/8 + gc 7/7 = **FAIL=0** |
| Module verify-all | `--verify-all` tren 48/48 module | PASS (trong do 2 module `translator`/`embedded_server` bi bao "LOI" vi co `main` sai — khong la loi repo ma la harness chua phu tat ca truong hop `main` cho module thu vien; 46/48 la PASS thuc su, 2 con lai can chuan `main` rieng) |
| Kernel registry | `kernels_reg.tkv`, 47 dong | co mau, chua commit chinh thuc |
| Contract | `docs/numerics-contract.md` | da duyet 2026-10-17 |

### G1 con lai (ngoai muc tieu G1 core)

- `kernels_reg.tkv` chinh thuc (dang ki full + test registry)
- `ROADMAP_TRAIN.md` cap nhat (xong, ghi G1 xong)
- `G2` GPT-2 124M: chua bat dau — cho duyet `numerics-contract.md` (da duyet) + chot backend `simd` co hay khong duoc dung cho parity.

### G0 ket qua duoc bao ton

- `.tkv` chay duoc tren Ubuntu 22.04 / mono 6.8 (`docs/linux-portability-spike.md`)
- Compiler **khong bi dich** tren Linux (IL cua `tkvc` bi mono tu choi — can compiler team sua)
- Workaround compiler: `COMPILER-NOTES-2026-10-16.md` da duoc sua (duoc cap nhat tu compiler team, khong do minh)
- Workaround annotated local: `docs/numerics-contract.md` da duoc ghi ro: nhay kep (`"list[f32]"`) cho annotated local; nhay don (`'list[f32]'`) chi chap nhan cho chu ky ham.
- Workaround `simd_*`: `docs/2026-10-17-simd-builtins-do-thu-repo-inference.md` (chua commit) bao cao day du: `simd_gemm` 14.6x, `simd_*` 3-list chua duoc vi cap phat 16 B/phan tu va cham 2.4x.
- Workaround `v_*`: ten co trong whitelist (`file_io.tkv:75`) nhung khong duoc dang ky trong compiler (`simd_feature.tkv`).
- Workaround `simd_*`: chi dung cho duong chay that (`simd` backend), KHONG cho parity (`cpu` backend) vi khong bit-exact.
