# BENCHMARKS — Bản tóm tắt tiếng Việt (cập nhật 2026-11-11)

Tài liệu này là bản tóm tắt tiếng Việt của `BENCHMARKS.md` (bản tiếng Anh —
nhật ký đầy đủ từ §4 đến §9.22). Bản VI cũ (2026-09-28, coi mọi số liệu là
stub vô nghĩa, "không có SIMD", đường dẫn compiler chết) đã lỗi thời và được
thay bằng trang này.

## Trạng thái hiện tại (2026-11-11, cùng phiên)

Full train step GPT-2 124M @seq1024: **4016 ms** (TKV) so với **4061.1 ms**
(PyTorch 2.14.0+cpu, torch threads=14) → **0.989×**. Loss EXACT
(`10.572291 / 10.756927 / 10.760399`). Gate f32 **135/135**, trainer **15/15**,
gpt2_full_tests **33/33**.

| phép đo | §9.22 (mới nhất) | torch cùng phiên |
|---|---|---|
| full step seq8 | 1047 ms | 981.4 ms (TKV 1.067×) |
| full step seq256 | 1594 ms | 1551.4 ms (TKV 1.027×) |
| full step seq1024 | **4016 ms** | 4061.1 ms (TKV **0.989×**) |
| fwd+bwd seq1024 | 3375.0 ms | 3395.3 ms |
| opt.step seq1024 (pha) | **453 ms** | 646.4 ms |
| mode-2 TỔNG seq1024 | **4047 ms** | 4097.6 ms (TOTAL e T=14) |

## Cách đo (recipe hiện hành)

Xem `BENCHMARKS.md` §1. Tóm tắt:

- TKV: `& $tkvc tv/tests/simd_bench_124m_seq.tkv out/bseq.exe main`, cấu hình
  `tv/tests/_seq_cfg.txt` = `<seq>/<mode>/<steps>/<warmup>` (`mode=0` fwd+bwd,
  `mode=1` full step best-of-3, `mode=2` phân rã 5 pha min-of-2), chạy bằng
  `dotnet out/bseq.exe` (cần `.runtimeconfig.json` cạnh exe — xem §9.17 của
  bản tiếng Anh).
- torch: `python tools/bench_torch.py d <seq>` (fwd+bwd & full step) và
  `python tools/bench_torch.py e <seq>` (phân rã pha, T=1 và T=14).
- **Luôn đo lại torch trong cùng phiên** khi công bố số mới; đánh dấu rõ
  so với §9.21 (khac phiên).

## Quy ước trình bày

- Mỗi bảng phải ghi rõ: máy, compiler (đường dẫn + ngày build), torch version,
  TKV/torch ratio, cùng phiên hay khác phiên.
- Mode 1 dao động mạnh; số tin cậy cho quyết định theo pha là mode 2.
- Suite `compare_rivals.tkv` là smoke legacy — hằng số ORT_*/TORCH_* trong đó
  là literal, không phải phép đo; không trích dẫn.
- `HANDOFF.md` là nhật ký kỹ thuật; mọi kết luận số học (numerics contract)
  xem `docs/numerics-contract.md`.
