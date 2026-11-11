# Cách Benchmark TokenVector.Inference vs PyTorch

Công thức đo apples-to-apples cho bước huấn luyện GPT-2 124M (benchmark mà ta
công bố). Nhật ký số liệu: `BENCHMARKS.md` (mới nhất: §9.22). Thư mục kết quả:
`benchmark/` (gitignored, trừ 2 file của mỗi phiến đo). Xem bản tiếng Anh: `BENCHMARKING.md`.

## Cần chuẩn bị

- Compiler self-host: `D:\TokenVector\Release\dist\tkvc-tv.exe` (ở dưới gọi là `$tkvc`).
- .NET 10 trên `PATH` (binary biên dịch cần `dotnet <exe>` và `.runtimeconfig.json`
  đặt cạnh — `tkvc-tv` **không** tự sinh file này).
- Python + `torch 2.14+cpu` + `transformers` (phía PyTorch).
- ~8 GB RAM trống; máy tham chiếu: Xeon E5-2695 v3 (14C/28T).

> **Đo 2026-11-11 (cùng phiên, §9.22):** full step @seq1024 = **4016 ms**
> (TKV) so với **4061.1 ms** (torch) → **0.989×**, loss EXACT.

## Thiết lập một lần

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
Set-Content tv/tests/_seq_cfg.txt "1024`n1`n1`n1"   # seq / mode / steps / warmup
```

Các trường của `_seq_cfg.txt`: `seq` (8/256/1024), `mode` (`0` = fwd+bwd,
`1` = full step, `2` = phân rã 5 pha), `steps`, `warmup`.

## Build binary bench phía TKV

```powershell
& $tkvc tv/tests/simd_bench_124m_seq.tkv benchmark/bseq.exe main
# thiếu runtimeconfig → lỗi "hostpolicy.dll not found":
Set-Content benchmark/bseq.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
```

## Đo TKV

```powershell
# một cấu hình — best-of-3 (chạy 3 lần, lấy min):
dotnet benchmark/bseq.exe
# ví dụ output:
#   BSEQ full-step simd seq=1024 ms=4016.0 loss=10.760399 -> 13.442921
# mode 0 in:  BSEQ fwd=... bwd=... total=...
# mode 2 in:  BPHASE seq=1024 fwd=1016.0 loss=109.0 bwd=2328.0 opt=453.0 copyback=141.0 TONG=4047.0 loss_val=...
```

Script matrix (TKV best-of-3 + torch cả hai tier, ghi log vào `benchmark/`):

```powershell
powershell -File benchmark/_bench_matrix.ps1
```

## Đo PyTorch (luôn đo lại cùng phiên)

```powershell
python benchmark/bench_torch.py d 1024   # tier d: fwd+bwd và full step
python benchmark/bench_torch.py e 1024   # tier e: phân rã zero/fwd/bwd/opt, T=1 và T=14
```

**Không bao giờ trích dẫn số torch khác phiên.** §9.21 từng mượn `4130.6`
của §9.18; số cùng phiên thực ra là `4061.1`. Mỗi bảng công bố phải ghi rõ
cùng phiên hay khác phiên.

## Đọc kết quả

| TKV in ra | Nghĩa là | So với |
|---|---|---|
| `BSEQ fwd=… bwd=… total=…` (mode 0) | fwd+bwd, không optimizer | torch tier d `fb()` |
| `BSEQ full-step simd` (mode 1) | một bước huấn luyện đầy đủ | torch tier d `step()` |
| `BPHASE … TONG=…` (mode 2) | 5 pha, min-of-2 | torch tier e |
| `BENCH E torch … seqN T=14` | pha phía torch | mode 2 |

Loss phải in **EXACT**: `10.572291` / `10.756927` / `10.760399`
(seq8/256/1024). Nếu khác — dừng lại, có gì đó sai.

## Thêm một lần đo vào nhật ký

1. Chạy matrix ở trên; lưu log thô vào `benchmark/_bench_matrix_<tag>_<yyyyMMdd_HHmmss>.log`.
2. Thêm mục `§9.x` vào `BENCHMARKS.md` với: CPU, đường dẫn + sha256 compiler,
   torch version + threads, ghi chú cùng/khác phiên, bảng TKV vs torch
   (mode 1 best-of-3, mode 2 TONG, tier `d`/`e`), dòng loss EXACT, và ghi chú
   phương sai (`mode 1` dao động — số tin cậy là `mode 2`).
3. Ghi vào `HANDOFF.md` (append-only).

## Những cạm bẫy

- `dotnet benchmark/bseq.exe` báo `hostpolicy.dll not found` → thiếu
  `*.runtimeconfig.json` cạnh exe (xem §9.17).
- `OMP: Warning #182` vô hại (OpenMP affinity).
- RAM: full step @124M giữ ~7.2 GB; đóng torch trước/sau để tránh nhiễu page file;
  không chạy torch và TKV đồng thời.
- `tv/tests/_seq_cfg.txt` bị các script ghi đè — nhớ trả về sau khi thí nghiệm.
- `mode 1` dao động ±5% giữa các lần chạy; quyết định theo pha dùng `mode 2`.
- Không bench với compiler thiếu native AVX2/FMA exports (`tkv_simd_native.dll`)
  — đường fallback scalar chậm ~2×.
