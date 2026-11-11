# 🚀 TokenVector.Inference: Runtime Deep Learning & Engine huấn luyện viết thuần `.tkv`

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: TokenVector .tkv](https://img.shields.io/badge/Language-TokenVector%20.tkv-blue.svg)](https://github.com/nguyenhungtran18/TokenVector)
[![Compiler: tkvc](https://img.shields.io/badge/Compiler-tkvc--tv-orange.svg)](https://github.com/nguyenhungtran18/TokenVector)

<img src="assets/tokenvector-logo.svg" alt="TokenVector logo" width="96" height="96">

**TokenVector.Inference** là runtime deep learning viết bằng **TokenVector
(`.tkv`)** — inference (tối ưu graph, quantization, micro-serving) **và**
engine huấn luyện GPT-2 124M có thời gian một bước ngang PyTorch (đo
2026-11-11, xem bên dưới). Xem bản tiếng Anh: `README.md`.

> **Trạng thái hiện tại, kiểm chứng 2026-11-11:**
> - Full train step GPT-2 124M @seq1024: **4016 ms** so với PyTorch
>   2.14.0+cpu (cùng phiên, 14 luồng) **4061.1 ms** → **0.989×**, loss
>   **EXACT** (`10.760399`), `gate_f32.sh` **135/135**, `trainer_tests`
>   **15/15**, `gpt2_full_tests` **33/33**. Chi tiết: `BENCHMARKS.md` §9.21–§9.22.
> - Nguồn là `.tkv`; pipeline tối ưu sinh IL P/Invoke thư viện native SIMD
>   (`tkv_simd_native.dll`, build bởi repo compiler) cho các kernel nóng.
>   Đường CPU là đường được hỗ trợ.
> - Còn lại: seq8/seq256 vẫn chậm hơn torch một chút (1.067×/1.027×);
>   working set ~7.2 GB @seq1024; mode-1 đo dao động — dùng mode-2 (phân rã
>   5 pha) khi ra quyết định.
>
> `HANDOFF.md` là nhật ký kỹ thuật (append-only). Mọi lỗ hổng đã biết —
> và mọi lần đo sai trên đường đi — đều được ghi ở đó.

## Cái gì đã có

| Hạng mục | Trạng thái |
|---|---|
| Inference runtime (graph, PTQ INT8/FP8, ONNX reader, batcher) | Chạy được, có test; xem `TEST_REPORT_VI.md` |
| Huấn luyện GPT-2 124M (fwd/bwd/AdamW/clip/checkpoint) | Chạy được; xem `BENCHMARKS.md` §9 |
| Kernel native SIMD (GEMM, fused bwd, AdamW) | Ship qua repo compiler (`tkv_simd_native.dll`) |
| GPU / NPU | **Không có** — chỉ CPU |

## Yêu cầu

- **Compiler TokenVector**, bản self-host hiện hành (ở đây gọi là `$tkvc`).
  Bản tham chiếu: `D:\TokenVector\Release\dist\tkvc-tv.exe`.
- **.NET 10 runtime** để chạy exe đã biên dịch. `tkvc-tv` **không** sinh
  `.runtimeconfig.json` cạnh output — thiếu nó sẽ báo
  `hostpolicy.dll not found`. Tạo một file:

  ```json
  {"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}
  ```

  (chi tiết: `BENCHMARKS.md` §9.17).

## Bố cục repo

```
tv/            Nguồn .tkv (runtime/, training/, tests/, tools/)
tools/         Công cụ phía host: bench_torch.py, test runners, generators
docs/          Tài liệu kỹ thuật;  docs/archive/ = kế hoạch đã xong/hết hiệu lực
reports/       Báo cáo parity & benchmark theo ngày
BENCHMARKS.md  Nhật ký benchmark chính (hiện tại: §9.22)
BENCHMARKING_VI.md  Hướng dẫn benchmark vs PyTorch (+ EN)
BENCHMARKING_VI.md  Hướng dẫn benchmark (+ EN)
HANDOFF.md     Nhật ký kỹ thuật (append-only)
```

Fixture parity GPT-2 124M (~2.3 GB) **không nằm trong git**; sinh lại bằng
`python tools/gen_gpt2_fixture.py` (xem `docs/handoff-gpt2-124m-fixture-design.md`).

## Bắt đầu nhanh

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"

# Inference quickstart (graph -> session -> một forward pass)
& $tkvc tv/examples/quickstart.tkv benchmark/quickstart.exe main
Set-Content benchmark/quickstart.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/quickstart.exe      # in ra: Y[0] = 2.25  /  quickstart: OK
```

API theo dạng graph; không có `tv.inference.load_onnx()` cũng như `NDArray`.
Xem `USER_GUIDE_VI.md` cho hướng dẫn API và `TOKENVECTOR_SYNTAX_SPEC_VI.md`
cho các quy tắc ngôn ngữ hay cắn (hằng số cấp module không import được;
`__init__` của record bị bỏ qua; `TkvInt` …).

## Build & test

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"

# Hai suite inference (mỗi case chạy trong một process riêng)
powershell -File tools\run_cases.ps1 -Compiler $tkvc

# Cổng f32, 10 suite, 135 assert
bash tools/gate_f32.sh

# Cổng trainer (MLP, AdamW vs reference PyTorch): kỳ vọng 15/15
& $tkvc tv/tests/trainer_tests.tkv out\trainer_tests.exe main
dotnet out\trainer_tests.exe
```

## Benchmark vs PyTorch

```powershell
# Phía PyTorch (cần python + torch 2.14+cpu + transformers)
python benchmark/bench_torch.py d 1024   # fwd+bwd và full step
python benchmark/bench_torch.py e 1024   # phân rã pha, T=1 / T=14

# Phía TokenVector (124M, _seq_cfg.txt chọn seq/mode/steps/warmup)
Set-Content tv/tests/_seq_cfg.txt "1024`n1`n1`n1"
& $tkvc tv/tests/simd_bench_124m_seq.tkv benchmark/bseq.exe main
dotnet benchmark/bseq.exe
```

Kết quả cùng phiên mới nhất (2026-11-11, `BENCHMARKS.md` §9.22):

| phép đo | TKV | PyTorch | TKV/torch |
|---|---:|---:|---:|
| full step @seq1024 | **4016** | 4061.1 | **0.989×** |
| full step @seq256 | 1594 | 1551.4 | 1.027× |
| full step @seq8 | 1047 | 981.4 | 1.067× |
| fwd+bwd @seq1024 | 3375.0 | 3395.3 | **0.994×** |
| opt.step @seq1024 (pha) | **453** | 646.4 | **−193 ms** |

## Bản đồ tài liệu

- `BENCHMARKS.md` / `BENCHMARKS_VI.md` — nhật ký benchmark
- `HANDOFF.md` — nhật ký kỹ thuật, mọi điều tra và ngõ cụt
- `USER_GUIDE_VI.md` — hướng dẫn API
- `TOKENVECTOR_SYNTAX_SPEC_VI.md` — quy tắc cú pháp ngôn ngữ
- `docs/numerics-contract.md` — hợp đồng số học fp32 (cái gì bit-exact với PyTorch, cái gì không)
- `docs/COMPILER_TEAM_REQUEST*.md` — các request gửi compiler team (#1–#14)
- `docs/archive/` — kế hoạch đã xong/hết hiệu lực (đừng thực thi)
- `RELEASE_NOTES_VI.md` — lịch sử release
- `ROADMAP_TRAIN.md` — lộ trình huấn luyện (chiến dịch đã hoàn thành)

## Hệ sinh thái

**TokenVector** là hệ sinh thái AI/HPC thế hệ mới xây trên ngôn ngữ `.tkv`
và compiler `tkvc`:

- [`TokenVector`](https://github.com/nguyenhungtran18/TokenVector) — ngôn ngữ & compiler
- [`TokenVector.Numerics`](https://github.com/nguyenhungtran18/TokenVector.Numerics) — NDArray / đại số tuyến tính
- [`TokenVector.Vision`](https://github.com/nguyenhungtran18/TokenVector.Vision) — vision 2D & 3D
- [`TokenVector.Text`](https://github.com/nguyenhungtran18/TokenVector.Text) — tokenization, embeddings, tiền xử lý LLM
- [`TokenVector.Data`](https://github.com/nguyenhungtran18/TokenVector.Data) — pipeline dữ liệu double-buffer
- **TokenVector.Inference** — repo này: runtime inference + huấn luyện GPT-2 124M
