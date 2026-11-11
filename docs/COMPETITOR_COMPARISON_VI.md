# Feature Comparison: TokenVector.Inference vs ONNX Runtime vs llama.cpp vs PyTorch

> Phạm vi: so sánh **chức năng** (làm được gì), không phải tốc độ. Toàn bộ nguồn trong repo là
> TokenVector (`.tkv`, 640+ file kể cả bản mirror sinh tự động, ~205 nghìn dòng); binary đã biên dịch P/Invoke
> thư viện native SIMD đồng hành `tkv_simd_native.dll` (build bởi repo compiler `D:\TokenVector`) cho các
> kernel nóng — không có mã C/C++ trong repository này. Đường scalar `.tkv` thuần vẫn là tham chiếu ngữ nghĩa.
> Host intrinsics chỉ còn cho cấp phát thô, OS/HTTP/JSON/time-threads, file/network IO, và
> byte codec (protobuf/utf8/FP8/FP16-bitcast).
> Chỉ đo tốc độ theo recipe ở `benchmark/BENCHMARKS.md` §1 (124M fwd/bwd/step; torch cùng phiên) —
> `tv/benchmarks/compare_rivals.tkv` là smoke suite legacy, không phải nguồn đo.

Chú thích: (✅) có · (giới hạn) một phần · (thiếu) chưa có

## 1. Phủ model

| Khả năng | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Định dạng đầu vào | ONNX (reader zero-copy, topo-sort Kahn, **writer/serializer**) + SafeTensors (F32/F16/BF16) + **GGUF** (`format/gguf.tkv`: container, metadata KV, F32/F16/BF16/int, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K/Q4_K verified, Q5_K provisional, **gated Q2_K/Q3_K/Q6_K** dequant + self-gate) | ONNX đầy đủ + shape inference | GGUF (+ tự tải từ HF) | Native / SafeTensors / export ONNX |
| Model lớn | ONNX single-file; không external-data / checkpoint chia shard | External data, >2 GB, symbolic shapes | GGUF chia shard, mmap, `--mlock` | SafeTensors chia shard, meta-device init |
| Phủ op | Danh sách đóng 82 op (`execution_node.tkv` + `translator.map_op`): Gemm đầy đủ, Gather/Slice/Split/Squeeze/Unsqueeze/Expand, Clip/Cast/Pad/Where, ReduceMean/Sum/Max, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, LogSoftmax, Trilu, TopK, Constant (fold), Quantize/DequantizeLinear, QLinearMatMul/Conv, MatMulInteger, DynamicQuantizeLinear, GatherND, ScatterElements, Range, EyeLike, ConstantOfShape, CumSum, InstanceNormalization, Mish, PRelu, Resize/Upsample, Einsum tổng quát, If/Loop (unroll), LSTM/GRU/RNN (unroll 2 chiều, Y/Y_h/Y_c), `StringConcat`/`StringEqual`/`StringLength` (bảng `str_values`), `Scan` unroll tĩnh (trip ≤ 128). **Op lạ báo lỗi rõ, không im lặng** + suy luận shape tĩnh. Còn thiếu: Sequence*/image ops (cần kiểu sequence) và Scan trip runtime | **200+ ops**, full opset | Native LLM-arch | Full ATen + custom op, autograd |
| Shape động | ✅ symbolic dimension, shape specialization, và chạy shape cụ thể theo request (`runtime/dynamic_shapes.tkv`) | symbolic dims | context biến đổi | dynamic + `torch.compile` |
| Training | ✅ Huấn luyện GPT-2 124M đầy đủ (`training/gpt2_ref124.tkv`: fwd/bwd/AdamW/clip/checkpoint, loss EXACT so với fixture HF; full step @seq1024 = **4016 ms** vs torch 4061.1, cùng phiên, 0.989× — `benchmark/BENCHMARKS.md` §9.22) + MLP+AdamW reference (`trainer_tests` 15/15) + LoRA | ✅ on-device training | ✗ chỉ inference | ✅ training đầy đủ |

## 2. Tính năng LLM

| Khả năng | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Attention | Softmax O(1) bộ nhớ theo hàng, causal + mask cộng + `kv_offset`, GQA/MQA | FlashAttention qua EP, GQA, ctx 128K | Flash attention, SWA cache, prompt caching | SDPA + FlashAttention-2/3 (GPU) |
| KV-cache | paged blocks (`paged_kv.tkv`: LRU, `PrefixStore` chia sẻ) + **prefix compute-skip** (`prefix_skip.tkv`: intern prefix nguyên block, decode suffix) | paged (GenAI / PagedAttention + quantized KV 1.29) | paged + tái dùng prefix | paged (vLLM) |
| Continuous batching | scheduler slots (waiting tới running, decode từng bước) + pack single-forward (block mask) | qua EP/Triton | mức token + parallel slots | qua vLLM/TRT-LLM |
| Sampling | temperature/top-k/top-p/min-p/penalty/greedy (LCG deterministic) | API `Generate()` | đầy đủ + grammar/JSON | đầy đủ |
| Tokenizer/template | BPE + WordPiece + SentencePiece-unigram; 5 built-in + Jinja-subset | qua EP | built-in + full Jinja | `transformers` |
| Pipeline text | `generate_text`/`chat_generate` nối vào server (usage, stop, seed) | GenAI đầy đủ | đầy đủ | đầy đủ |
| Speculative decoding | draft+target verify (`speculative.tkv`) + **tree multi-draft** (`tree_speculative.tkv`) + **beam** (`beam.tkv`) | tùy EP | draft / **EAGLE-3** / DFlash / DSpark / n-gram (simple/map-k/mod), mixable (`--spec-type`) | có |
| Tool-calling | loop + JSON-object mode + grammar masking GBNF-subset | GenAI | grammar, tool-call | hệ sinh thái |
| Đa modal | (giới hạn) preprocess vision (resize/normalize/patchify) + graph ViT builder; **thiếu weight encoder** | tùy EP | mtmd + mmproj (vision/audio/video) | có |

## 3. Lượng tử hóa

| Khả năng | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| INT8 weights | per-tensor sym/asym + **per-channel** + GEMM per-channel | static/dynamic, per-channel | Q8_0 | per-channel |
| INT8 activations | dynamic theo hàng | dynamic | mix K-quants | dynamic |
| INT4 | symmetric grouped + **asymmetric kiểu AWQ (zp/group)** + GEMM grouped (+ordered/act-order) + permute cols | AWQ + RTN; NVFP4 QMoE (1.29) | K-quants Q2 tới Q6 | torchao 0.18 (int4/int8/MXFP8 MoE, QAT) |
| FP8/FP16 | kiểu FP8 E4M3/E5M2 **+ GEMM compute E4M3** + **kiểu FP16 + GEMM compute** | FP16 + mixed; FLOAT8E8M0 (1.27+) | compute F16 | FP8/FP16/BF16; torchao float8 train |
| Calibration | MinMax, **KL kiểu TensorRT**, Percentile, lưới MSE | MinMax, Entropy, Percentile | imatrix K-quants | MinMax, Histogram, MSE |
| QDQ | fold cặp QDQ + pass riêng; chưa xuất **file** QDQ | model QDQ | n/a (GGUF native) | có |

## 4. Serving

| Khả năng | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| API OpenAI | `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (JSON + SSE), `/v1/rerank`, `/predict`, web UI chat (`GET /`) | (thiếu) cần Triton/OVMS | + `/v1/responses`, `/v1/rerank`, Anthropic-compat | qua sidecar |
| Streaming | SSE theo chunk text + `[DONE]` | n/a | SSE | có |
| Auth/giới hạn | API key, rate-limit/IP, max body, backpressure 429 | n/a | api-key, TLS | tùy frontend |
| Metrics | Prometheus + queue/dropped | profiler | `--metrics` | Prometheus |
| Batching | micro-batch 100-500us trên pool + slots liên tục | tuning EP | continuous + ubatch | dynamic |
| Khác | xem §2 (tool-calling, grammar, prefix-skip) | GenAI `Generate()` | draft/EAGLE-3/n-gram speculative | K8s autoscale |

## 5. Phần cứng, footprint và hệ sinh thái

| Khả năng | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| CPU | ✅ kernel native AVX2/FMA qua `tkv_simd_native.dll` đồng hành + fallback scalar `.tkv` (tham chiếu ngữ nghĩa); tái dùng slot theo liveness; **arena chưa zero-GC** | MLAS, mọi arch + mobile | mọi CPU, AMX/NEON/SVE | MKL/OpenBLAS |
| GPU | (thiếu) provider stub báo lỗi rõ | CUDA/TRT/DirectML/ROCm/Metal | CUDA/HIP/Metal/Vulkan, offload tách lớp | CUDA/ROCm/XPU |
| NPU/edge | (thiếu) | QNN, OpenVINO, CoreML | Hexagon (giới hạn) | ExecuTorch |
| Footprint | ✅ nguồn `.tkv` + 1 DLL native (đồng hành, repo compiler) | ~100 MB+ kèm EP | một binary | cỡ GB |
| Provider API | `device.tkv` (CPU live; accelerator stub) | EP plugin API | backend built-in | device/dispatch |
| Hệ sinh thái | tải HF-hub + cache, CLI (`serve/gen/export/calibrate/latency`), client HTTP qua API OpenAI-compatible | Olive, Model Zoo, Azure | HF GGUF, server distros | kho package + hub mô hình |
| Ngôn ngữ | chỉ `.tkv` | đa ngôn ngữ (C++ core + binding) | C/C++ API + 20+ binding | đa ngôn ngữ, binding dày đặc |

## 6. Tốc độ

Số latency/throughput chỉ ghi trong một lần chạy có ngày tháng của
Số đo tốc độ nằm ở `benchmark/BENCHMARKS.md` §9 (hiện tại: §9.22, cùng phiên với torch). Không sao chép số lịch sử vào tài liệu này.

## 7. Còn lại — lý do kỹ thuật chính xác từng mục

**Đã đóng trong đợt TRIZ này:** (1) LoRA + CheckpointArena, (2) HAL + kernel lane-abstract, (3) string-as-bytes + Scan driver/unroll tĩnh, (4) gated Q2_K/Q3_K/Q6_K + graph ViT, (5) tree-speculative + prefix-skip + beam. Op phủ thêm: `StringConcat`/`StringEqual`/`StringLength` (bảng `str_values`), `Scan` unroll tĩnh (trip ≤ 128); GGUF đọc Q2_K/Q3_K/Q6_K theo layout ggml + structural check + roundtrip self-gate.

**Bốn gap đã có regression runner** (`tv/tests/gap_tests.tkv`): SIMT matmul, dynamic-shape specialization, remap tên weight ViT SafeTensors, pretrain gradient clipping, và backend dispatch; build và chạy thực tế đều PASS.

- **Seam backend TokenVector** (`runtime/backend.tkv`): chọn provider và dispatch CPU/SIMT trên kernel thuần `.tkv`. `backend_status()` phân biệt `available`, `simulated` và `unavailable`. CUDA và OpenCL được báo **unavailable** — không có backend GPU trong runtime này.

- **Không có mã C/C++ trong repository này.** Chuyển đổi bit IEEE-754 binary32 và byte codec được viết bằng TokenVector (`runtime/f32_bits.tkv`). Các kernel nóng (GEMM, fused backward, AdamW) được P/Invoke từ IL biên dịch sang thư viện native SIMD đồng hành `tkv_simd_native.dll` (repo compiler, không nằm trong repo này); kernel scalar `.tkv` thuần vẫn là tham chiếu ngữ nghĩa và chạy được ở mọi nơi thiếu native export. Các file bridge CUDA/OpenCL/OpenVINO cũ đã bị xoá ngày 2026-09-26 (code chết không thể chạm tới).
1. **Pretraining quy mô LLM.** GPT-2 124M huấn luyện end-to-end (fwd/bwd/AdamW/clip, loss EXACT so với fixture HF, 0.989× torch @seq1024 — `benchmark/BENCHMARKS.md` §9.21–§9.22). Ở quy mô lớn hơn CPU pretraining vẫn chậm 10–100x; chỉ khả thi cùng mục 2.
2. **Compute GPU/NPU.** `.tkv` không có cấu trúc GPU (thread block, shared memory, MMA, launch kernel). Backend GPU = mở rộng **compiler** tv (repo chính) + shim CUDA/TensorRT/QNN. `device.tkv`/`hal.tkv` fail loud; `HalDispatch` là seam đăng ký khi có backend.
3. **Sequence container + image ops + Scan động.** String đã chạy trên `str_values` + UTF-8 (`string_tensor.tkv`); Scan tĩnh unroll khi trip ≤ 128. Còn thiếu: kiểu sequence độ dài biến đổi cho Sequence*, codec image, Scan trip phụ thuộc runtime (cần primitive loop thật hoặc compiler).
4. **IQ/TQ + weight vision + mmap shard.** Q2_K/Q3_K/Q6_K decode theo layout ggml bit-exact sau self-gate; IQ/TQ chưa có vector tham chiếu trong repo (vẫn raise loud — đúng). Vision graph builder đã có, thiếu weight thật. mmap shard cần intrinsic memory-map + index đa shard.
5. **Đo lại benchmark.** Số đo latency training nằm ở `benchmark/BENCHMARKS.md` §9 (hiện tại: §9.21–§9.22 — `benchmark/bseq_post14.exe` build bằng `D:\TokenVector\Release\dist\tkvc-tv.exe`, torch đo cùng phiên). `tv/benchmarks/compare_rivals.tkv` vẫn là smoke suite legacy — các hằng số ORT/TORCH trong đó là literal, không phải phép đo.

## 8. Khi nào chọn ai

- **TokenVector**: latency CPU cỡ micro-giây, endurance Zero-GC, nhúng gọn, serving OpenAI ở edge cho graph ONNX nhỏ
- **ONNX Runtime**: ONNX tổng quát, plugin EP GPU/NPU (CUDA/WebGPU/QNN), GenAI INT4/NVFP4 Windows, training/đa ngôn ngữ (1.29)
- **llama.cpp**: serving LLM text-in/out ngay hôm nay (GGUF, sampling, template, draft/EAGLE-3/n-gram speculative, mtmd multimodal, offload GPU; v0.4.0)
- **PyTorch**: training, nghiên cứu, custom op, serving GPU lớn (2.14 + torchao 0.18)

## 9. Quick start (`.tkv`)

```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc tv/examples/quickstart.tkv benchmark/quickstart.exe main
Set-Content benchmark/quickstart.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/quickstart.exe     # Y[0] = 2.25 / quickstart: OK
```

API làm việc trên **graph** — không có `tv.inference.load_onnx()`, không `NDArray`, không
`tensor.zeros`. Dựng `ExecutionGraph`, rồi `session_from_graph(g, 1)` và
`session.run_single(x, "X", "Y")`. Xem `USER_GUIDE_VI.md` và `tv/examples/quickstart.tkv`
cho snippet đầy đủ, và `docs/HANDOFF.md` cho các stub chưa nối.

Dòng CPU mô tả cái mà mã *đang là* (kernel native AVX2/FMA qua DLL đồng hành; fallback scalar `.tkv` làm tham chiếu ngữ nghĩa; arena chưa zero-GC). Số đo xem `benchmark/BENCHMARKS.md` §9.22 (cùng phiên với torch; full step 0.989× @seq1024).
