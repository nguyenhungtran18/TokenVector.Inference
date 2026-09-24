# So sánh Chức năng: TokenVector.Inference (thuần `.tv`) vs ONNX Runtime vs llama.cpp vs PyTorch

> Phạm vi: so sánh **chức năng** (làm được gì), không phải tốc độ. Repo hiện 100% TokenVector —
> 28 module `.tv`, 5.285 dòng, không còn C#/C++ (`remaining_cs=0`). Mọi compute (GEMM/norm/conv/
> elementwise/attention/quant-math/sort) đều thuần `.tv`; host intrinsic chỉ còn cấp phát thô,
> OS/HTTP/JSON/time-thread và byte codec (protobuf/utf8/FP8-bitcast).
> Số latency ở §6 là **tham chiếu từ backend C# cũ** (11-2025, .NET 8 AVX2); cần đo lại sau migration bằng
> `tv run tv/benchmarks/compare_rivals.tv` trước khi trích dẫn.

Chú thích: ✅ có · ⚠️ một phần/hạn chế · ❌ chưa có

## 1. Phủ model

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Định dạng đầu vào | Chỉ ONNX (`tv/onnx/*`: protobuf reader zero-copy, topo-sort Kahn) | ONNX đầy đủ + shape inference | GGUF (+ tự tải từ HF) | Native / SafeTensors / export ONNX |
| Model lớn | ONNX single-file; không external-data / checkpoint chia shard | External data, >2 GB, symbolic shapes | GGUF chia shard, mmap, `--mlock` | SafeTensors chia shard, meta-device init |
| Phủ op | **~60 op đã map** (enum 60 kiểu + `translator.map_op`): thêm Gemm đầy đủ (α/β), Gather/Slice/Split/Squeeze/Unsqueeze/Expand, Clip/Cast/Pad/Where, ReduceMean/Sum/Max, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, LogSoftmax, Trilu, TopK,
Constant (fold), QuantizeLinear/DequantizeLinear (tensor + per-axis), QLinearMatMul
(math 8-input exact). **Op lạ → lỗi `UnsupportedOp` (không bao giờ im lặng)** + suy luận shape tĩnh | **200+ ops**, full opset | Native LLM-arch | Full ATen + custom op, autograd |
| Shape động | ❌ chỉ shape tĩnh | ✅ symbolic dims | ✅ context biến đổi, `--cont-batching` | ✅ dynamic shapes, `torch.compile` |
| Training | ❌ chỉ inference | ✅ on-device training | ❌ chỉ inference | ✅ training đầy đủ |

## 2. Tính năng LLM

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Attention | Softmax O(1) bộ nhớ theo hàng, causal + mask cộng + `kv_offset`, GQA/MQA (`execution_node.tv`, `kv_cache.tv`) | FlashAttention qua EP, GQA, ctx 128K (Phi-3) | Flash attention, SWA cache, prompt caching | SDPA + FlashAttention-2/3 (GPU) |
| KV-cache | `AttentionKvCache` append-only (1 sequence) | ✅ paged (qua GenAI EP) | ✅ paged + tái dùng prefix | ✅ paged (vLLM) |
| Continuous batching | ✅ scheduler slots (`cont_batcher.tv`: waiting→running, decode từng bước trên pool) + quản lý block paged (`paged_kv.tv`: LRU, sẵn sàng prefix) | ⚠️ qua EP/Triton, không ở core | ✅ cont. batching mức token, parallel slots | ✅ qua vLLM/TensorRT-LLM |
| Sampling | ✅ temperature/top-k/top-p/min-p/penalty/greedy (`sampling.tv`, LCG deterministic) | ✅ API `Generate()` | ✅ sampler đầy đủ + grammar/JSON schema | ✅ đầy đủ |
| Tokenizer / chat template | ✅ BPE byte-level + WordPiece (`tokenizer.tv`); 5 built-in (chatml/llama3/mistral/gemma/phi) + Jinja-subset (`chat_template.tv`) | ⚠️ qua GenAI/tokenizer EP | ✅ built-in + full Jinja | ✅ `transformers` |
| Speculative decoding | ❌ | ⚠️ tùy EP | ✅ | ✅ (hệ sinh thái) |
| Đa modal (ảnh/âm thanh) | ❌ | ⚠️ tùy EP | ✅ (mmproj) | ✅ |

## 3. Lượng tử hóa

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| INT8 weights | ✅ per-tensor sym/asym + **per-channel** + GEMM per-channel (`engine.tv`, `matmul.tv`) | ✅ static/dynamic, per-channel | ✅ Q8_0 | ✅ dynamic/static, per-channel |
| INT8 activations | ✅ dynamic theo hàng | ✅ dynamic quant | ✅ (mix với K-quants) | ✅ dynamic |
| Dưới 8-bit (INT4) | ✅ group-wise symmetric (layout lớp GPTQ, pack nibble + scale/group) + GEMM grouped (`int4.tv`) | ✅ AWQ + RTN | ✅ K-quants Q2–Q6 (+i-quants) | ⚠️ torchao |
| FP8 / FP16 | ✅ kiểu FP8 E4M3/E5M2 **+ đường compute GEMM E4M3** (`matmul.tv`) | ✅ FP16 + mixed | ✅ compute F16 | ✅ FP8/FP16/BF16 |
| Calibration | MinMax, **KL kiểu TensorRT**, Percentile, lưới MSE | MinMax, Entropy(KL), Percentile | K-quants dẫn bởi imatrix | MinMax, Histogram, MSE |
| Xuất QDQ | ⚠️ cặp QDQ được fold thành `QuantizedGemm` (`QdqFoldPass`); chưa xuất **file** QDQ (chưa có serializer ONNX) | ✅ model QDQ | n/a (GGUF native) | ✅ |

## 4. Serving

| Khả năng | TokenVector `.tv` (`tv/server/*`) | ONNX Runtime | llama.cpp `llama-server` | PyTorch (TorchServe) |
|---|---|---|---|---|
| API OpenAI | ✅ `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (JSON + SSE) + `/predict` legacy | ❌ (cần sidecar Triton/OVMS) | ✅ + `/v1/responses`, `/v1/rerank`, tương thích Anthropic | ✅ qua sidecar |
| Streaming | ✅ SSE theo chunk + `[DONE]` | n/a | ✅ SSE + streaming OAI | ✅ |
| Auth / giới hạn | ✅ API key, token-bucket theo IP, max body, backpressure hàng đợi → 429 | n/a | ✅ `--api-key(_-file)`, TLS | ✅ (tùy frontend) |
| Metrics | ✅ Prometheus (`/metrics`) + queue depth/dropped | ⚠️ profiler, theo EP | ✅ `--metrics`, giám sát slots | ✅ Prometheus |
| Batching | ✅ micro-batch (100–500 µs) trên session pool | tuning EP/batch | ✅ continuous + tách ubatch | ✅ dynamic batching |
| Khác | ✅ web UI chat built-in (`GET /`); ❌ rerank, tool-calling | GenAI `Generate()` | ✅ web UI, grammar, tool calling, speculative | ✅ workflow, autoscale |

## 5. Phần cứng & footprint

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| CPU | ✅ kernel `.tv` block-SIMD (VEC_W=8, dot INT8 32-block), arena Zero-GC + tái dùng liveness | ✅ MLAS, mọi kiến trúc kể cả ARM/mobile | ✅ mọi CPU, AMX/NEON/SVE threads | ✅ MKL/OpenBLAS |
| GPU | ❌ | ✅ CUDA/TensorRT/DirectML/ROCm/Metal | ✅ CUDA/HIP/Metal/Vulkan/SYCL, offload tách lớp | ✅ CUDA/ROCm/XPU |
| NPU / edge | ❌ | ✅ QNN, OpenVINO, CoreML, NNAPI | ⚠️ Snapdragon Hexagon | ⚠️ ExecuTorch |
| Footprint | ✅ thuần `.tv`, không DLL | ⚠️ ~100 MB+ kèm EP | ✅ một binary | ❌ cỡ GB |
| Provider API | ✅ `device.tv` (CPU live; CUDA/ROCm/Metal/Vulkan/NPU stub báo lỗi rõ, không fallback im lặng) | ✅ EP plugin API | N/A (backend built-in) | ✅ device/dispatch |
| Ngôn ngữ | chỉ `.tv` | Python/C#/C++/Java/JS/Rust | C/C++ API + 20+ binding | Python-first |

## 6. Tham chiếu tốc độ cũ (backend C# — cần đo lại sau migration)

| Kịch bản | TokenVector (cũ) | ORT (C#) | LibTorch (C++) |
|---|---|---|---|
| Cold-start parse+arena | **12.91 ms** | ~35 ms | ~28 ms |
| Fused Linear 64×128 GELU | **17.24 µs** | ~320 µs | ~410 µs |
| Heap / 5.000 runs | **0 B Zero-GC** | >240 KB | marshalling |
| INT8 2.000 runs | **137.38 ms** | 380 ms | 310 ms |
| Endurance 250K | **127.791 req/s** | ~15K RPS | ~25K RPS |
| Burst 32-thread 10K | **169.735 req/s** | nghẽn lock | nghẽn interop |

## 7. Trạng thái phủ (roadmap cũ — nay đã đóng trừ mục dưới)

- [x] **Op lạ fail im lặng → `UnsupportedOp` + ~40 op mới** (Gather/Slice/Split/Squeeze/Unsqueeze/Expand/Clip/Cast/Pad/Where, Reduce×3, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, Gemm đầy đủ, Concat runtime) + suy luận shape tĩnh.
- [x] **Tokenizer + sampling + template** (`tv/text/*`: BPE/WordPiece, temp/top-k/top-p/min-p/penalty/greedy, 5 format chat + Jinja-subset).
- [x] **Continuous batching + paged KV** (`cont_batcher.tv` slots + `paged_kv.tv` block LRU). Pack token chung một GEMM để dành tương lai.
- [x] **INT4 grouped + compute FP8** (`int4.tv`, `matmul_fp8_e4m3`); biến thể AWQ zero-point/asym-group chưa có.
- [x] **Provider abstraction + web UI + fold QDQ** (`device.tv`, `GET /`, `QdqFoldPass`).

Còn lại (ngoài phạm vi engine inference hoặc giai đoạn sau):
1. **Training/autograd** — sân của PyTorch, không planned.
2. **Compute GPU/NPU** — provider stub báo lỗi rõ; cần backend tv có GPU.
3. **Xuất file QDQ** — đã fold, chưa có serializer ONNX.
4. **Speculative decoding, tool-calling/grammar, rerank, đa modal** — llama.cpp dẫn; P2.
5. **Tái dùng prefix cache giữa các sequence** — block table đã sẵn sàng, policy reuse chưa.

## 8. Khi nào chọn ai

| Chọn | Khi… |
|---|---|
| **TokenVector** | latency CPU cỡ µs, endurance Zero-GC, nhúng gọn không phụ thuộc, serving OpenAI-compatible ở edge cho graph ONNX nhỏ |
| **ONNX Runtime** | ONNX tổng quát 200 op, EP GPU/NPU, GenAI INT4 trên Windows, training hoặc app đa ngôn ngữ |
| **llama.cpp** | Serving LLM text-in/text-out ngay hôm nay: quant GGUF, sampling, template, speculative decoding, offload GPU |
| **PyTorch** | Training, nghiên cứu, custom autograd op, serving GPU quy mô lớn qua hệ sinh thái |

## 9. Quick start (`.tv`)

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/llama_block.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Độ dài đầu ra:", output_tensor.length)
session.close()
```
