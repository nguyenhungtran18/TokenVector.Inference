# Feature Comparison: TokenVector.Inference (pure `.tv`) vs ONNX Runtime vs llama.cpp vs PyTorch

> Scope: **functional** comparison (what each stack can do), not speed. Repo is now 100% TokenVector —
> 28 `.tv` modules, 5,285 lines, zero C#/C++ (`remaining_cs=0`). All compute (GEMM/norm/conv/
> elementwise/attention/quant-math/sort) is pure `.tv`; host intrinsics remain only for
> raw allocation, OS/HTTP/JSON/time-threads, and byte codecs (protobuf/utf8/FP8-bitcast).
> Latency numbers in §6 are **legacy C#-backend references** (Nov-2025, .NET 8 AVX2); re-measure post-migration with
> `tv run tv/benchmarks/compare_rivals.tv` before quoting them.

Legend: ✅ supported · ⚠️ partial/limited · ❌ missing

## 1. Model coverage

| Capability | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Input format | ONNX only (`tv/onnx/*`: zero-copy protobuf reader, Kahn topo-sort) | ONNX (full spec + shape inference) | GGUF (+ HF auto-download) | PyTorch native / SafeTensors / ONNX export |
| Large-model support | Single-file ONNX; no external-data / sharded checkpoints | External data, >2 GB, symbolic shapes | Sharded GGUF, mmap, `--mlock`/`--no-mmap` | Sharded SafeTensors, meta-device init |
| Op coverage | **~60 mapped** (`execution_node.tv` 60-type enum + `translator.map_op`): all previous + full Gemm (α/β), Gather/Slice/Split/Squeeze/Unsqueeze/Expand, Clip/Cast/Pad/Where, ReduceMean/Sum/Max, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, LogSoftmax, Trilu, TopK,
Constant (folded), QuantizeLinear/DequantizeLinear (tensor-wise + per-axis), QLinearMatMul
(exact 8-input math). **Unknown op → `UnsupportedOp` error (never silent)** + static shape
inference for intermediates | **200+ ops**, full ONNX opset, spec validation | LLM-arch native (Llama/Qwen/Gemma/Phi/…), not general ONNX | Full ATen + custom ops, autograd |
| Dynamic shapes | ❌ static shapes only | ✅ symbolic dims | ✅ variable context, `--cont-batching` | ✅ dynamic shapes, `torch.compile` |
| Training | ❌ inference-only | ✅ on-device training | ❌ inference-only | ✅ full training |

## 2. LLM inference features

| Capability | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Attention | O(1)-memory row-wise softmax, causal + additive mask + `kv_offset`, GQA/MQA (`tv/runtime/execution_node.tv`, `kv_cache.tv`) | FlashAttention via EPs, GQA, 128K ctx (Phi-3) | Flash attention, SWA cache, prompt caching | SDPA + FlashAttention-2/3 (GPU) |
| KV-cache | Append-only `AttentionKvCache` (single sequence) | ✅ paged (via GenAI EP) | ✅ paged + prefix reuse across slots | ✅ paged (vLLM-style in ecosystem) |
| Continuous batching | ✅ slot scheduler (`cont_batcher.tv`: waiting→running admission, per-step decode over session pool) + paged block manager (`paged_kv.tv`: LRU eviction, prefix-ready block tables) | ⚠️ via EP/Triton, not core | ✅ token-level cont. batching, parallel slots | ✅ via vLLM/TensorRT-LLM |
| Sampling | ✅ temperature/top-k/top-p/min-p/repeat-penalty/greedy (`sampling.tv`, deterministic LCG) | ✅ `Generate()` API | ✅ full sampler + grammars/JSON schema | ✅ full sampler |
| Tokenizer / chat template | ✅ byte-level BPE + WordPiece (`tokenizer.tv`); 5 built-ins (chatml/llama3/mistral/gemma/phi) + Jinja-subset custom (`chat_template.tv`) | ⚠️ via GenAI/tokenizer EP | ✅ built-in + full Jinja | ✅ `transformers` |
| Speculative decoding | ❌ | ⚠️ EP-dependent | ✅ | ✅ (ecosystem) |
| Multimodal (image/audio) | ❌ | ⚠️ EP-dependent | ✅ (mmproj) | ✅ |

## 3. Quantization

| Capability | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| INT8 weights | ✅ per-tensor sym/asym + **per-channel** + per-channel GEMM (`engine.tv`, `matmul.tv`) | ✅ static/dynamic, per-channel | ✅ Q8_0 | ✅ dynamic/static, per-channel |
| INT8 activations | ✅ dynamic row-wise | ✅ dynamic quant | ✅ (mixed with K-quants) | ✅ dynamic |
| Sub-8-bit (INT4) | ✅ group-wise symmetric (GPTQ-class layout, nibble-packed + per-group scales) + grouped GEMM (`int4.tv`) | ✅ AWQ + RTN (DirectML/CUDA) | ✅ K-quants Q2–Q6 (+i-quants) | ⚠️ torchao (niche) |
| FP8 / FP16 | ✅ FP8 E4M3/E5M2 types **+ E4M3 GEMM compute path** (`matmul.tv`) | ✅ FP16 + mixed precision | ✅ F16 compute | ✅ FP8 (Hopper+), FP16/BF16 |
| Calibration | MinMax, **TensorRT-style KL**, Percentile, MSE grid-search | MinMax, Entropy(KL), Percentile | Imatrix-guided K-quants | MinMax, Histogram, MSE |
| QDQ export | ⚠️ QDQ roundtrips folded to `QuantizedGemm` (`QdqFoldPass`); no QDQ **file** export (no ONNX serializer yet) | ✅ QDQ ONNX models | n/a (GGUF native) | ✅ quantized export |

## 4. Serving

| Capability | TokenVector `.tv` (`tv/server/*`) | ONNX Runtime | llama.cpp `llama-server` | PyTorch (TorchServe) |
|---|---|---|---|---|
| OpenAI API | ✅ `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (JSON + SSE) + legacy `/predict` | ❌ (needs Triton/OVMS sidecar) | ✅ + `/v1/responses`, `/v1/rerank`, Anthropic-compat | ✅ via sidecars |
| Streaming | ✅ SSE chunked + `[DONE]` | n/a | ✅ SSE + OAI streaming | ✅ |
| Auth / limits | ✅ API key, per-IP token-bucket, max body, queue backpressure → 429 | n/a | ✅ `--api-key(_-file)`, TLS | ✅ (frontend-dependent) |
| Metrics | ✅ Prometheus (`/metrics`) + queue depth/dropped | ⚠️ profiler, per-EP | ✅ `--metrics`, slots monitoring | ✅ Prometheus |
| Batching | ✅ micro-batch window (100–500 µs) over session pool | EP/batch tuning | ✅ continuous + ubatch split | ✅ dynamic batching |
| Extras | ✅ built-in chat web UI (`GET /`); ❌ rerank, tool-calling | GenAI `Generate()` | ✅ web UI, grammars, tool calling, speculative | ✅ workflows, autoscale (K8s) |

## 5. Hardware & footprint

| Capability | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| CPU | ✅ SIMD-blocked `.tv` (VEC_W=8, 32-blocked INT8 dot), Zero-GC arena + liveness reuse | ✅ MLAS, all archs incl. ARM/mobile | ✅ all CPUs, AMX/NEON/SVE threads | ✅ MKL/OpenBLAS |
| GPU | ❌ | ✅ CUDA/TensorRT/DirectML/ROCm/Metal | ✅ CUDA/HIP/Metal/Vulkan/SYCL/offload split | ✅ CUDA/ROCm/XPU |
| NPU / edge | ❌ | ✅ QNN, OpenVINO, CoreML, NNAPI | ⚠️ Snapdragon Hexagon | ⚠️ ExecuTorch |
| Footprint | ✅ pure `.tv`, no native DLLs | ⚠️ ~100 MB+ with EPs | ✅ single binary | ❌ GB-scale |
| Provider API | ✅ `device.tv` (CPU live; CUDA/ROCm/Metal/Vulkan/NPU stubs fail loudly, never silent) | ✅ EP plugin API | N/A (built-in backends) | ✅ device/dispatch |
| Languages | `.tv` only | Python/C#/C++/Java/JS/Rust | C/C++ API + 20+ bindings | Python-first |

## 6. Legacy speed references (C# backend — re-measure after migration)

| Scenario | TokenVector (legacy) | ORT (C#) | LibTorch (C++) |
|---|---|---|---|
| Cold-start parse+arena | **12.91 ms** | ~35 ms | ~28 ms |
| Fused Linear 64×128 GELU | **17.24 µs** | ~320 µs | ~410 µs |
| Heap / 5,000 runs | **0 B Zero-GC** | >240 KB | marshalling |
| INT8 2,000 runs | **137.38 ms** | 380 ms | 310 ms |
| 250K endurance | **127,791 req/s** | ~15K RPS | ~25K RPS |
| 32-thread 10K burst | **169,735 req/s** | lock contention | interop bottleneck |

## 7. Coverage status (was roadmap — now closed except below)

- [x] **Unknown-op fails silent → `UnsupportedOp` + ~40 new ops** (Gather/Slice/Split/Squeeze/Unsqueeze/Expand/Clip/Cast/Pad/Where, Reduce×3, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, full Gemm, Concat runtime) + static shape inference.
- [x] **Tokenizer + sampling + templates** (`tv/text/*`: BPE/WordPiece, temp/top-k/top-p/min-p/penalty/greedy, 5 built-in chat formats + Jinja subset).
- [x] **Continuous batching + paged KV** (`cont_batcher.tv` slots + `paged_kv.tv` LRU blocks). Token-packing into one GEMM remains future work.
- [x] **INT4 grouped + FP8 compute** (`int4.tv`, `matmul_fp8_e4m3`); AWQ-specific zero-point/asym-group variants not yet.
- [x] **Provider abstraction + web UI + QDQ folding** (`device.tv`, `GET /`, `QdqFoldPass`).

Remaining (explicitly out of scope for an inference engine or next phase):
1. **Training/autograd** — PyTorch-only territory, not planned.
2. **GPU/NPU compute** — providers stubbed with loud errors; needs a GPU-capable tv backend.
3. **QDQ file export** — folding exists, ONNX serializer doesn't.
4. **Speculative decoding, tool-calling/grammars, rerank, multimodal** — llama.cpp leads; queued P2.
5. **Shared-prefix cache reuse across sequences** — block tables are prefix-ready, reuse policy not yet.

## 8. When to pick whom

| Pick | When… |
|---|---|
| **TokenVector** | µs CPU latency, Zero-GC endurance, tiny dependency-free embed, OpenAI-compatible edge serving of small ONNX graphs |
| **ONNX Runtime** | 200-op general ONNX, GPU/NPU EPs, INT4 GenAI on Windows, training or multi-language apps |
| **llama.cpp** | Text-in/text-out LLM serving today: GGUF quants, sampling, templates, speculative decoding, GPU offload |
| **PyTorch** | Training, research, custom autograd ops, GPU-scale serving via ecosystem |

## 9. Quick start (`.tv`)

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/llama_block.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output length:", output_tensor.length)
session.close()
```
