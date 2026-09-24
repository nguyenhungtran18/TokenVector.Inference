# Feature Comparison: TokenVector.Inference (pure `.tkv`) vs ONNX Runtime vs llama.cpp vs PyTorch

> Scope: **functional** comparison (what each stack can do), not speed. Repo is 100% TokenVector —
> 47 `.tkv` modules, 9,970 lines. All compute and clients are pure `.tkv`;
> host intrinsics remain only for raw allocation, OS/HTTP/JSON/time-threads, file/network IO, and
> byte codecs (protobuf/utf8/FP8/FP16-bitcast).
> Measure speed only with `tkvc build tv/benchmarks/compare_rivals.tkv` on the pure engine.

Legend: ✅ supported · ⚠️ partial/limited · ❌ missing

## 1. Model coverage

| Capability | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Input format | ONNX (zero-copy reader, Kahn topo-sort, **writer/serializer**) + SafeTensors weights (F32/F16/BF16) + **GGUF** (`format/gguf.tkv`: container, KV metadata, F32/F16/BF16/ints, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K/Q4_K verified, Q5_K provisional) | ONNX (full spec + shape inference) | GGUF (+ HF auto-download) | PyTorch native / SafeTensors / ONNX export |
| Large-model support | Single-file ONNX; no external-data / sharded checkpoints | External data, >2 GB, symbolic shapes | Sharded GGUF, mmap, `--mlock`/`--no-mmap` | Sharded SafeTensors, meta-device init |
| Op coverage | **82-op closed list** (`execution_node.tkv` enum + `translator.map_op`): all previous + full Gemm (α/β), Gather/Slice/Split/Squeeze/Unsqueeze/Expand, Clip/Cast/Pad/Where, ReduceMean/Sum/Max, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, LogSoftmax, Trilu, TopK,
Constant (folded), QuantizeLinear/DequantizeLinear (tensor-wise + per-axis), QLinearMatMul,
QLinearConv, MatMulInteger, DynamicQuantizeLinear, GatherND, ScatterElements, Range, EyeLike,
ConstantOfShape, CumSum, InstanceNormalization, Mish, PRelu, Resize/Upsample, Einsum (generic),
If/Loop (unrolled), LSTM/GRU/RNN (unrolled, bidir, Y/Y_h/Y_c), `StringConcat`/`StringEqual`/`StringLength` (`str_values` side table), static `Scan` unroll (trip ≤ 128). **Unknown op → `UnsupportedOp`
error (never silent)** + static shape inference. Still missing: first-class Sequence*/image data
ops (need sequence dtype in `tv.numerics`) and Scan with runtime trip count | **200+ ops**, full ONNX opset, spec validation | LLM-arch native (Llama/Qwen/Gemma/Phi/…), not general ONNX | Full ATen + custom ops, autograd |
| Dynamic shapes | ❌ static shapes only | ✅ symbolic dims | ✅ variable context, `--cont-batching` | ✅ dynamic shapes, `torch.compile` |
| Training | ⚠️ small-model training (`training/autograd.tkv`: LinHead+SGD, MLP + Adam, softmax-CE, grad-checked) — no full-model pretraining | ✅ on-device training | ❌ inference-only | ✅ full training |

## 2. LLM inference features

| Capability | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Attention | O(1)-memory row-wise softmax, causal + additive mask + `kv_offset`, GQA/MQA (`tv/runtime/execution_node.tkv`, `kv_cache.tkv`) | FlashAttention via EPs, GQA, 128K ctx (Phi-3) | Flash attention, SWA cache, prompt caching | SDPA + FlashAttention-2/3 (GPU) |
| KV-cache | Append-only `AttentionKvCache` (single sequence) | ✅ paged (via GenAI EP) | ✅ paged + prefix reuse across slots | ✅ paged (vLLM-style in ecosystem) |
| Continuous batching | ✅ slot scheduler (`cont_batcher.tkv`: waiting→running admission, per-step decode over session pool) + paged block manager (`paged_kv.tkv`: LRU eviction, prefix-ready block tables) | ⚠️ via EP/Triton, not core | ✅ token-level cont. batching, parallel slots | ✅ via vLLM/TensorRT-LLM |
| Sampling | ✅ temperature/top-k/top-p/min-p/repeat-penalty/greedy (`sampling.tkv`, deterministic LCG) | ✅ `Generate()` API | ✅ full sampler + grammars/JSON schema | ✅ full sampler |
| Tokenizer / chat template | ✅ byte-level BPE + WordPiece + SentencePiece-unigram (`tv/text/*`); 5 built-ins (chatml/llama3/mistral/gemma/phi) + Jinja-subset custom | ⚠️ via GenAI/tokenizer EP | ✅ built-in + full Jinja | ✅ `transformers` |
| Speculative decoding | ✅ draft+target verify loop (`speculative.tkv`) | ⚠️ EP-dependent | ✅ | ✅ (ecosystem) |
| Multimodal (image/audio) | ❌ | ⚠️ EP-dependent | ✅ (mmproj) | ✅ |

## 3. Quantization

| Capability | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| INT8 weights | ✅ per-tensor sym/asym + **per-channel** + per-channel GEMM (`engine.tkv`, `matmul.tkv`) | ✅ static/dynamic, per-channel | ✅ Q8_0 | ✅ dynamic/static, per-channel |
| INT8 activations | ✅ dynamic row-wise | ✅ dynamic quant | ✅ (mixed with K-quants) | ✅ dynamic |
| Sub-8-bit (INT4) | ✅ group-wise symmetric + **AWQ-style asymmetric (per-group zp)** + grouped GEMM (+ordered/act-order variant) + col permute utils | ✅ AWQ + RTN (DirectML/CUDA) | ✅ K-quants Q2–Q6 (+i-quants) | ⚠️ torchao (niche) |
| FP8 / FP16 | ✅ FP8 E4M3/E5M2 types **+ E4M3 GEMM** + **FP16 type + GEMM compute path** | ✅ FP16 + mixed precision | ✅ F16 compute | ✅ FP8 (Hopper+), FP16/BF16 |
| Calibration | MinMax, **TensorRT-style KL**, Percentile, MSE grid-search | MinMax, Entropy(KL), Percentile | Imatrix-guided K-quants | MinMax, Histogram, MSE |
| QDQ export | ⚠️ QDQ roundtrips folded to `QuantizedGemm` (`QdqFoldPass`); no QDQ **file** export (no ONNX serializer yet) | ✅ QDQ ONNX models | n/a (GGUF native) | ✅ quantized export |

## 4. Serving

| Capability | TokenVector `.tkv` (`tv/server/*`) | ONNX Runtime | llama.cpp `llama-server` | PyTorch (TorchServe) |
|---|---|---|---|---|
| OpenAI API | ✅ `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (JSON + SSE) + legacy `/predict` | ❌ (needs Triton/OVMS sidecar) | ✅ + `/v1/responses`, `/v1/rerank`, Anthropic-compat | ✅ via sidecars |
| Streaming | ✅ SSE chunked + `[DONE]` | n/a | ✅ SSE + OAI streaming | ✅ |
| Auth / limits | ✅ API key, per-IP token-bucket, max body, queue backpressure → 429 | n/a | ✅ `--api-key(_-file)`, TLS | ✅ (frontend-dependent) |
| Metrics | ✅ Prometheus (`/metrics`) + queue depth/dropped | ⚠️ profiler, per-EP | ✅ `--metrics`, slots monitoring | ✅ Prometheus |
| Batching | ✅ micro-batch window (100–500 µs) over session pool | EP/batch tuning | ✅ continuous + ubatch split | ✅ dynamic batching |
| Extras | ✅ built-in chat web UI (`GET /`), `/v1/rerank`, tool-calling loop + JSON-object mode, **GBNF-subset grammar masking** (`text/grammar.tkv`), shared-prefix block store, packed single-forward batching | GenAI `Generate()` | ✅ web UI, grammars, tool calling, speculative | ✅ workflows, autoscale (K8s) |

## 5. Hardware & footprint

| Capability | TokenVector `.tkv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| CPU | ✅ SIMD-blocked `.tkv` (VEC_W=8, 32-blocked INT8 dot), Zero-GC arena + liveness reuse | ✅ MLAS, all archs incl. ARM/mobile | ✅ all CPUs, AMX/NEON/SVE threads | ✅ MKL/OpenBLAS |
| GPU | ❌ | ✅ CUDA/TensorRT/DirectML/ROCm/Metal | ✅ CUDA/HIP/Metal/Vulkan/SYCL/offload split | ✅ CUDA/ROCm/XPU |
| NPU / edge | ❌ | ✅ QNN, OpenVINO, CoreML, NNAPI | ⚠️ Snapdragon Hexagon | ⚠️ ExecuTorch |
| Footprint | ✅ pure `.tkv`, no native DLLs | ⚠️ ~100 MB+ with EPs | ✅ single binary | ❌ GB-scale |
| Provider API | ✅ `device.tkv` (CPU live; CUDA/ROCm/Metal/Vulkan/NPU stubs fail loudly, never silent) | ✅ EP plugin API | N/A (built-in backends) | ✅ device/dispatch |
| Ecosystem | ✅ HF-hub download+cache (`tools/hub.tkv`), CLI (`serve/gen/export/calibrate/latency`), HTTP clients over the OpenAI-compatible API, vision preprocess frontend (resize/normalize/patchify) | ✅ Olive, Model Zoo, Azure | ✅ HF GGUF ecosystem, server distributions | ✅ package index, model hub, accelerator wheels |

## 6. Speed references

Speed numbers live only in a dated run of `tv/benchmarks/compare_rivals.tkv` (see `BENCHMARKS.md`).
No historical backend figures are kept in this document.

## 7. Coverage status (closed list + remaining items with technical reasons)

Closed:
- [x] **Silent unknown-op → `UnsupportedOp` + 82-op coverage** (index/shape/reduce/pool/norm/elementwise/Gemm/quant/pooling/pool-family, `Constant` folding, `If`/`Loop` unrolling, `LSTM`/`GRU`/`RNN` unrolling with Y/Y_h/Y_c, `Resize`, generic `Einsum`, full QDQ family with exact math, `StringConcat`/`StringEqual`/`StringLength` via `str_values` side table, static `Scan` unroll) + static shape inference.
- [x] **Text pipeline** (`tv/text/*`: BPE/WordPiece/SentencePiece-unigram, temp/top-k/top-p/min-p/penalty/greedy, 5 chat formats + Jinja subset, `generate_text`, speculative single-chain, **multi-draft tree speculative** (`tree_speculative.tkv`), **constrained beam search** (`beam.tkv`), tool loop, GBNF-subset grammar masking).
- [x] **Serving** (slot continuous batching, paged KV + shared prefix store, **prefix compute-skip planner** `prefix_skip.tkv`, packed single-forward batching, OpenAI API + SSE + rerank + auth/limits/metrics, web UI).
- [x] **Quantization** (INT8 per-tensor/per-channel/dynamic, grouped INT4 sym + AWQ-asym + act-order, FP8/FP16 compute, 4 calibrations, QDQ folding; quantized int8/int32 IO + int8 initializers; **gated Q2_K/Q3_K/Q6_K dequant** from ggml reference layouts with structural checks + discrete roundtrip self-gate).
- [x] **Formats & ecosystem** (ONNX reader + writer/serializer, SafeTensors, GGUF container + verified quants, HF-hub cache, CLI, vision preprocess frontend + **ViT graph builder** `vision/vit.tkv`, micro-training with grad-checked autograd + **LoRA adapters + checkpoint arena** `training/lora.tkv`, **HAL lane-abstract kernels** `runtime/hal.tkv`).
- [x] **TRIZ groups implemented this pass** — (1) LoRA + CheckpointArena, (2) HAL + lane-abstract kernels, (3) String-as-bytes + Scan driver/unroll, (4) Gated K-quants + ViT graph, (5) Tree-speculative + prefix-skip + beam.

Remaining — each with its precise technical reason:

1. **Full-model pretraining.** Our autograd covers `LinHead` + ReLU + softmax-CE only; LoRA adapters + `CheckpointArena` give a second allocation policy for adapter fine-tunes, not full backward for all ops. Pretraining needs backward kernels for every forward op **plus** multi-GB weight streaming. CPU pretraining at LLM scale is 10–100x too slow — it only makes sense together with item 2.
2. **GPU/NPU compute.** The `.tkv` language as defined has no GPU constructs: no thread blocks, shared memory, tensor-core MMA, or kernel-launch API — every existing `extern def` is scalar/host-side (alloc, IO, codecs, time). A GPU backend means extending the tv **compiler** itself (TokenVector main repo, not this one) plus vendor SDK shims (CUDA/TensorRT/QNN). `device.tkv` / `hal.tkv` stubs exist so this fails loudly instead of silently; `HalDispatch` is the registration seam once a backend exists.
3. **Full sequence container + image data ops + dynamic Scan.** String tensors ride `str_values` + UTF-8 byte pools (`string_tensor.tkv`); static `Scan` unrolls when trip ≤ 128 (`_expand_scan`). Remaining: first-class variable-length **sequence** type for Sequence* ops, image tensor codecs, and Scan whose trip depends on runtime data (needs a real loop primitive or compiler support).
4. **IQ/TQ dequant + vision encoder weights + GGUF-shard mmap.** Q2_K/Q3_K/Q6_K are now decoded from bit-exact ggml layouts behind a self-gate; IQ/TQ nonlinear layouts still lack a checked reference vector set in-tree (silent-corruption risk remains — loud error is correct). Vision needs actual ViT/CLIP **weights** (graph builder exists). Sharded mmap needs a memory-mapped-file host intrinsic plus multi-shard index resolution.
5. **Re-measured benchmarks.** Every published latency number must come from `tv/benchmarks/compare_rivals.tkv` on this pure engine (`tkvc` at `D:\TokenVector\3.code\dist\tkvc.exe build …`). Algorithm work (tree/beam/prefix-skip) is in-tree; measurement is the remaining gap.

## 8. When to pick whom

| Pick | When… |
|---|---|
| **TokenVector** | µs CPU latency, Zero-GC endurance, tiny dependency-free embed, OpenAI-compatible edge serving of small ONNX graphs |
| **ONNX Runtime** | 200-op general ONNX, GPU/NPU EPs, INT4 GenAI on Windows, training or multi-language apps |
| **llama.cpp** | Text-in/text-out LLM serving today: GGUF quants, sampling, templates, speculative decoding, GPU offload |
| **PyTorch** | Training, research, custom autograd ops, GPU-scale serving via ecosystem |

## 9. Quick start (`.tkv`)

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/llama_block.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output length:", output_tensor.length)
session.close()
```
