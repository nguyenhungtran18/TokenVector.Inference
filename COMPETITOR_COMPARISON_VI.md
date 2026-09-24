# So sánh Chức năng: TokenVector.Inference (thuần `.tv`) vs ONNX Runtime vs llama.cpp vs PyTorch

> Phạm vi: so sánh **chức năng** (làm được gì), không phải tốc độ. Repo hiện 100% TokenVector —
> 47 module `.tv`, 9.970 dòng (+ client Python mỏng `bindings/tv_client.py`), không còn C#/C++.
> Mọi compute đều thuần `.tv`; host intrinsic chỉ còn cấp phát thô, OS/HTTP/JSON/time-thread,
> đọc/ghi file-mạng và byte codec.
> Số latency ở §6 là **tham chiếu từ backend C# cũ**; cần đo lại bằng
> `tv run tv/benchmarks/compare_rivals.tv` trước khi trích dẫn.

Chú thích: (✅) có · (giới hạn) một phần · (thiếu) chưa có

## 1. Phủ model

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Định dạng đầu vào | ONNX (reader zero-copy, topo-sort Kahn, **writer/serializer**) + SafeTensors (F32/F16/BF16) + **GGUF** (`format/gguf.tv`: container, metadata KV, F32/F16/BF16/int, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_K/Q4_K verified, Q5_K provisional) | ONNX đầy đủ + shape inference | GGUF (+ tự tải từ HF) | Native / SafeTensors / export ONNX |
| Model lớn | ONNX single-file; không external-data / checkpoint chia shard | External data, >2 GB, symbolic shapes | GGUF chia shard, mmap, `--mlock` | SafeTensors chia shard, meta-device init |
| Phủ op | Danh sách đóng 82 op (`execution_node.tv` + `translator.map_op`): Gemm đầy đủ, Gather/Slice/Split/Squeeze/Unsqueeze/Expand, Clip/Cast/Pad/Where, ReduceMean/Sum/Max, GAP/MaxPool/AvgPool, BatchNorm, Flatten, ArgMax, Shape/Size, Min/Max/Mean/Sum, Pow/Sqrt/Exp/Log/Neg/Reciprocal/Abs, LeakyReLU/ELU/Softplus, Dropout, LogSoftmax, Trilu, TopK, Constant (fold), Quantize/DequantizeLinear, QLinearMatMul/Conv, MatMulInteger, DynamicQuantizeLinear, GatherND, ScatterElements, Range, EyeLike, ConstantOfShape, CumSum, InstanceNormalization, Mish, PRelu, Resize/Upsample, Einsum tổng quát, If/Loop (unroll), LSTM/GRU/RNN (unroll 2 chiều, Y/Y_h/Y_c), `StringConcat`/`StringEqual`/`StringLength` (bảng `str_values`), `Scan` unroll tĩnh (trip ≤ 128). **Op lạ báo lỗi rõ, không im lặng** + suy luận shape tĩnh. Còn thiếu: Sequence*/image ops (cần kiểu sequence) và Scan trip runtime | **200+ ops**, full opset | Native LLM-arch | Full ATen + custom op, autograd |
| Shape động | (thiếu) chỉ shape tĩnh | symbolic dims | context biến đổi | dynamic + `torch.compile` |
| Training | (giới hạn) train model nhỏ (`autograd.tv`: LinHead+SGD, MLP + Adam, softmax-CE, gradcheck) | on-device training | chỉ inference | training đầy đủ |

## 2. Tính năng LLM

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| Attention | Softmax O(1) bộ nhớ theo hàng, causal + mask cộng + `kv_offset`, GQA/MQA | FlashAttention qua EP, GQA, ctx 128K | Flash attention, SWA cache, prompt caching | SDPA + FlashAttention-2/3 (GPU) |
| KV-cache | Append-only 1 sequence + block manager paged (LRU, prefix-store chia sẻ) | paged (GenAI EP) | paged + tái dùng prefix | paged (vLLM) |
| Continuous batching | scheduler slots (waiting tới running, decode từng bước) + pack single-forward (block mask) | qua EP/Triton | mức token + parallel slots | qua vLLM/TRT-LLM |
| Sampling | temperature/top-k/top-p/min-p/penalty/greedy (LCG deterministic) | API `Generate()` | đầy đủ + grammar/JSON | đầy đủ |
| Tokenizer/template | BPE + WordPiece + SentencePiece-unigram; 5 built-in + Jinja-subset | qua EP | built-in + full Jinja | `transformers` |
| Pipeline text | `generate_text`/`chat_generate` nối vào server (usage, stop, seed) | GenAI đầy đủ | đầy đủ | đầy đủ |
| Speculative decoding | draft+target verify + bonus token | tùy EP | có | có |
| Tool-calling | loop + JSON-object mode + grammar masking GBNF-subset | GenAI | grammar, tool-call | hệ sinh thái |
| Đa modal | (thiếu) chỉ preprocess (resize/normalize/patchify); encoder ngoài phạm vi | tùy EP | (mmproj) | có |

## 3. Lượng tử hóa

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| INT8 weights | per-tensor sym/asym + **per-channel** + GEMM per-channel | static/dynamic, per-channel | Q8_0 | per-channel |
| INT8 activations | dynamic theo hàng | dynamic | mix K-quants | dynamic |
| INT4 | symmetric grouped + **asymmetric kiểu AWQ (zp/group)** + GEMM grouped (+ordered/act-order) + permute cols | AWQ + RTN | K-quants Q2 tới Q6 | torchao |
| FP8/FP16 | kiểu FP8 E4M3/E5M2 **+ GEMM compute E4M3** + **kiểu FP16 + GEMM compute** | FP16 + mixed | compute F16 | FP8/FP16/BF16 |
| Calibration | MinMax, **KL kiểu TensorRT**, Percentile, lưới MSE | MinMax, Entropy, Percentile | imatrix K-quants | MinMax, Histogram, MSE |
| QDQ | fold cặp QDQ + pass riêng; chưa xuất **file** QDQ | model QDQ | n/a (GGUF native) | có |

## 4. Serving

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| API OpenAI | `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (JSON + SSE), `/v1/rerank`, `/predict`, web UI chat (`GET /`) | (thiếu) cần Triton/OVMS | + `/v1/responses`, `/v1/rerank`, Anthropic-compat | qua sidecar |
| Streaming | SSE theo chunk text + `[DONE]` | n/a | SSE | có |
| Auth/giới hạn | API key, rate-limit/IP, max body, backpressure 429 | n/a | api-key, TLS | tùy frontend |
| Metrics | Prometheus + queue/dropped | profiler | `--metrics` | Prometheus |
| Batching | micro-batch 100-500us trên pool + slots liên tục | tuning EP | continuous + ubatch | dynamic |
| Khác | xem §2 (tool-calling, grammar) | GenAI `Generate()` | speculative | K8s autoscale |

## 5. Phần cứng, footprint và hệ sinh thái

| Khả năng | TokenVector `.tv` | ONNX Runtime | llama.cpp | PyTorch |
|---|---|---|---|---|
| CPU | kernel `.tv` block-SIMD, arena Zero-GC | MLAS, mọi arch + mobile | mọi CPU, AMX/NEON/SVE | MKL/OpenBLAS |
| GPU | (thiếu) provider stub báo lỗi rõ | CUDA/TRT/DirectML/ROCm/Metal | CUDA/HIP/Metal/Vulkan, offload tách lớp | CUDA/ROCm/XPU |
| NPU/edge | (thiếu) | QNN, OpenVINO, CoreML | Hexagon (giới hạn) | ExecuTorch |
| Footprint | thuần `.tv`, không DLL | ~100 MB+ kèm EP | một binary | cỡ GB |
| Provider API | `device.tv` (CPU live; accelerator stub) | EP plugin API | backend built-in | device/dispatch |
| Hệ sinh thái | tải HF-hub + cache, CLI (`serve/gen/export/calibrate/latency`), client Python HTTP mỏng | Olive, Model Zoo, Azure | HF GGUF, server distros | PyPI, HF Hub, CUDA wheels |
| Ngôn ngữ | chỉ `.tv` (+ client `.py` mỏng) | Python/C#/C++/Java/JS/Rust | C/C++ API + 20+ binding | Python-first |

## 6. Tham chiếu tốc độ cũ (backend C# — cần đo lại)

| Kịch bản | TokenVector (cũ) | ORT | LibTorch |
|---|---|---|---|
| Cold-start | **12.91 ms** | ~35 ms | ~28 ms |
| Fused Linear 64x128 GELU | **17.24 us** | ~320 us | ~410 us |
| Heap / 5.000 runs | **0 B** | >240 KB | marshalling |
| INT8 2.000 runs | **137.38 ms** | 380 ms | 310 ms |
| Endurance 250K | **127.791 req/s** | ~15K | ~25K |
| Burst 32-thread 10K | **169.735 req/s** | nghẽn lock | nghẽn interop |

## 7. Còn lại — lý do kỹ thuật chính xác từng mục

**Đã đóng trong đợt TRIZ này:** (1) LoRA + CheckpointArena, (2) HAL + kernel lane-abstract, (3) string-as-bytes + Scan driver/unroll tĩnh, (4) gated Q2_K/Q3_K/Q6_K + graph ViT, (5) tree-speculative + prefix-skip + beam. Op phủ thêm: `StringConcat`/`StringEqual`/`StringLength` (bảng `str_values`), `Scan` unroll tĩnh (trip ≤ 128); GGUF đọc Q2_K/Q3_K/Q6_K theo layout ggml + structural check + roundtrip self-gate.

1. **Pretrain full-model.** Autograd hiện chỉ có `LinHead` + ReLU + softmax-CE; LoRA + `CheckpointArena` cho fine-tune adapter, không phải backward đủ mọi op. Pretrain LLM trên CPU vẫn chậm hơn GPU 10–100 lần — chỉ có nghĩa khi kèm mục 2.
2. **Compute GPU/NPU.** `.tv` không có cấu trúc GPU (thread block, shared memory, MMA, launch kernel). Backend GPU = mở rộng **compiler** tv (repo chính) + shim CUDA/TensorRT/QNN. `device.tv`/`hal.tv` fail loud; `HalDispatch` là seam đăng ký khi có backend.
3. **Sequence container + image ops + Scan động.** String đã chạy trên `str_values` + UTF-8 (`string_tensor.tv`); Scan tĩnh unroll khi trip ≤ 128. Còn thiếu: kiểu sequence độ dài biến đổi cho Sequence*, codec image, Scan trip phụ thuộc runtime (cần primitive loop thật hoặc compiler).
4. **IQ/TQ + weight vision + mmap shard.** Q2_K/Q3_K/Q6_K decode theo layout ggml bit-exact sau self-gate; IQ/TQ chưa có vector tham chiếu trong repo (vẫn raise loud — đúng). Vision graph builder đã có, thiếu weight thật. mmap shard cần intrinsic memory-map + index đa shard.
5. **Đo lại benchmark.** Số §6 vẫn là backend C# cũ; suite thuần chỉ chạy dưới toolchain `tv` không có ở đây. Thuật toán (tree/beam/prefix-skip) đã trong repo — còn thiếu đo lại.

## 8. Khi nào chọn ai

- **TokenVector**: latency CPU cỡ micro-giây, endurance Zero-GC, nhúng gọn, serving OpenAI ở edge cho graph ONNX nhỏ
- **ONNX Runtime**: ONNX tổng quát 200 op, EP GPU/NPU, GenAI INT4 Windows, training/đa ngôn ngữ
- **llama.cpp**: serving LLM text-in/out ngay hôm nay (GGUF, sampling, template, speculative, offload GPU)
- **PyTorch**: training, nghiên cứu, custom op, serving GPU lớn

## 9. Quick start (`.tv`)

```tokenvector
import tv.inference

session = tv.inference.load_onnx("models/llama_block.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output length:", output_tensor.length)
session.close()
```
