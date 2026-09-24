# TokenVector.Inference User Guide

## 1. Overview
`TokenVector.Inference` provides high-performance deep learning model serving, graph optimizations, and multi-precision quantization for the TokenVector language (`.tkv`) and `tkvc` compiler.

## 2. Loading ONNX Models
```tokenvector
import tv.inference

# Load from disk
session = tv.inference.load_onnx("model.onnx")
```

## 3. Hot-Path Execution
```tokenvector
import tv.inference
import tv.numerics.tensor

session = tv.inference.load_onnx("model.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output shape:", output_tensor.shape)
session.close()
```

Buffers are pre-sized for the arena-reuse hot path (Zero-GC design goal).

## 4. Post-Training Quantization (PTQ)
```tokenvector
import tv.quantization.engine

# MinMax / KL / percentile / MSE calibration on activations,
# then INT8 / INT4 / FP8 weight transforms — see tv/quantization/engine.tkv
```

CLI:
```powershell
tkvc build tv/tools/cli.tkv
# calibrate --model path.onnx ...
```

## 5. Micro-Serving & Dynamic Batching
```tokenvector
import tv.inference

session = tv.inference.load_onnx("model.onnx")
server = session.serve(port=8080)
# OpenAI-compatible: /v1/chat/completions, SSE streaming, /metrics
```

## 6. Toolchain
```powershell
# Compiler location
D:\TokenVector\3.code\dist\tkvc.exe build <source.tkv> [--out app.exe]
```
