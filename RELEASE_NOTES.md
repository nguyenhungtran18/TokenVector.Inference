# TokenVector.Inference Release Notes - Version 1.0.0

## Release Summary
We are thrilled to announce the initial release of **TokenVector.Inference 1.0.0**, the ultra-low latency model serving, graph optimization, and quantization runtime engine for the TokenVector ecosystem — **100% TokenVector (`.tkv`)**.

## Key Features
- **Zero-GC Hot Path Engine:** Unified memory arena allocation with tensor liveness interval analysis.
- **High-Performance ONNX Parser:** Pure `.tkv` zero-copy protobuf wire reader.
- **Graph Optimization Passes:** `ConstantFoldingPass`, `DeadCodeEliminationPass`, and `OperatorFusionPass` (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`).
- **INT8 & FP8 Precision Quantization:** Static PTQ, dynamic quantization, grouped INT4, and IEEE FP8 E4M3/E5M2 formats.
- **Embedded Micro-Server & Dynamic Batcher:** Sub-millisecond REST server with OpenAI-compatible API, SSE, auth, limits, and metrics.
- **TokenVector Stdlib Module:** `stdlib/tv/inference` for language `.tkv`, built and tested with `tkvc`.
