# TokenVector.Inference Release Notes - Version 1.0.0

## Release Summary
We are thrilled to announce the initial release of **TokenVector.Inference 1.0.0**, the ultra-low latency model serving, graph optimization, and quantization runtime engine for the TokenVector ecosystem.

## Key Features
- **Zero-GC Hot Path Engine:** Unified native memory arena allocation with tensor liveness intervals analysis.
- **High-Performance ONNX Parser:** Pure C# zero-copy protobuf wire reader (`FastProtobufReader`).
- **Graph Optimization Passes:** `ConstantFoldingPass`, `DeadCodeEliminationPass`, and `OperatorFusionPass` (`FusedLinear`, `FusedConv2D`, `FusedRMSNorm`).
- **INT8 & FP8 Precision Quantization:** Static PTQ, dynamic quantization, AVX2 / AVX-512 VNNI kernels, and IEEE FP8 E4M3/E5M2 formats.
- **Embedded Micro-Server & Dynamic Batcher:** Sub-millisecond REST/IPC server with lock-free channel queueing.
- **TokenVector Compiler Bridge:** Static CIL interop methods and stdlib `.tv` wrappers.
