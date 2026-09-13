# TokenVector.Inference User Guide

## 1. Overview
`TokenVector.Inference` provides high-performance deep learning model serving, graph optimizations, and multi-precision quantization for C# and the TokenVector compiler ecosystem.

## 2. Loading ONNX Models
```csharp
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

// Load directly from disk or byte buffer
using var session = new InferenceSession("model.onnx");
```

## 3. Zero-Allocation Hot Path Execution
To guarantee 0 Byte allocations on the managed heap:
```csharp
// Pre-allocate input and output tensors
var inputTensor = new NDArray<float>(1, 64);
var outputTensor = new NDArray<float>(1, 10);

NamedNDArray[] inputs = new NamedNDArray[1] { new("input", inputTensor) };
NamedNDArray[] outputs = new NamedNDArray[1] { new("output", outputTensor) };

// Hot-path loop (Zero GC)
for (int i = 0; i < 100000; i++)
{
    session.Run(inputs.AsSpan(), outputs.AsSpan());
}
```

## 4. Post-Training Quantization (PTQ)
```csharp
using TokenVector.Inference.Quantization;

float[] calibrationData = GetCalibrationActivations();
var qParams = QuantizationEngine.CalibrateMinMax(calibrationData, symmetric: true);

sbyte[] quantized = new sbyte[calibrationData.Length];
QuantizationEngine.QuantizeToInt8Symmetric(calibrationData, quantized, qParams.Scale);
```

## 5. Micro-Serving & Dynamic Batching
```csharp
using TokenVector.Inference.Server;

// Start embedded HTTP server on port 8080 (<1ms latency)
using var server = new EmbeddedServer(session, port: 8080);
```
