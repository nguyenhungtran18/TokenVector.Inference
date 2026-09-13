# TokenVector Language Syntax Specification: Inference Module

## 1. Import Directive
```tokenvector
import tv.inference
import tv.numerics.tensor
```

## 2. Session Loading and Initialization
```tokenvector
# Load ONNX model into native Zero-GC execution session
session = tv.inference.load_onnx("models/llama_block.onnx")
```

## 3. Synchronous Tensor Forward Pass
```tokenvector
# Create input tensor
input_tensor = tensor.zeros([1, 64], dtype=float32)

# Execute inference
output_tensor = session.run(input_tensor)

print("Output length:", output_tensor.length)
```

## 4. Closing Session
```tokenvector
session.close()
```
