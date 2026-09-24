# Hướng dẫn Sử dụng TokenVector.Inference (Tiếng Việt)

## 1. Tổng quan
`TokenVector.Inference` cung cấp nền tảng phục vụ suy luận mô hình học sâu, tối ưu hóa đồ thị tính toán và lượng tử hóa đa độ chính xác cho ngôn ngữ TokenVector (`.tkv`) và trình biên dịch `tkvc`.

## 2. Nạp Mô hình ONNX
```tokenvector
import tv.inference

# Nạp từ đường dẫn file
session = tv.inference.load_onnx("model.onnx")
```

## 3. Thực thi Hot Path
```tokenvector
import tv.inference
import tv.numerics.tensor

session = tv.inference.load_onnx("model.onnx")
input_tensor = tensor.zeros([1, 64], dtype=float32)
output_tensor = session.run(input_tensor)
print("Output shape:", output_tensor.shape)
session.close()
```

Buffer được cấp phát sẵn cho đường tái sử dụng arena (mục tiêu Zero-GC).

## 4. Lượng tử hóa Mô hình (PTQ)
```tokenvector
import tv.quantization.engine

# Hiệu chuẩn MinMax / KL / percentile / MSE trên activation,
# rồi chuyển trọng số INT8 / INT4 / FP8 — xem tv/quantization/engine.tkv
```

CLI:
```powershell
tkvc build tv/tools/cli.tkv
# calibrate --model path.onnx ...
```

## 5. Phục vụ Nhúng & Dynamic Batching
```tokenvector
import tv.inference

session = tv.inference.load_onnx("model.onnx")
server = session.serve(port=8080)
# API tương thích OpenAI: /v1/chat/completions, SSE streaming, /metrics
```

## 6. Toolchain
```powershell
# Vị trí trình biên dịch
D:\TokenVector\3.code\dist\tkvc.exe build <source.tkv> [--out app.exe]
```
