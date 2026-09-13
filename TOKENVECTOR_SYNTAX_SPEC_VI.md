# Đặc tả Cú pháp Ngôn ngữ TokenVector: Module Inference (Tiếng Việt)

## 1. Khai báo Import
```tokenvector
import tv.inference
import tv.numerics.tensor
```

## 2. Nạp và Khởi tạo Session
```tokenvector
# Nạp file ONNX vào phiên làm việc Zero-GC
session = tv.inference.load_onnx("models/llama_block.onnx")
```

## 3. Thực thi Suy luận Forward Pass
```tokenvector
# Khởi tạo tensor đầu vào
input_tensor = tensor.zeros([1, 64], dtype=float32)

# Thực thi suy luận
output_tensor = session.run(input_tensor)

print("Độ dài đầu ra:", output_tensor.length)
```

## 4. Giải phóng Tài nguyên Session
```tokenvector
session.close()
```
