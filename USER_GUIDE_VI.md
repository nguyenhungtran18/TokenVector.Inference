# Hướng dẫn Sử dụng TokenVector.Inference (Tiếng Việt)

## 1. Tổng quan
`TokenVector.Inference` cung cấp nền tảng phục vụ suy luận mô hình học sâu, tối ưu hóa đồ thị tính toán và lượng tử hóa đa độ chính xác cho C# và Trình biên dịch TokenVector (`.tv`).

## 2. Nạp Mô hình ONNX
```csharp
using TokenVector.Inference.Runtime;
using TokenVector.Numerics.Core;

// Nạp trực tiếp từ đường dẫn file hoặc byte array
using var session = new InferenceSession("model.onnx");
```

## 3. Thực thi Suy luận Zero-GC Hot Path
Để đảm bảo tuyệt đối 0 Byte cấp phát trên Heap trong vòng lặp phục vụ:
```csharp
// Khởi tạo sẵn buffer I/O bên ngoài hot path
var inputTensor = new NDArray<float>(1, 64);
var outputTensor = new NDArray<float>(1, 10);

NamedNDArray[] inputs = new NamedNDArray[1] { new("input", inputTensor) };
NamedNDArray[] outputs = new NamedNDArray[1] { new("output", outputTensor) };

// Vòng lặp Hot Path (0 Byte GC)
for (int i = 0; i < 100000; i++)
{
    session.Run(inputs.AsSpan(), outputs.AsSpan());
}
```

## 4. Lượng tử hóa Mô hình (PTQ)
```csharp
using TokenVector.Inference.Quantization;

float[] calibrationData = LayDuLieuHieuChuan();
var qParams = QuantizationEngine.CalibrateMinMax(calibrationData, symmetric: true);

sbyte[] quantized = new sbyte[calibrationData.Length];
QuantizationEngine.QuantizeToInt8Symmetric(calibrationData, quantized, qParams.Scale);
```

## 5. Phục vụ Nhúng & Dynamic Batching
```csharp
using TokenVector.Inference.Server;

// Khởi chạy Embedded Micro-Server trên cổng 8080 (độ trễ < 1ms)
using var server = new EmbeddedServer(session, port: 8080);
```
