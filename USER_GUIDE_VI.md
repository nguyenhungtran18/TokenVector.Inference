# Hướng dẫn Sử dụng TokenVector.Inference (Tiếng Việt)

## 1. Tổng quan
`TokenVector.Inference` là runtime đồ thị thuần TokenVector (`.tkv`): tối ưu hóa đồ thị,
lượng tử hóa INT8/INT4/FP8, và micro-batcher trong process. Mọi thứ dưới đây là API
**đã kiểm chứng**. `docs/HANDOFF.md` liệt kê lỗ hổng, `docs/SIMD_NOTES.md` tóm tắt ba tầng SIMD đang dùng.

## 2. Build và chạy ví dụ nhanh
```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc tv/examples/quickstart.tkv benchmark/quickstart.exe main
Set-Content benchmark/quickstart.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/quickstart.exe      # Y[0] = 2.25 / quickstart: OK
```

## 3. Dựng và chạy graph
Không có `NDArray` và không có `tv.inference.load_onnx()`. Tensor là `list[f32]` phẳng,
session đến từ một `ExecutionGraph`.

```tokenvector
import tv.runtime.session

# Hằng số cấp module không import được giữa module; phải khai báo lại.
OperatorType_FusedLinear = 2
ActivationFunction_Relu = 1

g = ExecutionGraph(1, "demo", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w            # list[f32], dài n*k
g.initializers["B"] = b            # list[f32], dài n
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("W", [8, 4])
g.get_or_create_tensor("B", [8])
g.get_or_create_tensor("Y", [1, 8])

node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")            # rồi append("W"), append("B")
node.outputs.append("Y")
g.nodes.append(node)

# 1 = chạy ConstantFolding, QdqFold, OperatorFusion, DeadCodeElimination.
session = session_from_graph(g, 1)
y = session.run_single(x, "X", "Y")
session.close()
```

`run_single(input, input_name, output_name)` nhận buffer `list[f32]` phẳng. Truyền `""`
cho tên thì tự lấy input/output đầu tiên của graph.

## 4. Parse ONNX từ bytes
`parse_bytes(buf)` chạy được với buffer trong bộ nhớ; `parse_file(path)` **chưa** chạy
được vì `parser_read_bytes` là stub trả `""`.

```tokenvector
import tv.onnx.parser
import tv.onnx.translator
import tv.runtime.session

model = parse_bytes(buf)      # buf: nội dung file ONNX
g = translate(model)
session = session_from_graph(g, 1)
```

## 5. Lượng tử hóa Mô hình (PTQ)
```tokenvector
import tv.quantization.engine
import tv.quantization.types

p = calibrate_minmax(x, 1)
xi = _nd_i8(len(x))
quantize_int8_symmetric(x, xi, p.scale)
# INT4 / FP8 E4M3+E5M2 / FP16: xem tv/quantization/*.tkv
```

## 6. Micro-batching
Đường đồng bộ của batcher chạy inference ngay trên thread gọi; chưa có worker thread
nên micro-batching thực chất chưa gộp batch.

```tokenvector
import tv.server.batcher
# __init__ của record BỊ BỎ QUA lúc biên dịch, nên session phải được thêm vào
# pool bằng tay - đó là điều factory này làm.
b = new_dynamic_batcher_with_session(session, 32, 200, 10000)
y = b.enqueue(x)        # không bao giờ block; chạy inline nếu không có worker
b.dispose()
```

Embedded REST server chưa listen: `http_listen()` là stub trả `0`. CLI có sẵn nhưng
`argv()` là stub, nên luôn chỉ in usage và không nhận tham số.

## 7. Dùng package NuGet từ .NET

`TokenVector.Inference` kèm managed assembly (`lib/net10.0/TokenVector.Inference.dll`)
build từ nguồn `.tkv`, plus toàn bộ source `.tkv`. Consumer .NET 10 tham chiếu như mọi
package khác, NHƯNG hai native DLL phải đặt cạnh output — NuGet **không** tự copy:

```
your-app/
  your-app.csproj          # thêm <PackageReference Include="TokenVector.Inference" Version="1.0.1" />
  bin/Debug/net10.0/
    TokenVector.Inference.dll       # từ package (tự copy)
    tkv_simd_native.dll             # native AVX2 kernels — copy từ tools/runtime/ của package TokenVector
    System.Numerics.Vectors.dll     # copy từ tools/runtime/ của package TokenVector
```

> `TokenVector` (compiler package) ships hai DLL này trong `tools/runtime/`;
> copy chúng cạnh app (hoặc publish folder). Thiếu chúng, hot kernels rơi về scalar
> (hoặc lỗi) ở P/Invoke.

Ví dụ tối thiểu graph → session → run (cùng mã có trong `packaging/smoke-test/`):

```csharp
using System;
using System.Collections.Generic;

class Demo
{
    static int Main()
    {
        var g = new ExecutionGraph(1, "smoke", 0);
        g.inputs.Add("X");
        g.outputs.Add("Y");

        var w = TKVApp._nd_f32(32);
        for (int i = 0; i < 32; i++) w[i] = 0.5f;
        var b = TKVApp._nd_f32(8);
        for (int i = 0; i < 8; i++) b[i] = 0.25f;
        g.initializers["W"] = w;
        g.initializers["B"] = b;
        g.get_or_create_tensor("X", new List<long> { 1, 4 }, 0);
        g.get_or_create_tensor("W", new List<long> { 8, 4 }, 0);
        g.get_or_create_tensor("B", new List<long> { 8 }, 0);
        g.get_or_create_tensor("Y", new List<long> { 1, 8 }, 0);

        var node = TKVApp.new_node("gemm", 2);   // OperatorType_FusedLinear
        node.activation = 1;                     // ActivationFunction_Relu
        node.inputs.Add("X"); node.inputs.Add("W"); node.inputs.Add("B");
        node.outputs.Add("Y");
        g.nodes.Add(node);

        var session = TKVApp.session_from_graph(g, 1);
        var x = TKVApp._nd_f32(4);
        for (int i = 0; i < 4; i++) x[i] = 1.0f;
        var y = session.run_single(x, "X", "Y");
        Console.WriteLine("Y[0] = " + y[0] + "  (expect 2.25)");
        session.close();
        return 0;
    }
}
```

Ghi chú API (mức IL `.tkv`): class + helpers của `TKVApp` là kiểu global, `ExecutionGraph(1, name, 0)` nhận sentinel `_ok` đầu tiên, constants (`OperatorType_*`, `ActivationFunction_*`) là số nguyên thô, tensor là `List<float>` / shape là `List<long>`.

## 8. Toolchain
```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc <entry-point.tkv> app.exe main
Set-Content app.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet app.exe

powershell -File tools\run_cases.ps1 -Compiler $tkvc     # 22 pass / 4 fail (baseline; gap .NET Framework)
bash tools/gate_f32.sh                               # 135/135 — con số phải xanh
powershell -File tools/build_inference.ps1 -SkipBenchmarks     # build + test
```

Chỉ build entry point. Build một module lẻ khiến `tkvc` ghi lại các bản sao import
`tv.*.tkv`, và đã quan sát thấy 9 file bị ghi đầy byte NUL.
