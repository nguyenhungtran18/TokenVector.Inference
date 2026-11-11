# TokenVector.Inference User Guide

## 1. Overview
`TokenVector.Inference` is a pure TokenVector (`.tkv`) graph runtime: graph optimization,
INT8/INT4/FP8 quantization, and an in-process micro-batcher. Everything below is the
**verified** API. `docs/HANDOFF.md` lists the gaps, `docs/SIMD_NOTES.md` summarizes the three SIMD layers in use.

## 2. Build and run the quickstart
```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc tv/examples/quickstart.tkv benchmark/quickstart.exe main
Set-Content benchmark/quickstart.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet benchmark/quickstart.exe      # Y[0] = 2.25 / quickstart: OK
```

## 3. Constructing and running a graph
There is no `NDArray` and no `tv.inference.load_onnx()`. Tensors are flat `list[f32]`,
and a session comes from an `ExecutionGraph`.

```tokenvector
import tv.runtime.session

# Module-level constants cannot be imported across modules; redeclare them.
OperatorType_FusedLinear = 2
ActivationFunction_Relu = 1

g = ExecutionGraph(1, "demo", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w            # list[f32], length n*k
g.initializers["B"] = b            # list[f32], length n
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("W", [8, 4])
g.get_or_create_tensor("B", [8])
g.get_or_create_tensor("Y", [1, 8])

node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")            # then append("W"), append("B")
node.outputs.append("Y")
g.nodes.append(node)

# 1 = run ConstantFolding, QdqFold, OperatorFusion, DeadCodeElimination.
session = session_from_graph(g, 1)
y = session.run_single(x, "X", "Y")
session.close()
```

`run_single(input, input_name, output_name)` takes flat `list[f32]` buffers. Passing
`""` for the names falls back to the graph's first input/output.

## 4. ONNX parsing from bytes
`parse_bytes(buf)` works on an in-memory buffer; `parse_file(path)` does **not** yet,
because `parser_read_bytes` is a stub returning `""`.

```tokenvector
import tv.onnx.parser
import tv.onnx.translator
import tv.runtime.session

model = parse_bytes(buf)      # buf: the ONNX file contents
g = translate(model)
session = session_from_graph(g, 1)
```

## 5. Post-Training Quantization (PTQ)
```tokenvector
import tv.quantization.engine
import tv.quantization.types

p = calibrate_minmax(x, 1)
xi = _nd_i8(len(x))
quantize_int8_symmetric(x, xi, p.scale)
# INT4 / FP8 E4M3+E5M2 / FP16: see tv/quantization/*.tkv
```

## 6. Micro-batching
The batcher's synchronous path runs inference on the calling thread; there is no
worker thread yet, so micro-batching does not actually batch.

```tokenvector
import tv.server.batcher
# __init__ of a record is SKIPPED at compile time, so the session must be
# appended to pool by hand - that is what this factory does.
b = new_dynamic_batcher_with_session(session, 32, 200, 10000)
y = b.enqueue(x)        # never blocks; runs inline if no worker picks it up
b.dispose()
```

The embedded REST server does not listen: `http_listen()` is a stub returning `0`.
The CLI is present but `argv()` is a stub, so it always prints usage and takes no
arguments.

## 7. Using the NuGet package from .NET

`TokenVector.Inference` ships a managed assembly (`lib/net10.0/TokenVector.Inference.dll`)
built from the `.tkv` sources, plus full sources under `src/`. Consumers on .NET 10 depend
on it like any other package, but two native DLLs must sit next to the app output — NuGet
does **not** copy them automatically:

```
your-app/
  your-app.csproj          # add <PackageReference Include="TokenVector.Inference" Version="1.0.1" />
  bin/Debug/net10.0/
    TokenVector.Inference.dll       # from the package (auto-copied)
    tkv_simd_native.dll             # native AVX2 kernels — copy from TokenVector package tools/runtime/
    System.Numerics.Vectors.dll     # copy from TokenVector package tools/runtime/
```

> `TokenVector` (the compiler package) ships these under `tools/runtime/`; copy them
> next to your app output (or publish folder). Without them, the hot kernels fall back
> to the scalar path (or fail) at P/Invoke time.

Minimal graph → session → run (full example with the same code in `packaging/smoke-test/`):

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
        return 0;   // prints SMOKE OK when run via packaging/smoke-test/run.ps1
    }
}
```

API notes (.tkv IL level): classes and `TKVApp` helpers are global-namespace types,
`ExecutionGraph(1, name, 0)` takes an `_ok` sentinel first, constants (`OperatorType_*`,
`ActivationFunction_*`) are raw integers, tensors are `List<float>` / `List<long>` shapes.

## 8. Toolchain
```powershell
$tkvc = "D:\TokenVector\Release\dist\tkvc-tv.exe"
& $tkvc <entry-point.tkv> app.exe main
Set-Content app.runtimeconfig.json '{"runtimeOptions":{"tfm":"net10.0","framework":{"name":"Microsoft.NETCore.App","version":"10.0.0"}}}'
dotnet app.exe

powershell -File tools\run_cases.ps1 -Compiler $tkvc     # 22 pass / 4 fail (baseline; .NET Framework gap)
bash tools/gate_f32.sh                               # 135/135 — the number that must stay green
powershell -File tools/build_inference.ps1 -SkipBenchmarks     # build + test
```

Build entry points only. Building a single module makes `tkvc` rewrite the generated
`tv.*.tkv` import copies, which has been observed to leave nine of them filled with
NUL bytes.
