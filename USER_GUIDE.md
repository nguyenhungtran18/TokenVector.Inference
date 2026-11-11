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

## 7. Toolchain
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
