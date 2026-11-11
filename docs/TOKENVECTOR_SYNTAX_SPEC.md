# TokenVector Language Syntax Specification: Inference Module

This file documents the syntax that `TokenVector.Inference` actually uses. The
previous version of this file showed an API that does not exist
(`tv.inference.load_onnx`, `tensor.zeros`, `NDArray`); it had been reduced to
whitespace by an accidental overwrite.

## 1. Imports
```tokenvector
import tv.runtime.session
import tv.onnx.parser
import tv.onnx.translator
```

**Two rules that bite everyone:**

- **Module-level constants are NOT importable.** Only functions and records cross
  module boundaries. Redeclare `OperatorType_*` and `ActivationFunction_*` in every
  file that uses them; take the values from `tv/runtime/session.tkv` and
  `tv/optimization/kernels.tkv`.
- **A record's `__init__` is skipped at compile time.** Construct records positionally
  in field order, one argument per non-container field, and finish initialization in a
  factory that assigns the container fields by hand (see
  `new_embedded_server` in `tv/server/embedded_server.tkv`).

## 2. Tensor buffers
There is no `NDArray` and no `tensor.zeros`. A tensor is a flat `list[f32]`.

```tokenvector
x = _nd_f32(4)        # 4 f32 values
for i in range(4):
    x[i] = 1.0
```

## 3. Graph construction
```tokenvector
g = ExecutionGraph(1, "demo", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("Y", [1, 8])
```

## 4. Nodes
```tokenvector
node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")
node.outputs.append("Y")
g.nodes.append(node)
```

## 5. Session and forward pass
```tokenvector
session = session_from_graph(g, 1)   # 1 = run the optimization passes
y = session.run_single(x, "X", "Y")
session.close()
```

`session_from_graph(graph, 0)` skips optimization. Both are correct; `(1)` is the
default you want.

## 6. Type annotation syntax
```tokenvector
def scale(a: "list[f32]", n: "i64") -> "f32":
```

Doubles are the only string delimiters in expressions. `len()` returns `TkvInt`, so
cast it before using it as a `while` bound or you get an infinite loop.

Runnable reference: `tv/examples/quickstart.tkv`.
