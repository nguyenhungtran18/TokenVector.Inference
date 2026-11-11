# Smoke test — `TokenVector.Inference.dll` từ .NET

Kiểm tra tối thiểu: assembly `lib/net10.0/TokenVector.Inference.dll` gọi được từ
ứng dụng .NET 10 (graph → session → forward), kỳ vọng `Y[0] = 2.25` (ReLU(X @ W^T + B) với X=[1,4]=1, W=0.5, B=0.25).

## Chạy

```powershell
# 1) build DLL (một lần):
packaging\pack.ps1
# 2) chạy smoke:
packaging\smoke-test\run.ps1          # mặc định runtime = D:\TokenVector\Release\runtime
# hoặc:
packaging\smoke-test\run.ps1 -RuntimeNativeDir <path-of-tkv_simd_native.dll>
```

## API đã dùng (mức IL, không namespace)

| Mã | Việc |
|---|---|
| `new ExecutionGraph(1, "smoke", 0)` | tạo execution graph |
| `g.inputs.Add("X")` / `g.outputs.Add("Y")` | định nghĩa input/output |
| `TKVApp._nd_f32(n)` | tạo `List<float>` mới |
| `g.initializers["W"] = w` | nạp weights |
| `g.get_or_create_tensor("X", new List<long>{1,4}, 0)` | shape/dtype (i64) |
| `TKVApp.new_node("gemm", 2)` | node `OperatorType_FusedLinear` |
| `node.activation = 1` | `ActivationFunction_Relu` |
| `TKVApp.session_from_graph(g, 1)` | build session (bật optimization passes) |
| `session.run_single(x, "X", "Y")` | forward một sample |
| `session.close()` | giải phóng arena |

Constants (`OperatorType_*`, `ActivationFunction_*`) không import được giữa các
module `.tkv` — truyền số nguyên trực tiếp.

## Yêu cầu

- `tkv_simd_native.dll` cạnh exe (hot kernels P/Invoke; nguồn: `D:\TokenVector\Release\runtime\`).
- `TokenVector.Inference.dll` cạnh reference (output build của pack.ps1).
