# Đặc tả cú pháp ngôn ngữ TokenVector: Module Inference

Tài liệu này mô tả cú pháp mà `TokenVector.Inference` thực sự dùng. Bản trước
trình bày một API không tồn tại (`tv.inference.load_onnx`, `tensor.zeros`, `NDArray`)
và đã bị ghi đè thành toàn khoảng trắng do nhầm lẫn.

## 1. Import
```tokenvector
import tv.runtime.session
import tv.onnx.parser
import tv.onnx.translator
```

**Hai luật gây khó nhất:**

- **Hằng số cấp module KHÔNG import được.** Chỉ hàm và record mới đi qua được ranh
  giới module. Phải khai báo lại `OperatorType_*` và `ActivationFunction_*` trong mọi
  file dùng tới; lấy giá trị từ `tv/runtime/session.tkv` và
  `tv/optimization/kernels.tkv`.
- **`__init__` của record bị BỎ QUA lúc biên dịch.** Dựng record bằng tham số
  positional theo thứ tự field, mỗi tham số cho một field không phải container, rồi
  hoàn tất khởi tạo trong factory bằng cách gán tay các field container (xem
  `new_embedded_server` trong `tv/server/embedded_server.tkv`).

## 2. Buffer tensor
Không có `NDArray` và không có `tensor.zeros`. Một tensor là `list[f32]` phẳng.

```tokenvector
x = _nd_f32(4)        # 4 gia tri f32
for i in range(4):
    x[i] = 1.0
```

## 3. Dựng graph
```tokenvector
g = ExecutionGraph(1, "demo", 0)
g.inputs.append("X")
g.outputs.append("Y")
g.initializers["W"] = w
g.get_or_create_tensor("X", [1, 4])
g.get_or_create_tensor("Y", [1, 8])
```

## 4. Node
```tokenvector
node = new_node("gemm", OperatorType_FusedLinear)
node.activation = ActivationFunction_Relu
node.inputs.append("X")
node.outputs.append("Y")
g.nodes.append(node)
```

## 5. Session và forward pass
```tokenvector
session = session_from_graph(g, 1)   # 1 = chay cac pass toi uu
y = session.run_single(x, "X", "Y")
session.close()
```

`session_from_graph(graph, 0)` thì bỏ qua tối ưu hóa. Cả hai đều đúng; `(1)` là lựa
chọn mặc định nên dùng.

## 6. Cú pháp chú thích kiểu
```tokenvector
def scale(a: "list[f32]", n: "i64") -> "f32":
```

Trong biểu thức, chuỗi chỉ nhận dấu nháy kép. `len()` trả về `TkvInt`, nên phải ép kiểu
trước khi dùng làm biên của `while`, nếu không sẽ lặp vô hạn.

Ví dụ chạy được: `tv/examples/quickstart.tkv`.
