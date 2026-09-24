# Ghi chú Phát hành TokenVector.Inference - Phiên bản 1.0.0 (Tiếng Việt)

## Tổng quan Phát hành
Phiên bản chính thức đầu tiên của **TokenVector.Inference 1.0.0** mang đến nền tảng phục vụ suy luận mô hình học sâu, tối ưu hóa đồ thị và lượng tử hóa siêu nhẹ — **100% TokenVector (`.tkv`)**.

## Tính năng Nổi bật
- **Động cơ Zero-GC Hot Path:** Cấp phát bộ nhớ arena một lần theo khoảng vòng đời Tensor.
- **Bộ Parser ONNX Hiệu năng cao:** Reader protobuf thuần `.tkv`, zero-copy.
- **Optimization Passes:** Gộp và tối ưu hóa toán tử (`ConstantFoldingPass`, `DeadCodeEliminationPass`, `OperatorFusionPass`).
- **Lượng tử hóa INT8 & FP8:** PTQ tĩnh, lượng tử hóa động, INT4 grouped, định dạng FP8 `E4M3`/`E5M2`.
- **Embedded Server & Micro-Batching:** Phục vụ nhúng độ trễ thấp, API OpenAI-compatible, SSE, auth, metrics.
- **Module stdlib TokenVector:** `stdlib/tv/inference` cho ngôn ngữ `.tkv`, build/test bằng `tkvc`.
