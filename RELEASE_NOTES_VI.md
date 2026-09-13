# Ghi chú Phát hành TokenVector.Inference - Phiên bản 1.0.0 (Tiếng Việt)

## Tổng quan Phát hành
Phiên bản chính thức đầu tiên của **TokenVector.Inference 1.0.0** mang đến nền tảng phục vụ suy luận mô hình học sâu, tối ưu hóa đồ thị và lượng tử hóa siêu nhẹ thuần 100% C# Native.

## Tính năng Nổi bật
- **Động cơ Zero-GC Hot Path:** Cấp phát bộ nhớ native arena một lần duy nhất theo khoảng vòng đời của Tensor.
- **Bộ Parser ONNX Hiệu năng cao:** Giải mã protobuf trực tiếp không sinh rác (`FastProtobufReader`).
- **Optimization Passes:** Gộp và tối ưu hóa toán tử (`ConstantFoldingPass`, `DeadCodeEliminationPass`, `OperatorFusionPass`).
- **Lượng tử hóa INT8 & FP8:** Hỗ trợ PTQ tĩnh, lượng tử hóa động, kernel SIMD AVX2/AVX-512 VNNI, và định dạng FP8 `E4M3`/`E5M2`.
- **Embedded Server & Micro-Batching:** Phục vụ nhúng với độ trễ phản hồi $< 1\text{ ms}$.
- **Cầu nối Trình biên dịch TokenVector:** Tích hợp trực tiếp module `stdlib/tv/inference` cho ngôn ngữ `.tv`.
