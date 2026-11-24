# Đường GPU IL-only — quyết định và lộ trình (2026-11-24)

## Quyết định của người dùng
- **Bỏ hoàn toàn artifact C++**: không sinh `.torch.cpp`, không sinh `.cuda.cu`, không viết tay `.cpp/.c` ở lớp bridge lẫn kernel.
- Mọi thứ (tensor GPU, allocator, kernel, autograd, binding) phải nằm trong **TokenVector/IL**; IL gọi trực tiếp CUDA/cuBLAS driver API qua P/Invoke.
- **Port từng kernel một**, mỗi kernel so sánh số liệu với ATen trước khi thay thế.

## Tình trạng hiện tại (khảo sát 2026-11-24)
- Compiler chỉ có 2 đường: `.tkv → IL → ilasm → .NET PE`, hoặc `.tkv → C/C++/CUDA source → gcc/nvcc → .so`.
- Không có backend sinh PTX/cubin, không có interpreter/VM `.tkv`, không có binding Python↔.NET có sẵn.
- Autograd GPU hiện **chỉ có trong C++ sinh** (`torch_feature.tkv`: backward/grad/put_grad/zero_grad/fp16 sync).
  Autograd thuần TokenVector chỉ có trên CPU (`tv/training/{tensor,autograd,tape,ops}.tkv`).
- Điểm tựa đã chứng minh: chương trình `.tkv` biên dịch trên Windows **chạy được trên Linux bằng mono** (`docs/linux-portability-spike.md`, 2026-10-17).
  Compiler tự thân thì mono chạy không được (IL `stloc.s 63`) — chỉ cần chạy PE sản phẩm, không cần compiler trên Colab.
- Mốc hiện tại cần bảo vệ: `TokenVector@9ccc17b`, `TokenVector.CudaBackend@92a2349`,
  bridge sinh C++ cho `2.36×` vs PyTorch eager, fingerprint `3.8906` (fp16/native), KHÔNG so sánh trực tiếp với HF fp32 `9.0058`.

## Lộ trình kernel-đầu-tiên (milestone)
1. **M0 — hạ tầng IL → CUDA**: khai báo P/Invoke CUDA driver/runtime + cuBLAS trong `.tkv`; biên dịch PE trên Windows; chạy dưới mono trên Colab T4.
2. **M1 — SAXPY**: `.tkv` cấp phát device buffer, copy H2D/D2H, gọi `cublasSaxpy`, so sánh kết quả với torch `axpy`; không có file C/C++ nào.
3. **M2 — emitter kernel**: thêm cơ chế để TokenVector sinh mã thiết bị (không phải C++) cho op elementwise; kernel đầu: `gelu` hoặc `add`; so sánh maxabs với ATen.
4. **M3 — thay dần forward**: matmul → layer_norm → causal SDPA → log_softmax/gather, mỗi kernel giữ sai số so với ATen và theo dõi fingerprint loss.
5. **M4 — autograd & optimizer**: tape/grad trên device, put grads, AdamW, fp16 master sync — thay thế `torch_feature.tkv`.
6. **M5 — binding & hiệu năng**: thay ctypes bằng giao tiếp Python↔.NET (pythonnet hoặc shared memory), đo lại tốc độ; chỉ claim speedup khi có số đo mới.

## Nguyên tắc xác nhận
- Mọi kết quả kèm HEAD + timestamp + tên kernel + phép so sánh cụ thể (maxabs / bitwise).
- Không xóa đường bridge sinh C++ cho đến khi M4 xác nhận xong; đường cũ là baseline an toàn.
- Không dùng C/C++ làm cầu trung gian ở bất kỳ milestone nào.
