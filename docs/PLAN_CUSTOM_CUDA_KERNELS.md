# KẾ HOẠCH — Custom CUDA kernels cho `.tkv` (Nguyen Hung Tran, 2026-11-17/18)

> **Nguồn chuẩn khi cập nhật:** mỗi mốc con số đều kèm notebook session + commit. Không claim trước khi đo.
> Loss native mode 0 giữ EXACT 25.407877; GPU bridge waiver numerics `loss_val` ~17.111x.

## 0. Bối cảnh & hiện thực hoá (ground truth 2026-11-17/18, Colab T4)

| Notebook | GPU bridge TONG (x3) | bwd | LMHEAD_BWD | LN_BWD / LN_FWD |
|---|---|---|---|---|
| (19) | 7 892 | 5 676 | 5 748 | 938 / 719 |
| (28) | **3 849** | **1 582** | **1 456** | 406 / ~1.9/call* |

*LN_FWD gồm 1 lần CUDA lazy-load ~465 ms chi phí đầu phiên (đã xử lý bằng warm-up, Release `1f4efad`; steady ~1.9 ms/call).

**Khoảng cách tới torch CUDA full step (~0.27 s, đo cùng VM phiên 2026-11-15/16/17)**: ~14×.
Không thể đóng bằng bridge ATen. Phải: (a) giảm op-time ATen bằng fused kernel,
và/hoặc (b) viết lại đường training step bằng CUDA C++ thuần (dự án lớn).

## 1. Phân bố op-time GPU (28) để chọn mục tiêu

- **LMHEAD_BWD 1 456 ms (54%)** — pha 1 softmax+scatter (fp32 read/write ~410 MB), pha 2 `mm` cuBLAS, pha 3 `addmm_out`. Phần lớn là memory-bandwidth + cuBLAS sgemm (đã tối ưu, chạm được có giới hạn).
- **LN_FWD/BWD ~500 + ~700 ms** — ATen elementwise/reduce kernels (~10–14 kernel launches/call/path).
- **ATTN (bridge) không đếm counter** — quên gọi thẳng TkvTorch* đường ATen; chi phí nhỏ trên GPU (phải xác nhận bằng CUDA profile).
- **`opt` ~430 ms** AdamW — đang là native path (throng phần tự viết trong simd_native); có thể cuốn vào kernel riêng.

## 2. Probes cần có (đã thêm vào notebook, chờ run)

1. `nvcc --version` — toolchain compile .cu trong Colab (nếu thiếu: fallback `.cpp` + ATen, hỏng ơn).
2. `nvidia-smi --query-gpu=compute_cap` — kiến trúc (T4 = sm_75, fp32 peak 8.1 TFLOPS).
3. CUDA kernels (backward) từ torch.profiler → biết SDPA path thực (math / mem_efficient / flash fp32) + softmax kernel nào đang chạy.

## 3. Thứ tự thử nghiệm (ROI ÷ độ khó)

**P0 — Fused LayerNorm fwd+bwd 1 kernel** (bit-closeness check vs ATen):
- Target: ≤0.2 ms/call (hiện ~1.9 fwd / 5.4 bwd). Tổng ~75 call mỗi loại mỗi run → ~400–500 ms giảm/step.
- Rủi ro: thấp (công thức chuẩn Welford; hiếm khi sai số học).
- Cấu trúc: pointer cast ai đọc/ghi; Launch trong .so shim riêng `tkv_ln_cuda.dll` hoặc nhúng thẳng vào bridge cu thể `y[i] = (x[i]-mu)*rstd*g[i]+b[i]`.
- Cổng: tính thường accuracy ~1e-6 vs ATen; loss_val phải giữ ~17.111x.

**P1 — Fused softmax-grad (LM head) `dlogits = (softmax(logits) − onehot(id))·scale`** — chỉ nếu profile cho thấy fp32 softmax là nút cổ chai thật (>200 ms). Softmax fp32 có thể viết 1-pass/2-pass blocked; ATen fp32 softmax đã khá tốt — chỉ làm khi đo thấy.

**P2 — Flash fp32 SDPA** — chỉ khi CUDA profile cho thấy SDPA đang math/naive chậm (>1 s/step). fp32 flash tùy chỉnh là task lớn (online softmax, tiling; ~500 dòng) — cân nhắc kỹ trước khi làm.

**P3 — Fused AdamW epilogue** (lr, b1, b2, eps, wd, bc1, bc2, coef đọc từ process env/.cfg) — task nhỏ nếu cần; hiện `opt` không phải nút cổ chai lớn.

**Khả năng cứu lớn nhất**: từng lớp tự nhiên của P0+P1 là **~400–700 ms/step**; cộng nhau vẫn far ~2.5–3.0 s. **KHÔNG thể chạm 0.27 s** mà không viết lại training step bằng CUDA C++ toàn bộ (10k+ LOC, dự án ≥ 2–4 tuần, rủi ro cao mất baseline native EXACT).

## 4. Thiết kế kỹ thuật

- File `Release/tools/cuda_native/tkv_p0_ln.cu` + `tkv_p0_ln.cpp` exports:
  `extern "C" __declspec(dllexport) int TkvLNCudaFwd(const float* x, const float* g, const float* b, float* y, int64_t n, int64_t h, double eps);`
  (và `Bwd` tương tự trả về dx/dg/db).
- Build cell mới `build-cuda` trong notebook (`nvcc -O3 -arch=sm_75 -shared -fPIC ...`).
- Attach vào shim `tkv_simd_native.c`: sau khi load bridge thì load thêm `tkv_ln_cuda.dll`; file gọi `TkvTorchLayerNormFwd` → gọi thí nghiệm CUDA trước (env `TKV_LN_CUDA=1`), rớt về ATen khi -9.
- Tests: `tests/cuda_ln_fwd_bwd.tkv` với tolerance 1e-6; loss GPU giữ nguyên ~17.11; native EXACT giữ nguyên.
- Gate nghiêm: `.tkv` `gpt2_layernorm` gọi `layernorm_fwd` shim → shim sẽ tự chọn CUDA nếu `TKV_LN_CUDA=1` và rc==0; rc!=0 → ATen → rc!=0 → f64 native — chuỗi fallback an toàn, không regression.

## 5. Các bước triển khai (khi bắt đầu)

1. (28/30) chạy probes → ghi nhận nvcc + arch + SDPA path.
2. Viết P0 kernel + build cell; test microbench độc lập trong cell (N=1024, H=768, fp32).
3. Integrate vào shim (env flag), chạy notebook bình thường; so sánh loss/time.
4. Nếu nhanh: promote; ghi `BENCHMARKS.md`; cập nhật plan này.
5. Lặp P1/P2 theo dữ kiện profile.

## 6. Các lưu ý / rủi ro

- **Numerics**: P0 dùng Welford từng-thread + final reduce — kết quả lệch ~1e-7 vs ATen — loss_val ~17.111x như hiện tại.
- **sm_75 fp32** không có tensor core fp32 (TF32 only, cấm bật TF32 vì cần match host) — kernel phải đứng vững trên fp32 thông thường.
- **Lazy CUDA module**: kernel function CUDA module nạp lười tại lần gọi đầu — warm-up ở constructor bridge/shim tương tự workaround `1f4efad`.
- **Anti-pattern**: đừng viết kernel rồi cũng `dlopen` qua ctypes rồi gọi trong `.tkv` cho mọi op — quá rời rạc, khó bench. Integrate vào shim như mọi native ops khác.
