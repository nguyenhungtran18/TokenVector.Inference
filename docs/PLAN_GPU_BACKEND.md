# PLAN — GPU backend cho TokenVector.Inference (CUDA / Colab)

**Ngày:** 2026-11-12
**Bối cảnh:** engine CPU-only hiện nay; README table 3 provider `Available/Unavailable/Unavailable`.
Mục tiêu: tokenvector train GPT-2 124M trên GPU Colab (T4) với loss EXACT ≤ 1e-5,
step-time ≥ 2× so với CPU, và giữ nguyên parity/gate hiện tại.

---

## 0. TRẠNG THÁI THỰC TẾ (2026-11-12) — hook P0 ĐÃ CÀI NHƯNG KHÔNG BAO GIỜ ĐƯỢC GỌI

Bằng chứng từ Colab T4 (`BENCHMARKS.md` §9.25):

- `TkvCUDA_Available() = 1` → dlopen `libcublas.so.12` + `libcudart.so.12` thành công.
- Nhưng `TkvCUDA_SgemmF32(a,b,o,n)` **chưa từng thực thi GEMM trên GPU**, vì nó được
  gọi từ `SgemmF32(a,b,o,T)` — hàm này chỉ là **kernel tile T×T** bên trong
  `SgemmRect` / `SgemmFull` / `SgemmPacked`, với `T` là tile size (< 512). Điều kiện
  `n >= 512` trong hook luôn sai.
- Ngoài ra, kể cả nếu điều kiện đúng, mỗi lần gọi vẫn `cudaMalloc` 3 buffer + 2 H2D
  + 1 D2H → nằm trong vòng lặp tile thì chắc chắn chậm hơn CPU.

**Hệ quả cho kế hoạch:** P0 (hook `SgemmF32`) coi như *chưa đạt*. Muốn GPU có tác dụng
phải làm đúng P1/P2: hook ở tầng GEMM **đầy đủ** (có `m,n,k`, `transA/transB`) và
**giữ weight/activation trên device** (device-resident), chỉ copy phần thay đổi.
Cần đo phase breakdown trên Colab trước khi quyết định hook ở đâu.

**Số đo mới nhất (2026-11-12, `BENCHMARKS.md` §9.26):** `.tkv` 26 230 ms vs torch CPU
10 243.7 ms (**2.56×** chậm hơn) vs torch CUDA 271.7 ms (**96.5×** chậm hơn).
GPU chỉ được "chạm" ở mức CUDA context (117 → 243 MiB, util 0–6%) — xác nhận hook
không bao giờ chạy GEMM. Vì vậy P1 bắt buộc phải hook `SgemmRect`/`SgemmFull`/
`SgemmPacked` (có `m,n,k`) và bỏ test `n >= 512`.

---

## 1. Hiện trạng cụ thể

| Thành phần | Trạng thái |
|---|---|
| Engine core | `.tkv` source, biên dịch bởi `tkvc-tv` (Windows) sang .NET IL |
| Native kernels | `libsimd_native.dll` (AVX2/FMA, pthread pool) |
| Threading | `pthread` (kernels.c); Windows fallback qua `CreateWaitableTimer` |
| Acceleration seam | `runtime/hal.tkv` → `HalDispatch`, `DeviceCaps`, `SimtCpuProvider`, `FfiAccelerator` |
| Backend selection | `runtime/backend.tkv` → `backend_status()` → thuộc `available/simulated/unavailable` |
| Doc GPU hiện tại | COMPETITOR_COMPARISON.md §"GPU/NPU compute" — ghi rõ: chỉ stub, không có
  thread block / shared memory / MMA / kernel-launch API; `HalDispatch` là seam đăng ký khi có backend |

**Ràng buộc cứng của `.tkv`:** ngôn ngữ không có abstraction GPU (không có grid/block,
kernel launch, tensor-core MMA intrinsic, async dispatch). Bất kỳ kernel nào "GPU"
cũng phải là **native library** (như `libsimd_native.dll` hiện nay) gọi qua FFI.

---

## 2. Lựa chọn kiến trúc

### Phương án A — Route qua vendor SDK (cuBLAS + FFI); **khuyến nghị làm trước**

- Viết native `tkv_cuda_native.dll/.so` (C; thin layer) → load `libcublas` qua
  `dlopen`/`LoadLibrary` ngay trong native lib (giống pattern `libsimd_native.dll`).
- Mọi GEMM (fwd/bwd) → `cublasSgemm` (row-major, transa/transb tuỳ bước).
- Softmax, LayerNorm, GELU, AdamW → cuBLAS/cusolver không có; hoặc:
  - LayerNorm + GELU → kernel CUDA ngắn (<200 dòng) trong `tkv_cuda_native.cu`
  - AdamW → giữ CPU (memory-bound, nhỏ, < 1ms/step) hoặc chuyển qua cuBLAS
  - Softmax → `cublasSgemv` + `CUBLAS_COMPENDD`? không tương thích trực tiếp;
    viết kernel softmax CUDA ~50 dòng
- `FfiAccelerator` đăng ký `"cuda"` provider khi `libcuda` available; CPU fallback
  giữ nguyên nguyên bản (flag chọn `-DUSE_CUDA=0` không dùng GPU).

**Ưu điểm:** tốc độ phát triển nhanh (2–4 tuần cho fwd/bwd GEMM + smoke parity);
tái sử dụng pattern FFI đã có; không phụ thuộc compiler team.

**Nhược điểm:** loss EXACT không còn đúng theo bitwise (cuBLAS ≠ AVX2 tích luỹ theo
thứ tự khác nhau) — chấp nhận `max_abs_diff ≤ 1e-5` trên 124M fixture;
sinh/phiên bản vendor lock-in (NVIDIA only).

**Giai đoạn:**
- **P0 (spike, 2–4 ngày).** Build native lib tối thiểu: GEMM `Sgemm`, một elementwise
  kernel, FFI gọi từ .tkv, lấy ra được kết quả đúng trên test 4×4.
  Điều kiện đạt = loss diff `≤ 1e-5` trên 1 bước GPT-2 124M (CPU vs GPU chỉ GEMM).
- **P1 (stub + wiring, 4–7 ngày).** `FfiAccelerator` → provider `"cuda"`; `HalDispatch`
  chọn `"cuda"` khi caps `CUDA_MP`; Lo từ CPU kernel thay trực tiếp bằng gọi native.
  Điều kiện đạt: gate_f32 chạy green (135/135) với `-DUSE_CUDA=1`.
- **P2 (bwd kernels, 1 tuần).** GEMM transa cho dW, dx; softmax+gelu fwd/bwd; parity vs PyTorch
  `[0,1,2,...]`. Điều kiện đạt: GPT-2 124M first-step loss diff `≤ 1e-5`, full-step 2–4× CPU.
- **P3 (RoCM, 3–5 ngày).** Lặp lại trên `rocblas` + `hiprtc`; target AMD APU/MI250.
  Không cần nếu Colab là NVIDIA.
- **P4 (multi-GPU, 1–2 tuần).** Data-parallel: mỗi GPU một replica, trung bình gradient.
  Điều kiện đạt: 1.8× vs 1 GPU trên 2×T4.

### Phương án B — Native CUDA kernels từ đầu (đúng kiến trúc)

Viết `.cu` cho GEMM, attention, norm — tự triển khai, không cuBLAS. **KHông khuyến nghị
làm trước**: 8–12 tuần, rủi ro lệch optimizer, phục thuộc nvcc version per GPU.

**Chỉ làm khi:** (a) Phương án A đã chứng minh GEMM qua FFI thành công nhưng Softmax/
LayerNorm trên `gkss` không đủ nhanh, hoặc (b) cần Chương trình không dùng vendor libs
(embedded/offline).

---

## 3. Phụ thuộc bắt buộc

| Phụ thuộc | Phiên bản | Ghi chú |
|---|---|---|
| CUDA Toolkit | 12.x (Colab `nvidia-cuda-toolkit` apt / pip `nvidia-cuda-runtime-cu12`) | bắt buộc nếu build từ `.cu`; không cần nếu FFI vào cuBLAS có sẵn |
| `libcuda.so` | tương thích với driver | `nvidia-smi` hiển thị ok |
| `libcublas.so.12` | apt `libcublas12` | hoặc pip `nvidia-cuda-runtime-cu12` |
| NVIDIA GPU | Compute Capability ≥ 7.0 (Colab T4, sm_75) | CC8.9/9.0 OK cho APUs |
| Linux/WSL2 (cho `.so`) | Ubuntu ≥ 20.04 | Windows sau (MSBuild vc++ + CUDA) |

Nếu chỉ Windows thì chọn `tkv_simd_native.dll` + `libcublas.dll` (Windows CUDA SDK).

---

## 4. Sơ đồ tích hợp (proposed)

```
.tkv fwd/bwd 
  └─► runtime/engine.tkv  (gọi native FFI)
        └─► tkv_cuda_native.dll/.so  (dlopen("libcublas.so.12"))
              └─► cublasSgemm / custom .cu kernels / cuLaunchKernel
        └─► tkv_simd_native.dll/.so  (fallback CPU AVX2/FMA)
```

**Point of no return:** nếu `cublas` fail lần đầu trong `Sgemm`, chọn automatic fallback
sang `AVX2` path (tương tự `TKV_GEMM_BLAS=0` hiện tại). Không crash cứng GPU missing.

---

## 5. Ảnh hưởng đo đạc

Đo chính (mỗi phase cuối):

| Metric | Hiện tại (CPU) | Sau P1 | Sau P2 | Mục tiêu |
|---|---:|---:|---:|---:|
| fwd+bwd seq1024 (min) | 3.4 s | ~2.0 s | ~1.5–2.0 s | < 2.0 s |
| opt.step (AdamW) | 0.45 s | 0.45 s (CPU) | 0.1 s | < 0.2 s |
| Loss EXACT | 1e-7 | < 1e-6 | < 1e-6 | < 1e-5 |
| Gate 135/135 | PASS | PASS | PASS | PASS |
| Trainer 15/15 | PASS | PASS | PASS | PASS |

*(Số liệu chưa đo, kỳ vọng dựa trên tỷ lệ cuBLAS/AVX2 GEMM ≈ 2–4× trên T4.)*

---

## 6. Rủi ro & giảm thiểu

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| Loss không còn EXACT sau khi thay CPU kernels | Chắc chắn (khác accumulation order) | tolerances, roll-forward; so khớp vs PyTorch thay vì bit-exact |
| Colab T4 bị throttle (thực tế 50% clocks trên free tier) | Cao | Đo thực tế; đừng quote peak |
| cuBLAS version mismatch (Colab cập nhật toolkit) | Trung bình | dlopen theo tên `libcublas.so.12`; fallback CPU nếu không tìm thấy |
| Multi-threading CPU contention with host BLAS | Thấp | `OMP_NUM_THREADS` env controllable; giữ single-thread host BLAS khi GPU active |
| `nuget`/`.NET` không thể load native lib (Windows < 10) | Thấp | `LoadLibraryEx(LOAD_WITH_ALTERED_SEARCH_PATH)` (đã có trong `tkv_simd_native.c`) |
| `%` loss EXACT budget bị đốt hết khi chạy 124M | Trung bình | log per-step diff; skip nếu < 1e-5 |

---

## 7. Thứ tự thực thi

1. **Spike P0** (2–4 ngày, Linux + WSL2 PXE): tải `nvidia-cuda-toolkit`, viết `tkv_cuda_native.c`
   gọi `cublasSgemm`, verify `dlopen` + `Sgemm` gọi được từ `.tkv`.
2. **P1 + wiring** (3–5 ngày): FFI provider `"cuda"` trong `backend.tkv`/`hal.tkv`, smoke test
   trên `tools/gate_f32.sh`.
3. **P2** (1–2 tuần): fwd/bwd GPT-2 124M native GPU path; loss parity; doc performance.
4. **P3** (3–5 ngày, optional): RoCM GEMM.
5. **P4** (1–2 tuần, optional): data-parallel DP trên 2× T4.

---

## 8. Các câu hỏi phải giải quyết TRƯỚC khi bắt đầu

1. Windows và Linux build riêng hay chung `.so`/`.dll`? — có vì `libsimd_native.dll` đã có cả hai.
2. ai build: local user (vnd) hay CI? — local chạy `dotnet build` (tương tự compiler self-host).
3. Colab free hay Colab Pro? — Free đủ (15 GB VRAM T4, 68 CUDA cores); Colab Pro+ nếu cần H100.
4. `tkv_cuda_native.dll` committed hay build on-demand? — Build on-demand; không commit binary CUDA.

---

## 9. Liệt kê files dự kiến chạm

- `runtime/backend.tkv` — thêm provider `"cuda"`
- `runtime/hal.tkv` — đăng ký `CudaSgemmCaps` trong `HalDispatch`
- `simd_native/` (mới) — `libcuda_native.c` / `.cu`
- `tools/gate_f32.sh` — flag `-DUSE_CUDA=1`
- `benchmark/_bench_matrix.ps1` — thêm `benchmark/_bench_matrix_cuda.sh`
- `docs/PLAN_GPU_BACKEND.md` — file này
