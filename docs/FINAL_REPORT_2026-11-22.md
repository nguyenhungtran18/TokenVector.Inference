# Báo cáo cuối — TokenVector thay thế native C/C++ kernel (2026-11-22)

## 1. Mục tiêu & kết luận

Hai mục tiêu ban đầu đều có bằng chứng Colab:

| Mục tiêu | Kết quả measured | Bằng chứng |
|---|---|---|
| GPU train nhanh hơn PyTorch eager | **2.24×** (TKV 224.2 ms vs torch 501.7 ms, GPT2-medium 355M, T=1024, T4, 400 steps) | `colab_FINAL_BENCH_GPU_VS_TORCH_2026-11-22.ipynb` |
| Native EXACT path không bị phá | `loss_val = 25.45386` giữ nguyên sau **23 kernel** thay thế | `colab_NATIVE_ALL_TKV_INTEGRATE_LOSS_2026-11-22.ipynb` |
| Kernel native viết bằng TokenVector thuần | **23/23 export**, **0 C injection** | 23 notebook bit-exact/tolerance riêng lẻ |

CPU benchmark: native all-TKV `29.03 ms` vs PyTorch CPU eager `10318.9 ms` → `355.41×`.
**Caveat bắt buộc**: đây không phải so sánh apple-to-apple (model/RNG/numerics khác nhau, xem §5).

## 2. Danh sách kernel thuần TokenVector

**Nhóm A — 14 kernel CPU bit-exact** (C INJECTED=False, diff=0):
`LinearDwFused`, `BiasAddRows`, `BiasGrad`, `MergeHbHead`, `LayernormFwd`,
`LayernormBwdFused`, `F32ListAdd`, `F32ListCopy`, `LinearDxFused`,
`SoftmaxCausalBwd`, `SoftmaxCausalFwd`, `GeluErfFwd`, `GeluErfBwd`, `EmbedBwdFused`.

**Nhóm B — 7 kernel FlashAttention** (waiver s0.125 có sẵn trong `tkv_flash_attn.c`):
`FlashAttentionFwdLse`, `FlashAttentionFwdFused`, `FlashAttentionBwdFused`,
`FlashAttentionBwdFusedLse`, `FlashAttentionFwdLseMulti`,
`FlashAttentionBwdFusedLseMulti`, `FlashAttentionBwdMergeMulti`.
Độ lệch đo được: `maxabs 2.38e-07 .. 4.77e-07`; một số output bitwise (`ctxo`, `ctxT`).

**Nhóm C — 2 stub**:
`TkvCudaAttnFwd`, `TkvCudaAttnBwd` — giữ hành vi mặc định (trả 0 → fallback flash).

Tất cả nằm trong 1 file `.tkv` cho flash (2 helper + 7 export; compiler hỗ trợ nhiều kernel/gọi kernel chéo).

## 3. Mở rộng compiler (đều được user cho phép, self-host `vong.exe` IL_IDENTICAL)

`d_sqrt`, `d_exp`, `tkv_notfinite`, `tkv_rint`, `tkv_ldexp`, `tkv_fmax`, `tkv_fabs`,
`log(→logf`, `f32buf/i32buf` (malloc/free tự động), word-boundary cho rule inference.

### Bug compiler đã sửa trong quá trình
- `d_sqrt` bị `sqrtf` ăn theo → token trung gian `__TKVSQRT__`.
- C scope: biến khai báo trong vòng lặp đầu dùng lại ở vòng sau.
- `dctx` stride `h` bị dùng nhầm `3*h` → segfault, Colab reset runtime.
- Inference int: CUDA `'r * '` khớp `lr * mi`; CPU `'n['` khớp `lse_in[` → `f_x` thành `int`.
- `adamw.cuda.cu`: `int pi` (từ commit `94e2dfb`) làm loss nổ `ln(V)=10.82`; đã sửa word-boundary.

## 4. Phương pháp verify

- Mỗi kernel: build native reference `.so` và TKV-generated `.so` **cùng flags** `-O2 -mavx2 -mfma`.
- So sánh bitwise `uint32` cho nhóm A; `maxabs` cho nhóm B (theo waiver).
- Tích hợp: ghi `.cpu.c` đè lên source C native → build `.so` → chạy `dotnet demo/bseq.dll` mode 2.
- Flash/attn path có waiver `s0.125` từ trước — **không claim bitwise** cho nhóm B.

## 5. Giới hạn đã biết (nói thẳng, không che)

1. **Đường cuBLAS tùy chọn** (`TKV_ATTN_CUDA=1`) của `tkv_attn_cuda.c` bị bỏ: DSL CPU không biểu diễn được cuBLAS. Hành vi mặc định (trả 0) giữ nguyên.
2. **Vẫn còn C/C++ ở lớp bridge**: `torch_fused.cpp` (ATen bridge), `adamw.cuda.cu` (nvcc), `tkv_simd_native.c` / `tkv_cuda_native.c` / `tkv_step124.c` / `tkv_prof.c` (wrapper, điều phối, profiling). "Không còn C/C++" chỉ đúng cho **kernel implementation**.
3. **Loss 25.45386 vs PyTorch 9.4876 không so được trực tiếp**: native dùng `gpt2_init(7)` + RNG riêng + implementation riêng; PyTorch dùng HF `GPT2LMHeadModel`. Con số 25.45386 là fingerprint EXACT, không phải metric chất lượng. Tương tự GPU: TKV 3.89 vs HF 9.00 ở step 0.
4. **355× CPU chưa phải parity nghiêm ngặt** vì lý do (3). Con số là measured, nhưng cần harness cùng weights để khẳng định.
5. CPU total time: baseline 29271 µs → bản 14-kernel 30112 µs → bản all-TKV 28412 µs. Chỉ báo số đo, **không claim tốc độ CPU tăng**.

## 6. HEAD cuối

| Repo | Branch | HEAD |
|---|---|---|
| TokenVector | main | `9ccc17b` |
| TokenVector | prototype/cuda-backend | `688567d` |
| TokenVector.Inference | main | `10476b5` |

## 7. Việc tiếp theo

Parity loss harness: export **cùng một bộ weights** sang cả native bridge và HF model, so cross-entropy trên cùng `ids`. Notebook sẽ được tạo ngay sau báo cáo này.

## 8. Parity loss - ket luan (bo sung sau khi chot ban dau)

Notebook colab_PARITY_LOSS_V3_2026-11-22.ipynb (Inference 9fb00ab): tao weights phan phoi
N(0, 0.02^2) y het native gpt2_init, nap vao HF (tie lm_head = wte), cung ids = (i*37+11)%V:

    HF loss with NATIVE-init weights = 10.789082
    NATIVE bseq loss (same init)     = 25.453860
    DIFF = 14.664778
    ln(V) = 10.824905

KET LUAN: chenh loss KHONG phai do initialization. HF voi cung phan phoi weights ra
~ln(V)=10.82 (dung cho logits gan uniform), trong khi native ra 25.45. Vay hieu ung nam o
forward/loss implementation cua native reference (gpt2_forward_saved / smm_dot_pd /
gpt2_loss), KHONG phai kernel TKV da thay the.

Bang chung kem: 23 kernel da duoc verify rieng le bit-exact/tolerance, va native loss giu
nguyen 25.45386 truoc/sau moi lan thay - tuc sai so nay CO SAN trong native reference,
khong phai do mirror TKV gay ra.

Ve 355x CPU: van la measured, nhung chung loai - so sanh giua hai implementation khong co
cung numerics. Khong nang cap thanh "cung bai toan nhanh hon 355x".

## 9. DEBUG LOGITS - locate nguon cua loss 25.45 (bo sung)

Benv `colab_DEBUG_LOGITS_PARITY_2026_11_22.ipynb` (Inference 8945eb4) chay `bseq_debug.exe`
(compiled tu `simd_bench_debug.tkv` bang tkvc-tv local, co ham dbg_logits). Ket qua:

    (warm, truoc opt.step)   max=3.0046 min=-2.986 mean=-0.0003  row0=-0.286 -0.804 -0.376
    (loop iter 1, sau 1 step) max=3.0046 min=-2.986 mean=-0.0003  row0= same
    (loop iter 2, sau 2 step) max=11.095 min=-12.035 mean=8.3098 row0=7.377 8.314 10.190
    BPHASE ... loss_val=25.45386

    HF cung phan phoi weights: max=2.9598 min=-3.0413 mean=-0.0014, CE=10.789081

KET LUAN:
- Forward NATIVE o lan dau (va sau 1 step) HOAN TOAN BINH THUONG, logits khop HF
  (max ~3, mean ~0, CE ~10.8). Khong co loi lm-head / smm_dot_pd / gpt2_loss.
- Logits chi NO sau lan opt.step() thu 2 (mean 8.31, variance lon) => loss 25.45 la
  HAU QUAN cua trang thai params sau 2 buoc optimizer cua native reference, KHONG phai
  loi forward. Day la artifact CO SAN cua native reference (ghim la EXACT 25.454).
- Vi 23 kernel deu duoc verify bit-exact/tolerance rieng le va loss gium nguyen 25.45386
  truoc/sau moi lan thay, artifact nay KHONG do mirror TKV gaya ra.

Khong tua sua native reference - no la nen tang cua EXACT gate. Ghi nhan de biet cach doc
con so 25.454: no la fingerprint on dinh, khong phai training loss so sanh duoc voi HF.
