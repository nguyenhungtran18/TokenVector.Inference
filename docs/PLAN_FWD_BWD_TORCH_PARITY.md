# PLAN — .tkv fwd+bwd tiến gần tốc độ torch (2026-11-16)

**Mục tiêu:** đo lại và đóng gap .tkv vs torch CPU trên cùng máy/thread; verify rằng
".tkv chậm hơn torch" trên Colab không phải do code, mà do (hoặc không phải chỉ)
thread mismatch và GPU bridge research.

**Đối tượng:** GPT-2 124M, seq=1024, `tv/tests/simd_bench_124m_seq.tkv`, loss EXACT mode 0.

**Không làm:** rewrite runtime; đổi compiler / native layer trừ khi phép trừ ROI lớn hơn 5%.

---

## 0. Sự thật về ".tkv chậm torch"

| Phiên | Máy | .tkv full step | torch | .tkv/torch |
|---|---|---:|---:|---:|
| 2026-11-11 (§9.22) | Windows, Xeon E5-2695 v3 | 4016 | 4061 | **0.99×** |
| 2026-11-11 (§9.22) | Windows, Xeon E5-2695 v3, mode 0 | 3375 | 3395 | 0.99× |
| 2026-11-15 (§9.44) | Colab T4, 2 vCPU | 15 548 | 10 906 (§9.43) | **1.43×** |
| 2026-11-15 (§9.44) | Colab T4,同上 | 15 783 (§9.45 GREEN) | 10 906 | **1.45×** |

Conclude: **trên cùng máy, cùng protocol, .tkv ~ parity (0.99×)**. Cùng VM với cùng hệ
(Windows + 2 vCPU), gap ~1.45×.

### 0a (2026-11-15) — thread mismatch: REJECTED
```
threads=2: TONG=16 733
threads=4: TONG=16 917
threads=8: TONG=17 153
default (12, 2-vCPU): ~16 231
```
Chênh lệch chỉ 3–5,7% — **không phải nguyên nhân**. Thanh liệu sau đây không kết thúc được.

### 0b (2026-11-15, `[10/9 torch-profile]`) — same-machine parity
```
aten::mm                       4.843s  52%  99 calls
aten::addmm                    1.591s  17%   48 calls
aten::_sdpa_flash_backward     1.260s  13%   24 calls (12 fwd + 12 bwd)
aten::mul / add / log_softmax  ~0.9s    ~10%

Self CPU backward: 9.291s
```
.tkv bwd wall (`TKV_PROF`): ~10.6 s → **gap ~1.1×**, không phải 1.45×.

| Op (fwd bwd) | .tkv ms | Ghi chú |
|---|---:|---|
| `ATTN_BWD_MERGE` | 5 663–6 507 ms | vs torch flash bwd 1.26 s — gap lớn nhất |
| `LMHEAD_BWD` | 6 507–8 121 ms | vs torch trong mm/addmm (~1.6–2 s) |
| `LIN_DW` + `LIN_DX` | 10 548–10 689 ms | vs torch mm (4.8 s + 1.6 s addmm) — gap ~2.1× |
| `ATTN_FWD` | 4 674–5 119 ms | vs torch flash fwd ~0.4–1.0 s |
| LN (fwd+bwd) | 697–1 132 ms | chỉ ~5–10% |
| GELU | ~1 175 ms | ~10% |
| CE | ~500–900 ms | ~8% |

Dominant ops: **GEMM (4.84 + 1.59 s vs 10.6 s)** + **attention bwd**.

## 0.5 GPU bridge hiện tại đã GREEN (2026-11-15)
Đã fix `torch_fused.cpp::host()` (CUDA mem-copy bug). BENCHMARKS.md §9.45.
GPU bridge GREEN nhưng vẫn khoảng **25× slower torch CUDA** (~8.2 s vs 270 ms).

## 1. Breakdown gây chậm (Colab 2026-11-15) — xem §0b

Cơ hội: **attention backward** (biggest), sau đó **GEMM** vs torch. LN/GELU nhỏ.

## 2. Chiến lược benchmark công bằng
1. **Windows local, same machine**: `.tkv` + torch cùng protocol (x3 median),
   loss EXACT trước khi merge vào official.
2. **Colab T4**: `TKV_GEMM_THREADS=2` tiếp tục là fair; thread không phải vấn đề.
3. **Ghi lại phiên** theo luật TRACKED_PLAN A5 + BENCHMARKS.

## 3. Priority thực thi

1. **Ghi nhận đỉnh**: .tkv/torch ngang trên Windows (§9.22); gap Colab là do VM.
2. **Hướng A — LN (1.1 s → 0.5 s)**: không ưu tiên, vì LN chỉ ~5–10%.
3. **Hướng B — Attention backward** (5.7 s → <2.5 s):
   - profile fallback path của `.tkv attention_bwd_merge_multi` xem có thể tái tối ưu không;
   - xem xét ATen path có bit-exact nhưng chậm; nếu issue, fix fallback;
   - nếu được, port flash-attn-2 recomputation (no extra memory).
4. **GEMM** — mổ xẻ OpenBLAS vs oneDNN trên Windows vs Colab, đảm bảo dùng đúng path.
5. **LMHEAD_BWD** — đo ATen path vs native fused, chọn nhanh.

**Sau khi fix các bước trên, trên Windows .tkv có thể parity hoặc thắng torch full step.**

## 4. Ràng buộc
- Loss EXACT mode 0 = 25.407877 luôn, nếu lệch = revert `_git adddemos` sạch thật.
- Không rewrite runtime trừ khi code present + ROI > 5%.

## 5. Câu hỏi mở
- Tại sao bwd attention .tkv hiện tại bị LN ~5x chậm hơn torch — rate percent trong `attention_bwd_merge_multi`.
- Có thể cấu hình .tkv kernel `LinearDwFused` vs `.tkv sgemm_packed` cho nhanh tương đương aten::mm 4.8 s?
- Nếu dự định chạy trên T4, cần chắc chắn `TKV_GEMM_BLAS` + `TkvCUDA_Available` consistent.
