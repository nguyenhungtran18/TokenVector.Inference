# ROADMAP — Huấn luyện LLM trong TokenVector (chứng minh thuật toán)

Ngày lập: **2026-10-17**
Trạng thái: **KHÔNG CODE** — tài liệu này chỉ là kế hoạch.
Phạm vi: chứng minh đúng thuật toán huấn luyện LLM; việc kiểm thử hiệu năng thực
sẽ diễn ra sau trên GPU Google. Ưu tiên số tuyệt đối với PyTorch.

## TRẠNG THÁI

| Giai đoạn | Trạng thái |
|---|---|
| **G0** | ✅ **XONG** (17-10-2027) — xem bên dưới |
| **G1** tensor core + autodiff thật | **XONG** (2026-10-17): `tensor.tkv`; `ops.tkv` (12 op fwd); fixture 47; `grad_tests.tkv` 20/20; `tape.tkv` 8/8; `parity_tests.tkv` 4/4; `kernels_reg.tkv` co mau |
| **G2** GPT-2 thu nho (256/64/4/2) | **FORWARD + BACKWARD XONG** (2026-10-18): `gpt2_ref.tkv` day du embed/LN row/MHA causal/MLP/tie head + `attention_bwd`/`block_bwd`/`gpt2_backward`; `gpt2_tests` **10/10 PASS** (hidden h0-h2, logits 2.68e-07, loss 0.0; grad wte 1.79e-07, ln1_g0 3.3e-09, attn_w0 3.4e-08, mlp_w1_0 2.5e-08, lnf_g 9.3e-09) vs `torch.autograd`; `run_cases` [8/8] + `--verify-all` xanh. Gate con lai: grad bit-exact (dang dung tol 1e-5 nhu grad_tests) |
| **G3** data pipeline | ✅ **XONG** (2026-10-19): tokenizer GPT-2 BPE parity `8ac487d` (9/9 case + roundtrip); packing + mask + shard + stream `d8a203a`/`pack_tests.tkv` **5/5 PASS** (pack 8x32, mask block 32x32 + causal4, shard roundtrip 16 shard/256 token, stream seed42 checksum 1184745045, determinism seed 42/43) vs reference `tools/gen_pack_fixture.py` |
| **G4** trainer | ✅ **XONG** (2026-10-19): `tv/training/trainer.tkv` — AdamW (decoupled WD) + warmup/cosine + clip norm + grad accumulation + checkpoint bit-exact (trọng số/m/v/RNG), vòng train/eval log loss-ppl-throughput, lệnh `train` trong `cli.tkv`; `trainer_tests` **15/15 PASS** (1 bước khớp PyTorch loss 1.2e-7 / grad norm 0.0 / trọng số 3e-8; overfit 20 mẫu eval_loss 0.01495 acc 1.000; resume 3+save/load+3 **bit-for-bit**). Lệch còn lại vs PyTorch ~2.4e-6 (thứ tự làm tròn f32), không phải 0 tuyệt đối |
| G5 harness + seam GPU | chưa bắt đầu |
| G6 phân tán (có thể bỏ) | chưa bắt đầu |

### Kết quả G0

| Việc | Kết quả |
|---|---|
| Spike Linux/Colab | `.tkv` **chạy được** trên Ubuntu/mono. Compiler **không biên dịch được** trên Linux (IL của chính compiler bị mono từ chối + thiếu `ilasm`). Chi tiết: `docs/linux-portability-spike.md` |
| Fixture từ PyTorch | `tools\gen_fixtures.py` (torch 2.14.0+cpu, fp32, seed 0) sinh 9 fixture vào `tv\tests\fixtures\` |
| Đối chiếu phía `.tkv` | `tv\tests\parity_tests.tkv` — **4/4 PASS**, đã nối vào `tools\run_cases.ps1` thành cổng `[4/4]` |
| Hợp đồng số học | `docs\numerics-contract.md` (chờ duyệt) |

Số đo đạt được: `gelu(tanh)` **0.0**, `cross_entropy` **0.0** (bit-exact);
`softmax` 1.86e-09; `rmsnorm` 1.19e-07.
Đồng thời xác nhận bẫy đã dự đoán: GELU tanh-approx lệch PyTorch mặc định **4.12e-04**
→ bắt buộc phải viết GELU erf ở G1.

Cổng hồi quy sau G0: `run_cases` **26/26 + gap 16/16 + parity 4/4**,
`--verify-all` **49/49 module**.

---

## 1. Quyết định đã chốt

| # | Quyết định |
|---|---|
| 1 | Mục tiêu: **full LLM** (pretrain + fine-tune), chỉ cần **chứng minh thuật toán** |
| 2 | Kiến trúc chuẩn đối chiếu: **GPT-2 124M** |
| 3 | Hệ quả so sánh: **golden fixture sinh từ PyTorch**, check-in sẵn vào repo |
| 4 | Cả CPU và GPU đều cần; phần GPU để sau, chỉ phải có **seam đúng** |
| 5 | Việc test trên GPU Google: kế hoạch riêng, không nằm trong tài liệu này |

## 2. Mục tiêu thật (viết không mời)

Sau kế hoạch này, repo **chứng minh được** rằng:

1. Thuật toán backward của từng op là **đúng**, có gradcheck tự động.
2. Một kiến trúc Transformer **đầy đủ** (GPT-2 124M) chạy forward/backward/train
   khớp **PyTorch từng chữ số** trên một cấu hình thực thi tất định (CPU).
3. Có thể **overfit** một tập dữ liệu cực nhỏ tới hội tụ, và **resume** checkpoint
   cho kết quả giống hệt lần chạy không ngắt.
4. Tách được phần thuật toán khỏi phần thực thi, để sau này gắn backend GPU mà
   **không sửa thuật toán**.

Không nằm trong mục tiêu: huấn luyện thật ở quy mô lớn, tốc độ, GPU, phân tán.

## 3. Định nghĩa "khớp 100% PyTorch"

Đây là điểm dễ hiểu sai nhất, nên đặt rõ trước.

| Đường thực thi | Cam kết |
|---|---|
| **CPU xác định** (fp32, seed cố định, thứ tự giảm tử tài liệu hoá) | **bit-exact 100%** với PyTorch |
| **GPU** | bật deterministic algorithm, tắt TF32 → khớp trong dung sai tuyên bố (mục tiêu rel ≤ 1e-5). **Không hứa bit-exact** |

Lý do: GPU giảm theo thứ tự khác CPU, TF32 mặc định trên Ampere+, và atomic
scatter-add không xác định thứ tự. Bit-exact giữa CPU và GPU là điều không tồn tại
ở PyTorch, nên không thể là mục tiêu.

**Hệ quả bắt buộc:** mọi phép giảm tử (sum/mean/matmul) phải có thứ tự tích lũy
**cố định và ghi trong tài liệu**. Không được để thứ tự phụ thuộc vào parallelization.

## 4. Hợp đồng số học (đóng băng — sửa sau này rất đắt)

| Mục | Chốt | Bẫy parity đã biết |
|---|---|---|
| Kiểu trọng số chính | `fp32` | |
| Kiểu tích luỹ | `fp32`, không fast-math | TF32 làm lệch ngay |
| GELU | **erf chính xác** | tanh-approx lệch ~1e-3 so với PyTorch mặc định |
| LayerNorm / RMSNorm | trừ mean/square-mean rồi chia `+eps`; tính bằng fp32 | chỗ đặt `eps` khác nhau là khác số |
| Softmax | trừ max → exp → tổng → chia, theo đúng trục, thứ tự cố định | |
| CrossEntropy | log-sum-exp + NLL, `mean` trên token **không** mask | nhiều token mask vẫn phải nằm trong mẫu số đúng |
| Dropout | **tự implement RNG có seed** | không thể dùng RNG của hệ đối với đường parity |
| Init trọng số | normal(0, 0.02) với đúng thứ tự phát sinh | thứ tự sinh khác → khác số |
| Adam/AdamW | betas 0.9/0.999, eps 1e-8, bias-correction theo step, weight decay tách khỏi gradient | dễ nhầm Adam với AdamW |
| LR schedule | đúng công thức `get_linear_schedule_with_warmup` | |
| Embedding backward | sắp xếp chỉ số trước khi cộng | thứ tự cộng ảnh hưởng số |
| Gradient accumulation | cộng theo thứ tự micro-batch cố định | |

### 4.1 Bẫy riêng của GPT-2 trên HuggingFace — phải biết trước

| Vấn đề | Chi tiết |
|---|---|
| `Conv1D` không phải `Linear` | HF GPT-2 lưu trọng số cỡ `[in, out]`, ngược chiều với `Linear`. Sai chiều là sai số. |
| Không scale attention theo tầng | `scale_attn_by_inverse_layer_idx = False` cho `gpt2` |
| Softmax attention tính ở fp32 | |
| Dropout mặc định | `embd_pdrop=0.1`, `attn_pdrop=resid_pdrop=0.1` → ở chế độ train có **dropout thật**. Xem 4.2. |
| Tokenizer | phải trùng GPT-2 BPE (tiktoken / HF `gpt2`), kể cả cách tách từ bằng regex |

### 4.2 Về dropout — phải tách khỏi mục tiêu parity

Khớp bit-exact cả phần dropout đòi hỏi phải phát lại đúng PRNG của PyTorch
(Philox) và đúng thứ tự tiêu thụ số ngẫu nhiên. Đó là việc riêng, rủi ro cao.

Đề xuất tách:
- **Parity giai đoạn đầu**: chạy với dropout = 0 (tương đương chế độ eval của HF).
  Khi đó parity là khả thi và là bằng chứng thuật toán đúng.
- **Parity có dropout**: mục tiêu riêng, sau. Nếu làm thì phải phát lại Philox.
- Hạt giống tự viết vẫn bắt buộc: để kết quả **tái lập được**, không phải để khớp PyTorch.

## 5. Cơ chế chứng minh — golden fixture từ PyTorch

### 5.1 Nguyên tắc

Không so sánh "kết quả cuối cùng". So sánh **từng tầng**, để khi sai biết sai ở đâu.

### 5.2 Một fixture gồm gì

Với mỗi ca kiểm thử, lưu:

| Thành phần | Kích thướt | Mục đích |
|---|---|---|
| `input_ids` | `[B, T]` int64 | đầu vào |
| `loss` | 1 số fp32 | so sánh tổng |
| `hidden` theo tầng | **thống kê** mỗi tầng (mean, std, L2) + vài phần tử chọn sẵn | khoanh vùng lỗi mà không giữ tensor khổng lồ |
| `logits` | **mẫu thưa**: vài vị trí × top-k xác suất | kiểm tra đầu ra mà không giữ 50257×T |
| `grad` | một số tham số đã chọn, toàn bộ | kiểm tra gradient |
| `config` + `seed` + `phiên bản torch` | JSON | tái lập |

Lý do lấy mẫu: logits GPT-2 124M ở T=1024 là 50257×1024×4B ≈ 200 MB — không thể
check-in. Thống kê theo tầng + mẫu thưa vừa đủ định vị lỗi, vừa nhỏ.

### 5.3 Quy trình

1. Viết script Python sinh fixture từ `transformers` + `torch` (chạy 1 lần trên máy có GPU hoặc CPU).
2. Chốt môi trường: `torch` phiên bản ghi trong fixture; `attn_implementation="eager"`; `fp32`; `device="cpu"`.
3. Check-in fixture vào repo cạnh test.
4. Test của TokenVector đọc fixture và so khớp.

### 5.4 Fixture tối thiểu cho G1 (chỉ op)

Mỗi op một fixture: input → output → gradient. Gradcheck **và** so fixture.
Hai tầng kiểm chứng độc lập: gradcheck bắt lỗi công thức, fixture bắt lỗi khớp với
cách PyTorch **thực sự** làm (thứ tự giảm tử, cách reshape, cách mask).

## 6. Kiến trúc đích

```
                 ┌──────────────────────────────────────────┐
                 │  Toán thuật (không biết backend)          │
                 │  autograd tape · GPT-2 · trainer         │
                 └───────────────┬──────────────────────────┘
                                 │ gọi qua interface thuần
                 ┌───────────────▼──────────────────────────┐
                 │  Kernel registry (tên op → hiện thực)    │
                 ├───────────────┬──────────────────────────┤
                 │  CPU backend  │  GPU backend (để sau)    │
                 │  scalar/SIMD  │  seam, chưa viết          │
                 └───────────────┴──────────────────────────┘
```

Yêu cầu cứng: **đổi backend không được sửa một dòng thuật toán nào.** Đó là tiêu chí
để "sau này test trên GPU Google" thực sự khả thi thay vì phải viết lại.

## 7. Các giai đoạn

Quy ước cỡ: **S** ≈ vài ngày, **M** ≈ 1–2 tuần, **L** ≈ 1 tháng trở lên.

### G0 — Spike khả thi + chốt hợp đồng số học  *(S)*

**Nội dung**
- Chạy thử 1 forward + 1 backward tối giản trên CPU.
- Trả lời câu hỏi sống còn: `tkvc` có biên dịch/chạy được trên Linux/Colab không?
  (nếu không: mono? dotnet? WSL? hay chỉ sinh PE Windows?)
- Ghi tài liệu `docs/numerics-contract.md` chốt mục 4.
- Chốt định dạng fixture và **sinh thử 1 fixture op** từ PyTorch để xác nhận đường
  làm việc trước khi xây gì.

**Gate**
- Forward/backward CPU chạy đúng.
- Câu hỏi Colab có câu trả lời dứt khoát (có/không/khác).
- Đường sinh fixture PyTorch → đọc kiểm tra trong `.tkv` chạy một vòng.

---

### G1 — Tensor core backend-agnostic + autodiff thật  *(L)*

**Nội dung**
- Abstraction tensor: shape, dtype, offset, đọc/ghi phần tử.
- **Kernel registry**: đăng ký hiện thực theo tên op. CPU backend hiện tại là
  hiện thực duy nhất.
- **Autodiff thật**: tape op, gradient tích luỹ về node, backward theo topo.
- Backward cho: matmul, add/mul, layernorm, rmsnorm, embedding (index-add có sort),
  softmax, cross-entropy (theo chuỗi + mask), gelu, silu, swiglu.
- GELU chính xác (erf) và bản tanh-approx để đối chiếu.
- Phát hiện in-place nguy hiểm; activation checkpointing (tái tính thay vì lưu).
- Gradient accumulator theo micro-batch có thứ tự cố định.

**Gate**
- Gradcheck pass cho **mọi** op.
- **Mọi op khớp 100% fixture PyTorch** (bit-exact trên CPU).
- `tools\run_cases.ps1` và `--verify-all` vẫn xanh (không hồi quy).

---

### G2 — GPT-2 124M  *(L)*

**Nội dung**
- Embedding (wte/wpe) + 12 block + head LM + liên kết trọng số (weight tying).
- Đúng công thức HF: `Conv1D` (chiều `[in, out]`), scale `1/sqrt(d_head)`, không
  scale theo tầng, softmax attention ở fp32.
- `n_positions = 1024`, vocab 50257.
- Weight init đúng normal(0, 0.02), thứ tự phát sinh cố định.
- Attention: causal mask + padding mask, và kiểm tra đường mask đúng.

**Gate**
- Forward: loss khớp **bit-exact** với HF GPT-2 124M trên 1 batch cố định.
- Hidden-state thống kê theo tầng khớp → phát hiện sai ở tầng nào.
- Backward: gradient các tham số chọn mẫu khớp bit-exact.

---

### G3 — Data pipeline  *(M)*

**Nội dung**
- Tokenizer: khớp GPT-2 BPE (kể cả regex tách từ) — kiểm bằng fixture token id.
- Packing nhiều chuỗi thành chuỗi dài `T`, có attention mask đúng.
- Chia shard, prefetch, seed tất định, số thứ tự mẫu tái lập.

**Gate**
- Token id của một đoạn văn bất kỳ khớp HF ** tuyệt đối** (so từng id).
- Cùng seed → cùng thứ tự mẫu, cùng checksum.

---

### G4 — Trainer  *(M)*

**Nội dung**
- AdamW + warmup/cosine đúng công thức, grad accumulation, clip norm.
- Checkpoint save/load: trọng số + trạng thái optimizer + step + RNG + lịch LR.
- Vòng lặp train/eval, log loss/ppl/throughput.
- Lệnh `train` trong `cli.tkv` (hiện mới có `serve|gen|export|calibrate|latency`).

**Gate**
- **Overfit** tập ~20 mẫu: loss giảm về gần 0.
- **Resume bit-exact**: dừng giữa chừng rồi chạy tiếp cho kết quả y hệt chạy liền.
- Khớp 1 bước train với PyTorch cùng hyper-parameter (bit-exact).

---

### G5 — Harness + seam GPU  *(M)*

**Nội dung**
- Script sinh fixture (PyTorch) — chạy 1 lần, check-in kết quả.
- Bộ test đối chiếu: 1 lệnh cho báo cáo tỉ lệ op/khối khớp.
- Notebooks Colab (để dùng sau): chạy CPU reference vs PyTorch GPU, rồi bật backend GPU.
- Viết interface backend để khi thêm GPU chỉ cần một module mới.

**Gate**
- Một lệnh → báo cáo `N/N khớp 100%`.
- Thêm backend giả (trả về kết quả sai có chủ ý) → bộ test **bắt được**, chứng minh
  harness thật sự kiểm tra chứ không phải luôn xanh.

---

### G6 — Phân tán  *(L — có thể bỏ)*

Data parallel (allreduce), ZeRO-1/2, checkpoint phân mảnh.
Với mục tiêu "chứng minh thuật toán", **G6 không cần**: gradient accumulation trên
một thiết bị đã chứng minh được thuật toán. Chỉ làm khi thật sự cần scale.

## 8. Ngoài phạm vi

- Backend GPU, CUDA, OpenCL (chỉ làm seam).
- Phân tán nhiều node.
- Tối ưu tốc độ, đa luồng hoá GEMM, dùng SIMD để tăng tốc.
- Quantization trong huấn luyện (QLoRA).
- Sinh văn bản sau huấn luyện chất lượng.

## 9. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| `tkvc` không chạy được trên Linux/Colab | **cao** | G0 spike phải trả lời trước mọi thứ |
| Bẫy parity kiến trúc HF (Conv1D, dropout, init) | cao | liệt kê sẵn ở mục 4.1/4.2 |
| Bĩu căn số tích luỹ khi song song hóa | cao | cấm song song hóa trong đường parity |
| Fixture quá lớn để check-in | trung bình | đã xử lý ở 5.2 (lấy mẫu + thống kê) |
| Autodiff không đủ sức cho kiến trúc mới | trung bình | registry op mở, ghi rõ op nào có backward |
| Tokenizer lệch HF | trung bình | fixture token id bắt buộc ở G3 |
| Phạm vi nở ra (muốn làm cả GPU) | trung bình | G6/GPU nằm ngoài phạm vi, ghi rõ |

## 10. Ba việc đầu tiên

1. **G0-spike-1**: kiểm tra `tkvc` trên Linux/Colab — build, chạy, kết luận.
2. **G0-spike-2**: sinh fixture PyTorch cho một op đơn giản (`gelu` hoặc `softmax`),
   nạp trong `.tkv`, so khớp. Xác nhận mọi mắt xích đều thông trước khi xây G1.
3. **G0-doc**: viết `docs/numerics-contract.md` chốt mục 4, đưa để duyệt.

## 11. Quy ước kiểm thử dùng lại

| Cổng | Lệnh |
|---|---|
| Hồi quy toàn repo | `powershell -File tools\run_cases.ps1` |
| Toàn bộ thân hàm | `tkvc <module> <out.exe> --verify-all` |
| Pipeline đầy đủ | `powershell -File build_inference.ps1` |

Mọi gate trong tài liệu này đều phải giữ ba cổng trên ở trạng thái xanh.
