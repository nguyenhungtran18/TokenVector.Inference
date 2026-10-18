# HANDOFF — TokenVector.Inference (Pure TokenVector) — 2026-10-17

Cập nhật sau khi compiler self-host `D:\TokenVector\Release\dist\tkvc-tv.exe`
được sửa và repo chuyển hết sang dùng nó.

---

## 1. Trạng thái: XANH TOÀN BỘ

| Hạng mục | Kết quả |
|---|---|
| `tools\run_cases.ps1` | `pass=26 fail=0 crash=0` |
| `gap_tests` | `PASS (16/16)` |
| `build_inference.ps1 -SkipBenchmarks` | Completed Successfully |
| `build_inference.ps1` (đủ benchmark) | Completed Successfully |
| 48/48 module biên dịch được | probe gộp toàn bộ module: `ALL_MODULES_IMPORTED: 48` |
| 5/5 entry point biên dịch + chạy | `inference_tests`, `gap_tests`, `quickstart`, `compare_rivals`, `cli` |
| **Toàn bộ thân hàm type-check** | `--verify-all` **PASS cả 5/5 entry** (compiler 17:13 17-10-2026) |
| Benchmark stress | `ok=500 err=0` (ổn định 4/4 lần chạy) |
| `quickstart` | `Y[0] = 2.25` |

### Số đo thật (B-05 đã đóng, trước đây mọi thời gian đều là `0`)

```
Cold-start parse+arena : 78.0 ms
Single-sample FusedLinear 64x128 GELU : 32.0 us
Heap per 500 inferences : -728 B   <-- Zero-GC đạt, đo thật
FP32 641.0 ms | INT8 1797.0 ms | speedup 0.357x
Hot-path endurance : 64.5k req/s | Concurrent 32T/500 : ok=500 err=0
```

Không còn lỗi biên dịch nào trong repo.

---

## 2. Git

| Commit | Nội dung |
|---|---|
| `53e817f` | Gọi record constructor theo `__init__` của compiler mới (~130 chỗ) |
| `70256f2` | Sửa 2 test cuối (LSTM overflow, LoRA) → `inference_tests` 26/26 |
| `5d8a270` | Sửa `compare_rivals`: pool sai kiểu, gán list vào field, race worker |

Working tree sạch, trừ file chưa commit: `COMPILER-NOTES-2026-10-16.md`
(thông báo của phía compiler, để bạn quyết định có đưa vào repo không).

---

## 3. Quy tắc compiler mới — PHẢI NHỚ KHI SỬA CODE

### 3.1 Quy tắc đồng bộ file `tv.*.tkv` (nguồn sai kinh điển)

`import tv.runtime.session` từ `tv/quantization/x.tkv` resolve về
**bản colocated** `tv/quantization/tv.runtime.session.tkv`, KHÔNG phải
`tv/runtime/session.tkv`.

**Sau mỗi lần sửa file canonical `tv/<mod>/<name>.tkv`, phải copy sang TẤT CẢ
bản `tv.*.tkv` cùng tên ở mọi thư mục.**

Hiện có ~40 nhóm mirror. Script đồng bộ dùng trong phiên này:
quét mọi `tv/*.tkv` không phải `tv.*` làm canonical, so với mọi file
`tv.*.tkv` (bao gồm `stdlib/tv/inference/`), copy lệch. **Luôn chạy script
này trước khi build**, nếu không rất dễ gặp "module biên dịch ở bản cũ".

Ngoại lệ: `tv.numerics.tensor` / `tv.numerics.linalg` chỉ tồn tại dưới dạng
stub trong `stdlib/tv/inference/` (không có canonical) — đừng xóa.

### 3.2 Positional của record constructor

- `__init__` **có tham số bắt buộc** → positional theo đúng thứ tự tham số,
  **không** có `_ok` ẩn.
- `__init__` **chỉ có tham số mặc định** hoặc không có → positional map theo
  **tất cả field không container, kể cả `_ok`**.

Hai trường hợp cùng tồn tại trong repo, phải đọc `__init__` mới biết dùng
cách nào. Sai thì lỗi kiểu `record 'X' can N tham so (...)`.

### 3.3 Record field

- Không dùng `"inferred"` — phải khai báo kiểu cụ thể.
- Kiểu phải **resolve được** từ import ở đúng thư mục (xem 3.1).

### 3.4 Biên dịch lazy

Probe `import <module>` **không** kiểm tra thân hàm. Phải build entry point
thật thì lỗi trong thân hàm mới lộ. Đừng tin vào "N/N module PASS".

---

## 4. Trạng thái lỗi compiler

### 4.1 Đã được compiler đóng (17-10-2026)

| Mục | Trạng thái |
|---|---|
| B-01 `list[f64]` → `list[f32]` | đã kiểm tra mismatch tại call site |
| B-02 `*`/`+` i64 unchecked | đã sửa |
| B-05 `now_ms`/`now_us`/`gc_*` | đã có, số đo chạy thật |
| B-06 chẩn đoán record ctor | in cả 2 khả năng |
| B-07 `--verify-all` | đã có, PASS 5/5 entry |

### 4.2 Còn lại (đề nghị đã gửi `D:\TokenVector\Release\docs\e_handoff\2026-10-17-inference-repo-verify-after-b01-b07.md`)

| Mức | Vấn đề | Workaround trong repo |
|---|---|---|
| P1 | `gc_collect()` gọi dạng lệnh độc lập bị từ chối (thiếu trong `_BUILTIN_LENH_DOC_LAP`, `file_io.tkv:190`) | **đã đóng** |
| P1 | `gc_counts()` luôn trả `""` → gate `0/0/0` không bao giờ xanh | **đã đóng** (trả `2/2/2`) |
| P2 | `simd_*` từ chối `list[f32]` (`shape=list`) | **đã đóng** — `simd_add` chạy đúng |
| P2 | `--verify-all` bắt buộc có entry `main` → không kiểm được module thư viện | **đã đóng** — verify được cả 48 module |
| P3 | Thông báo lỗi annotated local không nhắc quy tắc **nháy kép** | dùng `"list[f32]"` |
| — | `sys_argv()` vẫn trả `[]` | `cli.tkv` luôn in usage |

> **Đính chính:** "annotated local không biên dịch được" **không phải bug compiler**.
> TokenVector yêu cầu annotation dùng **nháy kép**; với nháy kép cả 3 dạng đều đúng.
> Nháy đơn vẫn hợp lệ trong chữ ký hàm (repo dùng nháy đơn ở mọi chữ ký).

### 4.3 Workaround đã có thể gỡ (B-01/B-02 đã đóng)

| Workaround | Vị trí | Gỡ thử |
|---|---|---|
| copy phần tử thay vì truyền thẳng `list` | `tv/onnx/translator.tkv` | `_slice_shape` |
| LCG 31-bit thay vì `i64` | `tv/training/autograd.tkv` | `lora_forward_and_step` |

Gỡ **từng cái một**, chạy lại `tools\run_cases.ps1` sau mỗi lần.

Ghi chú compiler (file ở repo compiler, chưa commit):
- `D:\TokenVector\Release\docs\e_handoff\2026-10-16-inference-repo-list-alias-and-overflow-bugs.md`
- `D:\TokenVector\Release\docs\e_handoff\2026-10-16-inference-repo-record-container-type.md`
  (kèm bảng **file:line cần sửa trong `compiler_bootstrap/compiler/il_features/`**)

---

## 5. Bài học đã tốn thời gian — đừng lặp lại

1. **Đừng static-reverse-engineer compiler.** Viết probe tối thiểu rồi chạy —
   nhanh hơn nhiều. Nhiều giả thuyết ("gán list vào field hỏng", "ctor lồng
   nhau làm null") hoá ra **không tái hiện được** khi tách nhỏ; nguyên nhân
   thật là thiếu coerce `f64`→`f32`.
2. **Lỗi "lazy" đòi phải gọi thật hàm.** Khi sửa `lin_head` / `lora_linear_create`,
   phải build + chạy `inference_tests`, không dừng ở "module compile được".
3. **Sửa canonical rồi sync mirror ngay**, rồi mới build. Đây là nguồn gây
   hiểu nhầm nhiều nhất.
4. **Khi compiler có lỗi, ghi note ở `D:\TokenVector\Release\docs\e_handoff\`**
   kèm file:line cần sửa. Sửa trong `compiler_bootstrap/compiler/**` —
   **KHÔNG sửa `build/selfhost/tv_gop.tkv`** (đó là bản merge).
5. **Sửa bug phải kèm hồi quy.** `gap_tests` đã bắt được 4 lỗi thật.

---

## 6. Việc còn lại (nếu muốn tiếp)

| Việc | Ghi chú |
|---|---|
| **G2 backward**: `gpt2_ref.tkv` co day du `attention_bwd`/`block_bwd`/`gpt2_backward` (recompute, khong luu trung gian); `gpt2_tests` **10/10 PASS** (forward 5 + grad 5 so `g2_grad_*`, wte 1.79e-07) | bug da fix: `block_bwd` thieu **term residual** cua `dx` (r = x + attn) — dang `dln1_g` PASS nhung grad tang duoi sai; quirk: khong duoc `list[i].data[j]` (gan ra bien) |
| **G3 — Data pipeline**: tokenizer parity (GPT-2 BPE HF) + packing + shard | `tokenizer.tkv` đã có; fixture token id bắt buộc |
| **G4 — Trainer**: AdamW + schedule + grad accumulation + checkpoint + CLI `train` | `autograd.tkv` đã có Adam, `lora_forward_and_step`; cần `grad_tests` mở rộng cho chuỗi kiến trúc |
| **G5 — Harness**: Colab reference script (`compare_rivals` + GPU) | `docs/linux-portability-spike.md`: `.tkv` chạy được, compiler chưa; `simd_gemm` 14.6x (vuông); `simd_*` 3-arg chưa dùng được |
| **G6 — Phân tán**: DDP/ZeRO-1/2 (tùy chọn) | `atomic_add`, `thread_spawn` đã có; `simd_*` chưa dùng cho parity |
| Số đo benchmark thật | cần timing thật thay stub; hiện **không dùng được** |
| SIMD | cần `System.Numerics.Vectors.dll` cạnh exe (B-04 đã đóng); `simd_gemm` 14.6x vuông, không áp dụng cho M=16 K=256 N=256; `simd_*` 3-arg chậm 2.4x + cấp phát 16 B/phần tử |
| Ghi note compiler | `docs/e_handoff/` đã có 4 note `2026-10-16-*`; `COMPILER-NOTES` chưa stage |
| **Bit-exact vs PyTorch** | **đã đo + CHẤP NHẬN có kiểm soát** (2026-10-18): matmul/mean KHÔNG thể bit-exact (MKL sgemm AVX2, mọi ứng viên thứ tự đều sai; tuần tự hiện tại đã là tốt nhất — 1499/2048 vs MKL 5.7e-06); exp/sqrt/add bit-exact OK. Hạn mức thật: forward 2.68e-07, grad 1.79e-07 << 1e-5. Công cụ **`tools/verify_parity.ps1` ĐÃ XÂY** (3 lớp: fixture oracle diff / bảng bit-diff vs baseline / gate run_cases; report `reports/parity-<ngay>.md`, exit tường minh) — lần chạy đầu XANH: 100/100 fixture khớp bản sinh lại. Kèm `tools/compare_fixtures.py`, generator nhận `--out`. Lưu ý: `.ps1` phải thuần ASCII (PS5.1 + UTF-8 không BOM → smart-quote) → `docs/accumulation-order-vs-pytorch-2026-10-18.md`; probe: `tools/probe_accumulation_order.py` |

---

## 7. Lệnh kiểm tra nhanh

```powershell
powershell -File tools\run_cases.ps1          # 26 case + gap 16 + parity 4 + grad 20 + tape 8 + gc 7 + gpt2 5
powershell -File build_inference.ps1          # đầy đủ, gồm benchmark
```

Nhớ chạy **script đồng bộ mirror** (mục 3.1) trước khi build.
