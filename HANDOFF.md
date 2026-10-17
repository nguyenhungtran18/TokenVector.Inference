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
| Ép biên dịch thân hàm | 46/48 module sạch, **366 hàm** được gọi trong probe |
| Benchmark stress | `ok=500 err=0` (ổn định 4/4 lần chạy) |
| `quickstart` | `Y[0] = 2.25` |

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

## 4. Lỗi compiler CÒN LẠI (đã tự kiểm chứng lại 2026-10-17)

| # | Vấn đề | Triệu chứng | Workaround trong repo |
|---|---|---|---|
| 1 | Truyền `list[f64]` (suy ra từ `[]` + `.append(1.0)`) vào tham số `list[f32]` | `len()` đúng, **phần tử sai**, không báo lỗi | copy từng phần tử: `for e in src: dst.append(e)` |
| 2 | `*`/`+` i64 dùng checked arithmetic | `OverflowException` | LCG 31-bit: `(s * 1103515245 + 12345) & 0x7FFFFFFF` |
| 3 | `argv()` | trả `[]`, CLI không nhận tham số | `cli.tkv` luôn in usage |
| 4 | `simd_*` | biên dịch được, chạy lỗi `FileNotFoundException: System.Numerics.Vectors` | chưa có |
| 5 | `now_ms`/`now_us`/`gc_*` | stub trả `0` → benchmark in `0.0 ms`, tỷ lệ so sánh ra `infx` | coi là smoke test, **không dùng làm số đo** |

Lưu ý: workaround "khai báo biến đúng kiểu" (`_src: 'list[f32]'`) trong thông
báo của phía compiler **chưa biên dịch được**:
`il_codegen: khong dich duoc dong: "_a: 'list[f32]' = []"`.

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
| Commit `COMPILER-NOTES-2026-10-16.md` | quyết định của bạn |
| Số đo benchmark thật | cần timing thật thay stub; hiện **không dùng được** |
| SIMD | cần `System.Numerics.Vectors.dll` cạnh exe |
| Xác minh 2 module chưa sạch | `tv/onnx/translator.tkv`, `tv/server/embedded_server.tkv` — lỗi của **probe harness** (thiếu số đối số record), chưa phải lỗi repo; cần probe vòng thích ứng nhiều hơn |
| Phủ nốt 82 hàm chưa ép biên dịch | 448 hàm cấp module, 366 đã sạch; phần còn lại có tham số kiểu record phức tạp |

---

## 7. Lệnh kiểm tra nhanh

```powershell
cd D:\TokenVector.Inference
powershell -File tools\run_cases.ps1          # 26 case + gap 16
powershell -File build_inference.ps1          # đầy đủ, gồm benchmark
```

Nhớ chạy **script đồng bộ mirror** (mục 3.1) trước khi build.
