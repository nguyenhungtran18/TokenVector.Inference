# THONG BAO COMPILER 2026-10-16 — Danh cho TokenVector.Inference

Compiler `D:\TokenVector\Release\dist\tkvc-tv.exe` da duoc sua de bien dich
duoc repo nay. Tom tat nhung gi da doi, viec repo can lam, va gioi han con lai.

## 1. Da sua trong compiler (test 125/125 pass, build IL_IDENTICAL)

| # | Truoc day | Bay gio |
|---|---|---|
| 1 | `.NET KeyNotFoundException: The given key was not present in the dictionary` khi gap bat ky loi kieu nao trong 3 nhom (so gán cho `str`, ternary bi LICM viet lai, slice temp) | Bao loi ro tieng Viet co vi tri/nguyen nhan |
| 2 | `_t = xs` (gan bien container) lam mat shape -> `khong suy duoc hinh dang` | Giu nguyen TypeAnn day du |
| 3 | Field `params: 'inferred'` gay KeyError | Bao loi ro: chi dung cho tham so ham, hay khai bao kieu cu the |
| 4 | `AST node khong ro: 1000412` (khong biet thieu cu phap gi) | Kem loai node (vd `bytes_lit`) |
| 5 | `'<bieu thuc>[i]' khong suy duoc` (khong biet base la gi) | Kem loai + ten + dtype/shape cua base |
| 6 | Ham khong ton tai (`spawn`, ...) | Goi y cu the (`thread_spawn`, doi ten neu trung builtin) |

## 2. Phat hien quan trong: file `tv.*.tkv` colocated phai dong bo

`import tv.runtime.session` tu `tv/quantization/x.tkv` resolve ve
`tv/quantization/tv.runtime.session.tkv` (cung thu muc), KHONG phai
`tv/runtime/session.tkv`. Repo co 13 ban copy `tv.runtime.session.tkv`
(+12 ban `tv.onnx.models.tkv`, 7 ban `tv.text.generate.tkv`).

**Quy tac: moi lan sua file canonical (`tv/<mod>/<name>.tkv`) thi copy de
len TAT CA ban `tv.*.tkv` cung ten trong moi thu muc.** Da dong bo:
13 session + 12 models + 7 generate (khi sua `sizes[red[i]]`, `.pop(i+1)`,
`b""`, `params: 'inferred'`). Neu them file moi, kiem tra lai toan bo.

## 3. Viec repo da lam / can lam (da xac minh bang bien dich that)

- [x] `_idiv`/`_imod` boc phep `//`/`%` (112+29+42 cho, tien to theo module)
- [x] Helper `_span0`/`_ints0`/`_enq0` (doc roi moi index, khong index truc tiep tren ket qua goi)
- [x] `_shp_at_sd(d, i)` thay cho `d.shape[i]` truc tiep
- [x] Ham trung ten giua 2 module -> doi ten duy nhat theo module
- [x] Xoa nhanh `else:` rong
- [x] `spawn(...)` -> `thread_spawn(worker_fn)` (lưu y: khac ngu nghia bound method)
- [x] `params: 'inferred'` -> `'SampleParams'` (7 file generate)
- [x] `self.raw = b""` -> `""` (12 file models, field khai bao `str`)
- [x] `_imod(rem, sizes)[red[i]]` -> `_imod(rem, sizes[red[i]])` (dau ngoac lot ra ngoai)
- [x] `.pop(i + 1)` -> gan bien tam truoc roi `.pop(tmp)` (4 cho)
- [ ] Doi ten ham `parallel_for` tu dinh nghia trong `runtime/backend.tkv`, `runtime/simt.tkv` (trung builtin - compiler da bao loi cung)
- [ ] `LayerNorm(d_in)` / `LayerNorm(hidden)` thieu 5 doi so (xem cach goi dung o dong ~953)
- [ ] `format/gguf.tkv`: so gán cho field `str` (compiler da chi ro vi tri)

## 4. Gioi han con lai (co workaround, khong chan build)

- **BUG 1 (xac nhan, nghiem trong)**: `Box(_src)` voi `_src: list[f64]` trong khi param la `list[f32]` -> chay ra gia tri rac, khong bao loi. Workaround: khai bao bien dung kieu (`_src: 'list[f32]'`) hoac copy tung phan tu. Compiler chua tu dong convert/bao loi o duong method call/`__init__` (duong top-level call da co coerce).
- **BUG 2**: `*`/`+` i64 dung checked arithmetic -> OverflowException thay vi wrap 64-bit. Workaround: dung LCG 31-bit (`* 1103515245 + 12345) & 0x7FFFFFFF`). Chua co `mul_wrap_u64`.
- **`argv()` tra ve `[]`**: CLI chua nhan tham so dong lenh. Cho `cli.tkv` (luon in usage).
- **`simd_*` chay loi**: thieu `System.Numerics.Vectors.dll` canh exe. Bien dich duoc, chay nem `FileNotFoundException`.
- **Benchmark repo in `0.0 ms`**: `now_ms`/`gc_allocated_bytes`/… la stub tra 0. Khong phai so that.
- **Bien dich lazy**: probe `import <module>` chi bat loi cap module. Muon chac phai bien dich ca entry point that (`cli.tkv`, `compare_rivals.tkv`).

## 5. Cach kiem tra nhanh sau moi lan sua repo

```powershell
cd D:\TokenVector.Inference
# bien dich 1 module + dependency cua no (qua import probe)
$v = "import tv.text.generate`n`ndef main() -> `"i32`":`n    print(`"OK`")`n    return 0"
[System.IO.File]::WriteAllText("tv\text\_probe.tkv", $v)
& D:\TokenVector\Release\dist\tkvc-tv.exe tv\text\_probe.tkv "$env:TEMP\p.exe" main
Remove-Item tv\text\_probe.tkv
# nho: import resolve ban colocated CUNG THU MUC - sua file canonical thi copy sang cac ban tv.*.tkv
```
